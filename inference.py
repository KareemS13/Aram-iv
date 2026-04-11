"""
inference.py — Next-day tropospheric NO2 prediction pipeline.

Assembles a single feature row for a target date using recent NO2
observations and the latest available ERA5 covariates, then runs
the trained ensemble model to produce a next-day forecast.

predict()       — single next-day forecast (used at runtime)
predict_range() — rolling 14-day forecast (used by the refresh workflow
                  to pre-compute the predictions cache)
"""

from datetime import date, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from config import ERA5_DAILY_CSV, MODEL_PATH, NO2_DAILY_CSV, get_logger
from era5_client import ERA5Client
from era5_processor import ERA5Processor
from features import FEATURE_COLS
from model import NO2Forecaster

logger = get_logger(__name__)


class InferencePipeline:
    """
    Run next-day NO2 inference given recent observations and ERA5 data.

    Parameters
    ----------
    model_path : Path
        Path to the pickled ensemble model.
    no2_csv : Path
        Daily NO2 series CSV (must include at least the last 14 days).
    era5_csv : Path
        Daily ERA5 covariates CSV.
    era5_client : ERA5Client | None
        Used to fetch today's ERA5 covariates if not already cached.
    era5_processor : ERA5Processor | None
        Used to aggregate ERA5 NetCDF to daily scalars.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        no2_csv: Path = NO2_DAILY_CSV,
        era5_csv: Path = ERA5_DAILY_CSV,
        era5_client: ERA5Client | None = None,
        era5_processor: ERA5Processor | None = None,
    ) -> None:
        self.forecaster = NO2Forecaster(model_path=model_path)
        self.no2_csv = Path(no2_csv)
        self.era5_csv = Path(era5_csv)
        self.era5_client = era5_client
        self.era5_processor = era5_processor or ERA5Processor()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_era5_current(self) -> pd.DataFrame:
        """
        Load ERA5 daily covariates, fetching the latest month if needed.

        Returns a DataFrame indexed by date.
        """
        era5 = pd.read_csv(self.era5_csv, index_col="date", parse_dates=True)

        today = pd.Timestamp.today().normalize()
        if era5.index.max() < today - pd.Timedelta(days=7) and self.era5_client:
            logger.warning(
                "ERA5 data is %d days old — fetching latest available month.",
                (today - era5.index.max()).days,
            )
            nc_path = self.era5_client.fetch_latest_available()
            era5 = self.era5_processor.process_all([nc_path])

        return era5

    def _calendar_features(self, target_date: date) -> dict:
        """Return calendar feature values for target_date."""
        dow = target_date.weekday()
        month = target_date.month
        doy = target_date.timetuple().tm_yday
        am_holidays = holidays.country_holidays("AM")

        return {
            "day_of_week": dow,
            "is_weekend": int(dow >= 5),
            "month": month,
            "day_of_year": doy,
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "is_holiday": int(target_date in am_holidays),
            "is_holiday_lag1": int((target_date - timedelta(days=1)) in am_holidays),
        }

    def _build_row_from_series(
        self,
        no2_series: list[float],
        era5_row: pd.Series,
        target_date: date,
    ) -> pd.DataFrame:
        """
        Construct a single-row feature DataFrame from an explicit NO2 series.

        Parameters
        ----------
        no2_series : list[float]
            NO2 values in µmol/m², most-recent last. At least 14 values required.
            For rolling forecasts, predicted values are appended to this list
            before each call so lags stay correct.
        era5_row : pd.Series
            ERA5 covariates for the best available proxy date (fixed across the
            forecast horizon since future ERA5 is unavailable).
        target_date : date
            The date being predicted (used for calendar features).
        """
        n = len(no2_series)

        def _lag(k):
            return float(no2_series[-k]) if n >= k else float("nan")

        def _roll(k):
            return float(np.mean(no2_series[-k:])) if n >= k else float("nan")

        lag1 = _lag(1)

        era5_features = {
            col: float(era5_row[col]) if col in era5_row.index else float("nan")
            for col in [
                "blh", "wind_u10", "wind_v10", "wind_speed",
                "temp_2m", "surface_pressure", "precip",
                "precip_log1p", "blh_log", "solar_rad",
            ]
        }

        cal = self._calendar_features(target_date)
        month = cal["month"]
        is_winter = int(month >= 12 or month <= 2)

        blh  = era5_features.get("blh", float("nan"))
        wind = era5_features.get("wind_speed", float("nan"))
        temp = era5_features.get("temp_2m", float("nan"))

        row = {
            "no2_lag1":  lag1,
            "no2_lag2":  _lag(2),
            "no2_lag3":  _lag(3),
            "no2_lag4":  _lag(4),
            "no2_lag5":  _lag(5),
            "no2_lag7":  _lag(7),
            "no2_lag14": _lag(14),
            "no2_roll3":  _roll(3),
            "no2_roll7":  _roll(7),
            "no2_roll14": _roll(14),
            "no2_roll30": _roll(30),
            **era5_features,
            **cal,
            "is_winter":     is_winter,
            "temp_x_lag1":   temp * lag1 if not np.isnan(temp) else float("nan"),
            "wind_x_lag1":   wind * lag1 if not np.isnan(wind) else float("nan"),
            "winter_x_lag1": is_winter * lag1,
            "blh_x_lag1":    blh  * lag1 if not np.isnan(blh)  else float("nan"),
        }

        available = [c for c in FEATURE_COLS if c in row]
        return pd.DataFrame([row])[available]

    def _build_inference_row(self, target_date: date) -> pd.DataFrame:
        """
        Construct a single-row feature DataFrame for target_date.

        Reads NO2 history and ERA5 from disk, then delegates to
        _build_row_from_series.
        """
        no2 = pd.read_csv(self.no2_csv, index_col="date", parse_dates=True)
        no2.sort_index(inplace=True)

        today_ts = pd.Timestamp(date.today())
        no2_hist = no2[no2.index <= today_ts]["no2_mean"].dropna() * 1e6
        if len(no2_hist) < 7:
            raise ValueError(
                f"Need at least 7 days of NO2 history; only {len(no2_hist)} available."
            )

        era5 = self._ensure_era5_current()
        era5_row = era5[era5.index <= today_ts].iloc[-1]

        return self._build_row_from_series(list(no2_hist), era5_row, target_date)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, target_date: date | None = None) -> dict:
        """
        Predict tropospheric NO2 for target_date (default: tomorrow).

        Returns
        -------
        dict with keys:
          target_date           : ISO date string
          predicted_no2_mol_m2  : float (mol/m²)
          predicted_no2_umol_m2 : float (µmol/m²)
          era5_lag_days         : int (how many days old the ERA5 proxy is)
        """
        if target_date is None:
            target_date = date.today() + timedelta(days=1)

        logger.info("Building inference row for %s …", target_date)
        X = self._build_inference_row(target_date)

        model, feature_cols = self.forecaster.load_model()
        X = X[feature_cols]
        prediction = float(model.predict(X)[0])

        era5 = pd.read_csv(self.era5_csv, index_col="date", parse_dates=True)
        era5_lag = (pd.Timestamp.today().normalize() - era5.index.max()).days

        result = {
            "target_date": target_date.isoformat(),
            "predicted_no2_umol_m2": round(prediction, 4),
            "predicted_no2_mol_m2": round(prediction / 1e6, 10),
            "era5_lag_days": era5_lag,
        }

        logger.info(
            "Forecast for %s: %.4f µmol/m² (ERA5 lag: %d days)",
            target_date, prediction, era5_lag,
        )
        return result

    def predict_range(self, start_date: date, n_days: int = 14) -> list[dict]:
        """
        Pre-compute a rolling n-day forecast starting from start_date.

        Each day's prediction is fed back as a lag for subsequent days,
        matching how the model would be used in practice when actual
        observations are unavailable. ERA5 covariates are fixed to the
        most recently available row (future ERA5 is unknown at run time).

        Parameters
        ----------
        start_date : date
            First date to predict (typically date.today() + 1 day).
        n_days : int
            Number of consecutive days to predict (default 14).

        Returns
        -------
        list of dicts, each with keys:
          target_date, predicted_no2_umol_m2, predicted_no2_mol_m2, era5_lag_days
        """
        # Load NO2 history once — series in µmol/m², most-recent last
        no2 = pd.read_csv(self.no2_csv, index_col="date", parse_dates=True)
        no2.sort_index(inplace=True)
        today_ts = pd.Timestamp(date.today())
        no2_hist = list(no2[no2.index <= today_ts]["no2_mean"].dropna() * 1e6)

        if len(no2_hist) < 14:
            raise ValueError(
                f"Need at least 14 days of NO2 history; only {len(no2_hist)} available."
            )

        # Load model and ERA5 once
        model, feature_cols = self.forecaster.load_model()
        era5 = self._ensure_era5_current()
        era5_row = era5[era5.index <= today_ts].iloc[-1]
        era5_lag = (pd.Timestamp.today().normalize() - era5.index.max()).days

        results = []
        for i in range(n_days):
            target_date = start_date + timedelta(days=i)
            X = self._build_row_from_series(no2_hist, era5_row, target_date)
            X = X[feature_cols]
            pred = float(model.predict(X)[0])

            # Feed prediction back as the next lag-1 value
            no2_hist.append(pred)

            results.append({
                "target_date": target_date.isoformat(),
                "predicted_no2_umol_m2": round(pred, 4),
                "predicted_no2_mol_m2": round(pred / 1e6, 10),
                "era5_lag_days": era5_lag,
            })
            logger.info("Range forecast %s: %.4f µmol/m²", target_date, pred)

        return results
