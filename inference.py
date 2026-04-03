"""
inference.py — Next-day tropospheric NO2 prediction pipeline.

Assembles a single feature row for a target date using recent NO2
observations and the latest available ERA5 covariates, then runs
the trained LightGBM model to produce a next-day forecast.
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
        Path to the pickled LightGBM model.
    no2_csv : Path
        Daily NO2 series CSV (must include at least the last 7 days).
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

    def _build_inference_row(self, target_date: date) -> pd.DataFrame:
        """
        Construct a single-row feature DataFrame for target_date (tomorrow).

        Uses:
        - no2_lag1/2/7 and rolling means from the NO2 history up to today
        - ERA5 covariates from today (best available proxy for tomorrow)
        - Calendar features computed for target_date
        """
        # Load NO2 history
        no2 = pd.read_csv(self.no2_csv, index_col="date", parse_dates=True)
        no2.sort_index(inplace=True)

        today = date.today()
        today_ts = pd.Timestamp(today)

        # Require at least 7 days of NO2 history ending on/before today
        # Convert mol/m² → µmol/m² to match training scale
        no2_hist = no2[no2.index <= today_ts]["no2_mean"].dropna() * 1e6
        if len(no2_hist) < 7:
            raise ValueError(
                f"Need at least 7 days of NO2 history; only {len(no2_hist)} available."
            )

        lag1 = float(no2_hist.iloc[-1])
        lag2 = float(no2_hist.iloc[-2]) if len(no2_hist) >= 2 else float("nan")
        lag7 = float(no2_hist.iloc[-7]) if len(no2_hist) >= 7 else float("nan")
        roll3 = float(no2_hist.iloc[-3:].mean()) if len(no2_hist) >= 3 else float("nan")
        roll7 = float(no2_hist.iloc[-7:].mean())

        # ERA5 covariates — use today's values as proxy for tomorrow
        era5 = self._ensure_era5_current()
        era5_row = era5[era5.index <= today_ts].iloc[-1]

        era5_features = {
            col: float(era5_row[col]) if col in era5_row.index else float("nan")
            for col in [
                "blh", "wind_u10", "wind_v10", "wind_speed",
                "temp_2m", "surface_pressure", "precip",
                "precip_log1p", "blh_log",
            ]
        }

        row = {
            "no2_lag1": lag1,
            "no2_lag2": lag2,
            "no2_lag7": lag7,
            "no2_roll3": roll3,
            "no2_roll7": roll7,
            **era5_features,
            **self._calendar_features(target_date),
        }

        # Align to expected feature order
        available = [c for c in FEATURE_COLS if c in row]
        return pd.DataFrame([row])[available]

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
        X = X[feature_cols]  # align to exact training columns
        prediction = float(model.predict(X)[0])

        # ERA5 lag diagnostic
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
            target_date,
            prediction,
            era5_lag,
        )
        return result
