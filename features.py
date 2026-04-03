"""
features.py — Build the feature matrix for next-day NO2 forecasting.

Joins the daily NO2 series with ERA5 covariates, constructs lag features,
rolling statistics, cyclic calendar encodings, and Armenian holiday flags.
The target variable is tomorrow's NO2 (no2_mean shifted by -1).
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from config import ERA5_DAILY_CSV, FEATURES_CSV, NO2_DAILY_CSV, PROCESSED_DIR, get_logger

logger = get_logger(__name__)

# Autoregressive lag offsets (days)
NO2_LAGS = [1, 2, 7]

# Rolling window sizes (days, backward-looking)
NO2_ROLLING = [3, 7]

# All input feature columns in order (no target)
FEATURE_COLS = [
    "no2_lag1", "no2_lag2", "no2_lag7",
    "no2_roll3", "no2_roll7",
    "blh", "wind_u10", "wind_v10", "wind_speed",
    "temp_2m", "surface_pressure", "precip",
    "precip_log1p", "blh_log",
    "day_of_week", "is_weekend", "month", "day_of_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_holiday", "is_holiday_lag1",
]


class FeatureEngineer:
    """
    Build the feature matrix for next-day tropospheric NO2 forecasting.

    Parameters
    ----------
    no2_csv : Path
        Daily NO2 CSV produced by NO2Processor (columns: no2_mean, no2_pixel_count).
    era5_csv : Path
        Daily ERA5 CSV produced by ERA5Processor.
    output_csv : Path
        Destination for the full feature matrix CSV.
    """

    def __init__(
        self,
        no2_csv: Path = NO2_DAILY_CSV,
        era5_csv: Path = ERA5_DAILY_CSV,
        output_csv: Path = FEATURES_CSV,
    ) -> None:
        self.no2_csv = Path(no2_csv)
        self.era5_csv = Path(era5_csv)
        self.output_csv = Path(output_csv)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature construction steps ────────────────────────────────────────────

    def _add_no2_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add autoregressive lag columns and backward rolling means for NO2."""
        for lag in NO2_LAGS:
            df[f"no2_lag{lag}"] = df["no2_mean"].shift(lag)
        for window in NO2_ROLLING:
            df[f"no2_roll{window}"] = (
                df["no2_mean"]
                .shift(1)  # exclude current day to avoid leakage
                .rolling(window=window, min_periods=max(2, window // 2))
                .mean()
            )
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal calendar and cyclic encoding features."""
        idx = df.index
        df["day_of_week"] = idx.dayofweek          # 0=Monday, 6=Sunday
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = idx.month
        df["day_of_year"] = idx.dayofyear

        # Cyclic encodings
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _add_holiday_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Armenian public holiday flags using the holidays library."""
        am_holidays = holidays.country_holidays("AM")
        df["is_holiday"] = df.index.map(lambda d: int(d.date() in am_holidays))
        df["is_holiday_lag1"] = df["is_holiday"].shift(1).fillna(0).astype(int)
        return df

    # ── Main build pipeline ───────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Full pipeline: load → lag → ERA5 merge → calendar → holiday → target.

        Target column 'no2_next' = no2_mean shifted by -1 (tomorrow's value).
        Drops the first 7 rows (NaN lags) and the last row (NaN target).

        Returns the complete feature matrix DataFrame.
        """
        # Load NO2 series — convert mol/m² → µmol/m² for numerical stability
        no2 = pd.read_csv(self.no2_csv, index_col="date", parse_dates=True)
        no2.index = pd.DatetimeIndex(no2.index)
        no2["no2_mean"] = no2["no2_mean"] * 1e6

        # Load ERA5 covariates
        era5 = pd.read_csv(self.era5_csv, index_col="date", parse_dates=True)
        era5.index = pd.DatetimeIndex(era5.index)

        # Merge on date index (inner join — only days with both sources)
        df = no2[["no2_mean"]].join(era5, how="inner")

        if len(df) < 30:
            raise ValueError(
                f"Too few rows after join ({len(df)}). "
                "Ensure NO2 and ERA5 date ranges overlap by at least 30 days."
            )

        # Build features
        df = self._add_no2_lag_features(df)
        df = self._add_calendar_features(df)
        df = self._add_holiday_flag(df)

        # Target: next-day NO2
        df["no2_next"] = df["no2_mean"].shift(-1)

        # Drop rows with NaN target or NaN in any required lag
        lag_cols = [f"no2_lag{l}" for l in NO2_LAGS]
        df.dropna(subset=["no2_next"] + lag_cols, inplace=True)

        # Keep only feature columns + target (drop raw no2_mean, pixel_count, etc.)
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        df = df[available_features + ["no2_next"]]

        df.to_csv(self.output_csv)
        logger.info(
            "Feature matrix: %d rows × %d features → %s",
            len(df),
            len(available_features),
            self.output_csv,
        )
        return df

    def load_or_build(self) -> pd.DataFrame:
        """Load feature matrix from CSV if it exists, else call build()."""
        if self.output_csv.exists():
            logger.info("Loading cached feature matrix from %s", self.output_csv)
            return pd.read_csv(self.output_csv, index_col="date", parse_dates=True)
        return self.build()

    @staticmethod
    def feature_columns() -> list[str]:
        """Return the ordered list of input feature column names (no target)."""
        return list(FEATURE_COLS)
