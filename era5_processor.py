"""
era5_processor.py — Convert ERA5 monthly NetCDF files into a daily scalar table.

Spatial aggregation: mean over all grid cells within BBOX.
Temporal aggregation: daily mean for most variables; daily sum for precipitation.
"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    BBOX,
    ERA5_DAILY_CSV,
    PROCESSED_DIR,
    RAW_ERA5_DIR,
    RAW_ERA5_SOLAR_DIR,
    SOLAR_DAILY_CSV,
    get_logger,
)

logger = get_logger(__name__)

# ERA5 NetCDF short names → output column names
_RENAME = {
    "blh": "blh",
    "u10": "wind_u10",
    "v10": "wind_v10",
    "t2m": "temp_2m",
    "sp": "surface_pressure",
    "tp": "precip",
}

# Variables that should be summed (not averaged) over the day
_SUM_VARS = {"tp"}


class ERA5Processor:
    """
    Convert ERA5 monthly NetCDF files into a daily scalar covariate table.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ERA5 monthly NetCDF files.
    output_csv : Path
        Destination CSV with one row per day.
    bbox : dict
        Bounding box with keys north/south/east/west.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_ERA5_DIR,
        output_csv: Path = ERA5_DAILY_CSV,
        bbox: dict = BBOX,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_csv = Path(output_csv)
        self.bbox = bbox
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_month_file(self, nc_path: Path) -> pd.DataFrame:
        """
        Open one ERA5 monthly NetCDF, spatially and temporally aggregate.

        Steps:
        1. Open dataset, clip to bbox.
        2. Spatial mean over latitude/longitude.
        3. Resample to daily: mean for most vars, sum for precipitation.
        4. Rename columns, derive wind_speed, apply log transforms.

        Returns a DataFrame indexed by date (one row per day).
        """
        # New CDS API wraps NetCDF inside a ZIP — detect and extract
        _tmp_dir = None
        with open(nc_path, "rb") as f:
            is_zip = f.read(2) == b"PK"
        if is_zip:
            _tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(nc_path, "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise ValueError(f"No .nc file found inside {nc_path.name}")
                for name in nc_names:
                    zf.extract(name, _tmp_dir)
            # Merge all .nc files inside the ZIP (instant + accumulated streams)
            datasets = [xr.open_dataset(str(Path(_tmp_dir) / n)) for n in nc_names]
            ds = xr.merge(datasets)
        else:
            ds = xr.open_dataset(str(nc_path))

        # New CDS API uses 'valid_time' instead of 'time' — rename to 'time'
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        elif "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.assign_coords(time=ds["valid_time"]).swap_dims({"valid_time": "time"})

        # Drop non-data dimensions/vars that interfere with spatial averaging
        for drop_var in ["number", "expver"]:
            if drop_var in ds.coords or drop_var in ds.data_vars:
                ds = ds.drop_vars(drop_var, errors="ignore")

        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"

        # Try clipping to bbox first; fall back to nearest grid point if bbox
        # is smaller than the ERA5 grid resolution (~0.25°) and returns nothing.
        lat_slice = slice(self.bbox["north"], self.bbox["south"])
        lon_slice = slice(self.bbox["west"], self.bbox["east"])
        ds_clipped = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})

        if ds_clipped[lat_dim].size == 0 or ds_clipped[lon_dim].size == 0:
            # Bbox too small — select nearest grid point to bbox centre
            centre_lat = (self.bbox["north"] + self.bbox["south"]) / 2
            centre_lon = (self.bbox["east"] + self.bbox["west"]) / 2
            ds_clipped = ds.sel(
                {lat_dim: centre_lat, lon_dim: centre_lon}, method="nearest"
            )
            ds_spatial = ds_clipped  # already a point — no spatial mean needed
        else:
            ds_spatial = ds_clipped.mean(dim=[lat_dim, lon_dim])

        # Only keep the ERA5 variable columns we care about
        keep_vars = list(_RENAME.keys())
        ds_spatial = ds_spatial[[v for v in keep_vars if v in ds_spatial.data_vars]]

        # Separate sum vs mean variables
        sum_vars = [v for v in ds_spatial.data_vars if v in _SUM_VARS]
        mean_vars = [v for v in ds_spatial.data_vars if v not in _SUM_VARS]

        frames: list[pd.DataFrame] = []

        if mean_vars:
            daily_mean = (
                ds_spatial[mean_vars]
                .resample(time="1D")
                .mean()
                .to_dataframe()
                .reset_index()
                .set_index("time")
            )
            frames.append(daily_mean)

        if sum_vars:
            daily_sum = (
                ds_spatial[sum_vars]
                .resample(time="1D")
                .sum()
                .to_dataframe()
                .reset_index()
                .set_index("time")
            )
            frames.append(daily_sum)

        if is_zip:
            for sub_ds in datasets:
                sub_ds.close()
        else:
            ds.close()

        df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        df.index.name = "date"
        df.index = pd.DatetimeIndex(df.index).normalize()

        # Rename to friendly column names
        df.rename(columns=_RENAME, inplace=True)

        # Drop any columns not in _RENAME values (e.g. expver, number)
        keep = list(_RENAME.values())
        df = df[[c for c in keep if c in df.columns]]

        return df

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features: wind_speed, precip_log1p, blh_log."""
        if "wind_u10" in df.columns and "wind_v10" in df.columns:
            df["wind_speed"] = np.sqrt(df["wind_u10"] ** 2 + df["wind_v10"] ** 2)
        if "precip" in df.columns:
            df["precip_log1p"] = np.log1p(df["precip"].clip(lower=0))
        if "blh" in df.columns:
            df["blh_log"] = np.log(df["blh"].clip(lower=1))
        return df

    # ── Public API ────────────────────────────────────────────────────────────

    def process_all(self, nc_paths: list[Path]) -> pd.DataFrame:
        """
        Process all monthly NetCDF files and concatenate into one DataFrame.

        Drops duplicate dates (can occur at month boundaries), sorts by date,
        adds derived features, and saves to self.output_csv.

        Returns the combined daily DataFrame.
        """
        parts: list[pd.DataFrame] = []
        for nc_path in sorted(nc_paths):
            try:
                part = self._process_month_file(nc_path)
                parts.append(part)
                logger.debug("Processed ERA5 %s: %d days", nc_path.name, len(part))
            except Exception as exc:
                logger.warning("Failed to process %s: %s", nc_path.name, exc)

        if not parts:
            logger.warning("No ERA5 files processed.")
            return pd.DataFrame()

        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = self._enrich(df)

        df.to_csv(self.output_csv)
        logger.info(
            "Saved ERA5 daily covariates: %d days → %s", len(df), self.output_csv
        )
        return df

    def load_or_build(self, nc_paths: list[Path]) -> pd.DataFrame:
        """Load existing CSV if present and up-to-date, else call process_all()."""
        if self.output_csv.exists():
            existing = pd.read_csv(
                self.output_csv, index_col="date", parse_dates=True
            )
            if not existing.empty:
                logger.info("Loading cached ERA5 daily CSV from %s", self.output_csv)
                return existing
        return self.process_all(nc_paths)


class SolarProcessor:
    """
    Convert ERA5 surface solar radiation downwards (ssrd) NetCDF files into
    a daily scalar series.

    ssrd is an accumulated field (J/m²) — hourly values are summed to get
    the daily total insolation, then saved as 'solar_rad'.

    Parameters
    ----------
    raw_dir : Path
        Directory containing yearly solar NetCDF files (e.g. solar_2018.nc).
    output_csv : Path
        Destination CSV with columns: date, solar_rad.
    bbox : dict
        Bounding box with keys north/south/east/west.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_ERA5_SOLAR_DIR,
        output_csv: Path = SOLAR_DAILY_CSV,
        bbox: dict = BBOX,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_csv = Path(output_csv)
        self.bbox = bbox
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _process_file(self, nc_path: Path) -> pd.DataFrame:
        """
        Open one solar NetCDF, spatially clip and aggregate, daily sum ssrd.

        Returns a DataFrame indexed by date with column 'solar_rad'.
        """
        ds = xr.open_dataset(str(nc_path))

        # Normalise time dimension name
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        elif "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.assign_coords(time=ds["valid_time"]).swap_dims({"valid_time": "time"})

        # Drop non-data vars
        for drop_var in ["number", "expver"]:
            ds = ds.drop_vars(drop_var, errors="ignore")

        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"

        lat_slice = slice(self.bbox["north"], self.bbox["south"])
        lon_slice = slice(self.bbox["west"], self.bbox["east"])
        ds_clipped = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})

        if ds_clipped[lat_dim].size == 0 or ds_clipped[lon_dim].size == 0:
            centre_lat = (self.bbox["north"] + self.bbox["south"]) / 2
            centre_lon = (self.bbox["east"] + self.bbox["west"]) / 2
            ds_spatial = ds.sel(
                {lat_dim: centre_lat, lon_dim: centre_lon}, method="nearest"
            )
        else:
            ds_spatial = ds_clipped.mean(dim=[lat_dim, lon_dim])

        # Daily sum of accumulated ssrd (J/m²)
        daily = (
            ds_spatial[["ssrd"]]
            .resample(time="1D")
            .sum()
            .to_dataframe()
            .reset_index()
            .set_index("time")
        )
        ds.close()

        daily.index.name = "date"
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        daily.rename(columns={"ssrd": "solar_rad"}, inplace=True)
        return daily[["solar_rad"]]

    def process_all(self, nc_paths: list[Path]) -> pd.DataFrame:
        """
        Process all solar NetCDF files and concatenate into one DataFrame.

        Saves result to self.output_csv and returns the DataFrame.
        """
        parts: list[pd.DataFrame] = []
        for nc_path in sorted(nc_paths):
            try:
                part = self._process_file(nc_path)
                parts.append(part)
                logger.debug("Processed solar %s: %d days", nc_path.name, len(part))
            except Exception as exc:
                logger.warning("Failed to process %s: %s", nc_path.name, exc)

        if not parts:
            logger.warning("No solar files processed.")
            return pd.DataFrame()

        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="first")].sort_index()

        df.to_csv(self.output_csv)
        logger.info(
            "Saved solar daily series: %d days → %s", len(df), self.output_csv
        )
        return df

    def load_or_build(self, nc_paths: list[Path]) -> pd.DataFrame:
        """Load existing CSV if present, else call process_all()."""
        if self.output_csv.exists():
            existing = pd.read_csv(
                self.output_csv, index_col="date", parse_dates=True
            )
            if not existing.empty:
                logger.info("Loading cached solar daily CSV from %s", self.output_csv)
                return existing
        return self.process_all(nc_paths)
