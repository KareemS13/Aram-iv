"""
no2_processor.py — Extract and aggregate Sentinel-5P TROPOMI NO2 L2 data.

Reads granule ZIP archives, applies QA filtering (qa_value > 0.75),
clips to the Yerevan bounding box, and produces a daily mean tropospheric
NO2 column density time series in mol/m².
"""

import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    BBOX,
    MIN_PIXEL_COUNT,
    NO2_DAILY_CSV,
    PROCESSED_DIR,
    QA_THRESHOLD,
    RAW_S5P_DIR,
    S5P_GROUP,
    S5P_LAT_VAR,
    S5P_LON_VAR,
    S5P_NO2_VAR,
    S5P_QA_VAR,
    get_logger,
)

logger = get_logger(__name__)


class NO2Processor:
    """
    Extract and spatially aggregate Sentinel-5P TROPOMI NO2 L2 data.

    Parameters
    ----------
    raw_dir : Path
        Directory containing downloaded S5P ZIP files.
    output_csv : Path
        Destination for the daily aggregated NO2 time series CSV.
    qa_threshold : float
        Minimum qa_value to retain a pixel.
    bbox : dict
        Bounding box with keys north/south/east/west.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_S5P_DIR,
        output_csv: Path = NO2_DAILY_CSV,
        qa_threshold: float = QA_THRESHOLD,
        bbox: dict = BBOX,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_csv = Path(output_csv)
        self.qa_threshold = qa_threshold
        self.bbox = bbox
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_nc_from_zip(self, zip_path: Path, tmp_dir: str) -> Path:
        """
        Extract the NO2 NetCDF file from a granule ZIP into tmp_dir.

        Returns the path to the extracted .nc file.
        Raises FileNotFoundError if no matching file is found.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_names = [n for n in zf.namelist() if "NO2" in n and n.endswith(".nc")]
            if not nc_names:
                # Some granules wrap the nc inside a subdirectory
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise FileNotFoundError(
                    f"No .nc file found inside {zip_path.name}"
                )
            # Prefer the file whose name matches the ZIP stem
            target = nc_names[0]
            for name in nc_names:
                if zip_path.stem in name:
                    target = name
                    break
            zf.extract(target, tmp_dir)
        return Path(tmp_dir) / target

    def _parse_sensing_date(self, nc_path: Path) -> date:
        """
        Parse the sensing start date from the granule filename.

        S5P filenames embed the sensing start at characters 20-27 (0-indexed):
          S5P_OFFL_L2__NO2____20230115T081234_...
                              ^^^^^^^^
        """
        stem = nc_path.stem
        # Walk up if nested (extract may preserve directory structure)
        name = stem.split("/")[-1].split("\\")[-1]
        date_str = name[20:28]  # YYYYMMDD
        return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))

    def _process_granule(
        self, nc_path: Path
    ) -> tuple[date, float, int] | None:
        """
        Process one S5P NO2 granule and return (sensing_date, mean_no2, pixel_count).

        Returns None if no valid pixels remain after masking.
        """
        try:
            ds = xr.open_dataset(str(nc_path), group=S5P_GROUP, mask_and_scale=True)
        except Exception as exc:
            logger.warning("Cannot open %s: %s", nc_path.name, exc)
            return None

        # Squeeze trivial time dimension (always size 1 per granule)
        ds = ds.squeeze("time", drop=True)

        lat = ds[S5P_LAT_VAR].values
        lon = ds[S5P_LON_VAR].values
        qa = ds[S5P_QA_VAR].values
        no2 = ds[S5P_NO2_VAR].values

        ds.close()

        # Spatial bbox mask
        spatial_mask = (
            (lat >= self.bbox["south"])
            & (lat <= self.bbox["north"])
            & (lon >= self.bbox["west"])
            & (lon <= self.bbox["east"])
        )
        # QA mask
        qa_mask = qa > self.qa_threshold

        valid = spatial_mask & qa_mask & np.isfinite(no2)
        pixel_count = int(valid.sum())

        if pixel_count == 0:
            return None

        mean_no2 = float(no2[valid].mean())
        sensing_date = self._parse_sensing_date(nc_path)
        return sensing_date, mean_no2, pixel_count

    # ── Public API ────────────────────────────────────────────────────────────

    def process_all(self, zip_paths: list[Path]) -> pd.DataFrame:
        """
        Process a list of granule ZIP files and aggregate to daily means.

        Multiple granules on the same day (multiple orbits) are combined
        using pixel-count-weighted averaging.

        Returns a DataFrame indexed by date with columns:
          no2_mean (mol/m²), no2_pixel_count.
        Saves result to self.output_csv.
        """
        # Accumulator: date → (weighted_sum, total_pixels)
        accum: dict[date, list[float]] = {}  # date → [weighted_sum, total_count]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for zip_path in zip_paths:
                try:
                    nc_path = self._extract_nc_from_zip(zip_path, tmp_dir)
                except FileNotFoundError as exc:
                    logger.warning("%s", exc)
                    continue

                result = self._process_granule(nc_path)
                if result is None:
                    logger.debug("No valid pixels in %s", zip_path.name)
                    continue

                sensing_date, mean_no2, pixel_count = result
                if sensing_date not in accum:
                    accum[sensing_date] = [0.0, 0]
                accum[sensing_date][0] += mean_no2 * pixel_count
                accum[sensing_date][1] += pixel_count

        if not accum:
            logger.warning("No valid granules processed.")
            return pd.DataFrame(columns=["no2_mean", "no2_pixel_count"])

        rows = []
        for d, (weighted_sum, total_count) in sorted(accum.items()):
            daily_mean = weighted_sum / total_count if total_count >= MIN_PIXEL_COUNT else float("nan")
            rows.append({"date": pd.Timestamp(d), "no2_mean": daily_mean, "no2_pixel_count": total_count})

        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)

        df.to_csv(self.output_csv)
        logger.info(
            "Saved daily NO2 series: %d days, %d NaN (low pixel count) → %s",
            len(df),
            df["no2_mean"].isna().sum(),
            self.output_csv,
        )
        return df

    def load_or_build(self, zip_paths: list[Path]) -> pd.DataFrame:
        """
        Load existing CSV if present, otherwise run process_all().

        Re-processes if the CSV is missing or its last date is earlier
        than the latest sensing date implied by zip_paths.
        """
        if self.output_csv.exists():
            existing = pd.read_csv(self.output_csv, index_col="date", parse_dates=True)
            if not existing.empty:
                last_csv_date = existing.index.max().date()
                last_zip_date = max(
                    date(
                        int(p.stem[20:24]), int(p.stem[24:26]), int(p.stem[26:28])
                    )
                    for p in zip_paths
                    if len(p.stem) >= 28
                )
                if last_csv_date >= last_zip_date:
                    logger.info("Loading cached NO2 daily CSV from %s", self.output_csv)
                    return existing
        return self.process_all(zip_paths)
