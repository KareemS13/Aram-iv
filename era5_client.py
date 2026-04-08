
"""
era5_client.py — ERA5 reanalysis data download via the CDS API.

Downloads monthly ERA5 single-level files for the Yerevan bounding box.
Requires a valid ~/.cdsapirc or CDS_URL / CDS_KEY environment variables.
"""

import calendar
import os
from datetime import date
from pathlib import Path

import cdsapi

from config import (
    ERA5_AREA,
    ERA5_DATASET,
    ERA5_TIMES,
    ERA5_VARIABLES,
    RAW_ERA5_DIR,
    get_logger,
)

logger = get_logger(__name__)


class ERA5Client:
    """
    Download ERA5 hourly single-level data for the Yerevan bounding box.

    One NetCDF file per calendar month is stored in download_dir. Requests
    are skipped if the file already exists.

    Parameters
    ----------
    download_dir : Path
        Directory where ERA5 NetCDF files are stored.
    cds_url : str | None
        CDS API URL override. Falls back to ~/.cdsapirc if None.
    cds_key : str | None
        CDS API key override. Falls back to ~/.cdsapirc if None.
    """

    def __init__(
        self,
        download_dir: Path = RAW_ERA5_DIR,
        cds_url: str | None = None,
        cds_key: str | None = None,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Build cdsapi client kwargs; omit keys not provided so the
        # library falls back to ~/.cdsapirc for missing values.
        client_kwargs: dict = {"quiet": True}
        url = (cds_url or os.environ.get("CDS_URL") or "").strip()
        key = (cds_key or os.environ.get("CDS_KEY") or "").strip()
        if url:
            client_kwargs["url"] = url
        if key:
            client_kwargs["key"] = key

        self._client = cdsapi.Client(**client_kwargs)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _month_file_path(self, year: int, month: int) -> Path:
        """Return the local path for a month's ERA5 NetCDF file."""
        return self.download_dir / f"era5_{year}{month:02d}.nc"

    # ── Download ──────────────────────────────────────────────────────────────

    def fetch_month(self, year: int, month: int) -> Path:
        """
        Download one calendar month of ERA5 hourly data for the Yerevan bbox.

        Skips if the local file already exists (idempotent).

        Returns the local path of the NetCDF file.
        """
        dest = self._month_file_path(year, month)
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("ERA5 already cached: %s", dest.name)
            return dest

        days_in_month = calendar.monthrange(year, month)[1]
        day_list = [f"{d:02d}" for d in range(1, days_in_month + 1)]

        request = {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": day_list,
            "time": ERA5_TIMES,
            "area": ERA5_AREA,
            "data_format": "netcdf",
        }

        logger.info("Fetching ERA5 %d-%02d …", year, month)
        try:
            self._client.retrieve(ERA5_DATASET, request, str(dest))
        except Exception as exc:
            if "not available yet" in str(exc) or "400" in str(exc):
                logger.warning(
                    "ERA5 %d-%02d not yet available on CDS — skipping.", year, month
                )
                return None
            raise
        logger.info("Saved ERA5 %d-%02d → %s (%.1f MB)", year, month, dest.name, dest.stat().st_size / 1e6)
        return dest

    def fetch_date_range(self, date_start: date, date_end: date) -> list[Path]:
        """
        Fetch all months spanning date_start..date_end.

        Returns a list of local NetCDF paths (one per calendar month).
        """
        paths: list[Path] = []
        year, month = date_start.year, date_start.month
        end_year, end_month = date_end.year, date_end.month

        while (year, month) <= (end_year, end_month):
            result = self.fetch_month(year, month)
            if result is not None:
                paths.append(result)
            month += 1
            if month > 12:
                month = 1
                year += 1

        logger.info(
            "ERA5 fetch_date_range complete: %d monthly files for %s → %s",
            len(paths),
            date_start,
            date_end,
        )
        return paths

    def fetch_latest_available(self) -> Path:
        """
        Fetch the most recently available ERA5 month (ERA5 has ~5-day latency).

        Tries the current month, then steps back one month at a time until
        a successful download is returned. Logs a warning about the lag.
        """
        from datetime import date as dt_date
        today = dt_date.today()
        year, month = today.year, today.month

        for _ in range(3):  # try up to 3 months back
            dest = self._month_file_path(year, month)
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            try:
                return self.fetch_month(year, month)
            except Exception as exc:
                logger.warning(
                    "ERA5 %d-%02d not yet available (%s), trying previous month.",
                    year, month, exc,
                )
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1

        raise RuntimeError("Could not retrieve any recent ERA5 data.")
