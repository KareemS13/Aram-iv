"""
s5p_client.py — Sentinel-5P TROPOMI NO2 ingestion via Google Earth Engine.

Replaces cdse_client.py + no2_processor.py. Uses the GEE Python API to
query the full TROPOMI L2 NO2 archive (2018-present) server-side, applying
QA filtering and spatial aggregation over the Yerevan bounding box, then
exports a daily mean NO2 time series CSV locally.

No large file downloads — GEE does all the heavy lifting in the cloud.
"""

from datetime import date
from pathlib import Path

import ee
import pandas as pd

from config import (
    BBOX,
    MIN_PIXEL_COUNT,
    NO2_DAILY_CSV,
    PROCESSED_DIR,
    QA_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)

# GEE asset ID for Sentinel-5P TROPOMI NO2 offline L2 product
S5P_COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"

# Band names in the GEE L3 collection
NO2_BAND = "tropospheric_NO2_column_number_density"  # mol/m²
# L3 products are already QA-filtered during processing; no qa_value band needed.


class S5PClient:
    """
    Query Sentinel-5P TROPOMI NO2 data via Google Earth Engine.

    Applies QA filtering, clips to the Yerevan bounding box, and
    aggregates to daily mean NO2 column density.

    Parameters
    ----------
    project : str
        GEE Cloud project ID (e.g. 'ee-kareemsaffarini9').
    output_csv : Path
        Destination for the daily NO2 time series CSV.
    qa_threshold : float
        Minimum qa_value to retain a pixel (default 0.75).
    bbox : dict
        Bounding box with keys north/south/east/west.
    """

    def __init__(
        self,
        project: str,
        output_csv: Path = NO2_DAILY_CSV,
        qa_threshold: float = QA_THRESHOLD,
        bbox: dict = BBOX,
    ) -> None:
        self.project = project
        self.output_csv = Path(output_csv)
        self.qa_threshold = qa_threshold
        self.bbox = bbox
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        import os, json, tempfile
        sa_json_str = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
        if sa_json_str:
            sa_info = json.loads(sa_json_str)
            sa_email = sa_info["client_email"]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(sa_info, f)
                key_file = f.name
            credentials = ee.ServiceAccountCredentials(sa_email, key_file)
            ee.Initialize(credentials, project=project)
            logger.info("GEE initialized via service account (project=%s)", project)
        else:
            ee.Initialize(project=project)
            logger.info("GEE initialized with project: %s", project)

        # GEE geometry for the Yerevan bounding box
        self._region = ee.Geometry.Rectangle([
            bbox["west"], bbox["south"],
            bbox["east"], bbox["north"],
        ])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _daily_mean(self, image: ee.Image) -> ee.Feature:
        """
        Reduce one S5P L3 image to a single daily mean NO2 value over the bbox.

        L3 products are pre-QA-filtered. Computes mean and pixel count,
        returns an ee.Feature with properties: date, no2_mean, no2_pixel_count.
        """
        date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")

        no2 = image.select(NO2_BAND)

        # Count valid pixels
        pixel_count = no2.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self._region,
            scale=3500,
            maxPixels=1e6,
        ).get(NO2_BAND)

        # Mean NO2 over valid pixels
        mean_no2 = no2.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self._region,
            scale=3500,
            maxPixels=1e6,
        ).get(NO2_BAND)

        return ee.Feature(None, {
            "date": date_str,
            "no2_mean": mean_no2,
            "no2_pixel_count": pixel_count,
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def _fetch_year(
        self, year: int, start_month: int = 1, end_month: int = 12
    ) -> list[dict]:
        """
        Fetch one calendar year of daily NO2 means from GEE.

        Processes in monthly chunks to stay within GEE's 5000-element
        getInfo() limit (~14 images/day × 31 days = ~430 per month).
        start_month / end_month let callers skip months already cached.
        """
        from datetime import date as dt_date
        rows = []
        for month in range(start_month, end_month + 1):
            import calendar as _cal
            last_day = _cal.monthrange(year, month)[1]
            m_start = f"{year}-{month:02d}-01"
            m_end = f"{year}-{month:02d}-{last_day}"

            collection = (
                ee.ImageCollection(S5P_COLLECTION)
                .filterDate(m_start, m_end)
                .filterBounds(self._region)
            )

            features = collection.map(self._daily_mean)
            try:
                result = ee.FeatureCollection(features).getInfo()
            except Exception as exc:
                logger.warning("GEE error for %d-%02d: %s — skipping.", year, month, exc)
                continue

            for feat in result.get("features", []):
                props = feat["properties"]
                date_val = props.get("date")
                no2 = props.get("no2_mean")
                count_val = props.get("no2_pixel_count", 0)
                if date_val is None:
                    continue
                if count_val is not None and count_val < MIN_PIXEL_COUNT:
                    no2 = float("nan")
                rows.append({
                    "date": pd.Timestamp(date_val),
                    "no2_mean": no2,
                    "no2_pixel_count": count_val,
                })
            logger.info("  %d-%02d: %d images fetched", year, month, len(result.get("features", [])))

        return rows

    def fetch(self, date_start: date, date_end: date) -> pd.DataFrame:
        """
        Fetch daily mean NO2 over Yerevan bbox for the given date range.

        If an existing CSV is found, only fetches dates after the last
        cached date and merges with the existing data (incremental update).

        Processes year by year, month by month to stay within GEE limits.
        Returns a DataFrame indexed by date with columns:
          no2_mean (mol/m²), no2_pixel_count.
        Saves to self.output_csv.
        """
        # Load existing data and resume from where we left off
        existing: pd.DataFrame | None = None
        if self.output_csv.exists():
            existing = pd.read_csv(self.output_csv, index_col="date", parse_dates=True)
            if not existing.empty:
                last_cached = existing.index.max().date()
                if last_cached >= date_end:
                    logger.info("NO2 CSV already up to date (last: %s).", last_cached)
                    return existing
                from datetime import timedelta
                date_start = last_cached + timedelta(days=1)
                logger.info(
                    "Resuming NO2 fetch from %s (last cached: %s) …",
                    date_start, last_cached,
                )

        logger.info(
            "Querying GEE S5P NO2 collection for %s → %s …", date_start, date_end
        )

        all_rows: list[dict] = []
        for year in range(date_start.year, date_end.year + 1):
            logger.info("Fetching year %d …", year)
            sm = date_start.month if year == date_start.year else 1
            em = date_end.month if year == date_end.year else 12
            all_rows.extend(self._fetch_year(year, start_month=sm, end_month=em))

        if not all_rows:
            logger.warning("No valid rows after processing GEE results.")
            if existing is not None and not existing.empty:
                return existing
            return pd.DataFrame(columns=["no2_mean", "no2_pixel_count"])

        df = pd.DataFrame(all_rows).set_index("date")
        df.index = pd.DatetimeIndex(df.index)

        # Aggregate multiple images on same day (weighted mean)
        def weighted_mean(g: pd.DataFrame) -> pd.Series:
            total = g["no2_pixel_count"].sum()
            if total < MIN_PIXEL_COUNT:
                return pd.Series({"no2_mean": float("nan"), "no2_pixel_count": total})
            wsum = (g["no2_mean"].fillna(0) * g["no2_pixel_count"]).sum()
            return pd.Series({"no2_mean": wsum / total, "no2_pixel_count": total})

        df = df.groupby(df.index).apply(weighted_mean)
        df.sort_index(inplace=True)

        # Merge with existing cached data
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        df.to_csv(self.output_csv)
        logger.info(
            "Saved daily NO2 series: %d days, %d NaN → %s",
            len(df),
            df["no2_mean"].isna().sum(),
            self.output_csv,
        )
        return df

    def load_or_fetch(self, date_start: date, date_end: date) -> pd.DataFrame:
        """Load existing CSV if present and up-to-date, else call fetch()."""
        if self.output_csv.exists():
            existing = pd.read_csv(
                self.output_csv, index_col="date", parse_dates=True
            )
            if not existing.empty:
                last_date = existing.index.max().date()
                if last_date >= date_end:
                    logger.info(
                        "Loading cached NO2 daily CSV from %s", self.output_csv
                    )
                    return existing
        return self.fetch(date_start, date_end)
