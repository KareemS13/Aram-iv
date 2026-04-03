"""
cdse_client.py — Copernicus Data Space Ecosystem (CDSE) client.

Handles OAuth2 token acquisition, OData catalogue search for Sentinel-5P
NO2 L2 granules intersecting the Yerevan bounding box, and authenticated
streaming download of granule ZIP files.
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from config import (
    BBOX_WKT,
    CDSE_CATALOGUE_URL,
    CDSE_DOWNLOAD_BASE,
    CDSE_TOKEN_URL,
    RAW_S5P_DIR,
    S5P_OFFL_PREFIX,
    S5P_RPRO_PREFIX,
    get_logger,
)

logger = get_logger(__name__)

_TOKEN_EXPIRY_BUFFER_S = 60  # refresh token this many seconds before expiry


class CDSEClient:
    """
    Client for the Copernicus Data Space Ecosystem OData API.

    Parameters
    ----------
    username : str
        CDSE account e-mail. Reads CDSE_USER env var if not provided.
    password : str
        CDSE account password. Reads CDSE_PASS env var if not provided.
    download_dir : Path
        Local directory where downloaded ZIP files are saved.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        download_dir: Path = RAW_S5P_DIR,
    ) -> None:
        self.username = username or os.environ["CDSE_USER"]
        self.password = password or os.environ["CDSE_PASS"]
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self._token: str | None = None
        self._token_expires_at: float = 0.0

    # ── Authentication ────────────────────────────────────────────────────────

    def _fetch_token(self) -> str:
        """POST to CDSE token endpoint and store expiry timestamp."""
        resp = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"CDSE token request failed {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.monotonic() + payload.get("expires_in", 600)
        logger.debug("CDSE token acquired, expires in %ss", payload.get("expires_in"))
        return self._token

    def _get_token(self) -> str:
        """Return a valid Bearer token, refreshing if near expiry."""
        if (
            self._token is None
            or time.monotonic() >= self._token_expires_at - _TOKEN_EXPIRY_BUFFER_S
        ):
            self._fetch_token()
        return self._token  # type: ignore[return-value]

    # ── Catalogue search ──────────────────────────────────────────────────────

    def _build_filter(self, date_start: date, date_end: date, prefix: str) -> str:
        """Construct the OData $filter string for one product prefix."""
        start_str = f"{date_start.isoformat()}T00:00:00.000Z"
        end_str = f"{(date_end + timedelta(days=1)).isoformat()}T00:00:00.000Z"
        return (
            f"Collection/Name eq 'SENTINEL-5P'"
            f" and startswith(Name,'{prefix}')"
            f" and ContentDate/Start ge {start_str}"
            f" and ContentDate/Start lt {end_str}"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{BBOX_WKT}')"
        )

    def _search_page(
        self, filter_str: str, skip: int, top: int
    ) -> tuple[list[dict], int]:
        """Fetch one page of OData results. Returns (items, total_count)."""
        params = {
            "$filter": filter_str,
            "$top": top,
            "$skip": skip,
            "$orderby": "ContentDate/Start asc",
            "$count": "True",
        }
        resp = requests.get(CDSE_CATALOGUE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("value", [])
        total = data.get("@odata.count", len(items))
        return items, total

    def search(
        self,
        date_start: date,
        date_end: date,
        product_prefix: str = S5P_OFFL_PREFIX,
        page_size: int = 100,
    ) -> list[dict]:
        """
        Search the CDSE catalogue for S5P NO2 granules in a date range.

        Returns list of product dicts with keys: Id, Name, ContentDate,
        ContentLength, Online.
        """
        filter_str = self._build_filter(date_start, date_end, product_prefix)
        all_items: list[dict] = []
        skip = 0

        while True:
            items, total = self._search_page(filter_str, skip, page_size)
            all_items.extend(items)
            logger.debug(
                "Search '%s': fetched %d/%d", product_prefix, len(all_items), total
            )
            if len(all_items) >= total or not items:
                break
            skip += page_size

        logger.info(
            "Found %d granules for %s [%s → %s]",
            len(all_items),
            product_prefix,
            date_start,
            date_end,
        )
        return all_items

    # ── Download ──────────────────────────────────────────────────────────────

    def download(self, product_id: str, product_name: str) -> Path | None:
        """
        Download a single product by its UUID to self.download_dir.

        Skips if the file already exists with non-zero size.
        Returns the local Path of the downloaded file, or None if the
        product is in the Long Term Archive (LTA) and not immediately available.
        """
        dest = self.download_dir / f"{product_name}.zip"
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("Already downloaded: %s", dest.name)
            return dest

        url = f"{CDSE_DOWNLOAD_BASE}('{product_id}')/$value"
        headers = {"Authorization": f"Bearer {self._get_token()}"}

        logger.info("Downloading %s …", product_name)
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            if resp.status_code == 422:
                logger.warning(
                    "Skipping %s — product is in Long Term Archive (offline).",
                    product_name,
                )
                return None
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

        logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    # ── High-level ────────────────────────────────────────────────────────────

    def fetch_date_range(
        self,
        date_start: date,
        date_end: date,
        prefer_rpro: bool = True,
    ) -> list[Path]:
        """
        Search and download all NO2 granules for a date range.

        Strategy: when prefer_rpro=True, search RPRO first. Any dates not
        covered by RPRO are filled with OFFL results. Falls back entirely
        to OFFL if RPRO returns nothing.

        Returns a list of local ZIP paths.
        """
        downloaded: list[Path] = []

        if prefer_rpro:
            rpro_items = self.search(date_start, date_end, S5P_RPRO_PREFIX)
        else:
            rpro_items = []

        # Collect RPRO sensing dates to identify gaps
        rpro_dates: set[str] = set()
        for item in rpro_items:
            rpro_dates.add(item["ContentDate"]["Start"][:10])
            path = self.download(item["Id"], item["Name"])
            if path is not None:
                downloaded.append(path)

        # Fill gaps with OFFL
        offl_items = self.search(date_start, date_end, S5P_OFFL_PREFIX)
        for item in offl_items:
            sensing_date = item["ContentDate"]["Start"][:10]
            if sensing_date not in rpro_dates:
                path = self.download(item["Id"], item["Name"])
                if path is not None:
                    downloaded.append(path)

        logger.info(
            "fetch_date_range complete: %d files for %s → %s",
            len(downloaded),
            date_start,
            date_end,
        )
        return downloaded
