"""
config.py — Shared constants, paths, and model hyperparameters.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Geographic ────────────────────────────────────────────────────────────────
YEREVAN_LAT = 40.18
YEREVAN_LON = 44.51

BBOX = {"north": 40.24, "south": 40.08, "east": 44.63, "west": 44.35}

# WKT polygon (lon lat order, closed ring) for OData spatial filter
BBOX_WKT = (
    "POLYGON((44.35 40.08,44.63 40.08,44.63 40.24,44.35 40.24,44.35 40.08))"
)

# ── CDSE / Sentinel-5P ────────────────────────────────────────────────────────
CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)

# Product type prefixes (RPRO = reprocessed preferred, OFFL = offline fallback)
S5P_RPRO_PREFIX = "S5P_RPRO_L2__NO2____"
S5P_OFFL_PREFIX = "S5P_OFFL_L2__NO2____"

# NetCDF group and variable names inside each S5P granule
S5P_GROUP = "PRODUCT"
S5P_NO2_VAR = "nitrogendioxide_tropospheric_column"  # mol/m²
S5P_QA_VAR = "qa_value"
S5P_LAT_VAR = "latitude"
S5P_LON_VAR = "longitude"

QA_THRESHOLD = 0.75
MIN_PIXEL_COUNT = 10  # days with fewer valid pixels are set to NaN

# ── ERA5 / CDS ────────────────────────────────────────────────────────────────
ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = [
    "boundary_layer_height",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
]
# CDS area order: [North, West, South, East]
# Add 1° buffer so MARS can clip the ERA5 grid (Yerevan bbox alone is smaller
# than the 0.25° ERA5 grid resolution and causes MARS to reject the request).
ERA5_AREA = [BBOX["north"] + 1.0, BBOX["west"] - 1.0, BBOX["south"] - 1.0, BBOX["east"] + 1.0]
ERA5_TIMES = [f"{h:02d}:00" for h in range(24)]

# Short variable names as they appear in ERA5 NetCDF output
ERA5_SHORT_NAMES = {
    "boundary_layer_height": "blh",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
}

# ── File / Directory paths ────────────────────────────────────────────────────
DATA_DIR = Path("data")
RAW_S5P_DIR = DATA_DIR / "raw_s5p"
RAW_ERA5_DIR = DATA_DIR / "raw_era5"
RAW_ERA5_SOLAR_DIR = DATA_DIR / "raw_era5_solar"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")

NO2_DAILY_CSV = PROCESSED_DIR / "no2_daily.csv"
ERA5_DAILY_CSV = PROCESSED_DIR / "era5_daily.csv"
SOLAR_DAILY_CSV = PROCESSED_DIR / "solar_daily.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
PREDICTIONS_CACHE = PROCESSED_DIR / "predictions_cache.json"
MODEL_PATH = MODELS_DIR / "ensemble_no2.pkl"

# ── Model hyperparameters ─────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test fraction is the remainder (0.15)

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 100,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": 0.7,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

RIDGE_PARAMS = {
    "alpha": 1.0,
}

# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
