"""
pipeline.py — CLI entry point for the Yerevan NO2 forecasting system.

Sub-commands
------------
ingest      Download Sentinel-5P and ERA5 data for a date range.
process     Extract NO2 and ERA5 daily series from raw files.
features    Build the feature matrix CSV.
train       Train the LightGBM model and evaluate on the test set.
predict     Run next-day inference and print result as JSON.
pipeline    Run all stages end-to-end.

Usage examples
--------------
    python pipeline.py ingest   --start-date 2019-01-01 --end-date 2024-12-31
    python pipeline.py process
    python pipeline.py features
    python pipeline.py train
    python pipeline.py predict
    python pipeline.py predict  --target-date 2025-06-01
    python pipeline.py pipeline --start-date 2019-01-01 --end-date 2024-12-31

Credentials
-----------
    GEE_PROJECT   GEE Cloud project ID (env var, or pass --gee-project)
    ERA5 credentials must be configured in ~/.cdsapirc
"""

import argparse
import json
import sys
from datetime import date, timedelta

from config import (
    ERA5_DAILY_CSV,
    FEATURES_CSV,
    FIGURES_DIR,
    MODEL_PATH,
    NO2_DAILY_CSV,
    RAW_ERA5_DIR,
    get_logger,
)

logger = get_logger("pipeline")

DEFAULT_START = date(2019, 1, 1)
DEFAULT_GEE_PROJECT = "ee-kareemsaffarini9"


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Fetch S5P NO2 via GEE and ERA5 monthly files via CDS."""
    from s5p_client import S5PClient
    from era5_client import ERA5Client

    start = args.start_date
    end = args.end_date
    gee_project = getattr(args, "gee_project", DEFAULT_GEE_PROJECT)
    logger.info("Ingesting data from %s to %s", start, end)

    # Sentinel-5P via GEE (produces NO2 daily CSV directly)
    s5p = S5PClient(project=gee_project, output_csv=NO2_DAILY_CSV)
    no2_df = s5p.fetch(start, end)
    logger.info("S5P NO2 daily series: %d days", len(no2_df))

    # ERA5 via CDS
    era5 = ERA5Client(download_dir=RAW_ERA5_DIR)
    nc_paths = era5.fetch_date_range(start, end)
    logger.info("Downloaded %d ERA5 monthly file(s)", len(nc_paths))


def cmd_process(_args: argparse.Namespace) -> None:
    """Build ERA5 daily CSV from raw NetCDF files."""
    from era5_processor import ERA5Processor

    nc_paths = sorted(RAW_ERA5_DIR.glob("era5_*.nc"))
    if not nc_paths:
        logger.error("No ERA5 NC files found in %s. Run 'ingest' first.", RAW_ERA5_DIR)
        sys.exit(1)

    era5_proc = ERA5Processor(output_csv=ERA5_DAILY_CSV)
    era5_df = era5_proc.process_all(nc_paths)
    logger.info("ERA5 daily covariates: %d days", len(era5_df))


def cmd_features(_args: argparse.Namespace) -> None:
    """Build the feature matrix CSV from processed daily CSVs."""
    from features import FeatureEngineer

    fe = FeatureEngineer(
        no2_csv=NO2_DAILY_CSV,
        era5_csv=ERA5_DAILY_CSV,
        output_csv=FEATURES_CSV,
    )
    df = fe.build()
    logger.info("Feature matrix: %d rows × %d columns", len(df), len(df.columns))


def cmd_train(_args: argparse.Namespace) -> None:
    """Train LightGBM model and evaluate on the held-out test set."""
    from model import NO2Forecaster

    forecaster = NO2Forecaster(
        features_csv=FEATURES_CSV,
        model_path=MODEL_PATH,
        figures_dir=FIGURES_DIR,
    )
    model = forecaster.train()
    forecaster.evaluate(model)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run next-day inference and print JSON result."""
    from inference import InferencePipeline

    target = args.target_date
    pipeline = InferencePipeline(
        model_path=MODEL_PATH,
        no2_csv=NO2_DAILY_CSV,
        era5_csv=ERA5_DAILY_CSV,
    )
    result = pipeline.predict(target_date=target)
    print(json.dumps(result, indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run all stages end-to-end: ingest → process → features → train → predict."""
    if not hasattr(args, "gee_project"):
        args.gee_project = DEFAULT_GEE_PROJECT
    cmd_ingest(args)
    cmd_process(args)
    cmd_features(args)
    cmd_train(args)
    cmd_predict(args)


# ── CLI definition ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Yerevan tropospheric NO2 forecasting system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared date arguments
    def add_date_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--start-date",
            type=_parse_date,
            default=DEFAULT_START,
            metavar="YYYY-MM-DD",
            help=f"Start date (default: {DEFAULT_START})",
        )
        p.add_argument(
            "--end-date",
            type=_parse_date,
            default=date.today() - timedelta(days=1),
            metavar="YYYY-MM-DD",
            help="End date (default: yesterday)",
        )

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Fetch S5P via GEE and ERA5 via CDS")
    add_date_args(p_ingest)
    p_ingest.add_argument("--gee-project", default=DEFAULT_GEE_PROJECT,
                          help=f"GEE Cloud project ID (default: {DEFAULT_GEE_PROJECT})")
    p_ingest.set_defaults(func=cmd_ingest)

    # process
    p_process = subparsers.add_parser("process", help="Build daily NO2 and ERA5 CSVs")
    p_process.set_defaults(func=cmd_process)

    # features
    p_features = subparsers.add_parser("features", help="Build feature matrix CSV")
    p_features.set_defaults(func=cmd_features)

    # train
    p_train = subparsers.add_parser("train", help="Train and evaluate LightGBM model")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = subparsers.add_parser("predict", help="Run next-day NO2 inference")
    p_predict.add_argument(
        "--target-date",
        type=_parse_date,
        default=None,
        metavar="YYYY-MM-DD",
        help="Date to forecast (default: tomorrow)",
    )
    p_predict.set_defaults(func=cmd_predict)

    # pipeline (all stages)
    p_all = subparsers.add_parser("pipeline", help="Run all stages end-to-end")
    add_date_args(p_all)
    p_all.add_argument("--gee-project", default=DEFAULT_GEE_PROJECT,
                       help=f"GEE Cloud project ID (default: {DEFAULT_GEE_PROJECT})")
    p_all.add_argument("--target-date", type=_parse_date, default=None,
                       metavar="YYYY-MM-DD", help="Date to forecast (default: tomorrow)")
    p_all.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
