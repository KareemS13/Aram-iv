"""
app.py — Flask web interface for the Yerevan NO2 forecasting system.

Single page with a "Run Prediction" button that:
  1. Fetches latest S5P NO2 via GEE (incremental)
  2. Fetches latest ERA5 monthly files via CDS (skips existing)
  3. Re-processes ERA5 daily covariates
  4. Runs next-day inference
  5. Returns the result as JSON
"""

from datetime import date, timedelta

from flask import Flask, jsonify, render_template, send_from_directory

from config import ERA5_DAILY_CSV, FIGURES_DIR, MODEL_PATH, NO2_DAILY_CSV, RAW_ERA5_DIR

GEE_PROJECT = "ee-kareemsaffarini9"
DEFAULT_START = date(2019, 1, 1)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        end = date.today() - timedelta(days=1)

        # 1. Fetch latest NO2 from GEE (incremental)
        from s5p_client import S5PClient
        s5p = S5PClient(project=GEE_PROJECT, output_csv=NO2_DAILY_CSV)
        s5p.fetch(DEFAULT_START, end)

        # 2. Fetch latest ERA5 monthly files (skips already-downloaded months)
        from era5_client import ERA5Client
        era5_client = ERA5Client(download_dir=RAW_ERA5_DIR)
        era5_client.fetch_date_range(DEFAULT_START, end)

        # 3. Re-process all ERA5 NetCDF → daily CSV
        from era5_processor import ERA5Processor
        nc_paths = sorted(RAW_ERA5_DIR.glob("era5_*.nc"))
        ERA5Processor(output_csv=ERA5_DAILY_CSV).process_all(nc_paths)

        # 4. Run inference
        from inference import InferencePipeline
        pipeline = InferencePipeline(
            model_path=MODEL_PATH,
            no2_csv=NO2_DAILY_CSV,
            era5_csv=ERA5_DAILY_CSV,
        )
        result = pipeline.predict()

        return jsonify({"status": "ok", **result})

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
