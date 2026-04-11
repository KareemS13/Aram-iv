"""
app.py — Flask web interface for the Yerevan NO2 forecasting system.

Predictions are pre-computed by the bi-weekly GitHub Actions workflow and
stored in data/processed/predictions_cache.json. The /predict route reads
that cache and returns the result for tomorrow instantly.
"""

import json
from datetime import date, timedelta

import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory

from config import FIGURES_DIR, NO2_DAILY_CSV, PREDICTIONS_CACHE

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not PREDICTIONS_CACHE.exists():
            return jsonify({
                "status": "error",
                "message": "Predictions not available yet. Data refreshes automatically every 2 weeks.",
            }), 503

        cache = json.loads(PREDICTIONS_CACHE.read_text())
        today = date.today().isoformat()

        # Return the nearest cached prediction whose date is >= today
        # (tolerates timezone drift between local cache generation and UTC server)
        entry = next(
            (p for p in sorted(cache.get("predictions", []), key=lambda x: x["target_date"])
             if p["target_date"] >= today),
            None,
        )

        if entry is None:
            return jsonify({
                "status": "error",
                "message": "Predictions are stale. The next scheduled refresh will update them.",
            }), 503

        # Latest observed NO2 from S5P (has ~3-5 day satellite lag)
        latest_observed = None
        latest_observed_date = None
        try:
            no2 = pd.read_csv(NO2_DAILY_CSV, index_col="date", parse_dates=True)
            no2_valid = no2["no2_mean"].dropna()
            if len(no2_valid):
                latest_observed = round(float(no2_valid.iloc[-1]) * 1e6, 2)
                latest_observed_date = no2_valid.index[-1].date().isoformat()
        except Exception:
            pass

        return jsonify({
            "status": "ok",
            **entry,
            "generated_at": cache.get("generated_at"),
            "forecast_14d": cache.get("predictions", []),
            "latest_observed": latest_observed,
            "latest_observed_date": latest_observed_date,
        })

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
