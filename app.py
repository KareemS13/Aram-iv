"""
app.py — Flask web interface for the Yerevan NO2 forecasting system.

Predictions are pre-computed by the bi-weekly GitHub Actions workflow and
stored in data/processed/predictions_cache.json. The /predict route reads
that cache and returns the result for tomorrow instantly.
"""

import json
from datetime import date, timedelta

from flask import Flask, jsonify, render_template, send_from_directory

from config import FIGURES_DIR, PREDICTIONS_CACHE

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

        return jsonify({"status": "ok", **entry, "generated_at": cache.get("generated_at")})

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
