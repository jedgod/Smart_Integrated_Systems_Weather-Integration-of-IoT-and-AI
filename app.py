"""
CTEC 671 – Integration of IoT and AI
Project 2: Weather Prediction Dashboard

Run with:
    export OPENWEATHER_API_KEY=your_key_here
    python app.py
Then open http://127.0.0.1:5000
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from weather_model import (
    fetch_current_weather,
    fetch_forecast,
    build_training_dataframe,
    train_model,
    predict_condition,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "ctec671-weather-demo-secret")

# In-memory cache so we don't retrain on every request
_cache = {
    "model": None,
    "label_encoder": None,
    "metrics": None,
    "feature_cols": None,
    "forecast": None,
    "current": None,
    "city": "New York",
    "last_trained": None,
}


def ensure_model(city: str = "New York", force: bool = False):
    """Fetch data & train model if not already cached (or if city changed)."""
    if (
        not force
        and _cache["model"] is not None
        and _cache["city"] == city
    ):
        return

    current = fetch_current_weather(city)
    forecast = fetch_forecast(city)
    df = build_training_dataframe(city)
    model, le, metrics, feats = train_model(df)

    _cache.update({
        "model": model,
        "label_encoder": le,
        "metrics": metrics,
        "feature_cols": feats,
        "forecast": forecast,
        "current": current,
        "city": city,
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/")
def dashboard():
    city = request.args.get("city", _cache.get("city", "New York"))
    try:
        ensure_model(city)
        error = None
    except Exception as e:
        error = str(e)
        # Provide fallback demo data so the UI still renders
        _cache["current"] = {
            "city": city, "country": "US",
            "temperature": 24.8, "humidity": 62, "pressure": 1013,
            "wind_speed": 5.6, "description": "Partly Cloudy",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _cache["forecast"] = []
        _cache["metrics"] = {"accuracy": 0.87, "precision": 0.86, "recall": 0.84, "f1": 0.85}

    current = _cache["current"]
    forecast = _cache.get("forecast") or []
    metrics = _cache.get("metrics") or {}

    # Prepare chart data (temperature trend from forecast)
    chart_labels = [r["datetime"][11:16] for r in forecast[:16]]  # HH:MM
    chart_temps = [r["temperature"] for r in forecast[:16]]
    chart_humidity = [r["humidity"] for r in forecast[:16]]

    # Recent table rows (newest first)
    recent = list(reversed(forecast[:8])) if forecast else []

    return render_template(
        "dashboard.html",
        current=current,
        metrics=metrics,
        recent=recent,
        chart_labels=chart_labels,
        chart_temps=chart_temps,
        chart_humidity=chart_humidity,
        city=city,
        error=error,
        last_trained=_cache.get("last_trained"),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    try:
        temp = float(data.get("temperature", 30))
        humidity = float(data.get("humidity", 65))
        pressure = float(data.get("pressure", 1013))
        wind = float(data.get("wind_speed", 3.0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric input"}), 400

    city = data.get("city", _cache.get("city", "New York"))
    try:
        ensure_model(city)
        if _cache["model"] is None:
            raise RuntimeError("Model not available")
        result = predict_condition(
            _cache["model"], _cache["label_encoder"],
            temperature=temp, humidity=humidity,
            pressure=pressure, wind_speed=wind,
        )
        return jsonify(result)
    except Exception as e:
        # Demo fallback so the UI still works without a valid API key
        return jsonify({
            "condition": "Light Rain",
            "confidence": 82,
            "all_probabilities": {"Light Rain": 82, "Clouds": 12, "Clear": 6},
            "note": f"Demo prediction (live model unavailable: {e})",
        })


@app.route("/api/refresh")
def api_refresh():
    city = request.args.get("city", "New York")
    try:
        ensure_model(city, force=True)
        return jsonify({"status": "ok", "city": city, "trained_at": _cache["last_trained"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/current")
def api_current():
    city = request.args.get("city", "New York")
    try:
        data = fetch_current_weather(city)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
