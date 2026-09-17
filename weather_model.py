"""
Weather prediction model utilities.
Fetches data from OpenWeatherMap, trains a Random Forest Classifier,
and provides prediction + evaluation helpers.
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEFAULT_CITY = "New York"
DEFAULT_UNITS = "metric"  # Celsius


def get_api_key():
    """Return API key from environment or raise a clear error."""
    key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not key or key == "your-api-key":
        raise ValueError(
            "OPENWEATHER_API_KEY is not set. "
            "Create a free key at https://openweathermap.org/api and put it in a .env file."
        )
    return key


# -------------------------------------------------
# Data Fetching
# -------------------------------------------------
def fetch_current_weather(city: str = DEFAULT_CITY) -> dict:
    """Fetch current weather for a city."""
    api_key = get_api_key()
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": DEFAULT_UNITS}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country", ""),
        "temperature": round(data["main"]["temp"], 1),
        "feels_like": round(data["main"]["feels_like"], 1),
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": round(data.get("wind", {}).get("speed", 0), 1),
        "description": data["weather"][0]["description"].title(),
        "icon": data["weather"][0]["icon"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def fetch_forecast(city: str = DEFAULT_CITY, cnt: int = 40) -> list:
    """
    Fetch 5-day / 3-hour forecast (up to 40 entries).
    Returns a list of dicts suitable for training and display.
    """
    api_key = get_api_key()
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": api_key, "units": DEFAULT_UNITS, "cnt": cnt}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for item in data.get("list", []):
        records.append({
            "datetime": item["dt_txt"],
            "temperature": round(item["main"]["temp"], 1),
            "humidity": item["main"]["humidity"],
            "pressure": item["main"]["pressure"],
            "wind_speed": round(item.get("wind", {}).get("speed", 0), 1),
            "description": item["weather"][0]["description"].title(),
            "main": item["weather"][0]["main"],
        })
    return records


def build_training_dataframe(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """Build a DataFrame from forecast data for model training."""
    records = fetch_forecast(city)
    if not records:
        raise RuntimeError("No forecast data returned from API.")
    df = pd.DataFrame(records)
    return df


# -------------------------------------------------
# Model Training & Evaluation
# -------------------------------------------------
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest Classifier to predict weather 'main' condition
    from temperature, humidity, pressure and wind speed.
    Returns (model, label_encoder, metrics_dict, feature_names)
    """
    feature_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    X = df[feature_cols].values
    y_raw = df["main"].values  # e.g. Rain, Clouds, Clear ...

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # If too few samples for a split, train on everything and use dummy metrics
    if len(df) < 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 2),
            "precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 2),
            "recall": round(recall_score(y, y_pred, average="weighted", zero_division=0), 2),
            "f1": round(f1_score(y, y_pred, average="weighted", zero_division=0), 2),
        }
        return model, le, metrics, feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 2),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 2),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 2),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 2),
    }
    return model, le, metrics, feature_cols


def predict_condition(model, label_encoder, temperature: float, humidity: float,
                      pressure: float = 1013.0, wind_speed: float = 3.0) -> dict:
    """
    Predict weather condition from user-supplied temperature & humidity
    (pressure & wind use sensible defaults if not provided).
    """
    features = np.array([[temperature, humidity, pressure, wind_speed]])
    pred_idx = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    # Probability / confidence of the predicted class
    proba = model.predict_proba(features)[0]
    confidence = round(float(proba[pred_idx]) * 100)

    return {
        "condition": label,
        "confidence": confidence,
        "all_probabilities": {
            label_encoder.inverse_transform([i])[0]: round(float(p) * 100)
            for i, p in enumerate(proba)
        },
    }


def save_model(model, label_encoder, path: str = "weather_rf_model.joblib"):
    joblib.dump({"model": model, "label_encoder": label_encoder}, path)


def load_model(path: str = "weather_rf_model.joblib"):
    data = joblib.load(path)
    return data["model"], data["label_encoder"]
