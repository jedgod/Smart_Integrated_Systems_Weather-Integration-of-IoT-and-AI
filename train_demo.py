"""
Standalone script that demonstrates the full pipeline
(without the web UI). Useful for verification / grading.

Usage:
    export OPENWEATHER_API_KEY=your_key
    python train_demo.py
"""

import os
from dotenv import load_dotenv
from weather_model import (
    fetch_current_weather,
    build_training_dataframe,
    train_model,
    predict_condition,
)

load_dotenv()

CITY = "New York"

def main():
    print("=" * 60)
    print("CTEC 671 – Project 2  |  IoT + AI Weather Prediction Demo")
    print("=" * 60)

    # 1. Current weather
    print(f"\n[1] Fetching current weather for {CITY} ...")
    try:
        current = fetch_current_weather(CITY)
        print(f"    Location    : {current['city']}, {current['country']}")
        print(f"    Temperature : {current['temperature']} °C")
        print(f"    Humidity    : {current['humidity']} %")
        print(f"    Pressure    : {current['pressure']} hPa")
        print(f"    Wind Speed  : {current['wind_speed']} m/s")
        print(f"    Condition   : {current['description']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        print("    (Set a valid OPENWEATHER_API_KEY to use live data)")
        return

    # 2. Build training set from forecast
    print(f"\n[2] Building training data from forecast ...")
    df = build_training_dataframe(CITY)
    print(f"    Samples collected: {len(df)}")
    print(df[["datetime", "temperature", "humidity", "main"]].head())

    # 3. Train model
    print(f"\n[3] Training Random Forest Classifier ...")
    model, le, metrics, feats = train_model(df)
    print(f"    Features used : {feats}")
    print(f"    Accuracy      : {metrics['accuracy']}")
    print(f"    Precision     : {metrics['precision']}")
    print(f"    Recall        : {metrics['recall']}")
    print(f"    F1-Score      : {metrics['f1']}")

    # 4. Predict
    print(f"\n[4] Predicting weather for Temperature=30°C, Humidity=65% ...")
    result = predict_condition(model, le, temperature=30, humidity=65)
    print(f"    Predicted condition : {result['condition']}")
    print(f"    Confidence          : {result['confidence']} %")
    print(f"    Class probabilities : {result['all_probabilities']}")

    print("\n" + "=" * 60)
    print("Demo complete. Run 'python app.py' for the full dashboard.")
    print("=" * 60)

if __name__ == "__main__":
    main()
