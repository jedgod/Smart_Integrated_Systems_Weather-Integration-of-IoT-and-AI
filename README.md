# CTEC 671 – Integration of IoT and AI  
## Project 2: Weather Prediction Dashboard

A complete web application that demonstrates the integration of **IoT-style weather data** (via the OpenWeatherMap API) with **Artificial Intelligence** (Random Forest Classifier) for weather condition prediction.

The user interface is designed to closely match the provided project dashboard mock-up.

---

### Features

| Feature | Description |
|---------|-------------|
| **Live Weather Cards** | Temperature, Humidity, Pressure, Wind Speed, Current Condition |
| **AI Prediction** | Enter temperature & humidity → predict weather condition + confidence |
| **Trend Chart** | Interactive temperature / humidity forecast chart (Chart.js) |
| **Recent Data Table** | Latest readings from OpenWeatherMap |
| **Model Performance** | Accuracy, Precision, Recall, F1-Score of the trained Random Forest |
| **Multi-city** | Switch between New York, London, Tokyo, Sydney, Dubai, Paris |
| **Responsive** | Works on desktop and tablet |

---

### Project Structure

```
Weather_Prediction_IoT_AI/
├── app.py                  # Flask application (main entry point)
├── weather_model.py        # Data fetching + Random Forest training/prediction
├── requirements.txt        # Python dependencies
├── .env.example            # Template for your API key
├── README.md               # This file
├── templates/
│   └── dashboard.html      # Main dashboard UI
└── static/
    └── css/
        └── dashboard.css   # Styles matching the mock-up
```

---

### Prerequisites

- Python 3.9 or higher
- A free OpenWeatherMap API key  
  → Sign up at https://openweathermap.org/api (Current Weather + 5-day Forecast are free)

---

### Setup & Run

1. **Open a terminal** and go into the project folder:
   ```bash
   cd Weather_Prediction_IoT_AI
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key**:
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env and replace the placeholder with your real key
   # OPENWEATHER_API_KEY=your_real_key_here
   ```

5. **Start the application**:
   ```bash
   python app.py
   ```

6. **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

---

### How the AI Works

1. The app calls the OpenWeatherMap **5-day / 3-hour forecast** endpoint for the selected city.
2. From each forecast entry it extracts:  
   `temperature`, `humidity`, `pressure`, `wind_speed` (features)  
   and the weather `main` category (target label: Rain, Clouds, Clear, …).
3. A **Random Forest Classifier** (scikit-learn) is trained on these samples.
4. When you click **Predict Weather**, the model receives your temperature & humidity values (plus default pressure/wind) and returns the predicted condition together with a confidence percentage.
5. Evaluation metrics (Accuracy, Precision, Recall, F1) are computed on a held-out test split and shown on the dashboard.

> **Note:** The original project sample used Linear Regression on text labels, which is not ideal. This implementation uses a proper multi-class classifier, matching the “Random Forest Classifier” shown in the UI mock-up.

---

### Demo Mode

If no valid `OPENWEATHER_API_KEY` is set, the dashboard still loads with realistic demo numbers and a fallback prediction so you can explore the interface immediately.

---

### Sample Prediction Flow

1. Leave Temperature = 30 °C and Humidity = 65 %.
2. Click **Predict Weather**.
3. The model returns a condition (e.g. “Rain” / “Clouds”) and a confidence score.
4. Change the values and predict again to see different results.

---

### Technologies Used

- **Backend:** Python, Flask  
- **Machine Learning:** scikit-learn (RandomForestClassifier)  
- **Data Source:** OpenWeatherMap API (IoT-style public weather data)  
- **Frontend:** HTML5, Bootstrap 5, Chart.js, custom CSS  
- **Environment:** python-dotenv  

---

### Course Deliverables Mapping

| Requirement | How it is satisfied |
|-------------|---------------------|
| Retrieve historical / forecast data | `fetch_forecast()` & `fetch_current_weather()` |
| Extract temperature & humidity | Feature extraction in `weather_model.py` |
| Train ML algorithm | Random Forest Classifier training |
| Predict future weather | `/api/predict` endpoint + UI form |
| Evaluate accuracy | Accuracy / Precision / Recall / F1 displayed |
| User interface | Full interactive dashboard matching the provided design |

---

### Author / Course

**CTEC 671 – Smart Integrated Systems / Integration of IoT and AI**  
Project 2 – due May 7, 2023 (adapted for current use)

---

### License

This project is provided for educational purposes as part of CTEC 671 coursework.
