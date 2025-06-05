# 🏁 2025 F1 Predictions

![Open Source](https://img.shields.io/badge/Open%20Source-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastF1](https://img.shields.io/badge/FastF1-API-orange?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?style=for-the-badge&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Science-purple?style=for-the-badge&logo=pandas)

---

# 🏎️ F1 Predictions 2025

**Predict Formula 1 race outcomes for the 2025 season using real F1 data, machine learning, and weather conditions via FastF1.**

---

## ✨ Features
- Interactive CLI: Select any configured 2025 race and get instant predictions
- Uses real F1 data (laps, qualifying, weather) via FastF1
- Weather-aware: Model incorporates temperature, humidity, rain probability, and wind speed
- Machine learning (Gradient Boosting) for robust, data-driven predictions
- Displays model error (MAE) for transparency
- Easily extensible for new races or features

---

## 🧠 How It Works
1. **Race Selection:** User picks a race from the list (2025 calendar, mapped to 2024 data)
2. **Data Loading:** FastF1 loads historical race and weather data for the selected event
3. **Feature Engineering:** Combines qualifying times, driver stats, and weather conditions
4. **Model Training:** Trains a Gradient Boosting Regressor on historical data
5. **Prediction:** Predicts race results for the selected event, factoring in weather
6. **Display:** Shows predicted finishing order, times, and model error

---

## 🛠️ Setup & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/djdestinycodes/2025_f1_predictions.git
   cd 2025_f1_predictions
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the interactive prediction system:**
   ```bash
   python src/main.py
   ```
5. **Follow the prompts** to select a race and view predictions.

---

## 🗂️ Project Structure
```
2025_f1_predictions/
├── src/
│   ├── main.py            # Main interactive CLI
│   └── race_predictor.py  # Core ML and prediction logic
├── data/
│   ├── f1_cache/          # FastF1 cache (auto-generated)
│   └── race_config.py     # Race and qualifying config (with weather)
├── archive/               # Standalone scripts for past races/experiments
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Configuration & Customization
- **Races:** All races are defined in `data/race_config.py`.
- **Weather:** Each race config includes temperature, humidity, rain probability, and wind speed. These are used for predictions and can be edited for "what-if" scenarios.
- **Adding a Race:** Copy an existing entry in `race_config.py`, update the name, race number, qualifying data, and weather.
- **Model:** The ML model is defined in `src/race_predictor.py` and can be tuned or swapped for other regressors.

---

## 🧩 How Weather Data Works
- **Source:** Weather data is pulled from FastF1's historical session data (when available) for maximum realism.
- **Fallback:** If FastF1 weather is missing, the config values in `race_config.py` are used.
- **No external APIs:** No OpenWeather or other services required—everything is F1-native!

---

## 📈 Model Performance
- Evaluated using **Mean Absolute Error (MAE)**
- Lower MAE = more accurate predictions
- MAE is shown after each prediction for transparency

---

## 🗃️ Archive Directory
The `archive/` folder contains standalone scripts for each race, named in the format:
- `<race_name>_<year>.py` (e.g., `monaco_2024.py`, `silverstone_2024.py`)
- Prediction experiments: `prediction_<date>_<description>.py`

These scripts are preserved for reference, reproducibility, and experimentation. The main workflow is now unified and interactive.

---

## 🧑‍💻 Troubleshooting
- **FastF1 cache issues:** If you see errors about missing data, try deleting the `data/f1_cache/` folder and rerunning.
- **API/data errors:** Ensure you have a stable internet connection for FastF1 to fetch data.
- **Python version:** Use Python 3.11+ for best compatibility.

---

## 🤝 Contributing & License
- **Open Source** under the MIT License
---

## 📣 Thanks!!!!!!
- **Credits:** This project is based on the original repository by [mar-antaya](https://github.com/mar-antaya/2025_f1_predictions).

---

🏎️ **Start predicting F1 races like a data scientist!** 🚦

