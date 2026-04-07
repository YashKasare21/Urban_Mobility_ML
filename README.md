# Urban Mobility ML

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚦 Live Demo

**[Live Dashboard](https://urbanmobilityml-bc2fd9cka9snxhswzmknqx.streamlit.app/)**

---

## Overview

**Urban Mobility ML** is a real-time traffic prediction system that uses **Sensor Fusion** to combine live traffic and weather data to predict congestion levels at Nala Sopara, Maharashtra.

### Architecture Overview

The system implements a complete data pipeline:

1. **Data Ingestion Layer**: Fetches live data from TomTom Traffic API and OpenWeatherMap API
2. **Data Processing Layer**: Cleans data, generates synthetic training samples, and engineers features
3. **ML Training Pipeline**: Trains multiple models (Logistic Regression, GMM, MLP, PCA) satisfying university lab outcomes
4. **Deployment Layer**: Streamlit dashboard with Folium maps for visualization

### Sensor Fusion Concept

The core innovation is combining multiple data sources:
- **TomTom Traffic API**: Real-time `currentSpeed` and `freeFlowSpeed` 
- **OpenWeatherMap API**: `temp`, `humidity`, `visibility`, `weather_main`

These features are fused to predict whether traffic is **Clear (0)** or **Congested (1)**.

---

## Machine Learning Pipeline

This project satisfies all required lab outcomes (LO3-LO6):

| Lab Outcome | Algorithm | Description | Status |
|------------|-----------|-------------|--------|
| **LO3** | Logistic Regression (GridSearchCV) | Linear classification with hyperparameter tuning (C, solver) | **Production** |
| **LO4** | Gaussian Mixture Model | EM clustering for hidden traffic patterns | Completed |
| **LO5** | Single Layer Perceptron | MLPClassifier with zero hidden layers | Completed |
| **LO6** | PCA | Dimensionality reduction to 2 components | Completed |

**Best Model**: Logistic Regression with GridSearchCV is used in production for its superior accuracy.

---

## Tech Stack

### Core Technologies
- **Python 3.12** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-Learn** - Machine learning

### APIs
- [TomTom Traffic API](https://developer.tomtom.com/traffic-api) - Real-time traffic flow data
- [OpenWeatherMap API](https://openweathermap.org/api) - Weather data

### Deployment
- **Streamlit** - Interactive dashboard
- **Folium** - Interactive maps
- **Docker** - Containerization

---

## Project Structure

```
Urban_Mobility_ML/
├── .env                    # API keys (not committed)
├── Dockerfile              # Docker container
├── .dockerignore          # Docker ignore rules
├── requirements.txt       # Python dependencies
├── 01_data_fetcher.py     # Fetch live data from APIs
├── 02_preprocessing.py    # Data cleaning & feature engineering
├── 02b_generate_mock_data.py  # Synthetic training data
├── 03_model_training.py  # ML pipeline (LO3-LO6)
├── app.py                 # Streamlit dashboard
├── models/
│   ├── traffic_model.pkl # Trained classifier (LogisticRegression)
│   └── scaler.pkl        # Fitted StandardScaler
├── data/
│   ├── live_traffic.csv  # Raw data
│   └── processed_traffic.csv # Cleaned data
├── screenshots/          # Dashboard screenshots
└── README.md
```

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YashKasare21/Urban_Mobility_ML.git
cd Urban_Mobility_ML
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
TOMTOM_API_KEY=your_tomtom_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

Get free API keys:
- **TomTom**: https://developer.tomtom.com/
- **OpenWeatherMap**: https://openweathermap.org/api

### 3. Run with Docker

```bash
docker build -t urban-mobility .
docker run -p 8501:8501 urban-mobility
```

The dashboard opens at `http://localhost:8501`

### 4. Or Run Locally

```bash
pip install -r requirements.txt
python3 01_data_fetcher.py
python3 02b_generate_mock_data.py
python3 02_preprocessing.py
python3 03_model_training.py
streamlit run app.py
```

---

## Dashboard Features

### Tab 1: Prediction Dashboard
- Real-time traffic metrics (speed, congestion ratio)
- Weather data visualization
- ML prediction (Clear/Jam)
- Interactive Folium map with dynamic markers

### Tab 2: Historical Trends
- Line chart showing traffic speed over time
- Temperature vs congestion scatter plot
- Data statistics

### Tab 3: Live Map
- Interactive Folium map centered on Nala Sopara
- Location marker with popup details

---

## Screenshots

### Dashboard View
![Dashboard](./screenshots/dashboard_v2.png)
*Live Urban Mobility Control Center with tabs*

### Traffic Jam Prediction
![Jam Prediction](./screenshots/prediction_jam.png)
*Red alert when congestion detected*

### Historical Trends
![Historical Trends](./screenshots/historical_trends.png)
*Traffic speed line chart over time*

### Live Map
![Live Map](./screenshots/live_map.png)
*Interactive Folium map with location marker*

---

## Model Performance

- **Logistic Regression (GridSearchCV)**: Best accuracy after hyperparameter tuning
- **GMM**: 2 components for traffic pattern discovery
- **MLP Perceptron**: Single layer neural network
- **PCA**: 2 components retaining majority of variance

The tuned `traffic_model.pkl` and `scaler.pkl` are saved in the `models/` directory for inference.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Author

**Yash Kasare**
- Department: Computer Engineering
- Institution: Universal College of Engineering
- Location: Mumbai, Maharashtra, India