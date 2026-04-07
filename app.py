import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import importlib
from dotenv import load_dotenv

load_dotenv()

data_fetcher = importlib.import_module("01_data_fetcher")
fetch_live_data = data_fetcher.fetch_live_data


def main():
    st.set_page_config(page_title="Live Urban Mobility Predictor", layout="wide")

    st.title("🚦 Live Urban Mobility Predictor")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Live Metrics")
        if st.button(
            "Check Live Traffic Status", type="primary", use_container_width=True
        ):
            tomtom_key = os.getenv("TOMTOM_API_KEY")
            weather_key = os.getenv("OPENWEATHER_API_KEY")
            lat = 19.4167
            lon = 72.8167

            with st.spinner("Fetching live data from TomTom & OpenWeather..."):
                live_data = fetch_live_data(tomtom_key, weather_key, lat, lon)

            if live_data["currentSpeed"] is None or live_data["freeFlowSpeed"] is None:
                st.error("Failed to fetch traffic data from TomTom API")
                return

            current_speed = live_data["currentSpeed"]
            free_flow = live_data["freeFlowSpeed"]
            temp = live_data["temp"]
            humidity = live_data["humidity"]
            visibility = live_data["visibility"]
            weather_main = live_data["weather_main"]

            congestion_ratio = current_speed / free_flow if free_flow else 0

            st.markdown("### Raw Traffic Data")
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Speed", f"{current_speed} km/h")
            m2.metric("Free Flow Speed", f"{free_flow} km/h")
            m3.metric("Congestion Ratio", f"{congestion_ratio:.2f}")

            st.markdown("### Weather Data")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Temperature", f"{temp}°C")
            w2.metric("Humidity", f"{humidity}%")
            w3.metric("Visibility", f"{visibility} m")
            w4.metric("Condition", weather_main)

            st.markdown("---")
            st.markdown("### 🔮 Prediction Pipeline")

            weather_cols = ["Clear", "Clouds", "Mist", "Rain"]
            features = {
                "temp": temp,
                "humidity": humidity,
                "visibility": visibility,
                "Clear": 0,
                "Clouds": 0,
                "Mist": 0,
                "Rain": 0,
            }
            if weather_main in weather_cols:
                features[weather_main] = 1

            X_live = pd.DataFrame([features])
            st.write("Feature vector:", X_live.T)

            scaler = joblib.load("models/scaler.pkl")
            model = joblib.load("models/traffic_model.pkl")

            X_live_scaled = scaler.transform(X_live)
            prediction = model.predict(X_live_scaled)[0]

            st.markdown("---")
            st.markdown("### 🎯 Final Prediction")

            if prediction == 1:
                st.error("🚨 **TRAFFIC JAM DETECTED!** 🚨")
                st.markdown(
                    "<h1 style='text-align: center; color: red; background-color: #ffcccc; padding: 20px; border-radius: 10px;'>⚠️ CONGESTED - AVOID THIS ROUTE</h1>",
                    unsafe_allow_html=True,
                )
            else:
                st.success("**TRAFFIC IS CLEAR!** ")
                st.markdown(
                    "<h1 style='text-align: center; color: green; background-color: #ccffcc; padding: 20px; border-radius: 10px;'>ROUTE CLEAR - GO AHEAD</h1>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"**Prediction made at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

    with col2:
        st.header("Model Info")
        st.info(
            "This app uses a Logistic Regression model trained on traffic and weather data from Nala Sopara, Maharashtra."
        )
        st.markdown("""
        **Features used:**
        - Temperature
        - Humidity  
        - Visibility
        - Weather Condition (one-hot encoded)
        
        **Target:**
        - 0 = Traffic Clear
        - 1 = Traffic Congested
        """)
        st.markdown("---")
        st.markdown("### Historical Data")
        if os.path.exists("data/processed_traffic.csv"):
            df_hist = pd.read_csv("data/processed_traffic.csv")
            st.write(f"Total records: {len(df_hist)}")
            st.write(df_hist.tail(10))
        else:
            st.warning("No historical data found")


if __name__ == "__main__":
    main()
