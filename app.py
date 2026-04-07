import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import importlib
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

load_dotenv()

data_fetcher = importlib.import_module("01_data_fetcher")
fetch_live_data = data_fetcher.fetch_live_data


def main():
    st.set_page_config(
        page_title="Live Urban Mobility Predictor", layout="wide", page_icon="🚦"
    )

    if "live_data" not in st.session_state:
        st.session_state.live_data = None

    st.title("🚦 Live Urban Mobility Control Center")
    st.markdown("---")

    LAT = 19.4167
    LON = 72.8167

    tab1, tab2, tab3 = st.tabs(
        ["📊 Prediction Dashboard", "📈 Historical Trends", "🗺️ Live Map"]
    )

    with tab1:
        col_main, col_info = st.columns([3, 1])

        with col_main:
            st.header("Live Traffic Prediction")
            if st.button(
                "Check Live Traffic Status", type="primary", use_container_width=True
            ):
                tomtom_key = os.getenv("TOMTOM_API_KEY")
                weather_key = os.getenv("OPENWEATHER_API_KEY")
                lat = LAT
                lon = LON

                with st.spinner("Fetching live data from TomTom & OpenWeather..."):
                    live_data = fetch_live_data(tomtom_key, weather_key, lat, lon)

                if (
                    live_data["currentSpeed"] is None
                    or live_data["freeFlowSpeed"] is None
                ):
                    st.error("Failed to fetch traffic data from TomTom API")
                    st.stop()

                current_speed = live_data["currentSpeed"]
                free_flow = live_data["freeFlowSpeed"]
                temp = live_data["temp"]
                humidity = live_data["humidity"]
                visibility = live_data["visibility"]
                weather_main = live_data["weather_main"]

                congestion_ratio = current_speed / free_flow if free_flow else 0

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

                scaler = joblib.load("models/scaler.pkl")
                model = joblib.load("models/traffic_model.pkl")

                X_live_scaled = scaler.transform(X_live)
                prediction = model.predict(X_live_scaled)[0]

                st.session_state.live_data = {
                    "current_speed": current_speed,
                    "free_flow": free_flow,
                    "temp": temp,
                    "humidity": humidity,
                    "visibility": visibility,
                    "weather_main": weather_main,
                    "congestion_ratio": congestion_ratio,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat(),
                }

            if st.session_state.live_data is not None:
                data = st.session_state.live_data
                current_speed = data["current_speed"]
                free_flow = data["free_flow"]
                temp = data["temp"]
                humidity = data["humidity"]
                visibility = data["visibility"]
                weather_main = data["weather_main"]
                congestion_ratio = data["congestion_ratio"]
                prediction = data["prediction"]

                st.markdown("### 🚗 Traffic Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Current Speed", f"{current_speed} km/h", delta_color="inverse"
                )
                m2.metric("Free Flow Speed", f"{free_flow} km/h")
                m3.metric(
                    "Congestion Ratio",
                    f"{congestion_ratio:.2f}",
                    delta="-0.5" if congestion_ratio < 0.7 else "0.5",
                    delta_color="inverse",
                )

                st.markdown("### 🌤️ Weather Data")
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Temperature", f"{temp}°C")
                w2.metric("Humidity", f"{humidity}%")
                w3.metric("Visibility", f"{visibility} m")
                w4.metric("Condition", weather_main)

                st.markdown("---")

                st.markdown("### 🎯 Traffic Status")

                if prediction == 1:
                    st.error("🚨 **TRAFFIC JAM DETECTED!** 🚨")
                    st.markdown(
                        "<h2 style='text-align: center; color: white; background-color: #dc2626; padding: 25px; border-radius: 10px;'>⚠️ CONGESTED - AVOID THIS ROUTE</h2>",
                        unsafe_allow_html=True,
                    )
                    marker_tooltip = "High Traffic"
                else:
                    st.success("✅ **TRAFFIC IS CLEAR!**")
                    st.markdown(
                        "<h2 style='text-align: center; color: white; background-color: #16a34a; padding: 25px; border-radius: 10px;'>🟢 ROUTE CLEAR - GO AHEAD</h2>",
                        unsafe_allow_html=True,
                    )
                    marker_tooltip = "Clear Route"

                st.markdown(f"**Prediction made at:** {data['timestamp']}")

                st.markdown("---")

                st.subheader("🗺️ Live Map - Nala Sopara")
                folium_map = folium.Map(
                    location=[LAT, LON], zoom_start=14, tiles="cartodbpositron"
                )
                folium.Marker(
                    [LAT, LON],
                    popup=f"Traffic: {'Jam' if prediction == 1 else 'Clear'}",
                    tooltip=marker_tooltip,
                    icon=folium.Icon(
                        color="red" if prediction == 1 else "green",
                        icon="warning-sign" if prediction == 1 else "ok-sign",
                    ),
                ).add_to(folium_map)
                st_folium(folium_map, width=700, height=400)

        with col_info:
            st.header("ℹ️ Model Info")
            st.info(
                "**Logistic Regression** model trained on traffic + weather data from Nala Sopara, Maharashtra."
            )
            st.markdown("""
            **Features:**
            - Temperature
            - Humidity
            - Visibility
            - Weather (one-hot)
            
            **Target:**
            - 0 = Clear
            - 1 = Congested
            """)
            st.markdown("---")
            st.caption("📍 Location: Nala Sopara")
            st.caption("Lat: 19.4167, Lon: 72.8167")

    with tab2:
        st.header("📈 Historical Traffic Trends")
        try:
            if os.path.exists("data/processed_traffic.csv"):
                df = pd.read_csv("data/processed_traffic.csv")
            elif os.path.exists("data/live_traffic.csv"):
                df = pd.read_csv("data/live_traffic.csv")
            else:
                raise FileNotFoundError("No data files found")

            st.write(f"**Total Records:** {len(df)}")

            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(str)
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], format="mixed", errors="coerce"
                )
                df = df.sort_values("timestamp")

                st.markdown("### 🚗 Current Speed Over Time")
                df_indexed = df.set_index("timestamp")
                st.line_chart(df_indexed["currentSpeed"])

                if "congestion_ratio" in df.columns:
                    st.markdown("### 🌡️ Temperature vs Congestion Ratio")
                    st.scatter_chart(
                        df, x="temp", y="congestion_ratio", color="traffic_state"
                    )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Data Statistics")
                st.write(df.describe())
            with col2:
                if "traffic_state" in df.columns:
                    st.markdown("### 🏷️ Traffic State Distribution")
                    st.write(df["traffic_state"].value_counts())

        except FileNotFoundError:
            st.warning("No historical data found. Run the data fetcher first!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    with tab3:
        st.header("🗺️ Live Location Map")
        st.markdown(f"**Nala Sopara, Maharashtra** - Lat: {LAT}, Lon: {LON}")

        base_map = folium.Map(
            location=[LAT, LON], zoom_start=14, tiles="cartodbpositron"
        )
        folium.Marker(
            [LAT, LON],
            popup="Nala Sopara - Target Location",
            tooltip="Click for details",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(base_map)
        st_folium(base_map, width=900, height=600)

        st.markdown("""
        ---
        **Map Controls:**
        - 🖱️ Scroll to zoom
        - 🖱️ Drag to pan
        - 📍 Click markers for info
        """)


if __name__ == "__main__":
    main()
