import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def fetch_live_data(tomtom_key: str, weather_key: str, lat: float, lon: float) -> dict:
    """
    Fetches live traffic and weather data from TomTom and OpenWeatherMap APIs.

    Args:
        tomtom_key: TomTom API key
        weather_key: OpenWeatherMap API key
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary containing traffic and weather data with timestamp
    """
    headers = {"Content-Type": "application/json"}

    tomtom_url = f"{TOMTOM_BASE_URL}/10/json"
    tomtom_params = {"key": tomtom_key, "point": f"{lat},{lon}"}

    weather_params = {"lat": lat, "lon": lon, "appid": weather_key, "units": "metric"}

    data = {"timestamp": datetime.now().isoformat()}

    try:
        tomtom_response = requests.get(
            tomtom_url, params=tomtom_params, headers=headers, timeout=10
        )
        tomtom_response.raise_for_status()
        tomtom_data = tomtom_response.json()

        if "flowSegmentData" in tomtom_data:
            flow_data = tomtom_data["flowSegmentData"]
            data["currentSpeed"] = flow_data.get("currentSpeed")
            data["freeFlowSpeed"] = flow_data.get("freeFlowSpeed")
        else:
            data["currentSpeed"] = None
            data["freeFlowSpeed"] = None

    except requests.exceptions.Timeout:
        print("TomTom API request timed out")
        data["currentSpeed"] = None
        data["freeFlowSpeed"] = None
    except requests.exceptions.RequestException as e:
        print(f"TomTom API error: {e}")
        data["currentSpeed"] = None
        data["freeFlowSpeed"] = None

    try:
        weather_response = requests.get(
            OPENWEATHER_BASE_URL, params=weather_params, timeout=10
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        data["temp"] = weather_data.get("main", {}).get("temp")
        data["humidity"] = weather_data.get("main", {}).get("humidity")
        data["visibility"] = weather_data.get("visibility", 10000)
        data["weather_main"] = (
            weather_data.get("weather", [{}])[0].get("main")
            if weather_data.get("weather")
            else None
        )

    except requests.exceptions.Timeout:
        print("OpenWeather API request timed out")
        data["temp"] = None
        data["humidity"] = None
        data["visibility"] = None
        data["weather_main"] = None
    except requests.exceptions.RequestException as e:
        print(f"OpenWeather API error: {e}")
        data["temp"] = None
        data["humidity"] = None
        data["visibility"] = None
        data["weather_main"] = None

    return data


def main():
    load_dotenv()

    tomtom_key = os.getenv("TOMTOM_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")

    lat = 19.4167
    lon = 72.8167

    print(f"Fetching live data for coordinates: Lat {lat}, Lon {lon}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    data = fetch_live_data(tomtom_key, weather_key, lat, lon)
    print(f"Fetched data: {data}")

    csv_path = "data/live_traffic.csv"
    columns = [
        "timestamp",
        "currentSpeed",
        "freeFlowSpeed",
        "temp",
        "humidity",
        "visibility",
        "weather_main",
    ]

    df_new = pd.DataFrame([data], columns=columns)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    os.makedirs("data", exist_ok=True)
    df_updated.to_csv(csv_path, index=False)
    print(f"Data appended to {csv_path}")


if __name__ == "__main__":
    main()
