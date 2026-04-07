import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_mock_data(num_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)

    base_temp = 27.0
    base_humidity = 65
    base_visibility = 10000
    base_free_flow = 18.0

    weather_conditions = ["Clear", "Clouds", "Rain", "Mist"]
    weather_weights = [0.5, 0.25, 0.15, 0.1]

    hours = list(range(24))
    hour_weights = [0.02] * 24
    hour_weights[7] = 0.08
    hour_weights[8] = 0.12
    hour_weights[9] = 0.10
    hour_weights[17] = 0.10
    hour_weights[18] = 0.12
    hour_weights[19] = 0.08

    data = []
    start_time = datetime(2026, 4, 7, 19, 0, 0)

    for i in range(num_rows):
        timestamp = start_time + timedelta(minutes=i * 5)
        hour = timestamp.hour

        weather_main = random.choices(weather_conditions, weights=weather_weights)[0]

        free_flow_speed = base_free_flow + np.random.uniform(-1, 2)
        free_flow_speed = max(15, min(20, free_flow_speed))

        if weather_main == "Rain":
            current_speed = free_flow_speed * np.random.uniform(0.4, 0.7)
        elif weather_main == "Mist":
            current_speed = free_flow_speed * np.random.uniform(0.7, 0.9)
        elif weather_main == "Clouds":
            current_speed = free_flow_speed * np.random.uniform(0.75, 0.95)
        else:
            if hour in [8, 9, 18, 19]:
                current_speed = free_flow_speed * np.random.uniform(0.5, 0.75)
            else:
                current_speed = free_flow_speed * np.random.uniform(0.8, 1.0)

        current_speed = max(5, min(free_flow_speed + 2, current_speed))

        if weather_main == "Rain":
            temp = base_temp + np.random.uniform(-3, 2)
            humidity = base_humidity + np.random.uniform(10, 25)
            visibility = np.random.uniform(2000, 5000)
        elif weather_main == "Mist":
            temp = base_temp + np.random.uniform(-2, 3)
            humidity = base_humidity + np.random.uniform(5, 15)
            visibility = np.random.uniform(1000, 4000)
        elif weather_main == "Clouds":
            temp = base_temp + np.random.uniform(-2, 4)
            humidity = base_humidity + np.random.uniform(0, 10)
            visibility = np.random.uniform(7000, 10000)
        else:
            temp = base_temp + np.random.uniform(-3, 5)
            humidity = base_humidity + np.random.uniform(-5, 10)
            visibility = base_visibility

        temp = round(temp, 2)
        humidity = int(round(humidity))
        visibility = int(round(visibility))
        current_speed = round(current_speed, 1)
        free_flow_speed = round(free_flow_speed, 1)

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "currentSpeed": current_speed,
                "freeFlowSpeed": free_flow_speed,
                "temp": temp,
                "humidity": humidity,
                "visibility": visibility,
                "weather_main": weather_main,
            }
        )

    return pd.DataFrame(data)


def main():
    csv_path = "data/live_traffic.csv"

    print(f"Generating 500 rows of synthetic data...")
    df_new = generate_mock_data(500)
    print(f"Generated {len(df_new)} rows")

    print(f"Appending to {csv_path}...")
    if pd.io.common.file_exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    df_updated.to_csv(csv_path, index=False)
    print(f"Data appended. Total rows now: {len(df_updated)}")
    print("\nFirst few rows:")
    print(df_updated.head())


if __name__ == "__main__":
    main()
