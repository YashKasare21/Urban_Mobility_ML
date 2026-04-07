import pandas as pd


def main():
    raw_csv_path = "data/live_traffic.csv"
    processed_csv_path = "data/processed_traffic.csv"

    print(f"Loading data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    print(f"Loaded {len(df)} rows")

    print(f"Dropping rows with missing values...")
    df_clean = df.dropna()
    print(f"Remaining rows after dropna: {len(df_clean)}")

    df_clean["congestion_ratio"] = df_clean["currentSpeed"] / df_clean[
        "freeFlowSpeed"
    ].replace(0, 1)
    df_clean.loc[df_clean["freeFlowSpeed"] == 0, "congestion_ratio"] = 0

    df_clean["traffic_state"] = df_clean["congestion_ratio"].apply(
        lambda x: 1 if x < 0.7 else 0
    )

    print(f"Saving processed data to {processed_csv_path}...")
    df_clean.to_csv(processed_csv_path, index=False)

    print("Preprocessing complete!")
    print("\nFirst few rows of processed data:")
    print(df_clean.head())


if __name__ == "__main__":
    main()
