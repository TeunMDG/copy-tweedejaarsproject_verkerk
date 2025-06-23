import pandas as pd

def load_and_process_data():
    # --- 1. Load the Data ---
    file_path = 'data/Energie_kWh_15_min.csv'

    try:
        df = pd.read_csv(
            file_path,
            delimiter=';',
            parse_dates=['timestamp'],
            dayfirst=True,
            na_values='NULL'
        )
        print("✅ File loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at '{file_path}'.")
        print("Please update the 'file_path' variable in the script.")
        exit()

    # --- 2. Process the Data ---
    df_processed = df.copy()
    numeric_cols = df.columns.drop('timestamp')

    for col in numeric_cols:
        df_processed[col] = df_processed[col].diff()
        df_processed[col] = df_processed[col].apply(lambda x: x if 0 <= x <= 200 else 0)
        df_processed[col] = df_processed[col].fillna(0)

    # --- 3. Aggregate to Hourly Usage ---
    df_hourly = df_processed.set_index('timestamp')
    df_hourly = df_hourly.resample('H').sum()

    # --- 4. Export to ./data ---
    output_path = 'data/hourly_energy_usage.csv'
    df_hourly.to_csv(output_path)
    print(f"✅ Hourly energy usage data exported to '{output_path}'")

# === Run immediately ===
load_and_process_data()