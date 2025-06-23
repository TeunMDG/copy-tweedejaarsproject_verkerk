import pandas as pd
import os

def load_and_process_data_freq(file_path, year):
    """
    Loads and processes frequency data for a given year.

    Args:
        file_path (str): Path to the raw frequency CSV.
        year (int): Year to analyze (e.g., 2024).

    Returns:
        pd.DataFrame: DataFrame with mean daily frequency for the given year.
    """
    # --- 1. Load the Data ---
    try:
        df = pd.read_csv(
            file_path,
            delimiter=';',
            parse_dates=['interval_15min'],
            dayfirst=True,
            na_values='NULL'
        )
        print(f"✅ File loaded successfully for year {year}!")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Error: The file was not found at '{file_path}'.")

    # --- 2. Process the Data ---
    df_processed = df.copy()
    numeric_cols = df.columns.drop('interval_15min')

    for col in numeric_cols:
        df_processed[col] = df_processed[col].diff()
        df_processed[col] = df_processed[col].apply(lambda x: x if 0 <= x <= 200 else 0)
        df_processed[col] = df_processed[col].fillna(0)

    # --- 3. Aggregate to Daily Usage ---
    df_daily = df_processed.set_index('interval_15min')
    df_daily = df_daily.resample('D').sum()

    # --- 4. Filter for the Selected Year ---
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"  # Covers full year (inclusive)
    df_year = df_daily.loc[start_date:end_date]

    if df_year.empty:
        raise ValueError(f"⚠️ No data found for {year} in the given range.")

    # --- 5. Calculate Mean Daily Frequency ---
    mean_values = df_year.mean()
    mean_df = mean_values.reset_index()
    mean_df.columns = ['Meter', f'Mean Daily Frequency ({year})']

    # --- 6. Save to CSV ---
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)  # Create dir if it doesn't exist
    output_file = os.path.join(output_dir, f"mean_daily_freq_{year}.csv")
    mean_df.to_csv(output_file, index=False)

    print(f"✅ Mean frequency data saved to '{output_file}'")
    return mean_df
