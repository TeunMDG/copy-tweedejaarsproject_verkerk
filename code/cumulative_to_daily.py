import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/Energie_kWh_15_min.csv'

def load_and_process_data(file_path,label):
    # --- 1. Load the Data ---
    try:
        df = pd.read_csv(
            file_path,
            delimiter=';',
            parse_dates=[label],
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
    numeric_cols = df.columns.drop(label)

    for col in numeric_cols:
        df_processed[col] = df_processed[col].diff()
        df_processed[col] = df_processed[col].apply(lambda x: x if 0 <= x <= 200 else 0)
        df_processed[col] = df_processed[col].fillna(0)

    # --- 3. Aggregate to Daily Usage ---
    df_daily = df_processed.set_index(label)
    df_daily = df_daily.resample('D').sum()

    return df_daily

# This creates df_daily when the module is imported
df_daily = load_and_process_data(file_path)

if __name__ == '__main__':
    print("\nAggregated daily usage data preview:")
    print(df_daily.head())

    # --- 4. Save the Daily Data to a New File ---
    output_file_path = 'daily_energy_usage.csv'
    df_daily.to_csv(output_file_path)
    print(f"\n✅ Daily energy usage data saved to '{output_file_path}'")

    # --- 5. Plot the Daily Usage ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    for column in df_daily.columns:
        ax.plot(df_daily.index, df_daily[column], label=column)

    ax.set_title('Daily Energy Usage', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.legend(title='Meter')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # --- 6. Show the Graph ---
    print("\nDisplaying the plot. Close the plot window to finish the script.")
    plt.show()