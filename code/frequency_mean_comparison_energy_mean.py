import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os

DATA_DIR = 'data'

def load_year_data(year):
    freq_path = os.path.join(DATA_DIR, f'mean_daily_freq_{year}.csv')
    usage_path = os.path.join(DATA_DIR, f'mean_daily_usage_{year}.csv')

    if not os.path.exists(freq_path) or not os.path.exists(usage_path):
        raise FileNotFoundError(f"Data for year {year} not found.")

    freq_df = pd.read_csv(freq_path)
    usage_df = pd.read_csv(usage_path)

    # Calculate efficiency = usage / frequency
    efficiency = usage_df.iloc[:, 1] / freq_df.iloc[:, 1]
    entries = freq_df.iloc[:, 0]  # Assuming same identifiers

    return entries.tolist(), efficiency.tolist()


def generate_efficiency_plot(year1, year2):
    entries1, eff1 = load_year_data(year1)
    entries2, eff2 = load_year_data(year2)

    if entries1 != entries2:
        raise ValueError("Entry names between years do not match.")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(len(entries1))

    ax.bar(index, eff1, bar_width, label=str(year1))
    ax.bar(index + bar_width, eff2, bar_width, label=str(year2))

    ax.set_xlabel('Fan Entries')
    ax.set_ylabel('Energy Efficiency (Usage / Frequency)')
    ax.set_title(f'KwH per cycle comparison: {year1} vs {year2}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(entries1, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return encoded
