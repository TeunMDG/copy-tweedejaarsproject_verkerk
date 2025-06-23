import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Correct paths to access data from parent directory
file_path_energy = 'data/daily_energy_usage.csv'
file_path_freq = 'data/og_data/frequentie.csv'

# Check if files exist before proceeding
if not os.path.exists(file_path_energy):
    raise FileNotFoundError(f"Energy file not found at: {file_path_energy}")
if not os.path.exists(file_path_freq):
    raise FileNotFoundError(f"Frequency file not found at: {file_path_freq}")

# Load data with proper date parsing
energy_df = pd.read_csv(file_path_energy, index_col='timestamp', parse_dates=True)
frequency_df = pd.read_csv(
    file_path_freq,
    index_col='interval_15min',
    parse_dates=True,
    delimiter=';',
    dayfirst=True  # Fix for European date format
)

# Print frequency columns for inspection
print("Frequency data columns:", frequency_df.columns.tolist())

# Create a new 'frequency' column as the mean of all frequency columns
frequency_df['frequency'] = frequency_df.mean(axis=1)

# Convert frequency data to daily data by taking the mean
daily_frequency_df = frequency_df[['frequency']].resample('D').mean()

# UNCOMMENT OUT LATER, WILL SAVE FILES
# output_dir = 'frequency_daily'
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, 'daily_frequency.csv')
# daily_frequency_df.to_csv(output_path)
# print(f"\nDaily frequency data saved to: {output_path}")

# Align the dataframes by their index
aligned_energy_df, aligned_frequency_df = energy_df.align(daily_frequency_df, join='inner', axis=0)

# Calculate efficiency - FIXED SECTION
efficiency_df = pd.DataFrame(index=aligned_energy_df.index)

for col in aligned_energy_df.columns:
    # Convert to numeric arrays to avoid recursion issues
    energy_vals = pd.to_numeric(aligned_energy_df[col], errors='coerce').values
    freq_vals = pd.to_numeric(aligned_frequency_df['frequency'], errors='coerce').values

    # Calculate efficiency using vectorized operations
    efficiency = np.zeros_like(energy_vals, dtype=float)
    valid_mask = (freq_vals != 0) & (~np.isnan(energy_vals)) & (~np.isnan(freq_vals))
    efficiency[valid_mask] = energy_vals[valid_mask] / freq_vals[valid_mask]

    efficiency_df[col] = efficiency

# Create directory for saving projections and metrics
os.makedirs('fan_efficiency_projections', exist_ok=True)
os.makedirs('model_metrics', exist_ok=True)

# Prepare to store metrics
all_metrics = []

# Model the predicted decline for each fan and plot
for fan_col in efficiency_df.columns:
    # Extract the efficiency data for this fan
    fan_data = efficiency_df[[fan_col]].dropna()

    # Skip if not enough data
    if len(fan_data) < 365:
        print(f"Skipping {fan_col} - insufficient data ({len(fan_data)} points)")
        continue

    # Split into train and test sets (last year as test)
    split_date = fan_data.index.max() - timedelta(days=365)
    train_data = fan_data[fan_data.index <= split_date]
    test_data = fan_data[fan_data.index > split_date]

    # Prepare features (days since first measurement)
    X_train = np.array([(date - train_data.index[0]).days for date in train_data.index]).reshape(-1, 1)
    y_train = train_data[fan_col].values

    X_test = np.array([(date - train_data.index[0]).days for date in test_data.index]).reshape(-1, 1)
    y_test = test_data[fan_col].values

    # Create and train model
    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-4, random_state=42))
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Store metrics
    fan_metrics = {
        'fan': fan_col,
        'r2': r2,
        'mae': mae,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'last_train_date': train_data.index[-1],
        'first_test_date': test_data.index[0],
        'last_test_date': test_data.index[-1]
    }
    all_metrics.append(fan_metrics)

    # Prepare full dataset for projection
    X_full = np.array([(date - fan_data.index[0]).days for date in fan_data.index]).reshape(-1, 1)
    y_full = fan_data[fan_col].values

    # Train final model on all data
    final_model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-4, random_state=42))
    final_model.fit(X_full, y_full)

    # Create future dates (10 years)
    last_date = fan_data.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365*10, freq='D')

    # Prepare future features
    X_future = np.array([(date - fan_data.index[0]).days for date in future_dates]).reshape(-1, 1)
    predicted_efficiency = final_model.predict(X_future)

    # Get predictions for full historical period
    y_pred_full = final_model.predict(X_full)

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot historical data
    plt.scatter(train_data.index, y_train, color='blue', alpha=0.7, s=25, label='Train Data')
    plt.scatter(test_data.index, y_test, color='green', alpha=0.7, s=25, label='Test Data')

    # Plot model fit
    plt.plot(fan_data.index, y_pred_full, 'r-', linewidth=2, label='Model Fit')

    # Plot future projection
    plt.plot(future_dates, predicted_efficiency, 'r--', linewidth=2, label='10-Year Projection')

    # Add vertical line for train/test split
    plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7)

    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Efficiency (kWh/Hz)', fontsize=12)
    plt.title(f'Efficiency Projection for {fan_col}\nTest R² = {r2:.3f}, Test MAE = {mae:.3f}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add metrics box
    metrics_text = (f"Test Metrics:\nR² = {r2:.3f}\nMAE = {mae:.3f}\n\n"
                   f"Data Points:\nTrain: {len(X_train)}\nTest: {len(X_test)}")
    plt.annotate(metrics_text, xy=(0.05, 0.15), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8),
                fontsize=10)

    # Add projection info
    proj_info = (f"Projection:\nStart: {last_date.strftime('%Y-%m-%d')}\n"
                 f"End: {future_dates[-1].strftime('%Y-%m-%d')}")
    plt.annotate(proj_info, xy=(0.75, 0.15), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8),
                fontsize=10)

    # Rotate date labels for better visibility
    plt.gcf().autofmt_xdate()

    # Save the plot
    plot_filename = f'fan_efficiency_projections/{fan_col}_projection.png'
    plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
    plt.close()

# Save metrics to CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('model_metrics/fan_efficiency_metrics.csv', index=False)

print("\nAnalysis complete. Results include:")
print(f"- Projection plots saved in 'fan_efficiency_projections' directory")
print(f"- Metrics saved to 'model_metrics/fan_efficiency_metrics.csv'")
print("\nMetrics Summary:")
print(metrics_df[['fan', 'r2', 'mae', 'train_size', 'test_size']])