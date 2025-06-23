import pandas as pd
from cumulative_to_daily import df_daily

def calculate_mean_usage(start_date: str, end_date: str):
    df_2024 = df_daily.loc[start_date:end_date]

    if df_2024.empty:
        return {
            "success": False,
            "message": f"No data found in the date range {start_date} to {end_date}."
        }

    mean_values = df_2024.mean()
    mean_df = mean_values.reset_index()
    mean_df.columns = ['Meter', 'Mean Daily kWh']
    return {
        "success": True,
        "data": mean_df.to_dict(orient="records")
    }




# old version:

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from cumulative_to_daily import df_daily

# # --- 3. Calculate Mean Usage for 2024 ---
# print("\n--- Part 2: Calculating mean usage for 2024 ---")

# # Define the date range. Note: .loc slicing is inclusive on both ends.
# start_date = '2024-01-01'
# end_date = '2025-01-01'

# # Filter the DataFrame to get only the data within the specified date range
# df_2024 = df_daily.loc[start_date:end_date]

# if df_2024.empty:
#     print(f"⚠️ Warning: No data found in the date range {start_date} to {end_date}.")
# else:
#     # Calculate the mean for each column in the filtered DataFrame
#     mean_values = df_2024.mean()

#     # Convert the results into a nice format for saving
#     mean_df = mean_values.reset_index()
#     mean_df.columns = ['Meter', 'Mean Daily kWh (2024)']

#     print("\nCalculated Mean Daily Usage for 2024:")
#     print(mean_df)

#     # Save the mean values to a new CSV file
#     mean_output_file = 'mean_daily_usage_2024.csv'
#     mean_df.to_csv(mean_output_file, index=False)

#     print(f"\n✅ Mean usage data saved to '{mean_output_file}'")

# print("\n--- Script Finished ---")
