import pandas as pd
import os
import matplotlib.pyplot as plt

# CONFIGURATION
INPUT_FILE = "../../data/raw/caiso_demand_2DA/dataset.csv"
OUTPUT_FILE_5MIN = "../../data/processed/demand_cleaned_5min.csv"
OUTPUT_FILE_HOURLY = "../../data/processed/demand_cleaned_hourly.csv"
TIME_COLUMN = "INTERVALSTARTTIME_GMT"
PRICE_COLUMN = "SCE-TAC"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE_5MIN), exist_ok=True)

# Cleaning function
def clean_demand_data(input_file, time_col, price_col):
    df = pd.read_csv(input_file)
    df = df.pivot_table(index=TIME_COLUMN, 
                          columns=['TAC_AREA_NAME'],
                          values=['MW'],
                          aggfunc='first').reset_index()
    
    # Parse and sort timestamps
    df[time_col] = pd.to_datetime(df[time_col], format='mixed')

    #Flatten Pivot table
    target_cols = df['MW'].columns
    df.columns = df.columns.to_series().str.join('')
    df.columns = df.columns.str.replace('MW', '')

    df = df.sort_values(time_col).drop_duplicates(subset=time_col)

    # Set index and fill missing intervals to 5-minute frequency
    df = df.set_index(time_col).asfreq("5T")

    # Interpolate missing values
    for col in target_cols:
        try:
            df[col] = df[col].interpolate(method="time")
        except:
            print('column ' + col + ' was dropped')

    return df

# Plotting function
def plot_time_series(df, price_col, title, freq="D", sample_size=60):
    df_sample = df[price_col].resample(freq).mean().iloc[:sample_size]
    plt.figure(figsize=(12, 4))
    plt.plot(df_sample.index, df_sample.values, label=title)
    plt.title(title)
    plt.ylabel("Price ($/MWh)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    df_5min = clean_demand_data(INPUT_FILE, TIME_COLUMN, PRICE_COLUMN)
    df_5min.to_csv(OUTPUT_FILE_5MIN)
    print(f"✅ 5-minute cleaned data saved to: {OUTPUT_FILE_5MIN}")

    # Downsample to hourly
    df_hourly = df_5min.resample("H").mean()
    df_hourly.to_csv(OUTPUT_FILE_HOURLY)
    print(f"✅ Hourly data saved to: {OUTPUT_FILE_HOURLY}")

    # Quick plots for sanity check
    print("📊 Plotting 5-minute (daily mean):")
    plot_time_series(df_5min, PRICE_COLUMN, "CAISO Demand (5-minute, Daily Avg)", freq="D")

    print("📊 Plotting hourly (daily mean):")
    plot_time_series(df_hourly, PRICE_COLUMN, "CAISO Demand (Hourly, Daily Avg)", freq="D")
