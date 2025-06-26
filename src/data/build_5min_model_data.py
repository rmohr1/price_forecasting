import pandas as pd
import os

# CONFIGURATION
TIME_COLUMN = "INTERVALSTARTTIME_GMT"

RTM_FILE = "../../data/processed/rtm_cleaned_5min.csv"
DAM_FILE = "../../data/processed/dam_cleaned_5min.csv"
LOAD_FILE = "../../data/processed/demand_cleaned_5min.csv"
RTPD_DMND_FILE = "../../data/processed/rtpd_dmnd_cleaned_5min.csv"
RTPD_PRC_FILE = "../../data/processed/rtpd_prc_cleaned_5min.csv"
OUTPUT_FILE = "../../data/processed/model_ready_dataset_5min.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def load_and_prepare():
    # Load input datasets
    rtm = pd.read_csv(RTM_FILE, parse_dates=[TIME_COLUMN]).set_index(TIME_COLUMN)
    rtm.columns = rtm.columns.str.replace('LMP', 'RTM') #add RTM flag to columns

    dam = pd.read_csv(DAM_FILE, parse_dates=[TIME_COLUMN]).set_index(TIME_COLUMN)
    dam.columns = dam.columns.str.replace('LMP', 'DAM') #add DAM flag to columns

    load = pd.read_csv(LOAD_FILE, parse_dates=[TIME_COLUMN]).set_index(TIME_COLUMN)
    load = load[['SCE-TAC']] #Select only SCE-TAC zone
    load = load.rename(columns={"SCE-TAC": "2DA_LOAD"}) #rename zone for clarity

    rtpd_dmnd = pd.read_csv(RTPD_DMND_FILE, parse_dates=[TIME_COLUMN]).set_index(TIME_COLUMN)
    rtpd_dmnd.columns = rtpd_dmnd.columns.str.replace("MW", "RTPD_DMND")

    rtpd_prc = pd.read_csv(RTPD_PRC_FILE, parse_dates=[TIME_COLUMN]).set_index(TIME_COLUMN)
    rtpd_prc.columns = rtpd_prc.columns.str.replace("PRC", "RTPD_PRC")

    # Join on timestamp
    df = rtm.join([dam, load, rtpd_dmnd, rtpd_prc], how="outer")

    # Add lag features
    df["RTM_PRC_lag75min"] = df["RTM_PRC"].shift(15)
    df["RTM_PRC_lag80min"] = df["RTM_PRC"].shift(16)
    df["RTM_PRC_lag85min"] = df["RTM_PRC"].shift(17)
    df["RTM_PRC_lag90min"] = df["RTM_PRC"].shift(18)

    df["RTM_PRC_lag24hr"] = df["RTM_PRC"].shift(24*15)
    df["2DA_LOAD_lag1"] = df["2DA_LOAD"].shift(1)
    df["RTPD_DMND_lag1"] = df['RTPD_DMND'].shift(1)
 

    # Time-based features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek >= 5
    df["is_night"] = (df.index.hour < 6) | (df.index.hour > 20)

    # Drop NA rows
    df = df.dropna()

    # Save to CSV
    df.to_csv(OUTPUT_FILE)
    print(f"✅ Modeling dataset saved to: {OUTPUT_FILE}")

    return df

# MAIN EXECUTION
if __name__ == "__main__":
    df_model = load_and_prepare()
    print(df_model.head())
