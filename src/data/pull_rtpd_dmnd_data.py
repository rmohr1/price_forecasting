from datetime import datetime, timedelta
import os
import time
import pandas as pd
from pull_data import oasis_api_pull

# CONFIGURATION
QUERY = 'SLD_FCST' #Report to query
MARKET = "RTM"  # Real-Time Market
SAVE_DIR = "../../data/raw/caiso_rtpd_dmnd"  # Directory to save cached files
START_DATE = "2023-01-01"
END_DATE = "2025-06-23"
CHUNK_DAYS = 7  # Number of days per API request
SLEEP_SECONDS = 5  # Pause between requests to avoid throttling
PARAMS = {
    'TAC_AREA_NAME' : 'SCE-TAC',
    'XML_DATA_ITEM' : 'SYS_FCST_15MIN_MW',
}

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_and_cache_data():
    
    start_dt = datetime.fromisoformat(START_DATE)
    end_dt = datetime.fromisoformat(END_DATE)
    
    all_chunks = []

    chunk_start_dt = start_dt
    while chunk_start_dt < end_dt:
        chunk_end_dt = min(chunk_start_dt + timedelta(days=CHUNK_DAYS), end_dt)
        chunk_str = f"{chunk_start_dt.date()}_to_{chunk_end_dt.date()}"
        cache_file = os.path.join(SAVE_DIR, f"{QUERY}_{MARKET}_{chunk_str}.csv")

        if os.path.exists(cache_file):
            print(f"[✓] Skipping cached chunk: {chunk_str}")
            df = pd.read_csv(cache_file, parse_dates=["INTERVALSTARTTIME_GMT"])
        else:
            print(f"[→] Fetching chunk: {chunk_str}")
            df = oasis_api_pull(QUERY, MARKET, chunk_start_dt, chunk_end_dt, params=PARAMS)
            df.to_csv(cache_file, index=False)
            time.sleep(SLEEP_SECONDS)
            #try:
            #    df = node.get_lmps(dt, chunk_end_dt, market=market)
            #    df.to_csv(cache_file, index=False)
            #    time.sleep(SLEEP_SECONDS)
            #except Exception as e:
            #    print(f"[!] Failed to fetch {chunk_str}: {e}")
            #    df = pd.DataFrame()

        all_chunks.append(df)
        chunk_start_dt = chunk_end_dt
    
    df = pd.concat(all_chunks, ignore_index=True)
    df.to_csv(os.path.join(SAVE_DIR, "dataset.csv"))

    return df

# MAIN EXECUTION
if __name__ == "__main__":
    df_full = fetch_and_cache_data()
    print(f"\n✅ Downloaded {len(df_full)} rows total.")
    print(df_full.head())
