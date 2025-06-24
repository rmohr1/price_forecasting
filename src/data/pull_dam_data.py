from datetime import datetime, timedelta
import os
import time
import pandas as pd
from pycaiso.oasis import Node

# CONFIGURATION
NODE_NAME = "SP15"  # Change to your desired node
MARKET = "DAM"  # Real-Time Market
SAVE_DIR = "../../data/raw/caiso_dam"  # Directory to save cached files
START_DATE = "2023-01-01"
END_DATE = "2025-06-23"
CHUNK_DAYS = 7  # Number of days per API request
SLEEP_SECONDS = 5  # Pause between requests to avoid throttling

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_and_cache_lmp(node_name, start_date, end_date, market, chunk_days=7):
    node = getattr(Node, node_name.upper())() if hasattr(Node, node_name.upper()) else Node(node_name)    
    
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    all_chunks = []

    dt = start_dt
    while dt < end_dt:
        chunk_end_dt = min(dt + timedelta(days=chunk_days), end_dt)
        chunk_str = f"{dt.date()}_to_{chunk_end_dt.date()}"
        cache_file = os.path.join(SAVE_DIR, f"{node_name}_{market}_{chunk_str}.csv")

        if os.path.exists(cache_file):
            print(f"[✓] Skipping cached chunk: {chunk_str}")
            df = pd.read_csv(cache_file, parse_dates=["INTERVALSTARTTIME_GMT"])
        else:
            print(f"[→] Fetching chunk: {chunk_str}")
            df = node.get_lmps(dt, chunk_end_dt, market=market)
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
        dt = chunk_end_dt
    
    df = pd.concat(all_chunks, ignore_index=True)
    df.to_csv(os.path.join(SAVE_DIR, "dataset.csv"))

    return df

# MAIN EXECUTION
if __name__ == "__main__":
    df_full = fetch_and_cache_lmp(NODE_NAME, START_DATE, END_DATE, MARKET, CHUNK_DAYS)
    print(f"\n✅ Downloaded {len(df_full)} rows total.")
    print(df_full.head())
