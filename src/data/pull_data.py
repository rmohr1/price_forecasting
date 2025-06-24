from pycaiso.oasis import Node
from datetime import datetime
import pandas as pd
import time

def fetch_lmp(node_name: str, start: str, end: str, market: str='DAM') -> pd.DataFrame:
    """
    Fetch LMP data for a CAISO node using pycaiso.

    :param node_name: e.g. "SP15" or a specific p-node
    :param start: "YYYY-MM-DD"
    :param end:   "YYYY-MM-DD"
    :returns: DataFrame with INTERVALSTARTTIME_GMT, LMP values
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    node = getattr(Node, node_name.upper())() if hasattr(Node, node_name.upper()) else Node(node_name)
    df = node.get_lmps(start_dt, end_dt, market=market)
    
    df = df.rename(columns={'INTERVALSTARTTIME_GMT':'timestamp', 'LMP':'lmp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #return df[['timestamp', 'lmp']]
    return df

def build_dataset(node_name: str, start: str, end: str, market: str='RTM', chunk_days: int=7, SLEEP_SECONDS: int=2) -> pd.DataFrame:
    """
    Build and cache LMP data for a CAISO node using pycaiso.

    :param node_name: e.g. "SP15" or a specific p-node
    :param start: "YYYY-MM-DD"
    :param end:   "YYYY-MM-DD"
    :returns: DataFrame with INTERVALSTARTTIME_GMT, LMP values
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    node = getattr(Node, node_name.upper())() if hasattr(Node, node_name.upper()) else Node(node_name)
    
    all_chunks = []

    dt = start_dt

    while t < end_dt:
        chunk_end_dt = min(dt + timedelta(days=chunk_days), end_dt)
        chunk_str = f"{start_dt.date()}_to_{chunk_end_dt.date()}"
        cache_file = os.path.join(SAVE_DIR, f"{node_name}_{market}_{chunk_str}.csv") 

        if os.path.exists(cache_file):
            print(f"Skipping cached chunk: {chunk_str}")
            df = pd.read_csv(cache_file, parse_dates=["INTERVALSTARTTIME_GMT"])
        else:
            print(f"Fetching chunk: {chunk_str}")
            try:
                df = node.get_lmps(start_dt, chunk_end_dt, market_run_id=market)
                df.to_csv(cache_file, index=False)
                time.sleep(SLEEP_SECONDS)
            except Exception as e:
                print(f"Failed to fetch {chunk_str}: {e}")
                df = pd.DataFrame()

        all_chunks.append(df)
        start_dt = chunk_end_dt

    df = pd.concat(all_chunks, ignore_index=True) 
    


if __name__ == '__main__':
    df = fetch_lmp("SP15", "2025-01-01", "2025-01-10")
    print(df.head())
