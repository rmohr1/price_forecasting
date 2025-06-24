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


if __name__ == '__main__':
    df = fetch_lmp("SP15", "2025-01-01", "2025-01-10")
    print(df.head())
