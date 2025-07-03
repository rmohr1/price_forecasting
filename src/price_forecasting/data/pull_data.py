from datetime import datetime, timedelta
import pandas as pd
import time
import requests
import re
import io
import zipfile
import pytz
import os

SLEEP_SECONDS = 20

def get_df(
    response,
    parse_dates=[2],
    sort_values=None,
    reindex_columns=None,
) -> pd.DataFrame:

    """
    Converts requests.response to pandas.DataFrame

    Args:
        r : requests response object
        parse_dates (bool, list): which columns to parse dates if any
        sort_values(list): which columsn to sort by if any

    Returns:
        df (pandas.DataFrame): pandas dataframe
    """

    with io.BytesIO() as buffer:
        try:
            buffer.write(response.content)
            buffer.seek(0)
            z = zipfile.ZipFile(buffer)

        except zipfile.BadZipFile as e:
            raise ValueError("Bad zip file", e)

        else:
            csv = z.open(z.namelist()[0])  # ignores all but first file in zip
            df = pd.read_csv(csv, parse_dates=parse_dates)

            if sort_values:
                df = df.sort_values(sort_values).reset_index(drop=True)

            if reindex_columns:
                df = df.reindex(columns=reindex_columns)

    return df

def oasis_api_pull(
        queryname: str, 
        market: str, 
        start: datetime, 
        end: datetime, 
        params: dict, 
        timeout=20
) -> pd.DataFrame:
    """
    Constructs a CAISO OASIS API query and returns a dataframe

    Args:
        queryname : report name to request
        market: report market
        start: start datetime
        end: end datetime
        params: dict of extra request specific parameters
        timeout: requests query timeout

    Returns:
        df (pandas.DataFrame): pandas dataframe
    """

    def _get_UTC_string(
            dt: datetime,
            local_tz: str='America/Los_Angeles', 
            fmt: str = "%Y%m%dT%H:%M-0000"
            ) -> str:
        """
        Convert date string to UTC string for API request

        Args:
            dt : input datetime to convert

        Returns:
            tz_ : API compatible formatted and localized time string
        """
        tz_ = pytz.timezone(local_tz)
        return tz_.localize(dt).astimezone(pytz.UTC).strftime(fmt)

    #convert datetime strings to API format
    start_str = _get_UTC_string(start)
    end_str = _get_UTC_string(end)

    #construct api request
    base_url = "https://oasis.caiso.com/oasisapi/SingleZip?"

    pull_params = {
        'queryname':queryname,
        'startdatetime': start_str,
        'enddatetime': end_str,
        'market_run_id': market,
        'resultformat': '6',
        'version': '1',
    }

    # add in any report specific parameters
    pull_params = pull_params | params

    #execute request
    resp = requests.get(base_url, params=pull_params, timeout=timeout)
    resp.raise_for_status()
    headers = resp.headers["content-disposition"]

    #check for empty data file
    if re.search(r"\.xml\.zip;$", headers):
        raise ValueError("No data available for this query.")

    df = get_df(resp)
    return df

def fetch_and_cache_data(
        queryname: str, 
        market: str, 
        start_date: str, 
        end_date: str, 
        save_dir: str,
        params: dict,
        chunk_days: int = 7,
        timeout=20
):
    """
    Pulls data from the CAISO OASIS database, caches it, and then assembles into raw dataset

    Args:
        queryname : report name to request
        market: report market
        start: start date string "YYYY-MM-DD"
        end: end date string "YYYY-MM-DD"
        save_dir: relative directory string, creates and saves data there
        params: dict of extra request specific parameters
        timeout: requests query timeout

    Returns:
        df (pandas.DataFrame): pandas dataframe
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    all_chunks = []

    chunk_start_dt = start_dt
    while chunk_start_dt < end_dt:
        #creates a time chunk and cache file
        chunk_end_dt = min(chunk_start_dt + timedelta(days=chunk_days), end_dt)
        chunk_str = f"{chunk_start_dt.date()}_to_{chunk_end_dt.date()}"
        cache_file = os.path.join(save_dir, f"{queryname}_{market}_{chunk_str}.csv")

        if os.path.exists(cache_file):
            print(f"[✓] Skipping cached chunk: {chunk_str}")
            df = pd.read_csv(cache_file, parse_dates=["INTERVALSTARTTIME_GMT"])
        else:
            print(f"[→] Fetching chunk: {chunk_str}")
            df = oasis_api_pull(queryname, market, chunk_start_dt, chunk_end_dt, params=params)
            df.to_csv(cache_file, index=False)
            time.sleep(SLEEP_SECONDS)

        all_chunks.append(df)
        chunk_start_dt = chunk_end_dt
    
    df = pd.concat(all_chunks, ignore_index=True)
    df.to_csv(os.path.join(save_dir, "dataset.csv"))

    return df


if __name__ == '__main__':
    start_dt = datetime.fromisoformat("2025-01-01")
    end_dt = datetime.fromisoformat("2025-01-10")  
    df = oasis_api_pull("SLD_FCST", "RTM", start_dt, end_dt)
    print(df.head())
