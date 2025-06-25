from datetime import datetime
import pandas as pd
import time
import requests
import re
import io
import zipfile
import pytz


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
            print("Bad zip file", e)

        else:
            csv = z.open(z.namelist()[0])  # ignores all but first file in zip
            df = pd.read_csv(csv, parse_dates=parse_dates)

            if sort_values:
                df = df.sort_values(sort_values).reset_index(drop=True)

            if reindex_columns:
                df = df.reindex(columns=reindex_columns)

    return df

def oasis_api_pull(queryname: str, market: str, start: datetime, end: datetime, params: dict, timeout=20):
    """
    Constructs a CAISO OASIS API query and returns a dataframe

    Args:
        queryname : report name to request
        market: report market
        start: "YYYY-MM-DD"
        end: "YYYY-MM-DD"
        timeout: requests query timeout
        params dict: extra request specific parameters


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
        print("No data available for this query.")

    df = get_df(resp)
    return df


if __name__ == '__main__':
    #df = fetch_lmp("SP15", "2025-01-01", "2025-01-10")
    start_dt = datetime.fromisoformat("2025-01-01")
    end_dt = datetime.fromisoformat("2025-01-10")  
    df = oasis_api_pull("SLD_FCST", "RTM", start_dt, end_dt)
    print(df.head())
