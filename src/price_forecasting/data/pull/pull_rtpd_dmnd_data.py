from price_forecasting.data.pull_data import fetch_and_cache_data


# CONFIGURATION
QUERYNAME = 'SLD_FCST' #Report to query
MARKET = "RTM"  # Real-Time Market
SAVE_DIR = "rtpd_dmnd"  # Directory to save cached files
START_DATE = "2023-01-01"
END_DATE = "2025-06-23"
CHUNK_DAYS = 7  # Number of days per API request
PARAMS = {
    'TAC_AREA_NAME' : 'SCE-TAC',
    'XML_DATA_ITEM' : 'SYS_FCST_15MIN_MW',
}

# Execute data pull
if __name__ == "__main__":
    df_full = fetch_and_cache_data(
        QUERYNAME, 
        MARKET, 
        START_DATE, 
        END_DATE, 
        SAVE_DIR,
        PARAMS,
        chunk_days = CHUNK_DAYS,
    )
    print(f"\n✅ Downloaded {len(df_full)} rows total.")
    print(df_full.head())
