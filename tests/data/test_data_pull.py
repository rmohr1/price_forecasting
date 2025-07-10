from datetime import datetime
import time

from price_forecasting.data.pull_data import oasis_api_pull

start_dt = datetime.fromisoformat("2025-01-01")
end_dt = datetime.fromisoformat("2025-01-02")  

def test_api_pull():
    time.sleep(1)
    df = oasis_api_pull("SLD_FCST", "RTM", start_dt, end_dt)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)

def test_dam_pull():
    time.sleep(1)
    import price_forecasting.data.pull.pull_dam_data as puller
    df = oasis_api_pull(puller.QUERYNAME, puller.MARKET, 
                        start_dt, end_dt, params=puller.PARAMS)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)

def test_load_pull():
    time.sleep(1)
    import price_forecasting.data.pull.pull_load_data as puller
    df = oasis_api_pull(puller.QUERYNAME, puller.MARKET, 
                        start_dt, end_dt, params=puller.PARAMS)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)

def test_rtm_pull():
    time.sleep(1)
    import price_forecasting.data.pull.pull_rtm_data as puller
    df = oasis_api_pull(puller.QUERYNAME, puller.MARKET, 
                        start_dt, end_dt, params=puller.PARAMS)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)

def test_rtpd_dmnd_pull():
    time.sleep(1)
    import price_forecasting.data.pull.pull_rtpd_dmnd_data as puller
    df = oasis_api_pull(puller.QUERYNAME, puller.MARKET, 
                        start_dt, end_dt, params=puller.PARAMS)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)

def test_rtpd_prc_pull():
    time.sleep(1)
    import price_forecasting.data.pull.pull_rtpd_prc_data as puller
    df = oasis_api_pull(puller.QUERYNAME, puller.MARKET, 
                        start_dt, end_dt, params=puller.PARAMS)
    assert("INTERVALSTARTTIME_GMT" in df.columns)
    assert(df.shape[0] > 0)