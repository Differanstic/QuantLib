import pandas as pd

import numpy as np
from datetime import datetime,timedelta

def getRelativePercentile(fyers,symbol,tf,start=None,end=None,range=None, is_backtest=False):
    
    if end :
      end = datetime.strptime(end,"%d%B%y") + timedelta(days=1)
      start = (end - timedelta(days = range)).strftime("%Y-%m-%d")
      end = end.strftime("%Y-%m-%d")
    else:
      end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
      start = (datetime.now() - timedelta(days = range)).strftime("%Y-%m-%d")
    
    
      
    df = fyers.get_fyers_historical_df(symbol,tf,start,end)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M:%S")
    df['time'] = df['timestamp'].dt.time
    df['avg_volume'] = df.groupby('time')['volume'].transform('mean')
    df['volume_percentile'] = (
        df.groupby('time')['volume']
      .rank(pct=True)
    )
    df['range'] = abs(df['close'] - df['low'] )
    df['avg_range'] = df.groupby('time')['range'].transform('mean')
    df['range_percentile'] = (
        df.groupby('time')['range']
      .rank(pct=True)
    )
    df['rvol'] = df['volume'] / df['avg_volume']
    
    result = pd.DataFrame()
    result = (
        df.groupby('time')
        .agg(
            avg_volume=('volume','mean'),
            avg_range=('range','mean')
        )
        .reset_index()
    )
    end = datetime.strptime(end,"%Y-%m-%d")-timedelta(days=1)
    if not is_backtest:
      df['timestamp'] = df['timestamp'] + pd.Timedelta(minutes=15)
    df = df[df['timestamp'] > end]
    return df,result