import pandas as pd
import numpy as np
from . import option_greeks as og
from datetime import datetime
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import math
import os


resource_dir = 'C:/Kotak_DB/Resources'
data_dir = 'C:/Kotak_DB/DB'

all_keys = ['timestamp','bp', 'bp1', 'bp2', 'bp3', 'bp4', 
                'bq', 'bq1', 'bq2', 'bq3', 'bq4', 
                'bno1', 'bno2', 'bno3', 'bno4', 'bno5', 
                'sp', 'sp1', 'sp2', 'sp3', 'sp4', 
                'sq', 'sq1', 'sq2', 'sq3', 'sq4', 
                'sno1', 'sno2', 'sno3', 'sno4', 'sno5']

def get_dates():
    path = 'C:/Users/manav/OneDrive/Desktop/sad-al-suud/Sadalsuud/DB/Index/'
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def loadNifty(date):
    index_path = f'{data_dir}/Index/{date}/Nifty 50.parquet'
    nifty_df = pd.read_parquet(index_path, engine='pyarrow')
    nifty_df = nifty_df.iloc[:, :2]
    nifty_df.columns = ['timestamp', 'spot_price']
    nifty_df['timestamp'] = pd.to_datetime(date + ' ' + nifty_df['timestamp'], format='%d%B%y %H:%M:%S:%f')
    return nifty_df


def loadOption(date,strike,load_mob:bool=False):
    
    d = datetime.strptime(date,'%d%B%y')
    cutoff = datetime.strptime('20August25','%d%B%y')
    if d < cutoff:
        df = pd.read_parquet(f'{data_dir}/NiftyFNO/{date}/{strike}.parquet', engine='pyarrow')
        df = df.iloc[:, :5]
        df.columns = ['timestamp', 'ltp', 'ltq', 'buyers', 'sellers'] 
    else:
        df = pd.read_parquet(f'{data_dir}/NiftyFNO/{date}/{strike}.parquet', engine='pyarrow')
        df = df.iloc[:, :6]
        df.columns = ['timestamp', 'ltp', 'ltq', 'buyers', 'sellers','oi'] 
        df['oi'] = df['oi'].replace(0, np.nan)
        df['oi'] = df['oi'].ffill()
        
    df['timestamp'] = pd.to_datetime(date + ' ' + df['timestamp'], format="%d%B%y %H:%M:%S:%f",errors='coerce')
    df['ltq'] = df['ltq'].replace(0, np.nan)
    df['buyers'] = df['buyers'].replace(0, np.nan)
    df['sellers'] = df['sellers'].replace(0, np.nan)
    df[['ltq', 'buyers', 'sellers']] = df[['ltq', 'buyers', 'sellers']].ffill()
    df = df.sort_values('timestamp')
    
    
    
    # MOB DF
    if load_mob:
        mob_df = pd.read_parquet(f'{data_dir}/NiftyFNOMOB/{date}/{strike}.parquet', engine='pyarrow')
        mob_df = mob_df.replace(0,np.nan)
        mob_df.ffill(inplace=True)

        mob_df.columns = all_keys
        mob_df['timestamp'] = pd.to_datetime(date + ' ' + mob_df['timestamp'], format='%d%B%y %H:%M:%S:%f')
        # Merging Option_DF and MOB_DF
        df = df.sort_values('timestamp')
        mob_df = mob_df.sort_values('timestamp')
        all_timestamps = pd.concat([mob_df['timestamp'], df['timestamp']]).drop_duplicates().sort_values()

        mob_df_reindexed = mob_df.set_index('timestamp').reindex(all_timestamps).ffill().reset_index()
        df_reindexed = df.set_index('timestamp').reindex(all_timestamps).ffill().reset_index()

        df = pd.concat([mob_df_reindexed,  df_reindexed.drop(columns='timestamp')], axis=1)

    
    
    
    expiry_cutoff = pd.to_datetime("2025-08-31")
    if (df["timestamp"] > expiry_cutoff).any():
        Expiry = "Tuesday"
    else:
        Expiry = "Thursday"
    iv_vals, delta_vals, gamma_vals, theta_vals = [], [], [], []
    df = og.calc_weekly_expiry(df,'timestamp',Expiry) 
    isCall = True if 'CE' in strike else False 
    df['strike'] = int(strike[:-2])
    T = df['expiry']
    for i in range(len(df)):
        
        iv = og.implied_volatility(df['ltp'].iloc[i], df['spot_price'].iloc[i], df['strike'].iloc[i], T.iloc[i], 0.1, call=isCall)
        d  = og.delta(df['spot_price'].iloc[i], df['strike'].iloc[i],T.iloc[i], 0.1, iv, call=isCall)
        g  = og.gamma(df['spot_price'].iloc[i], df['strike'].iloc[i], T.iloc[i], 0.1, iv)
        t  = og.theta(df['spot_price'].iloc[i], df['strike'].iloc[i], T.iloc[i], 0.1, iv, call=isCall)

        iv_vals.append(iv)
        delta_vals.append(d)
        gamma_vals.append(g)
        theta_vals.append(t)

    df['iv']     = iv_vals
    df['delta']  = delta_vals
    df['gamma']  = gamma_vals
    df['theta']  = theta_vals

    if 'gamma' in df.columns and 'oi' in df.columns:
        df['gex'] = df['gamma'] * df['oi'] * 75 * df['spot_price']
    if 'delta' in df.columns and 'oi' in df.columns:
        df['dex'] = df['delta'] * df['oi'] * 75 * df['spot_price']


  
    df.drop(columns=[ 'bno1', 'bno2', 'bno3', 'bno4', 'bno5', 'sno1', 'sno2', 'sno3', 'sno4', 'sno5'],inplace=True)
    return df 

def loadOptionChain(date, window, n=5):
    """
    Load option chain for a given date, restricting to n closest strikes to spot.
    
    Args:
        date (str): date folder
        window (str or int): time window for loadOptionMOB
        n (int): number of closest strikes to spot (both sides)
    """
    scripts = glob.glob(f"{data_dir}/NiftyFNO/{date}/*")

    nifty = loadNifty(date)
    spot = float(nifty['spot_price'].iloc[0])   # assuming loadNifty gives a DataFrame with 'spot_price'
    
    # extract strike from filename
    strikes = []
    for script in scripts:
        fname = script.split("\\")[1].split(".")[0]
        strike = int(fname[:-2])   # remove last 2 chars (CE/PE)
        strikes.append((strike, script))
    
    # find n closest strikes to spot
    strikes_sorted = sorted(strikes, key=lambda x: abs(x[0] - spot))
    closest_scripts = strikes_sorted[:n]
    
    def process_script(script, date, nifty, window):
        fname = script.split("\\")[1].split(".")[0]
        return fname, loadOptionMOB(date, fname, nifty, window)
    
    results = Parallel(n_jobs=-1)(
        delayed(process_script)(script, date, nifty, window) for _, script in closest_scripts
    )
    
    return dict(results)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Nifty 50 Stocks
def load_stock_mob(date,stock,window=500):
    try:
        df = pd.read_parquet(f'{data_dir}/NiftyEquity/{date}/{stock}.parquet', engine='pyarrow')
        df = df.iloc[:, :5]
        df.columns = ['timestamp', 'ltp', 'ltq', 'buyers', 'sellers']
        df['timestamp'] = pd.to_datetime(date + ' ' + df['timestamp'], format='%d%B%y %H:%M:%S:%f')
        df['ltq'] = df['ltq'].replace(0, np.nan)
        df['buyers'] = df['buyers'].replace(0, np.nan)
        df['sellers'] = df['sellers'].replace(0, np.nan)
        df[['ltq', 'buyers', 'sellers']] = df[['ltq', 'buyers', 'sellers']].ffill()
        # MOB DF
        mob_df = pd.read_parquet(f'{data_dir}/NiftyEquityMOB/{date}/{stock}.parquet', engine='pyarrow')
        mob_df = mob_df.replace(0,np.nan)
        mob_df.ffill(inplace=True)
        all_keys = ['timestamp1','bp', 'bp1', 'bp2', 'bp3', 'bp4', 
                    'bq', 'bq1', 'bq2', 'bq3', 'bq4', 
                    'bno1', 'bno2', 'bno3', 'bno4', 'bno5', 
                    'sp', 'sp1', 'sp2', 'sp3', 'sp4', 
                    'sq', 'sq1', 'sq2', 'sq3', 'sq4', 
                    'sno1', 'sno2', 'sno3', 'sno4', 'sno5']
        mob_df.columns = all_keys
        mob_df['timestamp'] = pd.to_datetime(date + ' ' + mob_df['timestamp1'], format='%d%B%y %H:%M:%S:%f')

        # Merging Option_DF and MOB_DF
        df = df.sort_values('timestamp')
        mob_df = mob_df.sort_values('timestamp')
        df = pd.merge_asof(mob_df,df,on='timestamp',direction='backward')
        
        df['tbq'] = df['bq'] + df['bq1'] + df['bq2'] + df['bq3'] + df['bq4']
        df['tbq'] = (df['tbq'] / df['bno1']).rolling(100,min_periods=2).sum()
        df['tsq'] = df['sq'] + df['sq1'] + df['sq2'] + df['sq3'] + df['sq4']
        df['tsq'] = (df['tsq'] / df['sno1']).rolling(100,min_periods=2).sum()
        
        df['tbo'] = (df['bno1'] + df['bno2'] + df['bno3'] + df['bno4'] + df['bno5']) 
        df['tso'] = (df['sno1'] + df['sno2'] + df['sno3'] + df['sno4'] + df['sno5'])
        df['buy_spread'] = df['ltp'] - df['bp']
        df['sell_spread'] =  df['sp'] - df['ltp']

        
        df['change'] = df['ltp'].diff()
        return df
    except Exception as e:
        print(f"Error loading {stock} for date {date}: {e}")
def _load_stock(file_path, stock, date):
    """Helper: load one stock CSV fast."""
    stock_df = pd.read_parquet(file_path,engine="pyarrow")
    stock_df = stock_df.iloc[:, :5]
    stock_df.columns = ["timestamp", "ltp", "ltq", "buyers", "sellers"]
    
    stock_df["timestamp"] = pd.to_datetime(
        date + " " + stock_df["timestamp"],
        format="%d%B%y %H:%M:%S:%f",
        errors="coerce"
    )
    # ffill numeric columns
    stock_df[["ltq", "buyers", "sellers"]] = stock_df[["ltq", "buyers", "sellers"]].ffill()

    # rename cols
    stock_df = stock_df.rename(columns={
        "ltp": f"ltp_{stock}",
        "ltq": f"ltq_{stock}",
        "buyers": f"buyers_{stock}",
        "sellers": f"sellers_{stock}",
    })
    return stock_df

def _load_stock(path, stock, date):
    stock_df = pd.read_parquet(path, engine="c")
    stock_df = df = df.iloc[:, :5]
    stock_df.columns = ["timestamp", f"ltp_{stock}", f"ltq_{stock}", f"buyers_{stock}", f"sellers_{stock}"]
    stock_df["timestamp"] = pd.to_datetime(
        date + " " + stock_df["timestamp"], format="%d%B%y %H:%M:%S:%f"
    )
    stock_df[[f"ltq_{stock}", f"buyers_{stock}", f"sellers_{stock}"]] = (
        stock_df[[f"ltq_{stock}", f"buyers_{stock}", f"sellers_{stock}"]].ffill()
    )
    return stock_df

def loadNifty50Stock(date, max_workers=8):
    stock_list = pd.read_parquet(f"{resource_dir}/nifty.parquet")["Symbol"].tolist()
    paths = [f"{data_dir}/NiftyEquity/{date}/{stock}.parquet" for stock in stock_list]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            tqdm(
                ex.map(_load_stock, paths, stock_list, [date] * len(stock_list)),
                total=len(stock_list),
            )
        )

    # Merge all with outer join on timestamp
    df = pd.concat(results).sort_values("timestamp")

    # Now group by timestamp (align across stocks)
    df = df.groupby("timestamp").first().reset_index()

    return df


## Helper Functions
def load_atm_scripts(date,window):
    '''
    Returns:
    Nifty,ATM_CE,ATM_PE
    '''
    nifty = loadNifty(date)
    spot = nifty['spot_price'][1]
    base_spot_floor = math.floor(spot / 50) * 50    
    ce_script = f'{base_spot_floor}CE'
    pe_script = f'{base_spot_floor}PE'
    ce = loadOptionMOB(date,ce_script,nifty,window)
    pe = loadOptionMOB(date,pe_script,nifty,window)
    return nifty,ce,pe