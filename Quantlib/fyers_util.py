import hashlib
import requests
from fyers_apiv3 import fyersModel
import pandas as pd
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import glob
from tqdm import tqdm
import os

class fyers_util:
    client_id = "NMZR8DS3BT-100"
    
    secret_key = os.getenv('FYERS_SECRET_KEY')
    if secret_key is None:
        raise ValueError("FYERS_SECRET_KEY environment variable is not set.")
    redirect_uri = "https://trade.fyers.in/api-login/redirect-uri/index.html"
    pin = "2232" 
    
    def get_token_path(self):
        """
        Returns a consistent cross-platform path for token.json.
        Windows → %APPDATA%\Quantlib\token.json
        Linux   → ~/.config/Quantlib/token.json
        """
        if os.name == "nt":  # Windows
            base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:  # Linux / macOS
            base_dir = Path.home() / ".config"

        token_dir = base_dir / "Quantlib"
        token_dir.mkdir(parents=True, exist_ok=True)

        return token_dir / "token.json"
    
        
    
    def __init__(self):
        self.tokenFile = self.get_token_path() 
        if Path(self.tokenFile).exists():
            with open(self.tokenFile, "r") as f:
                tokens = json.load(f)
                
            authTokenDate = datetime.strptime(tokens['auth_token_date'], '%Y-%m-%d %H:%M:%S')
            refreshTokenDate = datetime.strptime(tokens['refresh_token_date'], '%Y-%m-%d %H:%M:%S')
            isAboutToExpire = datetime.now() - refreshTokenDate >= timedelta(days=14)
            if isAboutToExpire:
                self.access_token,self.refresh_token,self.appIdHash = self.login()
            
            elif datetime.now() - authTokenDate > timedelta(hours=5):
                self.access_token =self.refreshAuthToken()
                print('Token-Refresh')
                print('Logged In - Token.json')    
            else :
                self.access_token = tokens['auth_token']
                self.appIdHash = tokens['app_id_hash']
        else: 
            self.access_token,self.refresh_token,self.appIdHash = self.login()
        thread = threading.Thread(target=self._token_refresher, daemon=True)
        thread.start()

    def refreshAuthToken(self):
        with open(self.tokenFile, "r") as f:
            tokens = json.load(f)
            url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
            payload = {
                "grant_type": "refresh_token",
                "appIdHash": tokens['app_id_hash'],
                "refresh_token": tokens["refresh_token"],
                "pin": self.pin
            }
            headers = {"Content-Type": "application/json"}

            r = requests.post(url, headers=headers, data=json.dumps(payload))
            r = r.json()
        tokens['auth_token'] = r['access_token']
        tokens['auth_token_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.tokenFile, "w") as f:
            json.dump(tokens, f, indent=4)
        return tokens['auth_token']

    def _token_refresher(self):
    
        while True:
            try:
                print(f"[{datetime.now()}] 🔄 Checking/refreshing token...")
                access_token = self.refreshAuthToken()
                print(f"[{datetime.now()}] ✅ Access token is now: {access_token[:10]}...")  # print first 10 chars
            except Exception as e:
                print(f"[{datetime.now()}] ❌ Error refreshing token:", e)

            
            time.sleep(5*60*60)

    def login(self): 
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code"
        )
        auth_code_url = session.generate_authcode()
        print(f"\n🔗 Open this URL in your browser and log in:\n{auth_code_url}")


        auth_code = input("\n📥 Paste the auth_code from redirected URL: ")
        combined = f"{self.client_id}:{self.secret_key}"
        appIdHash = hashlib.sha256(combined.encode()).hexdigest()
        validate_url = "https://api-t1.fyers.in/api/v3/validate-authcode"
        payload = {
            "grant_type": "authorization_code",
            "appIdHash": appIdHash,
            "code": auth_code
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(validate_url, json=payload, headers=headers)
        data = response.json()

        if data.get("s") == "ok":
            print("\n✅ Authentication Successful!")
            print("Access Token:", data["access_token"])
            print("Refresh Token:", data["refresh_token"])
        else:
            print("\n❌ Authentication Failed:")
            print("Message:", data.get("message", "Unknown error"))

        tokenData = {
            'auth_token' : data['access_token'],
            'auth_token_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'refresh_token' :  data['refresh_token'],
            'refresh_token_date' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'app_id_hash' : appIdHash
        }
        with open(self.tokenFile, "w") as f:
            json.dump(tokenData, f, indent=4)
        
        
        return data['access_token'], data["refresh_token"],appIdHash

    def get_fyers_historical_df(self, symbol: str, resolution: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Fyers API and return as a DataFrame.
    
        Parameters:
        - access_token (str): Valid Fyers access token
        - symbol (str): Symbol format like 'NSE:SBIN-EQ'
        - resolution (str): Timeframe (e.g. 'D', '5', '15', '60', '1')
        - start_date (str): Start date in 'YYYY-MM-DD'
        - end_date (str): End date in 'YYYY-MM-DD'
    
        Returns:
        - pd.DataFrame: DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Convert date strings to Unix timestamps
        from_timestamp = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
        to_timestamp = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))
    
        # Initialize Fyers client
        fyers = fyersModel.FyersModel(
            client_id=self.client_id,  # replace with actual client_id
            is_async=False,
            token=self.access_token,
            log_path=""
        )
    
        # Build request payload
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": str(from_timestamp),
            "range_to": str(to_timestamp),
            "cont_flag": "1"
        }
    
        # Get historical data
        response = fyers.history(data=data)
    
        # Handle response
        if response.get("s") == "ok" and "candles" in response:
            candles = response["candles"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
            df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
            return df
        else:
            raise ValueError(f"Failed to fetch data: {response}")
    

    def get_fyers_historical_full(fyer,symbol, start_date, end_date, resolution='D'):
        """
        Fetch historical data from Fyers API even if the date range exceeds API limit (50 days).
    
        symbol: str, e.g., 'NSE:NIFTY50-INDEX'
        start_date: str, 'YYYY-MM-DD'
        end_date: str, 'YYYY-MM-DD'
        resolution: str, e.g., 'D' for daily, '15' for 15-min
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        all_dfs = []
    
        while start <= end:
            chunk_end = min(start + timedelta(days=49), end)  # max 50 days per request
            print(f"Fetching: {start.date()} to {chunk_end.date()}")
    
            df_chunk = fyer.get_fyers_historical_df(
                symbol,
                resolution=resolution,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d")
            )
            all_dfs.append(df_chunk)
    
            start = chunk_end + timedelta(days=1)  # move to next chunk
    
        # Combine all chunks into one DataFrame
        full_df = pd.concat(all_dfs, ignore_index=True)
        return full_df

    def get_intraday_data(self,symbol,start_date,end_date,resolution):
        import time
        fyers = fyersModel.FyersModel(client_id=self.client_id,is_async=False,token=self.access_token)
        from_timestamp = int(time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S")))
        to_timestamp = int(time.mktime(time.strptime(end_date, "%Y-%m-%d %H:%M:%S")))
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": str(from_timestamp),
            "range_to": str(to_timestamp),
            "cont_flag": "1"
        }
    
                # Get historical data
        response = fyers.history(data=data)
        # Handle response
        if response.get("s") == "ok" and "candles" in response:
            candles = response["candles"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
            df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
            return df
        else:
            raise ValueError(f"Failed to fetch data: {response}")
    
    
    
# Load Methods
def load_index(date,exchange,index):
    nifty = pd.read_csv(f'E:/DB/{date}/{exchange}/INDEX/{index}.csv',encoding_errors='ignore',on_bad_lines='skip',engine='python')
    try:
        nifty['timestamp'] = pd.to_datetime(nifty['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
    except Exception as e:
        print(e)
    nifty = numericfy_df(nifty)
    nifty.dropna(subset=['timestamp','ltp'],inplace=True)
    return nifty.drop(columns=['exch_feed_time'])

def load_mob(dir):
    mob = pd.read_csv(dir,encoding_errors='ignore',on_bad_lines='skip',engine='python')
    mob['timestamp'] = pd.to_datetime(mob['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',dayfirst=True,errors='coerce')
    mob = numericfy_df(mob)
    return mob

def load_option(date,exchange,symbol,option,mob:bool):
    df = pd.read_csv(f'E:/DB/{date}/{exchange}/OPTIONS/{symbol}/{option}.csv',encoding_errors='ignore',on_bad_lines='skip',engine='python')
    df['strike'] = option[:-2]
    nifty = load_index(date,exchange,symbol)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',dayfirst=True,errors='coerce')
        df = df.dropna(subset=['timestamp'])
        nifty = nifty.dropna(subset=['timestamp'])
        nifty.rename(columns={'ltp':'spot_price'},inplace=True)
        df = df.sort_values('timestamp')
        nifty = nifty.sort_values('timestamp')
        df = pd.merge_asof(df,nifty[['timestamp', 'spot_price']],on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
        
        if mob:
            mob = load_mob(f'E:/DB/{date}/{exchange}/OPTIONS/{symbol}-MOB/{option}.csv')
            mob = mob.sort_values('timestamp')
            df = pd.merge_asof(df,mob,on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
            
        
    except Exception as e:
        print(option,e)
    df = numericfy_df(df)
    df = df[df['timestamp'].dt.date == int(date[0:2]) ]
    return df.drop(columns=['exch_feed_time'])
    
    
def load_option_chain(date,exchange,symbol,mob:bool):
    option_chain = {}
    files = glob.glob(f'E:/DB/{date}/{exchange}/OPTIONS/{symbol}/*')
    for f in tqdm(files):
        option = f.split('\\')[1].split('.')[0]
        try:
            option_chain[option] = load_option(date,exchange,symbol,option,mob)
        except Exception as e:
            print(option,e.add_note('Lol'))
    return option_chain

def load_stock(date,exchange,symbol,mob:bool):
    stock = pd.read_csv(f'E:/DB/{date}/{exchange}/EQUITY/{symbol}.csv',encoding_errors='ignore',on_bad_lines='skip',engine='python')
    try:
        stock['timestamp'] = pd.to_datetime(stock['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
        if mob:
            stock_mob = pd.read_csv(f'E:/DB/{date}/{exchange}/EQUITY-MOB/{symbol}.csv',encoding_errors='ignore',on_bad_lines='skip',engine='python')
            stock_mob['timestamp'] = pd.to_datetime(stock_mob['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
            stock = stock.sort_values('timestamp')
            stock_mob = stock_mob.sort_values('timestamp')
            stock = pd.merge_asof(stock,stock_mob,on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
            
            
    except Exception as e:
        print(e)
    stock = numericfy_df(stock)
    return stock.drop(columns=['exch_feed_time'])

import math    
import datetime as dt
def load_atm_options(date,exchange,symbol,mob:bool):
    underlying = load_index(date,exchange,symbol)
    start = underlying[underlying['timestamp'].dt.time >= dt.time(9, 15)]
    spot = start.iloc[0]['ltp']
    print('Spot:',spot)
    ce_spot = f'{math.floor(spot / 50) * 50}CE'
    pe_spot = f'{math.ceil(spot / 50) * 50 }PE' 
    print(ce_spot,pe_spot)
    ce = load_option(date,exchange,symbol,ce_spot,mob)
    pe = load_option(date,exchange,symbol,pe_spot,mob)
    return underlying,ce,pe

def numericfy_df(df):
    df.loc[:, df.columns != 'timestamp'] = df.loc[:, df.columns != 'timestamp'].apply(pd.to_numeric, errors='coerce')
    return df