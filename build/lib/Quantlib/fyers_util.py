import hashlib
import requests
from fyers_apiv3 import fyersModel
import pandas as pd
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import glob
from tqdm import tqdm
import os
import math    



class fyers_util:
    client_id = "NMZR8DS3BT-100"
    pin = "2232" 
    secret_key = os.getenv('FYERS_SECRET_KEY')
    
    if secret_key is None:
        raise ValueError("FYERS_SECRET_KEY environment variable is not set.")
    
    redirect_uri = "https://trade.fyers.in/api-login/redirect-uri/index.html"
    
    log_dir =  str(Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))) if os.name == 'nt' else str(Path.home() / ".config/fyers")
    
    def get_token_path(self):
        """
        Returns a consistent cross-platform path for token.json.
        Windows → %APPDATA%/Quantlib/token.json
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
        print(f"Token file path: {self.tokenFile}")

        try:
            with open(self.tokenFile, "r") as f:
                tokens = json.load(f)

            authTokenDate = datetime.strptime(
                tokens['auth_token_date'],
                '%Y-%m-%d %H:%M:%S'
            )

            isAboutToExpire = datetime.today().date() != authTokenDate.date()

            if isAboutToExpire:
                self.access_token, self.refresh_token, self.appIdHash = self.login()

            else:
                print("Logged In - Token.json")
                self.access_token = tokens['auth_token']
                self.appIdHash = tokens['app_id_hash']

        except Exception as e:
            print(f"Error reading token file: {e}. Initiating login.")
            self.access_token, self.refresh_token, self.appIdHash = self.login()    



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
    
    def get_fyers_historical_df(self, symbol: str,  start_date: str, end_date: str, resolution: str,) -> pd.DataFrame:
        """
        Fetch historical data from Fyers API and return as a DataFrame.

        Parameters:
        - symbol (str): Symbol format like 'NSE:SBIN-EQ'
        - resolution (str): Timeframe (e.g. 'D', '5', '15', '60', '1')
        - start_date (str): Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format
        - end_date (str): End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format

        Returns:
        - pd.DataFrame: DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        import time
        import pandas as pd
        from fyers_apiv3 import fyersModel

        # Detect date format and convert to timestamp
        def parse_date_to_timestamp(date_str):
            """Convert date string to Unix timestamp, handling both date-only and datetime formats."""
            try:
                # Try datetime format first (with time)
                dt = time.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                return int(time.mktime(dt))
            except ValueError:
                try:
                    # Try date-only format
                    dt = time.strptime(date_str, "%Y-%m-%d")
                    return int(time.mktime(dt))
                except ValueError:
                    raise ValueError(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")

        # Convert dates to Unix timestamps
        from_timestamp = parse_date_to_timestamp(start_date)
        to_timestamp = parse_date_to_timestamp(end_date)

        # Initialize Fyers model
        fyers = fyersModel.FyersModel(
            client_id=self.client_id, 
            is_async=False,
            token=self.access_token,
            log_path=self.log_dir
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

        if response.get("s") == "ok" and "candles" in response:
            candles = response["candles"]
            if candles:  # Check if we have data
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
                df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
                return df
            else:
                # No candles returned
                
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            # Handle no_data response gracefully
            if response.get("s") == "no_data":
                
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            else:
                raise ValueError(f"Failed to fetch data: {response}")

    
    def get_historic_db(self, symbol, start_date, end_date, resolution='D', data_dir="C:/historic_data/"):
        """
        Fetch historical data with caching.
        """
        from datetime import datetime, timedelta
        import os
        import pandas as pd

        os.makedirs(data_dir, exist_ok=True)

        symbol_clean = symbol.replace(":", "_").replace("-", "_")
        file_path = os.path.join(data_dir, f"{symbol_clean}_{resolution}.parquet")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        now = datetime.now()

        # Load existing data
        local_df = pd.DataFrame()
        if os.path.exists(file_path):
            try:
                local_df = pd.read_parquet(file_path)
                if not local_df.empty:
                    local_df["timestamp"] = pd.to_datetime(local_df["timestamp"])
                    local_df.sort_values("timestamp", inplace=True)
            except:
                local_df = pd.DataFrame()

        all_dfs = []
        fetch_ranges = []

        # Determine missing ranges
        if local_df.empty:
            fetch_ranges.append((start, end))
        else:
            local_start = local_df["timestamp"].min()
            local_end = local_df["timestamp"].max()

            # Check if today's data needs refresh
            refresh_today = False
            if end.date() >= now.date() and not local_df.empty:
                today_data = local_df[local_df["timestamp"].dt.date == now.date()]
                if today_data.empty or today_data["timestamp"].max() < now:
                    refresh_today = True
                    # Remove today's data from local_df
                    local_df = local_df[local_df["timestamp"].dt.date != now.date()]
                    # Update local_end after removing today's data
                    if not local_df.empty:
                        local_end = local_df["timestamp"].max()
                    else:
                        local_end = None

            if start < local_start:
                fetch_ranges.append((start, local_start - timedelta(days=1)))

            if end > local_end or refresh_today:
                if refresh_today and local_end is not None:
                    fetch_start = local_end + timedelta(days=1)
                elif refresh_today:
                    fetch_start = start
                else:
                    fetch_start = local_end + timedelta(days=1)

                fetch_end = end if end < now else now
                fetch_ranges.append((fetch_start, fetch_end))

        # Fetch in 50-day chunks
        for fetch_start, fetch_end in fetch_ranges:
            temp_start = fetch_start
            while temp_start <= fetch_end:
                chunk_end = min(temp_start + timedelta(days=49), fetch_end)

                # If chunk includes today, include time
                if chunk_end.date() == now.date():
                    end_str = now.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    end_str = chunk_end.strftime("%Y-%m-%d")

                try:
                    df_chunk = self.get_fyers_historical_df(
                        symbol,
                        temp_start.strftime("%Y-%m-%d"),
                        end_str
                        ,resolution
                    )
                    if not df_chunk.empty:
                        df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], format="%d-%m-%Y %H:%M:%S")
                        # Filter today's data to current time
                        if chunk_end.date() == now.date():
                            df_chunk = df_chunk[df_chunk['timestamp'] <= now]
                        all_dfs.append(df_chunk)
                except Exception as e:
                    print(f"Error fetching {temp_start} to {chunk_end}: {e}")

                temp_start = chunk_end + timedelta(days=1)

        # Combine and save
        if all_dfs:
            full_df = pd.concat([local_df] + all_dfs, ignore_index=True) if not local_df.empty else pd.concat(all_dfs, ignore_index=True)
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
            full_df.drop_duplicates(subset=["timestamp"], inplace=True)
            full_df.sort_values("timestamp", inplace=True)
            full_df.to_parquet(file_path, index=False)
        else:
            full_df = local_df
            if full_df.empty:
                full_df.to_parquet(file_path, index=False)

        # Return requested range
        if not full_df.empty:
            mask = (full_df["timestamp"] >= pd.to_datetime(start_date)) & (full_df["timestamp"] <= pd.to_datetime(end_date))
            return full_df.loc[mask].reset_index(drop=True)

        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    

    
        
    def option_chain(self,symbol:str,strike_count:int = 5,expiry: int = 0) -> dict:
        
        model = fyersModel.FyersModel(client_id=self.client_id, token=self.access_token,is_async=False, log_path=self.log_dir)
        
        data = {
            "symbol":symbol,
            "strikecount":strike_count,
            "timestamp": ""
        }
        response = model.optionchain(data=data)
        
        if expiry != 0:
            data['timestamp'] = response['data']['expiryData'][expiry]['expiry']
            response = model.optionchain(data=data)
        
        return response
        
    
    
    
# Load Methods
base_dir = "C:/DB/"
def load_index(date,exchange,index):
    nifty = pd.read_parquet(f'{base_dir}{date}/{exchange}/INDEX/{index}.parquet',engine='pyarrow')
    try:
        nifty['timestamp'] = (
            nifty['timestamp']
            .str.replace(r'(?<=\d{2}):(?=\d{6}$)', '.', regex=True)
            .pipe(pd.to_datetime, format="%d/%m/%Y %H:%M:%S.%f"))
        nifty = nifty.sort_values('timestamp')
        nifty = nifty.reset_index(drop=True)
    except Exception as e:
        print(e)
    nifty = numericfy_df(nifty)
    nifty.dropna(subset=['timestamp','ltp'],inplace=True)
    return nifty.drop(columns=['exch_feed_time'])

def load_mob(dir):
    mob = pd.read_parquet(dir,engine='pyarrow')
    mob['timestamp'] = pd.to_datetime(mob['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',dayfirst=True,errors='coerce')
    mob = numericfy_df(mob)
    return mob

def load_option(date,exchange,symbol,option,isIndex,mob:bool):
    df = pd.read_parquet(f'{base_dir}{date}/{exchange}/OPTIONS/{symbol}/{option}.parquet',engine='pyarrow')
    df['strike'] = option[:-2]
    underlying = load_index(date,exchange,symbol) if isIndex else load_stock(date,exchange,symbol,0)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',dayfirst=True,errors='coerce')
        df = df.dropna(subset=['timestamp'])
        underlying = underlying.dropna(subset=['timestamp'])
        underlying.rename(columns={'ltp':'spot_price'},inplace=True)
        df = df.sort_values('timestamp')
        underlying = underlying.sort_values('timestamp')
        df = pd.merge_asof(df,underlying[['timestamp', 'spot_price']],on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
        
        if mob:
            mob = load_mob(f'{base_dir}{date}/{exchange}/OPTIONS/{symbol}-MOB/{option}.parquet')
            mob = mob.sort_values('timestamp')
            df = pd.merge_asof(df,mob,on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
            
        
    except Exception as e:
        print(option,e)
    df = numericfy_df(df)
    
    return df.drop(columns=['exch_feed_time'])
     
def load_option_chain(date,exchange,symbol,isIndex=True,mob:bool=False):
    option_chain = {}
    files = glob.glob(f'{base_dir}{date}/{exchange}/OPTIONS/{symbol}/*')
    for f in tqdm(files):
        option = f.split('\\')[1].split('.')[0]
        try:
            option_chain[option] = load_option(date,exchange,symbol,option,isIndex,mob)
        except Exception as e:
            print(option,e.add_note('Lol'))
    return option_chain

def load_stock(date,exchange,symbol,mob:bool):
    stock = pd.read_parquet(f'{base_dir}{date}/{exchange}/EQUITY/{symbol}.parquet',engine='pyarrow')
    try:
        stock['timestamp'] = pd.to_datetime(stock['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
        if mob:
            stock_mob = pd.read_parquet(f'{base_dir}{date}/{exchange}/EQUITY-MOB/{symbol}.parquet',engine='pyarrow')
            stock_mob['timestamp'] = pd.to_datetime(stock_mob['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
            stock = stock.sort_values('timestamp')
            stock_mob = stock_mob.sort_values('timestamp')
            stock = pd.merge_asof(stock,stock_mob,on='timestamp',direction='backward',tolerance=pd.Timedelta('1s'))
            
            
    except Exception as e:
        print(e)
    stock = numericfy_df(stock)
    return stock.drop(columns=['exch_feed_time'])

def load_atm_options(date,exchange,symbol,strike_gap = 50,mob=False,is_index = False):
    
    underlying = load_index(date,exchange,symbol) if is_index else load_stock(date,exchange,symbol,0)
    start = underlying[underlying['timestamp'].dt.time >= dt.time(9, 15)]
    spot = start.iloc[0]['ltp']
    
    ce_spot = f'{math.floor(spot / strike_gap) * strike_gap}CE'
    pe_spot = f'{math.ceil(spot / strike_gap) * strike_gap }PE' 
    
    ce = load_option(date,exchange,symbol,ce_spot,isIndex=is_index,mob=mob)
    pe = load_option(date,exchange,symbol,pe_spot,isIndex=is_index,mob=mob)
    return underlying,ce,pe

### Futures Data
def load_futures(date,exchange,symbol,mob:bool,index=True):
    path = base_dir+date+'/'+exchange+'/FUTURES/'+symbol+'.parquet'
    mob_path = base_dir+date+'/'+exchange+'/FUTURES-MOB/'+symbol+'.parquet'
    df = pd.read_parquet(path,engine='pyarrow')
    underlying = load_index(date,exchange,symbol) if index else load_stock(date,exchange,symbol,0)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
    df.sort_values(by='timestamp',inplace=True)
    underlying.rename(columns={'ltp':'spot_price'},inplace=True)
    df = pd.merge_asof(df,underlying[['timestamp','spot_price']],on='timestamp',direction='backward',tolerance=pd.Timedelta('500s'),)
    
    if mob:
        mob = pd.read_parquet(mob_path,engine='pyarrow')
        mob['timestamp'] = pd.to_datetime(mob['timestamp'],format='%d/%m/%Y %H:%M:%S:%f',errors='coerce')
        mob.sort_values(by='timestamp',inplace=True)
        df = pd.merge_asof(df,mob,on='timestamp',direction='backward',tolerance=pd.Timedelta('500s'),)
        
    
    return df

def numericfy_df(df):
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_dates(reversed=False):
    paths = glob.glob(os.path.join(base_dir, "*"))
    names = [os.path.basename(p) for p in paths]

    def parse_date(s):
        return datetime.strptime(s, "%d%B%y")

    names = sorted(names, key=parse_date,reverse=reversed)
    return names


def fetchOI(symbol:str,
            date:str,
            start="09:00",
            end="15:00",
            
        ):
    
    url = "http://103.194.228.194:8001/option-chain/range"
    params = {
        "symbol": symbol,
        "date": date,
        "start": start,
        "end": end
    }

    response = requests.get(
            url,
            params=params
        )
    
    df = pd.DataFrame(
        response.json()["rows"]
    )
    
    return df.drop(columns=['filename', 'file_time'], errors='ignore')

