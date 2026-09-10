import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

class LiveFeatureBuilder:

    def __init__(
        self,
        f,
        symbol,
        resolution,
        preprocess_function,
        preprocess_kwargs=None,
        past_days=5,
        end_date=None,
        max_history=5000,
    ):

        self.fyers = f
        self.symbol = symbol
        self.resolution = resolution

        self.preprocess_function = preprocess_function
        self.preprocess_kwargs = preprocess_kwargs or {}

        self.max_history = max_history

        # ---------- Dates ----------

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # Go back N business days
        start_dt = end_dt - BDay(past_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        # ---------- Fetch History ----------

        self.df = self.fyers.get_historic_db(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            resolution=self.resolution,
        )
        self.df = self.df[(self.df["timestamp"].dt.time >= pd.to_datetime("09:15").time()) &(self.df["timestamp"].dt.time <= pd.to_datetime("15:30").time())]
        
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if end_date == now.strftime("%Y-%m-%d") and (now.hour > 9 or (now.hour == 9 and now.minute > 15)):
            
            market_open = now.normalize() + pd.Timedelta(hours=9, minutes=15)
            market_open = market_open.strftime("%Y-%m-%d %H:%M:%S")
            right_now = now.strftime("%Y-%m-%d %H:%M:%S")

            intraday_data = self.fyers.get_fyers_historical_df(self.symbol,market_open,right_now,self.resolution)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"],format="%d-%m-%Y %H:%M:%S")
            intraday_data["timestamp"] = pd.to_datetime(intraday_data["timestamp"],format="%d-%m-%Y %H:%M:%S")
            self.df = pd.concat([self.df,intraday_data],ignore_index=True)
        

        self.df, self.features = self.preprocess_function(self.df,**self.preprocess_kwargs)

    def update(self, candle):
        """
        Add a completed candle and regenerate features.

        Returns
        -------
        pd.Series
            Latest feature row.
        """

        self.df = pd.concat(
            [self.df, pd.DataFrame([candle])],
            ignore_index=True,
        )

        if len(self.df) > self.max_history:
            self.df = (
                self.df
                .iloc[-self.max_history:]
                .reset_index(drop=True)
            )

        self.df, self.features = self.preprocess_function(
            self.df,
            **self.preprocess_kwargs
        )

        return self.df.iloc[-1]

    def latest(self):
        return self.df.iloc[-1]

    def history(self):
        return self.df.copy()

    def get_features(self):
        return self.features

    def get_dataframe(self):
        return self.df