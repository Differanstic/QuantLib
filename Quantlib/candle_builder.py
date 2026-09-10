import pandas as pd


class CandleBuilder:
    """
    Build OHLCV candles from tick data.

    Required tick fields:
        timestamp
        ltp

    Optional tick fields:
        vol_traded_today
    """

    def __init__(self, timeframe="1min"):
        self.timeframe = timeframe
        self.current_period = None
        self.candle = None

    def update(self, tick):

        ts = pd.Timestamp(tick["timestamp"])
        period = ts.floor(self.timeframe)

        price = float(tick["ltp"])

        # Optional cumulative volume
        cum_volume = tick.get("vol_traded_today")
        if cum_volume is not None:
            cum_volume = float(cum_volume)

        # ---------------- First Candle ----------------

        if self.candle is None:

            self.current_period = period

            self.candle = {
                "timestamp": period,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,

                # internal
                "_start_volume": cum_volume,
                "_last_volume": cum_volume,
            }

            return None

        # ---------------- New Candle ----------------

        if period != self.current_period:

            completed = self.candle.copy()
            completed.pop("_start_volume")
            completed.pop("_last_volume")

            self.current_period = period

            self.candle = {
                "timestamp": period,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,

                "_start_volume": cum_volume,
                "_last_volume": cum_volume,
            }

            return completed

        # ---------------- Update ----------------

        self.candle["high"] = max(self.candle["high"], price)
        self.candle["low"] = min(self.candle["low"], price)
        self.candle["close"] = price

        # Update volume only if cumulative volume is available
        if cum_volume is not None:

            # First volume tick after starting without volume
            if self.candle["_start_volume"] is None:
                self.candle["_start_volume"] = cum_volume

            # Handle daily reset of cumulative volume
            elif (
                self.candle["_last_volume"] is not None
                and cum_volume < self.candle["_last_volume"]
            ):
                self.candle["_start_volume"] = cum_volume

            self.candle["_last_volume"] = cum_volume
            self.candle["volume"] = max(
                0,
                cum_volume - self.candle["_start_volume"]
            )

        return None

    def flush(self):

        if self.candle is None:
            return None

        candle = self.candle.copy()

        candle.pop("_start_volume")
        candle.pop("_last_volume")

        self.candle = None

        return candle