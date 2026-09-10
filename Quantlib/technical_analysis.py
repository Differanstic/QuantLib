import pandas as pd

import numpy as np
from datetime import datetime,timedelta


def _parse_analysis_date(value):
    if value is None:
        return datetime.now().date()

    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    formats = ("%Y-%m-%d", "%d%B%y", "%d%b%y", "%d-%m-%Y", "%d/%m/%Y")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    raise ValueError(
        "target_date must be a datetime/date or string in YYYY-MM-DD, DDMonthYY, "
        "DDMonYY, DD-MM-YYYY, or DD/MM/YYYY format"
    )


def _relative_percentile(current_value, history):
    history = history.dropna()
    if history.empty or pd.isna(current_value):
        return np.nan

    return (history <= current_value).mean()


def getRelativePercentile(
    fyers,
    symbol,
    tf,
    target_date=None,
    range=20,
    start=None,
    end=None,
    is_backtest=False
):
    """
    Analyse volume and candle range on target_date against the prior lookback range.

    Parameters
    ----------
    fyers : object
        Object exposing get_fyers_historical_df(symbol, tf, start, end).
    symbol : str
        Fyers symbol, for example 'NSE:SBIN-EQ'.
    tf : str
        Fyers timeframe/resolution, for example '5', '15', '60', or 'D'.
    target_date : str | datetime.date | datetime.datetime, optional
        Trading date to analyse. Defaults to today. For backward compatibility,
        `end` is used as the target date when target_date is omitted.
    range : int, optional
        Number of calendar days before target_date used as the baseline.
    start : str, optional
        Optional explicit baseline start date in YYYY-MM-DD format.
    end : str, optional
        Backward-compatible alias for target_date.
    is_backtest : bool, optional
        When False, shifts the returned target timestamps by one timeframe.

    Returns
    -------
    dict
        Contains metadata, target-day enriched candles, baseline profile by time,
        and a compact latest-candle summary.
    """
    if target_date is None:
        target_date = end

    lookback_days = 20 if range is None else int(range)
    analysis_date = _parse_analysis_date(target_date)
    fetch_end_date = analysis_date + timedelta(days=1)

    if start is None:
        fetch_start_date = analysis_date - timedelta(days=lookback_days)
        start = fetch_start_date.strftime("%Y-%m-%d")

    fetch_end = fetch_end_date.strftime("%Y-%m-%d")

    df = fyers.get_historic_db(symbol, start, fetch_end, tf)
    if df.empty:
        return {
            "metadata": {
                "symbol": symbol,
                "tf": tf,
                "target_date": analysis_date.strftime("%Y-%m-%d"),
                "start": start,
                "end": fetch_end,
                "lookback_days": lookback_days,
            },
            "target": pd.DataFrame(),
            "profile": pd.DataFrame(),
            "summary": {},
            "message": "No historical data returned for the requested period.",
        }

    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M:%S")
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    df['candle_range'] = (df['high'] - df['low']).abs()
    df['downside_range'] = (df['close'] - df['low']).abs()

    baseline = df[df['date'] < analysis_date].copy()
    target = df[df['date'] == analysis_date].copy()

    profile = (
        baseline.groupby('time')
        .agg(
            sample_count=('volume','count'),
            avg_volume=('volume','mean'),
            median_volume=('volume','median'),
            avg_candle_range=('candle_range','mean'),
            median_candle_range=('candle_range','median'),
            avg_downside_range=('downside_range','mean'),
        )
        .reset_index()
    )

    if not target.empty and not profile.empty:
        target = target.merge(profile, on='time', how='left')

        volume_history = baseline.groupby('time')['volume']
        range_history = baseline.groupby('time')['candle_range']
        downside_history = baseline.groupby('time')['downside_range']

        target['volume_percentile'] = target.apply(
            lambda row: _relative_percentile(row['volume'], volume_history.get_group(row['time']))
            if row['time'] in volume_history.groups else np.nan,
            axis=1
        )
        target['range_percentile'] = target.apply(
            lambda row: _relative_percentile(row['candle_range'], range_history.get_group(row['time']))
            if row['time'] in range_history.groups else np.nan,
            axis=1
        )
        target['downside_range_percentile'] = target.apply(
            lambda row: _relative_percentile(row['downside_range'], downside_history.get_group(row['time']))
            if row['time'] in downside_history.groups else np.nan,
            axis=1
        )
        target['rvol'] = target['volume'] / target['avg_volume']
        target['range_ratio'] = target['candle_range'] / target['avg_candle_range']

    if not is_backtest and not target.empty:
        try:
            target['timestamp'] = target['timestamp'] + pd.Timedelta(minutes=int(tf))
        except ValueError:
            pass

    summary = {}
    if not target.empty:
        latest = target.iloc[-1]
        summary = {
            "timestamp": latest['timestamp'],
            "close": latest['close'],
            "volume": latest['volume'],
            "avg_volume": latest.get('avg_volume', np.nan),
            "rvol": latest.get('rvol', np.nan),
            "volume_percentile": latest.get('volume_percentile', np.nan),
            "candle_range": latest['candle_range'],
            "avg_candle_range": latest.get('avg_candle_range', np.nan),
            "range_percentile": latest.get('range_percentile', np.nan),
            "range_ratio": latest.get('range_ratio', np.nan),
        }

    return {
        "metadata": {
            "symbol": symbol,
            "tf": tf,
            "target_date": analysis_date.strftime("%Y-%m-%d"),
            "start": start,
            "end": fetch_end,
            "lookback_days": lookback_days,
            "baseline_rows": len(baseline),
            "target_rows": len(target),
        },
        "target": target.reset_index(drop=True),
        "profile": profile,
        "summary": summary,
        "message": "Analysis complete." if not target.empty else "No candles found on target_date.",
    }
