import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm  # Optional: for progress bar
from scipy.stats import entropy, gaussian_kde, skew
from numpy.fft import fftfreq,fft
from numba import njit


### Calculation Algorithms
def check_target(df,price_col='ltp',tp_col='tp',sl_col='sl', lookahead=100):
    """
    For each row, check if TP or SL is hit first in the next `lookahead` ticks.
    
    Adds:
        - 'target_hit': 1 = TP hit, 0 = SL hit, -1 = neither
        - 'steps': number of ticks it took to hit TP/SL, else -1
    """
    df = df.dropna(subset=[tp_col,sl_col]).reset_index(drop=True)
    df = df.copy().reset_index(drop=True)  # Ensure integer index

    result = []
    steps_list = []

    ltp_array = df[price_col].values
    tp_array = df[tp_col].values
    sl_array = df[sl_col].values
    n = len(df)

    for i in range(n):
        tp = tp_array[i]
        sl = sl_array[i]

        hit = -1  # Default: neither hit
        steps = -1
        for j in range(i + 1, min(i + lookahead + 1, n)):
            ltp = ltp_array[j]
            if ltp >= tp:
                hit = 1
                steps = j - i
                break
            elif ltp <= sl:
                hit = 0
                steps = j - i
                break

        result.append(hit)
        steps_list.append(steps)

    df['target_hit'] = result
    df['steps'] = steps_list
    return df


### Time Functions
def tick_rate(df:pd.DataFrame,time_col:str,window):
    """Calc Tick Rate of DF

    Args:
        df (DataFrame): _description_
        time_col(str)
        window (Int): _description_
    Returns:
        df(Dataframe)
    """
    df['time_diff_ms'] = df[time_col].diff().dt.total_seconds() * 1000
    df['time_diff_ms_mean'] = df['time_diff_ms'].rolling(window=window).mean()
    df['tick_rate'] = 1_000 / df['time_diff_ms_mean']
    return df['tick_rate']


### Stats Functions
def ema(series, period) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ema_timeframe(series: pd.Series, timeframe: str) -> pd.Series:
    """
    Calculate EMA (Exponential Moving Average) based on a timeframe string.

    Parameters
    ----------
    series : pd.Series
        Price or value series with a DatetimeIndex or same frequency spacing.
    timeframe : str
        Time period like '1min', '5min', '15min', '1H', '1D', etc.

    Returns
    -------
    pd.Series
        EMA of the series using span derived from timeframe.
    """

    # --- infer frequency ---
    if isinstance(series.index, pd.DatetimeIndex):
        try:
            inferred_freq = pd.infer_freq(series.index)
            if inferred_freq is None:
                # fallback: assume 1s spacing if unknown
                inferred_freq = '1s'
        except Exception:
            inferred_freq = '1s'
    else:
        # if no datetime index, assume 1-unit frequency
        inferred_freq = '1s'

    # convert both to pandas Timedelta
    tf_delta = pd.to_timedelta(timeframe)
    freq_delta = pd.to_timedelta(inferred_freq)

    # number of samples per timeframe
    span = max(int(tf_delta / freq_delta), 1)

    # --- calculate EMA ---
    ema = series.ewm(span=span, adjust=False).mean()
    return ema

def mutual_info(df:pd.DataFrame,features:list,target_col:str):
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_classif
    X = df[features].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    y = df[target_col]
    mi = mutual_info_classif(X, y, discrete_features=False)
    mi_df = pd.DataFrame({'feature': X.columns, 'mutual_info': mi}).sort_values(by='mutual_info', ascending=False)
    return mi_df
    
    
    
@njit
def hurst(ts):
    N = len(ts)
    mean_ts = np.mean(ts)
    dev_ts = ts - mean_ts
    cumulative_dev = np.cumsum(dev_ts)
    R = np.max(cumulative_dev) - np.min(cumulative_dev)
    S = np.std(ts)
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(N)

def rolling_hurst(df, col='ltp', window=100):
    """Calc Rolling Hurst

    Args:
        df (Dataframe): _description_
        col (str, optional): _description_. Defaults to 'ltp'.
        window (int, optional): _description_. Defaults to 100.

    Returns:
        Series
    """
    values = df[col].values
    hursts = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(window, len(values)):
        hursts[i] = hurst(values[i - window:i])
    
    return pd.Series(hursts)

def rolling_kde(series, window=100, grid_points=100, bw_adjust=1.0):
    series = series.dropna()
    kde_peaks = np.full(len(series), np.nan)
    x_grid = np.linspace(series.min(), series.max(), grid_points)

    for i in tqdm(range(window, len(series)), desc="Rolling KDE"):
        window_data = series.iloc[i - window:i].values
        kde = gaussian_kde(window_data, bw_method='scott')
        kde.set_bandwidth(kde.factor * bw_adjust)
        kde_vals = kde(x_grid)
        kde_peaks[i] = x_grid[np.argmax(kde_vals)]  # Mode of KDE

    return pd.Series(kde_peaks, index=series.index)

def rolling_kde_skew(series, window=100, grid_points=100):
    series = series.dropna()
    kde_skews = np.full(len(series), np.nan)
    x_grid = np.linspace(series.min(), series.max(), grid_points)

    for i in tqdm(range(window, len(series))):
        window_data = series.iloc[i - window:i].values
        kde = gaussian_kde(window_data)
        kde_vals = kde(x_grid)
        kde_skews[i] = skew(kde_vals)

    return pd.Series(kde_skews, index=series.index)

def kde_tp_sl_prob(df:pd.DataFrame, window=100, sl_col='sl', target_col='tp', ltp_col='ltp', grid_size=500):
    """Calc Tp Hitting and SL Prob Based on KDE

    Args:
        df (Dataframe): _description_
        window (int, optional): _description_. Defaults to 100.
        sl_col (str, optional): _description_. Defaults to 'sl'.
        target_col (str, optional): _description_. Defaults to 'tp'.
        ltp_col (str, optional): _description_. Defaults to 'ltp'.
        grid_size (int, optional): _description_. Defaults to 500.

    Returns:
        DataFrame: _description_
    """
    prob_lt_sl = np.full(len(df), np.nan)
    prob_gt_target = np.full(len(df), np.nan)

    ltp = df[ltp_col].values
    sls = df[sl_col].values
    targets = df[target_col].values

    for i in range(window, len(df)):
        ltp_window = ltp[i - window:i]

        # Skip flat windows
        if np.std(ltp_window) < 1e-6:
            prob_lt_sl[i] = 0.0
            prob_gt_target[i] = 0.0
            continue

        kde = gaussian_kde(ltp_window)
        
        # Grid for integration (approximation)
        x_grid = np.linspace(np.min(ltp_window) - 3, np.max(ltp_window) + 3, grid_size)
        pdf = kde.evaluate(x_grid)
        dx = x_grid[1] - x_grid[0]
        cdf = np.cumsum(pdf) * dx  # cumulative distribution function

        # Find indices for sl and target on the x_grid
        sl_idx = np.searchsorted(x_grid, sls[i])
        tgt_idx = np.searchsorted(x_grid, targets[i])

        prob_lt_sl[i] = cdf[sl_idx] if sl_idx < len(cdf) else 1.0
        prob_gt_target[i] = 1.0 - cdf[tgt_idx] if tgt_idx < len(cdf) else 0.0

    # Add to DataFrame
    df['prob_lt_sl'] = prob_lt_sl
    df['prob_gt_target'] = prob_gt_target
    return df

def calc_fft_energy(df,col_name,window):
    """
        Calculate rolling mean for a given column.
    
        Args:
            df (pd.DataFrame): Input dataframe.
            col_name (str): Column name to compute rolling mean on.
            window (int): Size of the rolling window.
    
        Returns:
            pd.Series: Rolling mean of the column.
    """
    def get_fft_bands_energy(series, fs=1.0):
        N = len(series)
        freqs = fftfreq(N, d=1/fs)
        fft_vals = np.abs(fft(series))

        # Only use positive frequencies
        mask = freqs > 0
        freqs = freqs[mask]
        fft_vals = fft_vals[mask]

        total_energy = np.sum(fft_vals)

        low = np.sum(fft_vals[(freqs >= 0) & (freqs < 0.1)]) / total_energy
        mid = np.sum(fft_vals[(freqs >= 0.1) & (freqs < 0.3)]) / total_energy
        high = np.sum(fft_vals[freqs >= 0.3]) / total_energy

        dominant_freq = freqs[np.argmax(fft_vals)]
        spectral_entropy = entropy(fft_vals / np.sum(fft_vals))

        return low, mid, high, dominant_freq, spectral_entropy
    
    results = {
        'fft_low': [], 'fft_mid': [], 'fft_high': [],
        'fft_dom_freq': [], 'fft_entropy': []
    }
    series = df[col_name]
    for i in range(len(series)):
        if i < window:
            for key in results: results[key].append(np.nan)
            continue
        window_series = series.iloc[i - window:i]
        
        low, mid, high, dom_freq, spec_entropy = get_fft_bands_energy(window_series)
        results['fft_low'].append(low)
        results['fft_mid'].append(mid)
        results['fft_high'].append(high)
        results['fft_dom_freq'].append(dom_freq)
        results['fft_entropy'].append(spec_entropy)
    for key in results:
        df[f'{col_name}_{key}'] = results[key]
    return df

def QuantBS(df:pd.DataFrame,window:int,min_period = 2,operation = np.sum) -> pd.DataFrame:
    '''
    Quantise Buying and Selling Using LTQ
    '''
    df['change'] = df['ltp'].diff()
    
    df['bp'] = np.where(df['change'] > 0, df["ltq"] * df["change"], 0)
    df['bs'] = np.where(df['change'] < 0, abs(df["ltq"] * df["change"]), 0)

    df['call'] = df['bp'].rolling(window, min_periods=min_period).apply(operation, raw=True)
    df['put'] = df['bs'].rolling(window, min_periods=min_period).apply(operation, raw=True)
    return df

def calc_large_order_clusters(df, volume_col="ltq", window=500, percentile=0.99):
    x = df.copy()
    threshold = np.percentile(x[volume_col].dropna(), percentile*100)
    x["large_trade"] = (x[volume_col]  >= threshold).astype(int) 
    clusters = (x["large_trade"]
                  .rolling(window, min_periods=window)
                  .sum())
    
    x["large_orders"] = clusters
    return x, threshold

def calc_rolling_volatility(df:pd.DataFrame,window:int,price_col='ltp'):
    df = df.copy()
    df['log_ret'] = np.log(df['ltp'] / df['ltp'].shift(1))
    rolling_std = df['log_ret'].rolling(window=window).std()
    return rolling_std

def count_swing_hh_ll(df, price_col="ltp", window=10):
    """
    Count number of swing Higher Highs (HH) and Lower Lows (LL) 
    in the last `window` bars.
    
    HH = price == rolling max
    LL = price == rolling min
    """
    prices = df[price_col]

    rolling_max = prices.rolling(window+1, min_periods=1).max()
    rolling_min = prices.rolling(window+1, min_periods=1).min()

    # HH occurs if current price == highest in last window
    df["HH"] = (prices == rolling_max).astype(int)

    # LL occurs if current price == lowest in last window
    df["LL"] = (prices == rolling_min).astype(int)

    # cumulative counts in last n window
    df["HH"] = df["HH"].rolling(window, min_periods=1).sum()
    df["LL"] = df["LL"].rolling(window, min_periods=1).sum()

    return df



    