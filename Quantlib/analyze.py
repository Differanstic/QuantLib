import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Helper functions for the analysis
def calculate_vwap(df: pd.DataFrame) -> float:
    """Calculate Volume Weighted Average Price."""
    if 'volume' not in df.columns:
        return None
    return (df['ltp'] * df['volume']).sum() / df['volume'].sum()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(df) < period:
        return None
    
    high = df['ltp'].rolling(window=period).max()
    low = df['ltp'].rolling(window=period).min()
    close = df['ltp'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]


def calculate_realized_volatility(df: pd.DataFrame) -> float:
    """Calculate realized volatility (annualized)."""
    returns = df['ltp'].pct_change().dropna()
    if len(returns) < 2:
        return 0
    
    # Assuming 6.5 trading hours per day, 252 trading days
    hourly_std = returns.std()
    annualized_vol = hourly_std * np.sqrt(6.5 * 252)
    return annualized_vol * 100  # Return as percentage


def calculate_max_drawdown(prices: pd.Series) -> dict:
    """Calculate maximum drawdown."""
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max * 100
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    return {
        'drawdown_percent': max_dd,
        'drawdown_points': prices.loc[max_dd_idx] - cumulative_max.loc[max_dd_idx],
        'drawdown_start': cumulative_max.idxmax(),
        'drawdown_end': max_dd_idx,
        'drawdown_duration': max_dd_idx - cumulative_max.idxmax()
    }


def calculate_max_runup(prices: pd.Series) -> dict:
    """Calculate maximum runup."""
    cumulative_min = prices.cummin()
    runup = (prices - cumulative_min) / cumulative_min * 100
    max_ru = runup.max()
    max_ru_idx = runup.idxmax()
    
    return {
        'runup_percent': max_ru,
        'runup_points': prices.loc[max_ru_idx] - cumulative_min.loc[max_ru_idx],
        'runup_start': cumulative_min.idxmin(),
        'runup_end': max_ru_idx,
        'runup_duration': max_ru_idx - cumulative_min.idxmin()
    }


def calculate_volatility_ratio(df: pd.DataFrame) -> float:
    """Calculate volatility ratio (close-to-close vs. true range)."""
    if len(df) < 2:
        return 0
    
    close_to_close = df['ltp'].pct_change().std()
    true_range = calculate_atr(df)
    
    if true_range is None or true_range == 0:
        return 0
    
    return close_to_close / true_range


def identify_trend_direction(df: pd.DataFrame) -> str:
    """Identify the dominant trend direction."""
    if len(df) < 20:
        return 'insufficient_data'
    
    # Simple moving average trend detection
    short_window = min(10, len(df) // 4)
    long_window = min(30, len(df) // 2)
    
    short_ma = df['ltp'].rolling(window=short_window).mean().iloc[-1]
    long_ma = df['ltp'].rolling(window=long_window).mean().iloc[-1]
    current_price = df['ltp'].iloc[-1]
    
    if current_price > short_ma > long_ma:
        return 'strong_uptrend'
    elif current_price < short_ma < long_ma:
        return 'strong_downtrend'
    elif current_price > short_ma:
        return 'weak_uptrend'
    elif current_price < short_ma:
        return 'weak_downtrend'
    else:
        return 'rangebound'


def calculate_trend_strength(df: pd.DataFrame) -> float:
    """Calculate trend strength using linear regression."""
    if len(df) < 10:
        return 0
    
    x = np.arange(len(df))
    y = df['ltp'].values
    
    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return r_squared


def calculate_intraday_momentum(df: pd.DataFrame) -> dict:
    """Calculate intraday momentum metrics."""
    if len(df) < 2:
        return {}
    
    # Calculate returns for different time periods
    returns = {}
    periods = [1, 5, 15, 30, 60]  # minutes back
    
    for period in periods:
        if len(df) > period:
            returns[f'return_{period}min'] = (
                (df['ltp'].iloc[-1] - df['ltp'].iloc[-period-1]) / 
                df['ltp'].iloc[-period-1] * 100
            )
        else:
            returns[f'return_{period}min'] = None
    
    # Identify if momentum is accelerating
    if all(v is not None for v in returns.values()):
        momentum_acceleration = (
            returns['return_1min'] > returns['return_5min'] > 
            returns['return_15min'] > returns['return_30min']
        )
        returns['momentum_accelerating'] = bool(momentum_acceleration)
    
    return returns


def calculate_price_velocity(df: pd.DataFrame) -> float:
    """Calculate price velocity (points per hour)."""
    if len(df) < 2:
        return 0
    
    time_diff_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    if time_diff_hours == 0:
        return 0
    
    price_change = df['ltp'].iloc[-1] - df['ltp'].iloc[0]
    return price_change / time_diff_hours


def analyze_session_trend(df: pd.DataFrame) -> dict:
    """Analyze trend behavior in different session parts."""
    if len(df) < 10:
        return {}
    
    # Split session into parts
    session_parts = {
        'opening_30min': df.between_time('09:15', '09:45') if not df.empty else pd.DataFrame(),
        'morning': df.between_time('09:45', '12:00') if not df.empty else pd.DataFrame(),
        'afternoon': df.between_time('12:00', '14:30') if not df.empty else pd.DataFrame(),
        'closing_30min': df.between_time('14:30', '15:00') if not df.empty else pd.DataFrame(),
    }
    
    trends = {}
    for part_name, part_data in session_parts.items():
        if len(part_data) > 1:
            trends[part_name] = {
                'change': (part_data['ltp'].iloc[-1] - part_data['ltp'].iloc[0]) 
                         if len(part_data) > 0 else 0,
                'direction': 'up' if part_data['ltp'].iloc[-1] > part_data['ltp'].iloc[0] 
                           else 'down' if part_data['ltp'].iloc[-1] < part_data['ltp'].iloc[0] 
                           else 'flat',
                'volatility': part_data['ltp'].std() if len(part_data) > 1 else 0,
            }
    
    return trends


def calculate_time_in_range(df: pd.DataFrame, num_zones: int = 5) -> dict:
    """Calculate percentage of time spent in different price zones."""
    price_range = df['ltp'].max() - df['ltp'].min()
    if price_range == 0:
        return {}
    
    zone_width = price_range / num_zones
    zones = {}
    
    for i in range(num_zones):
        lower = df['ltp'].min() + i * zone_width
        upper = lower + zone_width
        
        time_in_zone = ((df['ltp'] >= lower) & (df['ltp'] < upper)).sum()
        percent_in_zone = (time_in_zone / len(df)) * 100
        
        zones[f'zone_{i+1}'] = {
            'range': f"{lower:.2f}-{upper:.2f}",
            'time_percent': percent_in_zone,
            'center_price': (lower + upper) / 2
        }
    
    return zones


def identify_price_zones(df: pd.DataFrame, num_zones: int = 3) -> list:
    """Identify most common price zones (clustering)."""
    from scipy import stats
    
    if len(df) < 10:
        return []
    
    # Use KDE to find price density peaks
    prices = df['ltp'].values
    kde = stats.gaussian_kde(prices)
    
    # Evaluate KDE over price range
    x = np.linspace(prices.min(), prices.max(), 100)
    density = kde(x)
    
    # Find peaks (simplified)
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i-1] and density[i] > density[i+1]:
            peaks.append({
                'price': x[i],
                'density': density[i],
                'zone_low': x[i] - (x[1] - x[0]) * 10,
                'zone_high': x[i] + (x[1] - x[0]) * 10
            })
    
    # Return top N zones by density
    peaks.sort(key=lambda x: x['density'], reverse=True)
    return peaks[:num_zones]


def identify_support_levels(df: pd.DataFrame, lookback: int = 20) -> list:
    """Identify potential support levels (local minima)."""
    if len(df) < lookback * 2:
        return []
    
    supports = []
    prices = df['ltp'].values
    
    for i in range(lookback, len(prices) - lookback):
        if (prices[i] == min(prices[i-lookback:i+lookback+1]) and 
            prices[i] < prices[i-lookback] and 
            prices[i] < prices[i+lookback]):
            supports.append({
                'price': prices[i],
                'time': df.index[i],
                'strength': calculate_level_strength(df, prices[i])
            })
    
    # Remove duplicates (nearby levels)
    unique_supports = []
    for sup in supports:
        if not any(abs(sup['price'] - u['price']) < (df['ltp'].std() * 0.1) 
                  for u in unique_supports):
            unique_supports.append(sup)
    
    return sorted(unique_supports, key=lambda x: x['strength'], reverse=True)[:5]


def identify_resistance_levels(df: pd.DataFrame, lookback: int = 20) -> list:
    """Identify potential resistance levels (local maxima)."""
    if len(df) < lookback * 2:
        return []
    
    resistances = []
    prices = df['ltp'].values
    
    for i in range(lookback, len(prices) - lookback):
        if (prices[i] == max(prices[i-lookback:i+lookback+1]) and 
            prices[i] > prices[i-lookback] and 
            prices[i] > prices[i+lookback]):
            resistances.append({
                'price': prices[i],
                'time': df.index[i],
                'strength': calculate_level_strength(df, prices[i])
            })
    
    # Remove duplicates
    unique_resistances = []
    for res in resistances:
        if not any(abs(res['price'] - u['price']) < (df['ltp'].std() * 0.1) 
                  for u in unique_resistances):
            unique_resistances.append(res)
    
    return sorted(unique_resistances, key=lambda x: x['strength'], reverse=True)[:5]


def calculate_level_strength(df: pd.DataFrame, level: float) -> float:
    """Calculate strength of a support/resistance level."""
    tolerance = df['ltp'].std() * 0.05
    touches = ((df['ltp'] >= level - tolerance) & 
               (df['ltp'] <= level + tolerance)).sum()
    recent_touches = ((df['ltp'].iloc[-20:] >= level - tolerance) & 
                      (df['ltp'].iloc[-20:] <= level + tolerance)).sum()
    
    return (touches * 0.7) + (recent_touches * 0.3)


def calculate_pivot_points(df: pd.DataFrame) -> dict:
    """Calculate traditional pivot points."""
    if len(df) < 1:
        return {}
    
    high = df['ltp'].max()
    low = df['ltp'].min()
    close = df['ltp'].iloc[-1]
    
    pp = (high + low + close) / 3
    
    return {
        'pivot': pp,
        'r1': (2 * pp) - low,
        'r2': pp + (high - low),
        'r3': high + 2 * (pp - low),
        's1': (2 * pp) - high,
        's2': pp - (high - low),
        's3': low - 2 * (high - pp)
    }


def get_session_high_low_times(df: pd.DataFrame) -> dict:
    """Get timestamps of session high and low."""
    if len(df) < 1:
        return {}
    
    high_idx = df['ltp'].idxmax()
    low_idx = df['ltp'].idxmin()
    
    return {
        'high_time': high_idx,
        'low_time': low_idx,
        'high_price': df['ltp'].max(),
        'low_price': df['ltp'].min(),
        'time_to_high': (high_idx - df.index[0]).total_seconds() / 60,
        'time_to_low': (low_idx - df.index[0]).total_seconds() / 60,
    }


def calculate_hourly_returns(df: pd.DataFrame) -> dict:
    """Calculate returns for each hour of the trading session."""
    if len(df) < 2:
        return {}
    
    # Group by hour
    df_hourly = df.resample('H').agg({
        'ltp': ['first', 'last', 'max', 'min']
    })
    
    returns = {}
    for hour, data in df_hourly.iterrows():
        hour_str = hour.strftime('%H:%M')
        open_price = data[('ltp', 'first')]
        close_price = data[('ltp', 'last')]
        
        if not np.isnan(open_price) and not np.isnan(close_price) and open_price != 0:
            returns[hour_str] = {
                'return_percent': ((close_price - open_price) / open_price) * 100,
                'high': data[('ltp', 'max')],
                'low': data[('ltp', 'min')],
                'range': data[('ltp', 'max')] - data[('ltp', 'min')],
            }
    
    return returns


def identify_volatile_hours(df: pd.DataFrame) -> list:
    """Identify most volatile hours of the session."""
    hourly_returns = calculate_hourly_returns(df)
    if not hourly_returns:
        return []
    
    volatile_hours = []
    for hour, data in hourly_returns.items():
        volatile_hours.append({
            'hour': hour,
            'volatility': abs(data['return_percent']),
            'range': data['range']
        })
    
    return sorted(volatile_hours, key=lambda x: x['volatility'], reverse=True)[:3]


def calculate_opening_range(df: pd.DataFrame, minutes: int = 30) -> dict:
    """Calculate opening range metrics."""
    if len(df) < 2:
        return {}
    
    # Get first N minutes of data
    start_time = df.index[0]
    end_time = start_time + pd.Timedelta(minutes=minutes)
    opening_data = df.loc[start_time:end_time]
    
    if len(opening_data) < 2:
        return {}
    
    return {
        'high': opening_data['ltp'].max(),
        'low': opening_data['ltp'].min(),
        'range': opening_data['ltp'].max() - opening_data['ltp'].min(),
        'breakout_upside': df['ltp'].max() > opening_data['ltp'].max(),
        'breakout_downside': df['ltp'].min() < opening_data['ltp'].min(),
        'opening_range_high_break_time': get_breakout_time(df, opening_data['ltp'].max(), 'above'),
        'opening_range_low_break_time': get_breakout_time(df, opening_data['ltp'].min(), 'below'),
    }


def calculate_closing_range(df: pd.DataFrame, minutes: int = 30) -> dict:
    """Calculate closing range metrics."""
    if len(df) < 2:
        return {}
    
    # Get last N minutes of data
    end_time = df.index[-1]
    start_time = end_time - pd.Timedelta(minutes=minutes)
    closing_data = df.loc[start_time:end_time]
    
    if len(closing_data) < 2:
        return {}
    
    return {
        'high': closing_data['ltp'].max(),
        'low': closing_data['ltp'].min(),
        'range': closing_data['ltp'].max() - closing_data['ltp'].min(),
        'close_relative_to_range': (
            (df['ltp'].iloc[-1] - closing_data['ltp'].min()) / 
            (closing_data['ltp'].max() - closing_data['ltp'].min()) * 100
            if (closing_data['ltp'].max() - closing_data['ltp'].min()) != 0 else 50
        ),
    }


def get_breakout_time(df: pd.DataFrame, level: float, direction: str) -> pd.Timestamp:
    """Get time when price breaks a level in specified direction."""
    if direction == 'above':
        breakout = df[df['ltp'] > level]
    else:  # below
        breakout = df[df['ltp'] < level]
    
    return breakout.index[0] if not breakout.empty else None

# ========================
# MOMENTUM INDICATOR FUNCTIONS
# ========================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0  # Neutral value
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_stochastic(ohlc_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Calculate Stochastic Oscillator."""
    if len(ohlc_data) < k_period + d_period:
        return {'k': 50.0, 'd': 50.0}
    
    low_min = ohlc_data['low'].rolling(window=k_period).min()
    high_max = ohlc_data['high'].rolling(window=k_period).max()
    
    k = 100 * ((ohlc_data['close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50.0,
        'd': d.iloc[-1] if not pd.isna(d.iloc[-1]) else 50.0,
        'overbought': k.iloc[-1] > 80 if not pd.isna(k.iloc[-1]) else False,
        'oversold': k.iloc[-1] < 20 if not pd.isna(k.iloc[-1]) else False
    }


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD indicator."""
    if len(prices) < slow + signal:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return {
        'macd': macd.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1],
        'bullish': histogram.iloc[-1] > 0
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        middle = prices.iloc[-1]
        return {
            'upper': middle,
            'middle': middle,
            'lower': middle,
            'bandwidth': 0,
            'percent_b': 0.5
        }
    
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    current_price = prices.iloc[-1]
    percent_b = ((current_price - lower.iloc[-1]) / 
                 (upper.iloc[-1] - lower.iloc[-1]) if 
                 (upper.iloc[-1] - lower.iloc[-1]) != 0 else 0.5)
    
    bandwidth = ((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100 
                if middle.iloc[-1] != 0 else 0)
    
    return {
        'upper': upper.iloc[-1],
        'middle': middle.iloc[-1],
        'lower': lower.iloc[-1],
        'bandwidth': bandwidth,
        'percent_b': percent_b,
        'squeeze': bandwidth < 10  # Bandwidth less than 10% indicates squeeze
    }


# ========================
# VOLUME ANALYSIS FUNCTIONS
# ========================

def analyze_volume_profile(df: pd.DataFrame, price_bins: int = 20) -> dict:
    """Analyze volume at different price levels."""
    if 'volume' not in df.columns or len(df) < 2:
        return {}
    
    # Create price bins
    price_min = df['ltp'].min()
    price_max = df['ltp'].max()
    bin_width = (price_max - price_min) / price_bins
    
    volume_profile = {}
    for i in range(price_bins):
        lower_bound = price_min + i * bin_width
        upper_bound = lower_bound + bin_width
        
        # Get volume in this price range
        mask = (df['ltp'] >= lower_bound) & (df['ltp'] < upper_bound)
        volume_in_range = df.loc[mask, 'volume'].sum()
        
        volume_profile[f'bin_{i+1}'] = {
            'price_range': f"{lower_bound:.2f}-{upper_bound:.2f}",
            'volume': volume_in_range,
            'volume_percent': (volume_in_range / df['volume'].sum() * 100) 
                            if df['volume'].sum() > 0 else 0
        }
    
    # Find point of control (price level with highest volume)
    max_volume_bin = max(volume_profile.items(), 
                        key=lambda x: x[1]['volume'])
    
    return {
        'volume_profile': volume_profile,
        'poc_price_range': max_volume_bin[1]['price_range'],
        'poc_volume': max_volume_bin[1]['volume'],
        'total_volume': df['volume'].sum(),
        'value_area': calculate_value_area(volume_profile, df['volume'].sum())
    }


def calculate_value_area(volume_profile: dict, total_volume: float) -> dict:
    """Calculate Value Area (70% of volume around POC)."""
    if not volume_profile or total_volume == 0:
        return {}
    
    # Sort price bins by volume
    sorted_bins = sorted(volume_profile.items(), 
                        key=lambda x: x[1]['volume'], 
                        reverse=True)
    
    # Find bins that contain 70% of total volume
    target_volume = total_volume * 0.7
    cumulative_volume = 0
    value_area_bins = []
    
    for bin_name, bin_data in sorted_bins:
        cumulative_volume += bin_data['volume']
        value_area_bins.append(bin_name)
        
        if cumulative_volume >= target_volume:
            break
    
    # Get price range of value area
    price_ranges = []
    for bin_name in value_area_bins:
        price_range = volume_profile[bin_name]['price_range']
        low, high = map(float, price_range.split('-'))
        price_ranges.append((low, high))
    
    if price_ranges:
        value_area_low = min(r[0] for r in price_ranges)
        value_area_high = max(r[1] for r in price_ranges)
        
        return {
            'low': value_area_low,
            'high': value_area_high,
            'range': value_area_high - value_area_low,
            'volume_percent': (cumulative_volume / total_volume * 100),
            'num_bins': len(value_area_bins)
        }
    
    return {}


def analyze_volume_trend(df: pd.DataFrame, window: int = 20) -> dict:
    """Analyze volume trend and its relation to price."""
    if 'volume' not in df.columns or len(df) < window:
        return {}
    
    # Calculate volume moving average
    volume_ma = df['volume'].rolling(window=window).mean()
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    # Volume trend (simple linear regression on volume)
    x = np.arange(len(df[-window:]))
    y = df['volume'].iloc[-window:].values
    
    if len(y) > 1:
        slope, _ = np.polyfit(x, y, 1)
        volume_trend = 'increasing' if slope > 0 else 'decreasing'
        trend_strength = abs(slope) / np.mean(y) if np.mean(y) > 0 else 0
    else:
        volume_trend = 'neutral'
        trend_strength = 0
    
    return {
        'current_volume': current_volume,
        'volume_ma': volume_ma.iloc[-1],
        'volume_ratio': volume_ratio,
        'volume_trend': volume_trend,
        'trend_strength': trend_strength,
        'high_volume': volume_ratio > 1.5,
        'low_volume': volume_ratio < 0.5
    }


def calculate_volume_price_correlation(df: pd.DataFrame, window: int = 20) -> dict:
    """Calculate correlation between volume and price movements."""
    if 'volume' not in df.columns or len(df) < window:
        return {}
    
    # Calculate returns and volume changes
    returns = df['ltp'].pct_change().dropna()
    volume_changes = df['volume'].pct_change().dropna()
    
    # Align the series
    common_index = returns.index.intersection(volume_changes.index)
    if len(common_index) < 10:
        return {}
    
    aligned_returns = returns.loc[common_index]
    aligned_volume = volume_changes.loc[common_index]
    
    # Calculate correlations
    correlation = aligned_returns.corr(aligned_volume)
    
    # Recent correlation (last N periods)
    recent_returns = aligned_returns.iloc[-window:]
    recent_volume = aligned_volume.iloc[-window:]
    
    recent_correlation = recent_returns.corr(recent_volume) if len(recent_returns) > 1 else 0
    
    return {
        'correlation': correlation if not pd.isna(correlation) else 0,
        'recent_correlation': recent_correlation if not pd.isna(recent_correlation) else 0,
        'correlation_strength': 'strong' if abs(correlation) > 0.7 else 
                               'moderate' if abs(correlation) > 0.3 else 'weak',
        'positive_divergence': correlation > 0.3,
        'negative_divergence': correlation < -0.3
    }


# ========================
# RISK METRICS FUNCTIONS
# ========================

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk (VaR) using historical simulation."""
    if len(returns) < 10:
        return 0
    
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    
    # Annualize if needed (assuming 6.5 hours per day, 252 days)
    var_annualized = var * np.sqrt(6.5 * 252)
    
    return {
        'daily_var_percent': var * 100,
        'daily_var_annualized': var_annualized * 100,
        'confidence_level': confidence_level
    }


def calculate_intraday_sharpe(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Calculate intraday Sharpe ratio."""
    if len(df) < 2:
        return 0
    
    # Calculate hourly returns (assuming data is roughly hourly)
    returns = df['ltp'].pct_change().dropna()
    
    if len(returns) < 2:
        return 0
    
    # Annualize returns and volatility
    # Assuming 6.5 trading hours per day, 252 trading days
    annualization_factor = np.sqrt(6.5 * 252)
    
    excess_returns = returns.mean() * (6.5 * 252) - (risk_free_rate / 100)
    volatility = returns.std() * annualization_factor
    
    if volatility == 0:
        return 0
    
    sharpe = excess_returns / volatility
    return sharpe


def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02, 
                          target_return: float = 0.0) -> float:
    """Calculate Sortino ratio (downside risk-adjusted return)."""
    if len(df) < 2:
        return 0
    
    returns = df['ltp'].pct_change().dropna()
    
    if len(returns) < 2:
        return 0
    
    # Calculate downside deviation
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        downside_dev = 0
    else:
        downside_dev = downside_returns.std()
    
    # Annualize
    annualization_factor = np.sqrt(6.5 * 252)
    excess_return = returns.mean() * (6.5 * 252) - (risk_free_rate / 100)
    
    if downside_dev == 0:
        return 0
    
    sortino = excess_return / (downside_dev * annualization_factor)
    return sortino


def calculate_calmar_ratio(df: pd.DataFrame, period_years: float = 1) -> float:
    """Calculate Calmar ratio (return vs max drawdown)."""
    if len(df) < 20:
        return 0
    
    # Calculate annualized return
    total_return = (df['ltp'].iloc[-1] / df['ltp'].iloc[0]) - 1
    
    # Assuming intraday data, scale to annual
    # Calculate hours elapsed
    hours_elapsed = (df.index[-1] - df.index[0]).total_seconds() / 3600
    if hours_elapsed == 0:
        return 0
    
    annualized_return = ((1 + total_return) ** (6.5 * 252 / hours_elapsed)) - 1
    
    # Get max drawdown
    drawdown_info = calculate_max_drawdown(df['ltp'])
    max_drawdown = abs(drawdown_info['drawdown_percent']) / 100
    
    if max_drawdown == 0:
        return 0
    
    calmar = annualized_return / max_drawdown
    return calmar


# ========================
# SUMMARY METRICS FUNCTIONS
# ========================

def classify_market_regime(analysis: dict) -> str:
    """Classify the current market regime based on multiple factors."""
    if not analysis:
        return 'unknown'
    
    try:
        # Extract key metrics
        volatility = analysis.get('volatility', {}).get('realized_volatility', 0)
        trend_strength = analysis.get('trend', {}).get('trend_strength', 0)
        rsi = analysis.get('momentum', {}).get('rsi', 50) if 'momentum' in analysis else 50
        atr = analysis.get('volatility', {}).get('avg_true_range', 0)
        price_range_pct = analysis.get('basic', {}).get('range_percent', 0)
        
        # Classify based on rules
        if volatility > 20:  # High volatility
            if trend_strength > 0.7:
                return 'trending_high_vol'
            else:
                return 'ranging_high_vol'
        elif volatility < 10:  # Low volatility
            if price_range_pct < 0.5:
                return 'consolidation_low_vol'
            else:
                return 'breakout_forming'
        else:  # Medium volatility
            if trend_strength > 0.6:
                if rsi > 70:
                    return 'overbought_trending'
                elif rsi < 30:
                    return 'oversold_trending'
                else:
                    return 'healthy_trend'
            else:
                if 40 < rsi < 60:
                    return 'balanced_range'
                else:
                    return 'range_extreme'
                    
    except Exception as e:
        return f'classification_error: {str(e)}'


def calculate_opportunity_score(analysis: dict) -> float:
    """Calculate a score from 0-100 indicating trading opportunity quality."""
    if not analysis:
        return 50.0
    
    try:
        score = 50.0  # Start at neutral
        
        # Trend strength contributes up to 20 points
        trend_strength = analysis.get('trend', {}).get('trend_strength', 0)
        score += trend_strength * 20
        
        # Volatility contributes up to 20 points (optimal range: 10-20%)
        volatility = analysis.get('volatility', {}).get('realized_volatility', 0)
        if 10 <= volatility <= 20:
            score += 20
        elif 5 <= volatility < 10 or 20 < volatility <= 25:
            score += 10
        elif volatility > 25 or volatility < 5:
            score -= 10
        
        # RSI position contributes up to 10 points (avoid extremes)
        rsi = analysis.get('momentum', {}).get('rsi', 50) if 'momentum' in analysis else 50
        if 30 <= rsi <= 70:
            score += 10
        elif 20 <= rsi < 30 or 70 < rsi <= 80:
            score += 5
        else:
            score -= 10
        
        # Volume confirmation contributes up to 10 points
        volume_ratio = analysis.get('volume', {}).get('volume_ratio', 1) if 'volume' in analysis else 1
        if 0.8 <= volume_ratio <= 1.2:
            score += 5
        elif volume_ratio > 1.5:  # High volume confirmation
            score += 10
        
        # Clear support/resistance contributes up to 10 points
        supports = analysis.get('structure', {}).get('support_levels', [])
        resistances = analysis.get('structure', {}).get('resistance_levels', [])
        if len(supports) > 0 and len(resistances) > 0:
            score += 10
        
        # Current price relative to range contributes up to 10 points
        # (Middle of range is better for mean reversion, edges better for breakout)
        # This depends on strategy - default to neutral
        
        # Clamp score between 0 and 100
        score = max(0, min(100, score))
        
        return round(score, 1)
        
    except Exception as e:
        print(f"Error calculating opportunity score: {e}")
        return 50.0


def generate_key_takeaways(analysis: dict) -> list:
    """Generate key insights from the analysis."""
    takeaways = []
    
    if not analysis:
        return ["Insufficient data for analysis"]
    
    try:
        # Trend insights
        trend = analysis.get('trend', {})
        if trend.get('trend_direction') in ['strong_uptrend', 'strong_downtrend']:
            strength = trend.get('trend_strength', 0)
            if strength > 0.7:
                takeaways.append(f"Strong {trend['trend_direction'].replace('_', ' ')} in place")
        
        # Volatility insights
        vol = analysis.get('volatility', {})
        realized_vol = vol.get('realized_volatility', 0)
        if realized_vol > 25:
            takeaways.append("Very high volatility - consider smaller position sizes")
        elif realized_vol < 8:
            takeaways.append("Low volatility - breakout potential building")
        
        # Momentum insights
        if 'momentum' in analysis:
            mom = analysis['momentum']
            rsi = mom.get('rsi', 50)
            if rsi > 80:
                takeaways.append("RSI indicates severely overbought conditions")
            elif rsi < 20:
                takeaways.append("RSI indicates severely oversold conditions")
            
            bb = mom.get('bollinger_bands', {})
            if bb.get('squeeze', False):
                takeaways.append("Bollinger Bands squeezing - volatility expansion likely")
        
        # Volume insights
        if 'volume' in analysis:
            vol_data = analysis['volume']
            if vol_data.get('high_volume', False):
                takeaways.append("High volume activity - institutional participation evident")
        
        # Structure insights
        struct = analysis.get('structure', {})
        supports = struct.get('support_levels', [])
        resistances = struct.get('resistance_levels', [])
        
        if len(supports) > 0 and len(resistances) > 0:
            current_price = analysis.get('basic', {}).get('close', 0)
            nearest_support = min(supports, key=lambda x: abs(x['price'] - current_price))
            nearest_resistance = min(resistances, key=lambda x: abs(x['price'] - current_price))
            
            dist_to_support = abs(current_price - nearest_support['price'])
            dist_to_resistance = abs(current_price - nearest_resistance['price'])
            
            if dist_to_support < dist_to_resistance:
                takeaways.append(f"Price closer to support ({dist_to_support:.1f} points away)")
            else:
                takeaways.append(f"Price closer to resistance ({dist_to_resistance:.1f} points away)")
        
        # If no specific takeaways, add general ones
        if not takeaways:
            regime = analysis.get('summary', {}).get('market_regime', 'unknown')
            takeaways.append(f"Market in {regime.replace('_', ' ')} regime")
            score = analysis.get('summary', {}).get('trading_opportunity_score', 50)
            if score > 70:
                takeaways.append("Good trading conditions based on multiple factors")
            elif score < 30:
                takeaways.append("Poor trading conditions - consider staying sidelines")
        
        return takeaways[:5]  # Return top 5 takeaways
        
    except Exception as e:
        return [f"Analysis error: {str(e)}"]


def calculate_overall_score(analysis: dict) -> dict:
    """Calculate overall market condition score with breakdown."""
    if not analysis:
        return {'total': 50, 'breakdown': {}}
    
    try:
        breakdown = {}
        total_score = 0
        max_possible = 0
        
        # 1. Trend Quality (0-25 points)
        trend = analysis.get('trend', {})
        trend_score = 0
        trend_strength = trend.get('trend_strength', 0)
        
        if trend_strength > 0.7:
            trend_score = 25
        elif trend_strength > 0.5:
            trend_score = 15
        elif trend_strength > 0.3:
            trend_score = 8
        else:
            trend_score = 3
        
        breakdown['trend'] = trend_score
        total_score += trend_score
        max_possible += 25
        
        # 2. Volatility Quality (0-25 points)
        vol = analysis.get('volatility', {})
        vol_score = 0
        realized_vol = vol.get('realized_volatility', 0)
        
        if 12 <= realized_vol <= 18:
            vol_score = 25
        elif 8 <= realized_vol < 12 or 18 < realized_vol <= 22:
            vol_score = 15
        elif 5 <= realized_vol < 8 or 22 < realized_vol <= 30:
            vol_score = 8
        else:
            vol_score = 3
        
        breakdown['volatility'] = vol_score
        total_score += vol_score
        max_possible += 25
        
        # 3. Momentum Quality (0-20 points)
        mom_score = 0
        if 'momentum' in analysis:
            momentum = analysis['momentum']
            rsi = momentum.get('rsi', 50)
            
            if 40 <= rsi <= 60:
                mom_score = 20
            elif 30 <= rsi < 40 or 60 < rsi <= 70:
                mom_score = 12
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                mom_score = 6
            else:
                mom_score = 2
        else:
            mom_score = 10  # Neutral if no momentum data
        
        breakdown['momentum'] = mom_score
        total_score += mom_score
        max_possible += 20
        
        # 4. Structure Quality (0-15 points)
        struct = analysis.get('structure', {})
        struct_score = 0
        supports = len(struct.get('support_levels', []))
        resistances = len(struct.get('resistance_levels', []))
        
        if supports >= 2 and resistances >= 2:
            struct_score = 15
        elif supports >= 1 and resistances >= 1:
            struct_score = 10
        elif supports >= 1 or resistances >= 1:
            struct_score = 5
        else:
            struct_score = 2
        
        breakdown['structure'] = struct_score
        total_score += struct_score
        max_possible += 15
        
        # 5. Volume Quality (0-15 points)
        vol_quality_score = 0
        if 'volume' in analysis:
            volume_data = analysis['volume']
            volume_ratio = volume_data.get('volume_ratio', 1)
            
            if 0.8 <= volume_ratio <= 1.2:
                vol_quality_score = 15
            elif 0.5 <= volume_ratio < 0.8 or 1.2 < volume_ratio <= 1.5:
                vol_quality_score = 8
            elif volume_ratio > 1.5:
                vol_quality_score = 12  # High volume can be good
            else:
                vol_quality_score = 3
        else:
            vol_quality_score = 7  # Neutral if no volume data
        
        breakdown['volume'] = vol_quality_score
        total_score += vol_quality_score
        max_possible += 15
        
        # Calculate final score (0-100 scale)
        if max_possible > 0:
            final_score = (total_score / max_possible) * 100
        else:
            final_score = 50
        
        return {
            'total': round(final_score, 1),
            'breakdown': breakdown,
            'interpretation': interpret_score(final_score)
        }
        
    except Exception as e:
        print(f"Error calculating overall score: {e}")
        return {'total': 50, 'breakdown': {}, 'interpretation': 'error'}


def interpret_score(score: float) -> str:
    """Interpret the overall score."""
    if score >= 80:
        return "Excellent trading conditions"
    elif score >= 65:
        return "Good trading conditions"
    elif score >= 50:
        return "Average trading conditions"
    elif score >= 35:
        return "Below average conditions"
    else:
        return "Poor trading conditions"


def analyze_intraday_index(index_data: pd.DataFrame, tick_interval: str = '1min') -> dict:
    """
    Comprehensive analysis of intraday index tick data.
    
    Args:
        index_data: DataFrame with 'timestamp' and 'ltp' columns
        tick_interval: Resampling interval for some calculations (default: '1min')
        
    Returns:
        Dictionary with comprehensive index analysis metrics
    """
    if len(index_data) < 2:
        return {}
    
    df = index_data.copy()
    df.set_index('timestamp', inplace=True)
    
    # Resample for certain calculations if needed
    if tick_interval:
        resampled = df['ltp'].resample(tick_interval).ohlc()
    else:
        resampled = None
    
    analysis = {}
    
    # ========================
    # 1. BASIC PRICE METRICS
    # ========================
    analysis['basic'] = {
        'open': df['ltp'].iloc[0],
        'high': df['ltp'].max(),
        'low': df['ltp'].min(),
        'close': df['ltp'].iloc[-1],
        'change_points': df['ltp'].iloc[-1] - df['ltp'].iloc[0],
        'change_percent': ((df['ltp'].iloc[-1] - df['ltp'].iloc[0]) / df['ltp'].iloc[0]) * 100,
        'range_points': df['ltp'].max() - df['ltp'].min(),
        'range_percent': ((df['ltp'].max() - df['ltp'].min()) / df['ltp'].iloc[0]) * 100,
        'average_price': df['ltp'].mean(),
        'median_price': df['ltp'].median(),
        'vwap': calculate_vwap(df) if 'volume' in df.columns else None,
    }
    
    # ========================
    # 2. VOLATILITY METRICS
    # ========================
    analysis['volatility'] = {
        'std_deviation': df['ltp'].std(),
        'avg_true_range': calculate_atr(df) if resampled is not None else None,
        'realized_volatility': calculate_realized_volatility(df),
        'max_drawdown': calculate_max_drawdown(df['ltp']),
        'max_runup': calculate_max_runup(df['ltp']),
        'volatility_ratio': calculate_volatility_ratio(df),
    }
    
    # ========================
    # 3. TREND ANALYSIS
    # ========================
    analysis['trend'] = {
        'trend_direction': identify_trend_direction(df),
        'trend_strength': calculate_trend_strength(df),
        'intraday_momentum': calculate_intraday_momentum(df),
        'price_velocity': calculate_price_velocity(df),
        'session_trend': analyze_session_trend(df),
    }
    
    # ========================
    # 4. PRICE DISTRIBUTION
    # ========================
    analysis['distribution'] = {
        'price_skewness': df['ltp'].skew(),
        'price_kurtosis': df['ltp'].kurtosis(),
        'time_in_range': calculate_time_in_range(df),
        'most_common_price_zones': identify_price_zones(df),
    }
    
    # ========================
    # 5. MARKET STRUCTURE
    # ========================
    analysis['structure'] = {
        'support_levels': identify_support_levels(df),
        'resistance_levels': identify_resistance_levels(df),
        'pivot_points': calculate_pivot_points(df),
        'session_high_low_time': get_session_high_low_times(df),
    }
    
    # ========================
    # 6. TIME-BASED ANALYSIS
    # ========================
    analysis['time_analysis'] = {
        'hourly_returns': calculate_hourly_returns(df),
        'most_volatile_hour': identify_volatile_hours(df),
        'opening_range': calculate_opening_range(df),
        'closing_range': calculate_closing_range(df),
    }
    
    # ========================
    # 7. MOMENTUM INDICATORS
    # ========================
    if resampled is not None:
        analysis['momentum'] = {
            'rsi': calculate_rsi(resampled['close']),
            'stochastic': calculate_stochastic(resampled),
            'macd': calculate_macd(resampled['close']),
            'bollinger_bands': calculate_bollinger_bands(resampled['close']),
        }
    
    # ========================
    # 8. VOLUME ANALYSIS (if available)
    # ========================
    if 'volume' in df.columns:
        analysis['volume'] = {
            'total_volume': df['volume'].sum(),
            'avg_volume': df['volume'].mean(),
            'volume_profile': analyze_volume_profile(df),
            'volume_trend': analyze_volume_trend(df),
            'volume_price_correlation': calculate_volume_price_correlation(df),
        }
    
    # ========================
    # 9. RISK METRICS
    # ========================
    analysis['risk'] = {
        'value_at_risk': calculate_var(df['ltp'].pct_change().dropna()),
        'sharpe_ratio_intraday': calculate_intraday_sharpe(df),
        'sortino_ratio': calculate_sortino_ratio(df),
        'calmar_ratio': calculate_calmar_ratio(df),
    }
    
    # ========================
    # 10. SUMMARY METRICS
    # ========================
    analysis['summary'] = {
        'market_regime': classify_market_regime(analysis),
        'trading_opportunity_score': calculate_opportunity_score(analysis),
        'key_takeaways': generate_key_takeaways(analysis),
        'overall_score': calculate_overall_score(analysis),
    }
    
    return analysis



def analyze_trades(trade_df,lot_size, initial_capital=25000):
    """
    Analyze individual trades from the trade DataFrame
    
    Parameters:
    -----------
    trade_df : pandas DataFrame
        DataFrame with trade details including:
        entry_idx, entry_price, entry_time, exit_idx, exit_price, exit_time,
        pnl, charges, exit_reason, option_type
    initial_capital : float
        Initial capital for calculating position sizing metrics
    
    Returns:
    --------
    dict: Dictionary containing comprehensive trade analysis
    """
    
    # Create a copy to avoid modifying original
    df = trade_df.copy()
    
    # Ensure datetime columns are proper datetime objects
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    analysis_results = {}
    
    print("="*80)
    print("TRADE-BY-TRADE ANALYSIS")
    print("="*80)
    
    # 1. BASIC TRADE METRICS
    print("\n1. BASIC TRADE METRICS:")
    print("-"*40)
    
    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]
    breakeven_trades = df[df['pnl'] == 0]
    
    win_rate = len(winning_trades) / total_trades * 100
    loss_rate = len(losing_trades) / total_trades * 100
    
    total_pnl = df['pnl'].sum()
    total_charges = df['charges'].sum()
    net_pnl = total_pnl 
    
    avg_pnl = df['pnl'].mean()
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({loss_rate:.1f}%)")
    print(f"Breakeven Trades: {len(breakeven_trades)}")
    print(f"Total P&L: ₹{total_pnl:.2f}")
    print(f"Total Charges: ₹{total_charges:.2f}")
    print(f"Net P&L: ₹{net_pnl:.2f}")
    print(f"Average P&L per Trade: ₹{avg_pnl:.2f}")
    print(f"Average Winning Trade: ₹{avg_win:.2f}")
    print(f"Average Losing Trade: ₹{avg_loss:.2f}")
    print(f"Profit Factor: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Profit Factor: N/A")
    
    # 2. TRADE DURATION ANALYSIS
    # 2. TRADE DURATION ANALYSIS (only if we have time columns)
    print("\n2. TRADE DURATION ANALYSIS:")
    print("-"*40)
    
    if 'entry_time' in df.columns and 'exit_time' in df.columns:
        df['trade_duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60  # in minutes
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        avg_duration = df['trade_duration'].mean()
        max_duration = df['trade_duration'].max()
        min_duration = df['trade_duration'].min()
        
        # Duration by win/loss
        win_duration = winning_trades['trade_duration'].mean() if len(winning_trades) > 0 else 0
        loss_duration = losing_trades['trade_duration'].mean() if len(losing_trades) > 0 else 0
        
        print(f"Average Trade Duration: {avg_duration:.1f} minutes")
        print(f"Longest Trade: {max_duration:.1f} minutes")
        print(f"Shortest Trade: {min_duration:.1f} minutes")
        print(f"Winning Trades Duration: {win_duration:.1f} minutes")
        print(f"Losing Trades Duration: {loss_duration:.1f} minutes")
    else:
        print("Trade duration not available - missing entry_time or exit_time columns")
        avg_duration = win_duration = loss_duration = 0
    
    # 3. OPTION TYPE ANALYSIS
    print("\n3. OPTION TYPE ANALYSIS:")
    print("-"*40)
    
    ce_trades = df[df['option_type'] == 'CE']
    pe_trades = df[df['option_type'] == 'PE']
    
    ce_win_rate = len(ce_trades[ce_trades['pnl'] > 0]) / len(ce_trades) * 100 if len(ce_trades) > 0 else 0
    pe_win_rate = len(pe_trades[pe_trades['pnl'] > 0]) / len(pe_trades) * 100 if len(pe_trades) > 0 else 0
    
    ce_pnl = ce_trades['pnl'].sum() if len(ce_trades) > 0 else 0
    pe_pnl = pe_trades['pnl'].sum() if len(pe_trades) > 0 else 0
    
    print(f"CE Trades: {len(ce_trades)} ({len(ce_trades)/total_trades*100:.1f}%)")
    print(f"PE Trades: {len(pe_trades)} ({len(pe_trades)/total_trades*100:.1f}%)")
    print(f"CE Win Rate: {ce_win_rate:.1f}%")
    print(f"PE Win Rate: {pe_win_rate:.1f}%")
    print(f"CE Total P&L: ₹{ce_pnl:.2f}")
    print(f"PE Total P&L: ₹{pe_pnl:.2f}")
    
    # 4. ENTRY/EXIT PRICE ANALYSIS
    print("\n4. ENTRY/EXIT PRICE ANALYSIS:")
    print("-"*40)
    
    df['price_change'] = df['exit_price'] - df['entry_price']
    df['price_change_pct'] = (df['price_change'] / df['entry_price']) * 100
    
    avg_price_change = df['price_change'].mean()
    avg_price_change_pct = df['price_change_pct'].mean()
    
    print(f"Average Price Change: ₹{avg_price_change:.2f}")
    print(f"Average Price Change %: {avg_price_change_pct:.2f}%")
    print(f"Maximum Price Increase: ₹{df['price_change'].max():.2f}")
    print(f"Maximum Price Decrease: ₹{df['price_change'].min():.2f}")
    
    # 5. EXIT REASON ANALYSIS
    print("\n5. EXIT REASON ANALYSIS:")
    print("-"*40)
    
    exit_reasons = df['exit_reason'].value_counts()
    print("Exit Reasons:")
    for reason, count in exit_reasons.items():
        pnl_for_reason = df[df['exit_reason'] == reason]['pnl'].sum()
        win_rate_reason = len(df[(df['exit_reason'] == reason) & (df['pnl'] > 0)]) / count * 100
        print(f"  {reason}: {count} trades, P&L: ₹{pnl_for_reason:.2f}, Win Rate: {win_rate_reason:.1f}%")
    
    # 6. TIME OF DAY ANALYSIS
    print("\n6. TIME OF DAY ANALYSIS:")
    print("-"*40)
    
    df['entry_hour'] = df['entry_time'].dt.hour
    df['exit_hour'] = df['exit_time'].dt.hour
    
    # Analyze by entry hour
    hourly_stats = []
    for hour in range(9, 16):  # Market hours typically 9 AM to 3:30 PM
        hour_trades = df[df['entry_hour'] == hour]
        if len(hour_trades) > 0:
            hour_pnl = hour_trades['pnl'].sum()
            hour_win_rate = len(hour_trades[hour_trades['pnl'] > 0]) / len(hour_trades) * 100
            hourly_stats.append((hour, len(hour_trades), hour_pnl, hour_win_rate))
    
    print("Performance by Entry Hour:")
    for hour, count, pnl, win_rate in sorted(hourly_stats, key=lambda x: x[2], reverse=True):
        print(f"  {hour}:00 - {count} trades, P&L: ₹{pnl:.2f}, Win Rate: {win_rate:.1f}%")
        
    # 6B. WEEKDAY ANALYSIS
    print("\n6B. DAY OF WEEK ANALYSIS:")
    print("-"*40)

    # Monday=0 ... Sunday=6
    df['weekday'] = df['entry_time'].dt.day_name()

    weekday_stats = (
        df.groupby('weekday')
          .agg(
              trades=('pnl', 'count'),
              total_pnl=('pnl', 'sum'),
              win_rate=('pnl', lambda x: (x > 0).mean() * 100)
          )
          .sort_values('total_pnl', ascending=False)
    )

    print(weekday_stats)

    best_day = weekday_stats['total_pnl'].idxmax()
    worst_day = weekday_stats['total_pnl'].idxmin()

    print(f"\nBest Day: {best_day}  → PnL: ₹{weekday_stats.loc[best_day, 'total_pnl']:.2f}")
    print(f"Worst Day: {worst_day} → PnL: ₹{weekday_stats.loc[worst_day, 'total_pnl']:.2f}")

    # Add to final output dictionary
    analysis_results['weekday_analysis'] = weekday_stats.to_dict()
    
    
    # 7. RISK-REWARD ANALYSIS
    print("\n7. RISK-REWARD ANALYSIS:")
    print("-"*40)
    
    # Calculate hypothetical risk (entry to lowest point) and reward (entry to highest point)
    # Note: This is simplified - actual risk/reward requires OHLC data
    if 'stop_price' in df.columns and df['stop_price'].notna().any():
        stops = df['stop_price'].dropna()
        avg_stop_distance = (df['entry_price'] - stops).abs().mean()
        print(f"Average Stop Distance: ₹{avg_stop_distance:.2f}")
    
    # 8. CONSECUTIVE WINS/LOSSES
    print("\n8. CONSECUTIVE WINS/LOSSES:")
    print("-"*40)
    
    df['is_win'] = df['pnl'] > 0
    df['is_loss'] = df['pnl'] < 0
    
    # Find streaks
    win_streaks = []
    loss_streaks = []
    current_streak = 0
    current_type = None
    
    for i, is_win in enumerate(df['is_win']):
        if is_win:
            if current_type == 'win':
                current_streak += 1
            else:
                if current_type == 'loss' and current_streak > 0:
                    loss_streaks.append(current_streak)
                current_streak = 1
                current_type = 'win'
        else:
            if current_type == 'loss':
                current_streak += 1
            else:
                if current_type == 'win' and current_streak > 0:
                    win_streaks.append(current_streak)
                current_streak = 1
                current_type = 'loss'
    
    # Add final streak
    if current_streak > 0:
        if current_type == 'win':
            win_streaks.append(current_streak)
        else:
            loss_streaks.append(current_streak)
    
    print(f"Longest Winning Streak: {max(win_streaks) if win_streaks else 0}")
    print(f"Longest Losing Streak: {max(loss_streaks) if loss_streaks else 0}")
    print(f"Average Winning Streak: {np.mean(win_streaks) if win_streaks else 0:.1f}")
    print(f"Average Losing Streak: {np.mean(loss_streaks) if loss_streaks else 0:.1f}")
    
    # 9. POSITION SIZING ANALYSIS
    print("\n9. POSITION SIZING ANALYSIS:")
    print("-"*40)
    
    # Calculate position size based on entry price
    df['position_size'] = df['entry_price'] * lot_size  # Assuming 1 lot = 50 shares
    avg_position_size = df['position_size'].mean()
    max_position_size = df['position_size'].max()
    min_position_size = df['position_size'].min()
    
    print(f"Average Position Size: ₹{avg_position_size:.2f}")
    print(f"Maximum Position Size: ₹{max_position_size:.2f}")
    print(f"Minimum Position Size: ₹{min_position_size:.2f}")
    print(f"Position Size as % of Capital: {(avg_position_size/initial_capital)*100:.1f}%")
    
    # 10. DISTRIBUTION ANALYSIS
    print("\n10. P&L DISTRIBUTION ANALYSIS:")
    print("-"*40)
    
    pnl_std = df['pnl'].std()
    pnl_skew = df['pnl'].skew()
    pnl_kurtosis = df['pnl'].kurtosis()
    
    print(f"P&L Standard Deviation: ₹{pnl_std:.2f}")
    print(f"P&L Skewness: {pnl_skew:.3f}")
    print(f"P&L Kurtosis: {pnl_kurtosis:.3f}")
    
    # Skewness interpretation
    if pnl_skew > 1:
        print("  → Right-skewed: More small losses, few large wins")
    elif pnl_skew < -1:
        print("  → Left-skewed: More small wins, few large losses")
    else:
        print("  → Fairly symmetric distribution")
    
    # 11. WORST/BEST TRADES
    print("\n11. EXTREME TRADES ANALYSIS:")
    print("-"*40)
    
    worst_trades = df.nsmallest(3, 'pnl')
    best_trades = df.nlargest(3, 'pnl')
    
    print("3 Worst Trades:")
    for idx, trade in worst_trades.iterrows():
        print(f"  Trade: P&L ₹{trade['pnl']:.2f}, Option: {trade['option_type']}, "
              f"Duration: {trade.get('trade_duration', 0):.1f} min, "
              f"Exit: {trade['exit_reason']}")
    
    print("\n3 Best Trades:")
    for idx, trade in best_trades.iterrows():
        print(f"  Trade: P&L ₹{trade['pnl']:.2f}, Option: {trade['option_type']}, "
              f"Duration: {trade.get('trade_duration', 0):.1f} min, "
              f"Exit: {trade['exit_reason']}")
    
    # 12. RECOMMENDATIONS
    print("\n" + "="*80)
    print("KEY INSIGHTS & RECOMMENDATIONS:")
    print("="*80)
    
    # Generate actionable insights
    if win_rate > 60 and avg_win > abs(avg_loss):
        print("✓ Strong strategy: High win rate with good risk-reward ratio")
    elif win_rate < 40 and avg_win > abs(avg_loss) * 2:
        print("✓ Low win rate but high risk-reward - acceptable if psychology can handle it")
    else:
        print("⚠ Review strategy: Consider improving win rate or risk-reward ratio")
    
    if ce_win_rate > pe_win_rate + 10:
        print("✓ CE trades significantly better - consider focusing on CE opportunities")
    elif pe_win_rate > ce_win_rate + 10:
        print("✓ PE trades significantly better - consider focusing on PE opportunities")
    
    if win_duration > loss_duration * 1.5:
        print("⚠ Winning trades take longer - consider taking profits earlier")
    elif loss_duration > win_duration * 1.5:
        print("⚠ Losing trades take longer - consider cutting losses faster")
    
    # Store all results in dictionary
    analysis_results = {
        'summary': {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'net_pnl': net_pnl,
            'avg_pnl': avg_pnl,
            'profit_factor': abs(avg_win/avg_loss) if avg_loss != 0 else 0
        },
        'by_option_type': {
            'ce': {
                'count': len(ce_trades),
                'pnl': ce_pnl,
                'win_rate': ce_win_rate
            },
            'pe': {
                'count': len(pe_trades),
                'pnl': pe_pnl,
                'win_rate': pe_win_rate
            }
        },
        'time_analysis': {
            'avg_duration': avg_duration,
            'win_duration': win_duration,
            'loss_duration': loss_duration
        },
        'distribution': {
            'std': pnl_std,
            'skew': pnl_skew,
            'kurtosis': pnl_kurtosis
        }
    }
    
    return analysis_results







def analyze_strategy(df, initial_capital=50000):
    """
    Analyze the trading strategy performance
    
    Parameters:
    -----------
    df : pandas DataFrame
        Daily strategy report with columns: date, ce_net, pe_net, net, index_change, 
        index_pct, index_range, index_volatility, ce_change, pe_change, ce_pct, pe_pct, ce_trades, pe_trades
    initial_capital : float
        Initial capital for calculating cumulative returns
    """
    
    # Create a copy to avoid modifying original
    analysis_df = df.copy()
    
    # Ensure date is datetime
    analysis_df['date'] = pd.to_datetime(analysis_df['date'], format='%d%B%y')
    
    print("="*80)
    print("STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    # 1. Basic Performance Metrics
    print("\n1. BASIC PERFORMANCE METRICS:")
    print("-"*40)
    
    total_pnl = analysis_df['net'].sum()
    total_ce_pnl = analysis_df['ce_net'].sum()
    total_pe_pnl = analysis_df['pe_net'].sum()
    total_trades = analysis_df['ce_trades'].sum() + analysis_df['pe_trades'].sum()
    
    print(f"Total P&L: ₹{total_pnl:.2f}")
    print(f"CE P&L Contribution: ₹{total_ce_pnl:.2f} ({total_ce_pnl/total_pnl*100:.1f}%)")
    print(f"PE P&L Contribution: ₹{total_pe_pnl:.2f} ({total_pe_pnl/total_pnl*100:.1f}%)")
    print(f"Total Trades: {total_trades}")
    print(f"Average Daily P&L: ₹{analysis_df['net'].mean():.2f}")
    print(f"Daily P&L Std Dev: ₹{analysis_df['net'].std():.2f}")
    
    # 2. Win/Loss Analysis
    print("\n2. WIN/LOSS ANALYSIS:")
    print("-"*40)
    
    winning_days = (analysis_df['net'] > 0).sum()
    losing_days = (analysis_df['net'] < 0).sum()
    flat_days = (analysis_df['net'] == 0).sum()
    
    win_rate = winning_days / len(analysis_df) * 100
    avg_win = analysis_df.loc[analysis_df['net'] > 0, 'net'].mean() if winning_days > 0 else 0
    avg_loss = analysis_df.loc[analysis_df['net'] < 0, 'net'].mean() if losing_days > 0 else 0
    
    print(f"Winning Days: {winning_days} ({win_rate:.1f}%)")
    print(f"Losing Days: {losing_days}")
    print(f"Flat Days: {flat_days}")
    print(f"Average Win: ₹{avg_win:.2f}")
    print(f"Average Loss: ₹{avg_loss:.2f}")
    print(f"Profit Factor: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Profit Factor: N/A")
    
    # 3. Index vs Strategy Correlation Analysis
    print("\n3. INDEX VS STRATEGY CORRELATION:")
    print("-"*40)
    
    # Calculate correlations
    correlation_index_pnl = analysis_df['index_change'].corr(analysis_df['net'])
    correlation_index_pct_pnl = analysis_df['index_pct'].corr(analysis_df['net'])
    
    print(f"Correlation (Index Change vs P&L): {correlation_index_pnl:.4f}")
    print(f"Correlation (Index % Change vs P&L): {correlation_index_pct_pnl:.4f}")
    
    # Analyze when strategy works best
    print("\n4. STRATEGY PERFORMANCE BY MARKET CONDITIONS:")
    print("-"*40)
    
    # Analyze by index movement direction
    up_market_days = analysis_df[analysis_df['index_change'] > 0]
    down_market_days = analysis_df[analysis_df['index_change'] < 0]
    
    if len(up_market_days) > 0:
        print(f"UP Market Days ({len(up_market_days)} days):")
        print(f"  Avg P&L: ₹{up_market_days['net'].mean():.2f}")
        print(f"  Win Rate: {(up_market_days['net'] > 0).sum()/len(up_market_days)*100:.1f}%")
    
    if len(down_market_days) > 0:
        print(f"DOWN Market Days ({len(down_market_days)} days):")
        print(f"  Avg P&L: ₹{down_market_days['net'].mean():.2f}")
        print(f"  Win Rate: {(down_market_days['net'] > 0).sum()/len(down_market_days)*100:.1f}%")
    
    # Analyze by volatility
    median_vol = analysis_df['index_volatility'].median()
    high_vol_days = analysis_df[analysis_df['index_volatility'] > median_vol]
    low_vol_days = analysis_df[analysis_df['index_volatility'] <= median_vol]
    
    print(f"\nHigh Volatility Days (> {median_vol:.2f}):")
    print(f"  Avg P&L: ₹{high_vol_days['net'].mean():.2f}")
    print(f"  Days: {len(high_vol_days)}")
    
    print(f"Low Volatility Days (≤ {median_vol:.2f}):")
    print(f"  Avg P&L: ₹{low_vol_days['net'].mean():.2f}")
    print(f"  Days: {len(low_vol_days)}")
    
    # 5. Trade Efficiency Analysis
    print("\n5. TRADE EFFICIENCY:")
    print("-"*40)
    
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_ce_pnl_per_trade = total_ce_pnl / analysis_df['ce_trades'].sum() if analysis_df['ce_trades'].sum() > 0 else 0
    avg_pe_pnl_per_trade = total_pe_pnl / analysis_df['pe_trades'].sum() if analysis_df['pe_trades'].sum() > 0 else 0
    
    print(f"Avg P&L per Trade: ₹{avg_pnl_per_trade:.2f}")
    print(f"Avg CE P&L per Trade: ₹{avg_ce_pnl_per_trade:.2f}")
    print(f"Avg PE P&L per Trade: ₹{avg_pe_pnl_per_trade:.2f}")
    
    # 6. Risk Metrics
    print("\n6. RISK METRICS:")
    print("-"*40)
    
    # Calculate daily returns based on initial capital
    daily_returns = analysis_df['net'] / initial_capital
    
    # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Volatility (daily): {daily_returns.std()*100:.4f}%")
    
    # 7. CE vs PE Analysis
    print("\n7. CE vs PE COMPARISON:")
    print("-"*40)
    
    ce_win_rate = (analysis_df['ce_net'] > 0).sum() / len(analysis_df) * 100
    pe_win_rate = (analysis_df['pe_net'] > 0).sum() / len(analysis_df) * 100
    
    print(f"CE Win Rate: {ce_win_rate:.1f}%")
    print(f"PE Win Rate: {pe_win_rate:.1f}%")
    print(f"CE Contribution to Total Trades: {analysis_df['ce_trades'].sum()/total_trades*100:.1f}%")
    print(f"PE Contribution to Total Trades: {analysis_df['pe_trades'].sum()/total_trades*100:.1f}%")
    
    # 8. When Strategy Works Best - Detailed Analysis
    print("\n8. WHEN STRATEGY WORKS BEST:")
    print("-"*40)
    
    # Find conditions for best performance
    top_30_percent = analysis_df.nlargest(int(len(analysis_df) * 0.3), 'net')
    
    print("Characteristics of Top 30% Performing Days:")
    print(f"  Avg Index Change: {top_30_percent['index_change'].mean():.2f}")
    print(f"  Avg Index Volatility: {top_30_percent['index_volatility'].mean():.2f}")
    print(f"  Avg Index Range: {top_30_percent['index_range'].mean():.2f}")
    
    
    # 9. Recommendations
    print("\n9. RECOMMENDATIONS:")
    print("-"*40)
    
    # Generate recommendations based on analysis
    if correlation_index_pnl > 0.3:
        print("✓ Strategy tends to perform better when market moves in certain direction")
    elif correlation_index_pnl < -0.3:
        print("✓ Strategy may be working as a hedge against market moves")
    else:
        print("✓ Strategy shows low correlation with market direction - may be market neutral")
    
    if avg_ce_pnl_per_trade > avg_pe_pnl_per_trade * 1.5:
        print("✓ Consider focusing more on CE trades")
    elif avg_pe_pnl_per_trade > avg_ce_pnl_per_trade * 1.5:
        print("✓ Consider focusing more on PE trades")
    
    if win_rate > 60:
        print("✓ High win rate strategy - good consistency")
    elif win_rate < 40:
        print("⚠ Low win rate - check if large wins compensate for frequent small losses")
    
    return analysis_df

# Additional visualization function
def plot_strategy_performance(df):
    """Create visualizations for strategy analysis"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Cumulative P&L
    df['cumulative_pnl'] = df['net'].cumsum()
    axes[0, 0].plot(df['date'], df['cumulative_pnl'], 'b-', linewidth=2)
    axes[0, 0].set_title('Cumulative P&L')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('P&L (₹)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Daily P&L Distribution
    axes[0, 1].hist(df['net'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Daily P&L Distribution')
    axes[0, 1].set_xlabel('Daily P&L (₹)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CE vs PE P&L Contribution
    ce_cumulative = df['ce_net'].cumsum()
    pe_cumulative = df['pe_net'].cumsum()
    axes[1, 0].plot(df['date'], ce_cumulative, 'r-', label='CE P&L', linewidth=2)
    axes[1, 0].plot(df['date'], pe_cumulative, 'g-', label='PE P&L', linewidth=2)
    axes[1, 0].set_title('CE vs PE Cumulative P&L')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('P&L (₹)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Index Change vs Strategy P&L
    axes[1, 1].scatter(df['index_change'], df['net'], alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_title('Index Change vs Strategy P&L')
    axes[1, 1].set_xlabel('Index Change')
    axes[1, 1].set_ylabel('Strategy P&L (₹)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Volatility vs P&L
    axes[2, 0].scatter(df['index_volatility'], df['net'], alpha=0.6, color='purple')
    axes[2, 0].set_title('Market Volatility vs Strategy P&L')
    axes[2, 0].set_xlabel('Index Volatility')
    axes[2, 0].set_ylabel('Strategy P&L (₹)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Trade Count vs P&L
    total_trades = df['ce_trades'] + df['pe_trades']
    axes[2, 1].scatter(total_trades, df['net'], alpha=0.6, color='orange')
    axes[2, 1].set_title('Number of Trades vs Daily P&L')
    axes[2, 1].set_xlabel('Total Trades per Day')
    axes[2, 1].set_ylabel('Daily P&L (₹)')
    axes[2, 1].grid(True, alpha=0.3)