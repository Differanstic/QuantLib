import pandas as pd

def indexCatDB(df:pd.DataFrame,tf):
    """
    Calculate change, range, and volatility over a given timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a datetime index and 'spot_price' column.
    tf : str
        Resample timeframe (e.g. '1T' = 1 min, '5T' = 5 min, '15T' = 15 min).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['change', 'range', 'vol'] for each timeframe window.
    """
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    grouped = df['spot_price'].resample(tf)
    
    stats = pd.DataFrame({
        'change': grouped.last() - grouped.first(),
        'range': grouped.max() - grouped.min(),
        'volatility': round(grouped.std(),4)
    })
    
    return stats.dropna()
