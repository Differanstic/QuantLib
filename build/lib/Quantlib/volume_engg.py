from numba import njit
import pandas as pd
import numpy as np

@njit
def _fast_profile(ltps, ltqs, bins, value_area_pct):
    """
    Fully Numba-compatible histogram-based POC/VAH/VAL computation.
    """
    n = ltps.size
    if n == 0:
        return np.nan, np.nan, np.nan

    ltp_min = np.min(ltps)
    ltp_max = np.max(ltps)
    if ltp_max == ltp_min:
        return ltp_min, ltp_min, ltp_min

    # Create histogram bins manually (Numba-safe)
    edges = np.linspace(ltp_min, ltp_max, bins + 1)
    hist = np.zeros(bins, dtype=np.float64)

    # Manually bin ltq into histogram
    for i in range(n):
        price = ltps[i]
        qty = ltqs[i]
        bin_idx = int((price - ltp_min) / (ltp_max - ltp_min) * bins)
        if bin_idx == bins:  # include right edge
            bin_idx = bins - 1
        hist[bin_idx] += qty

    total_vol = hist.sum()
    if total_vol == 0:
        return np.nan, np.nan, np.nan

    ltp_levels = 0.5 * (edges[:-1] + edges[1:])
    poc_idx = np.argmax(hist)
    poc = ltp_levels[poc_idx]

    # Value area (top 70%)
    sorted_idx = np.argsort(hist)[::-1]
    cum_vol = np.cumsum(hist[sorted_idx])
    cutoff = total_vol * value_area_pct

    mask = cum_vol <= cutoff
    if np.any(mask):
        vah = np.max(ltp_levels[sorted_idx][mask])
        val = np.min(ltp_levels[sorted_idx][mask])
    else:
        vah = val = poc

    return poc, vah, val
@njit
def _fast_cum_vp(ltp, ltq, bin_edges, value_area=0.7):
    n = len(ltp)
    n_bins = len(bin_edges) - 1
    hist = np.zeros(n_bins)
    poc_list = np.empty(n)
    vah_list = np.empty(n)
    val_list = np.empty(n)
    for i in range(n):
        # find bin index for current price
        bin_idx = np.searchsorted(bin_edges, ltp[i], side='right') - 1
        if 0 <= bin_idx < n_bins:
            hist[bin_idx] += ltq[i]
        total_vol = hist.sum()
        if total_vol == 0:
            poc_list[i] = np.nan
            vah_list[i] = np.nan
            val_list[i] = np.nan
            continue
        
        # Point of Control (max volume bin)
        max_idx = np.argmax(hist)
        poc_ltp = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2
        # Compute value area (top 70% of volume)
        sorted_idx = np.argsort(hist)[::-1]
        cum_vol = np.cumsum(hist[sorted_idx])
        cutoff = total_vol * value_area
        mask = cum_vol <= cutoff
        valid_idx = sorted_idx[mask]
        if len(valid_idx) > 0:
            vah = (bin_edges[valid_idx].max() + bin_edges[valid_idx].max() + 1) / 2
            val = (bin_edges[valid_idx].min() + bin_edges[valid_idx].min() + 1) / 2
        else:
            vah = poc_ltp
            val = poc_ltp
        poc_list[i] = poc_ltp
        vah_list[i] = vah
        val_list[i] = val
    return poc_list, vah_list, val_list

# Cumulative Volume Profile
def cumulative_volume_profile(df, ltp_col='ltp', vol_col='ltq', bins=100, value_area=0.7):
    """
    Super-fast cumulative volume profile (POC, VAH, VAL) using numba.
    """
    df = df.dropna(subset=[ltp_col, vol_col]).copy()
    df = df.sort_index()
    ltp = df[ltp_col].values.astype(np.float64)
    ltq = df[vol_col].values.astype(np.float64)
    bin_edges = np.linspace(ltp.min(), ltp.max(), bins + 1)
    poc, vah, val = _fast_cum_vp(ltp, ltq, bin_edges, value_area)
    df['poc'] = poc
    df['vah'] = vah
    df['val'] = val
    return df
def rolling_volume_profile_realtime(df: pd.DataFrame, timeframe='5min', bins=100, value_area_pct=0.7):
    """
    ⚡ Ultra-fast rolling volume profile (no lookahead bias).
    Each timeframe's profile is applied to the next timeframe.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    results = []
    prev_poc = prev_vah = prev_val = np.nan
    prev_time = None

    # Use resample (fast grouping)
    for g_time, g in df.resample(timeframe):
        if len(g) == 0:
            continue

        # Assign previous profile to next window
        if prev_time is not None:
            results.append([g_time, prev_poc, prev_vah, prev_val])

        ltps = g['ltp'].to_numpy(np.float64)
        ltqs = g['ltq'].to_numpy(np.float64)

        poc, vah, val = _fast_profile(ltps, ltqs, bins, value_area_pct)
        prev_poc, prev_vah, prev_val = poc, vah, val
        prev_time = g_time

    profile_df = pd.DataFrame(results, columns=['timestamp', 'r_poc', 'r_vah', 'r_val']).set_index('timestamp')

    out = df.merge(profile_df, left_index=True, right_index=True, how='outer')
    out[['r_poc', 'r_vah', 'r_val']] = out[['r_poc', 'r_vah', 'r_val']].ffill()
    out.reset_index(inplace=True)

    return out