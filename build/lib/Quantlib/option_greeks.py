# greeks.py
import numpy as np
from numba import njit
from datetime import datetime, timedelta, time
from scipy.optimize import brentq
from math import log, sqrt, exp, erf
import math

# ---------------------------------------------------------------------
# Core constants
# ---------------------------------------------------------------------
SQRT_2PI = np.sqrt(2.0 * np.pi)
INV_SQRT2 = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------
# Core normal pdf / cdf and Black–Scholes building blocks (Numba-optimized)
# ---------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def safe_sqrt(x):
    if x <= 0:
        return 1e-12
    return np.sqrt(x)

@njit(fastmath=True, cache=True)
def safe_div(a, b):
    if b == 0:
        return 1e-12
    return a / b

@njit(fastmath=True, cache=True)
def d1(S, K, T, r, sigma):
    T_ = safe_sqrt(T)
    sigT = sigma * T_
    if sigT == 0:
        sigT = 1e-12
    return (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigT

@njit(fastmath=True, cache=True)
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * safe_sqrt(T)

@njit(fastmath=True, cache=True)
def norm_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI

@njit(fastmath=True, cache=True)
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

@njit(fastmath=True, cache=True)
def call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * norm_cdf(d_1) - K * np.exp(-r * T) * norm_cdf(d_2)

@njit(fastmath=True, cache=True)
def put_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K - S, 0)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm_cdf(-d_2) - S * norm_cdf(-d_1)

@njit(fastmath=True, cache=True)
def delta(S, K, T, r, sigma, call=True):
    if sigma <= 0 or T <= 0:
        return 1.0 if (call and S > K) else 0.0
    d_1 = d1(S, K, T, r, sigma)
    return norm_cdf(d_1) if call else norm_cdf(d_1) - 1.0

@njit(fastmath=True, cache=True)
def gamma(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    denom = S * sigma * safe_sqrt(T)
    if denom == 0:
        return 0.0
    return norm_pdf(d_1) / denom

@njit(fastmath=True, cache=True)
def vega(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    return S * norm_pdf(d_1) * safe_sqrt(T)

@njit(fastmath=True, cache=True)
def theta(S, K, T, r, sigma, call=True):
    if sigma <= 0 or T <= 0:
        return 0.0
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    pdf = norm_pdf(d_1)
    sqrtT = safe_sqrt(T)

    if call:
        return (-S * pdf * sigma / (2 * sqrtT)
                - r * K * np.exp(-r * T) * norm_cdf(d_2))
    else:
        return (-S * pdf * sigma / (2 * sqrtT)
                + r * K * np.exp(-r * T) * norm_cdf(-d_2))

@njit(fastmath=True, cache=True)
def rho(S, K, T, r, sigma, call=True):
    d_2 = d2(S, K, T, r, sigma)
    if call:
        return K * T * np.exp(-r * T) * norm_cdf(d_2)
    else:
        return -K * T * np.exp(-r * T) * norm_cdf(-d_2)


# ---------------------------------------------------------------------
# Implied volatility (Brent root finding – still SciPy but fast in C)
# ---------------------------------------------------------------------
def implied_volatility(market_price, S, K, T, r, call=True):
    """
    Computes the implied volatility using Brent's method.

    Args:
        market_price (float): Observed market price of the option.
        S (float): Spot price of the underlying asset.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        call (bool): True for call, False for put.

    Returns:
        float: Implied volatility (sigma).
    """

    # local fast normal cdf (no SciPy, pure math)
    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def bs_price_local(sigma_):
        d1_ = (log(S / K) + (r + 0.5 * sigma_ * sigma_) * T) / (sigma_ * sqrt(T))
        d2_ = d1_ - sigma_ * sqrt(T)
        if call:
            return S * _norm_cdf(d1_) - K * exp(-r * T) * _norm_cdf(d2_)
        else:
            return K * exp(-r * T) * _norm_cdf(-d2_) - S * _norm_cdf(-d1_)

    def objective(sigma_):
        return bs_price_local(sigma_) - market_price

    try:
        # Brent's method requires a volatility bracket; 1e-6 to 5 is a reasonable range
        iv = brentq(objective, 1e-6, 5.0, maxiter=1000, xtol=1e-6)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------------
# Time-to-expiry helpers (Pandas-friendly, left as Python)
# ---------------------------------------------------------------------
def calc_weekly_expiry(df, time_col, target_day_str):
    """Calc Expiry of Option of DF

    Args:
        df (DataFrame): DataFrame with timestamp column.
        time_col (str): name of datetime column.
        target_day_str (str): like "Thursday".

    Returns:
        DataFrame: with 'expiry' column = T in years.
    """
    # Mapping day names to numbers (Monday=0, Sunday=6)
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2,
        'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
    }
    target_day_num = day_map[target_day_str.lower()]

    # Total seconds in a year (365 days)
    total_seconds_year = 365 * 24 * 60 * 60

    cutoff_time = time(15, 30, 0)

    def get_target_datetime(current_time):
        curr_weekday = current_time.weekday()
        days_ahead = (target_day_num - curr_weekday) % 7
        if days_ahead == 0 and current_time.time() > cutoff_time:
            days_ahead = 7  # next week's target day

        target_date = current_time.date() + timedelta(days=days_ahead)
        return datetime.combine(target_date, cutoff_time)

    df['expiry'] = df[time_col].apply(
        lambda t: (get_target_datetime(t) - t).total_seconds() / total_seconds_year
    )
    return df


# ---------------------------------------------------------------------
# Black–Scholes price wrapper (uses Numba BS core)
# ---------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type="C"):
    """
    Black-Scholes option pricing formula without dividend yield (q=0)

    Parameters:
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (in years)
    r : float - Risk-free interest rate
    sigma : float - Volatility
    option_type : str - "C" for Call, "P" for Put

    Returns:
    float - Theoretical option price
    """
    if option_type.upper() == "C":
        return float(call_price(S, K, T, r, sigma))
    else:
        return float(put_price(S, K, T, r, sigma))


# ---------------------------------------------------------------------
# Implied Spot solver
# ---------------------------------------------------------------------
def get_implied_spot(ltp, K, T, r, q, iv, option_type="C"):
    """
    Solve for implied spot S given option LTP, strike, T, r, q, and IV.
    This keeps the same function name/signature but fixes the bs_price misuse.
    """
    sigma = iv / 100.0 if iv > 1 else iv  # accept iv in % or decimal

    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def objective(S):
        # BS with continuous dividend yield q
        if S <= 0:
            return 1e9  # avoid log issue
        d1_ = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        d2_ = d1_ - sigma * sqrt(T)

        disc_r = exp(-r * T)
        disc_q = exp(-q * T)

        if option_type.upper() == "C":
            price = S * disc_q * _norm_cdf(d1_) - K * disc_r * _norm_cdf(d2_)
        else:
            price = K * disc_r * _norm_cdf(-d2_) - S * disc_q * _norm_cdf(-d1_)

        return price - ltp

    # Spot must be > 0, search in a broad range
    return brentq(objective, 1e-6, 1e6)


def get_time_to_expiry_in_years(expiry_date_str):
    """
    Returns time to expiry in years and seconds to expiry
    given an expiry date string in "DD-MM-YYYY" format.
    """
    current_time = datetime.now()
    expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%Y")
    expiry_datetime = datetime.combine(expiry_date.date(), time(15, 30))

    seconds_to_expiry = (expiry_datetime - current_time).total_seconds()
    expiry_in_years = seconds_to_expiry / (365 * 24 * 60 * 60)

    return expiry_in_years, seconds_to_expiry


# ---------------------------------------------------------------------
# High-level option Greeks wrapper (uses Numba Greeks internally)
# ---------------------------------------------------------------------
def option_greeks(S, K, T, r, sigma, call: bool):
    """
    Calculate option Greeks: Delta, Gamma, Vega, Theta (Black-Scholes model)

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to expiry (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Implied volatility (annualized, decimal)
    call : bool
        True for Call, False for Put

    Returns
    -------
    dict
        Dictionary with Delta, Gamma, Vega, Theta (per day)
    """
    if T <= 0.0 or sigma <= 0.0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    # Use Numba-accelerated primitives
    dlt = float(delta(S, K, T, r, sigma, call))
    gmm = float(gamma(S, K, T, r, sigma))
    vga = float(vega(S, K, T, r, sigma)) / 100.0  # per 1% change in vol
    th = float(theta(S, K, T, r, sigma, call)) / 365.0  # per day

    return {
        'delta': dlt,
        'gamma': gmm,
        'vega': vga,
        'theta': th
    }
