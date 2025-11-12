# greeks.py
import numpy as np
from numba import njit
from datetime import datetime, timedelta,time
from scipy.optimize import brentq
from scipy.stats import norm
from math import log, sqrt, exp, erf


SQRT_2PI = np.sqrt(2 * np.pi)

@njit
def norm_pdf(x):
    return np.exp(-0.5 * x**2) / SQRT_2PI

@njit
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

@njit
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

@njit
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

@njit
def call_price(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * norm_cdf(d_1) - K * np.exp(-r * T) * norm_cdf(d_2)

@njit
def put_price(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm_cdf(-d_2) - S * norm_cdf(-d_1)

@njit
def delta(S, K, T, r, sigma, call=True):
    d_1 = d1(S, K, T, r, sigma)
    if call:
        return norm_cdf(d_1)
    else:
        return norm_cdf(d_1) - 1.0

@njit
def gamma(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return norm_pdf(d_1) / (S * sigma * np.sqrt(T))

@njit
def vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * norm_pdf(d_1) * np.sqrt(T)

@njit
def theta(S, K, T, r, sigma, call=True):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm_pdf(d_1)
    cdf_d1 = norm_cdf(d_1)
    cdf_d2 = norm_cdf(d_2)
    
    if call:
        return (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2)
    else:
        return (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm_cdf(-d_2))

@njit
def rho(S, K, T, r, sigma, call=True):
    d_2 = d2(S, K, T, r, sigma)
    if call:
        return K * T * np.exp(-r * T) * norm_cdf(d_2)
    else:
        return -K * T * np.exp(-r * T) * norm_cdf(-d_2)





def implied_volatility(market_price, S, K, T, r, call=True):
    """
    Computes the implied volatility using Brent's method.

    Args:
        market_price (float): Observed market price of the option.
        S (float): Spot price of the underlying asset.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        option_type (str): 'call' or 'put'.

    Returns:
        float: Implied volatility (sigma).
    """

    def norm_cdf(x):
        return (1.0 + erf(x / sqrt(2.0))) / 2.0

    def bs_price(sigma):
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if call :
            return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
        else:
            return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    def objective(sigma):
        return bs_price(sigma) - market_price

    try:
        # Brent's method requires a volatility bracket; 1e-6 to 5 is a reasonable guess
        iv = brentq(objective, 1e-6, 5.0, maxiter=1000, xtol=1e-6)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def calc_weekly_expiry(df, time_col, target_day_str):
    """Calc Expiry of Option of DF

    Args:
        df (DataFrame): _description_
        time_col (str): _description_
        target_day_str ("Thursday"): _description_

    Returns:
        _type_: _description_
    """
    # Mapping day names to numbers (Monday=0, Sunday=6)
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2,
        'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
    }
    target_day_num = day_map[target_day_str.lower()]

    # Total seconds in a year (365 days)
    total_seconds_year = 365 * 24 * 60 * 60
    
    def get_target_datetime(current_time):
        # Current weekday
        curr_weekday = current_time.weekday()
        # Days until target day
        days_ahead = (target_day_num - curr_weekday) % 7
        if days_ahead == 0 and current_time.time() > datetime.strptime("15:30:00", "%H:%M:%S").time():
            days_ahead = 7  # next week's target day

        # Get upcoming target date
        target_date = current_time.date() + timedelta(days=days_ahead)
        # Combine with 15:30:00 time
        return datetime.combine(target_date, datetime.strptime("15:30:00", "%H:%M:%S").time())

    # Compute difference and divide by total seconds in year
    df['expiry'] = df[time_col].apply(lambda t: (get_target_datetime(t) - t).total_seconds() / total_seconds_year)
    return df


def bs_price(S, K, T, r, q, sigma, option_type="C"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.upper() == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# Implied Spot solver
def get_implied_spot(ltp, K, T, r, q, iv, option_type="C"):
    sigma = iv / 100 if iv > 1 else iv  # accept iv in % or decimal
    
    def objective(S):
        return bs_price(S, K, T, r, q, sigma, option_type) - ltp
    
    # Spot must be > 0, search in a broad range
    return brentq(objective, 1e-6, 1e6)

def get_time_to_expiry_in_years(expiry_date_str):
    '''
    Returns time to expiry in years and seconds to expiry given an expiry date string in "DD-MM-YYYY" format.
    '''
    
    current_time = datetime.now()
    expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%Y")
    
    expiry_datetime = datetime.combine(expiry_date.date(), time(15, 30))
    seconds_to_expiry = (expiry_datetime - current_time).total_seconds()
    expiry_in_years = seconds_to_expiry / (365 * 24 * 60 * 60)

    return expiry_in_years, seconds_to_expiry

import math
def option_greeks(S, K, T, r, sigma, call:bool):
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
        Implied volatility (annualized)
    option_type : str
        'C' for Call, 'P' for Put

    Returns
    -------
    dict
        Dictionary with Delta, Gamma, Vega, Theta
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    # Compute d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Common term
    pdf_d1 = norm.pdf(d1)

    if call:
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2))
    else:  # Put
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2))

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100  # per 1% change in vol

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta / 365  # per day
    }