import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone

def implied_volatility(option_price, S, K, T, r, option_type):
    if option_price <= 0 or T <= 0:
        return 0
    sigma = 0.2  # initial guess

    for _ in range(50):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "CE":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-6:
            break

        sigma -= (price - option_price) / vega

        if sigma <= 0:
            sigma = 0.0001

    return sigma

def bs_greeks(S, K, T, r, iv, opt_type):
    if T <= 0 or iv <= 0:
        return 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    delta = norm.cdf(d1) if opt_type == "CE" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T)) 
    if opt_type == 'PE':
        gamma = -gamma
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = -S * norm.pdf(d1) * iv / (2 * np.sqrt(T)) / 365

    return delta, gamma, vega, theta


def option_analysis(f,symbol: str, RISK_FREE: float=0.1, LOT_SIZE: int=65):
    data = f.option_chain(symbol)

    options = data["data"]["optionsChain"]
   
    expiry = int(data["data"]["expiryData"][0]["expiry"])
    spot = next(x["ltp"] for x in options if x["option_type"] == "")

    T = max(
        (datetime.fromtimestamp(expiry, tz=timezone.utc)
         - datetime.now(timezone.utc)).total_seconds(),
        0
    ) / (365 * 24 * 3600)

    rows = []
    net_delta_exp = 0
    net_gamma_exp = 0
    net_ce_oi = 0
    net_pe_oi = 0
    net_ce_volume = 0
    net_pe_volume = 0

    for o in options:
        if o["option_type"] not in ("CE", "PE"):
            continue

        if o["option_type"] == "CE":
            net_ce_oi += o["oi"]
            net_ce_volume += o.get("volume", 0)
        else:
            net_pe_oi += o["oi"]
            net_pe_volume += o.get("volume", 0)

        iv = implied_volatility(
            option_price=o["ltp"],
            S=spot,
            K=o["strike_price"],
            T=T,
            r=RISK_FREE,
            option_type=o["option_type"]
        )

        delta, gamma, vega, theta = bs_greeks(
            spot, o["strike_price"], T, RISK_FREE, iv, o["option_type"]
        )

        delta_exp = delta * o["oi"] * LOT_SIZE  
        gamma_exp = gamma * o["oi"] * LOT_SIZE * spot

        net_delta_exp += delta_exp
        net_gamma_exp += gamma_exp

        rows.append({
            "symbol": o['symbol'],
            "strike": o["strike_price"],
            "type": o["option_type"],
            "ltp": o["ltp"],
            "oi": o["oi"],
            "iv": round(iv * 100, 2),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 3),
            "theta": round(theta, 3),
            "delta_exposure": round(delta_exp, 0),
            "gamma_exposure": round(gamma_exp, 0)
        })

    max_delta = None
    max_gamma = None
    if max_delta is None or abs(delta_exp) > abs(max_delta["delta_exposure"]):
        max_delta = {
            "strike": o["strike_price"],
            "type": o["option_type"],
            "delta_exposure": round(delta_exp, 0),
            "delta": round(delta, 4),
            "oi": o["oi"]
        }

    # Largest Gamma Exposure
    if max_gamma is None or abs(gamma_exp) > abs(max_gamma["gamma_exposure"]):
        max_gamma = {
            "strike": o["strike_price"],
            "type": o["option_type"],
            "gamma_exposure": round(gamma_exp, 0),
            "gamma": round(gamma, 6),
            "oi": o["oi"]
        }
    

    return {
    "spot": round(spot, 2),
    "rows": rows,
    "net": {
        "delta_exposure": round(net_delta_exp, 0),
        "gamma_exposure": round(net_gamma_exp, 0),
        "ce_oi": net_ce_oi,
        "pe_oi": net_pe_oi,
        "net_ce_volume": net_ce_volume,
        "net_pe_volume": net_pe_volume,
        "max_delta": max_delta,
        "max_gamma": max_gamma
    }
}

