"""Black-Scholes Option Pricing, Greeks, and Historical Lot Size Model."""

import math
from datetime import datetime, date

# Standard normal distribution functions using math.erf
def _norm_cdf(x):
    """Cumulative distribution function for standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _norm_pdf(x):
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def get_historical_lot_size(trade_date):
    """
    Returns the exact NSE NIFTY contract lot size for a given historical date.
    - Up to 2021-04-30: 75
    - 2021-05-01 to 2024-04-25: 50
    - 2024-04-26 to 2024-11-19: 25
    - 2024-11-20 onwards: 65 (and 25)
    """
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date[:10], "%Y-%m-%d").date()
    elif isinstance(trade_date, datetime):
        trade_date = trade_date.date()

    if trade_date < date(2021, 5, 1):
        return 75
    elif trade_date < date(2024, 4, 26):
        return 50
    elif trade_date < date(2024, 11, 20):
        return 25
    else:
        return 65


def bs_price(spot, strike, t_years, r, sigma, option_type):
    """
    Calculate Black-Scholes European option theoretical price.
    """
    if t_years <= 1e-6:
        # At expiry payoff
        if option_type.upper() == "CE":
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)

    if sigma <= 1e-6:
        sigma = 1e-6

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)

    if option_type.upper() == "CE":
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    return max(0.05, price)


def bs_delta(spot, strike, t_years, r, sigma, option_type):
    """Calculate Black-Scholes Delta."""
    if t_years <= 1e-6:
        if option_type.upper() == "CE":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    if sigma <= 1e-6:
        sigma = 1e-6

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    if option_type.upper() == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def solve_iv(spot, strike, t_years, r, target_price, option_type):
    """Numerical solver for Implied Volatility using bisection method."""
    if target_price <= 0.05:
        return 0.15

    low_vol = 0.01
    high_vol = 3.0
    for _ in range(50):
        mid_vol = (low_vol + high_vol) / 2.0
        price = bs_price(spot, strike, t_years, r, mid_vol, option_type)
        diff = price - target_price
        if abs(diff) < 1e-4:
            return mid_vol
        if diff > 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol
    return (low_vol + high_vol) / 2.0


def find_delta_hedge_strike(spot, option_type, t_years, r, iv, target_delta=0.15, strike_step=50):
    """
    Finds the OTM strike with theoretical |delta| closest to target_delta.
    """
    atm_strike = round(spot / strike_step) * strike_step
    best_strike = atm_strike
    best_diff = 999.0

    # Scan up to 15 strikes OTM
    if option_type.upper() == "CE":
        candidates = [atm_strike + i * strike_step for i in range(1, 16)]
    else:
        candidates = [atm_strike - i * strike_step for i in range(1, 16)]

    for strike in candidates:
        d = abs(bs_delta(spot, strike, t_years, r, iv, option_type))
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = strike

    return best_strike


def get_trading_time_fraction(trade_datetime, expiry_datetime):
    """
    Calculate fraction of year remaining based on trading hours (375 mins/day, 250 days/year).
    """
    if trade_datetime >= expiry_datetime:
        return 1e-6
    # Total minutes between now and expiry session close
    total_seconds = (expiry_datetime - trade_datetime).total_seconds()
    total_days = total_seconds / 86400.0
    # Annualized time in trading days (250 days/yr)
    t_years = max(1e-6, total_days / 365.0)
    return t_years
