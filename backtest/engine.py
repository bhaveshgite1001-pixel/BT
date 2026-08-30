"""Core 10-Year Simulation Engine for NIFTY 15-Min ORB Delta-Hedged Credit Spread."""

import math
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from .option_pricing import (
    bs_price,
    bs_delta,
    find_delta_hedge_strike,
    get_historical_lot_size,
    get_trading_time_fraction,
)
from .charges_model import calculate_trade_friction
from .data_loader import load_nifty_1min_data, resample_to_15min_candles


def run_backtest_simulation(config):
    """
    Executes complete multi-year backtest simulation based on provided configuration.
    Returns dictionary with summary metrics, equity curve, yearly stats, and trade logs.
    """
    capital_cfg = config.get("capital", {})
    timing_cfg = config.get("timing", {})
    risk_cfg = config.get("risk_rules", {})
    option_cfg = config.get("options", {})
    friction_cfg = config.get("friction", {})
    sim_cfg = config.get("simulation", {})

    initial_capital = float(capital_cfg.get("initial_capital", 500000.0))
    sizing_mode = capital_cfg.get("sizing_mode", "compounding")
    fixed_lots = int(capital_cfg.get("fixed_lots", 1))
    max_lots = int(capital_cfg.get("max_lots", 10))
    margin_per_lot = float(capital_cfg.get("margin_per_lot", 60000.0))

    target_r_multiple = float(risk_cfg.get("target_r_multiple", 1.5))
    min_orb_range = float(risk_cfg.get("min_orb_range_pts", 40.0))
    max_orb_range = float(risk_cfg.get("max_orb_range_pts", 250.0))

    target_delta = float(option_cfg.get("target_hedge_abs_delta", 0.15))
    use_fixed_offset = bool(option_cfg.get("use_fixed_strike_offset", False))
    fixed_offset = float(option_cfg.get("fixed_hedge_offset_pts", 300.0))
    min_net_credit = float(option_cfg.get("min_net_credit_inr", 5.0))
    strike_step = int(option_cfg.get("strike_step", 50))

    start_yr = int(sim_cfg.get("start_year", 2015))
    end_yr = int(sim_cfg.get("end_year", 2026))

    # 1. Load Data
    df_1min = load_nifty_1min_data(start_year=start_yr, end_year=end_yr)
    df_15min = resample_to_15min_candles(df_1min)

    # Group 1-minute bars by date for fast intraday minute lookups
    df_1min["date"] = pd.to_datetime(df_1min["timestamp"]).dt.date
    df_1min_grouped = {d: group for d, group in df_1min.groupby("date")}

    # Group 15-minute bars by date
    days_15min = {d: group for d, group in df_15min.groupby("date")}
    unique_dates = sorted(list(days_15min.keys()))

    current_capital = initial_capital
    peak_capital = initial_capital
    
    trades = []
    daily_equity_series = []

    print(f"[INFO] Running simulation across {len(unique_dates):,} trading sessions ({start_yr} to {end_yr})...")

    for trade_date in unique_dates:
        day_15m = days_15min[trade_date]
        day_1m = df_1min_grouped.get(trade_date)
        
        if day_1m is None or len(day_15m) < 5:
            continue

        # Day base info
        day_vix = float(day_15m.iloc[0].get("vix", 15.0))
        sigma = max(0.08, (day_vix / 100.0))
        r_rate = 0.065
        contract_lot_size = get_historical_lot_size(trade_date)

        # 1. Form 1-Hour ORB (First 4 candles: 09:15-09:30, 09:30-09:45, 09:45-10:00, 10:00-10:15)
        orb_candles = day_15m.iloc[:4]
        orb_high = float(orb_candles["high"].max())
        orb_low = float(orb_candles["low"].min())
        orb_range = orb_high - orb_low

        day_traded = False

        # Range filter check
        if orb_range < min_orb_range or orb_range > max_orb_range:
            daily_equity_series.append({
                "date": str(trade_date),
                "equity": round(current_capital, 2),
                "daily_pnl": 0.0,
                "trades_count": 0
            })
            continue

        # Target and SL levels
        bullish_target = orb_high + target_r_multiple * orb_range
        bullish_sl = orb_low

        bearish_target = orb_low - target_r_multiple * orb_range
        bearish_sl = orb_high

        # 2. Scan remaining 15-min candles (10:15 onwards)
        post_orb_candles = day_15m.iloc[4:]
        
        for idx, candle in post_orb_candles.iterrows():
            c_time = candle["time"]
            c_close = float(candle["close"])
            c_ts = candle["timestamp"]

            if c_time > time(15, 20):
                break  # past trade window end

            signal = None
            if c_close > orb_high:
                signal = "BULLISH"
            elif c_close < orb_low:
                signal = "BEARISH"

            if signal:
                # 3. Position Sizing
                if sizing_mode == "fixed_lots":
                    lots = max(1, fixed_lots)
                else:
                    lots = max(1, int(current_capital // margin_per_lot))
                    lots = min(lots, max_lots)

                total_qty = lots * contract_lot_size

                # 4. Strike & Expiry Setup
                atm_strike = round(c_close / strike_step) * strike_step
                
                # Expiry datetime (assume current week expiry at 15:30)
                # If Thursday/Tuesday (expiry day), t_days = 0, else 1-3 days
                expiry_dt = datetime.combine(trade_date, time(15, 30))
                t_years = get_trading_time_fraction(c_ts, expiry_dt)

                if signal == "BULLISH":
                    # Sell ATM PE, Buy OTM PE
                    option_type = "PE"
                    sold_strike = atm_strike
                    if use_fixed_offset:
                        hedge_strike = sold_strike - fixed_offset
                    else:
                        hedge_strike = find_delta_hedge_strike(c_close, "PE", t_years, r_rate, sigma, target_delta, strike_step)
                else:
                    # Sell ATM CE, Buy OTM CE
                    option_type = "CE"
                    sold_strike = atm_strike
                    if use_fixed_offset:
                        hedge_strike = sold_strike + fixed_offset
                    else:
                        hedge_strike = find_delta_hedge_strike(c_close, "CE", t_years, r_rate, sigma, target_delta, strike_step)

                # Entry option pricing
                sell_entry_price = bs_price(c_close, sold_strike, t_years, r_rate, sigma, option_type)
                buy_entry_price = bs_price(c_close, hedge_strike, t_years, r_rate, sigma, option_type)
                net_credit = sell_entry_price - buy_entry_price

                if net_credit < min_net_credit:
                    continue  # credit too thin

                # 5. Minute-by-Minute Intraday Tracking
                entry_minute_ts = c_ts
                subsequent_1m = day_1m[day_1m["timestamp"] >= entry_minute_ts]

                exit_time = None
                exit_spot = c_close
                exit_reason = "EOD"

                for _, m_bar in subsequent_1m.iterrows():
                    m_ts = m_bar["timestamp"]
                    m_time = m_ts.time()
                    m_high = float(m_bar["high"])
                    m_low = float(m_bar["low"])

                    if signal == "BULLISH":
                        if m_high >= bullish_target:
                            exit_spot = bullish_target
                            exit_time = m_ts
                            exit_reason = "TARGET"
                            break
                        elif m_low <= bullish_sl:
                            exit_spot = bullish_sl
                            exit_time = m_ts
                            exit_reason = "STOP_LOSS"
                            break
                    else:
                        if m_low <= bearish_target:
                            exit_spot = bearish_target
                            exit_time = m_ts
                            exit_reason = "TARGET"
                            break
                        elif m_high >= bearish_sl:
                            exit_spot = bearish_sl
                            exit_time = m_ts
                            exit_reason = "STOP_LOSS"
                            break

                    if m_time >= time(15, 25):
                        exit_spot = float(m_bar["close"])
                        exit_time = m_ts
                        exit_reason = "EOD"
                        break

                if exit_time is None:
                    exit_time = subsequent_1m.iloc[-1]["timestamp"]
                    exit_spot = float(subsequent_1m.iloc[-1]["close"])
                    exit_reason = "EOD"

                # 6. Exit Option Pricing & P&L Calculation
                t_exit_years = get_trading_time_fraction(exit_time, expiry_dt)
                sell_exit_price = bs_price(exit_spot, sold_strike, t_exit_years, r_rate, sigma, option_type)
                buy_exit_price = bs_price(exit_spot, hedge_strike, t_exit_years, r_rate, sigma, option_type)

                # Gross P&L: (Sold entry - Sold exit) + (Bought exit - Bought entry)
                gross_pnl_pts = (sell_entry_price - sell_exit_price) + (buy_exit_price - buy_entry_price)
                gross_pnl_inr = gross_pnl_pts * total_qty

                friction = calculate_trade_friction(
                    sell_entry_price, sell_exit_price,
                    buy_entry_price, buy_exit_price,
                    total_qty, friction_cfg
                )

                net_pnl_inr = gross_pnl_inr - friction["total_friction"]
                current_capital += net_pnl_inr
                peak_capital = max(peak_capital, current_capital)

                trades.append({
                    "trade_id": len(trades) + 1,
                    "date": str(trade_date),
                    "signal": signal,
                    "lots": lots,
                    "qty": total_qty,
                    "contract_lot_size": contract_lot_size,
                    "orb_high": round(orb_high, 2),
                    "orb_low": round(orb_low, 2),
                    "orb_range": round(orb_range, 2),
                    "entry_time": str(c_ts.time())[:8],
                    "entry_spot": round(c_close, 2),
                    "sold_strike": sold_strike,
                    "hedge_strike": hedge_strike,
                    "option_type": option_type,
                    "sell_entry_price": round(sell_entry_price, 2),
                    "buy_entry_price": round(buy_entry_price, 2),
                    "net_credit": round(net_credit, 2),
                    "exit_time": str(exit_time.time())[:8],
                    "exit_spot": round(exit_spot, 2),
                    "sell_exit_price": round(sell_exit_price, 2),
                    "buy_exit_price": round(buy_exit_price, 2),
                    "exit_reason": exit_reason,
                    "gross_pnl": round(gross_pnl_inr, 2),
                    "total_taxes": friction["total_taxes"],
                    "slippage": friction["slippage_cost"],
                    "net_pnl": round(net_pnl_inr, 2),
                    "capital_after": round(current_capital, 2),
                })

                day_traded = True
                break  # 1 trade per day limit

        daily_equity_series.append({
            "date": str(trade_date),
            "equity": round(current_capital, 2),
            "daily_pnl": round(trades[-1]["net_pnl"], 2) if day_traded else 0.0,
            "trades_count": 1 if day_traded else 0
        })

    return {
        "initial_capital": initial_capital,
        "final_capital": round(current_capital, 2),
        "total_trades": len(trades),
        "trades": trades,
        "equity_series": daily_equity_series,
        "config": config,
    }
