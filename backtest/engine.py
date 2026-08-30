"""
Core 10-Year Simulation Engine for NIFTY 15-Minute ORB Strategies.
Supports both Option Buying (Long ATM CE/PE) and Option Selling (Credit Spreads).
"""

import os
import math
from datetime import datetime, time
import pandas as pd
import numpy as np

from backtest.data_loader import load_nifty_1min_data, resample_to_15min_candles
from backtest.option_pricing import (
    bs_price,
    find_delta_hedge_strike,
    get_historical_lot_size,
    get_trading_time_fraction
)
from backtest.charges_model import calculate_trade_friction


def run_backtest_simulation(config):
    """
    Executes the 10-year simulation loop across historical 1-minute NIFTY data.
    """
    # 1. Config Extraction
    strat_cfg = config.get("strategy", {})
    strat_mode = strat_cfg.get("mode", "option_buying")
    target_r_multiple = float(strat_cfg.get("target_r_multiple", 2.0))
    sl_r_multiple = float(strat_cfg.get("sl_r_multiple", 1.0))

    cap_cfg = config.get("capital", {})
    initial_capital = float(cap_cfg.get("initial_capital", 500000.0))
    sizing_mode = cap_cfg.get("sizing_mode", "fixed_lots")
    fixed_lots = int(cap_cfg.get("fixed_lots", 1))
    max_lots = int(cap_cfg.get("max_lots", 10))
    margin_per_lot = float(cap_cfg.get("margin_per_lot", 60000.0))

    risk_cfg = config.get("risk_rules", {})
    min_orb_range = float(risk_cfg.get("min_orb_range_pts", 40.0))
    max_orb_range = float(risk_cfg.get("max_orb_range_pts", 200.0))

    opt_cfg = config.get("options", {})
    strike_step = int(opt_cfg.get("strike_step", 50))
    target_delta = float(opt_cfg.get("target_hedge_abs_delta", 0.15))
    use_fixed_offset = bool(opt_cfg.get("use_fixed_strike_offset", False))
    fixed_offset = float(opt_cfg.get("fixed_hedge_offset_pts", 300.0))
    min_net_credit = float(opt_cfg.get("min_net_credit_inr", 5.0))

    friction_cfg = config.get("friction", {})
    sim_cfg = config.get("simulation", {})
    start_year = int(sim_cfg.get("start_year", 2015))
    end_year = int(sim_cfg.get("end_year", 2026))

    # 2. Data Ingestion
    df_1min = load_nifty_1min_data(start_year=start_year, end_year=end_year)
    df_1min["date"] = df_1min["timestamp"].dt.date

    # Group by date for high-speed indexed access
    grouped_days = {d: group for d, group in df_1min.groupby("date")}
    trading_days = sorted(grouped_days.keys())

    current_capital = initial_capital
    trades = []
    daily_equity_series = []

    r_rate = 0.07  # Risk-free rate (7% RBI repo rate proxy)
    sigma = 0.15   # Historical baseline IV

    for trade_date in trading_days:
        day_1m = grouped_days[trade_date]
        if len(day_1m) < 200:
            continue

        day_15m = resample_to_15min_candles(day_1m)
        if len(day_15m) < 5:
            continue

        # 1-Hour ORB Window (First four 15-min candles: 09:15-10:15)
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

        # Target and SL spot levels
        bullish_target = orb_high + (target_r_multiple * orb_range)
        bullish_sl = orb_high - (sl_r_multiple * orb_range)

        bearish_target = orb_low - (target_r_multiple * orb_range)
        bearish_sl = orb_low + (sl_r_multiple * orb_range)

        # Scan 15-min breakout candles (10:15 to 15:10)
        post_orb_candles = day_15m.iloc[4:]
        
        for idx, candle in post_orb_candles.iterrows():
            c_time = candle["time"]
            c_close = float(candle["close"])
            c_ts = candle["timestamp"]

            if c_time > time(15, 10):
                break

            signal = None
            if c_close > orb_high:
                signal = "BULLISH"
            elif c_close < orb_low:
                signal = "BEARISH"

            if signal:
                contract_lot_size = get_historical_lot_size(trade_date)

                # Position Sizing
                if sizing_mode == "fixed_lots":
                    lots = max(1, fixed_lots)
                else:
                    lots = max(1, int(current_capital // margin_per_lot))
                    lots = min(lots, max_lots)

                total_qty = lots * contract_lot_size
                atm_strike = round(c_close / strike_step) * strike_step
                expiry_dt = datetime.combine(trade_date, time(15, 30))
                t_years = get_trading_time_fraction(c_ts, expiry_dt)

                target_spot = bullish_target if signal == "BULLISH" else bearish_target
                sl_spot = bullish_sl if signal == "BULLISH" else bearish_sl

                # =============================================================
                # STRATEGY MODE 1: OPTION BUYING (LONG ATM CE / LONG ATM PE)
                # =============================================================
                if strat_mode == "option_buying":
                    option_type = "CE" if signal == "BULLISH" else "PE"
                    entry_price = bs_price(c_close, atm_strike, t_years, r_rate, sigma, option_type)

                    subsequent_1m = day_1m[day_1m["timestamp"] >= c_ts]
                    exit_time = subsequent_1m.iloc[-1]["timestamp"]
                    exit_spot = c_close
                    exit_reason = "EOD"

                    for _, m_bar in subsequent_1m.iterrows():
                        m_high = float(m_bar["high"])
                        m_low = float(m_bar["low"])
                        m_ts = m_bar["timestamp"]

                        if (signal == "BULLISH" and m_high >= target_spot) or (signal == "BEARISH" and m_low <= target_spot):
                            exit_spot = target_spot
                            exit_time = m_ts
                            exit_reason = "TARGET"
                            break

                        if (signal == "BULLISH" and m_low <= sl_spot) or (signal == "BEARISH" and m_high >= sl_spot):
                            exit_spot = sl_spot
                            exit_time = m_ts
                            exit_reason = "STOP_LOSS"
                            break

                        if m_ts.time() >= time(15, 25):
                            exit_spot = float(m_bar["close"])
                            exit_time = m_ts
                            exit_reason = "EOD"
                            break

                    t_exit = get_trading_time_fraction(exit_time, expiry_dt)
                    exit_price = bs_price(exit_spot, atm_strike, t_exit, r_rate, sigma, option_type)
                    gross_pnl_inr = (exit_price - entry_price) * total_qty

                    # Friction for single leg buy/sell
                    friction_cost = 40.0 + (0.001 * entry_price * total_qty) + (2 * float(friction_cfg.get("slippage_per_unit_leg", 0.5)) * total_qty)
                    net_pnl_inr = gross_pnl_inr - friction_cost
                    current_capital += net_pnl_inr

                    trades.append({
                        "trade_id": len(trades) + 1,
                        "date": str(trade_date),
                        "strategy": "OPTION_BUYING",
                        "signal": signal,
                        "lots": lots,
                        "qty": total_qty,
                        "contract_lot_size": contract_lot_size,
                        "orb_high": round(orb_high, 2),
                        "orb_low": round(orb_low, 2),
                        "orb_range": round(orb_range, 2),
                        "entry_time": str(c_ts.time())[:8],
                        "entry_spot": round(c_close, 2),
                        "strike": atm_strike,
                        "option_type": option_type,
                        "entry_price": round(entry_price, 2),
                        "exit_time": str(exit_time.time())[:8],
                        "exit_spot": round(exit_spot, 2),
                        "exit_price": round(exit_price, 2),
                        "exit_reason": exit_reason,
                        "target_spot": round(target_spot, 2),
                        "sl_spot": round(sl_spot, 2),
                        "gross_pnl": round(gross_pnl_inr, 2),
                        "taxes_friction": round(friction_cost, 2),
                        "net_pnl": round(net_pnl_inr, 2),
                        "capital_after": round(current_capital, 2),
                    })

                # =============================================================
                # STRATEGY MODE 2: CREDIT SPREAD SELLING
                # =============================================================
                else:
                    opt_type = "PE" if signal == "BULLISH" else "CE"
                    sold_strike = atm_strike
                    if use_fixed_offset:
                        hedge_strike = sold_strike - fixed_offset if opt_type == "PE" else sold_strike + fixed_offset
                    else:
                        hedge_strike = find_delta_hedge_strike(c_close, opt_type, t_years, r_rate, sigma, target_delta, strike_step)

                    sell_entry = bs_price(c_close, sold_strike, t_years, r_rate, sigma, opt_type)
                    buy_entry = bs_price(c_close, hedge_strike, t_years, r_rate, sigma, opt_type)
                    net_credit = sell_entry - buy_entry
                    if net_credit < min_net_credit:
                        continue

                    subsequent_1m = day_1m[day_1m["timestamp"] >= c_ts]
                    exit_time = subsequent_1m.iloc[-1]["timestamp"]
                    exit_spot = c_close
                    exit_reason = "EOD"

                    for _, m_bar in subsequent_1m.iterrows():
                        m_high = float(m_bar["high"])
                        m_low = float(m_bar["low"])
                        m_ts = m_bar["timestamp"]

                        if (signal == "BULLISH" and m_high >= target_spot) or (signal == "BEARISH" and m_low <= target_spot):
                            exit_spot = target_spot
                            exit_time = m_ts
                            exit_reason = "TARGET"
                            break

                        if (signal == "BULLISH" and m_low <= sl_spot) or (signal == "BEARISH" and m_high >= sl_spot):
                            exit_spot = sl_spot
                            exit_time = m_ts
                            exit_reason = "STOP_LOSS"
                            break

                        if m_ts.time() >= time(15, 25):
                            exit_spot = float(m_bar["close"])
                            exit_time = m_ts
                            exit_reason = "EOD"
                            break

                    t_exit = get_trading_time_fraction(exit_time, expiry_dt)
                    sell_exit = bs_price(exit_spot, sold_strike, t_exit, r_rate, sigma, opt_type)
                    buy_exit = bs_price(exit_spot, hedge_strike, t_exit, r_rate, sigma, opt_type)
                    gross_pnl_inr = ((sell_entry - sell_exit) + (buy_exit - buy_entry)) * total_qty

                    friction = calculate_trade_friction(
                        sell_entry, sell_exit,
                        buy_entry, buy_exit,
                        total_qty, friction_cfg
                    )
                    net_pnl_inr = gross_pnl_inr - friction["total_friction"]
                    current_capital += net_pnl_inr

                    trades.append({
                        "trade_id": len(trades) + 1,
                        "date": str(trade_date),
                        "strategy": "CREDIT_SPREAD",
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
                        "option_type": opt_type,
                        "net_credit": round(net_credit, 2),
                        "exit_time": str(exit_time.time())[:8],
                        "exit_spot": round(exit_spot, 2),
                        "exit_reason": exit_reason,
                        "target_spot": round(target_spot, 2),
                        "sl_spot": round(sl_spot, 2),
                        "gross_pnl": round(gross_pnl_inr, 2),
                        "taxes_friction": round(friction["total_friction"], 2),
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
