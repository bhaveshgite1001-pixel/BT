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
    find_premium_hedge_strike,
    get_historical_lot_size,
    get_trading_time_fraction
)
from backtest.charges_model import calculate_trade_friction

def get_intraday_iv(base_vix: float, t: time) -> float:
    """
    Simulates Intraday Implied Volatility Crush.
    Options are heavily inflated in the morning and crushed by afternoon.
    """
    if t < time(10, 30):
        return base_vix * 1.15  # +15% IV premium in the morning
    elif t < time(12, 0):
        return base_vix * 1.05  # +5% mid-day
    elif t < time(14, 0):
        return base_vix * 0.95  # -5% afternoon crush
    else:
        return base_vix * 0.90  # -10% deep afternoon crush


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
    target_sold_delta = float(opt_cfg.get("target_sold_abs_delta", 0.50))
    target_hedge_delta = float(opt_cfg.get("target_hedge_abs_delta", 0.05))
    use_fixed_offset = bool(opt_cfg.get("use_fixed_strike_offset", False))
    fixed_offset = float(opt_cfg.get("fixed_hedge_offset_pts", 300.0))
    min_net_credit = float(opt_cfg.get("min_net_credit_inr", 5.0))

    friction_cfg = config.get("friction", {})
    sim_cfg = config.get("simulation", {})
    start_year = int(sim_cfg.get("start_year", 2008))
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

    for trade_date in trading_days:
        day_1m = grouped_days[trade_date]
        if len(day_1m) < 200:
            continue

        day_15m = resample_to_15min_candles(day_1m)
        if len(day_15m) < 5:
            continue

        base_vix = min(1.0, max(0.08, float(day_1m.iloc[0]["vix"]) / 100.0))

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

                # Sizing moved to individual strategies
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
                    entry_sigma = get_intraday_iv(base_vix, c_ts.time())
                    entry_price = bs_price(c_close, atm_strike, t_years, r_rate, entry_sigma, option_type)
                    
                    cost_per_lot = entry_price * contract_lot_size
                    if sizing_mode == "fixed_lots":
                        lots = fixed_lots
                    else:
                        lots = min(max_lots, int(current_capital // cost_per_lot))
                        
                    if lots * cost_per_lot > current_capital:
                        lots = int(current_capital // cost_per_lot)
                    if lots < 1:
                        continue # cant afford
                    total_qty = lots * contract_lot_size

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
                    exit_sigma = get_intraday_iv(base_vix, exit_time.time())
                    exit_price = bs_price(exit_spot, atm_strike, t_exit, r_rate, exit_sigma, option_type)
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
                # STRATEGY MODE 2: PREMIUM BREAKDOWN (SHORT CE / SHORT PE)
                # =============================================================
                else:
                    if sizing_mode == "fixed_lots":
                        lots = fixed_lots
                    else:
                        lots = min(max_lots, int(current_capital // margin_per_lot))
                        
                    if lots * margin_per_lot > current_capital:
                        lots = int(current_capital // margin_per_lot)
                    if lots < 1:
                        continue # cant afford
                    total_qty = lots * contract_lot_size
                    # 1. At 9:15 AM
                    start_bar = day_1m.iloc[0]
                    spot_915 = float(start_bar["open"])
                    vix_915 = min(1.0, max(0.08, float(start_bar["vix"]) / 100.0))
                    expiry_dt = datetime.combine(trade_date, time(15, 30))
                    t_915 = get_trading_time_fraction(start_bar["timestamp"], expiry_dt)
                    sigma_915 = get_intraday_iv(vix_915, time(9, 15))

                    def get_200_strike(spot, opt_type):
                        best_strike = None
                        min_diff = 9999
                        base = round(spot/strike_step)*strike_step
                        for i in range(-40, 41):
                            k = base + (i*strike_step)
                            px = bs_price(spot, k, t_915, r_rate, sigma_915, opt_type)
                            if abs(px - 200) < min_diff:
                                min_diff = abs(px - 200)
                                best_strike = k
                        return best_strike
                        
                    ce_strike = get_200_strike(spot_915, "CE")
                    pe_strike = get_200_strike(spot_915, "PE")
                    if not ce_strike or not pe_strike:
                        continue
                    
                    # 2. Synthesize 9:15 to 10:15 Premium Chart to get ORB Lows
                    ce_orb_low = 9999.0
                    ce_orb_high = 0.0
                    pe_orb_low = 9999.0
                    pe_orb_high = 0.0
                    
                    for _, m_bar in day_1m[day_1m["timestamp"] <= c_ts].iterrows():
                        m_spot_h = float(m_bar["high"])
                        m_spot_l = float(m_bar["low"])
                        m_ts = m_bar["timestamp"]
                        m_t = get_trading_time_fraction(m_ts, expiry_dt)
                        m_vix = min(1.0, max(0.08, float(m_bar["vix"]) / 100.0))
                        m_sigma = get_intraday_iv(m_vix, m_ts.time())
                        
                        ce_px_h = bs_price(m_spot_h, ce_strike, m_t, r_rate, m_sigma, "CE")
                        ce_px_l = bs_price(m_spot_l, ce_strike, m_t, r_rate, m_sigma, "CE")
                        ce_orb_high = max(ce_orb_high, ce_px_h)
                        ce_orb_low = min(ce_orb_low, ce_px_l)
                        
                        pe_px_h = bs_price(m_spot_l, pe_strike, m_t, r_rate, m_sigma, "PE")
                        pe_px_l = bs_price(m_spot_h, pe_strike, m_t, r_rate, m_sigma, "PE")
                        pe_orb_high = max(pe_orb_high, pe_px_h)
                        pe_orb_low = min(pe_orb_low, pe_px_l)
                        
                    # 3. Track from 10:15 onwards for Premium Breakdowns
                    subsequent_1m = day_1m[day_1m["timestamp"] > c_ts]
                    traded_legs = []
                    positions = []
                    
                    last_3m_ce_close = 0
                    last_3m_pe_close = 0
                    
                    for _, m_bar in subsequent_1m.iterrows():
                        m_ts = m_bar["timestamp"]
                        t = m_ts.time()
                        
                        m_spot_h = float(m_bar["high"])
                        m_spot_l = float(m_bar["low"])
                        m_spot_c = float(m_bar["close"])
                        m_t = get_trading_time_fraction(m_ts, expiry_dt)
                        m_vix = min(1.0, max(0.08, float(m_bar["vix"]) / 100.0))
                        m_sigma = get_intraday_iv(m_vix, t)
                        
                        ce_s_px = bs_price(m_spot_c, ce_strike, m_t, r_rate, m_sigma, "CE")
                        pe_s_px = bs_price(m_spot_c, pe_strike, m_t, r_rate, m_sigma, "PE")
                        
                        if t.minute % 3 == 0:
                            last_3m_ce_close = ce_s_px
                            last_3m_pe_close = pe_s_px

                            if t <= time(15, 20):
                                # Check entries based on last 3-min close
                                if "CE" not in traded_legs and last_3m_ce_close > 0 and last_3m_ce_close < ce_orb_low:
                                    traded_legs.append("CE")
                                    if use_fixed_offset:
                                        ce_h_strike = ce_strike + fixed_offset
                                    else:
                                        ce_h_strike = find_delta_hedge_strike(m_spot_c, "CE", m_t, r_rate, m_sigma, target_hedge_delta, strike_step)
                                    
                                    ce_h_px = bs_price(m_spot_c, ce_h_strike, m_t, r_rate, m_sigma, "CE")
                                    positions.append({
                                        "type": "CE_SHORT", "entry": ce_s_px, "sl": ce_orb_high, "time": m_ts,
                                        "strike": ce_strike, "hedge_strike": ce_h_strike, "hedge_entry": ce_h_px, "spot": m_spot_c
                                    })
                                    
                                if "PE" not in traded_legs and last_3m_pe_close > 0 and last_3m_pe_close < pe_orb_low:
                                    traded_legs.append("PE")
                                    if use_fixed_offset:
                                        pe_h_strike = pe_strike - fixed_offset
                                    else:
                                        pe_h_strike = find_delta_hedge_strike(m_spot_c, "PE", m_t, r_rate, m_sigma, target_hedge_delta, strike_step)
                                        
                                    pe_h_px = bs_price(m_spot_c, pe_h_strike, m_t, r_rate, m_sigma, "PE")
                                    positions.append({
                                        "type": "PE_SHORT", "entry": pe_s_px, "sl": pe_orb_high, "time": m_ts,
                                        "strike": pe_strike, "hedge_strike": pe_h_strike, "hedge_entry": pe_h_px, "spot": m_spot_c
                                    })
                                    
                        # Monitor SL or EOD for active positions
                        active_positions = []
                        for pos in positions:
                            opt_type = "CE" if "CE" in pos["type"] else "PE"
                            opt_px_h = bs_price(m_spot_h if opt_type=="CE" else m_spot_l, pos["strike"], m_t, r_rate, m_sigma, opt_type)
                            
                            exit_px = None
                            exit_reason = None
                            
                            if opt_px_h >= pos["sl"]:
                                exit_px = pos["sl"]
                                exit_reason = "STOP_LOSS"
                            elif t >= time(15, 25):
                                exit_px = bs_price(m_spot_c, pos["strike"], m_t, r_rate, m_sigma, opt_type)
                                exit_reason = "EOD"
                                
                            if exit_px is not None:
                                buy_exit = bs_price(m_spot_c, pos["hedge_strike"], m_t, r_rate, m_sigma, opt_type)
                                
                                gross_pnl_inr = ((pos["entry"] - exit_px) + (buy_exit - pos["hedge_entry"])) * total_qty
                                friction = calculate_trade_friction(pos["entry"], exit_px, pos["hedge_entry"], buy_exit, total_qty, friction_cfg)
                                net_pnl_inr = gross_pnl_inr - friction["total_friction"]
                                current_capital += net_pnl_inr
                                
                                trades.append({
                                    "trade_id": len(trades) + 1,
                                    "date": str(trade_date),
                                    "strategy": "PREMIUM_BREAKDOWN",
                                    "signal": opt_type,
                                    "lots": lots,
                                    "qty": total_qty,
                                    "contract_lot_size": contract_lot_size,
                                    "orb_high": round(ce_orb_high if opt_type=="CE" else pe_orb_high, 2),
                                    "orb_low": round(ce_orb_low if opt_type=="CE" else pe_orb_low, 2),
                                    "orb_range": round((ce_orb_high - ce_orb_low) if opt_type=="CE" else (pe_orb_high - pe_orb_low), 2),
                                    "entry_time": str(pos["time"].time())[:8],
                                    "entry_spot": round(pos["spot"], 2),
                                    "sold_strike": pos["strike"],
                                    "hedge_strike": pos["hedge_strike"],
                                    "option_type": opt_type,
                                    "net_credit": round(pos["entry"] - pos["hedge_entry"], 2),
                                    "exit_time": str(m_ts.time())[:8],
                                    "exit_spot": round(m_spot_c, 2),
                                    "exit_reason": exit_reason,
                                    "target_spot": 0.0,
                                    "sl_spot": round(pos["sl"], 2),
                                    "gross_pnl": round(gross_pnl_inr, 2),
                                    "taxes_friction": round(friction["total_friction"], 2),
                                    "net_pnl": round(net_pnl_inr, 2),
                                    "capital_after": round(current_capital, 2),
                                })
                                day_traded = True
                            else:
                                active_positions.append(pos)
                        
                        positions = active_positions
                        
                        if t >= time(15, 25) and not positions:
                            break

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
