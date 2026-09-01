"""
Core 10-Year Simulation Engine for NIFTY 15-Minute ORB Strategies.
Supports both Option Buying (Long ATM CE/PE) and Option Selling (Credit Spreads).
Corrected version: independent strategy execution, correct stop-loss, accurate ORB range, proper daily P&L aggregation.
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
        return base_vix * 1.15
    elif t < time(12, 0):
        return base_vix * 1.05
    elif t < time(14, 0):
        return base_vix * 0.95
    else:
        return base_vix * 0.90


def run_backtest_simulation(config):
    """
    Executes the 10-year simulation loop across historical 1-minute NIFTY data.
    """
    # 1. Config Extraction
    strat_cfg = config.get("strategy", {})
    strat_mode = strat_cfg.get("mode", "option_buying")
    target_r_multiple = float(strat_cfg.get("target_r_multiple", 2.0))
    sl_r_multiple = float(strat_cfg.get("sl_r_multiple", 1.0))
    short_stop_loss_pct = float(strat_cfg.get("short_stop_loss_pct", 0.20))  # New: for premium breakdown

    cap_cfg = config.get("capital", {})
    initial_capital = float(cap_cfg.get("initial_capital", 500000.0))
    sizing_mode = cap_cfg.get("sizing_mode", "fixed_lots")
    fixed_lots = int(cap_cfg.get("fixed_lots", 1))
    max_lots = int(cap_cfg.get("max_lots", 10))
    margin_per_lot = float(cap_cfg.get("margin_per_lot", 60000.0))  # For spreads, this should be adjusted elsewhere

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

    grouped_days = {d: group for d, group in df_1min.groupby("date")}
    trading_days = sorted(grouped_days.keys())

    current_capital = initial_capital
    trades = []
    daily_equity_series = []

    r_rate = 0.07  # Risk-free rate

    for trade_date in trading_days:
        day_1m = grouped_days[trade_date]
        if len(day_1m) < 200:
            continue

        day_15m = resample_to_15min_candles(day_1m)
        if len(day_15m) < 5:
            continue

        base_vix = min(1.0, max(0.08, float(day_1m.iloc[0]["vix"]) / 100.0))

        daily_pnl = 0.0
        daily_trades_count = 0

        # =====================================================================
        # STRATEGY MODE 1: OPTION BUYING (LONG ATM CE/PE) using 15-min ORB
        # =====================================================================
        if strat_mode == "option_buying":
            # 1-Hour ORB Window (First four 15-min candles: 09:15-10:15)
            orb_candles = day_15m.iloc[:4]
            orb_high = float(orb_candles["high"].max())
            orb_low = float(orb_candles["low"].min())
            orb_range = orb_high - orb_low

            # Range filter
            if min_orb_range <= orb_range <= max_orb_range:
                bullish_target = orb_high + (target_r_multiple * orb_range)
                bullish_sl = orb_high - (sl_r_multiple * orb_range)
                bearish_target = orb_low - (target_r_multiple * orb_range)
                bearish_sl = orb_low + (sl_r_multiple * orb_range)

                # Track if we already traded a direction today
                bullish_traded = False
                bearish_traded = False

                post_orb_candles = day_15m.iloc[4:]
                for idx, candle in post_orb_candles.iterrows():
                    c_time = candle["time"]
                    c_close = float(candle["close"])
                    c_ts = candle["timestamp"]

                    if c_time > time(15, 10):
                        break

                    signal = None
                    if c_close > orb_high and not bullish_traded:
                        signal = "BULLISH"
                        bullish_traded = True
                    elif c_close < orb_low and not bearish_traded:
                        signal = "BEARISH"
                        bearish_traded = True

                    if signal:
                        contract_lot_size = get_historical_lot_size(trade_date)
                        atm_strike = round(c_close / strike_step) * strike_step
                        expiry_dt = datetime.combine(trade_date, time(15, 30))
                        t_years = get_trading_time_fraction(c_ts, expiry_dt)

                        target_spot = bullish_target if signal == "BULLISH" else bearish_target
                        sl_spot = bullish_sl if signal == "BULLISH" else bearish_sl

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
                            continue

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

                        # Friction for option buying (single leg, buy and sell)
                        friction_cost = 40.0 + (0.001 * entry_price * total_qty) + (2 * float(friction_cfg.get("slippage_per_unit_leg", 0.5)) * total_qty)
                        net_pnl_inr = gross_pnl_inr - friction_cost
                        current_capital += net_pnl_inr
                        daily_pnl += net_pnl_inr
                        daily_trades_count += 1

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

        # =====================================================================
        # STRATEGY MODE 2: PREMIUM BREAKDOWN (SHORT CE/PE with hedges)
        # Runs independently every day from 09:16 to 15:25
        # =====================================================================
        else:  # premium_breakdown
            # 1. At 09:16 AM, select short strikes closest to ₹200 premium
            # Get the 09:16 bar (or first available after 09:15)
            bars_915_916 = day_1m[(day_1m["timestamp"].dt.time >= time(9, 15)) & (day_1m["timestamp"].dt.time <= time(9, 16))]
            if len(bars_915_916) == 0:
                continue
            start_bar = bars_915_916.iloc[0]
            spot_915 = float(start_bar["open"])
            vix_915 = min(1.0, max(0.08, float(start_bar["vix"]) / 100.0))
            expiry_dt = datetime.combine(trade_date, time(15, 30))
            t_915 = get_trading_time_fraction(start_bar["timestamp"], expiry_dt)
            sigma_915 = get_intraday_iv(vix_915, time(9, 15))

            def get_200_strike(spot, opt_type):
                best_strike = None
                min_diff = 9999
                base = round(spot / strike_step) * strike_step
                for i in range(-40, 41):
                    k = base + (i * strike_step)
                    px = bs_price(spot, k, t_915, r_rate, sigma_915, opt_type)
                    if abs(px - 200) < min_diff:
                        min_diff = abs(px - 200)
                        best_strike = k
                return best_strike

            ce_strike = get_200_strike(spot_915, "CE")
            pe_strike = get_200_strike(spot_915, "PE")
            if not ce_strike or not pe_strike:
                continue

            # 2. Compute premium ORB range using 09:16 to 10:15 data (exactly)
            orb_period_mask = (day_1m["timestamp"].dt.time >= time(9, 16)) & (day_1m["timestamp"].dt.time <= time(10, 15))
            orb_1m = day_1m[orb_period_mask]
            if len(orb_1m) < 10:  # Need enough data
                continue

            ce_orb_high = 0.0
            ce_orb_low = 9999.0
            pe_orb_high = 0.0
            pe_orb_low = 9999.0

            for _, m_bar in orb_1m.iterrows():
                m_spot_h = float(m_bar["high"])
                m_spot_l = float(m_bar["low"])
                m_ts = m_bar["timestamp"]
                m_t = get_trading_time_fraction(m_ts, expiry_dt)
                m_vix = min(1.0, max(0.08, float(m_bar["vix"]) / 100.0))
                m_sigma = get_intraday_iv(m_vix, m_ts.time())

                # CE premium: high when spot high, low when spot low
                ce_px_h = bs_price(m_spot_h, ce_strike, m_t, r_rate, m_sigma, "CE")
                ce_px_l = bs_price(m_spot_l, ce_strike, m_t, r_rate, m_sigma, "CE")
                ce_orb_high = max(ce_orb_high, ce_px_h)
                ce_orb_low = min(ce_orb_low, ce_px_l)

                # PE premium: high when spot low, low when spot high
                pe_px_h = bs_price(m_spot_l, pe_strike, m_t, r_rate, m_sigma, "PE")
                pe_px_l = bs_price(m_spot_h, pe_strike, m_t, r_rate, m_sigma, "PE")
                pe_orb_high = max(pe_orb_high, pe_px_h)
                pe_orb_low = min(pe_orb_low, pe_px_l)

            # 3. Monitor from 10:15 to 15:25 for premium breakdowns
            monitor_1m = day_1m[day_1m["timestamp"].dt.time >= time(10, 15)]
            if len(monitor_1m) == 0:
                continue

            traded_legs = []
            positions = []  # list of active positions (short+hedge pair)
            last_3m_ce_close = 0.0
            last_3m_pe_close = 0.0

            for _, m_bar in monitor_1m.iterrows():
                m_ts = m_bar["timestamp"]
                t = m_ts.time()
                if t > time(15, 25):
                    break

                m_spot_h = float(m_bar["high"])
                m_spot_l = float(m_bar["low"])
                m_spot_c = float(m_bar["close"])
                m_t = get_trading_time_fraction(m_ts, expiry_dt)
                m_vix = min(1.0, max(0.08, float(m_bar["vix"]) / 100.0))
                m_sigma = get_intraday_iv(m_vix, t)

                # Option prices at close for current minute
                ce_s_px = bs_price(m_spot_c, ce_strike, m_t, r_rate, m_sigma, "CE")
                pe_s_px = bs_price(m_spot_c, pe_strike, m_t, r_rate, m_sigma, "PE")

                # Check for 3-minute boundary using exact time
                if t.minute % 3 == 0:
                    last_3m_ce_close = ce_s_px
                    last_3m_pe_close = pe_s_px

                    # Entry only before trade window ends (15:20)
                    if t <= time(15, 20):
                        # CE breakdown
                        if "CE" not in traded_legs and last_3m_ce_close > 0 and last_3m_ce_close < ce_orb_low:
                            # Determine hedge strike
                            if use_fixed_offset:
                                ce_h_strike = ce_strike + fixed_offset
                            else:
                                ce_h_strike = find_delta_hedge_strike(m_spot_c, "CE", m_t, r_rate, m_sigma, target_hedge_delta, strike_step)
                            ce_h_px = bs_price(m_spot_c, ce_h_strike, m_t, r_rate, m_sigma, "CE")
                            # Ensure net credit positive (optional filter)
                            if (ce_s_px - ce_h_px) >= min_net_credit:
                                positions.append({
                                    "type": "CE_SHORT",
                                    "entry": ce_s_px,
                                    "sl": ce_s_px * (1 + short_stop_loss_pct),  # 20% above entry
                                    "strike": ce_strike,
                                    "hedge_strike": ce_h_strike,
                                    "hedge_entry": ce_h_px,
                                    "spot": m_spot_c,
                                    "entry_time": m_ts
                                })
                                traded_legs.append("CE")

                        # PE breakdown
                        if "PE" not in traded_legs and last_3m_pe_close > 0 and last_3m_pe_close < pe_orb_low:
                            if use_fixed_offset:
                                pe_h_strike = pe_strike - fixed_offset
                            else:
                                pe_h_strike = find_delta_hedge_strike(m_spot_c, "PE", m_t, r_rate, m_sigma, target_hedge_delta, strike_step)
                            pe_h_px = bs_price(m_spot_c, pe_h_strike, m_t, r_rate, m_sigma, "PE")
                            if (pe_s_px - pe_h_px) >= min_net_credit:
                                positions.append({
                                    "type": "PE_SHORT",
                                    "entry": pe_s_px,
                                    "sl": pe_s_px * (1 + short_stop_loss_pct),
                                    "strike": pe_strike,
                                    "hedge_strike": pe_h_strike,
                                    "hedge_entry": pe_h_px,
                                    "spot": m_spot_c,
                                    "entry_time": m_ts
                                })
                                traded_legs.append("PE")

                # Check SL or EOD for active positions
                active_positions = []
                for pos in positions:
                    opt_type = "CE" if "CE" in pos["type"] else "PE"
                    # Current short option price (use close for simplicity)
                    short_px = ce_s_px if opt_type == "CE" else pe_s_px
                    # Check SL
                    exit_px = None
                    exit_reason = None

                    if short_px >= pos["sl"]:
                        exit_px = pos["sl"]
                        exit_reason = "STOP_LOSS"
                    elif t >= time(15, 25):
                        exit_px = short_px
                        exit_reason = "EOD"

                    if exit_px is not None:
                        # Determine hedge exit price (current)
                        hedge_px = bs_price(m_spot_c, pos["hedge_strike"], m_t, r_rate, m_sigma, opt_type)
                        # Gross P&L: short profit = entry - exit, hedge loss = hedge_exit - hedge_entry
                        gross_pnl_inr = ((pos["entry"] - exit_px) + (hedge_px - pos["hedge_entry"])) * total_qty
                        # Friction for two legs (buy/sell short + sell/buy hedge)
                        friction = calculate_trade_friction(pos["entry"], exit_px, pos["hedge_entry"], hedge_px, total_qty, friction_cfg)
                        net_pnl_inr = gross_pnl_inr - friction["total_friction"]
                        current_capital += net_pnl_inr
                        daily_pnl += net_pnl_inr
                        daily_trades_count += 1

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
                            "entry_time": str(pos["entry_time"].time())[:8],
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
                    else:
                        active_positions.append(pos)

                positions = active_positions

                if t >= time(15, 25) and not positions:
                    break

        # End of day: record equity
        daily_equity_series.append({
            "date": str(trade_date),
            "equity": round(current_capital, 2),
            "daily_pnl": round(daily_pnl, 2),
            "trades_count": daily_trades_count
        })

    return {
        "initial_capital": initial_capital,
        "final_capital": round(current_capital, 2),
        "total_trades": len(trades),
        "trades": trades,
        "equity_series": daily_equity_series,
        "config": config,
    }