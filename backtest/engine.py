"""
Core 10-Year Simulation Engine for NIFTY Intraday Strategies.
Supports:
  1. Option Buying (15-min ORB Breakout, Long ATM)
  2. Range-Bound Selling (Short Strangle/Iron Condor with hedges)
"""

import os
import math
from datetime import datetime, time
from collections import deque
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


# =============================================================================
# Friction Model (single-leg for option buying)
# =============================================================================
BROKERAGE_PER_ORDER = 20.0
STT_SELL_RATE = 0.0015
EXCHANGE_TXN_RATE = 3553.0 / 1_00_00_000
SEBI_FEE_PER_CRORE = 10.0
GST_RATE = 0.18
STAMP_DUTY_BUY_RATE = 0.00003

def calculate_single_leg_charges(premium, quantity, side):
    turnover = premium * quantity
    brokerage = BROKERAGE_PER_ORDER
    stt = math.floor(turnover * STT_SELL_RATE + 0.5) if side == "SELL" else 0.0
    exchange_txn = turnover * EXCHANGE_TXN_RATE
    sebi_fee = (turnover / 1_00_00_000) * SEBI_FEE_PER_CRORE
    gst = (brokerage + exchange_txn + sebi_fee) * GST_RATE
    stamp_duty = math.floor(turnover * STAMP_DUTY_BUY_RATE + 0.5) if side == "BUY" else 0.0
    total = brokerage + stt + exchange_txn + sebi_fee + gst + stamp_duty
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn_charge": exchange_txn,
        "sebi_fee": sebi_fee,
        "gst": gst,
        "stamp_duty": stamp_duty,
        "total_charges": total,
    }


def get_intraday_iv(base_vix: float, t: time) -> float:
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
    strat_mode = strat_cfg.get("mode", "option_buying")  # "option_buying" or "range_bound"
    target_r_multiple = float(strat_cfg.get("target_r_multiple", 2.0))
    sl_r_multiple = float(strat_cfg.get("sl_r_multiple", 1.0))
    short_stop_loss_pct = float(strat_cfg.get("short_stop_loss_pct", 0.20))

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
    target_sold_delta = float(opt_cfg.get("target_sold_abs_delta", 0.50))  # legacy
    short_selection_mode = opt_cfg.get("short_selection_mode", "premium")  # "premium" or "delta"
    short_target_premium = float(opt_cfg.get("short_target_premium", 200.0))
    short_target_premium_mode = opt_cfg.get("short_target_premium_mode", "fixed")  # "fixed" or "spot_pct"
    short_target_premium_pct = float(opt_cfg.get("short_target_premium_pct", 0.8))
    short_target_delta = float(opt_cfg.get("short_target_delta", 0.50))
    target_hedge_delta = float(opt_cfg.get("target_hedge_delta", 0.05))
    use_fixed_offset = bool(opt_cfg.get("use_fixed_strike_offset", False))
    fixed_offset = float(opt_cfg.get("fixed_hedge_offset_pts", 300.0))
    min_net_credit = float(opt_cfg.get("min_net_credit_inr", 5.0))

    friction_cfg = config.get("friction", {})
    slippage_per_unit = float(friction_cfg.get("slippage_per_unit_leg", 0.5))

    sim_cfg = config.get("simulation", {})
    start_year = int(sim_cfg.get("start_year", 2008))
    end_year = int(sim_cfg.get("end_year", 2026))
    
    timing_cfg = config.get("timing", {})
    orb_start_time = datetime.strptime(timing_cfg.get("orb_start_time", "09:15"), "%H:%M").time()
    orb_end_time = datetime.strptime(timing_cfg.get("orb_end_time", "10:15"), "%H:%M").time()
    trade_window_end_time = datetime.strptime(timing_cfg.get("trade_window_end", "15:20"), "%H:%M").time()
    eod_squareoff_time = datetime.strptime(timing_cfg.get("eod_squareoff_time", "15:25"), "%H:%M").time()

    # 2. Data Ingestion
    df_1min = load_nifty_1min_data(start_year=start_year, end_year=end_year)
    df_1min["date"] = df_1min["timestamp"].dt.date

    grouped_days = {d: group for d, group in df_1min.groupby("date")}
    trading_days = sorted(grouped_days.keys())

    current_capital = initial_capital
    trades = []
    daily_equity_series = []

    r_rate = 0.07

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
            orb_candles = day_15m.iloc[:4]
            orb_high = float(orb_candles["high"].max())
            orb_low = float(orb_candles["low"].min())
            orb_range = orb_high - orb_low

            if min_orb_range <= orb_range <= max_orb_range:
                bullish_target = orb_high + (target_r_multiple * orb_range)
                bullish_sl = orb_high - (sl_r_multiple * orb_range)
                bearish_target = orb_low - (target_r_multiple * orb_range)
                bearish_sl = orb_low + (sl_r_multiple * orb_range)

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
                        atm_strike = math.floor(c_close / strike_step + 0.5) * strike_step
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
                        if subsequent_1m.empty:
                            continue
                        exit_time = subsequent_1m.iloc[-1]["timestamp"]
                        exit_spot = c_close
                        exit_reason = "EOD"

                        for _, m_bar in subsequent_1m.iterrows():
                            m_high = float(m_bar["high"])
                            m_low = float(m_bar["low"])
                            m_ts = m_bar["timestamp"]

                            if signal == "BULLISH":
                                if m_low <= sl_spot:
                                    exit_spot = sl_spot
                                    exit_time = m_ts
                                    exit_reason = "STOP_LOSS"
                                    break
                                elif m_high >= target_spot:
                                    exit_spot = target_spot
                                    exit_time = m_ts
                                    exit_reason = "TARGET"
                                    break
                            else:
                                if m_high >= sl_spot:
                                    exit_spot = sl_spot
                                    exit_time = m_ts
                                    exit_reason = "STOP_LOSS"
                                    break
                                elif m_low <= target_spot:
                                    exit_spot = target_spot
                                    exit_time = m_ts
                                    exit_reason = "TARGET"
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

                        charges_buy = calculate_single_leg_charges(entry_price, total_qty, "BUY")
                        charges_sell = calculate_single_leg_charges(exit_price, total_qty, "SELL")
                        slippage_cost = 2 * slippage_per_unit * total_qty
                        total_friction = charges_buy["total_charges"] + charges_sell["total_charges"] + slippage_cost

                        net_pnl_inr = gross_pnl_inr - total_friction
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
                            "taxes_friction": round(total_friction, 2),
                            "net_pnl": round(net_pnl_inr, 2),
                            "capital_after": round(current_capital, 2),
                        })

        # =====================================================================
        # STRATEGY MODE 2: PREMIUM BREAKDOWN (SHORT CE/PE with hedges)
        # Runs independently every day from 09:16 to 15:25
        # =====================================================================
        elif strat_mode == "premium_breakdown":
            # 1. At ORB Start, select short strikes closest to ₹200 premium
            strike_time_end = (datetime.combine(trade_date, orb_start_time) + timedelta(minutes=1)).time()
            bars_start = day_1m[(day_1m["timestamp"].dt.time >= orb_start_time) & (day_1m["timestamp"].dt.time <= strike_time_end)]
            if len(bars_start) == 0:
                continue
            start_bar = bars_start.iloc[0]
            spot_915 = float(start_bar["open"])
            vix_915 = min(1.0, max(0.08, float(start_bar["vix"]) / 100.0))
            expiry_dt = datetime.combine(trade_date, time(15, 30))
            t_915 = get_trading_time_fraction(start_bar["timestamp"], expiry_dt)
            sigma_915 = get_intraday_iv(vix_915, orb_start_time)

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

            # 2. Compute premium ORB range dynamically
            orb_period_mask = (day_1m["timestamp"].dt.time > orb_start_time) & (day_1m["timestamp"].dt.time <= orb_end_time)
            orb_1m = day_1m[orb_period_mask]
            if len(orb_1m) < 10:
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

                ce_px_h = bs_price(m_spot_h, ce_strike, m_t, r_rate, m_sigma, "CE")
                ce_px_l = bs_price(m_spot_l, ce_strike, m_t, r_rate, m_sigma, "CE")
                ce_orb_high = max(ce_orb_high, ce_px_h)
                ce_orb_low = min(ce_orb_low, ce_px_l)

                pe_px_h = bs_price(m_spot_l, pe_strike, m_t, r_rate, m_sigma, "PE")
                pe_px_l = bs_price(m_spot_h, pe_strike, m_t, r_rate, m_sigma, "PE")
                pe_orb_high = max(pe_orb_high, pe_px_h)
                pe_orb_low = min(pe_orb_low, pe_px_l)

            # 3. Lot sizing for premium breakdown (must be defined before monitoring loop)
            contract_lot_size = get_historical_lot_size(trade_date)
            if sizing_mode == "fixed_lots":
                lots = fixed_lots
            else:
                # Use margin_per_lot (note: for spreads, adjust margin accordingly)
                lots = min(max_lots, int(current_capital // margin_per_lot))
            if lots * margin_per_lot > current_capital:
                lots = int(current_capital // margin_per_lot)
            if lots < 1:
                continue
            total_qty = lots * contract_lot_size

            # 4. Monitor from ORB End to EOD for premium breakdowns
            monitor_1m = day_1m[day_1m["timestamp"].dt.time >= orb_end_time]
            if len(monitor_1m) == 0:
                continue

            traded_legs = []
            positions = []
            last_3m_ce_close = 0.0
            last_3m_pe_close = 0.0

            for _, m_bar in monitor_1m.iterrows():
                m_ts = m_bar["timestamp"]
                t = m_ts.time()
                if t > eod_squareoff_time:
                    break

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

                    if t <= trade_window_end_time:
                        if "CE" not in traded_legs and last_3m_ce_close > 0 and last_3m_ce_close < ce_orb_low:
                            if use_fixed_offset:
                                ce_h_strike = ce_strike + fixed_offset
                            else:
                                ce_h_strike = find_delta_hedge_strike(m_spot_c, "CE", m_t, r_rate, m_sigma, target_hedge_delta, strike_step)
                            ce_h_px = bs_price(m_spot_c, ce_h_strike, m_t, r_rate, m_sigma, "CE")
                            if (ce_s_px - ce_h_px) >= min_net_credit:
                                positions.append({
                                    "type": "CE_SHORT",
                                    "entry": ce_s_px,
                                    "sl": ce_s_px * (1 + short_stop_loss_pct),
                                    "strike": ce_strike,
                                    "hedge_strike": ce_h_strike,
                                    "hedge_entry": ce_h_px,
                                    "spot": m_spot_c,
                                    "entry_time": m_ts
                                })
                                traded_legs.append("CE")

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

                active_positions = []
                for pos in positions:
                    opt_type = "CE" if "CE" in pos["type"] else "PE"
                    short_px = ce_s_px if opt_type == "CE" else pe_s_px
                    exit_px = None
                    exit_reason = None

                    if short_px >= pos["sl"]:
                        exit_px = pos["sl"]
                        exit_reason = "STOP_LOSS"
                    elif t >= eod_squareoff_time:
                        exit_px = short_px
                        exit_reason = "EOD"

                    if exit_px is not None:
                        hedge_px = bs_price(m_spot_c, pos["hedge_strike"], m_t, r_rate, m_sigma, opt_type)
                        gross_pnl_inr = ((pos["entry"] - exit_px) + (hedge_px - pos["hedge_entry"])) * total_qty
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

                if t >= eod_squareoff_time and not positions:
                    break
        # =====================================================================
        # STRATEGY MODE 3: RANGE-BOUND SELLING (Short Strangle with hedges)
        # =====================================================================
        else:  # range_bound
            # Wait until 10:15 AM (after ORB range)
            entry_time = time(10, 15)
            bars_at_entry = day_1m[day_1m["timestamp"].dt.time == entry_time]
            if len(bars_at_entry) == 0:
                continue
            entry_bar = bars_at_entry.iloc[0]
            spot_entry = float(entry_bar["open"])
            vix_entry = min(1.0, max(0.08, float(entry_bar["vix"]) / 100.0))
            expiry_dt = datetime.combine(trade_date, time(15, 30))
            t_entry = get_trading_time_fraction(entry_bar["timestamp"], expiry_dt)
            sigma_entry = get_intraday_iv(vix_entry, entry_time)

            # ---- Select short strikes ----
            def get_dynamic_target_premium(spot):
                if short_target_premium_mode == "spot_pct":
                    return spot * (short_target_premium_pct / 100.0)
                return short_target_premium

            def find_strike_by_premium(spot, opt_type, target_prem):
                best_strike = None
                min_diff = 9999.0
                base = math.floor(spot / strike_step + 0.5) * strike_step
                for i in range(-40, 41):
                    k = base + i * strike_step
                    px = bs_price(spot, k, t_entry, r_rate, sigma_entry, opt_type)
                    diff = abs(px - target_prem)
                    if diff < min_diff:
                        min_diff = diff
                        best_strike = k
                return best_strike

            from math import log, sqrt, erf
            def norm_cdf(x):
                return 0.5 * (1 + erf(x / sqrt(2)))
            def bs_delta(spot, strike, T, r, sigma, option_type):
                d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
                if option_type == "CE":
                    return norm_cdf(d1)
                else:
                    return -norm_cdf(-d1)

            def find_strike_by_delta(spot, opt_type, target_delta_abs):
                best_strike = None
                min_diff = 9999.0
                base = math.floor(spot / strike_step + 0.5) * strike_step
                for i in range(-40, 41):
                    k = base + i * strike_step
                    delta = bs_delta(spot, k, t_entry, r_rate, sigma_entry, opt_type)
                    diff = abs(abs(delta) - target_delta_abs)
                    if diff < min_diff:
                        min_diff = diff
                        best_strike = k
                return best_strike

            # Select CE and PE short strikes
            if short_selection_mode == "premium":
                target_prem = get_dynamic_target_premium(spot_entry)
                ce_strike = find_strike_by_premium(spot_entry, "CE", target_prem)
                pe_strike = find_strike_by_premium(spot_entry, "PE", target_prem)
            else:  # delta
                ce_strike = find_strike_by_delta(spot_entry, "CE", short_target_delta)
                pe_strike = find_strike_by_delta(spot_entry, "PE", short_target_delta)

            if not ce_strike or not pe_strike:
                continue

            # ---- Select hedge strikes ----
            if use_fixed_offset:
                ce_h_strike = ce_strike + fixed_offset
                pe_h_strike = pe_strike - fixed_offset
            else:
                ce_h_strike = find_delta_hedge_strike(spot_entry, "CE", t_entry, r_rate, sigma_entry, target_hedge_delta, strike_step)
                pe_h_strike = find_delta_hedge_strike(spot_entry, "PE", t_entry, r_rate, sigma_entry, target_hedge_delta, strike_step)

            # ---- Get option prices at entry ----
            ce_short_px = bs_price(spot_entry, ce_strike, t_entry, r_rate, sigma_entry, "CE")
            pe_short_px = bs_price(spot_entry, pe_strike, t_entry, r_rate, sigma_entry, "PE")
            ce_hedge_px = bs_price(spot_entry, ce_h_strike, t_entry, r_rate, sigma_entry, "CE")
            pe_hedge_px = bs_price(spot_entry, pe_h_strike, t_entry, r_rate, sigma_entry, "PE")

            # Net credit check (optional)
            if (ce_short_px - ce_hedge_px) < min_net_credit or (pe_short_px - pe_hedge_px) < min_net_credit:
                continue

            # ---- Lot sizing ----
            contract_lot_size = get_historical_lot_size(trade_date)
            if sizing_mode == "fixed_lots":
                lots = fixed_lots
            else:
                # Use margin_per_lot for combined position (short strangle)
                lots = min(max_lots, int(current_capital // margin_per_lot))
            if lots * margin_per_lot > current_capital:
                lots = int(current_capital // margin_per_lot)
            if lots < 1:
                continue
            total_qty = lots * contract_lot_size

            # ---- Set stop-loss levels (premium increase) ----
            ce_sl = ce_short_px * (1 + short_stop_loss_pct)
            pe_sl = pe_short_px * (1 + short_stop_loss_pct)

            # ---- Monitor from entry to 15:25 ----
            monitor_1m = day_1m[day_1m["timestamp"] >= entry_bar["timestamp"]]
            exit_time = monitor_1m.iloc[-1]["timestamp"]
            exit_reason = "EOD"
            exit_spot = spot_entry
            exit_ce_px = ce_short_px
            exit_pe_px = pe_short_px

            for _, m_bar in monitor_1m.iterrows():
                m_ts = m_bar["timestamp"]
                t = m_ts.time()
                if t > time(15, 25):
                    break

                m_spot_c = float(m_bar["close"])
                m_t = get_trading_time_fraction(m_ts, expiry_dt)
                m_vix = min(1.0, max(0.08, float(m_bar["vix"]) / 100.0))
                m_sigma = get_intraday_iv(m_vix, t)

                ce_px_now = bs_price(m_spot_c, ce_strike, m_t, r_rate, m_sigma, "CE")
                pe_px_now = bs_price(m_spot_c, pe_strike, m_t, r_rate, m_sigma, "PE")

                # Check stop-loss for either leg
                if ce_px_now >= ce_sl or pe_px_now >= pe_sl:
                    exit_time = m_ts
                    exit_spot = m_spot_c
                    exit_reason = "STOP_LOSS"
                    exit_ce_px = ce_px_now if ce_px_now >= ce_sl else ce_short_px
                    exit_pe_px = pe_px_now if pe_px_now >= pe_sl else pe_short_px
                    break

                # EOD exit
                if t >= time(15, 25):
                    exit_time = m_ts
                    exit_spot = m_spot_c
                    exit_reason = "EOD"
                    exit_ce_px = ce_px_now
                    exit_pe_px = pe_px_now
                    break

            # Calculate P&L for both legs combined
            ce_hedge_px_exit = bs_price(exit_spot, ce_h_strike, get_trading_time_fraction(exit_time, expiry_dt), r_rate, get_intraday_iv(base_vix, exit_time.time()), "CE")
            pe_hedge_px_exit = bs_price(exit_spot, pe_h_strike, get_trading_time_fraction(exit_time, expiry_dt), r_rate, get_intraday_iv(base_vix, exit_time.time()), "PE")

            gross_pnl_ce = ((ce_short_px - exit_ce_px) + (ce_hedge_px_exit - ce_hedge_px)) * total_qty
            gross_pnl_pe = ((pe_short_px - exit_pe_px) + (pe_hedge_px_exit - pe_hedge_px)) * total_qty
            gross_pnl_inr = gross_pnl_ce + gross_pnl_pe

            # Friction for 4 legs (2 shorts + 2 hedges)
            friction = calculate_trade_friction(
                ce_short_px, exit_ce_px, ce_hedge_px, ce_hedge_px_exit, total_qty, friction_cfg
            )
            friction_pe = calculate_trade_friction(
                pe_short_px, exit_pe_px, pe_hedge_px, pe_hedge_px_exit, total_qty, friction_cfg
            )
            total_friction = friction["total_friction"] + friction_pe["total_friction"]

            net_pnl_inr = gross_pnl_inr - total_friction
            current_capital += net_pnl_inr
            daily_pnl += net_pnl_inr
            daily_trades_count += 1

            trades.append({
                "trade_id": len(trades) + 1,
                "date": str(trade_date),
                "strategy": "RANGE_BOUND",
                "signal": "BOTH",
                "lots": lots,
                "qty": total_qty,
                "contract_lot_size": contract_lot_size,
                "orb_high": 0.0,
                "orb_low": 0.0,
                "orb_range": 0.0,
                "entry_time": "10:15:00",
                "entry_spot": round(spot_entry, 2),
                "sold_strike_ce": ce_strike,
                "sold_strike_pe": pe_strike,
                "hedge_strike_ce": ce_h_strike,
                "hedge_strike_pe": pe_h_strike,
                "option_type": "BOTH",
                "net_credit": round((ce_short_px - ce_hedge_px) + (pe_short_px - pe_hedge_px), 2),
                "exit_time": str(exit_time.time())[:8],
                "exit_spot": round(exit_spot, 2),
                "exit_reason": exit_reason,
                "target_spot": 0.0,
                "sl_spot_ce": round(ce_sl, 2),
                "sl_spot_pe": round(pe_sl, 2),
                "gross_pnl": round(gross_pnl_inr, 2),
                "taxes_friction": round(total_friction, 2),
                "net_pnl": round(net_pnl_inr, 2),
                "capital_after": round(current_capital, 2),
            })

        # End of day record
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