#!/usr/bin/env python3
"""
Black-Scholes Simulated Backtester for Hedged Premium Breakdown Strategy
Requires Nifty Spot 1-min data with VIX.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import csv
import os

# --- CONFIGURATION ---
DATA_FILE = "../BT/backtest_data/nifty_1min_2015_2026.csv"
OUTPUT_FILE = "premium_backtest_trades.csv"

SHORT_TARGET = 200.0
HEDGE_TARGET = 5.0
STOP_LOSS_PCT = 0.20
STRIKE_STEP = 50

# --- BLACK-SCHOLES PRICING ---
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_price(S, K, T, r, sigma, option_type="CE"):
    """
    S: Spot
    K: Strike
    T: Time to expiry in years
    r: Risk-free rate (e.g. 0.05 for 5%)
    sigma: Volatility (VIX / 100)
    """
    if T <= 0:
        return max(0, S - K) if option_type == "CE" else max(0, K - S)
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def get_dte_years(date_obj, current_time_str):
    # Find next Thursday
    days_ahead = 3 - date_obj.weekday()
    if days_ahead < 0:
        days_ahead += 7
    
    expiry_date = date_obj + timedelta(days=days_ahead)
    
    # Calculate exact time remaining (assuming 15:30 expiry time)
    current_dt = datetime.strptime(f"{date_obj.date()} {current_time_str}", "%Y-%m-%d %H:%M:%S")
    expiry_dt = datetime.strptime(f"{expiry_date.date()} 15:30:00", "%Y-%m-%d %H:%M:%S")
    
    delta = expiry_dt - current_dt
    years = delta.total_seconds() / (365.25 * 24 * 3600)
    return max(0.0001, years)

def find_target_strike(spot, vix, dte_years, option_type, target_premium):
    atm = round(spot / STRIKE_STEP) * STRIKE_STEP
    best_strike = atm
    min_diff = float("inf")
    best_premium = 0
    
    # scan +/- 50 strikes (2500 points)
    for i in range(-50, 51):
        strike = atm + (i * STRIKE_STEP)
        premium = bs_price(spot, strike, dte_years, 0.05, vix / 100.0, option_type)
        diff = abs(premium - target_premium)
        if diff < min_diff:
            min_diff = diff
            best_strike = strike
            best_premium = premium
            
    return best_strike, best_premium

# --- BACKTEST ENGINE ---
def run_backtest():
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Data file not found at {DATA_FILE}. Please ensure it exists.")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')
    
    grouped = df.groupby('date')
    
    trades = []
    
    print("Starting simulation...")
    
    for date, day_df in grouped:
        day_df = day_df.set_index('time')
        
        # 1. 09:15 Setup
        if '09:15:00' not in day_df.index:
            continue
            
        setup_row = day_df.loc['09:15:00']
        spot_0915 = setup_row['close']
        vix_0915 = setup_row['vix']
        dte_years = get_dte_years(pd.Timestamp(date), '09:15:00')
        
        ce_short_strike, ce_s_prem = find_target_strike(spot_0915, vix_0915, dte_years, "CE", SHORT_TARGET)
        pe_short_strike, pe_s_prem = find_target_strike(spot_0915, vix_0915, dte_years, "PE", SHORT_TARGET)
        
        # 2. Range Tracking 09:16 to 10:15
        ce_low = ce_s_prem
        ce_high = ce_s_prem
        pe_low = pe_s_prem
        pe_high = pe_s_prem
        
        tracking_mask = (day_df.index >= '09:16:00') & (day_df.index <= '10:15:00')
        tracking_df = day_df[tracking_mask]
        
        if tracking_df.empty:
            continue
            
        for time_str, row in tracking_df.iterrows():
            spot = row['close']
            vix = row['vix']
            t = get_dte_years(pd.Timestamp(date), time_str)
            
            ce_p = bs_price(spot, ce_short_strike, t, 0.05, vix/100, "CE")
            pe_p = bs_price(spot, pe_short_strike, t, 0.05, vix/100, "PE")
            
            ce_low = min(ce_low, ce_p)
            ce_high = max(ce_high, ce_p)
            pe_low = min(pe_low, pe_p)
            pe_high = max(pe_high, pe_p)
            
        # 3. 10:15 Hedge Setup
        if '10:15:00' not in day_df.index:
            continue
            
        hedge_row = day_df.loc['10:15:00']
        spot_1015 = hedge_row['close']
        vix_1015 = hedge_row['vix']
        dte_1015 = get_dte_years(pd.Timestamp(date), '10:15:00')
        
        ce_hedge_strike, _ = find_target_strike(spot_1015, vix_1015, dte_1015, "CE", HEDGE_TARGET)
        pe_hedge_strike, _ = find_target_strike(spot_1015, vix_1015, dte_1015, "PE", HEDGE_TARGET)
        
        # 4. Execution Loop (10:15 to 15:15, 3-min steps)
        exec_mask = (day_df.index > '10:15:00') & (day_df.index <= '15:15:00')
        exec_df = day_df[exec_mask]
        
        active_ce = False
        active_pe = False
        ce_pos = {}
        pe_pos = {}
        
        for time_str, row in exec_df.iterrows():
            minute = int(time_str.split(':')[1])
            if minute % 3 != 0:
                continue
                
            spot = row['close']
            vix = row['vix']
            t = get_dte_years(pd.Timestamp(date), time_str)
            
            # Current Prices
            ce_s_curr = bs_price(spot, ce_short_strike, t, 0.05, vix/100, "CE")
            pe_s_curr = bs_price(spot, pe_short_strike, t, 0.05, vix/100, "PE")
            ce_h_curr = bs_price(spot, ce_hedge_strike, t, 0.05, vix/100, "CE")
            pe_h_curr = bs_price(spot, pe_hedge_strike, t, 0.05, vix/100, "PE")
            
            # Check CE Entry
            if not active_ce and ce_s_curr < ce_low:
                # Sequence: Buy Hedge, Sell Short
                active_ce = True
                ce_pos = {
                    "entry_time": time_str,
                    "h_entry": ce_h_curr,
                    "s_entry": ce_s_curr,
                    "sl": ce_s_curr * (1.0 + STOP_LOSS_PCT)
                }
                
            # Check PE Entry
            if not active_pe and pe_s_curr < pe_low:
                active_pe = True
                pe_pos = {
                    "entry_time": time_str,
                    "h_entry": pe_h_curr,
                    "s_entry": pe_s_curr,
                    "sl": pe_s_curr * (1.0 + STOP_LOSS_PCT)
                }
                
            # Check SL
            if active_ce:
                if ce_s_curr >= ce_pos["sl"]:
                    # SL Hit
                    active_ce = False
                    trades.append({
                        "date": date, "leg": "CE", "entry_time": ce_pos["entry_time"], "exit_time": time_str,
                        "h_entry": ce_pos["h_entry"], "s_entry": ce_pos["s_entry"],
                        "s_exit": ce_s_curr, "h_exit": ce_h_curr, "exit_reason": "SL_HIT"
                    })
                    
            if active_pe:
                if pe_s_curr >= pe_pos["sl"]:
                    # SL Hit
                    active_pe = False
                    trades.append({
                        "date": date, "leg": "PE", "entry_time": pe_pos["entry_time"], "exit_time": time_str,
                        "h_entry": pe_pos["h_entry"], "s_entry": pe_pos["s_entry"],
                        "s_exit": pe_s_curr, "h_exit": pe_h_curr, "exit_reason": "SL_HIT"
                    })
                    
        # EOD Exit
        if active_ce:
            eod_row = day_df.loc['15:15:00'] if '15:15:00' in day_df.index else exec_df.iloc[-1]
            t = get_dte_years(pd.Timestamp(date), '15:15:00')
            ce_s_curr = bs_price(eod_row['close'], ce_short_strike, t, 0.05, eod_row['vix']/100, "CE")
            ce_h_curr = bs_price(eod_row['close'], ce_hedge_strike, t, 0.05, eod_row['vix']/100, "CE")
            trades.append({
                "date": date, "leg": "CE", "entry_time": ce_pos["entry_time"], "exit_time": "15:15:00",
                "h_entry": ce_pos["h_entry"], "s_entry": ce_pos["s_entry"],
                "s_exit": ce_s_curr, "h_exit": ce_h_curr, "exit_reason": "EOD"
            })
            
        if active_pe:
            eod_row = day_df.loc['15:15:00'] if '15:15:00' in day_df.index else exec_df.iloc[-1]
            t = get_dte_years(pd.Timestamp(date), '15:15:00')
            pe_s_curr = bs_price(eod_row['close'], pe_short_strike, t, 0.05, eod_row['vix']/100, "PE")
            pe_h_curr = bs_price(eod_row['close'], pe_hedge_strike, t, 0.05, eod_row['vix']/100, "PE")
            trades.append({
                "date": date, "leg": "PE", "entry_time": pe_pos["entry_time"], "exit_time": "15:15:00",
                "h_entry": pe_pos["h_entry"], "s_entry": pe_pos["s_entry"],
                "s_exit": pe_s_curr, "h_exit": pe_h_curr, "exit_reason": "EOD"
            })
            
    print(f"Simulation complete! {len(trades)} trades generated.")
    
    # Save and Analyze
    res_df = pd.DataFrame(trades)
    if res_df.empty:
        print("No trades triggered.")
        return
        
    # PnL Calculation (Short PnL = Entry - Exit, Hedge PnL = Exit - Entry)
    res_df["short_pnl"] = res_df["s_entry"] - res_df["s_exit"]
    res_df["hedge_pnl"] = res_df["h_exit"] - res_df["h_entry"]
    res_df["net_pnl"] = res_df["short_pnl"] + res_df["hedge_pnl"]
    
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Trades exported to {OUTPUT_FILE}")
    
    # Metrics
    wins = len(res_df[res_df["net_pnl"] > 0])
    losses = len(res_df[res_df["net_pnl"] <= 0])
    win_rate = (wins / len(res_df)) * 100
    avg_win = res_df[res_df["net_pnl"] > 0]["net_pnl"].mean()
    avg_loss = res_df[res_df["net_pnl"] <= 0]["net_pnl"].mean()
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    
    print("-" * 40)
    print("BACKTEST RESULTS (Black-Scholes Simulated)")
    print("-" * 40)
    print(f"Total Trades : {len(res_df)}")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Avg Win      : {avg_win:.2f} pts")
    print(f"Avg Loss     : {avg_loss:.2f} pts")
    print(f"Expectancy   : {expectancy:.2f} pts/trade")
    print(f"Total Net PnL: {res_df['net_pnl'].sum():.2f} pts")
    print("-" * 40)

if __name__ == "__main__":
    run_backtest()
