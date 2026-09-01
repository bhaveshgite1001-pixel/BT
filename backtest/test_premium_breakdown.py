import pandas as pd
from datetime import time, datetime
import math
import sys
import os
import csv
sys.path.append("/Users/tusharpatil/Downloads/BT")
from backtest.option_pricing import bs_price
from backtest.data_loader import load_nifty_1min_data

BASE_DIR = "/Users/tusharpatil/Downloads/BT"
CSV_FILE = os.path.join(BASE_DIR, "premium_trade_log.csv")

print("Loading NIFTY 1-min data for 2023...")
df_1min = load_nifty_1min_data(2023, 2023)
df_1min["date"] = df_1min["timestamp"].dt.date
grouped_days = {d: group for d, group in df_1min.groupby("date")}
print(f"Loaded {len(grouped_days)} trading days.")

r_rate = 0.07

def get_target_strike(spot, opt_type, target_premium, vix, days_to_expiry):
    best_strike = None
    min_diff = 9999
    base = round(spot/50)*50
    for i in range(-50, 51):
        k = base + (i*50)
        px = bs_price(spot, k, days_to_expiry/365.0, r_rate, vix, opt_type)
        if abs(px - target_premium) < min_diff:
            min_diff = abs(px - target_premium)
            best_strike = k
    return best_strike

# Initialize CSV for Dashboard
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "date", "entry_time", "exit_time", "symbol", "qty",
        "entry_premium", "exit_premium", "exit_reason",
        "net_pnl", "breakout_direction"
    ])

daily_results = []
total_pnl = 0

for trade_date, day_1m in grouped_days.items():
    if len(day_1m) < 300:
        continue  # skip incomplete days

    start_bar = day_1m.iloc[0]
    spot_915 = start_bar["open"]
    vix_915 = min(1.0, max(0.08, float(start_bar["vix"]) / 100.0))
    days_to_expiry = 1.8  # Assume 1.8 days to expiry on average for simulation

    ce_short_strike = get_target_strike(spot_915, "CE", 200, vix_915, days_to_expiry)
    pe_short_strike = get_target_strike(spot_915, "PE", 200, vix_915, days_to_expiry)

    ce_high, ce_low = 0, float('inf')
    pe_high, pe_low = 0, float('inf')

    for _, row in day_1m.iterrows():
        t = row["timestamp"].time()
        if t >= time(10, 15):
            break

        spot = row["open"]
        vix = min(1.0, max(0.08, float(row["vix"]) / 100.0))
        ce_px = bs_price(spot, ce_short_strike, days_to_expiry/365.0, r_rate, vix, "CE")
        pe_px = bs_price(spot, pe_short_strike, days_to_expiry/365.0, r_rate, vix, "PE")

        ce_high = max(ce_high, ce_px)
        ce_low = min(ce_low, ce_px)
        pe_high = max(pe_high, pe_px)
        pe_low = min(pe_low, pe_px)

    bar_1015 = day_1m[day_1m["timestamp"].dt.time == time(10, 15)]
    if bar_1015.empty:
        continue
    spot_1015 = bar_1015.iloc[0]["open"]
    vix_1015 = min(1.0, max(0.08, float(bar_1015.iloc[0]["vix"]) / 100.0))

    ce_hedge_strike = get_target_strike(spot_1015, "CE", 5, vix_1015, days_to_expiry)
    pe_hedge_strike = get_target_strike(spot_1015, "PE", 5, vix_1015, days_to_expiry)

    traded_legs = []
    positions = []
    daily_pnl = 0

    last_3m_ce_close = 0
    last_3m_pe_close = 0

    for _, row in day_1m.iterrows():
        t = row["timestamp"].time()
        if t <= time(10, 15):
            continue
            
        timestamp_str = t.strftime("%H:%M:%S")

        spot = row["close"]
        vix = min(1.0, max(0.08, float(row["vix"]) / 100.0))
        
        ce_s_px = bs_price(spot, ce_short_strike, days_to_expiry/365.0, r_rate, vix, "CE")
        pe_s_px = bs_price(spot, pe_short_strike, days_to_expiry/365.0, r_rate, vix, "PE")
        ce_h_px = bs_price(spot, ce_hedge_strike, days_to_expiry/365.0, r_rate, vix, "CE")
        pe_h_px = bs_price(spot, pe_hedge_strike, days_to_expiry/365.0, r_rate, vix, "PE")

        if t.minute % 3 == 0:
            last_3m_ce_close = ce_s_px
            last_3m_pe_close = pe_s_px

            if t <= time(15, 20):
                if "CE" not in traded_legs and last_3m_ce_close > 0 and last_3m_ce_close < ce_low:
                    traded_legs.append("CE")
                    positions.append({"type": "CE_SHORT", "entry": ce_s_px, "sl": ce_s_px * 1.20, "time": timestamp_str, "qty": 50, "symbol": f"CE_{ce_short_strike}"})
                    positions.append({"type": "CE_HEDGE", "entry": ce_h_px, "sl": None, "time": timestamp_str, "qty": 50, "symbol": f"CE_{ce_hedge_strike}"})

                if "PE" not in traded_legs and last_3m_pe_close > 0 and last_3m_pe_close < pe_low:
                    traded_legs.append("PE")
                    positions.append({"type": "PE_SHORT", "entry": pe_s_px, "sl": pe_s_px * 1.20, "time": timestamp_str, "qty": 50, "symbol": f"PE_{pe_short_strike}"})
                    positions.append({"type": "PE_HEDGE", "entry": pe_h_px, "sl": None, "time": timestamp_str, "qty": 50, "symbol": f"PE_{pe_hedge_strike}"})

        active_positions = []
        sl_hit_types = []
        for pos in positions:
            if "SHORT" in pos["type"]:
                current_px = ce_s_px if "CE" in pos["type"] else pe_s_px
                if current_px >= pos["sl"]:
                    sl_hit_types.append("CE" if "CE" in pos["type"] else "PE")
                    pnl_pts = pos["entry"] - current_px
                    pnl_rs = pnl_pts * pos["qty"]
                    daily_pnl += pnl_pts 
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([
                            trade_date.isoformat(), pos["time"], timestamp_str, pos["symbol"], pos["qty"],
                            round(pos["entry"], 2), round(current_px, 2), "SL_HIT", round(pnl_rs, 2), "CE" if "CE" in pos["type"] else "PE"
                        ])
                else:
                    active_positions.append(pos)
            else:
                active_positions.append(pos)

        remaining = []
        for pos in active_positions:
            if "HEDGE" in pos["type"]:
                leg_type = "CE" if "CE" in pos["type"] else "PE"
                if leg_type in sl_hit_types:
                    current_px = ce_h_px if leg_type == "CE" else pe_h_px
                    pnl_pts = current_px - pos["entry"]
                    pnl_rs = pnl_pts * pos["qty"]
                    daily_pnl += pnl_pts
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([
                            trade_date.isoformat(), pos["time"], timestamp_str, pos["symbol"], pos["qty"],
                            round(pos["entry"], 2), round(current_px, 2), "HEDGE_EXIT", round(pnl_rs, 2), leg_type
                        ])
                else:
                    remaining.append(pos)
            else:
                remaining.append(pos)

        positions = remaining

        if t >= time(15, 25) and positions:
            for pos in positions:
                leg_type = "CE" if "CE" in pos["type"] else "PE"
                if "SHORT" in pos["type"]:
                    current_px = ce_s_px if leg_type == "CE" else pe_s_px
                    pnl_pts = pos["entry"] - current_px
                else:
                    current_px = ce_h_px if leg_type == "CE" else pe_h_px
                    pnl_pts = current_px - pos["entry"]
                
                pnl_rs = pnl_pts * pos["qty"]
                daily_pnl += pnl_pts
                with open(CSV_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([
                        trade_date.isoformat(), pos["time"], timestamp_str, pos["symbol"], pos["qty"],
                        round(pos["entry"], 2), round(current_px, 2), "EOD_EXIT", round(pnl_rs, 2), leg_type
                    ])
            positions = []
            break

    daily_pnl_rupees = daily_pnl * 50
    total_pnl += daily_pnl_rupees
    daily_results.append({"date": trade_date, "pnl": daily_pnl_rupees})
    print(f"[{trade_date}] CE: {ce_short_strike} (L:{ce_low:.1f}) | PE: {pe_short_strike} (L:{pe_low:.1f}) | Traded: {traded_legs} | PnL: ₹{daily_pnl_rupees:.2f}")

print("=" * 60)
print(f"=== BACKTEST COMPLETE (2023) ===")
print(f"Total Traded Days  : {len(daily_results)}")
win_days = [r for r in daily_results if r['pnl'] > 0]
loss_days = [r for r in daily_results if r['pnl'] < 0]
print(f"Win Days           : {len(win_days)}")
print(f"Loss Days          : {len(loss_days)}")
print(f"Total PnL (Rupees) : ₹{total_pnl:,.2f}")
print("=" * 60)
