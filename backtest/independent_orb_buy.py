"""
Independent 10-year backtest of the NIFTY 15-min ORB option-buying strategy.
(Patched for Realistic Slippage and Intraday Volatility Crush)
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "backtest_data", "nifty_1min_2015_2026.csv")
OUT_TRADES = os.path.join(BASE_DIR, "independent_orb_buy_trades.csv")
OUT_REPORT = os.path.join(BASE_DIR, "independent_orb_buy_report.json")

INITIAL_CAPITAL = 500_000.0
TARGET_R = 2.0
SL_R = 1.0
MIN_RANGE = 40.0
MAX_RANGE = 200.0
STRIKE_STEP = 50
R_RATE = 0.07

# FIXED: Realistic live market slippage (5 points per leg)
SLIPPAGE = 5.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lot_size(d: date) -> int:
    if d < date(2021, 5, 1):
        return 75
    if d < date(2024, 4, 26):
        return 50
    if d < date(2024, 11, 20):
        return 25
    return 65


def get_intraday_iv(base_vix: float, t: time) -> float:
    """
    FIXED: Simulates Intraday Implied Volatility Crush.
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


def bs_price(spot: float, strike: float, t_years: float, sigma: float, opt: str) -> float:
    if t_years <= 1e-6:
        intrinsic = max(0.0, spot - strike) if opt == "CE" else max(0.0, strike - spot)
        return max(0.05, intrinsic)
    sigma = max(sigma, 1e-4)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (R_RATE + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if opt == "CE":
        px = spot * _norm_cdf(d1) - strike * math.exp(-R_RATE * t_years) * _norm_cdf(d2)
    else:
        px = strike * math.exp(-R_RATE * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return max(0.05, px)


def time_to_expiry(ts: pd.Timestamp, expiry: datetime) -> float:
    seconds = (expiry - ts.to_pydatetime()).total_seconds()
    return max(1e-6, seconds / (365.0 * 86400.0))


def buy_friction(entry_px: float, exit_px: float, qty: int) -> float:
    buy_to = entry_px * qty
    sell_to = exit_px * qty
    total_to = buy_to + sell_to
    brokerage = 40.0
    stt = sell_to * 0.001
    exchange = total_to * 0.00035
    sebi = total_to * 1e-6
    stamp = buy_to * 0.00003
    gst = (brokerage + exchange + sebi) * 0.18
    
    # Slippage is factored on quantity (Points slipped * quantity per leg * 2 legs)
    slippage = 2.0 * SLIPPAGE * qty
    return brokerage + stt + exchange + sebi + stamp + gst + slippage


def resample_15m(day: pd.DataFrame) -> pd.DataFrame:
    out = (
        day.set_index("timestamp")
        .resample("15min", origin="start")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), vix=("vix", "last"))
        .dropna()
        .reset_index()
    )
    out["bar_end"] = out["timestamp"] + pd.Timedelta(minutes=14)
    out["tod"] = out["timestamp"].dt.time
    return out


def run() -> dict:
    print(f"[INFO] Loading {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at {DATA_PATH}. Please verify the path.")
        return {"initial_capital": INITIAL_CAPITAL, "final_capital": INITIAL_CAPITAL, "trades": [], "equity": []}
        
    df["date"] = df["timestamp"].dt.date

    capital = INITIAL_CAPITAL
    trades: list[dict] = []
    equity: list[dict] = []

    for trade_date, day in df.groupby("date", sort=True):
        if len(day) < 200:
            equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": 0.0, "traded": 0})
            continue

        day = day.sort_values("timestamp").reset_index(drop=True)
        c15 = resample_15m(day)
        if len(c15) < 6:
            equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": 0.0, "traded": 0})
            continue

        orb = c15.iloc[:4]
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = orb_high - orb_low

        if orb_range < MIN_RANGE or orb_range > MAX_RANGE:
            equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": 0.0, "traded": 0})
            continue

        bull_tgt = orb_high + TARGET_R * orb_range
        bull_sl = orb_high - SL_R * orb_range
        bear_tgt = orb_low - TARGET_R * orb_range
        bear_sl = orb_low + SL_R * orb_range

        signal = None
        sig_row = None
        for _, row in c15.iloc[4:].iterrows():
            if row["tod"] > time(15, 10):
                break
            close = float(row["close"])
            if close > orb_high:
                signal = "BULLISH"
                sig_row = row
                break
            if close < orb_low:
                signal = "BEARISH"
                sig_row = row
                break

        if signal is None:
            equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": 0.0, "traded": 0})
            continue

        entry_ts = sig_row["bar_end"]
        path = day[day["timestamp"] > entry_ts]
        if path.empty:
            equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": 0.0, "traded": 0})
            continue

        entry_spot = float(sig_row["close"])
        opt = "CE" if signal == "BULLISH" else "PE"
        strike = int(round(entry_spot / STRIKE_STEP) * STRIKE_STEP)
        
        # Apply morning IV bump for realistic entry pricing
        base_vix = min(1.0, max(0.08, float(sig_row["vix"]) / 100.0))
        entry_sigma = get_intraday_iv(base_vix, entry_ts.time())
        
        expiry = datetime.combine(trade_date, time(15, 30))
        qty = lot_size(trade_date)
        entry_px = bs_price(entry_spot, strike, time_to_expiry(entry_ts, expiry), entry_sigma, opt)

        tgt = bull_tgt if signal == "BULLISH" else bear_tgt
        sl = bull_sl if signal == "BULLISH" else bear_sl

        exit_reason = "EOD"
        exit_spot = float(path.iloc[-1]["close"])
        exit_ts = path.iloc[-1]["timestamp"]

        highs = path["high"].to_numpy(dtype=float)
        lows = path["low"].to_numpy(dtype=float)
        closes = path["close"].to_numpy(dtype=float)
        stamps = path["timestamp"].to_numpy()

        for i in range(len(path)):
            ts_i = pd.Timestamp(stamps[i])
            tod = ts_i.time()
            hi, lo, cl = highs[i], lows[i], closes[i]

            if tod >= time(15, 25):
                exit_spot, exit_ts, exit_reason = cl, ts_i, "EOD"
                break

            hit_tgt = (signal == "BULLISH" and hi >= tgt) or (signal == "BEARISH" and lo <= tgt)
            hit_sl = (signal == "BULLISH" and lo <= sl) or (signal == "BEARISH" and hi >= sl)
            if hit_tgt and hit_sl:
                exit_spot, exit_ts, exit_reason = sl, ts_i, "STOP_LOSS"
                break
            if hit_sl:
                exit_spot, exit_ts, exit_reason = sl, ts_i, "STOP_LOSS"
                break
            if hit_tgt:
                exit_spot, exit_ts, exit_reason = tgt, ts_i, "TARGET"
                break

        # Apply afternoon IV crush for realistic exit pricing
        exit_sigma = get_intraday_iv(base_vix, pd.Timestamp(exit_ts).time())
        exit_px = bs_price(exit_spot, strike, time_to_expiry(pd.Timestamp(exit_ts), expiry), exit_sigma, opt)
        
        gross = (exit_px - entry_px) * qty
        fric = buy_friction(entry_px, exit_px, qty)
        net = gross - fric
        capital += net

        trades.append({
            "date": str(trade_date),
            "signal": signal,
            "option_type": opt,
            "lots": 1,
            "qty": qty,
            "orb_high": round(orb_high, 2),
            "orb_low": round(orb_low, 2),
            "orb_range": round(orb_range, 2),
            "entry_time": str(pd.Timestamp(entry_ts).time())[:8],
            "entry_spot": round(entry_spot, 2),
            "strike": strike,
            "iv": round(entry_sigma, 4),
            "entry_price": round(entry_px, 2),
            "exit_time": str(pd.Timestamp(exit_ts).time())[:8],
            "exit_spot": round(exit_spot, 2),
            "exit_price": round(exit_px, 2),
            "exit_reason": exit_reason,
            "target_spot": round(tgt, 2),
            "sl_spot": round(sl, 2),
            "gross_pnl": round(gross, 2),
            "friction": round(fric, 2),
            "net_pnl": round(net, 2),
            "capital_after": round(capital, 2),
        })
        equity.append({"date": str(trade_date), "equity": round(capital, 2), "daily_pnl": round(net, 2), "traded": 1})

    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(capital, 2),
        "trades": trades,
        "equity": equity,
    }


def analytics(raw: dict) -> dict:
    if not raw["trades"]:
        return {"summary": {}}
        
    trades = pd.DataFrame(raw["trades"])
    eq = pd.DataFrame(raw["equity"])
    eq["date"] = pd.to_datetime(eq["date"])
    eq["peak"] = eq["equity"].cummax()
    eq["dd_pct"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100.0
    eq["dd_inr"] = eq["equity"] - eq["peak"]

    init, final = raw["initial_capital"], raw["final_capital"]
    days = max(1, (eq["date"].iloc[-1] - eq["date"].iloc[0]).days)
    years = days / 365.25
    cagr = ((final / init) ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0

    w = trades[trades.net_pnl > 0]
    l = trades[trades.net_pnl <= 0]
    gp, gl = float(w.net_pnl.sum()) if len(w) else 0.0, abs(float(l.net_pnl.sum())) if len(l) else 0.0
    pf = (gp / gl) if gl > 0 else 0.0
    avg_w = gp / len(w) if len(w) else 0.0
    avg_l = gl / len(l) if len(l) else 0.0

    cw = cl = mw = ml = 0
    for p in trades.net_pnl:
        if p > 0:
            cw += 1
            cl = 0
            mw = max(mw, cw)
        else:
            cl += 1
            cw = 0
            ml = max(ml, cl)

    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    std = float(eq["ret"].std())
    dstd = float(eq.loc[eq["ret"] < 0, "ret"].std()) if (eq["ret"] < 0).any() else 0.0
    rf = 0.065 / 250.0
    mean = float(eq["ret"].mean())
    sharpe = ((mean - rf) / std * math.sqrt(250)) if std > 0 else 0.0
    sortino = ((mean - rf) / dstd * math.sqrt(250)) if dstd > 0 else 0.0

    trades["date"] = pd.to_datetime(trades["date"])
    trades["year"] = trades["date"].dt.year
    trades["month"] = trades["date"].dt.month
    trades["dow"] = trades["date"].dt.day_name()
    trades["hour"] = trades["entry_time"].str[:2].astype(int)

    yearly = []
    for yr, g in trades.groupby("year"):
        yw, yl = g[g.net_pnl > 0], g[g.net_pnl <= 0]
        ygp = float(yw.net_pnl.sum()) if len(yw) else 0.0
        ygl = abs(float(yl.net_pnl.sum())) if len(yl) else 0.0
        yearly.append({
            "year": int(yr),
            "trades": int(len(g)),
            "win_rate": round(len(yw) / len(g) * 100, 1),
            "net_pnl": round(float(g.net_pnl.sum()), 2),
            "pf": round((ygp / ygl) if ygl else 0.0, 2),
            "targets": int((g.exit_reason == "TARGET").sum()),
            "stops": int((g.exit_reason == "STOP_LOSS").sum()),
            "eod": int((g.exit_reason == "EOD").sum()),
        })

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap = {}
    for yr, g in trades.groupby("year"):
        heatmap[str(int(yr))] = {}
        for i, name in enumerate(months, 1):
            heatmap[str(int(yr))][name] = round(float(g.loc[g.month == i, "net_pnl"].sum()), 2)

    def grp(frame, key):
        rows = []
        for k, g in frame.groupby(key):
            rows.append({
                "key": str(k),
                "trades": int(len(g)),
                "win_rate": round((g.net_pnl > 0).mean() * 100, 1),
                "net_pnl": round(float(g.net_pnl.sum()), 2),
                "avg": round(float(g.net_pnl.mean()), 2),
            })
        return rows

    eq_m = eq.set_index("date").resample("ME").last().dropna()
    equity_monthly = [{"date": str(i)[:7], "equity": round(float(r.equity), 2)} for i, r in eq_m.iterrows()]

    summary = {
        "initial_capital": init,
        "final_capital": final,
        "net_pnl": round(final - init, 2),
        "roi_pct": round((final - init) / init * 100, 2),
        "cagr_pct": round(cagr, 2),
        "trades": int(len(trades)),
        "wins": int(len(w)),
        "losses": int(len(l)),
        "win_rate": round(len(w) / len(trades) * 100, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "payoff": round(avg_w / avg_l, 2) if avg_l else 0.0,
        "expectancy": round(float(trades.net_pnl.mean()), 2),
        "median": round(float(trades.net_pnl.median()), 2),
        "p05": round(float(trades.net_pnl.quantile(0.05)), 2),
        "p95": round(float(trades.net_pnl.quantile(0.95)), 2),
        "max_dd_inr": round(abs(float(eq.dd_inr.min())), 2),
        "max_dd_pct": round(abs(float(eq.dd_pct.min())), 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_wins": mw,
        "max_losses": ml,
        "sessions": int(len(eq)),
        "sessions_traded": int(eq.traded.sum()),
        "gross": round(float(trades.gross_pnl.sum()), 2),
        "friction": round(float(trades.friction.sum()), 2),
        "avg_entry": round(float(trades.entry_price.mean()), 2),
        "avg_iv": round(float(trades.iv.mean()), 4),
    }
    return {
        "summary": summary,
        "yearly": yearly,
        "heatmap": heatmap,
        "equity_monthly": equity_monthly,
        "by_exit": grp(trades, "exit_reason"),
        "by_signal": grp(trades, "signal"),
        "by_hour": grp(trades, "hour"),
        "by_dow": grp(trades, "dow"),
        "best": trades.nlargest(5, "net_pnl")[["date", "signal", "entry_price", "exit_price", "exit_reason", "net_pnl"]]
            .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        "worst": trades.nsmallest(5, "net_pnl")[["date", "signal", "entry_price", "exit_price", "exit_reason", "net_pnl"]]
            .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
    }


def main():
    raw = run()
    stats = analytics(raw)
    
    if not stats.get("summary"):
        print("[ERROR] Analytics failed. Ensure data was processed.")
        return
        
    pd.DataFrame(raw["trades"]).to_csv(OUT_TRADES, index=False)
    with open(OUT_REPORT, "w") as f:
        json.dump({"summary": stats["summary"], **{k: stats[k] for k in stats if k != 'summary'}}, f, default=str)
    
    s = stats["summary"]
    print("=" * 64)
    print("INDEPENDENT ORB OPTION-BUY BACKTEST (REALISTIC FRICTION)")
    print("=" * 64)
    print(f" Capital   {s['initial_capital']:,.0f}  ->  {s['final_capital']:,.2f}")
    print(f" Net P&L   {s['net_pnl']:,.2f}  ({s['roi_pct']:+.2f}%)   CAGR {s['cagr_pct']:.2f}%")
    print(f" Trades    {s['trades']}   Win {s['win_rate']:.1f}%  ({s['wins']}W/{s['losses']}L)")
    print(f" PF        {s['profit_factor']:.2f}   Payoff {s['payoff']:.2f}   Exp {s['expectancy']:.2f}  Med {s['median']:.2f}")
    print(f" Max DD    {s['max_dd_inr']:,.2f} ({s['max_dd_pct']:.2f}%)   Sharpe {s['sharpe']:.2f}  Sortino {s['sortino']:.2f}")
    print(f" Avg IV    {s['avg_iv']:.2%}   Avg premium {s['avg_entry']:.2f}   Friction {s['friction']:,.0f}")
    print("-" * 64)
    for y in stats["yearly"]:
        print(f" {y['year']}  n={y['trades']:<4}  WR {y['win_rate']:5.1f}%  PF {y['pf']:5.2f}  PnL {y['net_pnl']:12,.0f}  T/S/E {y['targets']}/{y['stops']}/{y['eod']}")
    print(f"\n[INFO] trades -> {OUT_TRADES}")
    print(f"[INFO] report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()