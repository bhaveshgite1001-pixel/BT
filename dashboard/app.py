#!/usr/bin/env python3
"""
NIFTY ORB Backtesting Terminal
A Flask-based web dashboard to visualize backtest trades.
"""

import os
import sys
import csv
from collections import defaultdict
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_FILES = {
    "index": {
        "trade": os.path.join(BASE_DIR, "trade_log.csv"),
    },
    "premium": {
        "trade": os.path.join(BASE_DIR, "premium_trade_log.csv"),
    }
}

def get_s_files():
    strategy = request.args.get("strategy", "premium")
    return STRATEGY_FILES.get(strategy, STRATEGY_FILES["premium"])

# ── Utilities ──────────────────────────────────────────────────────────────────
def read_trades():
    csv_file = get_s_files()["trade"]
    trades = []
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
        except Exception:
            pass
    return trades

def compute_stats(trades):
    if not trades:
        return {}
    # Track leg specific stats
    leg_stats = {
        "CE": {"pnl": 0.0, "wins": 0, "total": 0},
        "PE": {"pnl": 0.0, "wins": 0, "total": 0}
    }
    
    pnl_values, wins, exit_reasons, directions = [], 0, defaultdict(int), defaultdict(int)
    for t in trades:
        try:
            pnl = float(t.get("net_pnl", 0) or 0)
            pnl_values.append(pnl)
            if pnl > 0: wins += 1
            exit_reasons[t.get("exit_reason", "unknown")] += 1
            
            dir_val = t.get("breakout_direction", "unknown")
            directions[dir_val] += 1
            
            if dir_val in leg_stats:
                leg_stats[dir_val]["pnl"] += pnl
                leg_stats[dir_val]["total"] += 1
                if pnl > 0:
                    leg_stats[dir_val]["wins"] += 1
        except (ValueError, TypeError):
            pass
            
    total = len(pnl_values)
    total_pnl = sum(pnl_values)
    
    for leg in ["CE", "PE"]:
        t_leg = leg_stats[leg]["total"]
        leg_stats[leg]["win_rate"] = round((leg_stats[leg]["wins"] / t_leg) * 100, 1) if t_leg > 0 else 0
        leg_stats[leg]["pnl"] = round(leg_stats[leg]["pnl"], 2)

    return {
        "total_trades": total,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round((wins / total) * 100, 1) if total > 0 else 0,
        "wins": wins,
        "losses": total - wins,
        "avg_pnl": round(total_pnl / total, 2) if total > 0 else 0,
        "max_profit": round(max(pnl_values), 2) if pnl_values else 0,
        "max_loss": round(min(pnl_values), 2) if pnl_values else 0,
        "exit_reasons": dict(exit_reasons),
        "directions": dict(directions),
        "leg_stats": leg_stats
    }

def build_equity_curve(trades):
    curve, cumulative = [], 0.0
    for t in reversed(trades):
        try:
            pnl = float(t.get("net_pnl", 0) or 0)
            cumulative += pnl
            curve.append({"date": t.get("date", ""), "daily_pnl": round(pnl, 2), "cumulative": round(cumulative, 2)})
        except (ValueError, TypeError):
            pass
    return curve

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/trades")
def api_trades():
    trades = read_trades()
    return jsonify({"trades": trades, "stats": compute_stats(trades), "equity_curve": build_equity_curve(trades)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
