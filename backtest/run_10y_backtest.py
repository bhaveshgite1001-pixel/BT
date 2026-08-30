#!/usr/bin/env python3
"""CLI runner for 10-Year NIFTY 15-Min ORB Credit Spread Backtest."""

import argparse
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.config_loader import load_config
from backtest.engine import run_backtest_simulation
from backtest.analytics import compute_performance_metrics


def main():
    parser = argparse.ArgumentParser(description="10-Year NIFTY 15-Min ORB Credit Spread Backtester")
    parser.add_argument("--capital", type=float, help="Initial account capital (INR)")
    parser.add_argument("--sizing-mode", choices=["fixed_lots", "compounding"], help="Position sizing mode")
    parser.add_argument("--fixed-lots", type=int, help="Fixed lots to trade")
    parser.add_argument("--max-lots", type=int, help="Maximum lots cap in compounding mode")
    parser.add_argument("--target-mult", type=float, help="Target R-multiple extension (e.g. 1.5)")
    parser.add_argument("--start-year", type=int, help="Start year (e.g. 2015)")
    parser.add_argument("--end-year", type=int, help="End year (e.g. 2026)")
    parser.add_argument("--hedge-delta", type=float, help="Target hedge delta (e.g. 0.15)")
    parser.add_argument("--fixed-offset", type=float, help="Fixed hedge strike offset in points (e.g. 300)")
    parser.add_argument("--output-csv", default="backtest_results_10year.csv", help="Output trade log CSV path")

    args = parser.parse_args()

    # Build overrides dictionary
    overrides = {}
    if args.capital is not None:
        overrides.setdefault("capital", {})["initial_capital"] = args.capital
    if args.sizing_mode is not None:
        overrides.setdefault("capital", {})["sizing_mode"] = args.sizing_mode
    if args.fixed_lots is not None:
        overrides.setdefault("capital", {})["fixed_lots"] = args.fixed_lots
    if args.max_lots is not None:
        overrides.setdefault("capital", {})["max_lots"] = args.max_lots
    if args.target_mult is not None:
        overrides.setdefault("risk_rules", {})["target_r_multiple"] = args.target_mult
    if args.start_year is not None:
        overrides.setdefault("simulation", {})["start_year"] = args.start_year
    if args.end_year is not None:
        overrides.setdefault("simulation", {})["end_year"] = args.end_year
    if args.hedge_delta is not None:
        overrides.setdefault("options", {})["target_hedge_abs_delta"] = args.hedge_delta
    if args.fixed_offset is not None:
        overrides.setdefault("options", {})["use_fixed_strike_offset"] = True
        overrides.setdefault("options", {})["fixed_hedge_offset_pts"] = args.fixed_offset

    config = load_config(custom_overrides=overrides)

    print("=" * 70)
    print(" 🚀 10-YEAR NIFTY ORB CREDIT SPREAD BACKTEST")
    print("=" * 70)
    print(f"Capital: ₹{config['capital']['initial_capital']:,.2f} | Sizing: {config['capital']['sizing_mode']} (Max Lots: {config['capital']['max_lots']})")
    print(f"Target: {config['risk_rules']['target_r_multiple']}R | Range Filter: {config['risk_rules']['min_orb_range_pts']} - {config['risk_rules']['max_orb_range_pts']} pts")
    print(f"Period: {config['simulation']['start_year']} to {config['simulation']['end_year']}")
    print("=" * 70)

    raw_results = run_backtest_simulation(config)
    analytics = compute_performance_metrics(raw_results)
    s = analytics["summary"]

    print("\n" + "=" * 70)
    print(" 📊 PERFORMANCE SUMMARY (2015 – 2026)")
    print("=" * 70)
    print(f" Initial Capital:     ₹{s['initial_capital']:>14,.2f}")
    print(f" Final Capital:       ₹{s['final_capital']:>14,.2f}")
    print(f" Total Net P&L:       ₹{s['total_net_pnl']:>14,.2f} ({s['total_roi_pct']:+.2f}%)")
    print(f" CAGR:                 {s['cagr_pct']:>14.2f}%")
    print(f" Profit Factor:        {s['profit_factor']:>14.2f}")
    print(f" Win Rate:             {s['win_rate_pct']:>14.2f}% ({s['winning_trades']}W / {s['losing_trades']}L)")
    print(f" Max Drawdown:        -₹{s['max_drawdown_inr']:>13,.2f} ({s['max_drawdown_pct']:.2f}%)")
    print(f" Sharpe Ratio:         {s['sharpe_ratio']:>14.2f}")
    print(f" Sortino Ratio:        {s['sortino_ratio']:>14.2f}")
    print(f" Avg Win / Avg Loss:  ₹{s['avg_win_inr']:>6,.0f} / -₹{s['avg_loss_inr']:>5,.0f} (Ratio: {s['win_loss_ratio']:.2f})")
    print("=" * 70)

    print("\n 📅 YEAR-BY-YEAR BREAKDOWN:")
    print(f" {'Year':<6} | {'Trades':<7} | {'Win %':<8} | {'Net P&L (INR)':<15} | {'Profit Factor':<13}")
    print("-" * 62)
    for y in analytics["yearly_stats"]:
        print(f" {y['year']:<6} | {y['trades']:<7} | {y['win_rate_pct']:>6.1f}% | ₹{y['net_pnl']:>13,.2f} | {y['profit_factor']:>11.2f}")
    print("=" * 70)

    # Save CSV
    if raw_results["trades"]:
        import pandas as pd
        df = pd.DataFrame(raw_results["trades"])
        df.to_csv(args.output_csv, index=False)
        print(f"\n[INFO] Complete trade logs exported to: {args.output_csv}")


if __name__ == "__main__":
    main()
