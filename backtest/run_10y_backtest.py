"""
CLI entry point to execute 10-year quantitative backtest simulation.
Usage:
    python3 run_backtest.py --mode option_buying --target-r 2.0 --capital 500000
"""

import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from backtest.config_loader import load_config
from backtest.engine import run_backtest_simulation
from backtest.analytics import compute_performance_metrics, print_performance_summary


def main():
    parser = argparse.ArgumentParser(description="10-Year NIFTY 15-Minute ORB Strategy Backtester")
    parser.add_argument("--mode", type=str, choices=["option_buying", "credit_spread"], default="option_buying", help="Strategy Mode")
    parser.add_argument("--target-r", type=float, default=2.0, help="Target R-multiple (e.g. 1.5, 2.0, 2.5, 3.0)")
    parser.add_argument("--sl-r", type=float, default=1.0, help="Stop Loss R-multiple (e.g. 1.0)")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial Capital (INR)")
    parser.add_argument("--sizing-mode", type=str, choices=["fixed_lots", "compounding"], default="fixed_lots", help="Sizing mode")
    parser.add_argument("--fixed-lots", type=int, default=1, help="Fixed lots count")
    parser.add_argument("--max-lots", type=int, default=10, help="Max lots in compounding mode")
    parser.add_argument("--start-year", type=int, default=2015, help="Simulation start year")
    parser.add_argument("--end-year", type=int, default=2026, help="Simulation end year")
    parser.add_argument("--export-csv", type=str, default="backtest_trades_output.csv", help="Export trade log CSV filename")

    args = parser.parse_args()

    overrides = {
        "strategy": {
            "mode": args.mode,
            "target_r_multiple": args.target_r,
            "sl_r_multiple": args.sl_r,
        },
        "capital": {
            "initial_capital": args.capital,
            "sizing_mode": args.sizing_mode,
            "fixed_lots": args.fixed_lots,
            "max_lots": args.max_lots,
        },
        "simulation": {
            "start_year": args.start_year,
            "end_year": args.end_year,
        }
    }

    config = load_config(custom_overrides=overrides)
    print(f"\n[INFO] Starting Backtest: Mode={args.mode.upper()}, Target={args.target_r}R, SL={args.sl_r}R, Capital=Rs.{args.capital:,.0f} ({args.start_year}-{args.end_year})...")
    
    raw_results = run_backtest_simulation(config)
    analytics = compute_performance_metrics(raw_results)

    print_performance_summary(analytics)

    if args.export_csv and raw_results.get("trades"):
        import pandas as pd
        df_trades = pd.DataFrame(raw_results["trades"])
        csv_path = os.path.join(BASE_DIR, args.export_csv)
        df_trades.to_csv(csv_path, index=False)
        print(f"\n[INFO] Trade execution log exported to: {csv_path}\n")


if __name__ == "__main__":
    main()
