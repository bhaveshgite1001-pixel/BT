"""Performance Analytics, Risk Ratios, Yearly Breakdown, and Monthly Heatmap Generator."""

import math
import pandas as pd
import numpy as np


def compute_performance_metrics(results):
    """
    Computes comprehensive quant metrics, annual breakdowns, and monthly heatmap matrix.
    """
    initial_cap = float(results["initial_capital"])
    final_cap = float(results["final_capital"])
    trades = results.get("trades", [])
    equity_series = results.get("equity_series", [])

    total_net_pnl = final_cap - initial_cap
    total_roi_pct = (total_net_pnl / initial_cap) * 100.0 if initial_cap > 0 else 0.0

    if not trades:
        return {
            "summary": {
                "initial_capital": initial_cap,
                "final_capital": final_cap,
                "total_net_pnl": 0.0,
                "total_roi_pct": 0.0,
                "cagr_pct": 0.0,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "loss_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_inr": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "avg_win_inr": 0.0,
                "avg_loss_inr": 0.0,
                "win_loss_ratio": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
            },
            "yearly_stats": [],
            "monthly_heatmap": {},
            "drawdown_series": [],
        }

    df_trades = pd.DataFrame(trades)
    df_trades["date"] = pd.to_datetime(df_trades["date"])
    df_trades["year"] = df_trades["date"].dt.year
    df_trades["month"] = df_trades["date"].dt.month

    # Win / Loss Statistics
    winning_trades = df_trades[df_trades["net_pnl"] > 0]
    losing_trades = df_trades[df_trades["net_pnl"] <= 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    total_trades_count = len(df_trades)

    win_rate = (num_wins / total_trades_count) * 100.0 if total_trades_count > 0 else 0.0
    loss_rate = (num_losses / total_trades_count) * 100.0 if total_trades_count > 0 else 0.0

    gross_profit = winning_trades["net_pnl"].sum() if num_wins > 0 else 0.0
    gross_loss = abs(losing_trades["net_pnl"].sum()) if num_losses > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    avg_win = (gross_profit / num_wins) if num_wins > 0 else 0.0
    avg_loss = (gross_loss / num_losses) if num_losses > 0 else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # Consecutive Wins & Losses
    consec_wins = max_consec_wins = 0
    consec_losses = max_consec_losses = 0
    for pnl in df_trades["net_pnl"]:
        if pnl > 0:
            consec_wins += 1
            consec_losses = 0
            max_consec_wins = max(max_consec_wins, consec_wins)
        else:
            consec_losses += 1
            consec_wins = 0
            max_consec_losses = max(max_consec_losses, consec_losses)

    # Equity Curve & Drawdown Analysis
    df_equity = pd.DataFrame(equity_series)
    df_equity["date"] = pd.to_datetime(df_equity["date"])
    df_equity["peak"] = df_equity["equity"].cummax()
    df_equity["drawdown_inr"] = df_equity["equity"] - df_equity["peak"]
    df_equity["drawdown_pct"] = (df_equity["drawdown_inr"] / df_equity["peak"]) * 100.0

    max_dd_inr = abs(float(df_equity["drawdown_inr"].min()))
    max_dd_pct = abs(float(df_equity["drawdown_pct"].min()))

    # CAGR Calculation (over total trading days)
    total_days = (df_equity["date"].iloc[-1] - df_equity["date"].iloc[0]).days if len(df_equity) > 1 else 365
    years_count = max(0.1, total_days / 365.25)
    cagr_pct = (((final_cap / initial_cap) ** (1.0 / years_count)) - 1.0) * 100.0 if initial_cap > 0 and final_cap > 0 else 0.0

    # Sharpe & Sortino (Daily Returns based)
    df_equity["daily_return"] = df_equity["equity"].pct_change().fillna(0.0)
    mean_daily_return = df_equity["daily_return"].mean()
    std_daily_return = df_equity["daily_return"].std()
    downside_std = df_equity[df_equity["daily_return"] < 0]["daily_return"].std()

    risk_free_daily = 0.065 / 250.0  # 6.5% annual risk-free rate
    sharpe_ratio = ((mean_daily_return - risk_free_daily) / std_daily_return * math.sqrt(250)) if std_daily_return > 0 else 0.0
    sortino_ratio = ((mean_daily_return - risk_free_daily) / downside_std * math.sqrt(250)) if downside_std > 0 else 0.0

    # Year-by-Year Breakdown Table
    yearly_stats = []
    for yr, y_group in df_trades.groupby("year"):
        y_wins = y_group[y_group["net_pnl"] > 0]
        y_losses = y_group[y_group["net_pnl"] <= 0]
        y_net_pnl = y_group["net_pnl"].sum()
        y_gross_p = y_wins["net_pnl"].sum() if len(y_wins) > 0 else 0.0
        y_gross_l = abs(y_losses["net_pnl"].sum()) if len(y_losses) > 0 else 0.0
        y_pf = (y_gross_p / y_gross_l) if y_gross_l > 0 else (99.0 if y_gross_p > 0 else 0.0)
        
        yearly_stats.append({
            "year": int(yr),
            "trades": len(y_group),
            "win_rate_pct": round((len(y_wins) / len(y_group)) * 100.0, 1),
            "net_pnl": round(y_net_pnl, 2),
            "profit_factor": round(y_pf, 2),
            "target_hits": int((y_group["exit_reason"] == "TARGET").sum()),
            "sl_hits": int((y_group["exit_reason"] == "STOP_LOSS").sum()),
            "eod_exits": int((y_group["exit_reason"] == "EOD").sum()),
        })

    # Monthly Heatmap Matrix (Year x Month)
    monthly_heatmap = {}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for yr in sorted(df_trades["year"].unique()):
        yr_str = str(int(yr))
        monthly_heatmap[yr_str] = {}
        for m_idx in range(1, 13):
            m_trades = df_trades[(df_trades["year"] == yr) & (df_trades["month"] == m_idx)]
            m_pnl = m_trades["net_pnl"].sum() if len(m_trades) > 0 else 0.0
            monthly_heatmap[yr_str][month_names[m_idx - 1]] = round(m_pnl, 2)

    # Simplified Drawdown Series for Plotly/Chart.js
    drawdown_series = [
        {"date": str(row["date"])[:10], "drawdown_pct": round(float(row["drawdown_pct"]), 2)}
        for _, row in df_equity.iterrows()
    ]

    # CE / PE Leg Analysis
    leg_stats = {
        "CE": {"pnl": 0.0, "wins": 0, "losses": 0, "total": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_dd_inr": 0.0},
        "PE": {"pnl": 0.0, "wins": 0, "losses": 0, "total": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_dd_inr": 0.0}
    }
    
    if "option_type" in df_trades.columns:
        for opt_type in ["CE", "PE"]:
            leg_df = df_trades[df_trades["option_type"] == opt_type].copy()
            if len(leg_df) > 0:
                leg_wins = leg_df[leg_df["net_pnl"] > 0]
                leg_losses = leg_df[leg_df["net_pnl"] <= 0]
                
                leg_stats[opt_type]["pnl"] = round(float(leg_df["net_pnl"].sum()), 2)
                leg_stats[opt_type]["total"] = len(leg_df)
                leg_stats[opt_type]["wins"] = len(leg_wins)
                leg_stats[opt_type]["losses"] = len(leg_losses)
                leg_stats[opt_type]["win_rate"] = round((len(leg_wins) / len(leg_df)) * 100.0, 1)
                
                gross_profit = float(leg_wins["net_pnl"].sum()) if len(leg_wins) > 0 else 0.0
                gross_loss = abs(float(leg_losses["net_pnl"].sum())) if len(leg_losses) > 0 else 0.0
                
                leg_stats[opt_type]["profit_factor"] = round((gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0), 2)
                leg_stats[opt_type]["avg_win"] = round((gross_profit / len(leg_wins)) if len(leg_wins) > 0 else 0.0, 2)
                leg_stats[opt_type]["avg_loss"] = round((gross_loss / len(leg_losses)) if len(leg_losses) > 0 else 0.0, 2)
                
                # Leg-specific Drawdown
                leg_df["cum_pnl"] = leg_df["net_pnl"].cumsum()
                leg_df["peak"] = leg_df["cum_pnl"].cummax()
                leg_df["drawdown_inr"] = leg_df["cum_pnl"] - leg_df["peak"]
                leg_stats[opt_type]["max_dd_inr"] = round(abs(float(leg_df["drawdown_inr"].min())), 2)

    return {
        "summary": {
            "initial_capital": round(initial_cap, 2),
            "final_capital": round(final_cap, 2),
            "total_net_pnl": round(total_net_pnl, 2),
            "total_roi_pct": round(total_roi_pct, 2),
            "cagr_pct": round(cagr_pct, 2),
            "total_trades": total_trades_count,
            "winning_trades": num_wins,
            "losing_trades": num_losses,
            "win_rate_pct": round(win_rate, 2),
            "loss_rate_pct": round(loss_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_inr": round(max_dd_inr, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "leg_stats": leg_stats
        },
        "yearly_stats": yearly_stats,
        "monthly_heatmap": monthly_heatmap,
        "drawdown_series": drawdown_series,
    }


def print_performance_summary(analytics):
    """Prints a beautiful quantitative performance report to terminal."""
    s = analytics["summary"]
    print("=" * 65)
    print("       NIFTY 15-MINUTE ORB QUANTITATIVE STRATEGY REPORT       ")
    print("=" * 65)
    print(f" Initial Capital      : Rs. {s['initial_capital']:>14,.2f}")
    print(f" Final Capital        : Rs. {s['final_capital']:>14,.2f}")
    print(f" Total Net Profit     : Rs. {s['total_net_pnl']:>14,.2f} ({s['total_roi_pct']:+.2f}%)")
    print(f" CAGR                 : {s['cagr_pct']:>15.2f}%")
    print("-" * 65)
    print(f" Total Trades         : {s['total_trades']:>18}")
    print(f" Win Rate             : {s['win_rate_pct']:>17.1f}% ({s['winning_trades']}W / {s['losing_trades']}L)")
    print(f" Profit Factor        : {s['profit_factor']:>18.2f}")
    print(f" Avg Win / Avg Loss   : Rs. {s['avg_win_inr']:>8,.2f} / -Rs. {s['avg_loss_inr']:>8,.2f}")
    print(f" Payoff (Reward/Risk) : {s['win_loss_ratio']:>18.2f}")
    print("-" * 65)
    print(f" Max Drawdown (INR)   : Rs. -{s['max_drawdown_inr']:>13,.2f}")
    print(f" Max Drawdown (%)     : {s['max_drawdown_pct']:>17.2f}%")
    print(f" Sharpe Ratio         : {s['sharpe_ratio']:>18.2f}")
    print(f" Sortino Ratio        : {s['sortino_ratio']:>18.2f}")
    print(f" Max Consecutive Wins : {s['max_consecutive_wins']:>18}")
    print(f" Max Consec Losses    : {s['max_consecutive_losses']:>18}")
    print("-" * 65)
    
    if "leg_stats" in s:
        ce = s["leg_stats"]["CE"]
        pe = s["leg_stats"]["PE"]
        print(" LEG PERFORMANCE:")
        print(f" CE : {ce['win_rate']:>5.1f}% Win Rate | {ce['total']:>4} Trades | Rs. {ce['pnl']:>12,.2f}")
        print(f" PE : {pe['win_rate']:>5.1f}% Win Rate | {pe['total']:>4} Trades | Rs. {pe['pnl']:>12,.2f}")
        
    print("=" * 65)

    if analytics.get("yearly_stats"):
        print("\n📅 YEAR-BY-YEAR PERFORMANCE BREAKDOWN:")
        print(f" {'Year':<6} | {'Trades':<7} | {'Win %':<7} | {'PF':<6} | {'Net P&L (Rs.)':<16} | {'Targets':<8} | {'SL':<6}")
        print("-" * 68)
        for y in analytics["yearly_stats"]:
            print(f" {y['year']:<6} | {y['trades']:<7} | {y['win_rate_pct']:>5.1f}% | {y['profit_factor']:>6.2f} | Rs. {y['net_pnl']:>12,.2f} | {y['target_hits']:<8} | {y['sl_hits']:<6}")
        print("-" * 68)
