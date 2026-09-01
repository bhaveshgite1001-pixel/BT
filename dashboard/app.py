#!/usr/bin/env python3
"""
NIFTY 15-Minute ORB Delta-Hedged Credit Spread Backtest Dashboard
A Flask web application for interactive 10-year quantitative backtesting.
"""

import os
import sys
import csv
import json
import io
from flask import Flask, render_template, jsonify, request, Response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from backtest.config_loader import load_config
from backtest.engine import run_backtest_simulation
from backtest.analytics import compute_performance_metrics

app = Flask(__name__)
LATEST_BACKTEST_RESULT = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Returns the default configuration."""
    try:
        cfg = load_config()
        return jsonify({"success": True, "config": cfg})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def api_run_backtest():
    """Runs the backtest simulation with user-provided parameters."""
    global LATEST_BACKTEST_RESULT
    try:
        user_overrides = request.get_json() or {}
        config = load_config(custom_overrides=user_overrides)

        raw_results = run_backtest_simulation(config)
        analytics = compute_performance_metrics(raw_results)

        LATEST_BACKTEST_RESULT = {
            "summary": analytics["summary"],
            "yearly_stats": analytics["yearly_stats"],
            "monthly_heatmap": analytics["monthly_heatmap"],
            "equity_series": raw_results["equity_series"],
            "drawdown_series": analytics["drawdown_series"],
            "trades": raw_results["trades"],
            "config": config,
        }

        return jsonify({
            "success": True,
            "data": LATEST_BACKTEST_RESULT
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def api_export_csv():
    """Exports latest trade log to CSV."""
    global LATEST_BACKTEST_RESULT
    if not LATEST_BACKTEST_RESULT or not LATEST_BACKTEST_RESULT.get("trades"):
        return Response("No backtest results available to export.", status=404)

    trades = LATEST_BACKTEST_RESULT["trades"]
    keys = trades[0].keys()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(trades)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=nifty_orb_backtest_trades.csv"}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
