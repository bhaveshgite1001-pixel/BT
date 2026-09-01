#!/usr/bin/env python3
"""
NIFTY ORB Trading Dashboard
A Flask-based web dashboard to monitor the trading bot's status, logs,
and trade history. Includes an Admin panel to update token, .env config,
and bot settings — all from the browser.
"""

import os
import sys
import csv
import json
import functools
from datetime import datetime, date
import pytz
from collections import defaultdict

from flask import Flask, render_template, jsonify, request, Response
from dotenv import load_dotenv, set_key, dotenv_values

load_dotenv()

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_FILES = {
    "index": {
        "log": os.path.join(BASE_DIR, "orb_bot.log"),
        "trade": os.path.join(BASE_DIR, "trade_log.csv"),
        "state": os.path.join(BASE_DIR, "orb_state.json"),
        "lock": os.path.join(BASE_DIR, "orb_bot.lock"),
        "bot_name": "nifty_orb_option_seller"
    },
    "premium": {
        "log": os.path.join(BASE_DIR, "premium_orb_bot.log"),
        "trade": os.path.join(BASE_DIR, "premium_trade_log.csv"),
        "state": os.path.join(BASE_DIR, "premium_orb_state.json"),
        "lock": os.path.join(BASE_DIR, "premium_orb_bot.lock"),
        "bot_name": "premium_orb_seller"
    }
}
def get_s_files():
    from flask import request
    strategy = request.args.get("strategy", "index")
    return STRATEGY_FILES.get(strategy, STRATEGY_FILES["index"])
TOKEN_FILE       = os.path.join(BASE_DIR, "token.txt")
ENV_FILE         = os.path.join(BASE_DIR, ".env")

DASH_USER        = os.environ.get("DASHBOARD_USER", "admin")
DASH_PASSWORD    = os.environ.get("DASHBOARD_PASSWORD", "orb2025")
MAX_LOG_LINES    = 300

# Which .env keys are exposed to the admin panel (never expose password fields directly)
EDITABLE_KEYS = [
    "BROKER",
    "FIVEPAISA_APP_SOURCE",
    "FIVEPAISA_USER_ID",
    "FIVEPAISA_USER_PASSWORD",
    "FIVEPAISA_ENCRYPTION_KEY",
    "FIVEPAISA_API_KEY",
    "FIVEPAISA_TOTP_SECRET",
    "DHAN_CLIENT_ID",
    "DHAN_ACCESS_TOKEN",
    "DASHBOARD_USER",
    "DASHBOARD_PASSWORD",
]

# ── Auth ───────────────────────────────────────────────────────────────────────
def check_auth(username, password):
    return username == DASH_USER and password == DASH_PASSWORD

def requires_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="ORB Dashboard"'},
            )
        return f(*args, **kwargs)
    return decorated

# ── Helpers ────────────────────────────────────────────────────────────────────
def read_tail(filepath, n=MAX_LOG_LINES):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            buf = b""
            pos = size
            lines_found = 0
            while pos > 0 and lines_found < n:
                chunk = min(4096, pos)
                pos -= chunk
                f.seek(pos)
                buf = f.read(chunk) + buf
                lines_found = buf.count(b"\n")
            lines = buf.decode("utf-8", errors="replace").splitlines()
            return lines[-n:] if len(lines) > n else lines
    except Exception:
        return []

def read_trades():
    trades = []
    trade_csv = get_s_files()["trade"]
    if not os.path.exists(trade_csv):
        return trades
    try:
        with open(trade_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except Exception:
        pass
    return list(reversed(trades))

def read_state():
    state_file = get_s_files()["state"]
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return {}

def get_bot_status():
    is_running = False
    bot_name = get_s_files()["bot_name"]
    lock_file = get_s_files()["lock"]
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info["cmdline"] or [])
            if bot_name in cmdline and proc.pid != os.getpid():
                is_running = True
                break
    except Exception:
        if os.path.exists(lock_file) and os.path.getsize(lock_file) > 0:
            is_running = True

    if not is_running:
        if date.today().weekday() >= 5: return "WEEKEND"
        return "OFFLINE"

    state = read_state()
    status = state.get("status", "")
    if status == "closed":         return "DONE_TODAY"
    if status in ("in_trade", "open", "exit_pending"): return "IN_TRADE"
    if status == "orb_building":   return "ORB_BUILDING"
    if status == "waiting_breakout": return "WAITING"
    
    return "RUNNING"

def compute_stats(trades):
    if not trades:
        return {}
    # Track leg specific stats
    leg_stats = {
        "CE": {"pnl": 0.0, "wins": 0, "total": 0},
        "PE": {"pnl": 0.0, "wins": 0, "total": 0}
    }
    
    for t in trades:
        try:
            pnl = float(t.get("net_pnl", 0) or 0)
            pnl_values.append(pnl)
            if pnl > 0: wins += 1
            exit_reasons[t.get("exit_reason", "unknown")] += 1
            
            # Support both live bot (breakout_direction) and backtest bot (leg_type via direction) formats
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
@requires_auth
def index():
    return render_template("index.html")

@app.route("/api/status")
@requires_auth
def api_status():
    state = read_state()
    ist = pytz.timezone("Asia/Kolkata")
    return jsonify({"status": get_bot_status(), "state": state, "server_time": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/api/logs")
@requires_auth
def api_logs():
    return jsonify({"lines": read_tail(get_s_files()["log"], MAX_LOG_LINES)})

@app.route("/api/trades")
@requires_auth
def api_trades():
    trades = read_trades()
    return jsonify({"trades": trades, "stats": compute_stats(trades), "equity_curve": build_equity_curve(trades)})

@app.route("/api/health")
@requires_auth
def api_health():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return jsonify({
            "cpu_pct": cpu,
            "mem_used_mb": round(mem.used / 1024 / 1024, 1),
            "mem_total_mb": round(mem.total / 1024 / 1024, 1),
            "mem_pct": mem.percent,
            "disk_used_gb": round(disk.used / 1024**3, 2),
            "disk_total_gb": round(disk.total / 1024**3, 2),
            "disk_pct": disk.percent,
        })
    except ImportError:
        return jsonify({"error": "psutil not installed"})

# ── Admin APIs ─────────────────────────────────────────────────────────────────

@app.route("/api/admin/token", methods=["GET", "POST"])
@requires_auth
def admin_token():
    """Read or update token.txt"""
    if request.method == "GET":
        token = ""
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE) as f:
                    token = f.read().strip()
            except Exception:
                pass
        return jsonify({"token": token})

    data = request.get_json()
    new_token = (data.get("token") or "").strip()
    if not new_token:
        return jsonify({"ok": False, "error": "Token cannot be empty"}), 400
    try:
        with open(TOKEN_FILE, "w") as f:
            f.write(new_token + "\n")
        return jsonify({"ok": True, "message": "token.txt updated successfully!"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/config", methods=["GET", "POST"])
@requires_auth
def admin_config():
    """Read or update .env file"""
    if request.method == "GET":
        values = dotenv_values(ENV_FILE) if os.path.exists(ENV_FILE) else {}
        # Only return editable keys; mask passwords with asterisks
        result = {}
        for key in EDITABLE_KEYS:
            val = values.get(key, "")
            result[key] = val
        return jsonify({"config": result, "editable_keys": EDITABLE_KEYS})

    data = request.get_json()
    updates = data.get("updates", {})
    if not updates:
        return jsonify({"ok": False, "error": "No updates provided"}), 400
    try:
        for key, value in updates.items():
            if key in EDITABLE_KEYS:
                set_key(ENV_FILE, key, str(value))
        return jsonify({"ok": True, "message": f"Updated {len(updates)} setting(s). Restart the bot for changes to take effect."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/clear-state", methods=["POST"])
@requires_auth
def admin_clear_state():
    """Clear the bot state file (allows trading again today)"""
    try:
        state_file = get_s_files()["state"]
        if os.path.exists(state_file):
            os.remove(state_file)
        return jsonify({"ok": True, "message": "State cleared. The bot will start fresh."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/clear-log", methods=["POST"])
@requires_auth
def admin_clear_log():
    """Clear the bot log file"""
    try:
        log_file = get_s_files()["log"]
        open(log_file, "w").close()
        return jsonify({"ok": True, "message": "Log file cleared."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/bot/start", methods=["POST"])
@requires_auth
def admin_bot_start():
    """Start the trading bot in the background."""
    import subprocess, signal
    bot_name = get_s_files()["bot_name"]
    bot_script = os.path.join(BASE_DIR, f"{bot_name}.py")
    venv_python = os.path.join(BASE_DIR, "venv", "bin", "python")
    python_bin = venv_python if os.path.exists(venv_python) else "python3"

    # Check if already running
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info["cmdline"] or [])
            if bot_name in cmdline:
                return jsonify({"ok": False, "error": "Bot is already running (PID {})".format(proc.pid)}), 409
    except Exception:
        pass

    try:
        log_path = get_s_files()["log"]
        with open(log_path, "a") as log_out:
            proc = subprocess.Popen(
                [python_bin, bot_script],
                cwd=BASE_DIR,
                stdout=log_out,
                stderr=log_out,
                start_new_session=True,
            )
        return jsonify({"ok": True, "message": f"Bot started (PID {proc.pid}). Logs will appear in the Live view."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/bot/stop", methods=["POST"])
@requires_auth
def admin_bot_stop():
    """Gracefully stop the trading bot (SIGTERM)."""
    try:
        import psutil, signal
        bot_name = get_s_files()["bot_name"]
        killed = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info["cmdline"] or [])
            if bot_name in cmdline and proc.pid != os.getpid():
                proc.send_signal(signal.SIGTERM)
                killed.append(proc.pid)
        if killed:
            return jsonify({"ok": True, "message": f"Sent stop signal to bot (PID {killed}). It will finish any open order before exiting."})
        return jsonify({"ok": False, "error": "Bot is not currently running."}), 404
    except ImportError:
        return jsonify({"ok": False, "error": "psutil not installed — run: pip install psutil"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/admin/test-connection", methods=["POST"])
@requires_auth
def api_test_connection():
    try:
        sys.path.append(BASE_DIR)
        import importlib
        bot_name = get_s_files()["bot_name"]
        bot = importlib.import_module(bot_name)
        res = bot.test_api_connection()
        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


