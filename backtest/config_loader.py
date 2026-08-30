"""Configuration loader for the 10-Year Backtest Engine."""

import os
import json

try:
    import yaml
except ImportError:
    yaml = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

DEFAULT_CONFIG = {
    "capital": {
        "initial_capital": 500000.0,
        "sizing_mode": "compounding",  # "fixed_lots" or "compounding"
        "fixed_lots": 1,
        "max_lots": 10,
        "margin_per_lot": 60000.0,
        "risk_per_trade_pct": 0.05,
    },
    "timing": {
        "orb_start_time": "09:15",
        "orb_end_time": "10:15",
        "candle_timeframe_min": 15,
        "trade_window_end": "15:20",
        "eod_squareoff_time": "15:25",
    },
    "risk_rules": {
        "target_r_multiple": 1.5,
        "sl_mode": "opposite_boundary",
        "min_orb_range_pts": 40.0,
        "max_orb_range_pts": 250.0,
        "max_trades_per_day": 1,
    },
    "options": {
        "target_hedge_abs_delta": 0.15,
        "use_fixed_strike_offset": False,
        "fixed_hedge_offset_pts": 300.0,
        "min_net_credit_inr": 5.0,
        "strike_step": 50,
    },
    "friction": {
        "slippage_per_unit_leg": 0.50,
        "brokerage_per_order": 20.0,
        "enable_stt": True,
        "enable_exchange_charges": True,
        "enable_gst": True,
        "enable_stamp_duty": True,
    },
    "simulation": {
        "years": 10,
        "start_year": 2015,
        "end_year": 2026,
    }
}


def load_config(custom_overrides=None, config_file_path=None):
    """Load configuration with default fallbacks and optional custom dictionary overrides."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    
    path = config_file_path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        try:
            if yaml:
                with open(path, "r") as f:
                    file_conf = yaml.safe_load(f)
                    if isinstance(file_conf, dict):
                        _deep_update(config, file_conf)
            else:
                # If yaml not installed, try json or leave defaults
                with open(path, "r") as f:
                    content = f.read()
                    if content.startswith("{"):
                        _deep_update(config, json.loads(content))
        except Exception as e:
            print(f"[WARN] Error reading config file {path}: {e}")

    if custom_overrides and isinstance(custom_overrides, dict):
        _deep_update(config, custom_overrides)

    return config


def _deep_update(target, source):
    """Recursively update dictionary target with source."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            _deep_update(target[k], v)
        else:
            target[k] = v
