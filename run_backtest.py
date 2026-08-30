#!/usr/bin/env python3
"""
NIFTY 15-Minute ORB Delta-Hedged Credit Spread 10-Year Backtesting Engine
Repository: https://github.com/bhaveshgite1001-pixel/BT
"""

import os
import sys

# Add directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from backtest.run_10y_backtest import main

if __name__ == "__main__":
    main()
