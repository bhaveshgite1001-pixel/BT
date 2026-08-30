# NIFTY 15-Minute ORB Delta-Hedged Credit Spread Backtester (2015–2026)

A high-performance quantitative backtesting engine and interactive web workstation for the **NIFTY 15-Minute Opening Range Breakout (ORB) Delta-Hedged Credit Spread Strategy** across 10 years (~3,000 trading sessions, ~1.12 million 1-minute bars).

---

## 🚀 Key Features

* **High-Speed Two-Tier Caching**: Persistent disk caching + in-memory RAM cache executes 10-year backtests across 3,000+ sessions in **sub-seconds**.
* **Black-Scholes Options & Greeks**: Numerical Implied Volatility solver, dynamic intraday Theta decay ($T - t$ in trading minutes), and delta-solved ($\lvert\delta\rvert \approx 0.15$) hedge strikes.
* **Historical Era Adjustments**: Automatically switches lot sizes across regulatory eras ($75 \rightarrow 50 \rightarrow 65$) and handles transitions from monthly to weekly expiries.
* **Accurate Statutory Friction**: Models STT (0.1%), Exchange turnover (0.05%), GST (18%), Stamp duty (0.003%), flat ₹20 brokerage, and custom slippage.
* **Interactive Web Workstation**: Full Flask dashboard with interactive Chart.js Equity Curves, Underwater Drawdowns, Yearly Breakdown tables, Monthly Heatmap matrices, and CSV export.

---

## 📦 Installation

```bash
git clone https://github.com/bhaveshgite1001-pixel/BT.git
cd BT

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💻 Running the Web Dashboard

```bash
python3 dashboard/app.py
```
Open **`http://localhost:5001`** in your browser to interactively adjust parameters, view charts, and export results.

---

## ⚡ Running via CLI

```bash
# 1. Default 10-Year simulation
python3 run_backtest.py

# 2. Fixed 1 Lot with Rs. 1 Lakh Capital
python3 run_backtest.py --capital 100000 --sizing-mode fixed_lots --fixed-lots 1 --target-mult 1.5

# 3. Compounding with Rs. 5 Lakhs (Max 10 lots)
python3 run_backtest.py --capital 500000 --sizing-mode compounding --max-lots 10
```

---

## ⚙️ Configuration (`backtest/config.yaml`)

Every parameter (Capital, Max Lots, Margin per Lot, Target R, Timeframe, ORB Range Filters, Delta Hedge, and Taxes) is fully configurable via `backtest/config.yaml` or directly from the Dashboard UI.
