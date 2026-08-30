"""Statutory Taxes, Exchange Charges, Brokerage & Slippage Model for Indian F&O Options."""


def calculate_trade_friction(
    sell_entry_price,
    sell_exit_price,
    buy_entry_price,
    buy_exit_price,
    qty,
    friction_config=None
):
    """
    Calculates total friction (Slippage + Statutory Taxes + Brokerage) for a 2-leg credit spread.
    
    Leg 1 (Short ATM Option): Sold at sell_entry_price, Bought back at sell_exit_price.
    Leg 2 (Long OTM Hedge Option): Bought at buy_entry_price, Sold at buy_exit_price.
    """
    cfg = friction_config or {}
    
    slippage_rate = cfg.get("slippage_per_unit_leg", 0.50)
    brokerage_per_order = cfg.get("brokerage_per_order", 20.0)
    enable_stt = cfg.get("enable_stt", True)
    enable_exch = cfg.get("enable_exchange_charges", True)
    enable_gst = cfg.get("enable_gst", True)
    enable_stamp = cfg.get("enable_stamp_duty", True)

    # 1. Slippage (₹0.50 / unit per leg on entry and exit)
    # Leg 1 entry: sold at (price - slippage) -> slippage loss
    # Leg 1 exit: bought back at (price + slippage) -> slippage loss
    # Total slippage points across both legs entry + exit = 4 * slippage_rate
    slippage_cost = 4.0 * slippage_rate * qty

    # 2. Brokerage: 2 legs entry (2 orders) + 2 legs exit (2 orders) = 4 orders
    brokerage = 4.0 * brokerage_per_order

    # 3. Turnover Calculations
    # Sell side transactions: Short entry (sell) + Long exit (sell)
    sell_turnover = (sell_entry_price + buy_exit_price) * qty
    # Buy side transactions: Long entry (buy) + Short exit (buy)
    buy_turnover = (buy_entry_price + sell_exit_price) * qty
    total_turnover = sell_turnover + buy_turnover

    # 4. Securities Transaction Tax (STT) - 0.1% on Option Sell Turnover (revised from 0.0625%)
    stt = (sell_turnover * 0.001) if enable_stt else 0.0

    # 5. Exchange Turnover Charges - 0.05% on total premium turnover
    exchange_charges = (total_turnover * 0.0005) if enable_exch else 0.0

    # 6. SEBI Turnover Charges - ₹10 per crore (0.0001%)
    sebi_charges = total_turnover * 0.000001

    # 7. Stamp Duty - 0.003% on Buy side turnover
    stamp_duty = (buy_turnover * 0.00003) if enable_stamp else 0.0

    # 8. GST - 18% on (Brokerage + Exchange Charges + SEBI Charges)
    gst = ((brokerage + exchange_charges + sebi_charges) * 0.18) if enable_gst else 0.0

    total_taxes_and_brokerage = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst
    total_friction = slippage_cost + total_taxes_and_brokerage

    return {
        "slippage_cost": round(slippage_cost, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charges": round(exchange_charges, 2),
        "gst": round(gst, 2),
        "stamp_duty": round(stamp_duty, 2),
        "total_taxes": round(total_taxes_and_brokerage, 2),
        "total_friction": round(total_friction, 2),
    }
