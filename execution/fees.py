"""Exchange fee constants.

Single source of truth so paper-trading, backtest engine, and any future
analysis stay aligned. If Hyperliquid changes their fee schedule, update
here and every consumer picks it up.
"""
from __future__ import annotations

# Hyperliquid perpetual taker fee (0.035% = 3.5 bps).
TAKER_FEE_RATE: float = 0.00035
