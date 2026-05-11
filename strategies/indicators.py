"""Price-series indicators used by exit logic.

Single home for math-only helpers (no I/O, no state) so the live strategy and
the backtest engine can share the same definitions and stay aligned.
"""
from __future__ import annotations

import numpy as np


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    n: int = 14,
) -> np.ndarray:
    """Average True Range, simple rolling mean over the last ``n`` bars.

    True range at bar t = max(high[t]-low[t], |high[t]-close[t-1]|,
    |low[t]-close[t-1]|). At t=0 there is no prev close, so we fall back to
    the bar's own range (high[0]-low[0]).

    Returns an array of the same length as ``closes``; the first ``n-1`` entries
    average over however much history is available so the series is usable for
    early bars (rather than emitting NaNs that callers must remember to skip).
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    if not (highs.shape == lows.shape == closes.shape):
        raise ValueError("highs, lows, closes must have the same shape")
    if highs.size == 0:
        return np.array([], dtype=np.float64)
    if n < 1:
        raise ValueError("ATR window n must be >= 1")

    T = highs.size
    prev_close = np.concatenate(([closes[0]], closes[:-1]))   # at t=0, prev = close[0] → TR = high-low
    tr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_close),
        np.abs(lows - prev_close),
    ])
    atr = np.empty(T, dtype=np.float64)
    for t in range(T):
        start = max(0, t - n + 1)
        atr[t] = float(tr[start : t + 1].mean())
    return atr
