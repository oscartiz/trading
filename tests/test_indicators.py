"""Tests for strategies.indicators."""
from __future__ import annotations

import numpy as np
import pytest

from strategies.indicators import compute_atr


def test_atr_constant_series_is_zero():
    """Flat highs == lows == closes → true range zero → ATR zero."""
    n = 50
    flat = np.full(n, 100.0)
    atr = compute_atr(flat, flat, flat, n=14)
    assert atr.shape == (n,)
    assert np.allclose(atr, 0.0)


def test_atr_constant_bar_range_equals_that_range():
    """If every bar has identical high-low and no gap, ATR == high-low."""
    n = 30
    closes = np.full(n, 100.0)
    highs = closes + 1.0
    lows = closes - 1.0
    atr = compute_atr(highs, lows, closes, n=14)
    # Every TR = max(2, |1|, |-1|) = 2
    assert np.allclose(atr, 2.0)


def test_atr_handles_gaps_via_prev_close():
    """Overnight gap: TR includes |high - prev_close| / |low - prev_close|."""
    closes = np.array([100.0, 110.0, 110.0, 110.0])
    highs = np.array([101.0, 112.0, 111.0, 111.0])
    lows = np.array([99.0, 108.0, 109.0, 109.0])

    # TR[0] = high-low = 2 (no prev close, so we substitute close[0])
    # TR[1] = max(112-108=4, |112-100|=12, |108-100|=8) = 12
    # TR[2] = max(111-109=2, |111-110|=1, |109-110|=1) = 2
    # TR[3] = same as bar 2 = 2
    atr = compute_atr(highs, lows, closes, n=14)
    # n=14 means each entry averages over all bars up to that point
    assert atr[0] == pytest.approx(2.0)
    assert atr[1] == pytest.approx((2 + 12) / 2)
    assert atr[2] == pytest.approx((2 + 12 + 2) / 3)
    assert atr[3] == pytest.approx((2 + 12 + 2 + 2) / 4)


def test_atr_window_truncates_old_bars():
    """With n=2, the average uses only the most recent 2 bars."""
    closes = np.array([100.0, 100.0, 100.0, 100.0])
    highs = np.array([101.0, 102.0, 103.0, 110.0])      # last TR is big
    lows = np.array([99.0, 98.0, 97.0, 90.0])

    atr = compute_atr(highs, lows, closes, n=2)
    # TR sequence: [2, 4, 6, 20]  (constant closes → TR = high-low each bar after t=0)
    # actually TR[1] = max(102-98=4, |102-100|=2, |98-100|=2) = 4
    # TR[2] = max(103-97=6, |103-100|=3, |97-100|=3) = 6
    # TR[3] = max(110-90=20, |110-100|=10, |90-100|=10) = 20
    assert atr[0] == pytest.approx(2.0)
    assert atr[1] == pytest.approx((2 + 4) / 2)
    assert atr[2] == pytest.approx((4 + 6) / 2)
    assert atr[3] == pytest.approx((6 + 20) / 2)


def test_atr_validates_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        compute_atr(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))


def test_atr_validates_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        compute_atr(np.array([1.0]), np.array([1.0]), np.array([1.0]), n=0)


def test_atr_empty_input():
    atr = compute_atr(np.array([]), np.array([]), np.array([]), n=5)
    assert atr.shape == (0,)
