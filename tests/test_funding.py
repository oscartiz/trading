"""Tests for funding-rate accrual in the regime backtest engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting.regime_engine import _accrue_funding_between, run_regime_backtest
from strategies.configs import RegimeSwitchingConfig

from tests.conftest import build_prices
from tests.test_regime_engine import (
    StubClassifier, _patch_classifier, make_snap,
)


def test_accrue_funding_long_pays_positive_rate():
    """Longs pay shorts when funding_rate > 0 → returned cost is positive."""
    ts = np.array([100, 200, 300], dtype="int64")
    rates = np.array([0.0001, 0.0001, 0.0001])
    cost = _accrue_funding_between(ts, rates, 50, 350, "long", notional=1000.0)
    assert cost == pytest.approx(0.0001 * 3 * 1000.0)


def test_accrue_funding_short_receives_positive_rate():
    """Shorts receive funding from longs when funding_rate > 0 → cost is negative."""
    ts = np.array([100, 200, 300], dtype="int64")
    rates = np.array([0.0001, 0.0001, 0.0001])
    cost = _accrue_funding_between(ts, rates, 50, 350, "short", notional=1000.0)
    assert cost == pytest.approx(-0.0001 * 3 * 1000.0)


def test_accrue_funding_window_excludes_endpoints_correctly():
    """The half-open (start, end] window excludes the start tick."""
    ts = np.array([100, 200, 300], dtype="int64")
    rates = np.array([0.0001, 0.0001, 0.0001])
    # Start exactly at 100 — that funding tick should be excluded.
    cost = _accrue_funding_between(ts, rates, 100, 300, "long", notional=1000.0)
    assert cost == pytest.approx(0.0001 * 2 * 1000.0)


def test_accrue_funding_empty_inputs_returns_zero():
    ts = np.empty(0, dtype="int64")
    rates = np.empty(0, dtype=np.float64)
    assert _accrue_funding_between(ts, rates, 0, 1_000, "long", notional=1000.0) == 0.0


def _base_cfg(**overrides) -> RegimeSwitchingConfig:
    base = dict(
        train_window_bars=30,
        refit_every_bars=1000,
        signal_mode="smoothed",
        entry_proba=0.65,
        max_chop_proba=0.5,
        min_expected_return_per_bar=0.0,
        entry_confirmation_bars=0,
        same_regime_cooldown_bars=0,
        min_hold_bars=0,
        soft_exit_proba=None,
        position_size_usd=100.0,
    )
    base.update(overrides)
    return RegimeSwitchingConfig(**base)


def test_backtest_charges_funding_on_open_long():
    """A long that spans several funding settlements pays cumulative funding."""
    prices = build_prices([100.0] * 40 + [110.0] * 10)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bull] * 10 + [chop] * 10

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 5 funding settlements at 8h cadence, each charging +5bp on a long.
    funding_times = [start + timedelta(hours=30 + 8 * i) for i in range(5)]
    funding_df = pd.DataFrame({
        "timestamp": pd.to_datetime(funding_times, utc=True),
        "funding_rate": [0.0005] * 5,
    })

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", _base_cfg(), funding_df=funding_df)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side == "long"
    # Long paid all 5 funding ticks at $100 notional.
    assert t.funding > 0
    assert result.total_funding == pytest.approx(t.funding)
