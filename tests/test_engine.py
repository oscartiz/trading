"""Tests for the funding-rate backtest engine.

The engine merges funding settlements into hourly OHLC bars and simulates
entry/exit on funding-rate thresholds. Tests deliberately use small fixtures
with hand-built funding settlements so we can assert exact P&L.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backtesting.engine import run_backtest
from strategies.configs import FundingConfig

from tests.conftest import build_prices


def _funding_df(rows: list[tuple[datetime, float]]) -> pd.DataFrame:
    """Build a settlement dataframe — one row per 8h funding settlement."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([t for t, _ in rows], utc=True),
        "funding_rate": [r for _, r in rows],
    })


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


def _base_cfg(**overrides) -> FundingConfig:
    base = dict(
        entry_threshold=0.0002,
        exit_threshold=0.00005,
        stop_loss_pct=0.02,
        max_hold_hours=48,
        position_size_usd=100.0,
        poll_interval_seconds=600,
    )
    base.update(overrides)
    return FundingConfig(**base)


# --------------------------------------------------------------------------- #
#  Trade direction & price P&L                                                 #
# --------------------------------------------------------------------------- #


def test_positive_funding_opens_short():
    """Positive funding → strategy goes short (longs pay us)."""
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0005)])  # >> entry threshold
    cfg = _base_cfg(exit_threshold=0.0)        # never auto-exit on funding
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert len(result.trades) == 1
    assert result.trades[0].side == "short"


def test_negative_funding_opens_long():
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), -0.0005)])
    cfg = _base_cfg(exit_threshold=0.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert len(result.trades) == 1
    assert result.trades[0].side == "long"


def test_short_winning_price_pnl_when_price_falls():
    prices = build_prices([100.0] * 8 + [90.0] * 16)
    funding = _funding_df([(_t(0), 0.0005)])
    # Disable stop so price drop is allowed; force exit on max_hold instead.
    cfg = _base_cfg(exit_threshold=0.0001, max_hold_hours=8, stop_loss_pct=1.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    t = result.trades[0]
    assert t.side == "short"
    assert t.price_pnl == pytest.approx(10.0)


def test_long_losing_price_pnl_when_price_falls():
    prices = build_prices([100.0] * 8 + [90.0] * 16)
    funding = _funding_df([(_t(0), -0.0005)])
    cfg = _base_cfg(exit_threshold=0.0001, max_hold_hours=8, stop_loss_pct=1.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    t = result.trades[0]
    assert t.side == "long"
    assert t.price_pnl == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
#  Funding accrual at 8-hour settlements                                       #
# --------------------------------------------------------------------------- #


def test_funding_collected_only_at_settlement_rows():
    """Funding is accrued at the 8h settlement bars, not at every hourly bar.

    The entry-bar settlement does NOT accrue — the position is opened *after*
    that bar's settlement check. Only post-entry settlements add to funding.
    """
    prices = build_prices([100.0] * 24)
    # Settlements at hours 0 (entry, no accrual), 8 (accrue), 16 (accrue + exit).
    funding = _funding_df([(_t(0), 0.0005), (_t(8), 0.0005), (_t(16), 0.00001)])
    cfg = _base_cfg(exit_threshold=0.0001, max_hold_hours=24, position_size_usd=100.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    t = result.trades[0]
    # h=8 accrues |0.0005|×100=0.05; h=16 accrues |0.00001|×100=0.001 then exits.
    assert t.funding_collected == pytest.approx(0.05 + 0.001)


def test_entry_bar_settlement_does_not_accrue():
    """Confirm entry-bar settlement is excluded from funding accrual."""
    prices = build_prices([100.0] * 24)
    # Only one settlement, at the entry bar — should produce zero accrued funding.
    funding = _funding_df([(_t(0), 0.0005)])
    cfg = _base_cfg(exit_threshold=0.0, max_hold_hours=8, stop_loss_pct=1.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert result.trades[0].funding_collected == pytest.approx(0.0)


def test_funding_flipped_exit():
    """When funding flips sign while we're holding, exit reason = funding_flipped."""
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0005), (_t(8), -0.0005)])
    cfg = _base_cfg(exit_threshold=0.00001)  # don't trigger normalised
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert result.trades[0].exit_reason == "funding_flipped"


def test_funding_normalised_exit():
    """When |funding| drops below exit_threshold, exit reason = funding_normalised."""
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0005), (_t(8), 0.000001)])
    cfg = _base_cfg(exit_threshold=0.00005)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert result.trades[0].exit_reason == "funding_normalised"


# --------------------------------------------------------------------------- #
#  Stop loss & max hold                                                        #
# --------------------------------------------------------------------------- #


def test_stop_loss_long_uses_low():
    """Long stop must trigger when low ≤ entry × (1 - stop_pct)."""
    closes = [100.0] * 24
    lows = [100.0] * 8 + [90.0] + [100.0] * 15  # wick at hour 8 blows the 2% stop
    prices = build_prices(closes, lows=lows)
    funding = _funding_df([(_t(0), -0.0005), (_t(8), -0.0005)])
    cfg = _base_cfg(exit_threshold=0.0, stop_loss_pct=0.02)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    t = result.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(100.0 * (1 - 0.02))


def test_stop_loss_short_uses_high():
    closes = [100.0] * 24
    highs = [100.0] * 8 + [110.0] + [100.0] * 15
    prices = build_prices(closes, highs=highs)
    funding = _funding_df([(_t(0), 0.0005), (_t(8), 0.0005)])
    cfg = _base_cfg(exit_threshold=0.0, stop_loss_pct=0.02)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    t = result.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(100.0 * (1 + 0.02))


def test_max_hold_hours_force_exit():
    """A trade with persistent funding must close at max_hold_hours."""
    prices = build_prices([100.0] * 60)
    # Funding stays elevated the whole period, no flip, no normalisation.
    funding = _funding_df([(_t(h), 0.0005) for h in range(0, 60, 8)])
    cfg = _base_cfg(exit_threshold=0.0, max_hold_hours=24)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert result.trades[0].exit_reason == "max_hold_time"


# --------------------------------------------------------------------------- #
#  Fees & lifecycle                                                            #
# --------------------------------------------------------------------------- #


def test_fees_charged_on_every_trade():
    """funding_collected = funding_accumulated - 2 × fee_rate × notional.

    Entry bar settlement does not accrue, so the only settlement that contributes
    is the hour-8 one — which also triggers exit (funding_normalised).
    """
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0005), (_t(8), 0.000001)])
    cfg = _base_cfg(exit_threshold=0.00005, position_size_usd=100.0)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.001)
    expected_fees = 2 * 0.001 * 100.0           # 0.20 USD
    expected_funding_accrued = 0.000001 * 100.0  # accrued at h=8 just before exit
    t = result.trades[0]
    assert t.funding_collected == pytest.approx(expected_funding_accrued - expected_fees)


def test_open_position_recorded_at_backtest_end():
    """An open trade at the end of data must be recorded with backtest_end."""
    prices = build_prices([100.0] * 24)
    # Funding stays elevated through the entire backtest — never exits.
    funding = _funding_df([(_t(h), 0.0005) for h in range(0, 24, 8)])
    cfg = _base_cfg(exit_threshold=0.0, max_hold_hours=1000)
    result = run_backtest(funding, prices, "BTC", cfg, fee_rate=0.0)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "backtest_end"


def test_no_entry_when_funding_below_threshold():
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0001), (_t(8), 0.0001)])  # both well below 0.0002 threshold
    result = run_backtest(funding, prices, "BTC", _base_cfg(), fee_rate=0.0)
    assert len(result.trades) == 0


def test_equity_curve_length_matches_input_bars():
    prices = build_prices([100.0] * 24)
    funding = _funding_df([(_t(0), 0.0)])
    result = run_backtest(funding, prices, "BTC", _base_cfg(), fee_rate=0.0)
    assert len(result.equity_curve) == len(prices)
    assert not result.equity_curve.isna().any()
