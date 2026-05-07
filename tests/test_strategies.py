"""Tests for live strategy decision logic with the execution layer mocked.

Both FundingRateStrategy and RegimeSwitchingStrategy are tested by driving
their internal _check_entry / _maybe_exit / _maybe_enter methods directly,
or by stepping _tick() against a controlled FakeOrderManager.

These tests are critical because they cover the exact code path that fires
real market orders in production — any regression here is real money on the line.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from execution import Side
from risk import RiskConfig, RiskManager
from strategies.configs import FundingConfig, RegimeSwitchingConfig
from strategies.funding_rate import FundingRateStrategy
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_funding(funding_rate: float = 0.0,
                  mid: float = 100.0,
                  cfg_overrides: dict | None = None) -> tuple[FundingRateStrategy, FakeOrderManager]:
    info = FakeInfo(funding_rate=funding_rate, mid=mid)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    cfg = FundingConfig(**(cfg_overrides or {}))
    strat = FundingRateStrategy("BTC", om, risk, cfg)  # type: ignore[arg-type]
    return strat, om


def _make_regime(cfg_overrides: dict | None = None) -> tuple[RegimeSwitchingStrategy, FakeOrderManager]:
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    cfg = RegimeSwitchingConfig(**(cfg_overrides or {}))
    strat = RegimeSwitchingStrategy("BTC", om, risk, cfg)  # type: ignore[arg-type]
    return strat, om


def _snap(p_bear: float, p_chop: float, p_bull: float, er: float = 0.0) -> RegimeSnapshot:
    proba = np.array([p_bear, p_chop, p_bull], dtype=float)
    return RegimeSnapshot(
        regime=Regime(int(np.argmax(proba))),
        proba=proba,
        expected_return=er,
        expected_vol=0.01,
    )


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  FundingRateStrategy                                                         #
# --------------------------------------------------------------------------- #


def test_funding_enters_short_on_high_positive_funding():
    strat, om = _make_funding(funding_rate=0.0005, cfg_overrides={"entry_threshold": 0.0002})
    _run(strat._check_entry(0.0005, 100.0))
    assert strat._in_position is True
    assert strat._position_side == Side.SELL
    assert len(om.orders) == 1
    assert om.orders[0].side == Side.SELL
    # FundingConfig default position_size_usd=50, mid=100 → size 0.5
    assert om.orders[0].size == pytest.approx(0.5)


def test_funding_enters_long_on_high_negative_funding():
    strat, _om = _make_funding(funding_rate=-0.0005, cfg_overrides={"entry_threshold": 0.0002})
    _run(strat._check_entry(-0.0005, 100.0))
    assert strat._in_position is True
    assert strat._position_side == Side.BUY


def test_funding_skips_entry_below_threshold():
    strat, om = _make_funding(cfg_overrides={"entry_threshold": 0.0002})
    _run(strat._check_entry(0.0001, 100.0))   # below threshold
    assert strat._in_position is False
    assert om.orders == []


def test_funding_exit_on_normalised_rate():
    strat, _om = _make_funding(cfg_overrides={"entry_threshold": 0.0002, "exit_threshold": 0.00005})
    # Manually open a short.
    strat._in_position = True
    strat._position_side = Side.SELL
    strat._entry_price = 100.0
    strat._entry_time = datetime.now(timezone.utc)
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._check_exit(0.000001, 100.0))   # below exit_threshold

    assert strat._in_position is False


def test_funding_exit_on_flipped_funding():
    strat, _om = _make_funding(cfg_overrides={"entry_threshold": 0.0002, "exit_threshold": 0.0})
    strat._in_position = True
    strat._position_side = Side.SELL  # short receives positive funding
    strat._entry_price = 100.0
    strat._entry_time = datetime.now(timezone.utc)
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._check_exit(-0.0005, 100.0))    # funding flipped negative

    assert strat._in_position is False


def test_funding_stop_loss_long():
    """Long position with adverse 3% move should hit a 2% stop."""
    strat, _om = _make_funding(cfg_overrides={"stop_loss_pct": 0.02, "exit_threshold": 0.0})
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._entry_price = 100.0
    strat._entry_time = datetime.now(timezone.utc)
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._check_exit(-0.0005, 97.0))     # mid=97 → -3% on long

    assert strat._in_position is False


def test_funding_max_hold_force_close():
    strat, _om = _make_funding(cfg_overrides={"max_hold_hours": 1, "exit_threshold": 0.0})
    strat._in_position = True
    strat._position_side = Side.SELL
    strat._entry_price = 100.0
    # Entry was 2 hours ago.
    strat._entry_time = datetime.now(timezone.utc) - timedelta(hours=2)
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._check_exit(0.0005, 100.0))     # funding still strong

    assert strat._in_position is False


def test_funding_no_double_entry():
    """If already in position, _check_entry must not open a second order."""
    strat, om = _make_funding(funding_rate=0.0005, cfg_overrides={"entry_threshold": 0.0002})
    # Pretend we're already in.
    strat._in_position = True
    strat._position_side = Side.SELL

    # _tick branches on _in_position; calling _check_entry directly is the "wrong path"
    # but it should still not open a new order through the public _tick contract.
    _run(strat._tick())
    assert len(om.orders) == 0


# --------------------------------------------------------------------------- #
#  RegimeSwitchingStrategy                                                     #
# --------------------------------------------------------------------------- #


def test_regime_enters_long_on_high_p_bull_smoothed():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "min_expected_return_per_bar": 0.0,
        "max_chop_proba": 0.5,
    })
    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))

    assert strat._in_position is True
    assert strat._target_regime == Regime.BULL
    assert strat._position_side == Side.BUY
    assert len(om.orders) == 1
    assert om.orders[0].side == Side.BUY


def test_regime_enters_short_on_high_p_bear_smoothed():
    strat, _om = _make_regime(cfg_overrides={"signal_mode": "smoothed", "entry_proba": 0.65})
    snap = _snap(0.85, 0.10, 0.05, er=-0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is True
    assert strat._target_regime == Regime.BEAR
    assert strat._position_side == Side.SELL


def test_regime_blocks_entry_when_chop_dominant_smoothed():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.30,
        "max_chop_proba": 0.50,
    })
    # P(bull) > entry_proba but P(chop) > max_chop_proba.
    snap = _snap(0.10, 0.55, 0.35, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is False
    assert om.orders == []


def test_regime_viterbi_enters_on_label_below_proba_threshold():
    """In viterbi mode, the entry_proba threshold is bypassed in favour of MAP label."""
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "viterbi",
        "entry_proba": 0.85,
        "min_expected_return_per_bar": 0.0,
    })
    snap = _snap(0.30, 0.20, 0.50, er=0.001)
    _run(strat._maybe_enter(snap, Regime.BULL, 100.0))
    assert strat._in_position is True
    assert om.orders[0].side == Side.BUY


def test_regime_cooldown_blocks_same_regime_reentry():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "same_regime_cooldown_bars": 50,
    })
    # Pretend we just exited a bull trade.
    strat._last_exit_regime = Regime.BULL
    strat._last_exit_bar_index = 100
    strat._bar_index = 110   # only 10 bars elapsed, cooldown=50 still active

    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is False
    assert om.orders == []


def test_regime_cooldown_does_not_block_opposite_regime():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "same_regime_cooldown_bars": 50,
    })
    strat._last_exit_regime = Regime.BULL
    strat._last_exit_bar_index = 100
    strat._bar_index = 110

    # Strong bear after a recent bull exit — should be allowed.
    snap = _snap(0.85, 0.10, 0.05, er=-0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is True
    assert strat._target_regime == Regime.BEAR
    assert om.orders[0].side == Side.SELL


def test_regime_min_hold_blocks_soft_exit():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "exit_proba": 0.45,
        "min_hold_bars": 10,
        "stop_loss_pct": 1.0,        # disable stop
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._bar_index = 105
    strat._entry_bar_index = 100      # held 5 bars, below min_hold=10

    weak = _snap(0.30, 0.40, 0.30)
    _run(strat._maybe_exit(weak, None, 100.0))
    assert strat._in_position is True
    # Trying to close would have called market_order — assert no close was issued.
    assert om.orders == []


def test_regime_stop_loss_bypasses_min_hold():
    strat, _om = _make_regime(cfg_overrides={
        "stop_loss_pct": 0.05,
        "min_hold_bars": 100,         # very high — but stop should still fire
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._bar_index = 1
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    bull = _snap(0.05, 0.10, 0.85, er=0.001)
    # Mid = 90, that's -10% on a long → far below 5% stop.
    _run(strat._maybe_exit(bull, None, 90.0))
    assert strat._in_position is False


def test_regime_take_profit_fires_for_long():
    strat, _om = _make_regime(cfg_overrides={"take_profit_pct": 0.10})
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._bar_index = 50
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    bull = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_exit(bull, None, 115.0))   # +15% > 10% TP
    assert strat._in_position is False


def test_regime_viterbi_exit_on_label_flip():
    strat, _om = _make_regime(cfg_overrides={
        "signal_mode": "viterbi",
        "min_hold_bars": 0,
        "stop_loss_pct": 1.0,
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._bar_index = 10
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    snap = _snap(0.20, 0.60, 0.20)   # any soft posterior
    _run(strat._maybe_exit(snap, Regime.CHOP, 100.0))  # Viterbi flipped to CHOP
    assert strat._in_position is False


def test_regime_records_last_exit_state_on_close():
    strat, _om = _make_regime()
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._bar_index = 50
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._close_position())
    assert strat._in_position is False
    assert strat._last_exit_regime == Regime.BULL
    assert strat._last_exit_bar_index == 50
