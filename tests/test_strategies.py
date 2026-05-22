"""Tests for live strategy decision logic with the execution layer mocked.

The RegimeSwitchingStrategy is tested by driving its internal
_maybe_enter / _maybe_exit methods directly, or by stepping _tick() against a
controlled FakeOrderManager.

These tests are critical because they cover the exact code path that fires
real market orders in production — any regression here is real money on the line.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from execution import Side
from risk import RiskConfig, RiskManager
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


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
#  RegimeSwitchingStrategy                                                     #
# --------------------------------------------------------------------------- #


def test_regime_enters_long_on_high_p_bull_smoothed():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "min_expected_return_per_bar": 0.0,
        "max_chop_proba": 0.5,
        "entry_confirmation_bars": 0,
    })
    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))

    assert strat._in_position is True
    assert strat._target_regime == Regime.BULL
    assert strat._position_side == Side.BUY
    assert len(om.orders) == 1
    assert om.orders[0].side == Side.BUY


def test_regime_enters_short_on_high_p_bear_smoothed():
    strat, _om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed", "entry_proba": 0.65,
        "entry_confirmation_bars": 0, "entry_confirmation_bars_short": 0,
    })
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
        "entry_confirmation_bars": 0,
    })
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
        "entry_confirmation_bars": 0,
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
        "entry_confirmation_bars": 0,
    })
    strat._last_exit_regime = Regime.BULL
    strat._last_exit_bar_index = 100
    strat._bar_index = 110

    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is False
    assert om.orders == []


def test_regime_cooldown_does_not_block_opposite_regime():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "same_regime_cooldown_bars": 50,
        "entry_confirmation_bars": 0,
        "entry_confirmation_bars_short": 0,
    })
    strat._last_exit_regime = Regime.BULL
    strat._last_exit_bar_index = 100
    strat._bar_index = 110

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


def test_regime_confirmation_bars_blocks_premature_entry():
    """Streak < entry_confirmation_bars must suppress entry."""
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "entry_confirmation_bars": 5,
    })
    # Only 2 bars of streak — below the 5-bar requirement.
    strat._streak_target = Regime.BULL
    strat._streak_bars = 2

    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is False
    assert om.orders == []


def test_regime_confirmation_bars_allows_sustained_entry():
    """Streak ≥ entry_confirmation_bars must allow entry."""
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "entry_proba": 0.65,
        "entry_confirmation_bars": 5,
    })
    strat._streak_target = Regime.BULL
    strat._streak_bars = 6   # one over the requirement

    snap = _snap(0.05, 0.10, 0.85, er=0.001)
    _run(strat._maybe_enter(snap, None, 100.0))
    assert strat._in_position is True
    assert om.orders[0].side == Side.BUY


def test_update_streak_increments_on_same_target():
    strat, _om = _make_regime()
    strat._update_streak(Regime.BULL)
    strat._update_streak(Regime.BULL)
    strat._update_streak(Regime.BULL)
    assert strat._streak_target == Regime.BULL
    assert strat._streak_bars == 3


def test_update_streak_resets_on_target_flip():
    strat, _om = _make_regime()
    strat._update_streak(Regime.BULL)
    strat._update_streak(Regime.BULL)
    strat._update_streak(Regime.BEAR)   # different regime → reset to 1
    assert strat._streak_target == Regime.BEAR
    assert strat._streak_bars == 1


def test_update_streak_resets_on_no_candidate():
    strat, _om = _make_regime()
    strat._update_streak(Regime.BULL)
    strat._update_streak(None)          # no candidate → reset to 0
    assert strat._streak_target is None
    assert strat._streak_bars == 0


# --------------------------------------------------------------------------- #
#  Staged exit                                                                 #
# --------------------------------------------------------------------------- #


def test_staged_exit_trims_position_on_soft_weakening():
    """A held posterior in [exit_proba, soft_exit_proba) must trigger one
    partial reduce, not a full exit, and only fire once per position."""
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "exit_proba": 0.45,
        "soft_exit_proba": 0.60,
        "staged_exit_fraction": 0.5,
        "min_hold_bars": 0,
        "stop_loss_pct": 1.0,
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_size = 1.0
    strat._bar_index = 10
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    # Posterior at 0.55 — below soft_exit_proba (0.60) but above exit_proba (0.45).
    softening = _snap(0.20, 0.25, 0.55)
    _run(strat._maybe_exit(softening, None, 100.0))

    # Position stays open; exactly one reduce order at half size.
    assert strat._in_position is True
    assert strat._soft_exit_done is True
    assert len(om.orders) == 1
    assert om.orders[0].side == Side.SELL
    assert om.orders[0].size == 0.5

    # A second softening tick should not re-fire the partial.
    _run(strat._maybe_exit(softening, None, 100.0))
    assert len(om.orders) == 1


def test_staged_exit_does_not_fire_below_min_hold():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "exit_proba": 0.45,
        "soft_exit_proba": 0.60,
        "min_hold_bars": 50,
        "stop_loss_pct": 1.0,
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_size = 1.0
    strat._bar_index = 5
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    softening = _snap(0.20, 0.25, 0.55)
    _run(strat._maybe_exit(softening, None, 100.0))
    assert om.orders == []
    assert strat._soft_exit_done is False


def test_staged_exit_disabled_when_soft_exit_proba_none():
    strat, om = _make_regime(cfg_overrides={
        "signal_mode": "smoothed",
        "exit_proba": 0.45,
        "soft_exit_proba": None,
        "min_hold_bars": 0,
        "stop_loss_pct": 1.0,
    })
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_size = 1.0
    strat._bar_index = 10
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    # Same softening level — no staged exit when soft_exit_proba is disabled.
    softening = _snap(0.20, 0.25, 0.55)
    _run(strat._maybe_exit(softening, None, 100.0))
    assert om.orders == []


def test_full_exit_resets_soft_exit_flag():
    """After a full exit, _reset clears _soft_exit_done so the next position
    starts fresh."""
    strat, _om = _make_regime()
    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_size = 1.0
    strat._bar_index = 10
    strat._entry_bar_index = 0
    strat._soft_exit_done = True
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._close_position())
    assert strat._soft_exit_done is False


# --------------------------------------------------------------------------- #
#  Vol-scaled sizing                                                           #
# --------------------------------------------------------------------------- #


def test_vol_scaled_sizing_increases_notional_when_vol_is_low():
    """vol_target / expected_vol > 1 → notional grows (clipped to max_vol_scale)."""
    strat, _ = _make_regime(cfg_overrides={
        "position_size_usd": 100.0,
        "vol_target_per_bar": 0.01,
        "min_vol_scale": 0.5,
        "max_vol_scale": 2.0,
    })
    # Vol half the target → scale = 2.0 (caps at max_vol_scale=2.0).
    base = strat._resolve_base_notional(expected_vol=0.005)
    assert base == pytest.approx(200.0)


def test_vol_scaled_sizing_shrinks_notional_when_vol_is_high():
    strat, _ = _make_regime(cfg_overrides={
        "position_size_usd": 100.0,
        "vol_target_per_bar": 0.01,
        "min_vol_scale": 0.5,
        "max_vol_scale": 2.0,
    })
    # Vol double the target → scale = 0.5.
    base = strat._resolve_base_notional(expected_vol=0.02)
    assert base == pytest.approx(50.0)


def test_vol_scaling_disabled_when_target_is_none():
    strat, _ = _make_regime(cfg_overrides={
        "position_size_usd": 100.0,
        "vol_target_per_bar": None,
    })
    assert strat._resolve_base_notional(expected_vol=0.005) == pytest.approx(100.0)
    assert strat._resolve_base_notional(expected_vol=0.05) == pytest.approx(100.0)


def test_vol_scaling_no_op_when_expected_vol_nonpositive():
    """Defensive: a zero expected_vol must not blow up the size calc."""
    strat, _ = _make_regime(cfg_overrides={
        "position_size_usd": 100.0, "vol_target_per_bar": 0.01,
    })
    assert strat._resolve_base_notional(expected_vol=0.0) == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
#  Drift monitor                                                               #
# --------------------------------------------------------------------------- #


def test_drift_monitor_runs_when_breach_detected():
    """Smoke: a breach exercises the alert path without raising."""
    strat, _ = _make_regime(cfg_overrides={"regime_drift_alert_pct": 0.5})
    strat._in_position = True
    strat._target_regime = Regime.BULL
    strat._entry_state_means = (-0.003, 0.0, 0.005)    # entry-time bull_µ = 0.005

    # bull_µ shifts to 0.001 → drift = |0.001 - 0.005| / 0.005 = 80% (> 50%).
    new_means = np.array([-0.003, 0.0, 0.001])
    strat._check_regime_drift(new_means)


def test_drift_monitor_quiet_below_threshold():
    """A small drift must not raise or log ERROR."""
    strat, _ = _make_regime(cfg_overrides={"regime_drift_alert_pct": 0.5})
    strat._in_position = True
    strat._target_regime = Regime.BULL
    strat._entry_state_means = (-0.003, 0.0, 0.005)
    # 10% drift — below threshold.
    new_means = np.array([-0.003, 0.0, 0.0045])
    # Just assert no exception.
    strat._check_regime_drift(new_means)


def test_drift_monitor_skipped_when_flat():
    """No held position → no baseline to compare against → silent no-op."""
    strat, _ = _make_regime()
    strat._in_position = False
    strat._target_regime = None
    # Even with absurd means and no baseline, the call must return cleanly.
    strat._check_regime_drift(np.array([0.0, 0.0, 100.0]))


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
