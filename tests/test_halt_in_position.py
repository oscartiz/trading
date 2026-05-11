"""Regression test for the one-way drawdown halt contract.

When the risk manager halts the bot (drawdown breach), new entries must be
blocked but any held position must still exit cleanly on stop / regime /
time-cap. Otherwise capital can't be de-risked after a breach.

The strategy's exit path doesn't route through ``RiskManager.check_order``
(only entries do), so this test is the regression guard for that contract.
"""
from __future__ import annotations

import asyncio

import numpy as np

from execution import Side
from risk import RiskConfig, RiskManager
from runtime import StateStore, TradeJournal
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager, FakeOrderResult


def _build_halted_long_position(tmp_path, *, mid_at_entry: float = 100.0):
    """Strategy with: open long @ mid_at_entry, risk manager already halted."""
    info = FakeInfo(mid=mid_at_entry)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(
        max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=0.05,
    ))
    state = StateStore("regime_switching", "BTC", root=tmp_path)
    journal = TradeJournal("BTC", root=tmp_path)
    cfg = RegimeSwitchingConfig(min_hold_bars=0, max_hold_bars=10)
    s = RegimeSwitchingStrategy(    # type: ignore[arg-type]
        "BTC", om, risk, cfg, state_store=state, journal=journal,
    )

    # Seed: open long at $100 on bar 0; we're now somewhere later.
    s._in_position = True
    s._entry_price = mid_at_entry
    s._position_side = Side.BUY
    s._target_regime = Regime.BULL
    s._entry_bar_index = 0
    s._bar_index = 5
    om.position = {"coin": "BTC", "szi": "0.5"}

    # Halt the risk manager by feeding it a drawdown breach.
    risk.update_equity(1_000.0)
    risk.update_equity(800.0)        # 20% drawdown >> 5% limit
    assert risk.is_halted() is True

    return s, om, journal


def _flat_snapshot() -> RegimeSnapshot:
    """A noisy snapshot — values mostly don't matter, stop fires first."""
    return RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.10, 0.50, 0.40]),
        expected_return=0.0,
        expected_vol=0.01,
    )


def _weakened_bull_snapshot() -> RegimeSnapshot:
    """Held-regime posterior below exit threshold (0.45) — fires regime_weakening."""
    return RegimeSnapshot(
        regime=Regime.CHOP,
        proba=np.array([0.20, 0.50, 0.30]),    # P(bull)=0.30 < 0.45
        expected_return=0.0,
        expected_vol=0.01,
    )


# --------------------------------------------------------------------------- #
#  Exits fire despite halt                                                     #
# --------------------------------------------------------------------------- #


def test_stop_loss_fires_while_halted(tmp_path):
    s, om, journal = _build_halted_long_position(tmp_path, mid_at_entry=100.0)
    om.next_result = FakeOrderResult(success=True, order_id=99, fill_price=80.0)

    # Long @ $100, mid drops to $80 = -20% (stop is 12%) — stop fires regardless of min_hold.
    asyncio.run(s._maybe_exit(_flat_snapshot(), None, mid=80.0))

    assert len(om.orders) == 1, "halted bot must still close the position"
    assert om.orders[0].side == Side.SELL
    assert s._in_position is False

    rows = journal.read_all()
    assert len(rows) == 1
    assert rows[0]["event"] == "exit"
    assert "stop loss" in rows[0]["exit_reason"]


def test_regime_weakened_fires_while_halted(tmp_path):
    s, om, journal = _build_halted_long_position(tmp_path, mid_at_entry=100.0)
    om.next_result = FakeOrderResult(success=True, order_id=100, fill_price=101.0)

    # P&L flat-ish (101 vs 100 → +1%), held-regime posterior 0.30 < 0.45 → soft exit.
    asyncio.run(s._maybe_exit(_weakened_bull_snapshot(), None, mid=101.0))

    assert len(om.orders) == 1
    assert s._in_position is False
    row = journal.read_all()[0]
    assert "regime weakening" in row["exit_reason"]


def test_max_hold_fires_while_halted(tmp_path):
    s, om, journal = _build_halted_long_position(tmp_path, mid_at_entry=100.0)
    om.next_result = FakeOrderResult(success=True, order_id=101, fill_price=100.5)

    # Held regime still strong (P(bull)=0.60 > 0.45) so regime_weakening can't
    # fire; only the bar cursor exceeding max_hold can trigger the exit.
    strong_bull = RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.20, 0.20, 0.60]),
        expected_return=0.0,
        expected_vol=0.01,
    )
    s._bar_index = 100
    asyncio.run(s._maybe_exit(strong_bull, None, mid=100.5))

    assert len(om.orders) == 1
    assert s._in_position is False
    row = journal.read_all()[0]
    assert "max hold" in row["exit_reason"]


# --------------------------------------------------------------------------- #
#  Entries remain blocked while halted (the other half of the contract)       #
# --------------------------------------------------------------------------- #


def test_entries_blocked_while_halted(tmp_path):
    """The flip side: halted bot must NOT open new positions."""
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(
        max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=0.05,
    ))
    state = StateStore("regime_switching", "BTC", root=tmp_path)
    journal = TradeJournal("BTC", root=tmp_path)
    cfg = RegimeSwitchingConfig()
    s = RegimeSwitchingStrategy(    # type: ignore[arg-type]
        "BTC", om, risk, cfg, state_store=state, journal=journal,
    )
    s._streak_target = Regime.BULL
    s._streak_bars = 100

    risk.update_equity(1_000.0)
    risk.update_equity(800.0)
    assert risk.is_halted() is True

    bull = RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.05, 0.10, 0.85]),
        expected_return=0.001,
        expected_vol=0.01,
    )
    asyncio.run(s._maybe_enter(bull, None, mid=100.0))

    assert om.orders == [], "halt must block new entries"
    assert s._in_position is False
    # No journal row should be written for a blocked entry (check_order returns before order).
    assert journal.read_all() == []
