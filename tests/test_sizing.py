"""Tests for equity-relative position sizing."""
from __future__ import annotations

import asyncio

import numpy as np

from risk import RiskConfig, RiskManager
from runtime import StateStore, TradeJournal
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager, FakeOrderResult


def _build_strategy(
    tmp_path,
    *,
    pct: float | None = None,
    equity_source=None,
    fixed_usd: float = 100.0,
):
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    state = StateStore("regime_switching", "BTC", root=tmp_path)
    journal = TradeJournal("BTC", root=tmp_path)
    cfg = RegimeSwitchingConfig(
        position_size_usd=fixed_usd,
        position_size_pct=pct,
    )
    s = RegimeSwitchingStrategy(    # type: ignore[arg-type]
        "BTC", om, risk, cfg,
        state_store=state, journal=journal,
        equity_source=equity_source,
    )
    return s, om


def _bull_snapshot() -> RegimeSnapshot:
    return RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.05, 0.10, 0.85]),
        expected_return=0.001,
        expected_vol=0.01,
    )


def test_default_uses_fixed_usd(tmp_path):
    """No pct, no equity_source — falls back to position_size_usd."""
    s, _ = _build_strategy(tmp_path, fixed_usd=250.0)
    assert s._resolve_base_notional() == 250.0


def test_pct_sized_when_equity_source_available(tmp_path):
    s, _ = _build_strategy(
        tmp_path, pct=0.10, equity_source=lambda: 5_000.0, fixed_usd=100.0,
    )
    # 10% of $5000 equity = $500 base notional (before probability scaling).
    assert s._resolve_base_notional() == 500.0


def test_pct_without_equity_source_falls_back(tmp_path):
    """pct set but no equity_source wired — backtest/paper path."""
    s, _ = _build_strategy(tmp_path, pct=0.10, equity_source=None, fixed_usd=100.0)
    assert s._resolve_base_notional() == 100.0


def test_pct_falls_back_when_equity_source_raises(tmp_path):
    def _boom():
        raise RuntimeError("api timeout")
    s, _ = _build_strategy(
        tmp_path, pct=0.10, equity_source=_boom, fixed_usd=100.0,
    )
    assert s._resolve_base_notional() == 100.0


def test_pct_falls_back_when_equity_is_zero_or_negative(tmp_path):
    """A drawn-down account or wallet read returning 0 must not zero out trades."""
    s, _ = _build_strategy(
        tmp_path, pct=0.10, equity_source=lambda: 0.0, fixed_usd=100.0,
    )
    assert s._resolve_base_notional() == 100.0
    s2, _ = _build_strategy(
        tmp_path, pct=0.10, equity_source=lambda: -50.0, fixed_usd=100.0,
    )
    assert s2._resolve_base_notional() == 100.0


def test_entry_uses_pct_notional_end_to_end(tmp_path):
    """The full _maybe_enter path applies pct sizing × probability scaling."""
    s, om = _build_strategy(
        tmp_path, pct=0.10, equity_source=lambda: 1_000.0, fixed_usd=100.0,
    )
    om.next_result = FakeOrderResult(success=True, order_id=1, fill_price=100.0)
    s._streak_target = Regime.BULL
    s._streak_bars = 100

    asyncio.run(s._maybe_enter(_bull_snapshot(), None, mid=100.0))

    # base = 1000 * 0.10 = $100 ; proba_scale = max(0.85, 0.5) = 0.85 ; notional = $85
    # size = 85 / 100 = 0.85
    assert len(om.orders) == 1
    assert om.orders[0].size == 0.85


def test_entry_skipped_when_size_rounds_to_zero(tmp_path):
    """Tiny notional / large mid combinations must not send a 0-size order."""
    # mid is high (BTC scale) and equity is tiny → rounded size = 0
    s, om = _build_strategy(
        tmp_path, pct=0.10, equity_source=lambda: 0.001, fixed_usd=0.001,
    )
    s._streak_target = Regime.BULL
    s._streak_bars = 100

    asyncio.run(s._maybe_enter(_bull_snapshot(), None, mid=50_000.0))

    # No order should have been placed.
    assert om.orders == []
    assert s._in_position is False
