"""Tests for runtime.state.StateStore and per-strategy state persistence."""
from __future__ import annotations

import asyncio

from execution import Side
from risk import RiskConfig, RiskManager
from runtime import StateStore
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager


# --------------------------------------------------------------------------- #
#  StateStore                                                                  #
# --------------------------------------------------------------------------- #


def test_state_store_load_returns_empty_when_missing(tmp_path):
    store = StateStore("foo", "BTC", root=tmp_path)
    assert store.load() == {}


def test_state_store_round_trip(tmp_path):
    store = StateStore("foo", "BTC", root=tmp_path)
    payload = {"a": 1, "b": "hello", "c": None, "d": [1, 2, 3]}
    store.save(payload)
    assert store.load() == payload


def test_state_store_overwrites_atomically(tmp_path):
    store = StateStore("foo", "BTC", root=tmp_path)
    store.save({"v": 1})
    store.save({"v": 2})
    assert store.load() == {"v": 2}
    # No leftover .tmp file
    assert not list(tmp_path.glob("*.tmp"))


def test_state_store_clear_removes_file(tmp_path):
    store = StateStore("foo", "BTC", root=tmp_path)
    store.save({"v": 1})
    assert store.path.exists()
    store.clear()
    assert not store.path.exists()


def test_state_store_handles_corrupt_json(tmp_path):
    store = StateStore("foo", "BTC", root=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json")
    assert store.load() == {}


def test_state_store_isolates_by_name_and_coin(tmp_path):
    a = StateStore("strat_a", "BTC", root=tmp_path)
    b = StateStore("strat_b", "BTC", root=tmp_path)
    c = StateStore("strat_a", "ETH", root=tmp_path)
    a.save({"who": "a-btc"})
    b.save({"who": "b-btc"})
    c.save({"who": "a-eth"})
    assert a.load() == {"who": "a-btc"}
    assert b.load() == {"who": "b-btc"}
    assert c.load() == {"who": "a-eth"}


# --------------------------------------------------------------------------- #
#  Regime strategy persistence                                                 #
# --------------------------------------------------------------------------- #


def _make_regime(state: StateStore) -> tuple[RegimeSwitchingStrategy, FakeOrderManager]:
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    cfg = RegimeSwitchingConfig()
    return RegimeSwitchingStrategy("BTC", om, risk, cfg, state_store=state), om  # type: ignore[arg-type]


def test_regime_state_persists_full_position(tmp_path):
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s1, _ = _make_regime(store)
    s1._bar_index = 42
    s1._last_bar_open_ms = 1_700_000_000_000
    s1._bars_since_fit = 7
    s1._in_position = True
    s1._entry_price = 100.5
    s1._entry_bar_index = 40
    s1._position_side = Side.BUY
    s1._target_regime = Regime.BULL
    s1._streak_target = Regime.BULL
    s1._streak_bars = 3
    s1._last_exit_regime = Regime.BEAR
    s1._last_exit_bar_index = 25
    s1._save_state()

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s2, _ = _make_regime(store2)
    assert s2._bar_index == 42
    assert s2._last_bar_open_ms == 1_700_000_000_000
    assert s2._bars_since_fit == 7
    assert s2._in_position is True
    assert s2._entry_price == 100.5
    assert s2._entry_bar_index == 40
    assert s2._position_side == Side.BUY
    assert s2._target_regime == Regime.BULL
    assert s2._streak_target == Regime.BULL
    assert s2._streak_bars == 3
    assert s2._last_exit_regime == Regime.BEAR
    assert s2._last_exit_bar_index == 25


def test_regime_state_persists_cooldown_when_flat(tmp_path):
    """After exit, last_exit_regime and last_exit_bar_index must survive a restart
    so the cooldown gate keeps blocking re-entries."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s1, _ = _make_regime(store)
    s1._bar_index = 100
    s1._last_exit_regime = Regime.BULL
    s1._last_exit_bar_index = 95   # exited 5 bars ago
    s1._in_position = False
    s1._save_state()

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s2, _ = _make_regime(store2)
    assert s2._in_position is False
    assert s2._last_exit_regime == Regime.BULL
    assert s2._last_exit_bar_index == 95
    assert s2._bar_index == 100


def test_regime_state_handles_none_enums(tmp_path):
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s1, _ = _make_regime(store)
    # All enum fields None — pristine startup
    s1._save_state()
    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s2, _ = _make_regime(store2)
    assert s2._target_regime is None
    assert s2._last_exit_regime is None
    assert s2._streak_target is None
    assert s2._position_side is None


def test_regime_default_state_store_uses_default_dir(monkeypatch, tmp_path):
    """When no state_store is injected, the strategy creates one rooted at
    runtime.state.DEFAULT_STATE_DIR (which the autouse fixture has already
    redirected to tmp). This guards against accidental hard-coded paths."""
    from runtime import state as state_mod
    target = tmp_path / "redirected"
    monkeypatch.setattr(state_mod, "DEFAULT_STATE_DIR", target)

    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    s = RegimeSwitchingStrategy("BTC", om, risk, RegimeSwitchingConfig())  # type: ignore[arg-type]
    s._save_state()
    assert (target / "regime_switching_BTC.json").exists()


# --------------------------------------------------------------------------- #
#  Startup reconciliation — covers shared base-class behaviour                 #
# --------------------------------------------------------------------------- #


def _seed_open_position(s: RegimeSwitchingStrategy, side: Side) -> None:
    s._in_position = True
    s._position_side = side
    s._entry_price = 100.0
    s._entry_bar_index = 0
    s._target_regime = Regime.BULL if side == Side.BUY else Regime.BEAR
    s._save_state()


def test_reconcile_clean_startup_does_not_block(tmp_path):
    """No saved state, no exchange position — strategy is free to trade."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store)
    om.position = None
    s._check_existing_position()
    assert s._block_entries is False


def test_reconcile_matched_long_position_does_not_block(tmp_path):
    """Saved state says we're long; exchange agrees. Recovery is clean."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    pre, _ = _make_regime(store)
    _seed_open_position(pre, Side.BUY)

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store2)
    om.position = {"coin": "BTC", "szi": "0.5"}    # positive = long
    s._check_existing_position()
    assert s._block_entries is False


def test_reconcile_matched_short_position_does_not_block(tmp_path):
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    pre, _ = _make_regime(store)
    _seed_open_position(pre, Side.SELL)

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store2)
    om.position = {"coin": "BTC", "szi": "-0.5"}   # negative = short
    s._check_existing_position()
    assert s._block_entries is False


def test_reconcile_orphan_exchange_position_blocks_entries(tmp_path):
    """No saved state but exchange has a position — refuse to add to it."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store)
    om.position = {"coin": "BTC", "szi": "0.3"}
    s._check_existing_position()
    assert s._block_entries is True


def test_reconcile_phantom_state_blocks_entries(tmp_path):
    """Saved state says we're long but exchange is flat — someone closed under us."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    pre, _ = _make_regime(store)
    _seed_open_position(pre, Side.BUY)

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store2)
    om.position = None
    s._check_existing_position()
    assert s._block_entries is True


def test_reconcile_side_mismatch_blocks_entries(tmp_path):
    """Saved state says long but exchange is short — hard mismatch."""
    store = StateStore("regime_switching", "BTC", root=tmp_path)
    pre, _ = _make_regime(store)
    _seed_open_position(pre, Side.BUY)

    store2 = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store2)
    om.position = {"coin": "BTC", "szi": "-0.5"}
    s._check_existing_position()
    assert s._block_entries is True


def test_blocked_entries_skip_regime_maybe_enter(tmp_path):
    """Entries blocked → _maybe_enter must not place an order even on a clear bull signal."""
    import numpy as np

    from strategies.regime import RegimeSnapshot

    store = StateStore("regime_switching", "BTC", root=tmp_path)
    s, om = _make_regime(store)
    s._block_entries = True
    s._streak_target = Regime.BULL
    s._streak_bars = 100   # any confirmation bar requirement is met

    snap = RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.05, 0.10, 0.85]),
        expected_return=0.001,
        expected_vol=0.01,
    )
    asyncio.run(s._maybe_enter(snap, None, 100.0))
    assert s._in_position is False
    assert om.orders == []
