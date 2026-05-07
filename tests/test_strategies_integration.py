"""Integration-style tests that exercise full _tick() / _warm_up() cycles.

The live RegimeSwitchingStrategy threads candles → returns → classifier → orders.
These tests inject candles via FakeInfo and stub the classifier so we can verify
the full per-tick code path including bookkeeping, refit cadence, and order
emission, without ever invoking the real HMM.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from execution import Side
from risk import RiskConfig, RiskManager
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy
from strategies.configs import RegimeSwitchingConfig

from tests.conftest import FakeInfo, FakeOrderManager
from tests.test_regime_engine import StubClassifier, make_snap


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


def _candle(open_ms: int, close: float) -> dict:
    return {"t": open_ms, "T": open_ms + 3_600_000 - 1, "o": close, "h": close,
            "l": close, "c": close, "v": 1.0, "n": 1}


def _make_strategy(
    closes: list[float],
    classifier: StubClassifier,
    cfg_overrides: dict | None = None,
) -> tuple[RegimeSwitchingStrategy, FakeOrderManager]:
    """Build a strategy with pre-loaded candles in the warm-up window (last N hours).

    `_warm_up()` requests `(now - train_window_bars*1h, now)` so candle timestamps
    are anchored to that window. After warm-up `_last_bar_open_ms` will be the
    most recent timestamp, so callers inject new candles via the helpers below.
    """
    cfg_kwargs = dict(
        train_window_bars=10,
        refit_every_bars=10,
        signal_mode="smoothed",
        min_hold_bars=0,
        same_regime_cooldown_bars=0,
        position_size_usd=100.0,
        stop_loss_pct=0.10,
        take_profit_pct=None,
    )
    cfg_kwargs.update(cfg_overrides or {})
    cfg = RegimeSwitchingConfig(**cfg_kwargs)

    n = len(closes)
    now_ms = int(time.time() * 1000)
    # Distribute the warm-up candles inside the request window (now - n*1h, now].
    candles = [
        _candle(now_ms - (n - i) * 3_600_000, closes[i])
        for i in range(n)
    ]
    info = FakeInfo(mid=closes[-1], candles=candles)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))

    strat = RegimeSwitchingStrategy("BTC", om, risk, cfg)  # type: ignore[arg-type]
    strat._classifier = classifier   # type: ignore[assignment]
    return strat, om


def _prime_strategy_skipping_warm_up(
    strat: RegimeSwitchingStrategy,
    seed_closes: list[float],
    last_bar_offset_h: int = 6,
) -> int:
    """Set up `strat` as if warm-up already ran, leaving room for new candles.

    Pre-loads the closes deque and pins `_last_bar_open_ms` to `now - last_bar_offset_h`,
    so subsequent injected candles between (now - last_bar_offset_h, now - 1h] are
    accepted by `_poll_new_bars`. Returns the offset in ms for fixture math.
    """
    now_ms = int(time.time() * 1000)
    for c in seed_closes:
        strat._closes.append(float(c))
    strat._last_bar_open_ms = now_ms - last_bar_offset_h * 3_600_000
    return now_ms


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  _warm_up                                                                    #
# --------------------------------------------------------------------------- #


def test_warm_up_populates_closes_and_does_an_initial_fit():
    # train_window_bars=20 needs n_states*5=15 returns, so 16 closes is enough.
    closes = [100.0 + i for i in range(16)]
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)] * 5)
    strat, _om = _make_strategy(
        closes, classifier, cfg_overrides={"train_window_bars": 20},
    )

    _run(strat._warm_up())

    # All 16 closes fit in a deque of capacity 20.
    assert len(strat._closes) == len(closes)
    assert list(strat._closes)[-1] == closes[-1]
    # _refit must have called fit() now that there are >=15 returns.
    assert classifier.fit_calls == 1
    assert strat._last_bar_open_ms is not None


def test_warm_up_raises_when_no_candles_returned():
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)])
    info = FakeInfo(candles=[])
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    cfg = RegimeSwitchingConfig(train_window_bars=10)
    strat = RegimeSwitchingStrategy("BTC", om, risk, cfg)  # type: ignore[arg-type]
    strat._classifier = classifier  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="no candles"):
        _run(strat._warm_up())


# --------------------------------------------------------------------------- #
#  _poll_new_bars                                                              #
# --------------------------------------------------------------------------- #


def test_poll_new_bars_drops_still_forming_bar():
    """The currently-forming bar (open_time too close to now) must be excluded."""
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)])
    strat, _om = _make_strategy([100.0] * 5, classifier)
    now_ms = _prime_strategy_skipping_warm_up(strat, [100.0] * 10, last_bar_offset_h=6)
    strat.orders.info.candles.clear()    # remove the warm-up candles

    # Inject one already-closed bar (3h ago) and one that's still forming (now).
    closed_bar = _candle(now_ms - 3 * 3_600_000, 150.0)
    forming_bar = _candle(now_ms, 200.0)
    strat.orders.info.candles.extend([closed_bar, forming_bar])

    new = strat._poll_new_bars()
    # Only the closed bar should be returned; the forming one is excluded.
    assert len(new) == 1
    assert new[0]["c"] == 150.0


# --------------------------------------------------------------------------- #
#  _tick — entry, refit cadence, exit                                          #
# --------------------------------------------------------------------------- #


def test_tick_with_no_new_bars_is_noop():
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)])
    strat, om = _make_strategy([100.0] * 5, classifier)
    _prime_strategy_skipping_warm_up(strat, [100.0] * 10, last_bar_offset_h=6)
    # Strategy info has the warm-up candles loaded; clear them so no new bars exist.
    strat.orders.info.candles.clear()

    initial_orders = len(om.orders)
    initial_bar = strat._bar_index
    _run(strat._tick())
    assert len(om.orders) == initial_orders
    assert strat._bar_index == initial_bar


def test_tick_advances_bar_index_and_refits_on_cadence():
    """Each new closed bar increments bar_index; refit fires when bars_since_fit >= refit_every."""
    # Need plenty of room — set last_bar back ~7h so we can inject 5 hourly bars.
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)] * 10)
    strat, _om = _make_strategy(
        [100.0] * 5, classifier,
        cfg_overrides={"refit_every_bars": 3, "train_window_bars": 30},
    )
    # Pre-populate enough returns that _refit's size guard passes.
    seed = [100.0 + 0.01 * i for i in range(20)]
    now_ms = _prime_strategy_skipping_warm_up(strat, seed, last_bar_offset_h=7)
    strat.orders.info.candles.clear()

    fits_before = classifier.fit_calls
    # Inject 5 hourly bars in the (last_bar_open=now-7h, now-1h] window.
    for i in range(5):
        ts = now_ms - (6 - i) * 3_600_000   # now-6h, now-5h, ..., now-2h
        strat.orders.info.candles.append(_candle(ts, 100.0 + i))

    _run(strat._tick())

    assert strat._bar_index == 5
    assert classifier.fit_calls > fits_before


def test_tick_enters_long_when_smoothed_posterior_strong():
    """A strong-bull snapshot at the latest bar must trigger a market BUY."""
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    classifier = StubClassifier([bull] * 5)
    strat, om = _make_strategy(
        [100.0] * 5, classifier,
        cfg_overrides={
            "entry_proba": 0.65,
            "max_chop_proba": 0.5,
            "min_expected_return_per_bar": 0.0,
            "train_window_bars": 30,
        },
    )
    seed = [100.0 + 0.01 * i for i in range(20)]
    now_ms = _prime_strategy_skipping_warm_up(strat, seed, last_bar_offset_h=4)
    strat.orders.info.candles.clear()
    strat.orders.info.candles.append(_candle(now_ms - 3 * 3_600_000, 100.0))

    _run(strat._tick())

    assert strat._in_position is True
    assert strat._target_regime == Regime.BULL
    assert strat._position_side == Side.BUY
    assert len(om.orders) == 1
    assert om.orders[0].side == Side.BUY


def test_tick_does_not_enter_when_already_in_position():
    """If _in_position is True, _tick must run the exit branch only — no new orders."""
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    classifier = StubClassifier([bull] * 5)
    strat, om = _make_strategy(
        [100.0] * 5, classifier, cfg_overrides={"train_window_bars": 30},
    )
    seed = [100.0 + 0.01 * i for i in range(20)]
    now_ms = _prime_strategy_skipping_warm_up(strat, seed, last_bar_offset_h=4)
    strat.orders.info.candles.clear()
    strat.orders.info.candles.append(_candle(now_ms - 3 * 3_600_000, 100.0))

    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_bar_index = 0

    _run(strat._tick())
    assert len(om.orders) == 0     # no new entries


def test_tick_exits_when_smoothed_posterior_weakens():
    """An entered long must exit when the posterior drops below exit_proba."""
    weak = make_snap(0.30, 0.40, 0.30)
    classifier = StubClassifier([weak] * 5)
    strat, om = _make_strategy(
        [100.0] * 5, classifier,
        cfg_overrides={
            "exit_proba": 0.45, "min_hold_bars": 0, "stop_loss_pct": 1.0,
            "train_window_bars": 30,
        },
    )
    seed = [100.0 + 0.01 * i for i in range(20)]
    now_ms = _prime_strategy_skipping_warm_up(strat, seed, last_bar_offset_h=4)
    strat.orders.info.candles.clear()
    strat.orders.info.candles.append(_candle(now_ms - 3 * 3_600_000, 100.0))

    strat._in_position = True
    strat._position_side = Side.BUY
    strat._target_regime = Regime.BULL
    strat._entry_price = 100.0
    strat._entry_bar_index = 0
    strat.orders.position = {"szi": "1.0", "coin": "BTC"}

    _run(strat._tick())

    assert strat._in_position is False
    assert any(o.side == Side.SELL for o in om.orders)


def test_classify_now_returns_none_with_too_few_returns():
    """If the closes deque is too small, _classify_now must return (None, None)."""
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)])
    strat, _om = _make_strategy([100.0, 101.0], classifier, cfg_overrides={"train_window_bars": 2})

    # Manually shrink the deque so log_returns has < 10 elements.
    strat._closes.clear()
    strat._closes.extend([100.0])
    snap, viterbi = strat._classify_now()
    assert snap is None
    assert viterbi is None


def test_check_existing_position_warns_when_position_present():
    """If the exchange reports an open position on startup, the strategy logs a warning."""
    classifier = StubClassifier([make_snap(0.10, 0.80, 0.10)])
    strat, om = _make_strategy([100.0] * 15, classifier)
    om.position = {"szi": "0.5", "coin": "BTC"}
    # Should not raise; just emits a warning.
    strat._check_existing_position()
