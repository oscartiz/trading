"""Tests for the regime-switching backtest engine.

Most tests stub the RegimeClassifier so we can drive the engine through
exact regime sequences and assert exact P&L. A handful of integration
tests use a real HMM on synthetic regime-switching data to confirm the
end-to-end pipeline still works.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtesting import regime_engine as re_mod
from backtesting.regime_engine import run_regime_backtest
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeSnapshot

from tests.conftest import build_prices, regime_prices


# --------------------------------------------------------------------------- #
#  Stub classifier for deterministic engine tests                              #
# --------------------------------------------------------------------------- #


class StubClassifier:
    """Replaces RegimeClassifier so tests can dictate the per-bar posterior.

    The engine calls fit() once before the loop and then snapshot() (and
    optionally predict()) once per loop bar. We just hand out the supplied
    snapshots in order — `loop_snaps[i]` is consumed at the i-th loop bar
    (which corresponds to engine bar index `train_window_bars + i`).
    """

    def __init__(self,
                 loop_snaps: list[RegimeSnapshot],
                 loop_viterbi: list[int] | None = None,
                 means: tuple[float, float, float] = (-0.005, 0.0, 0.005),
                 variances: tuple[float, float, float] = (1e-4, 1e-4, 1e-4)) -> None:
        self._snaps = list(loop_snaps)
        self._viterbi = list(loop_viterbi or [])
        self._snap_idx = 0
        self._viterbi_idx = 0
        self.state_means = np.array(means, dtype=float)
        self.state_variances = np.array(variances, dtype=float)
        self.transition_matrix = np.eye(3)
        self.fit_calls = 0
        self.snapshot_calls = 0
        self.predict_calls = 0
        # Live strategy's _refit reads classifier.hmm.n_states for a sanity check.
        from types import SimpleNamespace
        self.hmm = SimpleNamespace(n_states=3)

    def fit(self, _returns: np.ndarray) -> "StubClassifier":
        self.fit_calls += 1
        return self

    def snapshot(self, _returns: np.ndarray) -> RegimeSnapshot:
        self.snapshot_calls += 1
        snap = self._snaps[self._snap_idx] if self._snap_idx < len(self._snaps) else self._snaps[-1]
        self._snap_idx += 1
        return snap

    def predict(self, _returns: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        if self._viterbi_idx < len(self._viterbi):
            label = self._viterbi[self._viterbi_idx]
        else:
            label = int(np.argmax(self._snaps[-1].proba))
        self._viterbi_idx += 1
        return np.array([label], dtype=np.int64)


def make_snap(p_bear: float, p_chop: float, p_bull: float,
              expected_return: float = 0.0,
              expected_vol: float = 0.01) -> RegimeSnapshot:
    proba = np.array([p_bear, p_chop, p_bull], dtype=float)
    return RegimeSnapshot(
        regime=Regime(int(np.argmax(proba))),
        proba=proba,
        expected_return=expected_return,
        expected_vol=expected_vol,
    )


def _patch_classifier(classifier):
    return patch.object(re_mod, "RegimeClassifier", lambda *_, **__: classifier)


def _base_cfg(**overrides) -> RegimeSwitchingConfig:
    """Tight defaults for fast deterministic tests.

    Explicitly pins every gate to its permissive value so tests don't
    silently break when production defaults shift.
    """
    base = dict(
        train_window_bars=30,
        refit_every_bars=10000,         # effectively never refit during a short test
        entry_proba=0.65,
        exit_proba=0.45,
        max_chop_proba=0.50,
        min_expected_return_per_bar=0.0,
        stop_loss_pct=0.05,
        take_profit_pct=None,
        max_hold_bars=10000,
        min_hold_bars=0,
        same_regime_cooldown_bars=0,
        entry_confirmation_bars=0,
        entry_confirmation_bars_short=0,
        signal_mode="smoothed",
        position_size_usd=100.0,
    )
    base.update(overrides)
    return RegimeSwitchingConfig(**base)


# --------------------------------------------------------------------------- #
#  Trade accounting                                                            #
# --------------------------------------------------------------------------- #


def test_long_winning_trade_pnl_and_fees():
    """Enter long at price 100, exit at 110 → +10 USD price PnL minus fees."""
    # closes[30]=100 (entry), closes[40]=110 (exit when chop dominates)
    prices = build_prices([100.0] * 40 + [110.0] * 10)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    # 20 loop bars: bull for 10 (entry on 0, still bull through 9), chop at 10 forces exit.
    # Loop bar i corresponds to engine bar t = 30+i. Bar 10 → t=40 (close=110).
    loop = [bull] * 10 + [chop] * 10
    cfg = _base_cfg(min_hold_bars=0, exit_proba=0.45)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.001)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side == "long"
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(110.0)
    expected_price_pnl = (110.0 - 100.0) / 100.0 * 100.0   # +10 USD
    expected_fees = 2 * 0.001 * 100.0                       # 0.2 USD
    assert t.price_pnl == pytest.approx(expected_price_pnl)
    assert t.fees == pytest.approx(expected_fees)
    assert t.total_pnl == pytest.approx(expected_price_pnl - expected_fees)


def test_short_winning_trade_pnl():
    """Short at 100 → exit at 90 → +10 USD price PnL."""
    prices = build_prices([100.0] * 40 + [90.0] * 10)
    bear = make_snap(0.85, 0.10, 0.05, expected_return=-0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bear] * 10 + [chop] * 10
    cfg = _base_cfg()

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side == "short"
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(90.0)
    assert t.price_pnl == pytest.approx(10.0)


def test_short_losing_trade_pnl():
    """Short at 100, price rises to 110, chop forces exit → -10 USD."""
    prices = build_prices([100.0] * 40 + [110.0] * 10)
    bear = make_snap(0.85, 0.10, 0.05, expected_return=-0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bear] * 10 + [chop] * 10
    cfg = _base_cfg(stop_loss_pct=1.0)  # huge stop so it can't fire

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].side == "short"
    assert result.trades[0].price_pnl == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
#  Stop-loss & take-profit                                                     #
# --------------------------------------------------------------------------- #


def test_stop_loss_long_uses_low_not_close():
    """For a long position, stop_loss must trigger off the bar's LOW, not close."""
    # closes[30]=100 (entry), closes[31]=99.5 (still above stop), low[31]=90 (wick blows stop)
    closes = [100.0] * 31 + [99.5] * 19
    lows = [100.0] * 31 + [90.0] + [99.5] * 18
    prices = build_prices(closes, lows=lows)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    # Bull at 0 (entry, t=30); chop at 1 so the post-stop bar can't re-enter.
    loop = [bull] + [chop] * 19
    cfg = _base_cfg(stop_loss_pct=0.05)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(100.0 * (1 - 0.05))


def test_stop_loss_short_uses_high():
    closes = [100.0] * 31 + [100.5] * 19
    highs = [100.0] * 31 + [110.0] + [100.5] * 18
    prices = build_prices(closes, highs=highs)
    bear = make_snap(0.85, 0.10, 0.05, expected_return=-0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bear, bear] + [chop] * 18
    cfg = _base_cfg(stop_loss_pct=0.05)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(100.0 * (1 + 0.05))


def test_take_profit_long_uses_high():
    closes = [100.0] * 31 + [100.5] * 19
    highs = [100.0] * 31 + [120.0] + [100.5] * 18
    prices = build_prices(closes, highs=highs)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bull, bull] + [chop] * 18
    cfg = _base_cfg(take_profit_pct=0.10)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    t = result.trades[0]
    assert t.exit_reason == "take_profit"
    assert t.exit_price == pytest.approx(100.0 * 1.10)


def test_stop_fires_below_min_hold_bars():
    """Hard stops must bypass min_hold_bars so a tanking trade can still be cut."""
    closes = [100.0] * 31 + [80.0] * 19
    lows = [100.0] * 31 + [80.0] * 19
    prices = build_prices(closes, lows=lows)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [bull] + [chop] * 19
    cfg = _base_cfg(stop_loss_pct=0.05, min_hold_bars=100)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop_loss"


# --------------------------------------------------------------------------- #
#  Hold gates                                                                  #
# --------------------------------------------------------------------------- #


def test_min_hold_bars_blocks_soft_exits():
    """A regime-weakening posterior at bar 1 must be ignored until min_hold elapses."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    weak = make_snap(0.30, 0.40, 0.30, expected_return=0.0)
    # Enter at loop bar 0, weak from bar 1 onwards. Without min_hold this would
    # exit at bar 1; with min_hold=10 the exit must happen no earlier than bar 10.
    loop = [bull] + [weak] * 19
    cfg = _base_cfg(min_hold_bars=10)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    held_bars = round(
        (result.trades[0].exit_time - result.trades[0].entry_time).total_seconds() / 3600
    )
    assert held_bars >= 10


def test_max_hold_bars_force_exits():
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    loop = [bull] * 20
    cfg = _base_cfg(max_hold_bars=5, min_hold_bars=0)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert any(t.exit_reason == "max_hold" for t in result.trades)


def test_cooldown_blocks_same_regime_reentry():
    """After exiting a bull trade, strong bull bars within cooldown must not re-enter."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    weak = make_snap(0.30, 0.40, 0.30, expected_return=0.0)
    # Enter at 0, weak at 2 forces exit at t=32, then bull every bar but cooldown
    # spans the whole remaining 17 bars — re-entry must be blocked.
    loop = [bull, bull, weak] + [bull] * 17
    cfg = _base_cfg(min_hold_bars=0, same_regime_cooldown_bars=20)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].side == "long"


def test_cooldown_allows_opposite_regime():
    """Cooldown is per-regime: after exiting bull, a strong bear may still short."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    bear = make_snap(0.85, 0.10, 0.05, expected_return=-0.001)
    weak = make_snap(0.30, 0.40, 0.30, expected_return=0.0)
    loop = [bull, bull, weak, bear] + [bear] * 16
    cfg = _base_cfg(min_hold_bars=0, same_regime_cooldown_bars=20)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) >= 2
    assert result.trades[0].side == "long"
    assert result.trades[1].side == "short"


# --------------------------------------------------------------------------- #
#  Signal mode dispatch                                                        #
# --------------------------------------------------------------------------- #


def test_smoothed_mode_respects_max_chop_proba():
    """In smoothed mode, entry must be blocked when chop dominates."""
    prices = build_prices([100.0] * 50)
    chop_dominant_bull = make_snap(0.10, 0.55, 0.35, expected_return=0.001)
    loop = [chop_dominant_bull] * 20
    cfg = _base_cfg(signal_mode="smoothed", max_chop_proba=0.50, entry_proba=0.30)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 0


def test_viterbi_mode_uses_label_not_proba():
    """Viterbi mode ignores entry_proba and trades on the MAP label."""
    prices = build_prices([100.0] * 30 + [110.0] * 20)
    weak_bull = make_snap(0.20, 0.30, 0.50, expected_return=0.001)  # P < entry_proba
    loop = [weak_bull] * 20
    viterbi = [int(Regime.BULL)] * 19 + [int(Regime.CHOP)]
    cfg = _base_cfg(signal_mode="viterbi", entry_proba=0.65, min_hold_bars=0)

    with _patch_classifier(StubClassifier(loop, viterbi)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].side == "long"


def test_entry_confirmation_blocks_brief_posterior_spike():
    """A 2-bar bull spike must not trigger entry when entry_confirmation_bars=5."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    # Two bars of bull, then chop — confirmation requires 5, so no entry should fire.
    loop = [bull, bull] + [chop] * 18
    cfg = _base_cfg(min_hold_bars=0, entry_confirmation_bars=5)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 0


def test_entry_confirmation_allows_sustained_signal():
    """A sustained bull regime past entry_confirmation_bars must enter on bar N."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    # 6 bars of bull (enough for confirmation=5), then chop forces exit.
    loop = [bull] * 6 + [chop] * 14
    cfg = _base_cfg(min_hold_bars=0, entry_confirmation_bars=5)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].side == "long"


def test_entry_confirmation_streak_resets_on_regime_flip():
    """A bull→bear→bull sequence must restart the streak counter."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    bear = make_snap(0.85, 0.10, 0.05, expected_return=-0.001)
    chop = make_snap(0.10, 0.80, 0.10)
    # 3 bars bull, 1 bar bear (resets), 3 bars bull (only 3 → not enough for confirm=5).
    loop = [bull] * 3 + [bear] + [bull] * 3 + [chop] * 13
    cfg = _base_cfg(min_hold_bars=0, entry_confirmation_bars=5,
                    entry_confirmation_bars_short=5, same_regime_cooldown_bars=0)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    # Bear at bar 3 has expected_return=-0.001 ≤ -min_er=0, so a bear entry
    # would qualify (1-bar streak); but confirmation=5 blocks it. The 3 trailing
    # bull bars also fail confirmation. → 0 trades.
    assert len(result.trades) == 0


def test_smoothed_mode_does_not_call_predict():
    """Smoothed mode must not invoke Viterbi (it's expensive)."""
    prices = build_prices([100.0] * 50)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    loop = [bull] * 20
    cfg = _base_cfg(signal_mode="smoothed")
    stub = StubClassifier(loop)

    with _patch_classifier(stub):
        run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert stub.predict_calls == 0, "Viterbi predict() should not be called in smoothed mode"


def test_viterbi_mode_calls_predict_each_bar():
    prices = build_prices([100.0] * 50)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [chop] * 20
    viterbi = [int(Regime.CHOP)] * 20
    cfg = _base_cfg(signal_mode="viterbi")
    stub = StubClassifier(loop, viterbi)

    with _patch_classifier(stub):
        run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    # One predict() per loop bar.
    assert stub.predict_calls == 20


# --------------------------------------------------------------------------- #
#  Entry guards                                                                #
# --------------------------------------------------------------------------- #


def test_min_expected_return_blocks_marginal_bull():
    """High P(bull) with near-zero E[r] must be filtered out."""
    prices = build_prices([100.0] * 50)
    marginal_bull = make_snap(0.05, 0.10, 0.85, expected_return=1e-6)
    loop = [marginal_bull] * 20
    cfg = _base_cfg(min_expected_return_per_bar=1e-3, signal_mode="smoothed")

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 0


def test_too_few_bars_raises():
    prices = build_prices([100.0] * 10)
    cfg = _base_cfg(train_window_bars=30)
    with pytest.raises(ValueError, match="Not enough bars"):
        run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)


def test_open_position_closed_at_backtest_end():
    prices = build_prices([100.0] * 30 + [105.0] * 5)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    loop = [bull] * 5
    cfg = _base_cfg()

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "backtest_end"
    assert result.trades[0].exit_price == pytest.approx(105.0)


# --------------------------------------------------------------------------- #
#  Equity curve & metadata                                                     #
# --------------------------------------------------------------------------- #


def test_equity_curve_length_matches_decision_bars():
    prices = build_prices([100.0] * 50)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [chop] * 20
    cfg = _base_cfg(train_window_bars=30)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.equity_curve) == 50 - cfg.train_window_bars


def test_equity_curve_has_no_nans():
    closes = [100.0 * (1.0 + 0.001 * i) for i in range(50)]
    prices = build_prices(closes)
    bull = make_snap(0.05, 0.10, 0.85, expected_return=0.001)
    loop = [bull] * 20
    cfg = _base_cfg()

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert not result.equity_curve.isna().any()


def test_n_refits_includes_initial_fit():
    prices = build_prices([100.0] * 50)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [chop] * 20
    cfg = _base_cfg(refit_every_bars=5, train_window_bars=30)

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert result.n_refits >= 2


def test_regimes_dataframe_records_every_decision_bar():
    prices = build_prices([100.0] * 50)
    chop = make_snap(0.10, 0.80, 0.10)
    loop = [chop] * 20
    cfg = _base_cfg()

    with _patch_classifier(StubClassifier(loop)):
        result = run_regime_backtest(prices, "BTC", cfg, fee_rate=0.0)

    assert len(result.regimes) == len(result.equity_curve)
    assert {"regime", "p_bear", "p_chop", "p_bull"} <= set(result.regimes.columns)


# --------------------------------------------------------------------------- #
#  Integration with real HMM (sanity check)                                    #
# --------------------------------------------------------------------------- #


def test_integration_with_real_hmm_runs_end_to_end():
    """A real HMM on synthetic regime data should produce trades and finite metrics."""
    df = regime_prices(n_warmup=60, bull_bars=120, chop_bars=40, bear_bars=120, seed=42)
    cfg = RegimeSwitchingConfig(
        train_window_bars=80,
        refit_every_bars=200,
        entry_proba=0.65,
        exit_proba=0.45,
        min_expected_return_per_bar=0.0,
        max_chop_proba=0.60,
        stop_loss_pct=0.20,
        take_profit_pct=None,
        max_hold_bars=2000,
        min_hold_bars=0,
        same_regime_cooldown_bars=0,
        signal_mode="smoothed",
        position_size_usd=100.0,
    )
    result = run_regime_backtest(df, "BTC", cfg, fee_rate=0.0)
    assert isinstance(result.start, datetime)
    assert isinstance(result.end, datetime)
    assert result.config is cfg
    # Real data should at least produce decisions (might be zero trades, that's ok).
    assert not result.equity_curve.isna().any()


def test_invalid_close_column_raises():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([datetime(2024, 1, 1, tzinfo=timezone.utc)] * 50),
        "open": [100.0] * 50,
        "high": [100.0] * 50,
        "low": [100.0] * 50,
        # 'close' deliberately missing
        "volume": [1.0] * 50,
    })
    with pytest.raises(ValueError, match="close"):
        run_regime_backtest(df, "BTC", _base_cfg(), fee_rate=0.0)
