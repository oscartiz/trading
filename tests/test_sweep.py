"""Tests for tools/regime_sweep.py — the parameter-sweep harness."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

# tools/ isn't on the package path by default; add it for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import regime_sweep  # noqa: E402
from backtesting.regime_engine import RegimeBacktestResult, RegimeTrade
from strategies.configs import RegimeSwitchingConfig


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


# --------------------------------------------------------------------------- #
#  SWEEPS list                                                                 #
# --------------------------------------------------------------------------- #


def test_sweeps_list_nonempty_and_unique_labels():
    labels = [s.label for s in regime_sweep.SWEEPS]
    assert len(labels) >= 5
    assert len(set(labels)) == len(labels), "sweep labels must be unique"


def test_sweep_signal_modes_are_valid():
    for s in regime_sweep.SWEEPS:
        assert s.signal_mode in ("smoothed", "viterbi")


def test_sweep_thresholds_are_valid_probabilities():
    for s in regime_sweep.SWEEPS:
        assert 0.0 < s.entry_proba < 1.0
        assert 0.0 < s.stop_loss_pct < 1.0


def test_sweep_can_construct_a_real_config():
    """Each sweep must produce a constructable RegimeSwitchingConfig."""
    s = regime_sweep.SWEEPS[0]
    cfg = RegimeSwitchingConfig(
        entry_proba=s.entry_proba,
        stop_loss_pct=s.stop_loss_pct,
        take_profit_pct=s.take_profit_pct,
        min_hold_bars=s.min_hold_bars,
        same_regime_cooldown_bars=s.same_regime_cooldown_bars,
        max_hold_bars=s.max_hold_bars,
        signal_mode=s.signal_mode,
    )
    assert cfg.entry_proba == s.entry_proba


# --------------------------------------------------------------------------- #
#  parse_date                                                                  #
# --------------------------------------------------------------------------- #


def test_sweep_parse_date():
    assert regime_sweep.parse_date("2024-01-01") == datetime(2024, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
#  run_sweep — uses mocks so no real HMM fits run                              #
# --------------------------------------------------------------------------- #


def _make_result(label: str, n_trades: int, pnl: float, sharpe: float) -> RegimeBacktestResult:
    trades = [
        RegimeTrade(
            entry_time=_t(0), exit_time=_t(8), side="long",
            entry_price=100.0, exit_price=100.0 + (pnl / max(n_trades, 1)),
            position_size_usd=100.0, entry_regime="bull", entry_proba=0.85,
            exit_reason="regime_weakened", fees=0.0,
        )
        for _ in range(n_trades)
    ]
    eq = pd.Series([0.0, pnl], index=pd.DatetimeIndex([_t(0), _t(8)]))
    return RegimeBacktestResult(
        trades=trades, equity_curve=eq, regimes=pd.DataFrame(),
        config=RegimeSwitchingConfig(),
        coin="BTC", start=_t(0), end=_t(24), n_refits=1,
    )


def test_run_sweep_invokes_each_config(monkeypatch):
    fake_prices = pd.DataFrame({
        "timestamp": pd.to_datetime([_t(0), _t(1)], utc=True),
        "open": [100.0, 100.0], "high": [101.0, 101.0], "low": [99.0, 99.0],
        "close": [100.0, 100.0], "volume": [1.0, 1.0],
    })
    monkeypatch.setattr(regime_sweep, "fetch_price_history", lambda *a, **kw: fake_prices)

    call_log: list[str] = []

    def fake_run(prices_df, coin, cfg, fee_rate=0.00035):
        # Just record the call and return a tiny result.
        call_log.append(cfg.signal_mode)
        return _make_result("any", n_trades=2, pnl=1.0, sharpe=0.5)

    monkeypatch.setattr(regime_sweep, "run_regime_backtest", fake_run)

    rows = regime_sweep.run_sweep("BTC", _t(0), _t(24), refit_every=10)
    assert len(rows) == len(regime_sweep.SWEEPS)
    assert all("trades" in r and "pnl_usd" in r and "sharpe" in r for r in rows)
    # Every sweep was driven through run_regime_backtest exactly once.
    assert len(call_log) == len(regime_sweep.SWEEPS)


def test_print_table_handles_empty_input(capsys):
    regime_sweep.print_table([])
    out = capsys.readouterr().out
    # Header lines should still appear; no row lines.
    assert "config" in out
    assert "exit reasons" in out


def test_print_table_sorts_by_pnl_descending(capsys):
    rows = [
        {"label": "low", "trades": 5, "win_pct": 40.0, "avg_hold_h": 10.0,
         "pnl_usd": -5.0, "max_dd_pct": -10.0, "sharpe": -0.5, "exits": {}},
        {"label": "high", "trades": 5, "win_pct": 60.0, "avg_hold_h": 10.0,
         "pnl_usd": 20.0, "max_dd_pct": -3.0, "sharpe": 1.5, "exits": {}},
        {"label": "mid", "trades": 5, "win_pct": 50.0, "avg_hold_h": 10.0,
         "pnl_usd": 5.0, "max_dd_pct": -7.0, "sharpe": 0.4, "exits": {}},
    ]
    regime_sweep.print_table(rows)
    out = capsys.readouterr().out
    # The "high" row should appear before "low" in the printed output (sorted desc by pnl).
    assert out.find("high") < out.find("mid") < out.find("low")
