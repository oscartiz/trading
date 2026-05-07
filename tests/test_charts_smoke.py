"""Smoke tests — chart functions must not raise on minimal valid input.

We don't pixel-compare; we just verify that the chart code paths execute and
produce a non-empty PNG. matplotlib is configured to use the Agg backend so
tests can run headless.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtesting.charts import plot_results
from backtesting.engine import BacktestResult, Trade
from backtesting.regime_charts import plot_regime_results
from backtesting.regime_engine import RegimeBacktestResult, RegimeTrade
from strategies.configs import FundingConfig, RegimeSwitchingConfig

from tests.conftest import build_prices


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


def _funding_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime([_t(0), _t(8), _t(16)], utc=True),
        "funding_rate": [0.0005, 0.00005, -0.0001],
    })


def _funding_result() -> BacktestResult:
    trades = [
        Trade(entry_time=_t(0), exit_time=_t(8), side="short",
              entry_price=100.0, exit_price=99.0, position_size_usd=100.0,
              funding_collected=0.4, exit_reason="funding_normalised"),
    ]
    equity = pd.Series(
        [0.0, 0.5, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4],
        index=pd.DatetimeIndex([_t(i) for i in range(8)]),
        name="pnl",
    )
    return BacktestResult(
        trades=trades, equity_curve=equity,
        config=FundingConfig(),
        coin="BTC", start=_t(0), end=_t(8),
    )


def _regime_result() -> RegimeBacktestResult:
    trades = [
        RegimeTrade(entry_time=_t(2), exit_time=_t(20), side="long",
                    entry_price=100.0, exit_price=110.0, position_size_usd=100.0,
                    entry_regime="bull", entry_proba=0.85,
                    exit_reason="regime_weakened", fees=0.07),
    ]
    n = 30
    equity = pd.Series(
        [0.0] * 5 + [float(i) for i in range(n - 5)],
        index=pd.DatetimeIndex([_t(i) for i in range(n)]),
        name="pnl",
    )
    regimes = pd.DataFrame({
        "regime": ["bull"] * 10 + ["chop"] * 5 + ["bear"] * 15,
        "p_bear": [0.05] * 10 + [0.10] * 5 + [0.85] * 15,
        "p_chop": [0.10] * 10 + [0.80] * 5 + [0.10] * 15,
        "p_bull": [0.85] * 10 + [0.10] * 5 + [0.05] * 15,
        "expected_return": [0.001] * 30,
    }, index=pd.DatetimeIndex([_t(i) for i in range(n)], name="timestamp"))
    return RegimeBacktestResult(
        trades=trades, equity_curve=equity, regimes=regimes,
        config=RegimeSwitchingConfig(),
        coin="BTC", start=_t(0), end=_t(n - 1), n_refits=2,
    )


def test_plot_results_funding_smoke(tmp_path: Path):
    result = _funding_result()
    prices = build_prices([100.0, 99.5, 99.0, 99.0, 99.0, 99.5, 99.0, 98.5],
                          start=_t(0))
    save_path = tmp_path / "funding.png"
    out = plot_results(result, prices, _funding_df(), save_path=save_path, show=False)
    assert out == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 1000   # non-empty PNG


def test_plot_regime_results_smoke(tmp_path: Path):
    result = _regime_result()
    n = 30
    prices = build_prices([100.0 + i for i in range(n)], start=_t(0))
    save_path = tmp_path / "regime.png"
    out = plot_regime_results(result, prices, save_path=save_path, show=False)
    assert out == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 1000


def test_plot_regime_results_handles_empty_trades(tmp_path: Path):
    """An empty trade list shouldn't crash the chart — the regime/equity panels still render."""
    result = _regime_result()
    result.trades.clear()
    n = 30
    prices = build_prices([100.0] * n, start=_t(0))
    save_path = tmp_path / "regime_empty.png"
    out = plot_regime_results(result, prices, save_path=save_path, show=False)
    assert save_path.exists() and save_path.stat().st_size > 1000
