"""Tests for backtesting/metrics.py and backtesting/regime_engine.py:regime_metrics.

Both metric helpers reduce a list of trades + equity curve to summary dict.
The tests build minimal trade lists and assert exact aggregates.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backtesting.engine import BacktestResult, Trade
from backtesting.metrics import compute_metrics
from backtesting.regime_engine import (
    RegimeBacktestResult,
    RegimeTrade,
    regime_metrics,
)
from strategies.configs import FundingConfig, RegimeSwitchingConfig


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


def _equity_series(values: list[float], start_hour: int = 0) -> pd.Series:
    idx = pd.DatetimeIndex([_t(start_hour + i) for i in range(len(values))])
    return pd.Series(values, index=idx, name="pnl")


def _funding_trade(
    *,
    entry_h: int = 0,
    exit_h: int = 8,
    side: str = "short",
    entry: float = 100.0,
    exit_: float = 100.0,
    funding: float = 0.0,
    reason: str = "funding_normalised",
    notional: float = 100.0,
) -> Trade:
    return Trade(
        entry_time=_t(entry_h),
        exit_time=_t(exit_h),
        side=side,  # type: ignore[arg-type]
        entry_price=entry,
        exit_price=exit_,
        position_size_usd=notional,
        funding_collected=funding,
        exit_reason=reason,
    )


def _regime_trade(
    *,
    entry_h: int = 0,
    exit_h: int = 8,
    side: str = "long",
    entry: float = 100.0,
    exit_: float = 110.0,
    fees: float = 0.0,
    reason: str = "regime_weakened",
    regime: str = "bull",
    notional: float = 100.0,
) -> RegimeTrade:
    return RegimeTrade(
        entry_time=_t(entry_h),
        exit_time=_t(exit_h),
        side=side,  # type: ignore[arg-type]
        entry_price=entry,
        exit_price=exit_,
        position_size_usd=notional,
        entry_regime=regime,
        entry_proba=0.85,
        exit_reason=reason,
        fees=fees,
    )


def _funding_result(trades: list[Trade], equity: pd.Series) -> BacktestResult:
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        config=FundingConfig(),
        coin="BTC",
        start=_t(0),
        end=_t(len(equity) + 1) if len(equity) > 0 else _t(0),
    )


def _regime_result(trades: list[RegimeTrade], equity: pd.Series) -> RegimeBacktestResult:
    return RegimeBacktestResult(
        trades=trades,
        equity_curve=equity,
        regimes=pd.DataFrame(),
        config=RegimeSwitchingConfig(),
        coin="BTC",
        start=_t(0),
        end=_t(len(equity) + 1) if len(equity) > 0 else _t(0),
        n_refits=1,
    )


# --------------------------------------------------------------------------- #
#  compute_metrics (funding)                                                   #
# --------------------------------------------------------------------------- #


def test_funding_metrics_empty_trades_returns_error():
    result = _funding_result([], _equity_series([0.0, 0.0, 0.0]))
    m = compute_metrics(result)
    assert "error" in m


def test_funding_metrics_win_rate():
    trades = [
        _funding_trade(funding=1.0),     # win
        _funding_trade(funding=2.0),     # win
        _funding_trade(funding=-1.0),    # loss
        _funding_trade(funding=0.0),     # tie (counts as not-win)
    ]
    equity = _equity_series([0.0, 1.0, 3.0, 2.0, 2.0])
    m = compute_metrics(_funding_result(trades, equity))
    assert m["n_trades"] == 4
    assert m["win_rate_pct"] == 50.0


def test_funding_metrics_total_pnl_decomposition():
    trades = [
        _funding_trade(side="short", entry=100.0, exit_=95.0, funding=2.0),
        _funding_trade(side="long",  entry=100.0, exit_=110.0, funding=-1.0),
    ]
    equity = _equity_series([0.0, 5.0, 7.0, 16.0, 16.0])
    m = compute_metrics(_funding_result(trades, equity))
    # short: price_pnl = -1 * (95-100)/100 * 100 = +5; total = 5 + 2 = 7
    # long:  price_pnl = (110-100)/100 * 100 = +10; total = 10 - 1 = 9
    assert m["price_pnl_usd"] == pytest.approx(15.0)
    assert m["funding_collected_usd"] == pytest.approx(1.0)
    assert m["total_pnl_usd"] == pytest.approx(16.0)


def test_funding_metrics_drawdown_is_negative():
    """Max drawdown is reported as the worst (most negative) equity-from-peak excursion."""
    trades = [_funding_trade(funding=10.0)]
    # Climb to 10, drop to -5 → drawdown of 15.
    equity = _equity_series([0.0, 5.0, 10.0, 4.0, -5.0])
    m = compute_metrics(_funding_result(trades, equity))
    assert m["max_drawdown_usd"] == pytest.approx(-15.0)


def test_funding_metrics_exit_reasons_aggregated():
    trades = [
        _funding_trade(reason="funding_normalised"),
        _funding_trade(reason="funding_normalised"),
        _funding_trade(reason="stop_loss"),
        _funding_trade(reason="max_hold_time"),
    ]
    equity = _equity_series([0.0] * 5)
    m = compute_metrics(_funding_result(trades, equity))
    assert m["exit_reasons"] == {
        "funding_normalised": 2,
        "stop_loss": 1,
        "max_hold_time": 1,
    }


def test_funding_metrics_sharpe_handles_zero_variance():
    """Constant equity curve → std=0 → Sharpe must be 0, not NaN/inf."""
    trades = [_funding_trade(funding=1.0)]
    equity = _equity_series([1.0, 1.0, 1.0, 1.0])
    m = compute_metrics(_funding_result(trades, equity))
    assert m["sharpe_ratio"] == 0.0


# --------------------------------------------------------------------------- #
#  regime_metrics                                                              #
# --------------------------------------------------------------------------- #


def test_regime_metrics_empty_trades_returns_error():
    result = _regime_result([], _equity_series([0.0]))
    m = regime_metrics(result)
    assert "error" in m


def test_regime_metrics_long_short_split():
    trades = [
        _regime_trade(side="long"),
        _regime_trade(side="long"),
        _regime_trade(side="short"),
    ]
    equity = _equity_series([0.0] * 4)
    m = regime_metrics(_regime_result(trades, equity))
    assert m["longs"] == 2
    assert m["shorts"] == 1


def test_regime_metrics_drawdown_uses_position_size():
    """regime_metrics scales drawdown by position_size_usd, not equity peak."""
    trades = [_regime_trade(entry=100.0, exit_=110.0)]   # +10 USD
    equity = _equity_series([0.0, 10.0, -15.0])           # peak 10, trough -15 → DD -25
    cfg = RegimeSwitchingConfig(position_size_usd=100.0)
    result = RegimeBacktestResult(
        trades=trades, equity_curve=equity,
        regimes=pd.DataFrame(),
        config=cfg, coin="BTC", start=_t(0), end=_t(3), n_refits=1,
    )
    m = regime_metrics(result)
    assert m["max_drawdown_usd"] == pytest.approx(-25.0)
    # Pct = max_dd_usd / position_size_usd × 100 = -25%
    assert m["max_drawdown_pct"] == pytest.approx(-25.0)


def test_regime_metrics_n_refits_propagates():
    trades = [_regime_trade()]
    equity = _equity_series([0.0, 5.0, 10.0])
    result = RegimeBacktestResult(
        trades=trades, equity_curve=equity,
        regimes=pd.DataFrame(),
        config=RegimeSwitchingConfig(), coin="BTC", start=_t(0), end=_t(3), n_refits=42,
    )
    m = regime_metrics(result)
    assert m["n_refits"] == 42


def test_regime_metrics_total_pnl_includes_fees():
    trades = [
        _regime_trade(entry=100.0, exit_=110.0, fees=0.50),  # +10 - 0.5 = 9.5
        _regime_trade(entry=100.0, exit_=95.0, fees=0.50, side="short"),  # +5 - 0.5 = 4.5
    ]
    equity = _equity_series([0.0, 9.5, 14.0])
    m = regime_metrics(_regime_result(trades, equity))
    assert m["total_pnl_usd"] == pytest.approx(14.0)


def test_regime_metrics_avg_hold_hours():
    trades = [
        _regime_trade(entry_h=0, exit_h=10),    # 10h
        _regime_trade(entry_h=0, exit_h=20),    # 20h
    ]
    equity = _equity_series([0.0, 5.0, 10.0])
    m = regime_metrics(_regime_result(trades, equity))
    assert m["avg_hold_hours"] == pytest.approx(15.0)
