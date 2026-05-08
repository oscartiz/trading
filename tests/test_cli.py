"""Tests for CLI entrypoint — regime_backtest.py.

Network calls are mocked so tests run offline; we drive main() through argparse
and the engine layer to confirm the wiring is correct.
"""
from __future__ import annotations

import csv
import importlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import regime_backtest as rb_cli
from backtesting.regime_engine import RegimeBacktestResult, RegimeTrade
from strategies.configs import RegimeSwitchingConfig


def _t(h: int) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h)


# --------------------------------------------------------------------------- #
#  parse_date                                                                  #
# --------------------------------------------------------------------------- #


def test_regime_parse_date():
    d = rb_cli.parse_date("2024-03-15")
    assert d == datetime(2024, 3, 15, tzinfo=timezone.utc)


def test_regime_parse_date_invalid_format_raises():
    with pytest.raises(ValueError):
        rb_cli.parse_date("03/15/2024")


# --------------------------------------------------------------------------- #
#  save_trades helpers                                                         #
# --------------------------------------------------------------------------- #


def _regime_result_with_trade() -> RegimeBacktestResult:
    trade = RegimeTrade(
        entry_time=_t(2), exit_time=_t(10), side="long",
        entry_price=100.0, exit_price=110.0, position_size_usd=100.0,
        entry_regime="bull", entry_proba=0.85,
        exit_reason="regime_weakened", fees=0.07,
    )
    eq = pd.Series([0.0, 9.93], index=pd.DatetimeIndex([_t(2), _t(10)]))
    return RegimeBacktestResult(
        trades=[trade], equity_curve=eq, regimes=pd.DataFrame(),
        config=RegimeSwitchingConfig(),
        coin="BTC", start=_t(0), end=_t(20), n_refits=1,
    )


def test_regime_save_trades_writes_csv(tmp_path):
    out = tmp_path / "regime_trades.csv"
    rb_cli.save_trades(_regime_result_with_trade(), out)
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["side"] == "long"
    assert rows[0]["entry_regime"] == "bull"
    assert rows[0]["exit_reason"] == "regime_weakened"


def test_regime_save_trades_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "subdir" / "trades.csv"
    rb_cli.save_trades(_regime_result_with_trade(), out)
    assert out.exists()


# --------------------------------------------------------------------------- #
#  setup_logging — must be idempotent and never raise                          #
# --------------------------------------------------------------------------- #


def test_regime_setup_logging_idempotent():
    rb_cli.setup_logging()
    rb_cli.setup_logging()   # second call must not crash


# --------------------------------------------------------------------------- #
#  Regime CLI main() — end-to-end with mocked network                          #
# --------------------------------------------------------------------------- #


def _make_synthetic_prices(n: int = 200) -> pd.DataFrame:
    """Enough bars for the engine's default 3000-bar train window? No — but we
    override train_window via env (see _patch_run). For CLI tests we instead
    patch run_regime_backtest directly so we don't need real bars."""
    timestamps = pd.to_datetime(
        [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)],
        utc=True,
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1.0] * n,
    })


def test_regime_cli_main_runs_with_defaults(monkeypatch, capsys):
    """main() must wire fetch → run_regime_backtest → print_regime_metrics without error."""
    fake_prices = _make_synthetic_prices()
    fake_result = _regime_result_with_trade()

    monkeypatch.setattr(rb_cli, "fetch_price_history", lambda *a, **kw: fake_prices)
    monkeypatch.setattr(rb_cli, "run_regime_backtest", lambda *a, **kw: fake_result)
    monkeypatch.setattr("sys.argv", ["regime_backtest.py", "--coin", "BTC"])

    rb_cli.main()
    out = capsys.readouterr().out
    assert "Regime-Switching Backtest" in out
    assert "BTC" in out


def test_regime_cli_main_passes_through_args(monkeypatch):
    """CLI args must surface in the constructed RegimeSwitchingConfig."""
    captured: dict = {}

    def fake_run(prices_df, coin, cfg, fee_rate):
        captured["cfg"] = cfg
        captured["coin"] = coin
        captured["fee_rate"] = fee_rate
        return _regime_result_with_trade()

    monkeypatch.setattr(rb_cli, "fetch_price_history",
                         lambda *a, **kw: _make_synthetic_prices())
    monkeypatch.setattr(rb_cli, "run_regime_backtest", fake_run)
    monkeypatch.setattr("sys.argv", [
        "regime_backtest.py",
        "--coin", "ETH",
        "--entry-proba", "0.80",
        "--stop-loss", "0.07",
        "--take-profit", "0",                # disables TP
        "--min-hold-bars", "48",
        "--cooldown-bars", "200",
        "--signal-mode", "smoothed",
        "--fee-rate", "0.0005",
    ])
    rb_cli.main()

    cfg = captured["cfg"]
    assert captured["coin"] == "ETH"
    assert captured["fee_rate"] == 0.0005
    assert cfg.entry_proba == 0.80
    assert cfg.stop_loss_pct == 0.07
    assert cfg.take_profit_pct is None        # 0 → None per CLI logic
    assert cfg.min_hold_bars == 48
    assert cfg.same_regime_cooldown_bars == 200
    assert cfg.signal_mode == "smoothed"


def test_regime_cli_save_trades_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(rb_cli, "fetch_price_history",
                         lambda *a, **kw: _make_synthetic_prices())
    monkeypatch.setattr(rb_cli, "run_regime_backtest",
                         lambda *a, **kw: _regime_result_with_trade())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "regime_backtest.py", "--coin", "BTC",
        "--start", "2024-01-01", "--end", "2025-01-01",
        "--save-trades",
    ])
    rb_cli.main()
    expected = tmp_path / "data" / "trades_regime_BTC_2024-01-01_2025-01-01.csv"
    assert expected.exists()


def test_module_can_be_imported_directly():
    """The CLI script must be importable as a module without executing main()."""
    importlib.reload(rb_cli)
    assert hasattr(rb_cli, "main")
