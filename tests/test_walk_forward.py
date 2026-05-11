"""Tests for the tools/walk_forward.py grid runner."""
from __future__ import annotations

import sys
from pathlib import Path

# tools/ isn't a package, so import it directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import walk_forward as wf    # noqa: E402

from strategies.configs import RegimeSwitchingConfig    # noqa: E402
from tests.conftest import regime_prices                # noqa: E402


# --------------------------------------------------------------------------- #
#  parse_csv_list                                                              #
# --------------------------------------------------------------------------- #


def test_parse_csv_list_basic():
    assert wf.parse_csv_list("BTC,ETH,SOL") == ["BTC", "ETH", "SOL"]


def test_parse_csv_list_strips_and_filters():
    assert wf.parse_csv_list(" BTC , ETH ,, SOL ") == ["BTC", "ETH", "SOL"]


# --------------------------------------------------------------------------- #
#  run_cell — patches fetch_price_history so tests don't hit the network      #
# --------------------------------------------------------------------------- #


def _patch_fetch(monkeypatch, df_or_exc):
    def _fake(*_a, **_kw):
        if isinstance(df_or_exc, Exception):
            raise df_or_exc
        return df_or_exc
    monkeypatch.setattr(wf, "fetch_price_history", _fake)


def test_run_cell_happy_path(monkeypatch):
    # Realistic synthetic series: long enough to clear train_window + a real loop.
    df = regime_prices(n_warmup=100, bull_bars=200, bear_bars=200, chop_bars=100, seed=1)
    _patch_fetch(monkeypatch, df)
    cfg = RegimeSwitchingConfig(train_window_bars=200, refit_every_bars=100)

    cell = wf.run_cell("BTC", 2024, cfg, start_buffer_days=10)

    assert cell.ok is True
    assert cell.error is None
    assert cell.metrics is not None
    # Backtest reports trades (or zero) — either way these keys exist.
    assert "n_trades" in cell.metrics
    assert "sharpe_ratio" in cell.metrics


def test_run_cell_records_error_on_short_data(monkeypatch):
    df = regime_prices(n_warmup=10, bull_bars=10, bear_bars=10, chop_bars=0, seed=1)
    _patch_fetch(monkeypatch, df)
    cfg = RegimeSwitchingConfig(train_window_bars=1000)   # data too short for this window

    cell = wf.run_cell("BTC", 2024, cfg, start_buffer_days=10)

    assert cell.ok is False
    assert cell.metrics is None
    assert "insufficient" in (cell.error or "")


def test_run_cell_records_error_on_fetch_failure(monkeypatch):
    _patch_fetch(monkeypatch, RuntimeError("binance 429"))
    cfg = RegimeSwitchingConfig()

    cell = wf.run_cell("BTC", 2024, cfg, start_buffer_days=10)

    assert cell.ok is False
    assert "binance 429" in (cell.error or "")


# --------------------------------------------------------------------------- #
#  aggregate                                                                   #
# --------------------------------------------------------------------------- #


def _cell(coin, year, **m) -> wf.CellResult:
    base = {
        "n_trades": 0, "longs": 0, "shorts": 0,
        "win_rate_pct": 0.0, "avg_hold_hours": 0.0,
        "total_pnl_usd": 0.0,
    }
    base.update(m)
    return wf.CellResult(coin, year, metrics=base)


def test_aggregate_sums_trades_and_pnl():
    cells = [
        _cell("BTC", 2022, n_trades=4, longs=2, shorts=2, total_pnl_usd=10.0,
              win_rate_pct=50.0, avg_hold_hours=20.0),
        _cell("BTC", 2023, n_trades=6, longs=5, shorts=1, total_pnl_usd=15.0,
              win_rate_pct=66.7, avg_hold_hours=30.0),
    ]
    agg = wf.aggregate(cells)
    assert agg["n_trades"] == 10
    assert agg["longs"] == 7
    assert agg["shorts"] == 3
    assert agg["total_pnl_usd"] == 25.0
    # Weighted win rate: (50*4 + 66.7*6) / 10 = (200 + 400.2) / 10 = 60.02
    assert agg["win_rate_pct"] == (50.0 * 4 + 66.7 * 6) / 10
    # Weighted hold: (20*4 + 30*6) / 10 = (80 + 180) / 10 = 26.0
    assert agg["avg_hold_hours"] == (20.0 * 4 + 30.0 * 6) / 10


def test_aggregate_ignores_failed_cells():
    cells = [
        _cell("BTC", 2022, n_trades=4, total_pnl_usd=10.0),
        wf.CellResult("BTC", 2023, error="boom"),
    ]
    agg = wf.aggregate(cells)
    assert agg["n_trades"] == 4
    assert agg["total_pnl_usd"] == 10.0


def test_aggregate_empty_returns_zero_row():
    agg = wf.aggregate([wf.CellResult("BTC", 2022, error="boom")])
    assert agg["n_trades"] == 0
    assert agg["total_pnl_usd"] == 0.0


# --------------------------------------------------------------------------- #
#  print_report — smoke test that all rows render without raising             #
# --------------------------------------------------------------------------- #


def test_print_report_handles_mixed_results(capsys):
    cells = [
        _cell("BTC", 2022, n_trades=3, longs=2, shorts=1, total_pnl_usd=5.0,
              win_rate_pct=66.7, avg_hold_hours=50.0,
              max_drawdown_pct=-2.5, sharpe_ratio=1.10,
              exit_reasons={"stop_loss": 1, "regime_weakened": 2}),
        wf.CellResult("BTC", 2023, error="data unavailable"),
        _cell("ETH", 2022, n_trades=2, longs=1, shorts=1, total_pnl_usd=-1.0,
              win_rate_pct=50.0, avg_hold_hours=40.0,
              max_drawdown_pct=-3.0, sharpe_ratio=-0.5,
              exit_reasons={"max_hold": 2}),
    ]
    wf.print_report(cells)
    out = capsys.readouterr().out
    assert "BTC" in out and "ETH" in out
    assert "data unavailable" in out
    assert "TOT" in out                        # per-coin and overall totals
    assert "Skipped cells" in out              # failure list at the end
