"""Parameter sweep for the regime-switching strategy.

Runs a list of configs against a single (coin, period) and prints a comparison
table. Reuses the cached price history so each run only pays for HMM fits.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# Allow running directly: `python tools/regime_sweep.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.data import fetch_price_history
from backtesting.regime_engine import regime_metrics, run_regime_backtest
from strategies.configs import RegimeSwitchingConfig


class Sweep(NamedTuple):
    label: str
    signal_mode: str
    entry_proba: float
    stop_loss_pct: float
    take_profit_pct: float | None
    min_hold_bars: int
    same_regime_cooldown_bars: int
    max_hold_bars: int
    entry_confirmation_bars: int = 0


SWEEPS: list[Sweep] = [
    # ── Reference: previous winner ─────────────────────────────────────────
    Sweep("ref-p85-stop12",          "smoothed", 0.85, 0.12, None, 72,  168,  720,   0),

    # ── Low-frequency, ride-the-regime: ~1 trade/month or fewer ────────────
    # Confirmation bars target the 4-12 hour range so the posterior has time to
    # settle but we don't lock ourselves out of every regime entirely.
    Sweep("lf-p85-cd720-conf4",      "smoothed", 0.85, 0.12, None, 72,  720,  1440,  4),
    Sweep("lf-p85-cd720-conf6",      "smoothed", 0.85, 0.12, None, 72,  720,  1440,  6),
    Sweep("lf-p85-cd720-conf8",      "smoothed", 0.85, 0.12, None, 72,  720,  1440,  8),
    Sweep("lf-p85-cd720-conf10",     "smoothed", 0.85, 0.12, None, 72,  720,  1440, 10),
    Sweep("lf-p85-cd720-conf12",     "smoothed", 0.85, 0.12, None, 72,  720,  1440, 12),
    Sweep("lf-p85-cd336-conf6",      "smoothed", 0.85, 0.12, None, 72,  336,  1440,  6),
    Sweep("lf-p85-cd336-conf8",      "smoothed", 0.85, 0.12, None, 72,  336,  1440,  8),
    Sweep("lf-p90-cd336-conf4",      "smoothed", 0.90, 0.12, None, 72,  336,  1440,  4),
    Sweep("lf-p90-cd336-conf6",      "smoothed", 0.90, 0.12, None, 72,  336,  1440,  6),
]


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run_sweep(coin: str, start: datetime, end: datetime, refit_every: int) -> list[dict]:
    prices_df = fetch_price_history(coin, start, end, interval="1h")
    rows: list[dict] = []
    for s in SWEEPS:
        cfg = RegimeSwitchingConfig(
            refit_every_bars=refit_every,
            entry_proba=s.entry_proba,
            stop_loss_pct=s.stop_loss_pct,
            take_profit_pct=s.take_profit_pct,
            min_hold_bars=s.min_hold_bars,
            same_regime_cooldown_bars=s.same_regime_cooldown_bars,
            entry_confirmation_bars=s.entry_confirmation_bars,
            max_hold_bars=s.max_hold_bars,
            signal_mode=s.signal_mode,
        )
        result = run_regime_backtest(prices_df, coin, cfg)
        m = regime_metrics(result)
        rows.append({
            "label": s.label,
            "trades": m.get("n_trades", 0),
            "win_pct": m.get("win_rate_pct", 0.0),
            "avg_hold_h": m.get("avg_hold_hours", 0.0),
            "pnl_usd": m.get("total_pnl_usd", 0.0),
            "max_dd_pct": m.get("max_drawdown_pct", 0.0),
            "sharpe": m.get("sharpe_ratio", 0.0),
            "exits": m.get("exit_reasons", {}),
        })
        print(f"  {s.label:28s}  done — {m.get('n_trades', 0)} trades, "
              f"P&L ${m.get('total_pnl_usd', 0.0):+.2f}, Sharpe {m.get('sharpe_ratio', 0.0):.2f}",
              flush=True)
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(f"{'config':28s} {'trades':>7s} {'hold(h)':>9s} {'win%':>6s} "
          f"{'P&L $':>10s} {'maxDD%':>8s} {'Sharpe':>7s}  exit reasons")
    print("-" * 110)
    rows_sorted = sorted(rows, key=lambda r: r["pnl_usd"], reverse=True)
    for r in rows_sorted:
        reasons = "  ".join(f"{k}={v}" for k, v in r["exits"].items())
        print(f"{r['label']:28s} {r['trades']:>7d} {r['avg_hold_h']:>9.1f} "
              f"{r['win_pct']:>6.1f} {r['pnl_usd']:>+10.2f} {r['max_dd_pct']:>+8.2f} "
              f"{r['sharpe']:>7.2f}  {reasons}")
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2025-01-01")
    parser.add_argument("--refit-every", type=int, default=336)
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    print(f"Sweep on {args.coin} {args.start}→{args.end}, refit every {args.refit_every} bars")
    print(f"{len(SWEEPS)} configurations")
    rows = run_sweep(args.coin, start, end, args.refit_every)
    print_table(rows)


if __name__ == "__main__":
    main()
