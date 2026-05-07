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


SWEEPS: list[Sweep] = [
    # Baseline — current defaults (TP at 6%, tight 3% stop, short max hold).
    Sweep("baseline-tp6-stop3",      "viterbi",  0.65, 0.03, 0.06, 24,  48,  240),

    # No TP, widen stop, longer max hold — Viterbi-driven.
    Sweep("viterbi-stop05-noTP",     "viterbi",  0.65, 0.05, None, 24,  168, 720),
    Sweep("viterbi-stop08-noTP",     "viterbi",  0.65, 0.08, None, 24,  168, 720),
    Sweep("viterbi-stop12-noTP",     "viterbi",  0.65, 0.12, None, 48,  168, 720),

    # Smoothed posterior with high-confidence entry threshold.
    Sweep("smoothed-p70-stop05",     "smoothed", 0.70, 0.05, None, 24,  48,  720),
    Sweep("smoothed-p70-stop08",     "smoothed", 0.70, 0.08, None, 24,  168, 720),
    Sweep("smoothed-p80-stop08",     "smoothed", 0.80, 0.08, None, 24,  168, 720),
    Sweep("smoothed-p80-stop12",     "smoothed", 0.80, 0.12, None, 48,  168, 720),
    Sweep("smoothed-p85-stop12",     "smoothed", 0.85, 0.12, None, 72,  168, 720),
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
