"""Walk-forward sweep: run the regime strategy across a grid of (coin × year).

Emits a per-cell metrics table plus a per-coin aggregation and an overall
totals row. The point is to validate that the strategy generalises across
assets and market regimes — a Sharpe figure on a single year of BTC isn't
worth much without 100+ trades across diverse conditions.

Usage::

    python tools/walk_forward.py
    python tools/walk_forward.py --coins BTC,ETH,SOL --years 2022,2023,2024
    python tools/walk_forward.py --refit-every 336 --start-buffer-days 130

A cell that fails (no data, fit failure) is recorded and skipped — one bad
period doesn't take the whole grid down. Failures are summarised at the end.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running directly: `python tools/walk_forward.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.data import fetch_price_history     # noqa: E402
from backtesting.regime_engine import regime_metrics, run_regime_backtest    # noqa: E402
from strategies.configs import RegimeSwitchingConfig    # noqa: E402


@dataclass
class CellResult:
    coin: str
    year: int
    metrics: dict | None = None          # populated on success
    error: str | None = None             # populated on failure

    @property
    def ok(self) -> bool:
        return self.metrics is not None


def run_cell(
    coin: str,
    year: int,
    cfg: RegimeSwitchingConfig,
    start_buffer_days: int,
) -> CellResult:
    """Run one (coin, year) backtest. Returns a CellResult — never raises."""
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    fetch_start = year_start - timedelta(days=start_buffer_days)
    try:
        prices_df = fetch_price_history(coin, fetch_start, year_end, interval=cfg.candle_interval)
        prices_df = prices_df[prices_df["timestamp"] < year_end].reset_index(drop=True)
        if len(prices_df) <= cfg.train_window_bars + 1:
            return CellResult(coin, year, error=f"insufficient data ({len(prices_df)} bars)")
        result = run_regime_backtest(prices_df, coin, cfg)
        return CellResult(coin, year, metrics=regime_metrics(result))
    except Exception as exc:
        return CellResult(coin, year, error=f"{type(exc).__name__}: {exc}")


def run_grid(
    coins: list[str],
    years: list[int],
    cfg: RegimeSwitchingConfig,
    start_buffer_days: int,
) -> list[CellResult]:
    cells: list[CellResult] = []
    for coin in coins:
        for year in years:
            print(f"  {coin} {year} ... ", end="", flush=True)
            cell = run_cell(coin, year, cfg, start_buffer_days)
            if cell.ok:
                m = cell.metrics or {}
                print(f"{m.get('n_trades', 0)} trades, "
                      f"P&L ${m.get('total_pnl_usd', 0.0):+.2f}, "
                      f"Sharpe {m.get('sharpe_ratio', 0.0):.2f}", flush=True)
            else:
                print(f"SKIP ({cell.error})", flush=True)
            cells.append(cell)
    return cells


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #


_HEADER = (
    f"{'coin':<6s} {'year':<6s} {'trades':>7s} {'L/S':>7s} "
    f"{'win%':>6s} {'hold(h)':>9s} {'P&L $':>10s} "
    f"{'maxDD%':>8s} {'Sharpe':>7s}  exit reasons"
)


def _row(cell: CellResult) -> str:
    m = cell.metrics or {}
    reasons = "  ".join(f"{k}={v}" for k, v in m.get("exit_reasons", {}).items())
    return (
        f"{cell.coin:<6s} {cell.year:<6d} "
        f"{m.get('n_trades', 0):>7d} "
        f"{m.get('longs', 0):>3d}/{m.get('shorts', 0):<3d} "
        f"{m.get('win_rate_pct', 0.0):>6.1f} "
        f"{m.get('avg_hold_hours', 0.0):>9.1f} "
        f"{m.get('total_pnl_usd', 0.0):>+10.2f} "
        f"{m.get('max_drawdown_pct', 0.0):>+8.2f} "
        f"{m.get('sharpe_ratio', 0.0):>7.2f}  {reasons}"
    )


def aggregate(cells: list[CellResult]) -> dict:
    """Sum of trades, P&L, longs, shorts, weighted win rate, etc."""
    ok = [c for c in cells if c.ok]
    if not ok:
        return {"n_trades": 0, "longs": 0, "shorts": 0, "win_rate_pct": 0.0,
                "total_pnl_usd": 0.0, "avg_hold_hours": 0.0}
    trades = sum(c.metrics["n_trades"] for c in ok)         # type: ignore[index]
    longs = sum(c.metrics["longs"] for c in ok)             # type: ignore[index]
    shorts = sum(c.metrics["shorts"] for c in ok)           # type: ignore[index]
    pnl = sum(c.metrics["total_pnl_usd"] for c in ok)       # type: ignore[index]
    # Win rate weighted by trade count.
    weighted_wins = sum(
        c.metrics["win_rate_pct"] * c.metrics["n_trades"]   # type: ignore[index]
        for c in ok
    )
    win_rate = (weighted_wins / trades) if trades else 0.0
    # Hold time weighted by trade count.
    weighted_hold = sum(
        c.metrics["avg_hold_hours"] * c.metrics["n_trades"]    # type: ignore[index]
        for c in ok
    )
    hold = (weighted_hold / trades) if trades else 0.0
    return {
        "n_trades": trades, "longs": longs, "shorts": shorts,
        "win_rate_pct": win_rate, "total_pnl_usd": pnl,
        "avg_hold_hours": hold,
    }


def print_report(cells: list[CellResult]) -> None:
    print("\n" + "=" * 110)
    print(_HEADER)
    print("-" * 110)

    coins_in_order: list[str] = []
    by_coin: dict[str, list[CellResult]] = {}
    for c in cells:
        if c.coin not in by_coin:
            coins_in_order.append(c.coin)
            by_coin[c.coin] = []
        by_coin[c.coin].append(c)

    for coin in coins_in_order:
        group = by_coin[coin]
        for cell in group:
            if cell.ok:
                print(_row(cell))
            else:
                print(f"{cell.coin:<6s} {cell.year:<6d}  (skipped: {cell.error})")
        agg = aggregate(group)
        print(
            f"{coin:<6s} {'TOT':<6s} "
            f"{agg['n_trades']:>7d} "
            f"{agg['longs']:>3d}/{agg['shorts']:<3d} "
            f"{agg['win_rate_pct']:>6.1f} "
            f"{agg['avg_hold_hours']:>9.1f} "
            f"{agg['total_pnl_usd']:>+10.2f}    -        -"
        )
        print("-" * 110)

    overall = aggregate(cells)
    print(
        f"{'ALL':<6s} {'TOT':<6s} "
        f"{overall['n_trades']:>7d} "
        f"{overall['longs']:>3d}/{overall['shorts']:<3d} "
        f"{overall['win_rate_pct']:>6.1f} "
        f"{overall['avg_hold_hours']:>9.1f} "
        f"{overall['total_pnl_usd']:>+10.2f}    -        -"
    )
    print("=" * 110)

    failures = [c for c in cells if not c.ok]
    if failures:
        print(f"\nSkipped cells ({len(failures)}):")
        for c in failures:
            print(f"  {c.coin} {c.year}: {c.error}")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "Walk-forward sweep",
    )
    parser.add_argument("--coins", default="BTC,ETH,SOL",
                        help="Comma-separated coins (default: BTC,ETH,SOL)")
    parser.add_argument("--years", default="2022,2023,2024",
                        help="Comma-separated years to backtest (default: 2022,2023,2024)")
    parser.add_argument("--refit-every", type=int, default=336,
                        help="Refit HMM every N bars (336=biweekly@1h)")
    parser.add_argument("--start-buffer-days", type=int, default=130,
                        help="Days of history before each year-start to seed the HMM "
                             "(default: 130 — slightly more than 3000 bars / 24h).")
    args = parser.parse_args()

    coins = parse_csv_list(args.coins)
    years = [int(y) for y in parse_csv_list(args.years)]

    cfg = RegimeSwitchingConfig(refit_every_bars=args.refit_every)

    print(f"Walk-forward: {len(coins)} coins × {len(years)} years "
          f"= {len(coins) * len(years)} cells")
    cells = run_grid(coins, years, cfg, args.start_buffer_days)
    print_report(cells)


if __name__ == "__main__":
    main()
