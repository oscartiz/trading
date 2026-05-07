"""CLI entry point for the 3-state regime-switching backtest.

Examples:
    python regime_backtest.py
    python regime_backtest.py --coin ETH --start 2023-01-01 --end 2024-01-01
    python regime_backtest.py --entry-proba 0.7 --refit-every 336
    python regime_backtest.py --coin BTC --start 2023-01-01 --save-trades --chart
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from backtesting.data import fetch_price_history
from backtesting.regime_engine import print_regime_metrics, run_regime_backtest
from strategies.configs import RegimeSwitchingConfig


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> {message}",
    )


def save_trades(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entry_time", "exit_time", "side", "entry_regime", "entry_price",
            "exit_price", "hold_hours", "total_pnl", "exit_reason",
        ])
        writer.writeheader()
        for t in result.trades:
            writer.writerow({
                "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M"),
                "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M"),
                "side": t.side,
                "entry_regime": t.entry_regime,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "hold_hours": round(t.hold_hours, 1),
                "total_pnl": round(t.total_pnl, 4),
                "exit_reason": t.exit_reason,
            })
    logger.info("Trade log saved → {}", path)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Backtest 3-state regime-switching strategy")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--interval", default="1h", help="Candle interval (1h, 4h, 1d, ...)")

    parser.add_argument("--train-window", type=int, default=3000,
                        help="Bars of history used to fit the HMM")
    parser.add_argument("--refit-every", type=int, default=168,
                        help="Refit HMM every N bars (168=weekly@1h)")

    parser.add_argument("--entry-proba", type=float, default=0.65)
    parser.add_argument("--exit-proba", type=float, default=0.45)
    parser.add_argument("--max-chop-proba", type=float, default=0.50)
    parser.add_argument("--min-er", type=float, default=1e-4,
                        help="Min |E[r]| per bar to enter (log-return units)")

    parser.add_argument("--stop-loss", type=float, default=0.03)
    parser.add_argument("--take-profit", type=float, default=0.06,
                        help="Set to 0 to disable")
    parser.add_argument("--max-hold-bars", type=int, default=240)
    parser.add_argument("--min-hold-bars", type=int, default=24,
                        help="Floor before regime-based exits (stops still fire below this)")
    parser.add_argument("--cooldown-bars", type=int, default=48,
                        help="No re-entering the same regime for N bars after exit")
    parser.add_argument("--signal-mode", choices=["smoothed", "viterbi"], default="viterbi",
                        help="Drive entries from soft posterior or Viterbi MAP label")

    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--leverage", type=int, default=1)
    parser.add_argument("--fee-rate", type=float, default=0.00035)

    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--chart", action="store_true",
                        help="Save price+regime+equity chart to data/charts/")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    prices_df = fetch_price_history(args.coin, start, end, interval=args.interval, force=args.force_download)

    cfg = RegimeSwitchingConfig(
        candle_interval=args.interval,
        train_window_bars=args.train_window,
        refit_every_bars=args.refit_every,
        entry_proba=args.entry_proba,
        exit_proba=args.exit_proba,
        max_chop_proba=args.max_chop_proba,
        min_expected_return_per_bar=args.min_er,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=(args.take_profit if args.take_profit > 0 else None),
        max_hold_bars=args.max_hold_bars,
        min_hold_bars=args.min_hold_bars,
        same_regime_cooldown_bars=args.cooldown_bars,
        signal_mode=args.signal_mode,
        position_size_usd=args.size,
        leverage=args.leverage,
    )

    result = run_regime_backtest(prices_df, args.coin, cfg, fee_rate=args.fee_rate)
    print_regime_metrics(result)

    if args.save_trades and result.trades:
        save_trades(result, Path(f"data/trades_regime_{args.coin}_{args.start}_{args.end}.csv"))

    if args.chart:
        from backtesting.regime_charts import plot_regime_results
        chart_path = plot_regime_results(result, prices_df, show=False)
        logger.info("Chart saved → {}", chart_path)


if __name__ == "__main__":
    main()
