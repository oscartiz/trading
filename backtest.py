"""CLI entry point for running funding rate backtests.

Usage:
    python backtest.py                            # BTC, 2024
    python backtest.py --coin ETH --start 2023-01-01 --end 2024-01-01
    python backtest.py --entry-threshold 0.0003 --save-trades
    python backtest.py --coin BTC --start 2024-01-01 --sweep-entry
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from backtesting.data import fetch_funding_history, fetch_price_history
from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics, print_metrics
from strategies.configs import FundingConfig


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True, format="<green>{time:HH:mm:ss}</green> {message}")


def save_trades(result, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entry_time", "exit_time", "side", "entry_price", "exit_price",
            "hold_hours", "funding_collected", "price_pnl", "total_pnl", "exit_reason",
        ])
        writer.writeheader()
        for t in result.trades:
            writer.writerow({
                "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M"),
                "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M"),
                "side": t.side,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "hold_hours": round(t.hold_hours, 1),
                "funding_collected": round(t.funding_collected, 6),
                "price_pnl": round(t.price_pnl, 6),
                "total_pnl": round(t.total_pnl, 6),
                "exit_reason": t.exit_reason,
            })
    logger.info("Trade log saved → {}", path)


def sweep_entry_thresholds(funding_df, prices_df, coin: str, args) -> None:
    """Run the backtest across a grid of entry thresholds and print a comparison table."""
    thresholds = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]
    print(f"\n{'Entry/hr':>12}  {'Trades':>6}  {'Win%':>6}  {'Total P&L':>10}  {'Sharpe':>7}  {'Max DD':>8}")
    print("-" * 60)
    for t in thresholds:
        cfg = FundingConfig(
            entry_threshold=t,
            exit_threshold=args.exit_threshold,
            stop_loss_pct=args.stop_loss,
            max_hold_hours=args.max_hold_hours,
            position_size_usd=args.size,
        )
        result = run_backtest(funding_df, prices_df, coin, cfg, fee_rate=args.fee_rate)
        m = compute_metrics(result)
        if "error" in m:
            print(f"{t:>12.4%}  {'—':>6}  {'—':>6}  {'—':>10}  {'—':>7}  {'—':>8}")
        else:
            print(
                f"{t:>12.4%}  {m['n_trades']:>6}  {m['win_rate_pct']:>5.1f}%"
                f"  {m['total_pnl_usd']:>+10.4f}  {m['sharpe_ratio']:>7.3f}"
                f"  {m['max_drawdown_pct']:>7.2f}%"
            )
    print()


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Backtest funding rate carry strategy")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--entry-threshold", type=float, default=0.0002,
                        help="Funding rate entry threshold per 8h period (default: 0.0002)")
    parser.add_argument("--exit-threshold", type=float, default=0.00005)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--max-hold-hours", type=int, default=48)
    parser.add_argument("--size", type=float, default=50.0, help="Position size in USD")
    parser.add_argument("--fee-rate", type=float, default=0.00035, help="Taker fee per side")
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--chart", action="store_true", help="Generate results chart (saved to data/charts/)")
    parser.add_argument("--sweep-entry", action="store_true",
                        help="Grid search over entry thresholds instead of single run")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    funding_df = fetch_funding_history(args.coin, start, end, force=args.force_download)
    prices_df = fetch_price_history(args.coin, start, end, force=args.force_download)

    if args.sweep_entry:
        sweep_entry_thresholds(funding_df, prices_df, args.coin, args)
        return

    cfg = FundingConfig(
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss_pct=args.stop_loss,
        max_hold_hours=args.max_hold_hours,
        position_size_usd=args.size,
    )
    result = run_backtest(funding_df, prices_df, args.coin, cfg, fee_rate=args.fee_rate)
    print_metrics(result)

    if args.save_trades and result.trades:
        save_trades(result, Path(f"data/trades_{args.coin}_{args.start}_{args.end}.csv"))

    if args.chart:
        from backtesting.charts import plot_results
        chart_path = plot_results(result, prices_df, funding_df, show=False)
        logger.info("Chart saved → {}", chart_path)


if __name__ == "__main__":
    main()
