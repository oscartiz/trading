"""Entry point: wires strategy → orders → risk."""
import argparse
import asyncio
import os
import sys

from loguru import logger

from config import settings
from execution import OrderManager, PaperOrderManager, build_clients
from risk import RiskConfig, RiskManager
from runtime import (
    Notifier,
    equity_watchdog,
    heartbeat_loop,
    make_live_equity_source,
    make_notifier_from_env,
    serve_metrics,
)
from strategies.configs import RegimeSwitchingConfig
from strategies.regime_switching import RegimeSwitchingStrategy


def setup_logging(notifier: Notifier) -> None:
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)
    logger.add("logs/trading.log", rotation="100 MB", retention="30 days", level="DEBUG")
    if notifier.enabled:
        logger.add(notifier.sink, level=notifier.min_level)


def build_strategy(
    coin: str, order_manager, risk: RiskManager, notifier: Notifier,
    equity_source, account_address: str | None,
) -> RegimeSwitchingStrategy:
    return RegimeSwitchingStrategy(
        coin=coin, order_manager=order_manager, risk=risk,
        config=RegimeSwitchingConfig(), notifier=notifier,
        equity_source=equity_source, account_address=account_address,
    )


async def main() -> None:
    notifier = make_notifier_from_env()
    setup_logging(notifier)

    parser = argparse.ArgumentParser(description="Run the regime-switching strategy live.")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--paper", action="store_true",
                        help="Paper-trade mode: real market data, simulated fills, no orders sent")
    parser.add_argument("--paper-equity", type=float, default=1_000.0,
                        help="Starting equity for paper mode")
    args = parser.parse_args()

    logger.info(
        "Starting | coin={} testnet={} paper={} alerts={}",
        args.coin, settings.testnet, args.paper, notifier.enabled,
    )

    info, exchange = build_clients(read_only=args.paper)
    # Equity-relative caps default to 15% of equity for max position and 10%
    # for any single order — wide enough that the BTC defaults (size=$100,
    # max_position=$150) keep working at small equities, and they scale up
    # cleanly as the account grows. Absolute caps stay as the floor.
    risk = RiskManager(RiskConfig(
        max_position_usd=150.0,
        max_order_usd=100.0,
        max_position_pct=0.15,
        max_order_pct=0.10,
        max_drawdown_pct=0.05,
    ))

    if args.paper:
        order_manager = PaperOrderManager(info, starting_equity=args.paper_equity)
        get_equity = order_manager.get_equity
        account_address = None
    else:
        assert exchange is not None
        order_manager = OrderManager(info, exchange)
        get_equity = make_live_equity_source(info, settings.account_address)
        account_address = settings.account_address

    strategy = build_strategy(
        args.coin, order_manager, risk, notifier, get_equity, account_address,
    )

    heartbeat_url = os.getenv("HEARTBEAT_URL") or None
    heartbeat_interval = float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300"))
    metrics_port = int(os.getenv("METRICS_PORT", "0") or 0) or None
    metrics_host = os.getenv("METRICS_HOST", "127.0.0.1")

    try:
        await asyncio.gather(
            strategy.run(),
            equity_watchdog(get_equity, risk, poll_seconds=60.0),
            heartbeat_loop(heartbeat_url, interval_seconds=heartbeat_interval),
            serve_metrics(metrics_port, host=metrics_host),
        )
    finally:
        notifier.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
