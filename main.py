"""Entry point: wires strategy → orders → risk."""
import argparse
import asyncio
import sys

from loguru import logger

from config import settings
from execution import OrderManager, PaperOrderManager, build_clients
from risk import RiskConfig, RiskManager
from runtime import equity_watchdog, make_live_equity_source
from strategies.configs import RegimeSwitchingConfig
from strategies.regime_switching import RegimeSwitchingStrategy


def setup_logging() -> None:
    import os
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)
    logger.add("logs/trading.log", rotation="100 MB", retention="30 days", level="DEBUG")


def build_strategy(coin: str, order_manager, risk: RiskManager) -> RegimeSwitchingStrategy:
    return RegimeSwitchingStrategy(
        coin=coin, order_manager=order_manager, risk=risk,
        config=RegimeSwitchingConfig(),
    )


async def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run the regime-switching strategy live.")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--paper", action="store_true",
                        help="Paper-trade mode: real market data, simulated fills, no orders sent")
    parser.add_argument("--paper-equity", type=float, default=1_000.0,
                        help="Starting equity for paper mode")
    args = parser.parse_args()

    logger.info(
        "Starting | coin={} testnet={} paper={}",
        args.coin, settings.testnet, args.paper,
    )

    info, exchange = build_clients(read_only=args.paper)
    risk = RiskManager(RiskConfig(
        max_position_usd=150.0,
        max_order_usd=100.0,
        max_drawdown_pct=0.05,
    ))

    if args.paper:
        order_manager = PaperOrderManager(info, starting_equity=args.paper_equity)
        get_equity = order_manager.get_equity
    else:
        assert exchange is not None
        order_manager = OrderManager(info, exchange)
        get_equity = make_live_equity_source(info, settings.account_address)

    strategy = build_strategy(args.coin, order_manager, risk)

    await asyncio.gather(
        strategy.run(),
        equity_watchdog(get_equity, risk, poll_seconds=60.0),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
