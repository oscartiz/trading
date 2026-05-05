"""Entry point: wires feed → strategy → orders."""
import asyncio
import sys

from loguru import logger

from config import settings
from data import MarketFeed
from execution import OrderManager, build_clients
from risk import RiskConfig, RiskManager
from strategies.example_momentum import MomentumStrategy

COINS = ["BTC"]


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.add("logs/trading.log", rotation="100 MB", retention="30 days", level="DEBUG")


async def main() -> None:
    setup_logging()
    logger.info("Starting | testnet={}", settings.testnet)

    info, exchange = build_clients()
    order_manager = OrderManager(info, exchange)
    risk = RiskManager(RiskConfig())

    strategies = [
        MomentumStrategy(coin="BTC", order_manager=order_manager, risk=risk),
    ]

    feed = MarketFeed(COINS)
    for strategy in strategies:
        feed.on_trade(strategy.coin, strategy.on_trade)
        feed.on_book(strategy.coin, strategy.on_book)

    await feed.run()


if __name__ == "__main__":
    asyncio.run(main())
