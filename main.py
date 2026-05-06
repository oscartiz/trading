"""Entry point: wires strategy → orders → risk."""
import argparse
import asyncio
import sys

from loguru import logger

from config import settings
from execution import OrderManager, build_clients
from risk import RiskConfig, RiskManager
from strategies.configs import FundingConfig, RegimeSwitchingConfig
from strategies.funding_rate import FundingRateStrategy
from strategies.regime_switching import RegimeSwitchingStrategy


def setup_logging() -> None:
    import os
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)
    logger.add("logs/trading.log", rotation="100 MB", retention="30 days", level="DEBUG")


def build_strategies(name: str, coin: str, order_manager: OrderManager, risk: RiskManager):
    if name == "funding":
        return [FundingRateStrategy(
            coin=coin, order_manager=order_manager, risk=risk,
            config=FundingConfig(
                entry_threshold=0.0002,
                exit_threshold=0.00005,
                stop_loss_pct=0.02,
                max_hold_hours=48,
                position_size_usd=50.0,
                poll_interval_seconds=600,
            ),
        )]
    if name == "regime":
        return [RegimeSwitchingStrategy(
            coin=coin, order_manager=order_manager, risk=risk,
            config=RegimeSwitchingConfig(),
        )]
    raise ValueError(f"Unknown strategy '{name}'. Available: funding, regime")


async def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run a trading strategy live.")
    parser.add_argument("--strategy", default="funding", choices=["funding", "regime"])
    parser.add_argument("--coin", default="BTC")
    args = parser.parse_args()

    logger.info("Starting | strategy={} coin={} testnet={}", args.strategy, args.coin, settings.testnet)

    info, exchange = build_clients()
    order_manager = OrderManager(info, exchange)
    risk = RiskManager(RiskConfig(
        max_position_usd=150.0,
        max_order_usd=100.0,
        max_drawdown_pct=0.05,
    ))

    strategies = build_strategies(args.strategy, args.coin, order_manager, risk)
    await asyncio.gather(*[s.run() for s in strategies])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
