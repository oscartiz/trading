"""Entry point: wires strategy → orders → risk."""
import asyncio
import sys

from loguru import logger

from config import settings
from execution import OrderManager, build_clients
from risk import RiskConfig, RiskManager
from strategies.funding_rate import FundingConfig, FundingRateStrategy


def setup_logging() -> None:
    import os
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)
    logger.add("logs/trading.log", rotation="100 MB", retention="30 days", level="DEBUG")


async def main() -> None:
    setup_logging()
    logger.info("Starting | testnet={}", settings.testnet)

    info, exchange = build_clients()
    order_manager = OrderManager(info, exchange)
    risk = RiskManager(RiskConfig(
        max_position_usd=150.0,   # never hold more than $150 total
        max_order_usd=100.0,      # single order cap
        max_drawdown_pct=0.05,    # halt if equity drops 5%
    ))

    strategies = [
        FundingRateStrategy(
            coin="BTC",
            order_manager=order_manager,
            risk=risk,
            config=FundingConfig(
                entry_threshold=0.0002,       # 0.02%/hr (~175%/yr) — only enter meaningful rates
                exit_threshold=0.00005,       # 0.005%/hr — exit when rate is no longer worth it
                stop_loss_pct=0.02,           # 2% stop
                max_hold_hours=48,
                position_size_usd=50.0,       # $50 notional
                poll_interval_seconds=600,    # check every 10 min
            ),
        ),
    ]

    await asyncio.gather(*[s.run() for s in strategies])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
