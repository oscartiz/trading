"""Background tasks that run alongside strategies in the main event loop."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from loguru import logger

if TYPE_CHECKING:
    from risk import RiskManager


async def equity_watchdog(
    get_equity: Callable[[], float],
    risk: "RiskManager",
    poll_seconds: float = 60.0,
) -> None:
    """Poll account equity and feed it to RiskManager.

    The watchdog never raises — a failed poll logs a warning and retries on the
    next tick, so a transient API hiccup doesn't take down the strategy loop.
    """
    logger.info("equity_watchdog started | poll_seconds={}", poll_seconds)
    while True:
        try:
            equity = get_equity()
            if equity > 0:
                risk.update_equity(equity)
        except Exception as exc:
            logger.warning("equity_watchdog poll failed: {}", exc)
        await asyncio.sleep(poll_seconds)


def make_live_equity_source(info, address: str) -> Callable[[], float]:
    """Build a get_equity() callable that reads Hyperliquid account value."""
    def get_equity() -> float:
        state = info.user_state(address)
        ms = state.get("marginSummary") or {}
        try:
            return float(ms.get("accountValue") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return get_equity
