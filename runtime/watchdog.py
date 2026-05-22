"""Background tasks that run alongside strategies in the main event loop."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from loguru import logger

from .metrics import REGISTRY as METRICS

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

    ``get_equity`` is expected to be a sync callable that may block on a
    REST round-trip; the watchdog runs it via ``asyncio.to_thread`` so a
    slow exchange response never freezes the event loop (and with it the
    heartbeat — the very signal that would page the operator).
    """
    logger.info("equity_watchdog started | poll_seconds={}", poll_seconds)
    while True:
        try:
            equity = await asyncio.to_thread(get_equity)
            if equity > 0:
                risk.update_equity(equity)
                METRICS.set("equity_usd", float(equity))
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
