"""Heartbeat / liveness loop.

GETs a watchdog URL on a fixed interval — works with healthchecks.io,
uptime-kuma's push monitor, or any plain HTTP endpoint that pages you when
pings stop arriving. With a strategy that trades ~once per month, the
process can die silently for days; this is the cheapest insurance against that.

Silently no-ops when ``HEARTBEAT_URL`` is unset, so backtests and local
test runs don't reach for the network.
"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
from loguru import logger


async def heartbeat_once(url: str, session: aiohttp.ClientSession) -> bool:
    """Issue a single heartbeat GET. Returns True on 2xx, False on any failure.

    Never raises — a transient watchdog outage shouldn't take down the strategy.
    """
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return True
    except Exception as exc:
        logger.warning("heartbeat failed: {}", exc)
        return False


async def heartbeat_loop(
    url: str | None,
    interval_seconds: float = 300.0,
    session: Any = None,
) -> None:
    """Run a heartbeat ping every ``interval_seconds``.

    If ``url`` is empty/None the loop returns immediately, so callers can
    unconditionally include this in their ``asyncio.gather`` set.
    """
    if not url:
        return
    logger.info("heartbeat started | url={} interval={}s", url, interval_seconds)
    owned_session = session is None
    if owned_session:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
    try:
        while True:
            await heartbeat_once(url, session)
            await asyncio.sleep(interval_seconds)
    finally:
        if owned_session:
            await session.close()
