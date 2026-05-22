"""Startup reconciliation against the exchange's view of the world.

The strategy already cross-checks its persisted *position* against the
exchange via ``Strategy._reconcile_with_exchange``. This module covers the
adjacent risk: an *open order* that the bot doesn't know about.

A submit-then-disconnect can leave a resting order on the book after a
crash. Restarting blindly would let the strategy place new orders on top
of it. Walking ``info.open_orders`` on boot and flagging any orphans gives
the operator a chance to investigate before the next tick fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class OpenOrderSnapshot:
    coin: str
    side: str            # "B" or "A"
    oid: int
    size: float
    limit_price: float | None


def fetch_open_orders(info: Any, address: str, coin: str) -> list[OpenOrderSnapshot]:
    """Return open orders for ``coin``. Never raises — a transient API error
    yields an empty list and a logged warning, since the alternative would
    be to crash the boot path."""
    try:
        raw = info.open_orders(address)
    except Exception as exc:
        logger.warning("open_orders fetch failed: {}", exc)
        return []
    out: list[OpenOrderSnapshot] = []
    for o in raw or []:
        if o.get("coin") != coin:
            continue
        try:
            size = float(o.get("sz", o.get("origSz", 0)) or 0)
            limit = o.get("limitPx") or o.get("px")
            out.append(OpenOrderSnapshot(
                coin=coin, side=str(o.get("side", "")),
                oid=int(o.get("oid", 0)),
                size=size,
                limit_price=float(limit) if limit not in (None, "") else None,
            ))
        except (TypeError, ValueError) as exc:
            logger.warning("malformed open order skipped: {} (raw={})", exc, o)
    return out


def reconcile_open_orders(
    info: Any, address: str, coin: str, expected_oids: set[int] | None = None,
) -> tuple[bool, list[OpenOrderSnapshot]]:
    """Return (clean, orphans) where ``clean`` is True iff every open order
    on ``coin`` is in ``expected_oids``. The strategy uses the return value
    to set ``_block_entries``.

    ``expected_oids=None`` is conservative: any open order is treated as
    an orphan. Callers that track their last-known resting order ID can
    pass it in so harmless resting limits don't trip the gate.
    """
    open_orders = fetch_open_orders(info, address, coin)
    expected = expected_oids or set()
    orphans = [o for o in open_orders if o.oid not in expected]
    clean = not orphans
    if not clean:
        logger.error(
            "{} | reconcile: {} orphan open order(s) on exchange — "
            "new entries blocked until manually cancelled",
            coin, len(orphans),
        )
        for o in orphans:
            logger.error("  oid={} side={} size={} px={}", o.oid, o.side, o.size, o.limit_price)
    return clean, orphans
