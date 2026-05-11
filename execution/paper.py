"""Paper-trading order manager.

Mirrors the surface area of `OrderManager` so strategies can't tell whether
they're running against the real exchange or this simulator. Fills happen at
the current mid (with optional slippage) and are accounted for in a per-coin
position book plus a running realised P&L.

The book is intentionally minimal — enough to satisfy the strategies' calls
to `market_order` and `get_position`, plus a `get_equity` accessor that the
drawdown watchdog can read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from .order_manager import OrderResult, Side


@dataclass
class PaperFill:
    coin: str
    side: Side
    size: float
    fill_price: float
    fee: float


class PaperOrderManager:
    """Same interface as OrderManager, but no orders go to the exchange."""

    def __init__(
        self,
        info: Any,
        starting_equity: float = 1_000.0,
        fee_rate: float = 0.00035,
        slippage_bps: float = 1.0,
    ) -> None:
        self.info = info
        self.starting_equity = starting_equity
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.realised_pnl: float = 0.0
        self.fees_paid: float = 0.0
        self.positions: dict[str, dict] = {}    # coin -> {"coin", "szi", "entryPx"}
        self.fills: list[PaperFill] = []
        self._next_oid: int = 1

    # ------------------------------------------------------------------ #
    #  Exchange surface — must match OrderManager                         #
    # ------------------------------------------------------------------ #
    def market_order(self, coin: str, side: Side, size: float) -> OrderResult:
        mid = self._get_mid(coin)
        if mid is None or mid <= 0:
            logger.warning("paper | {} | market_order rejected: no mid", coin)
            return OrderResult(success=False, error="no mid")
        slip = self.slippage_bps / 10_000.0
        fill_price = mid * (1 + slip) if side == Side.BUY else mid * (1 - slip)
        fee = abs(size) * fill_price * self.fee_rate
        self._apply_fill(coin, side, size, fill_price, fee)
        oid = self._next_oid
        self._next_oid += 1
        logger.info(
            "paper | {} | FILLED {} size={} @ {:.4f} (mid={:.4f}, fee=${:.4f})",
            coin, side, size, fill_price, mid, fee,
        )
        return OrderResult(
            success=True, order_id=oid, raw={"paper": True},
            fill_price=fill_price, fee=fee,
        )

    def limit_order(
        self, coin: str, side: Side, size: float, price: float, reduce_only: bool = False,
    ) -> OrderResult:
        # Simplification: paper limits fill immediately at the requested price.
        fee = abs(size) * price * self.fee_rate
        self._apply_fill(coin, side, size, price, fee)
        oid = self._next_oid
        self._next_oid += 1
        logger.info(
            "paper | {} | LIMIT FILLED {} size={} @ {:.4f} (fee=${:.4f})",
            coin, side, size, price, fee,
        )
        return OrderResult(
            success=True, order_id=oid, raw={"paper": True},
            fill_price=price, fee=fee,
        )

    def cancel_order(self, coin: str, order_id: int) -> OrderResult:
        return OrderResult(success=True, order_id=order_id, raw={"paper": True})

    def cancel_all(self, coin: str) -> None:
        return None

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> None:
        logger.info("paper | leverage noop | coin={} leverage={}x cross={}", coin, leverage, is_cross)

    def get_position(self, coin: str) -> dict | None:
        pos = self.positions.get(coin)
        if pos is None or float(pos["szi"]) == 0:
            return None
        return pos

    # ------------------------------------------------------------------ #
    #  Equity for the drawdown watchdog                                    #
    # ------------------------------------------------------------------ #
    def get_equity(self) -> float:
        unrealised = 0.0
        for pos in self.positions.values():
            szi = float(pos["szi"])
            if szi == 0:
                continue
            mid = self._get_mid(pos["coin"])
            if mid is None:
                continue
            entry = float(pos["entryPx"])
            unrealised += szi * (mid - entry)
        return self.starting_equity + self.realised_pnl + unrealised

    # ------------------------------------------------------------------ #
    #  Internal book-keeping                                               #
    # ------------------------------------------------------------------ #
    def _get_mid(self, coin: str) -> float | None:
        try:
            mids = self.info.all_mids()
            raw = mids.get(coin)
            return float(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("paper | failed to fetch mid for {}: {}", coin, exc)
            return None

    def _apply_fill(self, coin: str, side: Side, size: float, price: float, fee: float) -> None:
        delta = size if side == Side.BUY else -size
        pos = self.positions.get(coin) or {"coin": coin, "szi": "0", "entryPx": "0"}
        current_szi = float(pos["szi"])
        new_szi = current_szi + delta

        if current_szi == 0:
            pos["entryPx"] = str(price)
        elif (current_szi > 0) == (delta > 0):
            # Same side — weighted-average the entry
            old_value = current_szi * float(pos["entryPx"])
            new_value = old_value + delta * price
            pos["entryPx"] = str(new_value / new_szi) if new_szi != 0 else "0"
        else:
            # Reducing or flipping — realise P&L on the closed portion
            closed = min(abs(current_szi), abs(delta))
            entry = float(pos["entryPx"])
            direction = 1.0 if current_szi > 0 else -1.0
            self.realised_pnl += direction * (price - entry) * closed
            if abs(delta) > abs(current_szi):
                # Flipped: remaining size opens fresh at this price
                pos["entryPx"] = str(price)

        pos["szi"] = str(new_szi) if abs(new_szi) > 1e-12 else "0"
        if pos["szi"] == "0":
            pos["entryPx"] = "0"

        self.realised_pnl -= fee
        self.fees_paid += fee
        self.positions[coin] = pos
        self.fills.append(PaperFill(coin=coin, side=side, size=size, fill_price=price, fee=fee))
