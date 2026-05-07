from abc import ABC, abstractmethod

from loguru import logger

from execution import OrderManager, Side
from risk import RiskManager


class Strategy(ABC):
    """
    All strategies inherit from this. Subclass and implement on_trade and/or on_book.
    The run loop is managed externally (see main.py).
    """

    def __init__(self, coin: str, order_manager: OrderManager, risk: RiskManager) -> None:
        self.coin = coin
        self.orders = order_manager
        self.risk = risk
        # Set by _reconcile_with_exchange when restored state disagrees with the
        # live exchange position. Strategies must check this in their entry path.
        self._block_entries: bool = False

    async def on_trade(self, trade: dict) -> None:
        """Called for every trade tick on self.coin."""

    async def on_book(self, book: dict) -> None:
        """Called on every L2 book update for self.coin."""

    async def run(self) -> None:
        """Override for polling-based strategies. Default is a no-op."""

    @abstractmethod
    def name(self) -> str: ...

    # ------------------------------------------------------------------ #
    #  Startup reconciliation                                              #
    # ------------------------------------------------------------------ #
    def _reconcile_with_exchange(
        self, expected_in_position: bool, expected_side: Side | None,
    ) -> None:
        """Compare expected (restored) state against the exchange's view.

        Sets _block_entries when they disagree so the strategy refuses new
        entries until the position is manually flattened. Existing positions
        can still be exited via the strategy's normal exit logic.
        """
        pos = self.orders.get_position(self.coin)
        szi = float(pos["szi"]) if pos and pos.get("szi") is not None else 0.0
        exchange_in_position = szi != 0
        exchange_side: Side | None = None
        if szi > 0:
            exchange_side = Side.BUY
        elif szi < 0:
            exchange_side = Side.SELL

        if (exchange_in_position == expected_in_position
                and exchange_side == expected_side):
            if expected_in_position:
                logger.info(
                    "{} | startup state matches exchange | side={} szi={}",
                    self.coin, expected_side, szi,
                )
            return

        self._block_entries = True
        logger.error(
            "{} | startup mismatch | restored(in_position={}, side={}) vs "
            "exchange(szi={}, side={}) — new entries blocked until flat",
            self.coin, expected_in_position, expected_side, szi, exchange_side,
        )
