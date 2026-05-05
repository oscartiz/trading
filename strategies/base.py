from abc import ABC, abstractmethod

from execution import OrderManager
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

    async def on_trade(self, trade: dict) -> None:
        """Called for every trade tick on self.coin."""

    async def on_book(self, book: dict) -> None:
        """Called on every L2 book update for self.coin."""

    @abstractmethod
    def name(self) -> str: ...
