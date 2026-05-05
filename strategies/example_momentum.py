"""
Very simple momentum stub: buy when last N trades are net-positive volume, sell otherwise.
Not a real strategy — shows how to wire on_trade into orders + risk.
"""
from collections import deque

from loguru import logger

from execution import Side
from .base import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, *args, window: int = 20, size: float = 0.001, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window = window
        self._size = size
        self._sides: deque[str] = deque(maxlen=window)
        self._in_position = False

    def name(self) -> str:
        return "momentum"

    async def on_trade(self, trade: dict) -> None:
        self._sides.append(trade.get("side", ""))
        if len(self._sides) < self._window:
            return

        buys = self._sides.count("B")
        sells = self._sides.count("A")
        price = float(trade.get("px", 0))
        size_usd = price * self._size

        position = self.orders.get_position(self.coin)
        pos_usd = float(position["positionValue"]) if position else 0.0

        if buys > sells and not self._in_position:
            if self.risk.check_order("B", size_usd, pos_usd):
                result = self.orders.market_order(self.coin, Side.BUY, self._size)
                if result.success:
                    self._in_position = True
                    self.risk.register_order()
                    logger.info("Entered long {} @ {}", self.coin, price)

        elif sells > buys and self._in_position:
            result = self.orders.market_order(self.coin, Side.SELL, self._size)
            if result.success:
                self._in_position = False
                self.risk.release_order()
                logger.info("Exited long {} @ {}", self.coin, price)
