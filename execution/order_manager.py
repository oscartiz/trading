from dataclasses import dataclass, field
from enum import StrEnum

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from loguru import logger


class Side(StrEnum):
    BUY = "B"
    SELL = "A"


@dataclass
class OrderResult:
    success: bool
    order_id: int | None = None
    error: str | None = None
    raw: dict = field(default_factory=dict)
    fill_price: float | None = None
    fee: float | None = None


class OrderManager:
    def __init__(self, info: Info, exchange: Exchange) -> None:
        self.info = info
        self.exchange = exchange

    def market_order(self, coin: str, side: Side, size: float) -> OrderResult:
        is_buy = side == Side.BUY
        # slippage=True tells the SDK to use aggressive pricing for immediate fill
        result = self.exchange.market_open(coin, is_buy, size)
        return self._parse(result)

    def limit_order(
        self, coin: str, side: Side, size: float, price: float, reduce_only: bool = False
    ) -> OrderResult:
        is_buy = side == Side.BUY
        result = self.exchange.order(coin, is_buy, size, price, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)
        return self._parse(result)

    def cancel_order(self, coin: str, order_id: int) -> OrderResult:
        result = self.exchange.cancel(coin, order_id)
        return self._parse(result)

    def cancel_all(self, coin: str) -> None:
        open_orders = self.info.open_orders(self.exchange.account_address)
        for o in open_orders:
            if o["coin"] == coin:
                self.cancel_order(coin, o["oid"])

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> None:
        self.exchange.update_leverage(leverage, coin, is_cross)
        logger.info("Leverage set | coin={} leverage={}x cross={}", coin, leverage, is_cross)

    def get_position(self, coin: str) -> dict | None:
        state = self.info.user_state(self.exchange.account_address)
        for pos in state.get("assetPositions", []):
            if pos["position"]["coin"] == coin:
                return pos["position"]
        return None

    def _parse(self, raw: dict) -> OrderResult:
        status = raw.get("status")
        if status != "ok":
            error = str(raw.get("response", raw))
            logger.error("Order failed: {}", error)
            return OrderResult(success=False, error=error, raw=raw)
        data = raw.get("response", {}).get("data", {})
        statuses = data.get("statuses", [{}])
        first = statuses[0]
        filled = first.get("filled") or {}
        resting = first.get("resting") or {}
        oid = resting.get("oid") or filled.get("oid")
        fill_price: float | None = None
        if filled.get("avgPx") is not None:
            try:
                fill_price = float(filled["avgPx"])
            except (TypeError, ValueError):
                fill_price = None
        logger.info("Order ok | oid={} data={}", oid, data)
        return OrderResult(success=True, order_id=oid, raw=raw, fill_price=fill_price)
