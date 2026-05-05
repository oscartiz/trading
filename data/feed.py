import asyncio
import json
from collections import defaultdict, deque
from typing import Callable

import websockets
from loguru import logger

from config import settings

# WS endpoints follow the same base pattern but on a different subdomain
_WS_URLS = {
    True: "wss://api.hyperliquid-testnet.xyz/ws",
    False: "wss://api.hyperliquid.xyz/ws",
}


class MarketFeed:
    """
    Async WebSocket feed for Hyperliquid L2 book and trade data.

    Usage:
        feed = MarketFeed(["BTC", "ETH"])
        feed.on_trade("BTC", my_callback)
        await feed.run()
    """

    def __init__(self, coins: list[str], maxlen: int = 1000) -> None:
        self.coins = coins
        self._ws_url = _WS_URLS[settings.testnet]
        self._trade_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._book_callbacks: dict[str, list[Callable]] = defaultdict(list)
        self.trades: dict[str, deque] = {c: deque(maxlen=maxlen) for c in coins}
        self.books: dict[str, dict] = {c: {} for c in coins}

    def on_trade(self, coin: str, callback: Callable) -> None:
        self._trade_callbacks[coin].append(callback)

    def on_book(self, coin: str, callback: Callable) -> None:
        self._book_callbacks[coin].append(callback)

    async def run(self) -> None:
        async for ws in websockets.connect(self._ws_url):
            try:
                await self._subscribe(ws)
                async for raw in ws:
                    await self._dispatch(json.loads(raw))
            except websockets.ConnectionClosed:
                logger.warning("WS disconnected, reconnecting…")

    async def _subscribe(self, ws) -> None:
        for coin in self.coins:
            for sub_type in ("trades", "l2Book"):
                msg = json.dumps({"method": "subscribe", "subscription": {"type": sub_type, "coin": coin}})
                await ws.send(msg)
        logger.info("Subscribed to {} coins", len(self.coins))

    async def _dispatch(self, msg: dict) -> None:
        channel = msg.get("channel")
        data = msg.get("data", {})

        if channel == "trades":
            for trade in data:
                coin = trade.get("coin")
                if coin:
                    self.trades[coin].append(trade)
                    for cb in self._trade_callbacks.get(coin, []):
                        await _maybe_await(cb(trade))

        elif channel == "l2Book":
            coin = data.get("coin")
            if coin:
                self.books[coin] = data
                for cb in self._book_callbacks.get(coin, []):
                    await _maybe_await(cb(data))


async def _maybe_await(result):
    if asyncio.iscoroutine(result):
        await result
