"""Download and cache historical funding rates and OHLCV data.

Funding rates: Hyperliquid (full history available).
Price data: Binance perp klines (unlimited history, no key needed) — HL only keeps ~60 days.
"""
from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from hyperliquid.info import Info
from loguru import logger

CACHE_DIR = Path("data/cache")
_MAINNET_URL = "https://api.hyperliquid.xyz"
_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
_HL_BATCH_MS = 90 * 24 * 3_600_000   # 90-day batches for Hyperliquid
_BN_LIMIT = 1000                       # Binance max candles per request
_MAX_ITERS = 500

# Map HL coin names → Binance perpetual symbols
_BINANCE_SYMBOL: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ARB": "ARBUSDT",
    "AVAX": "AVAXUSDT",
    "MATIC": "MATICUSDT",
    "DOGE": "DOGEUSDT",
    "LINK": "LINKUSDT",
    "BNB": "BNBUSDT",
    "OP": "OPUSDT",
    "APT": "APTUSDT",
    "SUI": "SUIUSDT",
}

# Binance interval strings match HL (1h, 4h, 1d, etc.)


def _info(testnet: bool = False) -> Info:
    return Info(_TESTNET_URL if testnet else _MAINNET_URL, skip_ws=True)


def _cache_path(prefix: str, coin: str, start: datetime, end: datetime, extra: str = "") -> Path:
    name = f"{prefix}_{coin}{extra}_{int(start.timestamp())}_{int(end.timestamp())}.parquet"
    return CACHE_DIR / name


def fetch_funding_history(
    coin: str,
    start: datetime,
    end: datetime,
    testnet: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Return DataFrame[timestamp (UTC), funding_rate] at 8-hour settlement intervals."""
    path = _cache_path("funding", coin, start, end)
    if path.exists() and not force:
        logger.debug("Loading cached funding history from {}", path)
        return pd.read_parquet(path)

    logger.info("Fetching funding history for {} ({} → {})", coin, start.date(), end.date())
    info = _info(testnet)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    records: list[dict] = []

    cursor = start_ms
    for _ in range(_MAX_ITERS):
        if cursor >= end_ms:
            break
        batch = info.funding_history(coin, cursor, min(cursor + _HL_BATCH_MS, end_ms))
        if not batch:
            break
        records.extend(batch)
        last_ts = int(batch[-1]["time"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Fetched {} funding records", len(df))
    return df


def fetch_price_history(
    coin: str,
    start: datetime,
    end: datetime,
    interval: str = "1h",
    force: bool = False,
) -> pd.DataFrame:
    """Return DataFrame[timestamp (UTC), open, high, low, close, volume] from Binance perps."""
    path = _cache_path("prices_bn", coin, start, end, f"_{interval}")
    if path.exists() and not force:
        logger.debug("Loading cached price history from {}", path)
        return pd.read_parquet(path)

    symbol = _BINANCE_SYMBOL.get(coin)
    if symbol is None:
        raise ValueError(f"No Binance symbol mapping for '{coin}'. Add it to _BINANCE_SYMBOL in data.py.")

    logger.info("Fetching {} price history for {} from Binance ({} → {})", interval, coin, start.date(), end.date())
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    records: list[list] = []

    cursor = start_ms
    for _ in range(_MAX_ITERS):
        if cursor >= end_ms:
            break
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _BN_LIMIT,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        last_open_ts = int(batch[-1][0])
        if last_open_ts <= cursor:
            break
        cursor = last_open_ts + 1
        if len(batch) < _BN_LIMIT:
            break
        sleep(0.1)  # gentle rate limit

    # Binance kline columns: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(records, columns=[
        "t", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Fetched {} price candles", len(df))
    return df
