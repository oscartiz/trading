"""Download and cache historical price data for backtesting.

Price data: Binance perp klines (unlimited history, no key needed) — Hyperliquid
only retains a short rolling window, so we use Binance for historical bars.

Funding data: Binance funding-rate history at the same endpoint family.
Funding is charged every 8 hours; the backtest engine consumes the series
to assess the funding drag on multi-day positions.
"""
from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from loguru import logger

CACHE_DIR = Path("data/cache")
_BN_LIMIT = 1000                       # Binance max candles per request
_MAX_ITERS = 500
_BN_FUNDING_LIMIT = 1000               # Binance funding history max per request

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


def _cache_path(prefix: str, coin: str, start: datetime, end: datetime, extra: str = "") -> Path:
    name = f"{prefix}_{coin}{extra}_{int(start.timestamp())}_{int(end.timestamp())}.parquet"
    return CACHE_DIR / name


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


def fetch_funding_history(
    coin: str,
    start: datetime,
    end: datetime,
    force: bool = False,
) -> pd.DataFrame:
    """Return DataFrame[timestamp (UTC), funding_rate] from Binance perps.

    Each row is one funding settlement (every 8h on Binance). Use this to
    accrue per-position funding drag in the backtest engine — the live
    bot pulls equivalent values from Hyperliquid via the trading API and
    records them in the journal.
    """
    path = _cache_path("funding_bn", coin, start, end)
    if path.exists() and not force:
        logger.debug("Loading cached funding history from {}", path)
        return pd.read_parquet(path)

    symbol = _BINANCE_SYMBOL.get(coin)
    if symbol is None:
        raise ValueError(f"No Binance symbol mapping for '{coin}'. Add it in data.py.")

    logger.info("Fetching funding history for {} ({} → {})", coin, start.date(), end.date())
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    records: list[dict] = []
    cursor = start_ms

    for _ in range(_MAX_ITERS):
        if cursor >= end_ms:
            break
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _BN_FUNDING_LIMIT,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        records.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < _BN_FUNDING_LIMIT:
            break
        sleep(0.1)

    df = pd.DataFrame(records, columns=["fundingTime", "fundingRate", "symbol"])
    if df.empty:
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
                           "funding_rate": pd.Series([], dtype="float64")})
    else:
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = (
            df[["timestamp", "funding_rate"]]
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Fetched {} funding entries", len(df))
    return df
