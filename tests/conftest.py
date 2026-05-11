"""Shared fixtures and builders for the test suite.

These builders construct **deterministic** synthetic data so engine tests can
assert exact P&L and trade timing — they intentionally avoid any randomness.
For HMM-flavoured tests use the fixtures in test_hmm.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    """Redirect persistent strategy state into a per-test tmp dir.

    Strategies write JSON sidecar files to `state/{name}_{coin}.json` and an
    append-only journal to `state/fills_{coin}.jsonl`. Without this fixture,
    tests would share those files and pollute each other.
    """
    from runtime import journal as journal_mod
    from runtime import state as state_mod
    monkeypatch.setattr(state_mod, "DEFAULT_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(journal_mod, "DEFAULT_STATE_DIR", tmp_path / "state")

# ----- price-series builders ---------------------------------------------- #


def build_prices(
    closes: list[float],
    *,
    start: datetime | None = None,
    interval_hours: int = 1,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build an OHLC DataFrame from a list of closes.

    `highs` and `lows` default to the close so stop/TP tests can opt in to
    intrabar excursions explicitly.
    """
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n = len(closes)
    timestamps = [start + timedelta(hours=i * interval_hours) for i in range(n)]
    closes_arr = np.asarray(closes, dtype=float)
    highs_arr = np.asarray(highs if highs is not None else closes, dtype=float)
    lows_arr = np.asarray(lows if lows is not None else closes, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "open": closes_arr,
        "high": highs_arr,
        "low": lows_arr,
        "close": closes_arr,
        "volume": np.ones(n),
    })


def regime_prices(
    n_warmup: int = 60,
    bull_bars: int = 80,
    bear_bars: int = 80,
    chop_bars: int = 40,
    mu_bull: float = 0.004,
    mu_bear: float = -0.004,
    sigma_chop: float = 0.001,
    seed: int = 0,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Strong, well-separated regime sequence so the HMM converges cleanly."""
    rng = np.random.default_rng(seed)
    rets = np.concatenate([
        rng.normal(0.0, sigma_chop, size=n_warmup),
        rng.normal(mu_bull, 0.005, size=bull_bars),
        rng.normal(0.0, sigma_chop, size=chop_bars),
        rng.normal(mu_bear, 0.005, size=bear_bars),
        rng.normal(0.0, sigma_chop, size=chop_bars),
    ])
    closes = start_price * np.exp(np.cumsum(rets))
    return build_prices(list(closes))


def linear_prices(start: float, end: float, n_bars: int) -> pd.DataFrame:
    closes = np.linspace(start, end, n_bars).tolist()
    return build_prices(closes)


# ----- mock execution layer for live strategy tests ----------------------- #


@dataclass
class FakeOrderResult:
    success: bool = True
    order_id: int = 1
    error: str | None = None
    raw: dict = field(default_factory=dict)
    fill_price: float | None = None
    fee: float | None = None


@dataclass
class FakeOrder:
    coin: str
    side: Any   # Side enum value
    size: float


@dataclass
class FakeInfo:
    """Stand-in for hyperliquid.info.Info — only methods strategies call."""
    mid: float = 100.0
    candles: list[dict] = field(default_factory=list)
    user_state_data: dict = field(default_factory=lambda: {"assetPositions": []})

    def all_mids(self) -> dict:
        return {"BTC": self.mid, "ETH": self.mid, "SOL": self.mid}

    def candles_snapshot(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        return [c for c in self.candles if start_ms <= int(c["t"]) <= end_ms]

    def user_state(self, _addr: str) -> dict:
        return self.user_state_data


class FakeOrderManager:
    """Mocks execution.OrderManager — captures every call so tests can assert."""

    def __init__(self, info: FakeInfo | None = None) -> None:
        self.info = info or FakeInfo()
        self.orders: list[FakeOrder] = []
        self.set_leverage_calls: list[tuple[str, int, bool]] = []
        self.position: dict | None = None
        self.next_result: FakeOrderResult = FakeOrderResult(success=True)

    def market_order(self, coin: str, side: Any, size: float) -> FakeOrderResult:
        self.orders.append(FakeOrder(coin=coin, side=side, size=size))
        return self.next_result

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> None:
        self.set_leverage_calls.append((coin, leverage, is_cross))

    def get_position(self, coin: str) -> dict | None:
        return self.position
