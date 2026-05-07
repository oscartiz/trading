"""Tests for backtesting/data.py — cache behavior + fetch logic with HTTP mocked."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtesting import data as data_mod


def _t(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
#  _cache_path                                                                 #
# --------------------------------------------------------------------------- #


def test_cache_path_includes_coin_and_window():
    p = data_mod._cache_path("funding", "BTC", _t(2024), _t(2025))
    assert p.name == f"funding_BTC_{int(_t(2024).timestamp())}_{int(_t(2025).timestamp())}.parquet"
    assert p.parent.name == "cache"


def test_cache_path_extra_suffix_used_for_intervals():
    p = data_mod._cache_path("prices_bn", "ETH", _t(2024), _t(2025), extra="_4h")
    assert "_4h_" in p.name


# --------------------------------------------------------------------------- #
#  fetch_funding_history                                                       #
# --------------------------------------------------------------------------- #


def _redirect_cache(tmp_path: Path):
    return patch.object(data_mod, "CACHE_DIR", tmp_path)


def test_funding_cache_hit_skips_network(tmp_path):
    """When the parquet exists, no _info() / Info.funding_history call should happen."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([_t(2024)], utc=True),
        "funding_rate": [0.0001],
    })
    cache_path = tmp_path / f"funding_BTC_{int(_t(2024).timestamp())}_{int(_t(2025).timestamp())}.parquet"
    df.to_parquet(cache_path)

    fake_info_factory = MagicMock(side_effect=AssertionError("must not be called"))

    with _redirect_cache(tmp_path), patch.object(data_mod, "_info", fake_info_factory):
        out = data_mod.fetch_funding_history("BTC", _t(2024), _t(2025))

    assert len(out) == 1
    assert "funding_rate" in out.columns
    fake_info_factory.assert_not_called()


def test_funding_cache_miss_fetches_and_writes(tmp_path):
    fake_info = MagicMock()
    # First call returns two records; second call returns empty so the loop terminates.
    fake_info.funding_history.side_effect = [
        [
            {"time": int(_t(2024).timestamp() * 1000), "fundingRate": "0.0001"},
            {"time": int(_t(2024, 6, 1).timestamp() * 1000), "fundingRate": "0.0002"},
        ],
        [],
    ]
    info_factory = MagicMock(return_value=fake_info)

    with _redirect_cache(tmp_path), patch.object(data_mod, "_info", info_factory):
        out = data_mod.fetch_funding_history("BTC", _t(2024), _t(2025))

    assert len(out) == 2
    assert list(out.columns) == ["timestamp", "funding_rate"]
    assert out["funding_rate"].iloc[0] == pytest.approx(0.0001)
    cache_files = list(tmp_path.glob("funding_BTC_*.parquet"))
    assert len(cache_files) == 1


def test_funding_force_refetches_even_with_cache(tmp_path):
    """force=True must skip the cache and call the API."""
    cache_path = tmp_path / f"funding_BTC_{int(_t(2024).timestamp())}_{int(_t(2025).timestamp())}.parquet"
    pd.DataFrame({
        "timestamp": pd.to_datetime([_t(2024)], utc=True),
        "funding_rate": [0.0],
    }).to_parquet(cache_path)

    fake_info = MagicMock()
    fake_info.funding_history.return_value = [
        {"time": int(_t(2024, 3, 1).timestamp() * 1000), "fundingRate": "0.0009"},
    ]
    info_factory = MagicMock(return_value=fake_info)

    with _redirect_cache(tmp_path), patch.object(data_mod, "_info", info_factory):
        out = data_mod.fetch_funding_history("BTC", _t(2024), _t(2025), force=True)

    assert out["funding_rate"].iloc[0] == pytest.approx(0.0009)


def test_funding_paginates_until_cursor_does_not_advance(tmp_path):
    """If the API stops returning new rows the loop must terminate."""
    fake_info = MagicMock()
    # First call returns one row, second call returns same timestamp → loop must break.
    fake_info.funding_history.side_effect = [
        [{"time": 1_000, "fundingRate": "0.0001"}],
        [{"time": 1_000, "fundingRate": "0.0001"}],   # cursor would not advance
        AssertionError("should have exited"),
    ]
    info_factory = MagicMock(return_value=fake_info)

    with _redirect_cache(tmp_path), patch.object(data_mod, "_info", info_factory):
        out = data_mod.fetch_funding_history("BTC", _t(2024), _t(2025))

    assert len(out) == 1


# --------------------------------------------------------------------------- #
#  fetch_price_history                                                         #
# --------------------------------------------------------------------------- #


def test_price_cache_hit_skips_network(tmp_path):
    cache_path = tmp_path / f"prices_bn_BTC_1h_{int(_t(2024).timestamp())}_{int(_t(2025).timestamp())}.parquet"
    pd.DataFrame({
        "timestamp": pd.to_datetime([_t(2024)], utc=True),
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1.0],
    }).to_parquet(cache_path)

    requests_mock = MagicMock(side_effect=AssertionError("must not be called"))
    with _redirect_cache(tmp_path), patch.object(data_mod.requests, "get", requests_mock):
        out = data_mod.fetch_price_history("BTC", _t(2024), _t(2025), interval="1h")

    assert len(out) == 1
    requests_mock.assert_not_called()


def test_price_unknown_coin_raises(tmp_path):
    with _redirect_cache(tmp_path):
        with pytest.raises(ValueError, match="No Binance symbol"):
            data_mod.fetch_price_history("UNKNOWN_COIN", _t(2024), _t(2025))


def test_price_cache_miss_fetches_and_parses(tmp_path):
    """Single-page Binance response → DataFrame with the right shape."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, q, n, tbb, tbq, ignore]
        [int(_t(2024).timestamp() * 1000), "100", "101", "99", "100.5", "1.0",
         int(_t(2024).timestamp() * 1000) + 3_600_000, "0", 1, "0", "0", "0"],
    ]
    requests_mock = MagicMock(return_value=response)

    with _redirect_cache(tmp_path), patch.object(data_mod.requests, "get", requests_mock):
        out = data_mod.fetch_price_history("BTC", _t(2024), _t(2025), interval="1h")

    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out["close"].iloc[0] == pytest.approx(100.5)
    assert out["volume"].iloc[0] == pytest.approx(1.0)
    cache_files = list(tmp_path.glob("prices_bn_BTC*.parquet"))
    assert len(cache_files) == 1


def test_price_drops_duplicates_and_sorts(tmp_path):
    """Two batches where batch B's first row duplicates batch A's last must dedupe."""
    base_ms = int(_t(2024).timestamp() * 1000)

    def kline(ms):
        return [ms, "100", "101", "99", "100.5", "1.0", ms + 3_600_000, "0", 1, "0", "0", "0"]

    # Full first batch → loop continues. Second batch: a duplicate row + one new
    # row, length < limit → loop terminates.
    batch_a = [kline(base_ms + 3_600_000 * i) for i in range(1000)]
    last_a = batch_a[-1][0]
    batch_b = [kline(last_a), kline(last_a + 3_600_000)]   # dup + new

    response_a = MagicMock(raise_for_status=MagicMock())
    response_b = MagicMock(raise_for_status=MagicMock())
    response_a.json.return_value = batch_a
    response_b.json.return_value = batch_b
    requests_mock = MagicMock(side_effect=[response_a, response_b])

    with _redirect_cache(tmp_path), patch.object(data_mod.requests, "get", requests_mock):
        out = data_mod.fetch_price_history("BTC", _t(2024), _t(2025), interval="1h")

    # 1000 from batch A, 1 dup ignored, 1 new from batch B → 1001 unique.
    assert len(out) == 1001
    assert out["timestamp"].is_monotonic_increasing
    assert not out["timestamp"].duplicated().any()
