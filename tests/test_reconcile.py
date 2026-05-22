"""Tests for runtime.reconcile.reconcile_open_orders."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from runtime import reconcile_open_orders


@dataclass
class _FakeInfo:
    orders: list[dict]
    raise_exc: Exception | None = None

    def open_orders(self, _addr: str) -> list[dict]:
        if self.raise_exc:
            raise self.raise_exc
        return self.orders


def test_clean_when_no_open_orders():
    info = _FakeInfo(orders=[])
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC")
    assert clean is True
    assert orphans == []


def test_orphan_open_order_flagged():
    info = _FakeInfo(orders=[
        {"coin": "BTC", "side": "B", "oid": 42, "sz": "0.5", "limitPx": "60000"}
    ])
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC")
    assert clean is False
    assert len(orphans) == 1
    assert orphans[0].oid == 42
    assert orphans[0].size == 0.5
    assert orphans[0].limit_price == 60000


def test_known_oid_is_not_orphan():
    info = _FakeInfo(orders=[
        {"coin": "BTC", "side": "B", "oid": 42, "sz": "0.5", "limitPx": "60000"}
    ])
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC", expected_oids={42})
    assert clean is True
    assert orphans == []


def test_other_coin_orders_ignored():
    info = _FakeInfo(orders=[
        {"coin": "ETH", "side": "B", "oid": 99, "sz": "1.0", "limitPx": "3000"}
    ])
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC")
    assert clean is True
    assert orphans == []


def test_api_failure_returns_clean_empty():
    """A transient API failure must not crash boot — log a warning and proceed."""
    info = _FakeInfo(orders=[], raise_exc=RuntimeError("api timeout"))
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC")
    assert clean is True
    assert orphans == []


def test_malformed_order_skipped(caplog):
    """A single malformed order must not poison the reconcile result."""
    info = _FakeInfo(orders=[
        {"coin": "BTC", "side": "B", "oid": "not-an-int", "sz": "0.5"},
        {"coin": "BTC", "side": "B", "oid": 7, "sz": "1.0", "limitPx": "60000"},
    ])
    clean, orphans = reconcile_open_orders(info, "0xabc", "BTC")
    # The well-formed orphan still gets flagged.
    assert clean is False
    assert {o.oid for o in orphans} == {7}
