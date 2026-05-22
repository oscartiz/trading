"""Unit tests for execution.order_manager.OrderManager._parse.

The Hyperliquid SDK returns deeply-nested dicts whose shape changes between
'filled' and 'resting' orders, success and failure. The strategy depends on
OrderResult.fill_price being correctly populated for the journal and the
notifier, and on success/error being correctly classified. Pin the shapes we
rely on so the SDK can't silently change them out from under us.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from execution.order_manager import OrderManager, OrderResult


def _make_manager() -> OrderManager:
    """Build an OrderManager wired to dummies — only _parse is under test."""
    return OrderManager(info=MagicMock(), exchange=MagicMock())


def test_parse_success_with_fill_price():
    """A market order that filled returns success + fill_price."""
    raw = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 4242, "avgPx": "65000.5", "totalSz": "0.001"}}
                ]
            }
        },
    }
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.order_id == 4242
    assert result.fill_price == 65000.5
    assert result.error is None
    assert result.raw is raw


def test_parse_success_with_resting_no_fill_price():
    """A limit order that rests on the book has an oid but no fill_price."""
    raw = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"resting": {"oid": 999}}
                ]
            }
        },
    }
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.order_id == 999
    assert result.fill_price is None


def test_parse_failure_returns_error_string():
    """status != 'ok' is treated as failure; response payload becomes the error."""
    raw = {"status": "err", "response": "insufficient margin"}
    result = _make_manager()._parse(raw)
    assert result.success is False
    assert result.order_id is None
    assert result.fill_price is None
    assert "insufficient margin" in (result.error or "")


def test_parse_handles_missing_response_block():
    """A success status with no response.data shape must not raise."""
    raw = {"status": "ok"}
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.order_id is None
    assert result.fill_price is None


def test_parse_handles_non_numeric_avg_px():
    """A malformed avgPx string must not raise; fill_price falls back to None."""
    raw = {
        "status": "ok",
        "response": {
            "data": {"statuses": [{"filled": {"oid": 1, "avgPx": "n/a"}}]}
        },
    }
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.order_id == 1
    assert result.fill_price is None


def test_parse_returns_oid_when_filled_and_resting_both_absent():
    """An empty statuses entry returns success with no oid/fill — defensive."""
    raw = {"status": "ok", "response": {"data": {"statuses": [{}]}}}
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.order_id is None
    assert result.fill_price is None


def test_parse_failure_with_dict_response_stringifies():
    """If the response payload is a dict it should still serialise into error."""
    raw = {"status": "err", "response": {"code": 429, "message": "rate limit"}}
    result = _make_manager()._parse(raw)
    assert result.success is False
    # The dict gets str()'d — the actual content must survive into the error
    # field so the alerting webhook has something useful to display.
    assert "429" in (result.error or "")
    assert "rate limit" in (result.error or "")


def test_parse_returned_object_is_OrderResult():
    """Smoke test on the contract: _parse returns the right type."""
    raw = {"status": "ok", "response": {"data": {"statuses": [{}]}}}
    assert isinstance(_make_manager()._parse(raw), OrderResult)


def test_parse_captures_live_fee_from_filled_status():
    """The Hyperliquid SDK puts the fee on the filled status — the journal
    relies on this being captured so live P&L reconciliation isn't fee-blind."""
    raw = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "avgPx": "65000.0", "totalSz": "0.001", "fee": "0.0228"}}
                ]
            }
        },
    }
    result = _make_manager()._parse(raw)
    assert result.success is True
    assert result.fee == 0.0228


def test_parse_falls_back_to_feeUsd_field():
    """Older SDK versions named the field feeUsd — accept both."""
    raw = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "avgPx": "100", "totalSz": "1", "feeUsd": "0.42"}}
                ]
            }
        },
    }
    result = _make_manager()._parse(raw)
    assert result.fee == 0.42


def test_parse_missing_fee_stays_none():
    """No fee field in the response → OrderResult.fee is None (not zero, to
    distinguish 'unknown' from 'free trade')."""
    raw = {
        "status": "ok",
        "response": {
            "data": {"statuses": [{"filled": {"oid": 1, "avgPx": "100", "totalSz": "1"}}]}
        },
    }
    result = _make_manager()._parse(raw)
    assert result.fee is None
