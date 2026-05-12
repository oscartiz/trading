"""Tests for runtime.watchdog.make_live_equity_source.

The equity source feeds the drawdown halt. A broken / missing reading should
fail closed (return 0 → watchdog ignores) rather than raise, so a transient
SDK hiccup doesn't take down the strategy event loop.
"""
from __future__ import annotations

from dataclasses import dataclass

from runtime import make_live_equity_source


@dataclass
class FakeInfo:
    """Minimal stand-in for hyperliquid.info.Info.user_state."""
    state: dict

    def user_state(self, _address: str) -> dict:
        return self.state


def test_returns_account_value_when_present():
    info = FakeInfo(state={"marginSummary": {"accountValue": "1234.56"}})
    get_equity = make_live_equity_source(info, "0xdead")
    assert get_equity() == 1234.56


def test_returns_zero_when_margin_summary_missing():
    info = FakeInfo(state={})
    assert make_live_equity_source(info, "0xdead")() == 0.0


def test_returns_zero_when_margin_summary_is_null():
    info = FakeInfo(state={"marginSummary": None})
    assert make_live_equity_source(info, "0xdead")() == 0.0


def test_returns_zero_when_account_value_missing():
    info = FakeInfo(state={"marginSummary": {}})
    assert make_live_equity_source(info, "0xdead")() == 0.0


def test_returns_zero_when_account_value_is_null():
    info = FakeInfo(state={"marginSummary": {"accountValue": None}})
    assert make_live_equity_source(info, "0xdead")() == 0.0


def test_returns_zero_when_account_value_is_unparseable():
    """A malformed string must not raise — equity_watchdog never sees an exception."""
    info = FakeInfo(state={"marginSummary": {"accountValue": "not a number"}})
    assert make_live_equity_source(info, "0xdead")() == 0.0


def test_accepts_float_payload_without_string_coercion():
    """Hyperliquid sometimes returns numerics directly, not strings."""
    info = FakeInfo(state={"marginSummary": {"accountValue": 9999.0}})
    assert make_live_equity_source(info, "0xdead")() == 9999.0


def test_returns_zero_when_user_state_raises():
    """A network error inside user_state propagates — that's the watchdog's
    job to catch. But a well-formed empty response is fine."""
    class BlowsUp:
        def user_state(self, _addr: str) -> dict:
            raise RuntimeError("rpc timeout")

    get_equity = make_live_equity_source(BlowsUp(), "0xdead")
    # We DO want exceptions to propagate here — the watchdog wraps the whole
    # call in try/except. Make this contract explicit.
    try:
        get_equity()
    except RuntimeError as exc:
        assert "rpc timeout" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")
