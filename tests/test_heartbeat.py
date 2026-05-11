"""Tests for runtime.heartbeat."""
from __future__ import annotations

import asyncio

import pytest

from runtime import heartbeat_loop, heartbeat_once


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeSession:
    """Minimal stand-in for aiohttp.ClientSession — only what heartbeat_once uses."""

    def __init__(self, responder=None) -> None:
        self.calls: list[str] = []
        self.responder = responder or (lambda url: _FakeResp(200))

    def get(self, url: str):
        self.calls.append(url)
        return self.responder(url)

    async def close(self) -> None:
        return None


def test_heartbeat_once_success():
    sess = _FakeSession()
    ok = asyncio.run(heartbeat_once("https://ping.invalid/x", sess))    # type: ignore[arg-type]
    assert ok is True
    assert sess.calls == ["https://ping.invalid/x"]


def test_heartbeat_once_swallows_http_error():
    sess = _FakeSession(responder=lambda url: _FakeResp(503))
    ok = asyncio.run(heartbeat_once("https://ping.invalid/x", sess))    # type: ignore[arg-type]
    assert ok is False


def test_heartbeat_once_swallows_network_error():
    class _Raiser:
        def get(self, url):
            class _Boom:
                async def __aenter__(self): raise ConnectionError("dns")
                async def __aexit__(self, *a): return False
            return _Boom()
    ok = asyncio.run(heartbeat_once("https://x", _Raiser()))    # type: ignore[arg-type]
    assert ok is False


def test_heartbeat_loop_returns_immediately_when_url_unset():
    # Must not hit the network or sleep — completes promptly.
    asyncio.run(asyncio.wait_for(heartbeat_loop(None), timeout=1.0))
    asyncio.run(asyncio.wait_for(heartbeat_loop(""), timeout=1.0))


def test_heartbeat_loop_pings_then_sleeps():
    """Run the loop in the background, cancel after one ping, verify it pinged."""
    sess = _FakeSession()

    async def runner():
        task = asyncio.create_task(
            heartbeat_loop(
                "https://ping.invalid/x",
                interval_seconds=10.0,        # long sleep so we cancel mid-wait
                session=sess,
            )
        )
        # Let it run one iteration and enter the sleep.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(runner())
    assert sess.calls == ["https://ping.invalid/x"]
