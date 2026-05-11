"""Tests for runtime.notify.Notifier."""
from __future__ import annotations

import json
from types import SimpleNamespace

from loguru import logger

from runtime import Notifier, make_notifier_from_env


class _FakeResp:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_urlopen(monkeypatch, captured: list):
    def _fake_urlopen(req, timeout=None):
        captured.append({
            "url": req.full_url,
            "data": req.data,
            "headers": dict(req.headers),
            "method": req.get_method(),
            "timeout": timeout,
        })
        return _FakeResp()
    monkeypatch.setattr("runtime.notify.urllib.request.urlopen", _fake_urlopen)


def test_notifier_disabled_when_no_url(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    n = Notifier(webhook_url=None)
    assert n.enabled is False
    n.send("hello")
    assert captured == []


def test_notifier_posts_json_payload(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    n = Notifier(webhook_url="https://example.invalid/hook", timeout=2.0)
    n.send("position opened", level="ENTRY")

    assert len(captured) == 1
    call = captured[0]
    assert call["url"] == "https://example.invalid/hook"
    assert call["method"] == "POST"
    assert call["timeout"] == 2.0
    body = json.loads(call["data"].decode())
    assert body == {"content": "[ENTRY] position opened"}


def test_notifier_swallows_http_errors(monkeypatch, capsys):
    def _boom(*_a, **_kw):
        raise ConnectionError("no route to host")
    monkeypatch.setattr("runtime.notify.urllib.request.urlopen", _boom)
    n = Notifier(webhook_url="https://example.invalid/hook")
    # Must not raise.
    n.send("anything")
    err = capsys.readouterr().err
    assert "webhook failed" in err
    assert "no route to host" in err


def test_notifier_sink_routes_loguru_record(monkeypatch):
    """When wired as a loguru sink, WARN+ log lines should fire send()."""
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    n = Notifier(webhook_url="https://example.invalid/hook")

    sink_id = logger.add(n.sink, level="WARNING")
    try:
        logger.info("not loud enough")        # below threshold
        logger.warning("drawdown halt")
        logger.error("refit failed")
    finally:
        logger.remove(sink_id)

    levels = [json.loads(c["data"].decode())["content"] for c in captured]
    # INFO filtered, WARNING and ERROR delivered
    assert len(levels) == 2
    assert any("WARNING" in m and "drawdown halt" in m for m in levels)
    assert any("ERROR" in m and "refit failed" in m for m in levels)


def test_make_notifier_from_env_reads_envvars(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hook.invalid/x")
    monkeypatch.setenv("ALERT_MIN_LEVEL", "ERROR")
    n = make_notifier_from_env()
    assert n.webhook_url == "https://hook.invalid/x"
    assert n.min_level == "ERROR"
    assert n.enabled is True


def test_make_notifier_from_env_no_url_disabled(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    n = make_notifier_from_env()
    assert n.enabled is False


def test_sink_uses_record_level_name(monkeypatch):
    """Direct unit-level check: sink translates the loguru record into send()."""
    sent: list = []
    n = Notifier(webhook_url="https://example.invalid/hook")
    monkeypatch.setattr(n, "send", lambda msg, level="INFO": sent.append((msg, level)))
    fake_message = SimpleNamespace(record={
        "message": "halt fired",
        "level": SimpleNamespace(name="ERROR"),
    })
    n.sink(fake_message)
    assert sent == [("halt fired", "ERROR")]
