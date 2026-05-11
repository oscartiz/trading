"""Webhook alerting.

A tiny notifier that pushes one-line messages to a Discord-compatible webhook
(also works for any receiver that accepts JSON ``{"content": "..."}``).

Wired two ways:
    1. As a loguru sink for WARNING+ — catches halts, refit failures,
       reconciliation mismatches, and any other elevated log line.
    2. Explicit ``notifier.send(...)`` calls from strategy entries and exits,
       which log at INFO and would otherwise be filtered out.

When ``ALERT_WEBHOOK_URL`` is unset the notifier silently no-ops, so unit tests
and local backtests don't reach for the network.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any


class Notifier:
    def __init__(
        self,
        webhook_url: str | None = None,
        min_level: str = "WARNING",
        timeout: float = 5.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.min_level = min_level
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str, level: str = "INFO") -> None:
        if not self.enabled:
            return
        try:
            data = json.dumps({"content": f"[{level}] {message}"}).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,    # type: ignore[arg-type]
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception as exc:
            # Write to stderr (not loguru) to avoid recursing through our own sink.
            sys.stderr.write(f"[notify] webhook failed: {exc}\n")

    def sink(self, message: Any) -> None:
        """Loguru sink callback. Registered via ``logger.add(notifier.sink, ...)``."""
        record = message.record
        self.send(record["message"], level=record["level"].name)


def make_notifier_from_env() -> Notifier:
    return Notifier(
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        min_level=os.getenv("ALERT_MIN_LEVEL", "WARNING"),
    )
