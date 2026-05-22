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

Dispatch modes:
    sync (default)   — ``send()`` calls urllib inline. Useful in tests.
    async_dispatch   — ``send()`` enqueues; a background daemon thread drains
                       the queue and issues the HTTP requests. The asyncio
                       event loop never blocks on a slow webhook.
                       ``make_notifier_from_env()`` opts in to this mode.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import urllib.request
from typing import Any


class Notifier:
    def __init__(
        self,
        webhook_url: str | None = None,
        min_level: str = "WARNING",
        timeout: float = 5.0,
        async_dispatch: bool = False,
        queue_max_size: int = 200,
    ) -> None:
        self.webhook_url = webhook_url
        self.min_level = min_level
        self.timeout = timeout
        self.async_dispatch = bool(async_dispatch) and bool(webhook_url)
        self._queue: queue.Queue[str] | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        if self.async_dispatch:
            self._queue = queue.Queue(maxsize=queue_max_size)
            self._worker = threading.Thread(
                target=self._drain, name="notifier-worker", daemon=True,
            )
            self._worker.start()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str, level: str = "INFO") -> None:
        if not self.enabled:
            return
        payload = f"[{level}] {message}"
        if self.async_dispatch and self._queue is not None:
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                sys.stderr.write("[notify] queue full, dropping message\n")
            return
        self._post(payload)

    def sink(self, message: Any) -> None:
        """Loguru sink callback. Registered via ``logger.add(notifier.sink, ...)``."""
        record = message.record
        self.send(record["message"], level=record["level"].name)

    def flush(self, timeout: float = 2.0) -> bool:
        """Block until the queue drains (or timeout). Returns True on full drain."""
        if self._queue is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty():
                return True
            time.sleep(0.01)
        return self._queue.empty()

    def close(self) -> None:
        """Stop the background worker. Safe to call multiple times."""
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _drain(self) -> None:
        assert self._queue is not None
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._post(payload)

    def _post(self, payload: str) -> None:
        try:
            data = json.dumps({"content": payload}).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,    # type: ignore[arg-type]
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception as exc:
            sys.stderr.write(f"[notify] webhook failed: {exc}\n")


def make_notifier_from_env() -> Notifier:
    return Notifier(
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        min_level=os.getenv("ALERT_MIN_LEVEL", "WARNING"),
        async_dispatch=True,
    )
