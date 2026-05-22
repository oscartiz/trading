"""Minimal Prometheus-style /metrics endpoint.

A single shared `MetricsRegistry` holds the strategy's last observed gauge
values. The HTTP server emits them in Prometheus text format on /metrics.

Stdlib-only on the server side — uses aiohttp (already a dependency) for
the async HTTP server. No Prometheus client lib is pulled in; for a single
process and a handful of gauges, hand-rolling the text format is simpler
than the dependency.

Wiring: the strategy calls ``registry.set("regime_p_bull", 0.83)`` each
tick. Grafana / a Prometheus scraper reads ``http://host:port/metrics``.

When `METRICS_PORT` is unset, ``serve_metrics`` returns immediately so the
overhead is zero in tests, paper mode, or anywhere the user hasn't asked
for it.
"""
from __future__ import annotations

import threading
from typing import Mapping

from aiohttp import web
from loguru import logger


class MetricsRegistry:
    """Thread-safe key→float gauge registry."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._labels: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def set(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        with self._lock:
            self._values[name] = float(value)
            if labels:
                self._labels[name] = dict(labels)

    def snapshot(self) -> dict[str, tuple[float, dict[str, str]]]:
        with self._lock:
            return {k: (v, dict(self._labels.get(k, {}))) for k, v in self._values.items()}

    def render(self) -> str:
        """Emit Prometheus text format."""
        lines: list[str] = []
        snap = self.snapshot()
        for name in sorted(snap):
            value, labels = snap[name]
            if labels:
                label_str = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")
        return "\n".join(lines) + ("\n" if lines else "")


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Process-wide singleton — strategies, watchdog, and HTTP handler share it.
REGISTRY = MetricsRegistry()


async def _handle_metrics(request: web.Request) -> web.Response:
    return web.Response(
        text=REGISTRY.render(),
        content_type="text/plain",
        charset="utf-8",
    )


async def serve_metrics(port: int | None, host: str = "127.0.0.1") -> None:
    """Run the /metrics HTTP server. No-op when port is None/0.

    Binds to localhost by default — the metrics endpoint is meant to be
    scraped from the same host (or via SSH tunnel), not exposed publicly.
    """
    if not port:
        return
    app = web.Application()
    app.router.add_get("/metrics", _handle_metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=int(port))
    await site.start()
    logger.info("metrics endpoint started | http://{}:{}/metrics", host, port)
    try:
        # Block forever — the caller cancels via asyncio.gather/cancel.
        await _forever()
    finally:
        await runner.cleanup()


async def _forever() -> None:
    import asyncio
    while True:
        await asyncio.sleep(3600)
