"""Append-only trade journal.

One JSONL file per coin at ``state/fills_{coin}.jsonl``. Each entry is a single
``{ts, coin, event, ...}`` record describing an entry or exit fill. The journal
is the audit trail used to reconcile P&L against the exchange after the fact —
without it, only position state survives a restart and execution issues
(slippage spikes, partial fills) leave no forensic trace.

Writes are append-only and flushed before the file handle closes, so a crash
mid-write at worst loses the line currently being written, never a previous
record.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = Path("state")


class TradeJournal:
    """Per-coin append-only JSONL journal."""

    def __init__(self, coin: str, root: Path | str | None = None) -> None:
        base = Path(root) if root is not None else DEFAULT_STATE_DIR
        self.path = base / f"fills_{coin}.jsonl"

    def record(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": _utcnow_iso(), **event}
        with self.path.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
