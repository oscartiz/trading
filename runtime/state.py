"""Per-strategy JSON state store.

Each strategy owns one file at `state/{strategy}_{coin}.json` containing the
in-memory bookkeeping it needs to recover from a restart: position flags,
entry price, cooldown counters, regime streak, last bar processed, etc.

Writes are atomic (write-temp + rename) so a crash mid-save can't leave a
half-written file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = Path("state")


class StateStore:
    def __init__(self, strategy_name: str, coin: str, root: Path | str | None = None) -> None:
        # Resolve root at call time (not at class definition) so tests can
        # monkey-patch DEFAULT_STATE_DIR for isolation.
        base = Path(root) if root is not None else DEFAULT_STATE_DIR
        self.path = base / f"{strategy_name}_{coin}.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open() as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
