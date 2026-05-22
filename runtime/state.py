"""Per-strategy JSON state store.

Each strategy owns one file at `state/{strategy}_{coin}.json` containing the
in-memory bookkeeping it needs to recover from a restart: position flags,
entry price, cooldown counters, regime streak, last bar processed, etc.

Writes are atomic (write-temp + rename) so a crash mid-save can't leave a
half-written file.

Every save stamps a `_schema_version` so the loader can refuse to read a
file produced by a *newer* code version (which would silently misread
renamed or repurposed keys). Files written before this scheme existed are
treated as legacy v1 — the strategy-side reader defaults missing keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STATE_DIR = Path("state")
_SCHEMA_KEY = "_schema_version"


class StateSchemaError(RuntimeError):
    """Raised when a persisted state file declares a newer schema than this
    code version knows how to read. The operator must roll forward or delete
    the file — silently dropping fields would corrupt the next save."""


class StateStore:
    def __init__(
        self,
        strategy_name: str,
        coin: str,
        root: Path | str | None = None,
        schema_version: int = 1,
    ) -> None:
        # Resolve root at call time (not at class definition) so tests can
        # monkey-patch DEFAULT_STATE_DIR for isolation.
        base = Path(root) if root is not None else DEFAULT_STATE_DIR
        self.path = base / f"{strategy_name}_{coin}.json"
        self.schema_version = int(schema_version)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        stored = int(data.pop(_SCHEMA_KEY, 1))
        if stored > self.schema_version:
            raise StateSchemaError(
                f"{self.path} was written by schema v{stored} but this binary "
                f"only understands up to v{self.schema_version}. Roll forward "
                f"the binary or remove the file before restarting."
            )
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(data)
        payload[_SCHEMA_KEY] = self.schema_version
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
