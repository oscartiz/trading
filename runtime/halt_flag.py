"""File-based kill switch.

A `state/halt.flag` file lets the operator pause new entries without killing
the process. The strategy checks for it once per tick — if present, it
refuses to enter new positions and emits one warning (transition log only,
not a per-tick spam).

Existing positions are unaffected: the halt only gates entries, exits still
fire on their normal triggers. Same one-way semantics as the drawdown halt.

Usage::

    touch state/halt.flag        # halt new entries
    rm    state/halt.flag        # resume
"""
from __future__ import annotations

from pathlib import Path

# Imported via the module rather than re-bound at import time so the test
# fixture's monkey-patch of state.DEFAULT_STATE_DIR is honoured here too.
from . import state as _state_mod

_DEFAULT_FLAG = "halt.flag"


def halt_flag_path(root: Path | str | None = None, name: str = _DEFAULT_FLAG) -> Path:
    base = Path(root) if root is not None else _state_mod.DEFAULT_STATE_DIR
    return base / name


def is_halted(root: Path | str | None = None, name: str = _DEFAULT_FLAG) -> bool:
    return halt_flag_path(root, name).exists()


def set_halt(root: Path | str | None = None, name: str = _DEFAULT_FLAG) -> Path:
    path = halt_flag_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def clear_halt(root: Path | str | None = None, name: str = _DEFAULT_FLAG) -> None:
    path = halt_flag_path(root, name)
    if path.exists():
        path.unlink()
