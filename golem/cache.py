"""Shared cache helpers for reusable preprocessing artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def shared_cache_root() -> Path:
    """Return the shared cache root for reusable Golem artifacts."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    path = base / "golem"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(name: str) -> Path:
    """Return a named subdirectory under the shared cache root."""
    path = shared_cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
