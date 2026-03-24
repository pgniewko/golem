"""Golem — Descriptor pretraining for Graph Transformers on molecular descriptors."""

from importlib.metadata import PackageNotFoundError, version

_FALLBACK_VERSION = "0.1.0"


def _resolve_version() -> str:
    """Prefer installed package metadata when available."""
    try:
        return version("golem")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = _resolve_version()
