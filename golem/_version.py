"""Derive a PEP 440 version string from Git or installed metadata."""

import os

from . import _version_utils


def _normalize_prerelease(ver: str) -> str:
    """Convert common pre-release tag formats to PEP 440."""
    return _version_utils._normalize_prerelease(ver)


def _get_version_from_git() -> str:
    """Return the git-derived version for a source checkout."""
    return _version_utils._get_version_from_git()


def _get_version_from_metadata() -> str:
    """Return the installed distribution version."""
    return _version_utils._get_version_from_metadata()


def _get_version() -> str:
    """Return a PEP 440-compliant version string."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        try:
            return _get_version_from_git()
        except Exception:
            pass

    try:
        return _get_version_from_metadata()
    except Exception:
        return "0+unknown"


__version__: str = _get_version()
