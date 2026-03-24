"""Derive golem version and source metadata from Git or installed metadata."""

from __future__ import annotations

import os
import re
import subprocess
from importlib.metadata import version
from typing import Any, Dict


def _repo_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_git(args: list[str]) -> str:
    """Run a git command in the repository root and return stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=_repo_dir(),
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed"
        raise RuntimeError(msg)
    return result.stdout.strip()


def _normalize_prerelease(ver: str) -> str:
    """Convert common pre-release tag formats to PEP 440."""
    ver = re.sub(r"[-.]?alpha[.-]?", "a", ver)
    ver = re.sub(r"[-.]?beta[.-]?", "b", ver)
    ver = re.sub(r"[-.]?rc[.-]?", "rc", ver)
    return ver


def _get_version_from_git() -> str:
    """Return a git-derived version string for a source checkout."""
    try:
        desc = _run_git(["describe", "--tags", "--long", "--dirty"])
    except Exception:
        sha = _run_git(["rev-parse", "--short=12", "HEAD"])
        dirty = bool(_run_git(["status", "--porcelain", "--untracked-files=normal"]))
        resolved = f"0+g{sha}"
        if dirty:
            resolved = f"{resolved}.dirty"
        return resolved

    dirty = desc.endswith("-dirty")
    if dirty:
        desc = desc[:-6]

    desc = desc.lstrip("v")
    parts = desc.rsplit("-", 2)
    if len(parts) != 3 or not parts[2].startswith("g"):
        resolved = f"0+g{desc}"
    else:
        ver, distance, sha = parts[0], parts[1], parts[2][1:]
        ver = _normalize_prerelease(ver)
        if int(distance) == 0:
            resolved = ver
        else:
            resolved = f"{ver}.dev{distance}+{sha}"

    if dirty:
        if "+" in resolved:
            return f"{resolved}.dirty"
        return f"{resolved}+dirty"
    return resolved


def _get_version_from_metadata() -> str:
    """Return the installed distribution version."""
    return version("golem")


def _get_version() -> str:
    """Return a version string, preferring git metadata in source checkouts."""
    try:
        return _get_version_from_git()
    except Exception:
        try:
            return _get_version_from_metadata()
        except Exception:
            return "0+unknown"


def get_source_info() -> Dict[str, Any]:
    """Return exact git source metadata for the current checkout when available."""
    try:
        status = _run_git(["status", "--porcelain", "--branch", "--untracked-files=normal"])
        return {
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _run_git(["rev-parse", "HEAD"]),
            "short_commit": _run_git(["rev-parse", "--short=12", "HEAD"]),
            "describe": _run_git(["describe", "--tags", "--long", "--dirty", "--always"]),
            "dirty": any(line and not line.startswith("##") for line in status.splitlines()),
        }
    except Exception:
        return {}


__version__: str = _get_version()
