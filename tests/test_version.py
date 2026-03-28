import importlib.metadata

from click.testing import CliRunner

import golem
import golem._version as version_mod
from golem.cli import main


def test_normalize_prerelease_tags():
    assert version_mod._normalize_prerelease("1.2.3-alpha.2") == "1.2.3a2"
    assert version_mod._normalize_prerelease("1.2.3-beta.4") == "1.2.3b4"
    assert version_mod._normalize_prerelease("1.2.3-rc.5") == "1.2.3rc5"


def test_version_prefers_git_in_source_checkout(monkeypatch):
    """Source checkouts should use git instead of potentially stale metadata."""

    def fake_git_version():
        return "1.2.3.dev4+abc1234"

    def fail_metadata(_name):
        raise AssertionError("metadata lookup should not be used in a git checkout")

    monkeypatch.setattr(version_mod, "_get_version_from_git", fake_git_version)
    monkeypatch.setattr(importlib.metadata, "version", fail_metadata)
    monkeypatch.setattr(version_mod.os.path, "isdir", lambda path: path.endswith(".git"))

    assert version_mod._get_version() == "1.2.3.dev4+abc1234"


def test_version_falls_back_to_metadata_outside_git(monkeypatch):
    """Installed packages without git metadata should use importlib.metadata."""
    monkeypatch.setattr(version_mod.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    assert version_mod._get_version() == "9.9.9"


def test_version_falls_back_to_unknown_without_git_or_metadata(monkeypatch):
    monkeypatch.setattr(version_mod.os.path, "isdir", lambda path: False)

    def fail_metadata(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", fail_metadata)

    assert version_mod._get_version() == "0+unknown"


def test_cli_reports_runtime_version():
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert golem.__version__ in result.output
