"""Tests for golem._version."""

import golem._version as version_mod


class TestVersionModule:
    """Git-aware version/source metadata helpers."""

    def test_get_version_from_git_parses_describe_output(self, monkeypatch):
        monkeypatch.setattr(
            version_mod,
            "_run_git",
            lambda args: "v0.1.0-3-gabc123def456-dirty" if args == ["describe", "--tags", "--long", "--dirty"] else "",
        )
        assert version_mod._get_version_from_git() == "0.1.0.dev3+abc123def456.dirty"

    def test_get_version_from_git_falls_back_to_commit_hash(self, monkeypatch):
        def fake_run_git(args):
            if args == ["describe", "--tags", "--long", "--dirty"]:
                raise RuntimeError("no tags")
            if args == ["rev-parse", "--short=12", "HEAD"]:
                return "abc123def456"
            if args == ["status", "--porcelain", "--untracked-files=normal"]:
                return " M golem/pretrain.py"
            raise AssertionError(f"Unexpected git args: {args}")

        monkeypatch.setattr(version_mod, "_run_git", fake_run_git)
        assert version_mod._get_version_from_git() == "0+gabc123def456.dirty"

    def test_get_source_info_returns_empty_when_git_unavailable(self, monkeypatch):
        monkeypatch.setattr(version_mod, "_run_git", lambda args: (_ for _ in ()).throw(RuntimeError("no git")))
        assert version_mod.get_source_info() == {}
