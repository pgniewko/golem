"""Tests for golem._version."""

import golem._version as version_mod


class TestVersionModule:
    """Git-aware version/source metadata helpers."""

    def test_get_version_from_git_parses_describe_output(self, monkeypatch):
        def fake_run_git(args):
            if args == ["describe", "--tags", "--long"]:
                return "v0.1.0-3-gabc123def456"
            raise AssertionError(f"Unexpected git args: {args}")

        monkeypatch.setattr(version_mod, "_run_git", fake_run_git)
        assert version_mod._get_version_from_git() == "0.1.0.dev3+abc123def456"

    def test_get_version_from_git_returns_exact_tag_when_distance_is_zero(self, monkeypatch):
        def fake_run_git(args):
            if args == ["describe", "--tags", "--long"]:
                return "v0.1.0-0-gabc123def456"
            raise AssertionError(f"Unexpected git args: {args}")

        monkeypatch.setattr(version_mod, "_run_git", fake_run_git)
        assert version_mod._get_version_from_git() == "0.1.0"

    def test_get_version_from_git_normalizes_prerelease_tags(self, monkeypatch):
        def fake_run_git(args):
            if args == ["describe", "--tags", "--long"]:
                return "v0.1.0-beta.2-1-gabc123def456"
            raise AssertionError(f"Unexpected git args: {args}")

        monkeypatch.setattr(version_mod, "_run_git", fake_run_git)
        assert version_mod._get_version_from_git() == "0.1.0b2.dev1+abc123def456"

    def test_get_version_returns_unknown_when_git_and_metadata_are_unavailable(self, monkeypatch):
        monkeypatch.setattr(version_mod, "_get_version_from_git", lambda: (_ for _ in ()).throw(RuntimeError("no git")))
        monkeypatch.setattr(version_mod, "_get_version_from_metadata", lambda: (_ for _ in ()).throw(RuntimeError("no metadata")))
        assert version_mod._get_version() == "0+unknown"

    def test_get_source_info_returns_empty_when_git_unavailable(self, monkeypatch):
        monkeypatch.setattr(version_mod, "_run_git", lambda args: (_ for _ in ()).throw(RuntimeError("no git")))
        assert version_mod.get_source_info() == {}
