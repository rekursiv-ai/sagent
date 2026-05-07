"""Tests for sagent.bin.check_wheel."""

from __future__ import annotations

from pathlib import Path

import zipfile

import pytest

from sagent.bin import check_wheel


_ENTRY_POINTS = """[console_scripts]
sagent = sagent.bin.cli:main
sagent-slack = sagent.bin.slack:main
"""
_RECIPE = """
system_prompt:
  base: default/prompt.md
  env: default/prompt_env.md
compactor:
  full: default/compactor.md
  partial: default/compactor_partial.md
tool_descriptions:
  Bash: default/tools_bash.md
  WebSearch: default/tools_websearch.md
"""
_BASE_FILES = {
    "sagent/__init__.py": "",
    "sagent/bin/cli.py": "",
    "sagent/bin/slack.py": "",
    "sagent/lib/web/fetch.py": "",
    "sagent/lib/web/search.py": "",
    "sagent/assets/slack/default.md": "slack",
    "sagent-0.1.0.dist-info/entry_points.txt": _ENTRY_POINTS,
}
_RECIPE_FILES = {
    "sagent/assets/sagent.yaml": _RECIPE,
    "sagent/assets/default/prompt.md": "prompt",
    "sagent/assets/default/prompt_env.md": "env",
    "sagent/assets/default/compactor.md": "compactor",
    "sagent/assets/default/compactor_partial.md": "partial",
    "sagent/assets/default/tools_bash.md": "bash",
    "sagent/assets/default/tools_websearch.md": "websearch",
}


def _write_wheel(path: Path, files: dict[str, str]) -> None:
    """Write a minimal zip file with the requested files."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_sagent_wheel(tmp_path: Path, files: dict[str, str]) -> None:
    """Write one candidate Sagent wheel under a temporary dist directory."""
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(dist / "sagent-0.1.0-py3-none-any.whl", files)


class TestCheckWheel:
    """Wheel validation tests for import surface and recipe assets."""

    def test_accepts_required_public_wheel_surface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Accept a wheel with entry points, modules, and recipe assets."""
        monkeypatch.chdir(tmp_path)
        _write_sagent_wheel(tmp_path, _BASE_FILES | _RECIPE_FILES)

        assert check_wheel.main() == 0

    def test_rejects_missing_prompt_assets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject a wheel missing an asset referenced by system_prompt."""
        monkeypatch.chdir(tmp_path)
        _write_sagent_wheel(
            tmp_path,
            _BASE_FILES | {"sagent/assets/sagent.yaml": _RECIPE},
        )

        with pytest.raises(SystemExit, match=r"default/prompt\.md"):
            check_wheel.main()

    def test_rejects_recipe_referenced_missing_tool_asset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject a missing asset referenced by tool_descriptions."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        del files["sagent/assets/default/tools_websearch.md"]
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"default/tools_websearch\.md"):
            check_wheel.main()

    def test_rejects_missing_included_asset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject missing include targets and report the include chain."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/default/tools_websearch.md"] = "{{include: missing.md}}"
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(
            SystemExit,
            match=r"default/tools_websearch\.md -> missing\.md",
        ):
            check_wheel.main()

    def test_rejects_malformed_recipe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject invalid YAML before asset validation."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = "tool_descriptions: ["
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"invalid sagent/assets/sagent\.yaml"):
            check_wheel.main()

    def test_rejects_non_string_asset_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject recipe asset entries that are not strings."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = "tool_descriptions:\n  Bash: 1\n"
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"tool_descriptions\.Bash"):
            check_wheel.main()

    def test_rejects_include_cycle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject recursive include graphs with a readable cycle path."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/default/tools_bash.md"] = "{{include: b.md}}"
        files["sagent/assets/b.md"] = "{{include: default/tools_bash.md}}"
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(
            SystemExit,
            match=r"default/tools_bash\.md -> b\.md -> default/tools_bash\.md",
        ):
            check_wheel.main()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
