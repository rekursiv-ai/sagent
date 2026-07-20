"""Tests for ``bin.check_wheel``: wheel surface and recipe asset validation."""

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
    "sagent/lib/web/fetch/__init__.py": "",
    "sagent/lib/web/fetch/fetch.py": "",
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


_PYPROJECT = """\
[tool.hatch.build.targets.wheel]
packages = ["sagent"]
exclude = ["**/*_test.py", "**/conftest.py"]
"""


def _write_wheel(path: Path, files: dict[str, str]) -> None:
    """Write a minimal zip file with the requested files."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_source_tree(tmp_path: Path) -> None:
    """Materialize ``pyproject.toml`` and the canonical source ``.py`` tree.

    ``check_wheel`` derives the modules it expects from the build config and
    the on-disk package, so the check needs a source tree to scan. The tree
    mirrors the ``.py`` entries in ``_BASE_FILES`` plus test/conftest files
    the wheel exclude globs must drop.
    """
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    modules = [name for name in _BASE_FILES if name.endswith(".py")]
    # Excluded by the wheel globs: present in source, must not be required.
    modules += ["sagent/lib/web/fetch/fetch_test.py", "sagent/lib/web/conftest.py"]
    for name in modules:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _write_sagent_wheel(tmp_path: Path, files: dict[str, str]) -> None:
    """Write one candidate Sagent wheel plus the source tree it validates."""
    _write_source_tree(tmp_path)
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
            _ = check_wheel.main()

    def test_rejects_recipe_referenced_missing_tool_asset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject a missing asset referenced by tool_descriptions."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        del files["sagent/assets/default/tools_websearch.md"]
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"default/tools_websearch\.md"):
            _ = check_wheel.main()

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
            _ = check_wheel.main()

    def test_rejects_malformed_recipe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject invalid YAML before asset validation."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = "tool_descriptions: ["
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"invalid sagent/assets/sagent\.yaml"):
            _ = check_wheel.main()

    def test_rejects_non_string_asset_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject recipe asset entries that are not strings."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = "tool_descriptions:\n  Bash: 1\n"
        _write_sagent_wheel(tmp_path, files)

        with pytest.raises(SystemExit, match=r"tool_descriptions\.Bash"):
            _ = check_wheel.main()

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
            _ = check_wheel.main()


class TestCheckWheelErrors:
    """Error paths for missing wheels, modules, and recipe shape."""

    def test_rejects_no_wheels(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dist").mkdir()
        with pytest.raises(SystemExit, match=r"uv build produced no Sagent wheel"):
            _ = check_wheel.main()

    def test_rejects_missing_entry_points_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        del files["sagent-0.1.0.dist-info/entry_points.txt"]
        _write_sagent_wheel(tmp_path, files)
        with pytest.raises(SystemExit, match=r"entry_points\.txt"):
            _ = check_wheel.main()

    def test_rejects_source_module_missing_from_wheel(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reject a wheel that dropped a ``.py`` present in the source tree."""
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        del files["sagent/lib/web/fetch/fetch.py"]
        _write_sagent_wheel(tmp_path, files)
        with pytest.raises(
            SystemExit,
            match=r"missing source modules: sagent/lib/web/fetch/fetch\.py",
        ):
            _ = check_wheel.main()

    def test_rejects_missing_console_script(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent-0.1.0.dist-info/entry_points.txt"] = (
            "[console_scripts]\nsagent = sagent.bin.cli:main\n"
        )
        _write_sagent_wheel(tmp_path, files)
        with pytest.raises(SystemExit, match=r"sagent-slack"):
            _ = check_wheel.main()

    def test_rejects_recipe_not_a_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = "- one\n- two\n"
        _write_sagent_wheel(tmp_path, files)
        with pytest.raises(SystemExit, match=r"expected mapping"):
            _ = check_wheel.main()

    def test_skips_non_mapping_recipe_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = (
            "system_prompt: not-a-mapping\n"
            "compactor:\n  full: default/compactor.md\n"
            "  partial: default/compactor_partial.md\n"
            "tool_descriptions:\n  Bash: default/tools_bash.md\n"
            "  WebSearch: default/tools_websearch.md\n"
        )
        _write_sagent_wheel(tmp_path, files)
        # Skipping the non-mapping section means no missing-asset error.
        assert check_wheel.main() == 0

    def test_rejects_absolute_asset_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        files = dict(_BASE_FILES | _RECIPE_FILES)
        files["sagent/assets/sagent.yaml"] = (
            "tool_descriptions:\n  Bash: /etc/passwd\n"
            "  WebSearch: default/tools_websearch.md\n"
            "system_prompt:\n  base: default/prompt.md\n"
            "  env: default/prompt_env.md\n"
            "compactor:\n  full: default/compactor.md\n"
            "  partial: default/compactor_partial.md\n"
        )
        _write_sagent_wheel(tmp_path, files)
        with pytest.raises(SystemExit, match=r"must stay inside sagent/assets"):
            _ = check_wheel.main()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
