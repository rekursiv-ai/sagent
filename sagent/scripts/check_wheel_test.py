"""Tests for ``scripts.check_wheel``: the wheel's modules and console scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zipfile

import pytest

from sagent.scripts import check_wheel


if TYPE_CHECKING:
    from pathlib import Path


_ENTRY_POINTS = """[console_scripts]
sagent = sagent.tui.main:main
"""
_BASE_FILES = {
    "sagent/__init__.py": "",
    "sagent/tui/main.py": "",
    "sagent/tools/bash.py": "",
    "sagent/tools/read.py": "",
    "sagent-0.1.0.dist-info/entry_points.txt": _ENTRY_POINTS,
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


# ``check_wheel`` derives the modules it expects from the build config and the on-disk
# package, so the check needs a source tree to scan. The tree mirrors the ``.py``
# entries in ``_BASE_FILES`` plus test/conftest files the wheel exclude globs must drop.
def _write_source_tree(tmp_path: Path) -> None:
    """Materialize ``pyproject.toml`` and the canonical source ``.py`` tree."""
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    modules = [name for name in _BASE_FILES if name.endswith(".py")]
    # Excluded by the wheel globs: present in source, must not be required.
    modules += ["sagent/tools/bash_test.py", "sagent/tools/conftest.py"]
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


def _main_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: dict[str, str],
) -> str:
    """Run ``main`` on a wheel of ``files``; return its ``SystemExit`` message."""
    monkeypatch.chdir(tmp_path)
    _write_sagent_wheel(tmp_path, files)
    with pytest.raises(SystemExit) as raised:
        _ = check_wheel.main()
    return str(raised.value)


class TestCheckWheel:
    """A wheel holding every module and console script passes."""

    def test_accepts_required_public_wheel_surface(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_sagent_wheel(tmp_path, _BASE_FILES)

        assert check_wheel.main() == 0

    def test_a_gitignored_module_is_not_required(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Hatch leaves ``.gitignore``d files out of the wheel, so so does the check."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git").mkdir()
        (tmp_path / ".gitignore").write_text("generated/\n")
        _write_sagent_wheel(tmp_path, _BASE_FILES)
        (tmp_path / "sagent" / "generated").mkdir()
        (tmp_path / "sagent" / "generated" / "x.py").write_text("")

        assert check_wheel.main() == 0


class TestCheckWheelErrors:
    """Each failure says exactly what is wrong, joined as a reader expects."""

    def test_no_wheel(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "dist").mkdir()
        with pytest.raises(SystemExit) as raised:
            _ = check_wheel.main()
        assert str(raised.value) == "uv build produced no Sagent wheel"

    def test_rejects_two_wheels_rather_than_guessing_the_fresh_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Text order put ``0.9.0`` after ``0.10.0``, so the stale wheel won."""
        monkeypatch.chdir(tmp_path)
        _write_sagent_wheel(tmp_path, _BASE_FILES)
        _write_wheel(tmp_path / "dist" / "sagent-0.9.0-py3-none-any.whl", {})
        with pytest.raises(SystemExit, match="exactly one"):
            _ = check_wheel.main()

    def test_missing_modules_are_comma_joined(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        files = dict(_BASE_FILES)
        del files["sagent/tools/bash.py"], files["sagent/tools/read.py"]
        assert _main_error(monkeypatch, tmp_path, files) == (
            "wheel is missing source modules: sagent/tools/bash.py, sagent/tools/read.py"
        )

    def test_missing_entry_points(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        files = dict(_BASE_FILES)
        del files["sagent-0.1.0.dist-info/entry_points.txt"]
        assert _main_error(monkeypatch, tmp_path, files) == (
            "wheel is missing *.dist-info/entry_points.txt"
        )

    @pytest.mark.parametrize(
        "entry_points",
        [
            "[console_scripts]\n",
            "[console_scripts]\n# sagent = sagent.tui.main:main\n",
            "[gui_scripts]\nsagent = sagent.tui.main:main\n",
            "[console_scripts]\nsagent = sagent.bin.cli:main\n",
            "",
        ],
        ids=["empty", "comment", "other-section", "wrong-target", "no-section"],
    )
    def test_missing_console_script(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        entry_points: str,
    ) -> None:
        files = _BASE_FILES | {"sagent-0.1.0.dist-info/entry_points.txt": entry_points}
        assert _main_error(monkeypatch, tmp_path, files) == (
            "wheel is missing required console scripts: sagent = sagent.tui.main:main"
        )


@pytest.mark.parametrize(
    "pyproject",
    [
        "",
        "[tool]\n",
        "[tool.hatch]\n",
        "[tool.hatch.build]\n",
        "[tool.hatch.build.targets]\n",
        "[tool.hatch.build.targets.wheel]\n",
    ],
    ids=["empty", "tool", "hatch", "build", "targets", "wheel"],
)
def test_a_build_config_without_packages_expects_no_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pyproject: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(pyproject)
    assert check_wheel.expected_modules() == frozenset()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
