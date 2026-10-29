#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Validate the Sagent wheel.

The set of modules that must ship is derived from the source tree and the
build config in ``pyproject.toml`` -- no hand-maintained file list.
'''
# fmt: on

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import TypedDict, override

import configparser
import tomllib
import zipfile

from treekle import from_plain

from sagent.lib.files.glob import glob


def main() -> int:
    """Validate the one wheel under ``dist/`` and return a shell exit code.

    Returns:
      exit_code: ``0`` on success; this function raises ``SystemExit`` on
        failure rather than returning non-zero.

    Raises:
      SystemExit: If ``dist/`` holds no wheel or several, or source modules or
        console scripts are missing from the wheel.

    """
    # Exactly one: two builds side by side cannot be told apart by name order
    # (``0.9.0`` sorts after ``0.10.0``), and the build writes into a fresh dist/.
    wheels = sorted(Path("dist").glob("sagent-*.whl"))
    if not wheels:
        raise SystemExit("uv build produced no Sagent wheel")
    if len(wheels) > 1:
        raise SystemExit(
            "dist/ must hold exactly one Sagent wheel; found "
            + ", ".join(w.name for w in wheels),
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = frozenset(archive.namelist())
        missing = sorted(expected_modules() - names)
        if missing:
            raise SystemExit("wheel is missing source modules: " + ", ".join(missing))
        entry_points_name = _entry_points_name(names)
        if entry_points_name is None:
            raise SystemExit("wheel is missing *.dist-info/entry_points.txt")
        entry_points = archive.read(entry_points_name).decode()
    required_entry_points = {"sagent": "sagent.tui.main:main"}
    scripts = _console_scripts(entry_points)
    missing_entry_points = [
        f"{name} = {target}"
        for name, target in required_entry_points.items()
        if scripts.get(name) != target
    ]
    if missing_entry_points:
        raise SystemExit(
            "wheel is missing required console scripts: "
            + ", ".join(missing_entry_points),
        )
    return 0


# Derived from ``[tool.hatch.build.targets.wheel]`` in ``pyproject.toml``: every ``.py``
# under each configured package that git tracks (hatch honors ``.gitignore``),
# minus the wheel ``exclude`` globs. Reading the build
# config here keeps the check from drifting when modules are added, renamed, or
# restructured.
def expected_modules() -> frozenset[str]:
    """Return the package ``.py`` paths, relative to the cwd, the wheel must hold.

    Returns:
      modules: Every ``.py`` under each configured package that git would track,
        minus the wheel ``exclude`` globs.

    """
    # TOML is UTF-8 by spec; bytes skip the locale.
    table: object = tomllib.loads(Path("pyproject.toml").read_bytes().decode())
    for key in ("tool", "hatch", "build", "targets", "wheel"):
        table = from_plain(table, dict[str, object], default={}).get(key)
    wheel = from_plain(table, _WheelTarget, default=_WheelTarget())
    packages: list[str] = wheel.get("packages", [])
    excludes: list[str] = wheel.get("exclude", [])
    expected: set[str] = set()
    for package in packages:
        for path in glob(Path(package), "**/*.py"):
            posix = path.as_posix()
            if not any(fnmatch(posix, pat) for pat in excludes):
                expected.add(posix)
    return frozenset(expected)


class _WheelTarget(TypedDict, total=False):
    """``[tool.hatch.build.targets.wheel]``, the keys this check reads."""

    packages: list[str]
    exclude: list[str]


def _console_scripts(entry_points: str) -> dict[str, str]:
    """Return ``[console_scripts]`` entries of an ``entry_points.txt``."""
    parser = _CaseSensitiveParser(delimiters=("=",), interpolation=None)
    parser.read_string(entry_points)
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


class _CaseSensitiveParser(configparser.ConfigParser):
    """``ConfigParser`` that keeps script names' case, as the wheel does."""

    @override
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _entry_points_name(names: frozenset[str]) -> str | None:
    """Return the wheel entry-points path, if present."""
    for name in names:
        if name.endswith(".dist-info/entry_points.txt"):
            return name
    return None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
