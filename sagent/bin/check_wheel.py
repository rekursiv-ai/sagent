#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Validate the Sagent wheel.

The set of modules that must ship is derived from the source tree and the
build config in ``pyproject.toml`` -- no hand-maintained file list. Prompt
assets are validated from ``sagent/assets/sagent.yaml`` so recipe changes
cannot drift from packaged files.
'''
# fmt: on

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Final, cast, override

import configparser
import re
import tomllib
import zipfile

import yaml

from sagent.lib.files.glob import glob


_RECIPE_PATH: Final = "sagent/assets/sagent.yaml"


def main() -> int:
    """Validate the one wheel under ``dist/`` and return a shell exit code.

    Returns:
      exit_code: ``0`` on success; this function raises ``SystemExit`` on
        failure rather than returning non-zero.

    Raises:
      SystemExit: If ``dist/`` holds no wheel or several, source modules or
        console scripts are missing from the wheel, or the recipe references
        missing/invalid assets.

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
        _validate_recipe_assets(archive, names)
    required_entry_points = {
        "sagent": "sagent.bin.cli:main",
        "sagent-slack": "sagent.bin.slack:main",
    }
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
    config = tomllib.loads(Path("pyproject.toml").read_bytes().decode())
    wheel = (
        config.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    packages = [str(v) for v in wheel.get("packages", [])]
    excludes = [str(v) for v in wheel.get("exclude", [])]
    expected: set[str] = set()
    for package in packages:
        for path in glob(Path(package), "**/*.py"):
            posix = path.as_posix()
            if not any(fnmatch(posix, pat) for pat in excludes):
                expected.add(posix)
    return frozenset(expected)


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


def _validate_recipe_assets(archive: zipfile.ZipFile, names: frozenset[str]) -> None:
    """Validate recipe-selected assets and their include graph."""
    recipe = _load_recipe(archive)
    for asset in _recipe_assets(recipe):
        _validate_asset(archive, names, asset, ())


def _load_recipe(archive: zipfile.ZipFile) -> dict[str, object]:
    """Load the wheel recipe as a strict mapping."""
    try:
        loaded = yaml.safe_load(archive.read(_RECIPE_PATH).decode())
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid {_RECIPE_PATH}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"invalid {_RECIPE_PATH}: expected mapping")
    return cast(dict[str, object], loaded)


def _recipe_assets(recipe: dict[str, object]) -> list[str]:
    """Return asset paths referenced by recipe sections that package prompts."""
    assets: list[str] = []
    for section_name in ("system_prompt", "compactor", "tool_descriptions"):
        section = recipe.get(section_name)
        if not isinstance(section, dict):
            continue
        values = cast(dict[object, object], section)
        for key, value in values.items():
            if not isinstance(value, str):
                raise SystemExit(
                    f"invalid {_RECIPE_PATH}: {section_name}.{key} must be a string",
                )
            assets.append(_normalize_asset(value, f"{section_name}.{key}"))
    return assets


def _validate_asset(
    archive: zipfile.ZipFile,
    names: frozenset[str],
    asset: str,
    parents: tuple[str, ...],
) -> None:
    """Validate one asset and recurse through ``{{include: ...}}`` edges."""
    chain = (*parents, asset)
    if asset in parents:
        raise SystemExit("wheel asset include cycle: " + " -> ".join(chain))
    wheel_path = "sagent/assets/" + asset
    if wheel_path not in names:
        raise SystemExit("wheel is missing recipe asset: " + " -> ".join(chain))
    text = archive.read(wheel_path).decode()
    for match in re.finditer(r"\{\{include:\s*(.+?)\}\}", text):
        included = _normalize_asset(match.group(1).strip(), " -> ".join(chain))
        _validate_asset(archive, names, included, chain)


def _normalize_asset(path: str, context: str) -> str:
    """Normalize a recipe asset path and reject escapes from assets/."""
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise SystemExit(
            f"invalid {_RECIPE_PATH}: {context} must stay inside sagent/assets",
        )
    return parsed.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
