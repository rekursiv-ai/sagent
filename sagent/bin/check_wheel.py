#!/usr/bin/env python3
"""Validate that `uv build` produced a runnable Sagent wheel.

The static required-entry list covers import and entry-point structure. Prompt
assets are validated from ``sagent/assets/sagent.yaml`` so recipe changes cannot
drift from packaged files.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import cast

import re
import zipfile

import yaml


_RECIPE_PATH = "sagent/assets/sagent.yaml"
_ASSET_PREFIX = "sagent/assets/"
_RE_INCLUDE = re.compile(r"\{\{include:\s*(.+?)\}\}")
_REQUIRED_ENTRIES = (
    "sagent/__init__.py",
    "sagent/bin/cli.py",
    "sagent/bin/slack.py",
    _RECIPE_PATH,
    "sagent/assets/slack/default.md",
    "sagent/lib/web/fetch.py",
    "sagent/lib/web/search.py",
)
_REQUIRED_ENTRY_POINTS = (
    "sagent = sagent.bin.cli:main",
    "sagent-slack = sagent.bin.slack:main",
)
_RECIPE_ASSET_SECTIONS = ("system_prompt", "compactor", "tool_descriptions")


def main() -> int:
    """Validate the freshest wheel under ``dist/`` and return a shell exit code.

    Returns:
      exit_code: ``0`` on success; this function raises ``SystemExit`` on
        failure rather than returning non-zero.

    Raises:
      SystemExit: If no wheel is found, required entries are missing, or
        the recipe references missing/invalid assets.

    """
    wheels = sorted(Path("dist").glob("sagent-*.whl"))
    if not wheels:
        raise SystemExit("uv build produced no Sagent wheel")
    with zipfile.ZipFile(wheels[-1]) as archive:
        names = archive.namelist()
        missing: list[str] = [name for name in _REQUIRED_ENTRIES if name not in names]
        entry_points_name = _entry_points_name(names)
        if entry_points_name is None:
            missing.append("*.dist-info/entry_points.txt")
        if missing:
            raise SystemExit(
                "wheel is missing required wheel entries: " + ", ".join(missing)
            )
        assert entry_points_name is not None
        entry_points = archive.read(entry_points_name).decode()
        _validate_recipe_assets(archive, frozenset(names))
    if not any(name.startswith("sagent/") and name.endswith(".py") for name in names):
        raise SystemExit("wheel contains no sagent/*.py modules")
    missing_entry_points = [
        entry for entry in _REQUIRED_ENTRY_POINTS if entry not in entry_points
    ]
    if missing_entry_points:
        raise SystemExit(
            "wheel is missing required console scripts: "
            + ", ".join(missing_entry_points)
        )
    return 0


def _entry_points_name(names: list[str]) -> str | None:
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
    for section_name in _RECIPE_ASSET_SECTIONS:
        section = recipe.get(section_name, {})
        if not isinstance(section, dict):
            continue
        values = cast(dict[object, object], section)
        for key, value in values.items():
            if not isinstance(value, str):
                raise SystemExit(
                    f"invalid {_RECIPE_PATH}: {section_name}.{key} must be a string"
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
    wheel_path = _ASSET_PREFIX + asset
    if wheel_path not in names:
        raise SystemExit("wheel is missing recipe asset: " + " -> ".join(chain))
    text = archive.read(wheel_path).decode()
    for match in _RE_INCLUDE.finditer(text):
        included = _normalize_asset(match.group(1).strip(), " -> ".join(chain))
        _validate_asset(archive, names, included, chain)


def _normalize_asset(path: str, context: str) -> str:
    """Normalize a recipe asset path and reject escapes from assets/."""
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise SystemExit(
            f"invalid {_RECIPE_PATH}: {context} must stay inside sagent/assets"
        )
    return parsed.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
