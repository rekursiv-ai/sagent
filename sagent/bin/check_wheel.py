#!/usr/bin/env python3
"""Validate that `uv build` produced a runnable Sagent wheel."""

from __future__ import annotations

from pathlib import Path

import zipfile


_REQUIRED_ENTRIES = (
    "sagent/__init__.py",
    "sagent/bin/cli.py",
    "sagent/bin/slack.py",
    "sagent/assets/sagent.yaml",
    "sagent/assets/default/prompt.md",
    "sagent/assets/default/tools_bash.md",
    "sagent/assets/slack/default.md",
    "sagent/lib/web/fetch.py",
    "sagent/lib/web/search.py",
)
_REQUIRED_ENTRY_POINTS = (
    "sagent = sagent.bin.cli:main",
    "sagent-slack = sagent.bin.slack:main",
)


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
