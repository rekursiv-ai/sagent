#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Convert old sagent session logs into journals today's sagent resumes.

Each old ``<id>/session.jsonl`` under the paths given is rewritten in place
as today's journal, holding what its model last saw. A log of today's records
and a log with nothing to resume are skipped. A resume by id converts its one
session itself.

Examples:
  sh sagent/scripts/convert_sessions.py --dry-run
  sh sagent/scripts/convert_sessions.py ~/.local/share/rekursiv-ai/sagent/projects/_home_me_repo

'''
# fmt: on

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse

from sagent.lib.userdirs import data_dir
from sagent.session.legacy_journal import convert, is_legacy, upgrade


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Convert every old log under the paths given; print what was done.

    Args:
      argv: The command line, less the program; ``sys.argv``'s when ``None``.

    Returns:
      code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    root = data_dir() / "rekursiv-ai" / "sagent" / "projects"
    done: Counter[str] = Counter()
    for path in _logs(flags.paths or [root]):
        outcome = _convert(path, dry_run=flags.dry_run)
        done[outcome] += 1
        if outcome in ("converted", "would convert"):
            print(f"{outcome}: {path}")
    print(
        ", ".join(f"{n} {outcome}" for outcome, n in sorted(done.items())) or "no logs",
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    paths: list[Path]
    dry_run: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags."""
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Old logs, or directories searched for them (default: the projects root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Say what would be converted; write nothing.",
    )


def _logs(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield each ``session.jsonl`` the paths name or hold, in order; a missing one holds none."""
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("session.jsonl"))
        elif path.is_file():
            yield path


def _convert(path: Path, *, dry_run: bool) -> str:
    """Convert one log; return what became of it."""
    if not is_legacy(path):
        return "skipped: not old"
    if dry_run:
        return (
            "would convert" if len(convert(path)) > 1 else "skipped: nothing to resume"
        )
    return "converted" if upgrade(path) else "skipped: nothing to resume"


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
