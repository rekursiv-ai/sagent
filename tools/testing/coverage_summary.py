"""Fail when the changed sources, taken together, fall below ``fail_under``.

The ``pytest`` gate measures whole directories but tests only the changed
files, so its own total counts every untested sibling. This applies the floor
to the changed, non-omitted sources alone, a file no test executed counting
0%, and names each one short of it. No floor passes everything.

Never ``coverage report --include``: it matched 2,727 globs per file for 48s.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fnmatch
import os


if TYPE_CHECKING:
    from collections.abc import Sequence

    import coverage


def shortfall(sources: Sequence[str]) -> list[str]:
    """Return one line per source below the floor, then the total; [] when met.

    Reads ``.coverage`` and the coverage config from the working directory.

    Args:
      sources: The changed, non-test sources.

    Returns:
      lines: Empty when the floor holds or there is none.

    """
    # Here, not at the top: the ``testing`` gates import this module, and an
    # export without pytest-cov has no coverage; the gate calls this only with it.
    import coverage  # noqa: PLC0415 -- optional in exports; see above.

    measured = coverage.Coverage()
    # Closed on every path: coverage leaves an unreadable file's connection
    # open, and it then fails whichever later test the collector runs in.
    try:
        measured.load()
        return _shortfall(measured, sources=sources)
    finally:
        measured.get_data().close(force=True)


def _shortfall(measured: coverage.Coverage, *, sources: Sequence[str]) -> list[str]:
    """Return ``shortfall``'s lines from data already loaded."""
    floor = measured.config.fail_under
    if not floor:
        return []
    rows = [
        (path, len(statements), len(missed), gaps)
        for path, statements, _, missed, gaps in map(
            measured.analysis2,
            [
                path
                for path in sources
                if not any(
                    fnmatch.fnmatch(str(Path(path).absolute()), pattern)
                    for pattern in measured.config.run_omit
                )
            ],
        )
    ]
    total = sum(statements for _, statements, _, _ in rows)
    covered = total - sum(missed for _, _, missed, _ in rows)
    if not total or 100 * covered >= floor * total:
        return []
    return [
        *(
            f"{os.path.relpath(path)}  {missed}/{statements} missed  {gaps}"
            for path, statements, missed, gaps in rows
            if statements and 100 * (statements - missed) < floor * statements
        ),
        (
            f"Changed files are {100 * covered / total:.2f}% covered, below "
            f"fail_under={floor:g}."
        ),
    ]
