"""Tests for the floor on the changed sources' coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gc

import coverage
import pytest

from tools.testing import coverage_summary


if TYPE_CHECKING:
    from pathlib import Path


def _measure(
    root: Path,
    *,
    fail_under: int,
    measured: dict[str, tuple[int, int]],
    unrun: tuple[str, ...] = (),
) -> None:
    """Write ``pyproject.toml``, the sources, and a ``.coverage`` for them."""
    (root / "pyproject.toml").write_text(
        '[tool.coverage.run]\nomit = ["*/skipped/*"]\n'
        f"[tool.coverage.report]\nfail_under = {fail_under}\n",
    )
    lines: dict[str, list[int]] = {}
    for name, (statements, executed) in measured.items():
        module = root / name
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("".join(f"x{i} = {i}\n" for i in range(statements)))
        lines[str(module)] = list(range(1, executed + 1))
    for name in unrun:
        module = root / name
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("".join(f"x{i} = {i}\n" for i in range(4)))
    data = coverage.CoverageData(basename=str(root / ".coverage"))
    data.add_lines(lines)
    data.write()


def test_only_changed_files_below_the_floor_are_listed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed file no test executed counts 0%; an unchanged sibling not at all."""
    _measure(
        tmp_path,
        fail_under=98,
        measured={"pkg/low.py": (10, 5), "pkg/full.py": (50, 49), "pkg/sib.py": (4, 0)},
        unrun=("pkg/unrun.py",),
    )
    monkeypatch.chdir(tmp_path)

    assert coverage_summary.shortfall(
        ["pkg/low.py", "pkg/full.py", "pkg/unrun.py"],
    ) == [
        "pkg/low.py  5/10 missed  6-10",
        "pkg/unrun.py  4/4 missed  1-4",
        "Changed files are 84.38% covered, below fail_under=98.",
    ]


@pytest.mark.parametrize(
    ("measured", "fail_under", "short"),
    [
        ({"pkg/a.py": (50, 49), "pkg/b.py": (50, 50)}, 98, False),
        ({"pkg/a.py": (50, 48), "pkg/b.py": (50, 49)}, 98, True),
        ({"pkg/a.py": (10, 5)}, 0, False),
        ({"pkg/a.py": (10, 5), "pkg/skipped/c.py": (10, 0)}, 50, False),
    ],
    ids=["at-floor", "below-floor", "no-floor", "omitted"],
)
def test_the_floor_applies_to_the_changed_sources_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measured: dict[str, tuple[int, int]],
    fail_under: int,
    short: bool,
) -> None:
    _measure(tmp_path, fail_under=fail_under, measured=measured)
    monkeypatch.chdir(tmp_path)

    assert bool(coverage_summary.shortfall(list(measured))) is short


def test_a_total_exactly_at_the_floor_passes_and_each_file_is_judged_alike(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """99 of 100 statements at ``fail_under = 99`` is on the floor, not under it."""
    _measure(
        tmp_path,
        fail_under=99,
        measured={"pkg/a.py": (100, 99), "pkg/b.py": (200, 197)},
    )
    monkeypatch.chdir(tmp_path)

    assert coverage_summary.shortfall(["pkg/a.py"]) == []
    # a.py, on the floor, is not named; b.py, 0.5 under it, is.
    assert coverage_summary.shortfall(["pkg/a.py", "pkg/b.py"]) == [
        "pkg/b.py  3/200 missed  198-200",
        "Changed files are 98.67% covered, below fail_under=99.",
    ]


@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.parametrize("fail_under", [0, 98])
def test_the_data_file_is_closed_when_the_summary_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_under: int,
) -> None:
    """An open ``.coverage`` outlives the gate and fails a later test on gc."""
    _measure(tmp_path, fail_under=fail_under, measured={"pkg/low.py": (10, 5)})
    monkeypatch.chdir(tmp_path)

    coverage_summary.shortfall(["pkg/low.py"])
    gc.collect()


@pytest.mark.filterwarnings("error::ResourceWarning")
def test_an_unreadable_data_file_fails_and_leaves_nothing_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection left open fails whichever later test the collector runs in."""
    (tmp_path / ".coverage").write_bytes(b"not a database")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(coverage.exceptions.DataError):
        coverage_summary.shortfall(["a.py"])
    gc.collect()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
