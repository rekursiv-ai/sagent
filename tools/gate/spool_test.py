"""Tests for the spool every gate reads its files from."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools.gate import spool


if TYPE_CHECKING:
    from pathlib import Path


def test_chunks_accumulate_and_drain_once(tmp_path: Path) -> None:
    """pre-commit splits a long file list into chunks; the gate sees them all once."""
    spool.append(tmp_path, gate="ty", key=7, paths=["a.py", "b c.py"])
    spool.append(tmp_path, gate="ty", key=7, paths=["d.py"])

    assert spool.drain(tmp_path, gate="ty", key=7) == ["a.py", "b c.py", "d.py"]
    assert spool.drain(tmp_path, gate="ty", key=7) is None


def test_runs_and_gates_do_not_share_a_spool(tmp_path: Path) -> None:
    spool.append(tmp_path, gate="ty", key=7, paths=["a.py"])
    spool.append(tmp_path, gate="ty", key=8, paths=["b.py"])
    spool.append(tmp_path, gate="pytest", key=7, paths=["c.py"])

    assert spool.drain(tmp_path, gate="ty", key=7) == ["a.py"]
    assert spool.drain(tmp_path, gate="ty", key=8) == ["b.py"]
    assert spool.drain(tmp_path, gate="pytest", key=7) == ["c.py"]


def test_an_empty_spool_differs_from_none(tmp_path: Path) -> None:
    """A gate that wrote nothing to check still says it ran."""
    spool.write(tmp_path, gate="typecheck", key=7, paths=[])

    assert spool.drain(tmp_path, gate="typecheck", key=7) == []


def test_write_replaces_rather_than_appends(tmp_path: Path) -> None:
    spool.write(tmp_path, gate="typecheck", key=7, paths=["a.py", "b.py"])
    spool.write(tmp_path, gate="typecheck", key=7, paths=["c.py"])

    assert spool.drain(tmp_path, gate="typecheck", key=7) == ["c.py"]


def test_the_directory_is_private(tmp_path: Path) -> None:
    """Paths of uncommitted work must not be readable by other users."""
    spool.append(tmp_path / "spool", gate="ty", key=1, paths=["a.py"])

    assert (tmp_path / "spool").stat().st_mode & 0o777 == 0o700


_UV = "uv --quiet run --frozen --group pre-commit python -m tools.gate ty"


@pytest.mark.parametrize(
    ("commands", "expected"),
    [
        ({6: "python -m x", 5: _UV, 4: "/v/bin/python /v/bin/pre-commit run -a"}, 4),
        ({6: "python -m x", 5: _UV, 4: "python -mpre_commit hook-impl --x"}, 4),
        ({6: "python -m x", 5: _UV, 4: "bash"}, 1),
    ],
    ids=["run", "git-hook", "none"],
)
def test_the_key_is_the_pre_commit_process_not_uv(
    commands: dict[int, str],
    expected: int,
) -> None:
    """``uv ... --group pre-commit`` names pre-commit too, and must not match."""
    parents = {6: 5, 5: 4, 4: 1}

    assert (
        spool.run_key(6, process=lambda pid: (parents[pid], commands[pid])) == expected
    )


def test_an_unknown_process_ends_the_walk() -> None:
    assert spool.run_key(6, process=lambda _: None) == 1


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
