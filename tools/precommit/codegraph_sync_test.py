"""Tests for the CodeGraph index sync."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.precommit import codegraph_sync


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _sync(tmp_path: Path, *, installed: bool = True) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        return 0

    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    codegraph_sync.sync(
        root,
        caches=tmp_path / "caches",
        run=run,
        which=lambda name: f"/bin/{name}" if installed else None,
    )
    return calls


def test_a_missing_index_is_linked_into_the_cache_and_initialized(
    tmp_path: Path,
) -> None:
    """The ~387 MB index stays out of the checkout."""
    calls = _sync(tmp_path)

    assert (tmp_path / "repo" / ".codegraph").readlink() == tmp_path / "caches" / "repo"
    assert calls == [["codegraph", "init", "--yes"], ["codegraph", "sync", "--quiet"]]


def test_an_existing_index_is_only_synced(tmp_path: Path) -> None:
    (tmp_path / "repo" / ".codegraph").mkdir(parents=True)
    (tmp_path / "repo" / ".codegraph" / "codegraph.db").write_text("")

    assert _sync(tmp_path) == [["codegraph", "sync", "--quiet"]]


def test_without_the_cli_after_install_nothing_else_runs(tmp_path: Path) -> None:
    calls = _sync(tmp_path, installed=False)

    assert calls == [["npm", "install", "-g", "@colbymchenry/codegraph"]]


def test_an_unusable_cache_leaves_the_gates_on_their_floor(tmp_path: Path) -> None:
    """No index is a narrower check, never a failed commit."""
    (tmp_path / "caches").write_text("")

    assert _sync(tmp_path) == []
    assert not (tmp_path / "repo" / ".codegraph").is_symlink()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
