"""Tests for files only their owner may read: opened, or replaced whole."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import pathlib
import stat

import pytest

from sagent.lib.files.owner_only import open_private, write_private


if TYPE_CHECKING:
    from pathlib import Path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_a_new_file_and_its_directories_are_private(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "secret.json"
    write_private(path, "{}")
    assert path.read_text() == "{}"
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700
    assert _mode(path.parent.parent) == 0o700


def test_a_file_readable_by_others_is_made_private_when_replaced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "secret.json"
    path.write_text("old")
    path.chmod(0o644)
    write_private(path, b"new")
    assert path.read_bytes() == b"new"
    assert _mode(path) == 0o600


def test_a_replace_leaves_no_staged_file_even_when_it_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "secret.json"
    write_private(path, "one")

    def broken(self: Path, target: Path) -> Path:
        del self, target
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_private(path, "two")
    assert path.read_text() == "one"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["secret.json"]


def test_a_replace_stages_beside_the_file_under_a_hidden_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beside it, so the rename never crosses a filesystem."""
    path = tmp_path / "secret.json"
    real = pathlib.Path.replace
    staged: list[Path] = []

    def recorded(self: Path, target: Path) -> Path:
        staged.append(self)
        return real(self, target)

    monkeypatch.setattr("pathlib.Path.replace", recorded)
    write_private(path, "one")
    (where,) = staged
    assert where.parent == tmp_path
    assert where.name.startswith(".secret.json.")


def test_a_failure_after_the_rename_raises_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged file is gone by then, which is no second failure."""
    path = tmp_path / "secret.json"
    real = pathlib.Path.replace

    def interrupted(self: Path, target: Path) -> Path:
        real(self, target)
        raise KeyboardInterrupt

    monkeypatch.setattr("pathlib.Path.replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_private(path, "one")
    assert path.read_text() == "one"


def test_a_directory_made_by_another_process_meanwhile_is_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raced = tmp_path / "raced"
    raced.mkdir()
    real = pathlib.Path.exists

    def unseen(self: Path) -> bool:
        return self != raced and real(self)

    monkeypatch.setattr("pathlib.Path.exists", unseen)
    write_private(raced / "secret.json", "one")
    assert (raced / "secret.json").read_text() == "one"


def test_a_new_file_is_created_private_before_any_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No moment passes in which another user could open it."""

    def skipped(fd: int, mode: int) -> None:
        del fd, mode

    monkeypatch.setattr("os.fchmod", skipped)
    path = tmp_path / "lock"
    os.close(open_private(path, os.O_RDWR))
    assert _mode(path) == 0o600


def test_an_opened_file_is_made_private_even_if_it_existed(tmp_path: Path) -> None:
    path = tmp_path / "d" / "journal.jsonl"
    path.parent.mkdir()
    path.write_text("kept\n")
    path.chmod(0o644)
    fd = open_private(path, os.O_RDWR | os.O_APPEND)
    try:
        os.write(fd, b"more\n")
    finally:
        os.close(fd)
    assert path.read_text() == "kept\nmore\n"
    assert _mode(path) == 0o600


def test_an_opened_file_is_created_with_its_directories(tmp_path: Path) -> None:
    path = tmp_path / "x" / "lock"
    os.close(open_private(path, os.O_RDWR))
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


def test_an_exclusive_open_refuses_a_file_that_exists(tmp_path: Path) -> None:
    path = tmp_path / "once"
    os.close(open_private(path, os.O_WRONLY | os.O_EXCL))
    with pytest.raises(FileExistsError):
        open_private(path, os.O_WRONLY | os.O_EXCL)


@pytest.mark.parametrize("opening", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_directory_creation_stops_at_the_first_existing_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opening: bool,
    missing: bool,
) -> None:
    parent = tmp_path / "a" / "b" if missing else tmp_path
    path = parent / "secret"
    visited: list[Path] = []
    real = pathlib.Path.exists

    def recorded(directory: Path) -> bool:
        visited.append(directory)
        return real(directory)

    monkeypatch.setattr(pathlib.Path, "exists", recorded)
    if opening:
        os.close(open_private(path, os.O_RDWR))
    else:
        write_private(path, "one")
    assert visited == ([parent, parent.parent, tmp_path] if missing else [tmp_path])
    assert _mode(path) == 0o600
    if missing:
        assert _mode(parent) == 0o700
        assert _mode(parent.parent) == 0o700


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
