from __future__ import annotations

from pathlib import Path
from typing import cast

import fcntl
import os
import subprocess
import sys
import time

import pytest

from sagent.lib.files import tempclaim


def test_a_claim_is_removed_with_its_lock_on_exit(tmp_path: Path) -> None:
    with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
        (directory / "file").touch()
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            directory.name,
            f"{directory.name}.lock",
        ]

    assert list(tmp_path.iterdir()) == []


def test_a_claim_sweeps_a_dead_owners_directory_and_keeps_a_live_one(
    tmp_path: Path,
) -> None:
    dead, live = tmp_path / "run-dead", tmp_path / "run-live"
    for directory in (dead, live):
        (directory / "popen-gw0").mkdir(parents=True)
        directory.with_suffix(".lock").touch()
    held = os.open(live.with_suffix(".lock"), os.O_RDONLY)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
            remaining = {path.name for path in tmp_path.iterdir()}
    finally:
        os.close(held)

    assert remaining == {
        "run-live",
        "run-live.lock",
        directory.name,
        f"{directory.name}.lock",
    }


def test_sweep_removes_only_old_unlocked_entries(tmp_path: Path) -> None:
    for name in ("run-old", "run-young"):
        (tmp_path / name).mkdir()
    (tmp_path / "run-old.pending").touch()
    os.utime(tmp_path / "run-old", (0, 0))
    os.utime(tmp_path / "run-old.pending", (0, 0))

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=60)

    assert [path.name for path in tmp_path.iterdir()] == ["run-young"]


def test_sweep_ignores_entries_without_the_prefix(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    other.with_suffix(".lock").touch()
    os.utime(other, (0, 0))

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=0)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "other",
        "other.lock",
    ]


def test_sweep_skips_claims_owned_by_another_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = tmp_path / "run-foreign"
    foreign.mkdir()
    monkeypatch.setattr(os, "getuid", lambda: foreign.stat().st_uid + 1)

    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=0)

    assert foreign.is_dir()


def test_reap_ignores_a_lock_removed_before_open(tmp_path: Path) -> None:
    missing = tmp_path / "gone.lock"
    tempclaim._reap(missing)
    assert not missing.exists()


@pytest.mark.cli_python_subprocess
def test_a_killed_owners_claim_is_swept_by_the_next_claim(tmp_path: Path) -> None:
    owner = subprocess.Popen(  # noqa: S603 -- fixed Python and module entry point.
        [
            sys.executable,
            "-c",
            (
                "import sys, pathlib\n"
                "from sagent.lib.files import tempclaim\n"
                "with tempclaim.claimed_directory(pathlib.Path(sys.argv[1]), prefix='run-')"
                " as d:\n"
                "    (d / 'file').touch()\n"
                "    print(d, flush=True)\n"
                "    sys.stdin.read()\n"
            ),
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert owner.stdout is not None
    orphan = Path(cast(str, owner.stdout.readline()).strip())  # ty: ignore[redundant-cast] -- Basedpyright's typeshed types Popen.stdout as IO[Any].
    owner.kill()
    owner.communicate(timeout=30)
    assert orphan.is_dir()

    with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
        remaining = sorted(path.name for path in tmp_path.iterdir())

    assert remaining == [directory.name, f"{directory.name}.lock"]


def test_remove_tree_deletes_directories_a_test_made_read_only(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "run" / "staged" / "agent"
    staged.mkdir(parents=True)
    (staged / "file").touch()
    staged.chmod(0o500)
    staged.parent.chmod(0o500)

    tempclaim.remove_tree(tmp_path / "run")

    assert not (tmp_path / "run").exists()


def test_dotted_claims_keep_distinct_locks_and_reap_the_exact_directory(
    tmp_path: Path,
) -> None:
    with (
        tempclaim.claimed_directory(tmp_path, prefix="a.b-") as first,
        tempclaim.claimed_directory(tmp_path, prefix="a.b-") as second,
    ):
        assert first.is_dir()
        assert second.is_dir()
        assert (tmp_path / f"{first.name}.lock").is_file()
        assert (tmp_path / f"{second.name}.lock").is_file()
    assert list(tmp_path.iterdir()) == []
    dead = tmp_path / "a.b-dead"
    dead.mkdir()
    (tmp_path / f"{dead.name}.lock").touch()
    unrelated = tmp_path / "a"
    unrelated.mkdir()
    tempclaim.sweep(tmp_path, prefix="a.b-")
    assert not dead.exists()
    assert unrelated.is_dir()


def test_a_claim_dying_before_its_lock_is_in_view_leaves_a_pending_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``.pending`` name is what ``sweep`` reaps once it is old."""

    def killed(fd: int, operation: int) -> None:
        del fd, operation
        raise KeyboardInterrupt

    monkeypatch.setattr(fcntl, "flock", killed)
    with (
        pytest.raises(KeyboardInterrupt),
        tempclaim.claimed_directory(tmp_path, prefix="run-"),
    ):
        pass
    (left,) = tmp_path.iterdir()
    assert left.name.startswith("run-")
    assert left.suffix == ".pending"
    assert len(left.stem) == len("run-") + 32


def test_a_lock_is_readable_by_its_owner_alone(tmp_path: Path) -> None:
    with tempclaim.claimed_directory(tmp_path, prefix="run-") as directory:
        mode = directory.with_name(f"{directory.name}.lock").stat().st_mode
    assert mode & 0o777 == 0o600


def test_reap_goes_on_when_another_sweep_took_the_lock_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dead = tmp_path / "run-dead"
    dead.mkdir()
    lock = tmp_path / "run-dead.lock"
    lock.touch()
    remove = tempclaim.remove_tree

    def raced(path: Path) -> None:
        remove(path)
        lock.unlink()

    monkeypatch.setattr(tempclaim, "remove_tree", raced)
    tempclaim._reap(lock)
    assert list(tmp_path.iterdir()) == []


def test_remove_tree_of_nothing_is_a_no_op(tmp_path: Path) -> None:
    tempclaim.remove_tree(tmp_path / "absent")
    assert list(tmp_path.iterdir()) == []


def test_remove_tree_gives_up_quietly_on_what_it_cannot_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tree this user cannot make writable is left, not raised over."""
    if os.geteuid() == 0:
        pytest.skip("root unlinks beneath a read-only directory")
    staged = tmp_path / "run" / "staged"
    (staged / "agent").mkdir(parents=True)
    staged.chmod(0o500)

    def refused(self: Path, mode: int) -> None:
        del mode
        raise PermissionError(self)

    monkeypatch.setattr(Path, "chmod", refused)
    tempclaim.remove_tree(tmp_path / "run")
    monkeypatch.undo()
    staged.chmod(0o700)
    assert (staged / "agent").is_dir()


def test_sweep_by_default_removes_an_unlocked_entry_older_than_a_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, young = tmp_path / "run-old", tmp_path / "run-young"
    for directory in (old, young):
        directory.mkdir()
    os.utime(old, (1_000, 1_000))
    os.utime(young, (1_001, 1_001))
    monkeypatch.setattr(time, "time", lambda: 1_000 + 86_400.5)
    tempclaim.sweep(tmp_path, prefix="run-")
    assert [path.name for path in tmp_path.iterdir()] == ["run-young"]


def test_sweep_keeps_an_entry_exactly_at_the_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "run-edge"
    entry.mkdir()
    os.utime(entry, (1_000, 1_000))
    monkeypatch.setattr(time, "time", lambda: 1_060)
    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=60)
    assert entry.is_dir()


def test_sweep_keeps_an_old_directory_whose_owner_lives(tmp_path: Path) -> None:
    live = tmp_path / "run-live"
    live.mkdir()
    live.with_suffix(".lock").touch()
    os.utime(live, (0, 0))
    held = os.open(live.with_suffix(".lock"), os.O_RDONLY)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=60)
    finally:
        os.close(held)
    assert live.is_dir()


def test_sweep_goes_on_past_a_vanished_or_foreign_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gone, foreign, old = (
        tmp_path / f"run-{name}" for name in ("gone", "foreign", "old")
    )
    for directory in (foreign, old):
        directory.mkdir()
    os.utime(old, (0, 0))
    real = Path.lstat

    def lstat(self: Path) -> os.stat_result:
        info = real(self)
        if self != foreign:
            return info
        return os.stat_result((*info[:4], info.st_uid + 1, *info[5:10]))

    def glob(self: Path, pattern: str) -> list[Path]:
        del self, pattern
        return [gone, foreign, old]

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "glob", glob)
    tempclaim.sweep(tmp_path, prefix="run-", max_age_sec=60)
    assert foreign.is_dir()
    assert not old.exists()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
