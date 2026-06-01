"""Tests for ``lib.atomic_file``: tmp-rename atomic writes."""

from __future__ import annotations

from pathlib import Path

import os
import stat

import pytest

from sagent.lib.atomic_file import atomic_write_bytes


def test_atomic_write_bytes_fsyncs_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytes payloads are fsynced before rename so a crash can't lose them."""
    dst = tmp_path / "out.bin"
    fsynced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    atomic_write_bytes(dst, b"payload")
    assert fsynced, "atomic_write_bytes must fsync before rename"
    assert dst.read_bytes() == b"payload"


def test_atomic_write_bytes_with_file_mode_fsyncs_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``file_mode`` path also fsyncs the data before the rename."""
    dst = tmp_path / "creds.bin"
    fsynced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    atomic_write_bytes(dst, b"secret", file_mode=0o600)
    assert fsynced
    assert dst.read_bytes() == b"secret"


def test_atomic_write_bytes_creates_file(tmp_path: Path) -> None:
    dst = tmp_path / "out.bin"
    atomic_write_bytes(dst, b"payload")
    assert dst.read_bytes() == b"payload"


def test_atomic_write_bytes_creates_parent_dirs(tmp_path: Path) -> None:
    dst = tmp_path / "a" / "b.bin"
    atomic_write_bytes(dst, b"data")
    assert dst.read_bytes() == b"data"


def test_atomic_write_bytes_with_file_mode(tmp_path: Path) -> None:
    dst = tmp_path / "creds.bin"
    atomic_write_bytes(dst, b"secret", file_mode=0o600)
    mode = stat.S_IMODE(dst.stat().st_mode)
    assert mode == 0o600


def test_atomic_write_bytes_file_mode_ignores_restrictive_umask(
    tmp_path: Path,
) -> None:
    dst = tmp_path / "exec.sh"
    old_umask = os.umask(0o077)
    try:
        atomic_write_bytes(dst, b"#!/bin/sh\n", file_mode=0o755)
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(dst.stat().st_mode)
    assert mode == 0o755


def test_atomic_write_bytes_with_file_mode_retries_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dst = tmp_path / "creds.bin"
    payload = b"secret payload"
    real_write = os.write
    write_sizes: list[int] = []

    def partial_write(fd: int, data: bytes) -> int:
        if not write_sizes:
            write_sizes.append(len(data) // 2)
            return real_write(fd, data[: write_sizes[-1]])
        write_sizes.append(len(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", partial_write)

    atomic_write_bytes(dst, payload, file_mode=0o600)

    assert write_sizes == [len(payload) // 2, len(payload) - len(payload) // 2]
    assert dst.read_bytes() == payload


def test_atomic_write_bytes_rollback_on_open_failure(tmp_path: Path) -> None:
    # Force the os.open path to fail by pointing at a non-writable parent.
    bad_parent = tmp_path / "ro"
    bad_parent.mkdir()
    bad_parent.chmod(0o500)
    dst = bad_parent / "out.bin"
    try:
        with pytest.raises(PermissionError):
            atomic_write_bytes(dst, b"x", file_mode=0o600)
    finally:
        bad_parent.chmod(0o700)
    assert not dst.exists()


def test_atomic_write_bytes_follows_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"orig")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    atomic_write_bytes(link, b"new")
    # Symlink still points at the same file; target updated through the link.
    assert link.is_symlink()
    assert target.read_bytes() == b"new"


def test_atomic_write_bytes_symlink_to_nonexistent_does_not_create_target_dirs(
    tmp_path: Path,
) -> None:
    """Dangling symlink must not materialise the foreign target directory."""
    target = tmp_path / "foreign" / "x.bin"  # parent does not exist
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    atomic_write_bytes(link, b"data")
    assert not (tmp_path / "foreign").exists()
    # The link itself was replaced with a regular file holding the payload
    # (the link's parent is ``tmp_path``, which exists).
    assert link.read_bytes() == b"data"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
