"""Tests for ``lib.atomic_file``: tmp-rename atomic writes."""

from __future__ import annotations

from pathlib import Path

import os
import stat

import pytest

from sagent.lib.atomic_file import (
    atomic_write,
    atomic_write_binary,
    atomic_write_bytes,
)


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"
    with atomic_write(dst) as f:
        _ = f.write("hello world")
    assert dst.read_text() == "hello world"


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    dst = tmp_path / "a" / "b" / "c.txt"
    with atomic_write(dst) as f:
        _ = f.write("nested")
    assert dst.read_text() == "nested"


def test_atomic_write_text_rollback_on_exception(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"
    dst.write_text("original")

    def _do_write() -> None:
        with atomic_write(dst) as f:
            _ = f.write("clobber")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _do_write()
    # Existing file is preserved; tmp file is removed.
    assert dst.read_text() == "original"
    siblings = [p for p in tmp_path.iterdir() if p.name != "out.txt"]
    assert siblings == []


def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"
    dst.write_text("v1")
    with atomic_write(dst) as f:
        _ = f.write("v2")
    assert dst.read_text() == "v2"


def test_atomic_write_binary_creates_file(tmp_path: Path) -> None:
    dst = tmp_path / "out.bin"
    payload = b"\x00\x01\x02hello"
    with atomic_write_binary(dst) as f:
        _ = f.write(payload)
    assert dst.read_bytes() == payload


def test_atomic_write_binary_rollback_on_exception(tmp_path: Path) -> None:
    dst = tmp_path / "out.bin"
    dst.write_bytes(b"orig")

    def _do_write() -> None:
        with atomic_write_binary(dst) as f:
            _ = f.write(b"clobber")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _do_write()
    assert dst.read_bytes() == b"orig"


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


def test_atomic_write_follows_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("orig")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with atomic_write(link) as f:
        _ = f.write("new")
    # Symlink still points at the same file; target updated through the link.
    assert link.is_symlink()
    assert target.read_text() == "new"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
