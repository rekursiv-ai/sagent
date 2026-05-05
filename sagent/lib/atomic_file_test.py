"""Tests for lib.atomic_file."""

from __future__ import annotations

from pathlib import Path

import stat

import pytest

from sagent.lib.atomic_file import (
    atomic_write,
    atomic_write_binary,
    atomic_write_bytes,
)


def test_atomic_write_text(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "out.txt"
    with atomic_write(dest) as f:
        _ = f.write("hello\n")
        _ = f.write("world\n")
    assert dest.read_text() == "hello\nworld\n"
    assert list(tmp_path.glob("**/*.tmp.*")) == []


def test_atomic_write_binary(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "out.bin"
    with atomic_write_binary(dest) as f:
        _ = f.write(b"\x00\xff")
        _ = f.write(b"\x01")
    assert dest.read_bytes() == b"\x00\xff\x01"
    assert list(tmp_path.glob("**/*.tmp.*")) == []


def test_atomic_write_bytes(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "out.bin"
    atomic_write_bytes(dest, b"\x01\x02")
    assert dest.read_bytes() == b"\x01\x02"
    assert list(tmp_path.glob("**/*.tmp.*")) == []


def test_atomic_write_bytes_with_file_mode(tmp_path: Path) -> None:
    """``file_mode=0o600`` makes the destination owner-only."""
    dest = tmp_path / "creds.json"
    atomic_write_bytes(dest, b'{"token": "secret"}', file_mode=0o600)
    assert dest.read_bytes() == b'{"token": "secret"}'
    # Mode bits - strip the file-type prefix.
    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600


def _raise_after_write(path: Path) -> None:
    with atomic_write(path) as f:
        _ = f.write("partial")
        raise RuntimeError("boom")


def test_atomic_write_leaves_target_untouched_on_error(tmp_path: Path) -> None:
    dest = tmp_path / "out.txt"
    _ = dest.write_text("original")
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after_write(dest)
    assert dest.read_text() == "original"


def _raise_after_binary_write(path: Path) -> None:
    with atomic_write_binary(path) as f:
        _ = f.write(b"partial")
        raise RuntimeError("boom")


def test_atomic_write_binary_leaves_target_untouched_on_error(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "out.bin"
    _ = dest.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="boom"):
        _raise_after_binary_write(dest)
    assert dest.read_bytes() == b"original"
    assert list(tmp_path.glob("**/*.tmp.*")) == []


def test_atomic_write_bytes_preserves_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    _ = target.write_text("old")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    atomic_write_bytes(link, b"new")
    assert link.is_symlink()
    assert link.resolve() == target
    assert target.read_text() == "new"


def test_atomic_write_text_preserves_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    _ = target.write_text("old")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with atomic_write(link) as f:
        _ = f.write("new")
    assert link.is_symlink()
    assert target.read_text() == "new"


def test_atomic_write_binary_preserves_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    _ = target.write_bytes(b"old")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with atomic_write_binary(link) as f:
        _ = f.write(b"new")
    assert link.is_symlink()
    assert target.read_bytes() == b"new"
