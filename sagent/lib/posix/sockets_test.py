"""Tests for Unix socket addresses: bindable however long, and whether one answers."""

from __future__ import annotations

from pathlib import Path

import os
import socket
import tempfile

import pytest

from sagent.lib.posix.sockets import UNIX_SOCKET_PATH_MAX_BYTES, accepts, socket_address


def _bind(path: Path) -> None:
    """Bind and release ``path``, removing the socket file after."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
    finally:
        listener.close()
        path.unlink(missing_ok=True)


def test_a_long_path_binds_through_an_alias_to_its_own_directory(
    tmp_path: Path,
) -> None:
    """AF_UNIX limits the encoded address even when the filesystem does not."""
    logical = tmp_path / ("long-state-segment-" * 12) / "host.sock"
    path = socket_address(logical)

    _bind(path)

    assert len(os.fsencode(path)) <= UNIX_SOCKET_PATH_MAX_BYTES
    assert path.resolve().parent == logical.parent.resolve()


def test_a_long_relative_path_targets_the_callers_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A relative path must not become relative to the alias directory."""
    monkeypatch.chdir(tmp_path)
    logical = Path("long-relative-segment-" * 8) / "host.sock"
    logical.parent.mkdir()
    path = socket_address(logical)

    _bind(path)

    assert path.resolve().parent == logical.parent.resolve()


def test_an_address_of_exactly_the_limit_is_used_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only an address past the kernel's limit needs a short alias."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", "runtime")
    logical = Path("s" * UNIX_SOCKET_PATH_MAX_BYTES)
    path = socket_address(logical)

    _bind(path)

    assert path == logical


def test_rejects_an_alias_root_with_group_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A shared alias root would let another local user replace the socket."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", "runtime")
    alias_root = Path(tempfile.gettempdir()) / f"t-{os.getuid():x}"
    alias_root.mkdir(parents=True)
    alias_root.chmod(0o750)

    with pytest.raises(PermissionError, match="unsafe socket alias root"):
        socket_address(Path("long-logical-segment-" * 8) / "host.sock")


def test_rejects_an_alias_that_points_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An alias another process planted must not redirect the socket."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", "runtime")
    Path("runtime").mkdir()
    logical = Path("long-logical-segment-" * 8) / "host.sock"
    planted = socket_address(logical).parent
    planted.unlink()
    planted.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(OSError, match="unsafe socket alias:"):
        socket_address(logical)


def test_a_temporary_directory_too_long_for_an_alias_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = tmp_path / ("long-runtime-segment-" * 6)
    runtime.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(runtime))

    with pytest.raises(OSError, match="temporary directory is too long for AF_UNIX"):
        socket_address(tmp_path / ("long-logical-segment-" * 8) / "host.sock")


def test_a_listening_socket_accepts_and_a_missing_one_does_not(
    tmp_path: Path,
) -> None:
    address = tmp_path / "s"
    assert not accepts(address)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(address))
        listener.listen()
        assert accepts(address)
    finally:
        listener.close()
    assert not accepts(address)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
