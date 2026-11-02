"""Unix socket addresses: one short enough to bind, and whether one answers.

``sockaddr_un.sun_path`` holds 104 bytes on Darwin and the BSDs, so a socket
under a long per-user state directory cannot be bound there at all, though the
filesystem takes the path. :func:`socket_address` keeps the socket's inode
where its owner wants it and hands back a short alias to bind and connect
through; :func:`accepts` says whether anything listens there now.

Typical usage example:

  address = socket_address(state_dir() / "rekursiv-ai" / "tool" / "host.sock")
  if not accepts(address):
      ...  # start whatever serves it
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import hashlib
import os
import socket
import stat


__all__ = ["UNIX_SOCKET_PATH_MAX_BYTES", "accepts", "socket_address"]


UNIX_SOCKET_PATH_MAX_BYTES: Final = 103
"""Portable pathname payload for ``sockaddr_un.sun_path``.

Darwin and the BSDs provide 104 bytes including the trailing NUL. Linux is
slightly larger, but using its limit would leave the same path broken on a
developer's Mac.
"""


def socket_address(logical_path: Path) -> Path:
    """Return a bindable AF_UNIX address for a logical socket path.

    A stable alias in an owner-only temporary directory points to the logical
    parent, shortening the kernel address while keeping the socket inode where
    its owner put it.

    Args:
      logical_path: Desired socket location.

    Returns:
      path: Direct path when it fits, otherwise a stable short alias.

    Raises:
      OSError: The temporary root cannot safely hold a short alias.

    """
    if len(os.fsencode(logical_path)) <= UNIX_SOCKET_PATH_MAX_BYTES:
        return logical_path
    return _aliased_socket_path(logical_path)


def accepts(address: Path, *, timeout_sec: float = 0.5) -> bool:
    """Return whether something accepts connections at ``address``.

    Args:
      address: The socket.
      timeout_sec: How long the connect may take.

    Returns:
      accepting: Whether a connect succeeded; the connection is closed at once.

    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(timeout_sec)
        probe.connect(str(address))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


def _aliased_socket_path(logical_path: Path) -> Path:
    """Create a short, secure alias to a long socket parent directory."""
    logical_parent = logical_path.parent.absolute()
    logical_parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Imported only for the exceptional long-path case. A thin client that
    # imports this on every invocation would otherwise pay for ``tempfile``.
    import tempfile  # noqa: PLC0415 -- Only the long-path case needs it.

    alias_root = Path(tempfile.gettempdir()) / f"t-{os.getuid():x}"
    alias_root.mkdir(mode=0o700, exist_ok=True)
    root_info = alias_root.lstat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.getuid()
        or root_info.st_mode & 0o077
    ):
        raise PermissionError(f"unsafe socket alias root: {alias_root}")

    alias_name = hashlib.blake2b(os.fsencode(logical_parent), digest_size=8).hexdigest()
    alias_parent = alias_root / alias_name
    try:
        alias_parent.symlink_to(logical_parent, target_is_directory=True)
    except FileExistsError:
        if (
            not alias_parent.is_symlink()
            or alias_parent.resolve() != logical_parent.resolve()
        ):
            raise OSError(f"unsafe socket alias: {alias_parent}") from None

    address = alias_parent / logical_path.name
    if len(os.fsencode(address)) > UNIX_SOCKET_PATH_MAX_BYTES:
        raise OSError(f"temporary directory is too long for AF_UNIX: {address}")
    return address
