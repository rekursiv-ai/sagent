"""The per-user daemon lifecycle, shared by local clients and TLS serving."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import asyncio
import fcntl
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

from sagent.host.server import DEFAULT_TOOLS, Server
from sagent.host.strings import ServeDirs, ServerConfig, configuration
from sagent.host.wire import (
    MAX_LINE_BYTES,
    DaemonControl,
    DaemonStatus,
    encoded,
    received,
)
from sagent.lib.files.owner_only import open_private
from sagent.lib.files.source_version import source_version
from sagent.lib.posix.sockets import socket_address


_CWD: Final[Path] = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncio.streams import StreamReader, StreamWriter
    from collections.abc import Sequence
    from subprocess import Popen


def fingerprint() -> str:
    """Fingerprint the daemon's sources and shared dependencies without importing them.

    Returns:
      version: The fingerprint of source files currently on disk.

    """
    root = _CWD.parent
    outside = tuple(
        f"../{path.relative_to(root.parent)}"
        for name in ("lib", "treekle")
        for path in sorted((root.parent / name).rglob("*.py"))
        if not path.name.endswith("_test.py")
    )
    return source_version(root, outside=outside)


async def control(dirs: ServeDirs, request: DaemonControl) -> DaemonStatus:
    """Send an owner-only control request without starting a daemon.

    Args:
      dirs: The daemon's private state and credential directories.
      request: The control operation to send once.

    Returns:
      status: The daemon's identity and listener configuration.

    """
    reader: StreamReader
    writer: StreamWriter
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(
            socket_address(dirs.state / "daemon.sock"),
            limit=MAX_LINE_BYTES,
        ),
        timeout=5,
    )
    try:
        writer.write(encoded(request))
        await writer.drain()
        status = await asyncio.wait_for(received(reader), timeout=30)
        if not isinstance(status, DaemonStatus):
            raise ConnectionError(
                "The daemon accepted a control request but returned no status.",
            )
        return status
    finally:
        writer.close()
        await writer.wait_closed()


async def ensure(dirs: ServeDirs, *, version: str = "") -> DaemonStatus:
    """Connect to this user's daemon, starting or upgrading it under a lock.

    Args:
      dirs: The daemon's private state and credential directories.
      version: The caller's source fingerprint; empty computes it from disk.

    Returns:
      status: The ready daemon, after any required replacement.

    """
    version = version or fingerprint()
    fd = open_private(dirs.state / "daemon-start.lock", os.O_RDWR)
    try:
        await _locked(fd)
        deadline = time.monotonic() + 30
        status: DaemonStatus | None = None
        process = None
        while time.monotonic() < deadline:
            try:
                status = await control(dirs, DaemonControl(source_version=version))
            except (FileNotFoundError, ConnectionRefusedError):
                if process is not None and (code := process.poll()) is not None:
                    raise RuntimeError(
                        f"The sagent daemon exited with status {code};"
                        f" see {dirs.state / 'daemon.log'}.",
                    ) from None
                if process is None and status is None:
                    process = _spawn(dirs, version=version)
            else:
                if not status.restarting:
                    return status
            # The daemon is another process; only its socket reports its status.
            await asyncio.sleep(0.02)  # house-ignore[sleep] -- See above.
        raise TimeoutError(
            f"The sagent daemon did not become ready; see {dirs.state / 'daemon.log'}.",
        )
    finally:
        os.close(fd)


class Flags(Protocol):
    """The private daemon process's startup arguments."""

    state: Path
    secrets: Path
    source_version: str
    handoff_fd: int | None


def main(argv: Sequence[str] | None = None) -> int:
    """Serve until an owner requests replacement or shutdown.

    Args:
      argv: Private startup arguments, or the process arguments when omitted.

    Returns:
      code: Zero after an orderly daemon shutdown.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    handoff = ""
    if flags.handoff_fd is not None:
        with os.fdopen(flags.handoff_fd) as snapshot:
            handoff = snapshot.read()
    asyncio.run(
        _serve(
            ServeDirs(state=flags.state, secrets=flags.secrets),
            version=flags.source_version,
            handoff=handoff,
        ),
    )
    return 0


async def _serve(dirs: ServeDirs, *, version: str, handoff: str = "") -> None:
    fd = open_private(dirs.state / "daemon.lock", os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        address = socket_address(dirs.state / "daemon.sock")
        address.unlink(missing_ok=True)
        saved = configuration(
            dirs,
            default=ServerConfig(listen=(), tools=tuple(DEFAULT_TOOLS)),
        )
        async with Server(
            dirs,
            listen=saved.listen,
            tools=saved.tools,
            local_socket=address,
            source_version=version,
            handoff=handoff,
        ) as server:
            while True:
                _ = await server.stopping.wait()
                if not server.replacement:
                    return
                try:
                    state = await server.handoff()
                except Exception:
                    logger.exception("Daemon replacement failed")
                    continue
                try:
                    with tempfile.TemporaryFile(dir=dirs.state) as snapshot:
                        _ = snapshot.write(state.encode())
                        snapshot.flush()
                        snapshot.seek(0)
                        os.set_inheritable(snapshot.fileno(), True)
                        os.execv(  # noqa: S606 -- Replace this interpreter with the same fixed module.
                            sys.executable,
                            [
                                sys.executable,
                                "-P",
                                "-m",
                                "sagent",
                                "daemon",
                                "--state",
                                str(dirs.state),
                                "--secrets",
                                str(dirs.secrets),
                                "--source-version",
                                server.replacement,
                                "--handoff-fd",
                                str(snapshot.fileno()),
                            ],
                        )
                except Exception:
                    await server.resume(state)
                    logger.exception("Daemon replacement failed")
    finally:
        os.close(fd)


async def _locked(fd: int) -> None:
    """Take ``fd``'s exclusive lock, waiting on the event loop rather than a thread."""
    # Not `to_thread(flock)`: a cancelled wait left that thread blocked in flock, and
    # on macOS the close in `ensure`'s finally then blocks until flock returns, so
    # cancelling a start hung the whole event loop until the lock came free.
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            await asyncio.sleep(0.02)
        else:
            return


def _spawn(dirs: ServeDirs, *, version: str) -> Popen[bytes]:
    """Start outside the client's terminal; return its handle and reap it independently."""
    fd = open_private(dirs.state / "daemon.log", os.O_WRONLY | os.O_APPEND)
    try:
        process = subprocess.Popen(  # noqa: S603 -- Fixed interpreter and module, owner-owned paths.
            [
                sys.executable,
                "-P",
                "-m",
                "sagent",
                "daemon",
                "--state",
                str(dirs.state),
                "--secrets",
                str(dirs.secrets),
                "--source-version",
                version,
            ],
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=fd,
            start_new_session=True,
        )
    finally:
        os.close(fd)
    threading.Thread(target=process.wait, daemon=True).start()
    return process


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--handoff-fd", type=int)
