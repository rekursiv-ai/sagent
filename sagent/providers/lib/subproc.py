"""Persistent vendor-CLI subprocess with NDJSON stdin/stdout.

Wraps an ``asyncio.subprocess`` whose stdin/stdout speak line-oriented
JSON (claude's stream-json mode or gemini's ACP). Drains stderr to a
bounded ring buffer for diagnostics, owns an optional tmpdir that is
``rm -rf``'d on close, and exposes ``write_line`` / ``read_json_line``
/ ``close``.

The class is provider-agnostic: every CLI-wrapping provider feeds it
``argv`` + ``env`` + ``tmpdir`` produced by its own recipe, then
speaks the wire protocol on the returned handle.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import cast

import asyncio
import contextlib
import json
import logging
import shutil

from sagent.lib.json import MutableJSON
from sagent.types.exceptions import log_task_exception


__all__ = ["Subproc", "SubprocessTransportError"]

logger = logging.getLogger(__name__)


_STDERR_TAIL_LINES = 100
_TERMINATE_GRACE_SEC = 2.0
_READ_IDLE_TIMEOUT_SEC = 60.0


class SubprocessTransportError(RuntimeError):
    """Subprocess transport failed and the provider may respawn."""


class Subproc:
    """A managed CLI subprocess with NDJSON stdin/stdout and tmpdir cleanup.

    Args:
      argv: Executable + arguments. ``argv[0]`` is invoked verbatim.
      env: Full environment for the child. ``None`` inherits the parent's.
      tmpdir: Directory deleted by :meth:`close`. ``None`` leaves nothing
          for the wrapper to clean.
      cwd: Working directory for the child. ``None`` inherits.
      read_timeout_sec: Maximum idle seconds while waiting for one stdout line.

    """

    def __init__(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        tmpdir: Path | None = None,
        cwd: Path | None = None,
        read_timeout_sec: float = _READ_IDLE_TIMEOUT_SEC,
    ) -> None:
        self._argv = argv
        self._env = env
        self._tmpdir = tmpdir
        self._cwd = cwd
        self._read_timeout_sec = read_timeout_sec
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None
        self.sagent_google_cli_state: object | None = None
        self._closed = False

    async def start(self) -> None:
        """Spawn the subprocess and start the stderr drain.

        Raises:
          RuntimeError: If the executable is not on ``PATH``.

        """
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=str(self._cwd) if self._cwd is not None else None,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._stderr_task.add_done_callback(
            log_task_exception(logger, "subprocess stderr drainer crashed"),
        )

    async def write_line(self, line: str) -> None:
        """Write one NDJSON line (newline appended) to the child's stdin.

        Args:
          line: Serialized JSON without trailing newline.

        Raises:
          RuntimeError: If the subprocess exited before the write completed.

        """
        proc = self._proc
        assert proc is not None
        assert proc.stdin is not None
        try:
            proc.stdin.write(line.encode() + b"\n")
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise SubprocessTransportError(
                f"subprocess stdin closed: {self._diagnostic()}",
            ) from exc

    async def read_line(self) -> str:
        """Read one decoded line from stdout (newline stripped).

        Returns:
          line: One line of stdout, or ``""`` on EOF.

        """
        proc = self._proc
        assert proc is not None
        assert proc.stdout is not None
        try:
            raw = await asyncio.wait_for(
                proc.stdout.readline(), timeout=self._read_timeout_sec
            )
        except TimeoutError as exc:
            raise SubprocessTransportError(
                f"subprocess stdout idle timeout after {self._read_timeout_sec:g}s: "
                f"{self._diagnostic()}",
            ) from exc
        if not raw:
            return ""
        return raw.rstrip(b"\n").decode("utf-8", errors="replace")

    async def read_json_line(
        self, *, skip_non_json: bool = False
    ) -> MutableJSON | None:
        """Read until a valid JSON object line appears.

        Args:
          skip_non_json: Discard non-JSON lines instead of raising on them.
              Set when banner/log lines may precede the protocol stream.

        Returns:
          obj: Decoded JSON object, or ``None`` on EOF.

        Raises:
          SubprocessTransportError: A protocol line failed to parse and
              ``skip_non_json`` is False.

        """
        while True:
            line = await self.read_line()
            if not line:
                return None
            stripped = line.lstrip()
            if not stripped:
                continue
            if skip_non_json and not stripped.startswith("{"):
                logger.debug("skipping non-JSON stdout: %s", line[:120])
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if skip_non_json:
                    logger.debug("skipping malformed stdout: %s", line[:120])
                    continue
                raise SubprocessTransportError(
                    f"non-JSON line on stdout: {line[:200]!r}: {self._diagnostic()}"
                ) from exc
            if isinstance(obj, dict):
                return cast(MutableJSON, obj)
            logger.debug("skipping non-object JSON: %s", line[:120])

    async def close(self) -> None:
        """Terminate the subprocess and remove the owned tmpdir.

        Idempotent. Sends SIGTERM, waits ``_TERMINATE_GRACE_SEC``, then
        SIGKILL on timeout. ``rm -rf`` the tmpdir after the child has
        released its file handles.
        """
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                _ = await asyncio.wait_for(proc.wait(), _TERMINATE_GRACE_SEC)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                _ = await proc.wait()
        if self._stderr_task is not None:
            _ = self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 -- drain failure must not mask close
                logger.debug("subprocess close: stderr drain raised: %s", exc)
        if self._tmpdir is not None and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    @property
    def pid(self) -> int | None:
        """Underlying child PID, ``None`` if not started."""
        return self._proc.pid if self._proc is not None else None

    @property
    def is_alive(self) -> bool:
        """Whether the subprocess is started and has not exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def stderr_tail(self) -> str:
        """Last ``_STDERR_TAIL_LINES`` lines of stderr, newline-joined."""
        return "\n".join(self._stderr_tail)

    def _diagnostic(self) -> str:
        """Format a one-line snapshot for error messages."""
        rc = self._proc.returncode if self._proc is not None else "unstarted"
        tail = self.stderr_tail[-400:]
        return f"argv={self._argv[0]!r} rc={rc} stderr_tail={tail!r}"

    async def _drain_stderr(self) -> None:
        """Read stderr forever, buffering the tail for diagnostics."""
        proc = self._proc
        assert proc is not None
        assert proc.stderr is not None
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                return
            self._stderr_tail.append(
                raw.rstrip(b"\n").decode("utf-8", errors="replace")
            )
