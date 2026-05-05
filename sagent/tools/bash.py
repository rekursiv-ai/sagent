"""Bash tool.

Parsing, matcher helpers, and the read-only command classifier all
live in :mod:`sagent.tools.lib.bash`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import contextlib
import logging
import os
import secrets
import signal
import subprocess
import time

from sagent.custom_types import Message, MultipartMessage, TextMessage
from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    ToolState,
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.tools.lib.bash import Node, cached_parse_bash


logger = logging.getLogger(__name__)


@runtime_checkable
class _BashMatchPeer(Protocol):
    """Duck-type for sibling tools that provide a bash-lint hook."""

    def bash_match(self, trees: Sequence[Node]) -> str | None: ...


def _suppress_oserror() -> contextlib.suppress:
    """OSError handler for process-group signals (race: proc may have exited)."""
    return contextlib.suppress(OSError, ProcessLookupError)


# Mid-stream bash output line cap. Large outputs (test suites, log
# dumps) trim the middle to keep tool_result under the per-tool
# persist threshold, preserving head/tail for diagnosis.
_BASH_MAX_LINES = 500
_BASH_HEAD_LINES = 250
_BASH_TAIL_LINES = 250
BASH_DEFAULT_TIMEOUT_MS = 120_000
BASH_MAX_TIMEOUT_MS = 600_000


# Kept-alive references to background subprocesses so their Popen
# objects don't get garbage-collected mid-run. GC fires ``__del__``
# which emits ``ResourceWarning: subprocess N still running`` and
# (under ``asyncio`` debug mode) kills the child. Retaining the
# ``Popen`` here trades a tiny leak (sizeof(Popen) per bg job,
# released at interpreter exit via ``atexit``) for clean semantics.
_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


def _render_bash_description(text: str) -> str:
    """Substitute Sagent's static bash timeout values into prompt text."""
    return (
        text.replace(
            "${GET_MAX_TIMEOUT_MS()/60000}",
            str(BASH_MAX_TIMEOUT_MS // 60_000),
        )
        .replace(
            "${GET_DEFAULT_TIMEOUT_MS()/60000}",
            str(BASH_DEFAULT_TIMEOUT_MS // 60_000),
        )
        .replace("${GET_MAX_TIMEOUT_MS()}", str(BASH_MAX_TIMEOUT_MS))
        .replace("${GET_DEFAULT_TIMEOUT_MS()}", str(BASH_DEFAULT_TIMEOUT_MS))
    )


def _trim_bash_output(lines: list[str]) -> str:
    """Cap output at _BASH_MAX_LINES, dropping the middle if over."""
    if len(lines) <= _BASH_MAX_LINES:
        return "\n".join(lines)
    head = lines[:_BASH_HEAD_LINES]
    tail = lines[-_BASH_TAIL_LINES:]
    omitted = len(lines) - _BASH_HEAD_LINES - _BASH_TAIL_LINES
    return "\n".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(tail)


class Bash:
    """Execute shell commands."""

    name: str = "Bash"
    tool_id: str = "application/x-tool-bash"
    description: str = _render_bash_description(load_tool_description("Bash"))
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": BASH_MAX_TIMEOUT_MS,
                    "description": (
                        "Optional timeout in milliseconds. Must be between"
                        f" 1 and {BASH_MAX_TIMEOUT_MS}."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Clear, concise description of what this command does"
                        " in active voice."
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run this command in the background.",
                },
            },
            "required": ["command"],
        }
    )

    def __init__(
        self,
        *,
        peers: Sequence[object] = (),
    ) -> None:
        """Collect ``bash_match`` matchers from peer tools.

        Peers are sibling tools in the same agent. Any peer whose class
        defines ``bash_match(self, trees: Sequence[Node]) -> str | None``
        is registered as a lint source; its output is prepended to the
        ``tool_result`` inside a ``<system-reminder>`` block when it
        fires - same framing as ``changed_files_context`` so the model
        treats it as ambient metadata rather than tool output.
        """
        self._peer_matchers: tuple[Callable[[Sequence[Node]], str | None], ...] = tuple(
            peer.bash_match for peer in peers if isinstance(peer, _BashMatchPeer)
        )

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: Human-readable summary of the command.

        """
        directive = get_directive(msg)
        cmd = str(directive.get("command", ""))
        cmd = cmd.replace("\r\n", "⏎").replace("\n", "⏎").replace("\r", "⏎")
        cmd = cmd.replace("\t", " ")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"Bash {cmd}" if cmd else "Bash"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Execute the command and return the result.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message with stdout/stderr.

        """
        directive = get_directive(msg)
        command = str(directive.get("command", ""))
        timeout = int_val(directive.get("timeout"), BASH_DEFAULT_TIMEOUT_MS)
        run_in_background = bool_val(directive.get("run_in_background"), False)
        result = await run_sync(
            self._run,
            command=command,
            timeout=timeout,
            run_in_background=run_in_background,
        )
        if self._peer_matchers:
            nudges = self._collect_nudges(command)
            if nudges:
                body = "\n".join(f"[bash-lint] {h}" for h in nudges)
                banner = f"<system-reminder>\n{body}\n</system-reminder>"
                text = cast(str, result.content)
                return MultipartMessage(
                    (
                        TextMessage(f"{banner}\n\n{text}", "text/plain"),
                        *(TextMessage(h, "text/x-hint-tool-use-nudge") for h in nudges),
                    ),
                    "multipart/mixed",
                )
        return result

    def _collect_nudges(self, command: str) -> list[str]:
        trees = cached_parse_bash(command, get_tool_state().bash_parse_cache)
        if trees is None:
            return []
        nudges: list[str] = []
        for matcher in self._peer_matchers:
            nudge = matcher(trees)
            if nudge:
                nudges.append(nudge)
        return nudges

    def _run(
        self,
        *,
        command: str,
        timeout: int = BASH_DEFAULT_TIMEOUT_MS,
        run_in_background: bool = False,
    ) -> str:
        state = get_tool_state()
        if not Path(state.bash_cwd).is_dir():
            state.bash_cwd = (
                state.start_cwd if Path(state.start_cwd).is_dir() else str(Path.home())
            )
        timeout = int(timeout)
        timeout_s = max(1, min(timeout // 1000, BASH_MAX_TIMEOUT_MS // 1_000))
        if run_in_background:
            return _run_background(command, state=state)
        return _run_foreground(command, state=state, timeout_s=timeout_s)


def _run_background(command: str, *, state: ToolState) -> str:
    """Spawn a detached background process."""
    proc = subprocess.Popen(  # noqa: S603 -- trusted fixed argv, not user input
        ["/bin/bash", "-c", command],
        cwd=state.bash_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _BACKGROUND_PROCESSES[:] = [p for p in _BACKGROUND_PROCESSES if p.poll() is None]
    _BACKGROUND_PROCESSES.append(proc)
    logger.info("Background process started: pid=%d", proc.pid)
    return f"(running in background, pid={proc.pid})"


def _run_foreground(command: str, *, state: ToolState, timeout_s: int) -> str:
    """Run a command in the foreground, draining pipes continuously."""
    sentinel = f"__SAGENT_CWD_{secrets.token_hex(4)}__"
    tracked_cmd = f"trap 'echo {sentinel}=$(pwd)' EXIT\n{command}"
    proc = subprocess.Popen(  # noqa: S603 -- trusted fixed argv, not user input
        ["/bin/bash", "-c", tracked_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=state.bash_cwd,
        start_new_session=True,
    )
    stdout, stderr, reason = _communicate_with_abort(
        proc, state=state, timeout_s=timeout_s
    )
    if reason is not None:
        killed_lines = stdout.split("\n")
        if stderr:
            killed_lines.extend(stderr.split("\n"))
        return _trim_bash_output(killed_lines).strip() + f"\n[{reason}]"
    return _process_output(proc, stdout, stderr, sentinel=sentinel, state=state)


def _communicate_with_abort(
    proc: subprocess.Popen[str],
    *,
    state: ToolState,
    timeout_s: int,
) -> tuple[str, str, str | None]:
    """Drain pipes via ``communicate()``, polling for abort/timeout.

    Returns:
      stdout: Captured stdout.
      stderr: Captured stderr.
      reason: ``None`` on normal exit, or a human-readable reason
          (``"interrupted after Ns"`` / ``"timeout after Ns"``).

    """
    start = time.monotonic()
    deadline = start + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if state.abort_event.is_set():
            return _kill_and_drain(proc, start=start, reason="interrupted")
        if remaining <= 0:
            return _kill_and_drain(proc, start=start, reason="timeout")
        try:
            stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
            return stdout, stderr, None
        except subprocess.TimeoutExpired:
            continue


def _kill_and_drain(
    proc: subprocess.Popen[str],
    *,
    start: float,
    reason: str,
) -> tuple[str, str, str]:
    """SIGTERM/SIGKILL a process group and drain remaining output."""
    with _suppress_oserror():
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        with _suppress_oserror():
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    stdout, stderr = proc.communicate()
    elapsed = time.monotonic() - start
    return stdout, stderr, f"{reason} after {elapsed:.1f}s"


def _process_output(
    proc: subprocess.Popen[str],
    stdout: str,
    stderr: str,
    *,
    sentinel: str,
    state: ToolState,
) -> str:
    """Extract cwd sentinel, trim output, and append exit code."""
    output_lines: list[str] = []
    for line in stdout.split("\n"):
        if line.startswith(f"{sentinel}="):
            new_cwd = line[len(f"{sentinel}=") :]
            if new_cwd and Path(new_cwd).is_dir():
                state.bash_cwd = new_cwd
        else:
            output_lines.append(line)
    if stderr:
        output_lines.extend(stderr.split("\n"))
    out = _trim_bash_output(output_lines)
    if proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return out.strip() or "(no output)"
