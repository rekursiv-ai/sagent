"""Bash tool.

Parsing, matcher helpers, and the read-only command classifier all
live in :mod:`sagent.tools.lib.bash`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import asyncio
import atexit
import contextlib
import logging
import os
import re
import secrets
import signal
import subprocess
import time

from sagent.agent.state import ToolState, get_tool_state
from sagent.lib.custom_json import JSON, bool_val, int_val, json_freeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.lib.bash import Node, cached_parse_bash
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

# Matches ``[exit code: N]`` appended by ``_run`` on non-zero exits.
_BASH_EXIT_RE = re.compile(r"(?:^|\n)\[exit code: (\d+)\]\s*$")


@runtime_checkable
class _BashMatchPeer(Protocol):
    """Duck-type for sibling tools that provide a bash-lint hook."""

    def bash_match(self, trees: Sequence[Node]) -> str | None: ...


def _suppress_oserror() -> contextlib.suppress:
    """OSError handler for process-group signals (race: proc may have exited)."""
    return contextlib.suppress(OSError, ProcessLookupError)


BASH_DEFAULT_TIMEOUT_MS = 120_000  # config-globals: ignore -- default timeout dial
BASH_MAX_TIMEOUT_MS = 600_000  # config-globals: ignore -- max timeout dial

# Kept-alive references to background subprocesses so their Popen
# objects don't get garbage-collected mid-run.
_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


def reap_background_processes() -> None:
    """Reap completed detached Bash children retained for lifetime tracking.

    For each retained child, ``poll()`` collects its exit status if it has
    finished. Finished children are then dropped from the retention list;
    crucially the ``poll()`` itself reaps the OS child, so the ``Popen``
    object is never garbage-collected with an unreaped process -- the
    condition that makes ``Popen.__del__`` emit a ``ResourceWarning``.
    Still-running children stay retained.
    """
    _BACKGROUND_PROCESSES[:] = [p for p in _BACKGROUND_PROCESSES if p.poll() is None]


def _reap_at_exit() -> None:
    """Reap retained detached children at interpreter shutdown.

    A child that finished after the last
    :func:`reap_background_processes` call is otherwise garbage-collected
    by ``Popen.__del__`` while still unwaited, emitting a spurious
    ``ResourceWarning`` at exit -- seen under ``pytest -n`` worker
    teardown. A final ``poll()`` of each handle reaps every finished
    child (setting its ``returncode``), which is exactly the condition
    ``Popen.__del__`` checks, so no finished child warns.

    A child still genuinely running at exit is a different case: clearing
    the list drops the last reference, so ``Popen.__del__`` runs and emits
    ``ResourceWarning`` for it. That warning is accurate -- a
    ``start_new_session`` child is being left running past this process --
    and is not suppressed here.
    """
    for proc in _BACKGROUND_PROCESSES:
        with contextlib.suppress(Exception):
            proc.poll()
    _BACKGROUND_PROCESSES.clear()


atexit.register(_reap_at_exit)


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


class Bash:
    """Execute shell commands."""

    name: str = "Bash"
    tool_id: str = "application/x-tool-bash"
    description: str = _render_bash_description(load_tool_description("Bash"))
    clearable_results: bool = True
    emit_tool_summary: bool = False
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
            },
            "required": ["command"],
            # ``run_as_fully_detached`` is a deliberate Python-only escape
            # hatch (orphan subprocesses); strict additionalProperties keeps
            # the LLM from invoking it via tool calls.
            "additionalProperties": False,
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
        is registered as a lint source; its output is surfaced via the
        result's ``hint`` field for renderer-only display, AND the same
        nudges are prepended to ``content`` as a ``<system-reminder>``
        block so the model sees them too.
        """
        self._peer_matchers: tuple[Callable[[Sequence[Node]], str | None], ...] = tuple(
            peer.bash_match for peer in peers if isinstance(peer, _BashMatchPeer)
        )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this tool invocation.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          label: Compact one-line label for renderer display.

        """
        cmd = str(args.get("command", ""))
        cmd = cmd.replace("\r\n", "⏎").replace("\n", "⏎").replace("\r", "⏎")
        cmd = cmd.replace("\t", " ")
        return f"Bash {cmd}" if cmd else "Bash"

    def summary_result(self, result: ToolResult) -> str | None:
        """Return a one-line receipt: line count plus nonzero exit code.

        Args:
          result: The completed ``ToolResult``.

        Returns:
          receipt: Line-count receipt (with exit code on nonzero), or
            ``None`` when summaries are disabled or the result is an error.

        """
        if not self.emit_tool_summary or result.is_error:
            return None
        text = result.content
        exit_match = _BASH_EXIT_RE.search(text)
        body = text[: exit_match.start()] if exit_match else text
        lines = body.count("\n") + (0 if body.endswith("\n") or not body else 1)
        if exit_match:
            return f"{lines}L · exit {exit_match.group(1)}"
        return f"{lines}L"

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          text: Supplemental prompt text; empty for Bash.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: Bash has no static path to serialize on."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the command and return the result.

        Cancellation propagates natively: ``asyncio.create_subprocess_exec``
        starts the bash process in its own session group, and on
        ``CancelledError`` we ``SIGTERM`` (then ``SIGKILL``) the whole
        group before re-raising.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: ``ToolResult`` carrying stdout/stderr, the exit code,
            and any bash-lint nudges. Output is already bounded by the
            producers (``_process_output`` / ``_kill_and_drain``), which
            truncate the body before appending trailing diagnostics.

        """
        command = str(args.get("command", ""))
        timeout = int_val(args.get("timeout"), BASH_DEFAULT_TIMEOUT_MS)
        run_as_fully_detached = bool_val(
            args.get("run_as_fully_detached"),
            False,
        )
        state = get_tool_state()
        _ensure_valid_cwd(state)
        if run_as_fully_detached:
            text = _run_as_fully_detached(command, state=state)
        else:
            timeout_s = max(1, min(int(timeout) // 1000, BASH_MAX_TIMEOUT_MS // 1_000))
            text = await _run_foreground(command, state=state, timeout_s=timeout_s)
        body = text
        if not self._peer_matchers:
            return ToolResult(call_id="", content=body)
        nudges = self._collect_nudges(command)
        if not nudges:
            return ToolResult(call_id="", content=body)
        # Bake the banner into ``content`` so the model sees it, and
        # surface the same nudges on ``hint`` for the renderer.
        banner_body = "\n".join(f"[bash-lint] {h}" for h in nudges)
        banner = f"<system-reminder>\n{banner_body}\n</system-reminder>"
        return ToolResult(
            call_id="",
            content=f"{banner}\n\n{body}",
            hint="\n".join(nudges),
        )

    def _collect_nudges(self, command: str) -> list[str]:
        """Run peer ``bash_match`` matchers and return any nudges."""
        trees = cached_parse_bash(command, get_tool_state().bash_parse_cache)
        if trees is None:
            return []
        nudges: list[str] = []
        for matcher in self._peer_matchers:
            nudge = matcher(trees)
            if nudge:
                nudges.append(nudge)
        return nudges


def _ensure_valid_cwd(state: ToolState) -> None:
    """Reset ``state.bash_cwd`` to ``start_cwd`` (or ``$HOME``) if it's gone."""
    if not Path(state.bash_cwd).is_dir():
        state.bash_cwd = (
            state.start_cwd if Path(state.start_cwd).is_dir() else str(Path.home())
        )


def _run_as_fully_detached(command: str, *, state: ToolState) -> str:
    """Spawn an unmanaged detached background process."""
    proc = subprocess.Popen(  # noqa: S603 -- trusted fixed argv, not user input
        ["/bin/bash", "-c", command],
        cwd=state.bash_cwd,
        # Bash must not share the REPL terminal; interactive children
        # otherwise steal prompt-toolkit keystrokes from the user.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    reap_background_processes()
    _BACKGROUND_PROCESSES.append(proc)
    logger.info("Fully detached process started: pid=%d", proc.pid)
    return f"(running in background, pid={proc.pid})"


async def _run_foreground(command: str, *, state: ToolState, timeout_s: int) -> str:
    """Run a foreground bash command via :mod:`asyncio.subprocess`."""
    sentinel = f"__SAGENT_CWD_{secrets.token_hex(4)}__"
    tracked_cmd = f"trap 'echo {sentinel}=$(pwd)' EXIT\n{command}"
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        tracked_cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=state.bash_cwd,
        start_new_session=True,
    )
    start = time.monotonic()
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
    except TimeoutError:
        return await _kill_and_drain(proc, start=start, reason="timeout")
    except asyncio.CancelledError:
        await _kill_process_group(proc)
        raise
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return _process_output(proc, stdout, stderr, sentinel=sentinel, state=state)


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM then SIGKILL the process group; await reaping each step."""
    if proc.returncode is not None:
        return
    with _suppress_oserror():
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        _ = await asyncio.wait_for(proc.wait(), timeout=0.5)
    except TimeoutError:
        with _suppress_oserror():
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        _ = await proc.wait()


async def _kill_and_drain(
    proc: asyncio.subprocess.Process,
    *,
    start: float,
    reason: str,
) -> str:
    """Kill the process group, drain remaining output, format reason line."""
    await _kill_process_group(proc)
    stdout_bytes, stderr_bytes = await proc.communicate()
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    elapsed = time.monotonic() - start
    body = truncate(stdout.strip(), TOOL_RESULT_MAX_CHARS)
    if stderr:
        body += f"\n{stderr.strip()}"
    return body + f"\n[{reason} after {elapsed:.1f}s]"


def _process_output(
    proc: asyncio.subprocess.Process,
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
    # Truncate stdout, THEN append the diagnostics -- never the other way
    # round. ``truncate`` keeps the head, so folding stderr and the exit
    # code into the body first drops both on any oversized run and makes a
    # failed command read as a successful one.
    out = truncate("\n".join(output_lines).strip(), TOOL_RESULT_MAX_CHARS)
    if stderr:
        out += f"\n{stderr.strip()}"
    if proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return out.strip() or "(no output)"
