"""Bash tool.

Parsing, matcher helpers, and the read-only command classifier all
live in :mod:`sagent.tools.lib.bash`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import asyncio
import atexit
import contextlib
import logging
import os
import secrets
import signal
import subprocess
import time

from sagent.agent.state import ToolState, get_tool_state
from sagent.lib.custom_json import bool_val, int_val, json_freeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    BashMatcher,
    Node,
    cached_parse_bash,
    is_read_only,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

# Shared by every write-capable Bash call, so the runtime coalesces them
# into one sequential group. The value is arbitrary; only its sameness
# matters -- ``_partition_cohort`` groups on key equality.
_WRITER_KEY: Final = "bash:writer"


def _suppress_oserror() -> contextlib.suppress:
    """OSError handler for process-group signals (race: proc may have exited)."""
    return contextlib.suppress(OSError)


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Bash:
    """Execute shell commands.

    The ``Tool`` protocol's fields are ``ClassVar`` so they stay off
    ``__init__``: a display knob sharing a name with ``description``
    would otherwise shadow the model-facing prompt on every instance and
    ship its own value to the provider.

    Attributes:
      peers: Sibling tools supplying ``bash_match`` lint hooks. Any peer
          whose class defines ``bash_match(self, trees) -> str | None``
          is registered as a lint source; its output is surfaced on the
          result's ``hint`` field, and the same nudges are prepended to
          ``content`` as a ``<system-reminder>`` so the model sees them.
      command_head_rows: Leading command rows kept. The first line of a
          shell command identifies it.
      command_tail_rows: Trailing command rows kept. The last line
          usually carries the pipe or redirect that says what comes back.
      command_lang: Pygments lexer for the command row.
      output: Whether the result body renders in the pane.
      output_head_rows: Leading body rows kept.
      output_tail_rows: Trailing body rows kept, after a ``⋯ N lines ⋯``
          marker. Head and tail together bound what the terminal prints.
      output_max_width: Cell width cap; ``0`` uses the pane width.
      output_wrap: ``wrap`` continues an over-wide line on the next row,
          ``chop`` keeps its head and marks the cut.

    """

    # Unannotated on purpose: an annotated name becomes a dataclass
    # FIELD, which puts it on ``__init__`` -- and a display knob sharing
    # a name would then shadow the model-facing prompt on every instance.
    # Without the annotation these stay plain class attributes, excluded
    # from the constructor and unshadowable.
    name = "Bash"
    tool_id = "application/x-tool-bash"
    description = _render_bash_description(load_tool_description("Bash"))
    clearable_results = True
    directive_schema = json_freeze(
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

    peers: Sequence[object] = ()
    command_head_rows: Annotated[int, CLI_SETTABLE] = 3
    command_tail_rows: Annotated[int, CLI_SETTABLE] = 1
    command_lang: Annotated[str, CLI_SETTABLE] = "bash"
    output: Annotated[Toggle, CLI_SETTABLE] = "on"
    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"

    def summary(self, args: Mapping[str, object]) -> str:
        """Return the header and input rows for this invocation.

        Line 0 is the header -- the model's own ``description`` argument,
        present on 96% of real calls. The rest is the command verbatim,
        which the renderer marks with ``⎿`` as the call's input.

        Newlines are preserved rather than folded to ``⏎``: a heredoc
        rendered as one long line is unreadable, and the renderer bounds
        the rows so a long script cannot flood the pane.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          label: Header line, then the command lines.

        """
        cmd = str(args.get("command", "")).replace("\r\n", "\n").replace("\r", "\n")
        cmd = cmd.replace("\t", "    ").strip("\n")
        desc = str(args.get("description", "")).strip()
        header = f"Bash {desc}".rstrip()
        return f"{header}\n{cmd}" if cmd else header

    @property
    def _peer_matchers(self) -> tuple[Callable[[Sequence[Node]], str | None], ...]:
        """Lint hooks contributed by sibling tools.

        Recomputed rather than cached: ``functools.cached_property``
        needs a ``__dict__``, which ``slots=True`` removes.
        """
        return tuple(
            peer.bash_match for peer in self.peers if isinstance(peer, BashMatcher)
        )

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          text: Supplemental prompt text; empty for Bash.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Serialize writers against each other; run reads in parallel.

        Bash has no static path to key on, so the key is the command's
        EFFECT: anything that cannot mutate state runs concurrently,
        and everything else queues behind the other writers in its
        cohort. Two concurrent writers race and the loser's edit is
        silently lost.

        Biased hard toward serializing. An unparseable command, an
        unrecognized utility, any redirect or control-flow construct all
        read as a writer -- a false positive costs sequential dispatch,
        a false negative costs data.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          key: ``None`` for a read-only command, else a shared constant
              so every writer in the cohort runs one at a time.

        """
        trees = cached_parse_bash(
            str(args.get("command", "")), get_tool_state().bash_parse_cache
        )
        if trees is not None and is_read_only(trees):
            return None
        return _WRITER_KEY

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
    # ``printf`` with a LEADING newline, not ``echo``: a command whose
    # stdout lacks a trailing newline otherwise fuses with the sentinel
    # on one line, which fails the prefix test below -- cwd tracking
    # silently stops and the raw token is shown to the model.
    tracked_cmd = f'trap \'printf "\\n%s=%s\\n" "{sentinel}" "$(pwd)"\' EXIT\n{command}'
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
    # ``communicate()`` is driven by a task we own, so a timeout cancels
    # the WAIT and not the read: the partial stdout/stderr collected
    # before the kill stays reachable. Re-awaiting a cancelled
    # ``communicate()`` instead returns only what buffered afterwards,
    # dropping the very output the drain exists to recover.
    comm = asyncio.ensure_future(proc.communicate())
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            asyncio.shield(comm),
            timeout=timeout_s,
        )
    except TimeoutError:
        await _kill_process_group(proc)
        stdout_bytes, stderr_bytes = await comm
        reason = f"timeout after {time.monotonic() - start:.1f}s"
    except asyncio.CancelledError:
        _ = comm.cancel()
        await _kill_process_group(proc)
        raise
    else:
        reason = ""
    return _process_output(
        proc,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        sentinel=sentinel,
        state=state,
        reason=reason,
    )


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


def _process_output(
    proc: asyncio.subprocess.Process,
    stdout: str,
    stderr: str,
    *,
    sentinel: str,
    state: ToolState,
    reason: str = "",
) -> str:
    """Extract the cwd sentinel, bound the output, append diagnostics.

    The single exit path for every outcome -- clean, failed, timed out --
    so the sentinel is stripped and the bound applied exactly once.

    Args:
      proc: The finished (or killed) process, read for its exit code.
      stdout: Decoded stdout, still carrying the cwd sentinel line.
      stderr: Decoded stderr.
      sentinel: Token the tracking trap echoes with the final cwd.
      state: Tool state whose ``bash_cwd`` the sentinel updates.
      reason: Non-empty when the run did not finish on its own, e.g.
          ``"timeout after 1.0s"``.

    Returns:
      text: Bounded body followed by any diagnostics.

    """
    body_lines: list[str] = []
    for line in stdout.split("\n"):
        if line.startswith(f"{sentinel}="):
            new_cwd = line[len(f"{sentinel}=") :]
            if new_cwd and Path(new_cwd).is_dir():
                state.bash_cwd = new_cwd
        else:
            body_lines.append(line)
    # Diagnostics are what says the command FAILED, so they are bounded
    # and appended separately: ``truncate`` keeps the head, so folding
    # them into the body first drops them on exactly the runs that
    # overflow, and a broken build then reads as a clean one. Each half
    # gets its own share of the cap, so neither can crowd out the other
    # and the total stays bounded however noisy one of them is.
    diagnostics = truncate(stderr.strip(), TOOL_RESULT_MAX_CHARS // 4)
    body_budget = TOOL_RESULT_MAX_CHARS - len(diagnostics)
    out = truncate("\n".join(body_lines).strip(), body_budget)
    if diagnostics:
        out = f"{out}\n{diagnostics}" if out else diagnostics
    if reason:
        out += f"\n[{reason}]"
    elif proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return out.strip() or "(no output)"
