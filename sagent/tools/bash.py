"""Bash tool.

Parsing, matcher helpers, and the read-only command classifier all
live in :mod:`sagent.tools.lib.bash`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import asyncio
import atexit
import contextlib
import functools
import io
import logging
import os
import secrets
import signal
import subprocess
import time

from treekle import from_plain, immutable

from sagent.agent.runtime import EXCLUSIVE_KEY
from sagent.agent.state import (
    ToolState,
    approx_tokens,
    get_tool_state,
)
from sagent.lib import debug_log
from sagent.tools.core import (
    bound_by_tokens,
    load_tool_description,
    result_token_budget,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import (
    BashMatcher,
    cached_parse_bash,
    is_read_only,
)
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from collections.abc import Callable

    from bashlex.ast import (
        node as Node,  # noqa: N812 -- PascalCase for the type name; bashlex spells it lowercase.
    )


logger = logging.getLogger(__name__)

BASH_DEFAULT_TIMEOUT_MS = 120_000  # house-ignore[globals] -- Default timeout dial.
BASH_MAX_TIMEOUT_MS = 600_000  # house-ignore[globals] -- Max timeout dial.

# Behind a wrapper this process cannot tell whether the shell's directory exists, so
# the shell's own script enters it: ``$2`` is the tool's directory, ``$3`` where the
# tool started.
_ENTER_CWD: Final = 'cd -- "$2" 2>/dev/null || cd -- "$3" 2>/dev/null || cd /\n'

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
      exec_prefix: Words placed before ``/bin/bash``, so a wrapper command can run
          the shell somewhere else, such as under another user or in a container.
          Empty runs the shell directly. The wrapper must run its arguments with
          its standard streams and exit status unchanged. Not a command-line
          setting.
      env: Environment of the spawned process; ``None`` inherits this process's.
          With a wrapper it is the wrapper's environment, so the shell sees what
          the wrapper carries across. Never shown in ``repr``. Not a command-line
          setting.
      drain_sec: Seconds a stopped call waits for the output of a process that
          outlives the stop, before it gives up on that output. Not a command-line
          setting.
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
    max_result_chars = 30_000
    directive_schema = immutable(
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
        },
    )

    peers: Sequence[object] = ()
    exec_prefix: Sequence[str] = ()
    # Left out of ``repr``: the environment may hold secrets, and a tool's repr ends
    # up in logs and tracebacks.
    env: Mapping[str, str] | None = field(default=None, repr=False)
    drain_sec: float = 1.0
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

    # Recomputed rather than cached: ``functools.cached_property`` needs a ``__dict__``,
    # which ``slots=True`` removes.
    @property
    def _peer_matchers(self) -> tuple[Callable[[Sequence[Node]], str | None], ...]:
        """Lint hooks contributed by sibling tools."""
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
        """Serialize writers against every keyed call; run reads in parallel.

        Bash has no static path to key on, so the key is the command's
        EFFECT: anything that cannot mutate state runs concurrently, and
        everything else queues behind every keyed call in its cohort --
        other writers and same-file Read/Edit/Write alike, since a ``sed
        -i`` may touch any file. Two concurrent writers race and the
        loser's edit is silently lost.

        Biased hard toward serializing. An unparseable command, an
        unrecognized utility, any redirect or control-flow construct all
        read as a writer -- a false positive costs sequential dispatch,
        a false negative costs data.

        Args:
          args: Parsed tool directive mapping.

        Returns:
          key: ``None`` for a read-only command, else ``EXCLUSIVE_KEY``.

        """
        trees = cached_parse_bash(
            str(args.get("command", "")),
            get_tool_state().bash_parse_cache,
        )
        if trees is not None and is_read_only(trees):
            return None
        return EXCLUSIVE_KEY

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the command and return the result.

        Cancellation propagates natively: ``asyncio.create_subprocess_exec``
        starts the bash process in its own session group, and on
        ``CancelledError`` we ``SIGTERM`` (then ``SIGKILL``) the whole
        group before re-raising. Behind an ``exec_prefix`` the group may be
        out of this process's reach, so the shell's own guard ends it instead
        (see :func:`_guard`).

        Args:
          args: Parsed tool directive mapping.

        Returns:
          result: ``ToolResult`` carrying stdout/stderr, the exit code,
            and any bash-lint nudges. Output is already bounded by the
            producers (``_process_output`` / ``_kill_and_drain``), which
            truncate the body before appending trailing diagnostics.

        """
        _register_exit_reaper()
        command = str(args.get("command", ""))
        # ``float``: the schema says ``number``, so ``1.5`` is a valid timeout.
        timeout = from_plain(
            args.get("timeout"),
            float,
            default=BASH_DEFAULT_TIMEOUT_MS,
        )
        run_as_fully_detached = from_plain(
            args.get("run_as_fully_detached"),
            bool,
            default=False,
        )
        state = get_tool_state()
        # A wrapped shell may sit where this process cannot look, so its script
        # judges the directory instead.
        if not self.exec_prefix:
            _ensure_valid_cwd(state)
        if run_as_fully_detached:
            text = _run_as_fully_detached(
                command,
                state=state,
                exec_prefix=self.exec_prefix,
                env=self.env,
            )
        else:
            text = await _run_foreground(
                command,
                state=state,
                timeout_s=_timeout_seconds(timeout),
                exec_prefix=self.exec_prefix,
                env=self.env,
                drain_sec=self.drain_sec,
            )
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

    # Every Bash call is traced, nudged or not. Whether the banner changes behaviour is
    # unmeasurable from the matchers alone -- replaying them over old sessions counts
    # shapes, not deliveries -- so the delivered rate and what follows it have to be
    # recorded as they happen.
    def _collect_nudges(self, command: str) -> list[str]:
        """Run peer ``bash_match`` matchers and return any nudges."""
        trees = cached_parse_bash(command, get_tool_state().bash_parse_cache)
        if trees is None:
            return []
        nudges = [
            n for matcher in self._peer_matchers if (n := matcher(trees)) is not None
        ]
        debug_log.trace("bash_nudge", nudged=bool(nudges), count=len(nudges))
        return nudges


def _ensure_valid_cwd(state: ToolState) -> None:
    """Reset ``state.bash_cwd`` to ``start_cwd`` (or ``/``) if it's gone."""
    if not Path(state.bash_cwd).is_dir():
        state.bash_cwd = state.start_cwd if Path(state.start_cwd).is_dir() else "/"


def _run_as_fully_detached(
    command: str,
    *,
    state: ToolState,
    exec_prefix: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> str:
    """Spawn an unmanaged detached background process."""
    # Behind a wrapper the shell enters its directory itself, as the foreground
    # path does. No guard: nothing waits on a detached process, so nothing could
    # tell it to stop.
    argv = (
        [
            *exec_prefix,
            "/bin/bash",
            "-c",
            f'{_ENTER_CWD}eval "$1"',
            "bash",
            command,
            state.bash_cwd,
            state.start_cwd,
        ]
        if exec_prefix
        else ["/bin/bash", "-c", command]
    )
    proc = subprocess.Popen(  # noqa: S603 -- The agent's own command runs through a fixed bash argv; that is this tool's purpose.
        argv,
        cwd="/" if exec_prefix else state.bash_cwd,
        env=env,
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


# Float, not floor division: ``// 1000`` mapped 1999ms to 1s -- half the requested
# budget -- and anything under 1000ms to 0, which the old ``max(1, ...)`` then raised to
# a full second, 1000x the request. The schema's ``minimum: 1`` makes a 1ms timeout
# legal, so it must survive.
def _timeout_seconds(timeout_ms: float) -> float:
    """Convert the directive's millisecond timeout to wait_for seconds."""
    return min(max(timeout_ms, 1), BASH_MAX_TIMEOUT_MS) / 1000


async def _run_foreground(
    command: str,
    *,
    state: ToolState,
    timeout_s: float,
    exec_prefix: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    drain_sec: float = 1.0,
) -> str:
    """Run a foreground bash command via :mod:`asyncio.subprocess`."""
    sentinel = f"__SAGENT_CWD_{secrets.token_hex(4)}__"
    # ``printf`` with a LEADING newline, not ``echo``: a command whose
    # stdout lacks a trailing newline otherwise fuses with the sentinel
    # on one line, which fails the prefix test below -- cwd tracking
    # silently stops and the raw token is shown to the model.
    report = f'printf "\\n%s=%s\\n" "{sentinel}" "$(pwd)"'
    # The trap is what survives an early ``exit``; the trailing statement is
    # what survives the command installing its own trap -- ``trap - EXIT``
    # silently disabled tracking outright, so the tool kept a stale cwd and
    # every later relative path resolved against the wrong directory. Two
    # identical sentinel lines are harmless: the reader applies each and they
    # agree.
    #
    # ``__rc`` then ``exit`` restores the command's own status: a bare
    # trailing statement made every run exit with printf's 0, so ``false``
    # and a failing build both reported success.
    #
    # ``eval "$1"`` with the command passed as an ARGUMENT, not spliced
    # into this text: bash reads a script line-wise, so a command ending
    # in a continuation (``echo hi \``) consumed the very next line and
    # printed ``__rc=0`` as its own output.
    tracked_cmd = f"trap '{report}' EXIT\neval \"$1\"\n__rc=$?\n{report}\nexit $__rc"
    # Behind a wrapper this process may be unable to signal the shell, so the wrapper
    # starts from ``/``, the shell's script enters its directory itself, and the shell
    # takes a pipe held here as its stdin. That pipe closing is how it hears that it
    # must stop.
    held = _HeldPipe() if exec_prefix else None
    try:
        # ``$0``: names the shell in ``$0`` and in its error prefixes, so a
        # syntax error still reads ``bash: ...`` as it did when the command was
        # the ``-c`` string itself.
        if held is None:
            argv = ["/bin/bash", "-c", tracked_cmd, "bash", command]
        else:
            argv = [
                *exec_prefix,
                "/bin/bash",
                "-c",
                _guard() + tracked_cmd,
                "bash",
                command,
                state.bash_cwd,
                state.start_cwd,
            ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL if held is None else held.reader,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=state.bash_cwd if held is None else "/",
            env=env,
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
            await _kill_process_group(proc, held=held)
            drained = await _drain(proc, comm, grace_sec=drain_sec)
            stdout_bytes, stderr_bytes = drained or (b"", b"")
            reason = f"timeout after {time.monotonic() - start:.1f}s"
            if drained is None:
                reason += "; the shell did not stop and its output is lost"
        except asyncio.CancelledError:
            _ = comm.cancel()
            await _kill_process_group(proc, held=held)
            raise
        else:
            reason = ""
            if held is not None:
                held.release()
    finally:
        if held is not None:
            held.close()
    return _process_output(
        proc,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        sentinel=sentinel,
        state=state,
        reason=reason,
        check_cwd=held is None,
    )


# The shell runs where this process may have no power to signal it -- ``killpg``
# answers EPERM across users -- so the only thing that reaches it is what it can see:
# its stdin, a pipe this process holds the write end of (see :class:`_HeldPipe`). A
# watcher inside the shell reads that pipe. One line means the call ended normally
# and it leaves; end-of-file with no line means the call timed out, was cancelled, or
# this process died, and it ends the process group: TERM, then KILL after
# ``kill_after_sec``. A killed caller therefore leaves no shell behind.
#
# The watcher is started from a second subshell so it is no child of the shell: a bare
# ``wait`` in the command would otherwise wait on it until the call timed out. It
# ignores TERM because it sits in the group it signals, and dying of its own TERM
# would skip the KILL that a command trapping TERM needs. It keeps the pipe on fd 9
# and the command gets /dev/null, as it does without a wrapper.
def _guard(*, kill_after_sec: float = 2.0) -> str:
    """Return the script prologue that lets the shell be stopped from outside."""
    return (
        f"{_ENTER_CWD}exec 9<&0 </dev/null\n"
        "( ( trap '' TERM; read -r _ <&9 ||"
        f" {{ kill -TERM 0; sleep {kill_after_sec:g}; kill -KILL 0; }} )"
        " >/dev/null 2>&1 & )\n"
        "exec 9<&-\n"
    )


class _HeldPipe:
    """A shell's stdin, whose write end this process keeps while the shell runs."""

    def __init__(self) -> None:
        reader, writer = os.pipe()
        self.reader = io.FileIO(reader)
        self._writer = io.FileIO(writer, "w")

    def release(self) -> None:
        """End the call normally: the one line that tells the guard to stand down."""
        _ = self._writer.write(b"\n")
        self._writer.close()

    def close(self) -> None:
        """Close both ends; with no line written first, the guard ends the group."""
        self._writer.close()
        self.reader.close()


# Keyed on the GROUP, never the leader: ``start_new_session`` makes the group id the
# leader's pid, and that group outlives the leader. Skipping the kill once the leader
# exited, or asking ``getpgid`` of a reaped leader, left a backgrounded descendant
# running with the pipe open, so the drain below never saw EOF.
#
# With ``held`` the group may be out of reach, so closing the pipe is the stop and this
# process waits at most ``wait_sec`` for the wrapper to end: longer than the guard's
# grace, and never forever.
async def _kill_process_group(
    proc: asyncio.subprocess.Process,
    *,
    held: _HeldPipe | None = None,
    wait_sec: float = 5.0,
) -> None:
    """SIGTERM then SIGKILL the process group; await the leader's reaping."""
    if held is not None:
        held.close()
        try:
            _ = await asyncio.wait_for(proc.wait(), timeout=wait_sec)
        except TimeoutError:
            # Without this the drain would wait on the wrapper as long as it lives.
            logger.warning("Shell outlived its stop; killing wrapper pid=%d", proc.pid)
            with _suppress_oserror():
                proc.kill()
        return
    with _suppress_oserror():
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        _ = await asyncio.wait_for(proc.wait(), timeout=0.5)
    with _suppress_oserror():
        os.killpg(proc.pid, signal.SIGKILL)
    _ = await proc.wait()


# A descendant that left the group (its own ``setsid``) survives the group kill and
# can hold the pipe open forever, so the drain is bounded and then the pipes closed.
# ``communicate`` ends in a wait for the process, so with the pipes closed the only
# thing left to wait for is a process that may be out of reach (see :func:`_guard`),
# and that wait is bounded too. Its output is lost with it, because
# ``communicate`` returns the output only together with the exit.
async def _drain(
    proc: asyncio.subprocess.Process,
    comm: asyncio.Future[tuple[bytes, bytes]],
    *,
    grace_sec: float = 1.0,
) -> tuple[bytes, bytes] | None:
    """Collect what the killed command wrote; None if its process would not end."""
    try:
        return await asyncio.wait_for(asyncio.shield(comm), timeout=grace_sec)
    except TimeoutError:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.feed_eof()
    try:
        return await asyncio.wait_for(comm, timeout=grace_sec)
    except TimeoutError:
        logger.warning("Shell outlived its stop; giving up on its output")
        return None


# The single exit path for every outcome -- clean, failed, timed out -- so the sentinel
# is stripped and the bound applied exactly once.
def _process_output(
    proc: asyncio.subprocess.Process,
    stdout: str,
    stderr: str,
    *,
    sentinel: str,
    state: ToolState,
    reason: str = "",
    check_cwd: bool = True,
) -> str:
    """Extract the cwd sentinel, bound the output, append diagnostics."""
    body_lines: list[str] = []
    for line in stdout.split("\n"):
        if line.startswith(f"{sentinel}="):
            new_cwd = line[len(f"{sentinel}=") :]
            if new_cwd and (not check_cwd or Path(new_cwd).is_dir()):
                state.bash_cwd = new_cwd
        else:
            body_lines.append(line)
    # Diagnostics are what says the command FAILED, so they are bounded
    # and appended separately: ``truncate`` keeps the head, so folding
    # them into the body first drops them on exactly the runs that
    # overflow, and a broken build then reads as a clean one. Each half
    # gets its own share of the cap, so neither can crowd out the other
    # and the total stays bounded however noisy one of them is.
    budget = result_token_budget()
    diagnostics, _ = bound_by_tokens(
        (f"{line}\n" for line in stderr.strip().split("\n")),
        budget=budget // 4,
    )
    diagnostics = diagnostics.strip()
    body, kept = bound_by_tokens(
        (f"{line}\n" for line in body_lines),
        budget=max(1, budget - approx_tokens(diagnostics)),
    )
    out = body.strip()
    if kept < len(body_lines):
        out += f"\n... ({len(body_lines) - kept} more lines omitted; budget reached)"
    if diagnostics:
        out = f"{out}\n{diagnostics}" if out else diagnostics
    if reason:
        out += f"\n[{reason}]"
    elif proc.returncode != 0:
        out += f"\n[exit code: {proc.returncode}]"
    return out.strip() or "(no output)"


def _suppress_oserror() -> contextlib.suppress:
    """OSError handler for process-group signals (race: proc may have exited)."""
    return contextlib.suppress(OSError)


# A child that finished after the last :func:`reap_background_processes` call is
# otherwise garbage-collected by ``Popen.__del__`` while still unwaited, emitting a
# spurious ``ResourceWarning`` at exit -- seen under ``pytest -n`` worker teardown. A
# final ``poll()`` of each handle reaps every finished child (setting its
# ``returncode``), which is exactly the condition ``Popen.__del__`` checks, so no
# finished child warns.
#
# A child still genuinely running at exit is a different case: clearing the list drops
# the last reference, so ``Popen.__del__`` runs and emits ``ResourceWarning`` for it.
# That warning is accurate -- a ``start_new_session`` child is being left running past
# this process -- and is not suppressed here.
def _reap_at_exit() -> None:
    """Reap retained detached children at interpreter shutdown."""
    for proc in _BACKGROUND_PROCESSES:
        with contextlib.suppress(Exception):
            proc.poll()
    _BACKGROUND_PROCESSES.clear()


@functools.cache
def _register_exit_reaper() -> None:
    """Register detached-child cleanup when Bash is first used."""
    atexit.register(_reap_at_exit)
