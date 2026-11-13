"""Bash: run a command in the agent's working directory.

A ``cd`` in the command moves the agent's ``cwd`` for later tools. A command
that times out, or whose call is cancelled, is stopped with its process group.
The first lines of output are streamed as the command prints them. A long
output is truncated in the middle, so the rest is streamed when the command
ends.

Typical usage example:

  streaming_result = Bash().call(
      context,
      {"command": "pytest -q", "description": "Run the tests", "timeout": 60_000},
  )
  result = await settled(streaming_result)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

import asyncio
import contextlib
import io
import logging
import os
import secrets
import signal
import time

from sagent.lib.posix.child import Child, finish
from sagent.session.types import BashResult
from sagent.tools.command_analyzer import (
    ReadOnly,
    Rewrite,
    ShellTool,
    moves,
    rewrite,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import (
    Arg,
    Stream,
    ToolStreamingResult,
    json_schema,
    parse,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Bash",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Bash:
    """Run a shell command."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        command: Annotated[str, "The command to execute."]
        description: Annotated[str, "What this command does, in active voice."]
        timeout: Annotated[
            float,
            "Kill the command after this many milliseconds.",
            Arg(minimum=1, maximum=600_000),
        ] = 600_000

    name = "Bash"
    description = (files(__package__) / "assets" / "bash.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    display: ToolDisplay = field(
        default=ToolDisplay(
            command=DisplaySpec(lines=(3, 1)),
            output=DisplaySpec(),
            command_language="bash",
        ),
    )

    exec_prefix: Sequence[str] = ()
    """A wrapper before ``/bin/bash``, such as ``sudo -u agent``; it must pass streams through."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """The environment, or ``None`` to inherit; kept out of ``repr`` for secrets. A
    ``COLUMNS`` here wins over the context's."""

    drain_sec: float = 1.0
    """How long to wait for a stopped command's output."""

    grace_sec: float = 0.5
    """How long an unwrapped command has, after TERM, to close its output."""

    read_only: ReadOnly = field(default_factory=ReadOnly.default)
    """The commands that only read; ``rewrite`` may turn a line of them into ``peers``' calls."""

    peers: Sequence[ShellTool] = ()
    """Tools that ``rewrite`` may hand a command to, such as Read for ``cat``."""

    max_result_chars: int = 30_000
    """How much text one result may hold, if below the context's."""

    nice: int | None = 10
    """How far below this program's CPU priority commands run, so a build defers to
    the desktop; ``None`` runs them at its own."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Bash <description>``, then the command's lines."""
        header = f"Bash {str(arguments.get('description', '')).strip()}".rstrip()
        command = str(arguments.get("command", ""))
        command = command.replace("\r\n", "\n").strip("\n")
        return f"{header}\n{command}" if command else header

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``"*"`` for a command that may move the shell, else ``None``.

        Commands run side by side, and beside every other call, as a shell's
        jobs do; only Read, Write, and Edit keep an order, by file. Every call
        made after a command that may move the shell waits for it, since those
        calls start in the directory where it ends.

        Args:
          context: The calling agent's state; unused.
          arguments: What the model passed to Bash.

        Returns:
          key: ``"*"`` or ``None``.

        """
        del context
        command = arguments.get("command")
        return "*" if not isinstance(command, str) or moves(command) else None

    def rewrite(
        self,
        arguments: Mapping[str, PlainTree],
    ) -> Rewrite | tuple[Rewrite, ...] | str | None:
        """Return the calls peers make instead of this one, if the command is theirs.

        The agent applies this before recording the call, so the model's
        history shows ``Read`` where it wrote ``cat``, and a Read and a Grep
        where it wrote ``sed -n 1,9p a.py; grep -n x b.py``.

        Args:
          arguments: What the model passed to Bash.

        Returns:
          result: A peer's call to make instead; several, for a line that only
            reads; a message refusing a command that is other tools' work but
            not as calls made apart; or ``None`` to run it.

        """
        command = arguments.get("command")
        if not isinstance(command, str):
            return None
        return rewrite(
            command,
            peers=self.peers,
            reads=self.read_only.reads(command),
        )

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[BashResult, str, str]:
        """Run ``command`` in ``context.cwd``, and move it where the shell ends.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Bash.Arguments``.

        Returns:
          streaming_result: The command's output as it is printed, and a
            ``BashResult`` with its exit code; the exit code is ``None`` when
            the shell did not stop.

        """
        stdout, stderr = Stream[str](), Stream[str]()
        return ToolStreamingResult(
            result=asyncio.ensure_future(self._run(context, arguments, stdout, stderr)),
            stdout=stdout,
            stderr=stderr,
        )

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
        stdout: Stream[str],
        stderr: Stream[str],
    ) -> BashResult:
        """Run the command and return its result; close both streams on any exit."""
        try:
            a = parse(arguments, Bash.Arguments)
            return BashResult(
                command=("bash", "-c", a.command),
                exit_code=await self._run_command(context, a, stdout, stderr),
            )
        finally:
            stdout.close()
            stderr.close()

    async def _run_command(
        self,
        context: ToolContext,
        a: Bash.Arguments,
        stdout: Stream[str],
        stderr: Stream[str],
    ) -> int | None:
        """Run ``a.command``, write its output to the streams, and return its exit code."""
        cwd = context.cwd
        # A wrapped shell's directory may be invisible here; ``_guard`` falls back.
        if not self.exec_prefix:
            cwd = await asyncio.to_thread(_existing, cwd)
            if cwd != context.cwd:
                context.cwd = cwd
        sentinel = f"__SAGENT_CWD_{secrets.token_hex()}__"
        script = _script(sentinel)
        held = _HeldPipe() if self.exec_prefix else None
        argv = (
            ["/bin/bash", "-c", script, "bash", a.command]
            if held is None
            else [
                *self.exec_prefix,
                "/bin/bash",
                "-c",
                _guard() + script,
                "bash",
                a.command,
                str(cwd),
            ]
        )
        cap = min(self.max_result_chars, context.max_result_chars)
        spill = (
            context.spill_dir / f"bash-{secrets.token_hex(4)}.txt"
            if context.spill_dir is not None
            else None
        )
        note = f"; full output: {spill}" if spill is not None else ""
        stdout_lines = _StdoutLines(
            writer=_EarlyWriter(
                stream=stdout,
                early_chars=_early_chars(cap, note=note),
            ),
            report_prefix=f"{sentinel}=",
        )
        # Stderr gets its own quarter, so floods of stdout cannot hide the error.
        stderr_cap = cap // 4
        stderr_writer = _EarlyWriter(stream=stderr, early_chars=stderr_cap)
        started = time.monotonic()
        reason: str | None = None
        pipes: list[_OutputPipe] = []
        try:
            pipes.append(await _OutputPipe.open())
            pipes.append(await _OutputPipe.open())
            child = await Child.start(
                *argv,
                group=True,
                grace_sec=self.grace_sec,
                nice=self.nice,
                stdin=asyncio.subprocess.DEVNULL if held is None else held.reader,
                stdout=pipes[0].writer,
                stderr=pipes[1].writer,
                cwd=cwd if held is None else "/",
                env=_environment(self.env, columns=context.columns),
            )
            # The shell has its own copies: one held here would keep each pipe
            # from ever reaching its end.
            for pipe in pipes:
                pipe.writer.close()
            # Shielded, so a timeout cancels the wait, not the read of what was written.
            read_task = asyncio.ensure_future(
                _read_output(
                    child,
                    (pipes[0].reader, stdout_lines.add_line),
                    (pipes[1].reader, stderr_writer.add_line),
                ),
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(read_task),
                    timeout=timedelta(milliseconds=a.timeout).total_seconds(),
                )
            except TimeoutError:
                await finish(_stop(child, held=held, output=read_task))
                reason = f"timeout after {time.monotonic() - started:.1f}s"
                if not await _drain(pipes, read_task, grace_sec=self.drain_sec):
                    reason += (
                        "; the shell did not stop, and the rest of its output is lost"
                    )
            except asyncio.CancelledError:
                try:
                    await finish(_stop(child, held=held, output=read_task))
                finally:
                    _ = read_task.cancel()
                raise
            else:
                if held is not None:
                    held.release()
        finally:
            for pipe in pipes:
                pipe.close()
            if held is not None:
                held.close()
        stdout_lines.finish()
        # The script reports the directory before the command, then after it. Commands
        # run side by side, so one that stayed put must not undo another's ``cd``.
        # A wrapped shell is compared with the directory asked for instead, since
        # ``_guard`` starts it in ``/`` when that directory is gone.
        reported_dirs = stdout_lines.reported_dirs
        start_dirs = [str(cwd)] if held is not None else reported_dirs[:1]
        for reported_dir in reported_dirs:
            end_cwd = Path(reported_dir)
            if (
                reported_dir not in start_dirs
                and end_cwd.is_absolute()
                and (held is not None or await asyncio.to_thread(end_cwd.is_dir))
            ):
                context.cwd = end_cwd
        diagnostics, _ = context.fit(
            [f"{line}\n" for line in "".join(stderr_writer.lines).strip().split("\n")],
            cap=stderr_cap,
        )
        diagnostics = diagnostics.strip()
        if reason is not None:
            diagnostics += f"\n[{reason}]"
        lines = stdout_lines.writer.lines
        final_text, truncated = _final_text(
            context,
            lines,
            reserve=len(diagnostics),
            cap=cap,
            note=note,
        )
        if truncated and spill is not None:
            await asyncio.to_thread(_spill, spill, lines)
        stdout_lines.writer.write_rest(final_text)
        stderr_writer.write_rest(diagnostics.strip())
        return child.process.returncode


@dataclass(slots=True, kw_only=True)
class _EarlyWriter:
    """Writes a stream's first lines as they arrive, and the rest at the end.

    The final text is known only when the command ends: it is stripped, and a
    long one is truncated in the middle. A line is written early only when the
    final text is certain to contain it: a later non-blank line has arrived,
    and the output so far is within ``early_chars``.
    """

    stream: Stream[str]
    early_chars: int
    """How many characters from the start the final text always keeps."""

    lines: list[str] = field(default_factory=list[str])
    """Every line added, each with its newline."""

    _written: list[str] = field(default_factory=list[str])
    _next_unwritten: int = 0
    """Index in ``lines`` of the first line neither written nor skipped as a leading blank."""

    _chars_added: int = 0

    def add_line(self, line: str) -> None:
        """Add the next line and write the earlier lines that are now certain.

        Args:
          line: The next line of output, without its newline.

        """
        self.lines.append(f"{line}\n")
        self._chars_added += len(line) + 1
        if not line.strip() or self._chars_added > self.early_chars:
            return
        newest = len(self.lines) - 1
        for earlier in self.lines[self._next_unwritten : newest]:
            # The final text is stripped, so a leading blank line is never written.
            text = earlier if self._written else earlier.lstrip()
            if text:
                self._written.append(text)
                self.stream.write(text)
        self._next_unwritten = newest

    def write_rest(self, final_text: str) -> None:
        """Write the part of ``final_text`` that was not written early."""
        rest = final_text.removeprefix("".join(self._written))
        for line in rest.splitlines(keepends=True):
            self.stream.write(line)


@dataclass(slots=True, kw_only=True)
class _StdoutLines:
    """Separates the command's output lines from the script's directory reports."""

    writer: _EarlyWriter
    report_prefix: str
    """What a report line starts with: the sentinel and ``=``."""

    reported_dirs: list[str] = field(default_factory=list[str])
    """Each directory the script reported; the last is where the shell ended."""

    _pending_line: str | None = None

    # A line is classified only once the next line is known, so it waits here.
    def add_line(self, line: str) -> None:
        """Add the next line of standard output, given without its newline."""
        if self._pending_line is not None:
            self._route(self._pending_line, next_line=line)
        self._pending_line = line

    def finish(self) -> None:
        """Route the last line, which no line follows."""
        if self._pending_line is not None:
            self._route(self._pending_line, next_line=None)
        self._pending_line = None

    def _route(self, line: str, *, next_line: str | None) -> None:
        # The report's leading newline and the stream's final newline each read
        # as an empty line the command never printed.
        if not line and (next_line is None or next_line.startswith(self.report_prefix)):
            return
        if line.startswith(self.report_prefix):
            self.reported_dirs.append(line.removeprefix(self.report_prefix))
        else:
            self.writer.add_line(line)


# By chunk, not ``readline``: a line longer than the reader's limit raises there.
async def _read_lines(
    reader: asyncio.StreamReader,
    add_line: Callable[[str], None],
) -> None:
    """Call ``add_line`` with each line of ``reader``, without its newline."""
    partial = b""
    while chunk := await reader.read(1 << 16):
        *complete, partial = (partial + chunk).split(b"\n")
        for line in complete:
            add_line(line.decode(errors="replace"))
    add_line(partial.decode(errors="replace"))


async def _read_output(
    child: Child,
    *readers: tuple[asyncio.StreamReader, Callable[[str], None]],
) -> None:
    """Read each stream to its end, line by line, then wait for the shell to exit."""
    _ = await asyncio.gather(
        *(_read_lines(reader, add_line) for reader, add_line in readers),
    )
    _ = await child.exited()


# A child that ran ``setsid`` survives the group kill and can hold the pipes open.
async def _drain[T](
    pipes: Sequence[_OutputPipe],
    read_task: asyncio.Future[T],
    *,
    grace_sec: float,
) -> bool:
    """Return whether the stopped command's output was read to its end."""
    try:
        await asyncio.wait_for(asyncio.shield(read_task), timeout=grace_sec)
    except TimeoutError:
        # Closing a pipe here ends its read, with what was read already kept.
        for pipe in pipes:
            pipe.close()
    else:
        return True
    try:
        await asyncio.wait_for(read_task, timeout=grace_sec)
    except TimeoutError:
        logging.getLogger(__name__).warning("Shell outlived its stop; output lost")
        return False
    return True


@dataclass(frozen=True, slots=True, kw_only=True)
class _OutputPipe:
    """A pipe the shell writes to, whose read end this process closes itself.

    asyncio closes a subprocess pipe of its own only at the pipe's end, or when
    the process object is collected. A job that left the shell's group keeps
    the pipe from ending, so its descriptor would stay open here until then.
    """

    reader: asyncio.StreamReader
    transport: asyncio.ReadTransport
    writer: io.FileIO
    """The end the shell inherits."""

    @classmethod
    async def open(cls) -> Self:
        """Open a pipe whose read end the running loop reads.

        Returns:
          pipe: The pipe, both ends open.

        """
        read_fd, write_fd = os.pipe()
        read_end, writer = io.FileIO(read_fd), io.FileIO(write_fd, "w")
        reader = asyncio.StreamReader()
        try:
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader),
                read_end,
            )
        except BaseException:
            read_end.close()
            writer.close()
            raise
        return cls(reader=reader, transport=transport, writer=writer)

    def close(self) -> None:
        """Close both ends; the reader then ends after what it has read."""
        self.writer.close()
        self.transport.close()


class _HeldPipe:
    """A wrapped shell's stdin, whose write end is held while the shell runs."""

    def __init__(self) -> None:
        reader, writer = os.pipe()
        self.reader = io.FileIO(reader)
        self._writer = io.FileIO(writer, "w")

    def release(self) -> None:
        """Tell the guard the call ended normally."""
        self._writer.write(b"\n")
        self._writer.close()

    def close(self) -> None:
        """Close both ends; without ``release`` first, the guard ends the group."""
        self._writer.close()
        self.reader.close()


# The report runs on every way out: the EXIT trap after an early ``exit`` or a
# failure under ``set -e``, and the last line after a normal end. A command's own
# ``trap ... EXIT`` replaces the trap, so ``trap`` is wrapped to put the report
# back in front of the command's handler; ``$?`` passes through to it unchanged.
# Only the shell's own trap: a subshell's ``cd`` never moved the shell.
# ``eval "$1"`` stops a trailing ``\\`` from joining the next line, and
# ``exit $__rc`` keeps the command's status.
def _script(sentinel: str) -> str:
    """Return the script that runs the command ``$1`` and reports where it ended."""
    # The leading newline keeps output without a final one off the report.
    report = f'printf "\\n%s=%s\\n" "{sentinel}" "$(pwd)"'
    return (
        f"__sagent_report() {{ local rc=$?; {report}; return $rc; }}\n"
        "builtin trap __sagent_report EXIT\n"  # pragma: no mutate -- bash reads trap signal names case-insensitively.
        "trap() {\n"
        '  builtin trap "$@" || return\n'
        '  [ "$BASH_SUBSHELL" = 0 ] || return 0\n'
        "  local line\n"
        "  line=$(builtin trap -p EXIT)\n"  # pragma: no mutate -- bash reads trap signal names case-insensitively.
        '  case "$line" in *__sagent_report*) return 0 ;; esac\n'
        '  eval "set -- $line"\n'
        '  builtin trap -- "__sagent_report; ${3:-}" EXIT\n'  # pragma: no mutate -- bash reads trap signal names case-insensitively.
        "}\n"
        "__sagent_report\n"
        'eval "$1"\n'
        "__rc=$?\n"
        "__sagent_report\n"
        "exit $__rc"
    )


# ``_ends`` gives the start at least two fifths of the room. The room is ``cap``
# less the largest diagnostics (a quarter of ``cap`` plus one line of why the
# command stopped) and less the line that counts the omitted lines.
def _early_chars(cap: int, *, note: str) -> int:
    """Return how many characters from the output's start the final text always keeps."""
    room = cap - (cap // 4 + 160) - len(_omitted(10**18, note=note))
    return max(0, room) * 2 // 5


def _final_text(
    context: ToolContext,
    lines: Sequence[str],
    *,
    reserve: int,
    cap: int,
    note: str,
) -> tuple[str, bool]:
    """Return the text the model reads for ``lines``, and whether it was truncated."""
    body, kept = context.fit(lines, reserve=reserve, cap=cap)
    truncated = kept < len(lines)
    if truncated:
        body = _ends(context, lines, reserve=reserve, cap=cap, note=note)
    return (f"{body.strip()}\n" if body.strip() else ""), truncated


# A failure prints last, so the end is kept as well as the start: the end gets the
# larger share, but no line too long for it, and the start keeps at least part of
# its first line. Room is left for the longest the cut can say: every line.
def _ends(
    context: ToolContext,
    lines: Sequence[str],
    *,
    reserve: int,
    cap: int,
    note: str,
) -> str:
    """Return the start and end of ``lines``, and how many lines between are left out."""
    reserve += len(_omitted(len(lines), note=note))
    room = min(cap, context.max_result_chars) - reserve
    tail_room = room * 3 // 5
    _, tail = context.fit(reversed(lines[1:]), cap=max(1, tail_room))
    if len(lines[-1]) > tail_room:
        tail = 0
    end = "".join(lines[len(lines) - tail :])
    reserve += len(end)
    # A first line cut short says so, past the room ``fit`` was given.
    if len(lines[0]) > room - len(end):
        reserve += len(f"\n... ({len(lines[0]):,} chars omitted)")
    head, kept = context.fit(lines[: len(lines) - tail], reserve=reserve, cap=cap)
    return f"{head.rstrip()}{_omitted(len(lines) - kept - tail, note=note)}{end}"


def _omitted(count: int, *, note: str) -> str:
    """Return the line between a cut output's start and end: how many lines it left out."""
    return f"\n... ({count:,} lines omitted{note})\n"


def _spill(path: Path, stdout: Sequence[str]) -> None:
    """Write the whole of ``stdout`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(stdout).encode())


def _environment(
    env: Mapping[str, str] | None,
    *,
    columns: int | Callable[[], int],
) -> dict[str, str]:
    """Return ``env``, or this process's, with ``COLUMNS`` unless ``env`` sets it."""
    shown = {"COLUMNS": str(columns() if callable(columns) else columns)}
    # An inherited COLUMNS is the terminal's, not where the output is shown.
    return {**os.environ, **shown} if env is None else {**shown, **env}


def _existing(path: Path) -> Path:
    """Return ``path``, or its nearest ancestor that still exists."""
    return next(p for p in (path, *path.parents) if p.is_dir())


# ``killpg`` cannot reach a wrapped shell run as another user, so a watcher in
# it reads stdin, a pipe held here: a line means done; end of file without one
# (timeout, cancel, or this process dying) means TERM then KILL the group. It
# starts from a second subshell so the command's ``wait`` skips it, and ignores
# TERM so it lives to send the KILL.
def _guard(*, kill_after_sec: float = 2.0) -> str:
    """Return the prologue that enters the directory and lets the shell be stopped."""
    return (
        'cd -- "$2" 2>/dev/null || cd /\n'
        "exec 9<&0 </dev/null\n"
        "( ( trap '' TERM; read -r _ <&9 ||"
        f" {{ kill -TERM 0; sleep {kill_after_sec:g}; kill -KILL 0; }} )"
        " >/dev/null 2>&1 & )\n"
        "exec 9<&-\n"
    )


# The group, not the leader: a backgrounded child outlives the leader and would
# hold the pipe open.
async def _stop[T](
    child: Child,
    *,
    held: _HeldPipe | None,
    output: asyncio.Future[T],
    wait_sec: float = 5.0,
) -> None:
    """End the shell's process group: TERM, a grace for ``output`` to end, then KILL."""
    proc = child.process
    if held is not None:
        held.close()
        try:
            await asyncio.wait_for(child.exited(), timeout=wait_sec)
        except TimeoutError:
            logging.getLogger(__name__).warning("Killing wrapper pid=%d", proc.pid)
            with contextlib.suppress(OSError):
                proc.kill()
        return
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    # Waited on the output, not the shell: a shell in ``wait`` ends at the TERM,
    # and a KILL then would cut short its jobs' own handlers for it.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.shield(output), timeout=child.grace_sec)
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    await child.exited()
