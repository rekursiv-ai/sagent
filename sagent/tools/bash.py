"""Bash: run a command in the agent's working directory.

A ``cd`` in the command moves the agent's ``cwd`` for later tools. A command
that times out, or whose call is cancelled, is stopped with its process group.
Each line the command prints is written to its stream as it arrives, so the
record keeps every line. A long output is cut in the middle for the model,
which then reads the call's ``shown``: known only when the command ends.

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
import collections
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
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
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
            float | None,
            "Kill the command after this many milliseconds; omit to let it run.",
            Arg(minimum=1),
        ] = None

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

    max_record_chars: int = 4_000_000
    """How much of each stream the record keeps; past it, output goes to the spill
    file alone, so a command that prints without end holds no more in memory. At
    least ``max_result_chars``: a view of a record cut below it would join its
    start and end as if nothing lay between."""

    nice: int | None = 10
    """How far below this program's CPU priority commands run, so a build defers to
    the desktop; ``None`` runs them at its own."""

    def __post_init__(self) -> None:
        if self.max_record_chars < self.max_result_chars:
            raise ValueError(
                f"max_record_chars ({self.max_record_chars}) must be at least"
                f" max_result_chars ({self.max_result_chars}).",
            )

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
          streaming_result: Each line on its stream as it is printed, a
            ``BashResult`` with the exit code, ``None`` when the shell did not
            stop, and, when the output had to be cut, what the model reads as
            its ``shown``.

        """
        stdout, stderr = Stream[str](), Stream[str]()
        return whole_output(
            self._run(context, arguments, stdout, stderr),
            stdout=stdout,
            stderr=stderr,
        )

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
        stdout: Stream[str],
        stderr: Stream[str],
    ) -> ToolOutput[BashResult, str, str]:
        """Run the command; its result, and when the output was cut, its view."""
        a = parse(arguments, Bash.Arguments)
        code, shown = await self._run_command(context, a, stdout, stderr)
        return ToolOutput(
            result=BashResult(command=("bash", "-c", a.command), exit_code=code),
            shown=shown,
        )

    async def _run_command(
        self,
        context: ToolContext,
        a: Bash.Arguments,
        stdout: Stream[str],
        stderr: Stream[str],
    ) -> tuple[int | None, str | None]:
        """Run ``a.command``, writing each line as printed; return its code and view."""
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
        # Stderr gets its own quarter, so floods of stdout cannot hide the error.
        stderr_cap = cap // 4
        stdout_lines = _StdoutLines(
            printed=_Printed(
                stream=stdout,
                limit=self.max_record_chars,
                tail_chars=cap,
                spill=spill,
            ),
            report_prefix=f"{sentinel}=",
        )
        stderr_printed = _Printed(
            stream=stderr,
            limit=self.max_record_chars,
            tail_chars=stderr_cap,
        )
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
                    (pipes[1].reader, stderr_printed.add_line),
                ),
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(read_task),
                    timeout=None
                    if a.timeout is None
                    else timedelta(milliseconds=a.timeout).total_seconds(),
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
            stdout_lines.printed.close()
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
        # Cut as stdout is, to its ends: a failing command's error is often last.
        error_lines, errors_unheld = stderr_printed.ends()
        errors = [f"{line}\n" for line in "".join(error_lines).strip().split("\n")]
        diagnostics, _ = context.view(
            errors,
            cap=stderr_cap,
            note=f", {errors_unheld:,} not recorded" if errors_unheld else "",
        )
        diagnostics = diagnostics.strip()
        if reason is not None:
            diagnostics += f"\n[{reason}]"
            stderr_printed.note(f"[{reason}]")
        printed = stdout_lines.printed
        lines, unheld = printed.ends()
        # Written before the view names it, so a spill that fails is never named.
        if (
            printed.spill is not None
            and not printed.spilled
            and context.view(lines, reserve=len(diagnostics), cap=cap)[1]
            and not await asyncio.to_thread(_spill, printed.spill, lines)
        ):
            printed.spill = None
        note = f"; full output: {printed.spill}" if printed.spill is not None else ""
        final_text, _ = context.view(
            lines,
            reserve=len(diagnostics),
            cap=cap,
            note=(f", {unheld:,} not recorded" if unheld else "") + note,
        )
        code = child.process.returncode
        # The record keeps every line up to its cap; only what the model reads is
        # cut, and then it reads the view, which says the exit code as ``text``
        # would. A view that differs only in the blank lines around it is no view.
        view = final_text + diagnostics.strip()
        if (
            not printed.unrecorded
            and not stderr_printed.unrecorded
            and view.strip()
            == "".join(
                (*lines, *stderr_printed.lines),
            ).strip()
        ):
            return code, None
        return code, view + (f"\n[exit {code}]" if code not in (None, 0) else "")


@dataclass(slots=True, kw_only=True)
class _Printed:
    """One stream's lines, each written to its stream as it comes and kept.

    A command with no timeout may print without end, so past ``limit`` chars a
    line is no longer written or kept: it goes to ``spill`` alone, and the last
    ``tail_chars`` of them are held for the model's view of the end.
    """

    stream: Stream[str]
    limit: int
    tail_chars: int
    spill: Path | None = None
    lines: list[str] = field(default_factory=list[str])
    """Every line written, each with its newline."""

    tail: collections.deque[str] = field(default_factory=collections.deque[str])
    """The last lines past ``limit``, at most ``tail_chars`` of them."""

    unrecorded: int = 0
    """How many lines came past ``limit``."""

    _chars: int = 0
    _tail_held: int = 0
    _spilled: io.TextIOWrapper | None = None
    _done: bool = False

    def add_line(self, line: str) -> None:
        """Write the next line and keep it; past ``limit``, spill and hold its end.

        Args:
          line: The line, without its newline.

        """
        text = f"{line}\n"
        if not self.unrecorded and self._chars + len(text) <= self.limit:
            self._chars += len(text)
            self.lines.append(text)
            self.stream.write(text)
            return
        if not self.unrecorded:
            self._overflow()
        self.unrecorded += 1
        if self._spilled is not None:
            try:
                _ = self._spilled.write(text)
            except OSError as error:
                self._give_up(error)
        self.tail.append(text)
        self._tail_held += len(text)
        while (
            len(self.tail) > 1
            and self._tail_held - len(self.tail[0]) >= self.tail_chars
        ):
            self._tail_held -= len(self.tail.popleft())

    def note(self, text: str) -> None:
        """Write and keep the harness's own line, such as a timeout's, past any cap."""
        self.lines.append(f"{text}\n")
        self.stream.write(f"{text}\n")

    def _overflow(self) -> None:
        """Say the rest goes unrecorded, and start the spill with what was kept."""
        where = ""
        if self.spill is not None:
            # A spill that cannot be written, as on a full disk, is given up: the
            # command runs on, and the model still reads the output's ends.
            try:
                self.spill.parent.mkdir(parents=True, exist_ok=True)
                self._spilled = self.spill.open("w", encoding="utf-8")
                _ = self._spilled.write("".join(self.lines))
            except OSError as error:
                self._give_up(error)
            else:
                where = f"; all of it is in {self.spill}"
        self.stream.write(
            f"[output past {self.limit:,} chars is not recorded{where}]\n",
        )

    def close(self) -> None:
        """Finish the spill, if one was started; one that fails to flush is given up."""
        spilled, self._spilled = self._spilled, None
        if spilled is None:
            return
        try:
            spilled.close()
        except OSError as error:
            self._give_up(error)
        else:
            self._done = True

    def _give_up(self, error: OSError) -> None:
        """Drop a spill that could not be written; the model is told of none."""
        logging.getLogger(__name__).warning(
            "bash spill not written: %s",
            self.spill,
            exc_info=error,
        )
        spilled, self._spilled, self.spill = self._spilled, None, None
        if spilled is not None:
            with contextlib.suppress(OSError):
                spilled.close()

    @property
    def spilled(self) -> bool:
        """Whether the spill holds every line already."""
        return self._done

    def ends(self) -> tuple[list[str], int]:
        """Return the lines kept for the model's view, and how many lie unheld between."""
        return [*self.lines, *self.tail], self.unrecorded - len(self.tail)


@dataclass(slots=True, kw_only=True)
class _StdoutLines:
    """Separates the command's output lines from the script's directory reports."""

    printed: _Printed
    report_prefix: str
    """What a report line starts with: the sentinel and ``=``."""

    reported_dirs: list[str] = field(default_factory=list[str])
    """Each directory the script reported; the last is where the shell ended."""

    _blank_held: bool = False

    # The report's leading newline and the stream's final newline each read as an
    # empty line the command never printed, so an empty line waits for the next:
    # only then is it known to be the command's. Any other line is said at once.
    def add_line(self, line: str) -> None:
        """Add the next line of standard output.

        Args:
          line: The line, without its newline.

        """
        if line.startswith(self.report_prefix):
            self._blank_held = False
            self.reported_dirs.append(line.removeprefix(self.report_prefix))
            return
        if self._blank_held:
            self.printed.add_line("")
        self._blank_held = not line
        if line:
            self.printed.add_line(line)

    def finish(self) -> None:
        """Drop an empty line still held: the stream's final newline."""
        self._blank_held = False


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
    # After a final newline nothing is left: no line was printed there.
    if partial:
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


def _spill(path: Path, stdout: Sequence[str]) -> bool:
    """Write the whole of ``stdout`` to ``path``; return whether it could be."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(stdout).encode())
    except OSError as error:
        logging.getLogger(__name__).warning(
            "bash spill not written: %s",
            path,
            exc_info=error,
        )
        return False
    return True


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
