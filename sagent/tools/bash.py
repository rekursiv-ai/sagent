"""Bash: run a command in the agent's working directory.

A ``cd`` in the command moves the agent's ``cwd`` for later tools. A command
that times out, or whose call is cancelled, is stopped with its process group.

Typical usage example:

  result = await Bash().call(context, {"command": "pytest -q", "timeout": 60_000})
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import asyncio
import contextlib
import io
import logging
import os
import secrets
import signal
import time

from sagent.lib.posix.child import Child, finish
from sagent.session.types import ShellCommandResult
from sagent.tools.command_analyzer import ReadOnly, Rewrite, ShellTool, rewrite
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import Arg, parse, schema_of


if TYPE_CHECKING:
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
        timeout: Annotated[
            float,
            "Kill the command after this many milliseconds.",
            Arg(minimum=1, maximum=600_000),
        ] = 600_000
        description: Annotated[
            str,
            "What this command does, in active voice.",
        ] = ""

    name = "Bash"
    description = (files(__package__) / "assets" / "bash.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

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
    """The environment, or ``None`` to inherit; kept out of ``repr`` for secrets."""

    drain_sec: float = 1.0
    """How long to wait for a stopped command's output."""

    grace_sec: float = 0.5
    """How long an unwrapped command has, after TERM, to close its output."""

    read_only: ReadOnly = field(default_factory=ReadOnly.default)
    """Which commands only read, and so may run beside other calls."""

    peers: Sequence[ShellTool] = ()
    """Tools that ``rewrite`` may hand a command to, such as Read for ``cat``."""

    max_result_chars: int = 30_000
    """How much text one result may hold, if below the context's."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Bash <description>``, then the command's lines."""
        header = f"Bash {str(arguments.get('description', '')).strip()}".rstrip()
        command = str(arguments.get("command", ""))
        command = command.replace("\r\n", "\n").replace("\t", "    ").strip("\n")
        return f"{header}\n{command}" if command else header

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None`` for a command that only reads, else ``"*"`` to run alone.

        Args:
          context: The calling agent's state; unused.
          arguments: What the model passed to Bash.

        Returns:
          key: ``None`` or ``"*"``.

        """
        del context
        command = arguments.get("command")
        return (
            None if isinstance(command, str) and self.read_only.reads(command) else "*"
        )

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

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ShellCommandResult:
        """Run ``command`` in ``context.cwd``, and move it where the shell ends.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Bash.Arguments``.

        Returns:
          result: What the command printed, and how it exited.

        """
        a = parse(arguments, Bash.Arguments)
        # A wrapped shell's directory may be invisible here; ``_guard`` falls back.
        if not self.exec_prefix:
            context.cwd = await asyncio.to_thread(_existing, context.cwd)
        sentinel = f"__SAGENT_CWD_{secrets.token_hex(4)}__"
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
                str(context.cwd),
            ]
        )
        started = time.monotonic()
        reason = ""
        try:
            child = await Child.start(
                *argv,
                group=True,
                grace_sec=self.grace_sec,
                stdin=asyncio.subprocess.DEVNULL if held is None else held.reader,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd if held is None else "/",
                env=self.env,
            )
            proc = child.process
            # Shielded, so a timeout cancels the wait, not the read of what was written.
            communicate = asyncio.ensure_future(proc.communicate())
            try:
                out, err = await asyncio.wait_for(
                    asyncio.shield(communicate),
                    timeout=a.timeout / 1000,
                )
            except TimeoutError:
                await finish(_stop(child, held=held, output=communicate))
                drained = await _drain(proc, communicate, grace_sec=self.drain_sec)
                out, err = drained or (b"", b"")
                reason = f"timeout after {time.monotonic() - started:.1f}s"
                if drained is None:
                    reason += "; the shell did not stop, and its output is lost"
            except asyncio.CancelledError:
                try:
                    await finish(_stop(child, held=held, output=communicate))
                finally:
                    communicate.cancel()
                raise
            else:
                if held is not None:
                    held.release()
        finally:
            if held is not None:
                held.close()
        stdout: list[str] = []
        lines = out.decode(errors="replace").split("\n")
        for i, line in enumerate(lines):
            following = lines[i + 1] if i + 1 < len(lines) else None
            # The report's leading newline, and the stream's final one, end no
            # line the command printed; kept, each is counted as one omitted.
            if not line and (following is None or following.startswith(f"{sentinel}=")):
                continue
            if not line.startswith(f"{sentinel}="):
                stdout.append(line + "\n")
            elif (cwd := Path(line.removeprefix(f"{sentinel}="))).is_absolute() and (
                held is not None or await asyncio.to_thread(cwd.is_dir)
            ):
                context.cwd = cwd
        # Stderr gets its own quarter, so floods of stdout cannot hide the error.
        stderr = [
            f"{line}\n" for line in err.decode(errors="replace").strip().split("\n")
        ]
        cap = min(self.max_result_chars, context.max_result_chars)
        diagnostics, _ = context.fit(stderr, reserve=cap * 3 // 4, cap=cap)
        diagnostics = diagnostics.strip()
        if reason:
            diagnostics += f"\n[{reason}]"
        spill = (
            context.spill_dir / f"bash-{secrets.token_hex(4)}.txt"
            if context.spill_dir is not None
            else None
        )
        note = f"; full output: {spill}" if spill is not None else ""
        body, kept = context.fit(
            stdout,
            reserve=len(diagnostics) + len(note) + 80,
            cap=cap,
        )
        body = body.strip()
        if kept < len(stdout):
            if spill is not None:
                await asyncio.to_thread(_spill, spill, stdout)
            body += f"\n... ({len(stdout) - kept:,} more lines omitted{note})"
        return ShellCommandResult(
            call_id="",
            command=("bash", "-c", a.command),
            stdout=f"{body}\n" if body else "",
            stderr=diagnostics.strip(),
            exit_code=proc.returncode,
        )


# A child that ran ``setsid`` survives the group kill and can hold the pipes open.
async def _drain(
    proc: asyncio.subprocess.Process,
    communicate: asyncio.Future[tuple[bytes, bytes]],
    *,
    grace_sec: float,
) -> tuple[bytes, bytes] | None:
    """Return what the stopped command wrote, or ``None`` if it would not end."""
    try:
        return await asyncio.wait_for(asyncio.shield(communicate), timeout=grace_sec)
    except TimeoutError:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.feed_eof()
    try:
        return await asyncio.wait_for(communicate, timeout=grace_sec)
    except TimeoutError:
        logging.getLogger(__name__).warning("Shell outlived its stop; output lost")
        return None


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
        "builtin trap __sagent_report EXIT\n"
        "trap() {\n"
        '  builtin trap "$@" || return\n'
        '  [ "$BASH_SUBSHELL" = 0 ] || return 0\n'
        "  local line\n"
        "  line=$(builtin trap -p EXIT)\n"
        '  case "$line" in *__sagent_report*) return 0 ;; esac\n'
        '  eval "set -- $line"\n'
        '  builtin trap -- "__sagent_report; ${3:-}" EXIT\n'
        "}\n"
        'eval "$1"\n'
        "__rc=$?\n"
        "__sagent_report\n"
        "exit $__rc"
    )


def _spill(path: Path, stdout: Sequence[str]) -> None:
    """Write the whole of ``stdout`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(stdout), encoding="utf-8")


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
async def _stop(
    child: Child,
    *,
    held: _HeldPipe | None,
    output: asyncio.Future[tuple[bytes, bytes]],
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
