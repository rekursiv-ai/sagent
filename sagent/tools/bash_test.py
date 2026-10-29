"""Tests for the Bash tool: a real ``/bin/bash`` run in ``tmp_path``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

import asyncio
import contextlib
import functools
import logging
import os
import re
import shlex
import signal
import sys
import time

import pytest

from sagent.lib.posix.child import Child
from sagent.tools import bash
from sagent.tools.bash import Bash, _drain, _guard, _HeldPipe, _stop
from sagent.tools.command_analyzer import Rewrite
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Awaitable, Callable, Collection, Iterator, Sequence

    from treekle import PlainTree

    from sagent.session.types import ShellCommandResult


# The shell writes its pid and its child's into the directory it starts in, so a
# test can find both again and check that neither survives.
_SLEEPER: Final = "echo $$ > shell.pid; sleep 60 & echo $! > child.pid; wait"

# A wrapper that starts the shell in a session of its own, outside the group the
# tool makes, which is where a wrapper that changes the user leaves it.
_OTHER_SESSION: Final = ("/usr/bin/setsid", "--wait")

# Runs a Bash call in a process of its own, so a test can kill the caller.
_CALLER: Final = """\
import asyncio
import sys
from pathlib import Path

from sagent.tools.bash import Bash
from sagent.tools.types import ToolContext

directory, command, *prefix = sys.argv[1:]
context = ToolContext(cwd=Path(directory))
asyncio.run(
    Bash(exec_prefix=tuple(prefix)).call(context, {"command": command, "timeout": 120_000}),
)
"""


@pytest.fixture
def started() -> Iterator[list[int]]:
    """Collect the pids a test starts; whatever still runs at teardown is killed."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def test_summary_heads_with_the_description_then_the_command() -> None:
    tool = Bash()
    assert tool.summary({"command": "ls", "description": " List files "}) == (
        "Bash List files\nls"
    )
    assert tool.summary({"command": "ls"}) == "Bash\nls"
    assert tool.summary({}) == "Bash"


def test_summary_shows_the_command_as_the_pane_prints_it() -> None:
    assert Bash().summary({"command": "\nif x;\r\n\tthen y\n"}) == (
        "Bash\nif x;\n    then y"
    )
    # Only blank lines are dropped: indentation and the last word are the command's.
    assert Bash().summary({"command": "  echo X\n"}) == "Bash\n  echo X"


def test_schema_requires_a_command_and_bounds_the_timeout() -> None:
    schema = Bash.schema
    assert schema["required"] == ["command"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    timeout = properties["timeout"]
    assert isinstance(timeout, Mapping)
    assert (timeout["minimum"], timeout["maximum"]) == (1, 600_000)
    assert "600000" in Bash.description
    assert "30 s" in Bash.description


def test_a_command_with_no_timeout_is_killed_at_the_longest_allowed() -> None:
    """A hung command never runs on forever, holding what it touches."""
    properties = Bash.schema["properties"]
    assert isinstance(properties, Mapping)
    timeout = properties["timeout"]
    assert isinstance(timeout, Mapping)
    assert Bash.Arguments(command="x").timeout == timeout["maximum"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "true", "run_as_fully_detached": True},
        {"command": "true", "timeout": 600_001},
        {"command": "true", "timeout": 0},
        {"timeout": 5},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await Bash().call(ToolContext(cwd=tmp_path), arguments)


@pytest.mark.parametrize(
    "command",
    ["cat foo.py", "grep -n pat foo.py", "ls -la", "git status --short"],
)
def test_a_command_that_only_reads_runs_beside_others(
    tmp_path: Path,
    command: str,
) -> None:
    assert Bash().serialize_key(ToolContext(cwd=tmp_path), {"command": command}) is None


@pytest.mark.parametrize(
    "command",
    [
        "rm victim",
        "echo hi > victim",
        "sed -i 's/a/b/' foo.py",
        "git checkout .",
        "cd /srv && head -20 f",
        "cat <<'EOF'\nhi\nEOF",
        3,
        None,
    ],
)
def test_anything_else_runs_alone(tmp_path: Path, command: PlainTree) -> None:
    key = Bash().serialize_key(ToolContext(cwd=tmp_path), {"command": command})
    assert key == "*"


def test_a_command_another_tool_does_becomes_its_call() -> None:
    tool = Bash(peers=(Read(),))
    assert tool.rewrite({"command": "cat -n a.py"}) == Rewrite(
        tool=Read(),
        arguments={"file_path": "a.py"},
    )
    assert tool.rewrite({"command": "make"}) is None
    assert tool.rewrite({"command": 7}) is None
    assert Bash().rewrite({"command": "cat a.py"}) is None


def test_the_environment_never_appears_in_the_repr() -> None:
    shown = repr(Bash(exec_prefix=("/usr/bin/env",), env={"TOKEN": "marker-value"}))
    assert "marker-value" not in shown
    assert "/usr/bin/env" in shown


@pytest.mark.asyncio
async def test_a_command_reports_what_it_printed_and_how_it_exited(
    tmp_path: Path,
) -> None:
    result = await _run(tmp_path, "echo out; echo err >&2; exit 3")
    assert result.command == ("bash", "-c", "echo out; echo err >&2; exit 3")
    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err", 3)
    assert result.text() == "out\nerr\n[exit 3]"


@pytest.mark.asyncio
async def test_a_silent_command_reports_nothing(tmp_path: Path) -> None:
    result = await _run(tmp_path, "true")
    assert (result.stdout, result.stderr, result.exit_code) == ("", "", 0)


@pytest.mark.asyncio
async def test_the_command_runs_in_the_context_directory(tmp_path: Path) -> None:
    result = await _run(tmp_path, "pwd")
    assert result.stdout == f"{tmp_path}\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "code"),
    [("false", 1), ("exit 3", 3), ("true", 0), ("(exit 4); echo x", 0)],
)
async def test_the_commands_own_exit_status_is_kept(
    tmp_path: Path,
    command: str,
    code: int,
) -> None:
    assert (await _run(tmp_path, command)).exit_code == code


@pytest.mark.asyncio
async def test_a_cd_moves_the_directory_later_calls_start_in(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    await Bash().call(context, {"command": "cd sub"})
    assert context.cwd == tmp_path / "sub"
    result = await Bash().call(context, {"command": "pwd"})
    assert result.stdout == f"{tmp_path / 'sub'}\n"


@pytest.mark.asyncio
async def test_the_cwd_report_never_reaches_the_output(tmp_path: Path) -> None:
    """``printf`` adds no newline, so the report must not fuse with it."""
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    result = await Bash().call(context, {"command": "cd sub; printf hello"})
    assert result.stdout == "hello\n"
    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "trap - EXIT; cd sub",
        'trap "echo mine" EXIT; cd sub',
        "trap '' EXIT; cd sub",
        "cd sub",
        "cd sub; exit 5",
        "trap - EXIT; cd sub; exit 5",
        "trap 'rm -f nothing' EXIT; cd sub; exit 1",
        "trap 'true' INT EXIT; trap -p >/dev/null; cd sub; exit 1",
        "set -e; trap - EXIT; cd sub; false",
    ],
)
async def test_the_cwd_is_tracked_whether_the_command_exits_or_owns_the_trap(
    tmp_path: Path,
    command: str,
) -> None:
    """A trap the command sets still reports the directory, however it ends."""
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    result = await Bash().call(context, {"command": command})
    assert "__SAGENT_CWD_" not in result.stdout
    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
async def test_a_trap_the_command_sets_still_runs_and_sees_its_status(
    tmp_path: Path,
) -> None:
    result = await _run(tmp_path, "trap 'echo \"bye rc=$?\"' EXIT; echo hi; exit 3")
    assert (result.stdout, result.exit_code) == ("hi\nbye rc=3\n", 3)


@pytest.mark.asyncio
async def test_a_trap_in_a_subshell_reports_no_directory(tmp_path: Path) -> None:
    """A subshell's ``cd`` never moved the shell, so its exit must not say it did."""
    context = ToolContext(cwd=tmp_path)
    result = await Bash().call(
        context,
        {"command": "(trap 'echo x' EXIT; cd /); echo y"},
    )
    assert result.stdout == "x\ny\n"
    assert context.cwd == tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["echo hi \\", "echo a &&"])
async def test_a_dangling_command_cannot_swallow_the_wrappers_lines(
    tmp_path: Path,
    command: str,
) -> None:
    result = await _run(tmp_path, command)
    assert "__SAGENT_CWD_" not in result.text()
    assert "__rc" not in result.text()


@pytest.mark.asyncio
async def test_a_vanished_directory_falls_back_to_its_nearest_ancestor(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path / "gone" / "deeper")
    result = await Bash().call(context, {"command": "pwd"})
    assert result.stdout == f"{tmp_path}\n"
    assert context.cwd == tmp_path


@pytest.mark.asyncio
async def test_a_reported_directory_that_is_gone_is_not_adopted(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path)
    await Bash().call(context, {"command": 'mkdir sub && cd sub && rmdir "$PWD"'})
    assert context.cwd == tmp_path


@pytest.mark.asyncio
async def test_the_environment_given_replaces_the_inherited_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGENT_TEST_VARIABLE", "leaked")
    command = 'printf "%s|%s" "${SAGENT_TEST_VARIABLE-unset}" "$KEPT"'
    inherited = await _run(tmp_path, command)
    replaced = await _run(tmp_path, command, tool=Bash(env={"KEPT": "yes"}))
    assert inherited.stdout == "leaked|\n"
    assert replaced.stdout == "unset|yes\n"


@pytest.mark.asyncio
async def test_the_command_reads_no_input(tmp_path: Path) -> None:
    result = await _run(tmp_path, 'read -r line; echo "status=$?"')
    assert result.stdout == "status=1\n"


@pytest.mark.asyncio
async def test_long_output_is_cut_on_a_line_and_says_how_much_was_omitted(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, max_result_chars=200)
    result = await Bash().call(
        context,
        {
            "command": "for i in $(seq 1 100); do echo line$i; done; echo boom >&2; exit 3",
        },
    )
    assert result.stdout.startswith("line1\nline2\n")
    kept = sum(row.startswith("line") for row in result.stdout.splitlines())
    assert 0 < kept < 100
    assert result.stdout.endswith(f"\n... ({100 - kept} more lines omitted)\n")
    assert result.stderr == "boom"
    assert result.text().endswith("boom\n[exit 3]")
    assert len(result.stdout) + len(result.stderr) <= context.max_result_chars


@pytest.mark.asyncio
async def test_output_past_bash_s_own_cap_is_cut_and_spilled_whole(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, spill_dir=tmp_path / "session")
    result = await Bash().call(
        context,
        {"command": "for i in $(seq 1 10000); do echo line-$i-padding; done"},
    )
    assert len(result.stdout) <= Bash().max_result_chars
    spilled = re.search(r"; full output: (\S+)\)\n$", result.stdout)
    assert spilled is not None
    whole = await asyncio.to_thread(Path(spilled[1]).read_text, encoding="utf-8")
    assert whole.splitlines() == [f"line-{i}-padding" for i in range(1, 10_001)]


@pytest.mark.asyncio
async def test_output_cut_with_nowhere_to_spill_names_no_file(tmp_path: Path) -> None:
    result = await Bash(max_result_chars=200).call(
        ToolContext(cwd=tmp_path),
        {"command": "seq 1 1000"},
    )
    assert re.search(r"\.\.\. \(\d+ more lines omitted\)\n$", result.stdout)
    assert len(result.stdout) <= 200


@pytest.mark.asyncio
async def test_stderr_keeps_a_quarter_so_stdout_cannot_be_crowded_out(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, max_result_chars=400)
    result = await Bash().call(
        context,
        {"command": "for i in $(seq 1 200); do echo err$i >&2; done; echo out"},
    )
    assert result.stdout == "out\n"
    assert result.stderr.startswith("err1\nerr2\n")
    assert len(result.stderr) <= 100


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [60, 60.5])
async def test_a_timeout_stops_the_command_and_keeps_what_it_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_ms: float,
) -> None:
    _stop_when(monkeypatch, written=(tmp_path / "printed",))
    began = time.monotonic()
    result = await _run(
        tmp_path,
        "echo before; echo yes > printed; sleep 30",
        limit_ms=timeout_ms,
    )
    assert time.monotonic() - began < 5
    assert result.stdout == "before\n"
    assert re.fullmatch(r"\[timeout after 0\.\ds\]", result.stderr)
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_a_timeout_ends_a_job_that_outlives_the_shell(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backgrounded job holds the pipe after the shell exits; the group kill ends it."""
    _stop_when(monkeypatch, exited=True)
    result = await asyncio.wait_for(
        _run(tmp_path, "sleep 60 & echo $! > child.pid; echo started", limit_ms=1),
        timeout=5,
    )
    (child,) = await _written_pids(tmp_path, "child.pid")
    started.append(child)
    assert result.stdout == "started\n"
    assert "[timeout after" in result.stderr
    assert await _gone_within([child])


@pytest.mark.asyncio
async def test_a_timeout_asks_every_job_to_stop_before_killing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate reaches the whole group first, and a job has the grace to clean up.

    The shell, in ``wait``, ends at the TERM; the job's handler takes longer. The
    grace lasts until the job closes the output, not its whole length.
    """
    job = (
        "trap 'sleep 0.05; echo term > got.term; exit' TERM; echo yes > trapped;"
        " while :; do sleep 0.01; done"
    )
    _stop_when(monkeypatch, written=(tmp_path / "trapped",))
    await asyncio.wait_for(
        _run(tmp_path, f'sh -c "{job}" & wait', limit_ms=1, tool=Bash(grace_sec=60)),
        timeout=5,
    )
    assert _read_text(tmp_path / "got.term") == "term"


@pytest.mark.asyncio
async def test_a_timeout_kills_jobs_that_ignore_terminate(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stop_when(monkeypatch, written=(tmp_path / "shell.pid", tmp_path / "child.pid"))
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            f"trap '' TERM; {_SLEEPER}",
            limit_ms=1,
            tool=Bash(grace_sec=0.02),
        ),
        timeout=5,
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    assert "[timeout after" in result.stderr
    assert await _gone_within(pids)


@pytest.mark.asyncio
async def test_a_job_that_leaves_the_group_cannot_hang_the_call(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``setsid`` escapes the group kill; the drain gives up on its pipes."""
    _stop_when(monkeypatch, exited=True)
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "setsid sleep 60 & echo $! > child.pid; echo started",
            limit_ms=1,
            tool=Bash(drain_sec=0.05, grace_sec=0.01),
        ),
        timeout=5,
    )
    started.extend(await _written_pids(tmp_path, "child.pid"))
    assert result.stdout == "started\n"
    assert "[timeout after" in result.stderr


@pytest.mark.asyncio
async def test_a_stopped_shell_reports_its_exit_status_without_waitid(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before 3.13, macOS has no ``os.waitid``; the stop must not need it.

    The ``setsid`` job holds the pipes, so the status comes from the stop, not
    from the read of what the shell wrote.
    """
    monkeypatch.delattr(os, "waitid")
    _stop_when(monkeypatch, exited=True)
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "setsid sleep 60 & echo $! > child.pid; echo started",
            limit_ms=1,
            tool=Bash(drain_sec=0.01, grace_sec=0.01),
        ),
        timeout=5,
    )
    started.extend(await _written_pids(tmp_path, "child.pid"))
    assert result.exit_code is not None


@pytest.mark.asyncio
async def test_a_cancelled_call_stops_the_shell_and_its_jobs(
    tmp_path: Path,
    started: list[int],
) -> None:
    call = asyncio.create_task(_run(tmp_path, _SLEEPER, limit_ms=60_000))
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)
    assert await _gone_within(pids)


@pytest.mark.asyncio
async def test_a_call_cancelled_while_the_shell_starts_stops_it_and_its_jobs(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel lands before the call holds the shell, which has started its jobs."""
    real = asyncio.create_subprocess_exec
    handed = asyncio.Event()

    async def slow(
        *argv: str,
        stdin: int | IO[bytes] | None,
        stdout: int | IO[bytes] | None,
        stderr: int | IO[bytes] | None,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        pass_fds: Collection[int],
        limit: int,
        start_new_session: bool,
    ) -> Process:
        process = await real(
            *argv,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
            env=env,
            pass_fds=pass_fds,
            limit=limit,
            start_new_session=start_new_session,
        )
        await handed.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", slow)
    call = asyncio.create_task(_run(tmp_path, _SLEEPER, limit_ms=60_000))
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    call.cancel()
    handed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)
    assert await _gone_within(pids)


@pytest.mark.asyncio
async def test_a_wrapper_goes_before_the_shell_and_the_command_stays_one_word(
    tmp_path: Path,
) -> None:
    wrapper, record = _recording_prefix(tmp_path)
    result = await _run(
        tmp_path,
        "echo hi",
        tool=Bash(exec_prefix=(*wrapper, "/usr/bin/env", "A=1")),
    )
    args = record.read_bytes().split(b"\0")[:-1]
    assert args[:4] == [b"/usr/bin/env", b"A=1", b"/bin/bash", b"-c"]
    assert args[5:] == [b"bash", b"echo hi", str(tmp_path).encode()]
    assert result.stdout == "hi\n"


@pytest.mark.asyncio
async def test_a_wrapped_shell_enters_the_directory_and_tracks_cd(
    tmp_path: Path,
) -> None:
    wrapper, record = _recording_prefix(tmp_path)
    (tmp_path / "sub").mkdir()
    tool = Bash(exec_prefix=wrapper, env={"KEPT": "yes"})
    context = ToolContext(cwd=tmp_path)
    shown = await tool.call(context, {"command": 'echo "$PWD $KEPT"'})
    await tool.call(context, {"command": "cd sub"})
    assert shown.stdout == f"{tmp_path} yes\n"
    assert context.cwd == tmp_path / "sub"
    assert _read_text(Path(f"{record}.cwd")) == "/"


@pytest.mark.asyncio
async def test_a_wrapped_shell_falls_back_to_the_root_when_its_directory_is_gone(
    tmp_path: Path,
) -> None:
    wrapper, _ = _recording_prefix(tmp_path)
    context = ToolContext(cwd=tmp_path / "gone")
    result = await Bash(exec_prefix=wrapper).call(context, {"command": "pwd"})
    assert result.stdout == "/\n"
    assert context.cwd == Path("/")


@pytest.mark.asyncio
async def test_a_wrapped_call_never_looks_at_the_shells_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shell may run where this process cannot look, so it takes its word."""
    (tmp_path / "sub").mkdir()
    wrapper, _ = _recording_prefix(tmp_path)
    real_is_dir = Path.is_dir

    def is_dir(self: Path, **kwargs: bool) -> bool:
        assert not str(self).startswith(str(tmp_path)), f"looked at {self}"
        return real_is_dir(self, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    context = ToolContext(cwd=tmp_path)
    await Bash(exec_prefix=wrapper).call(context, {"command": "cd sub"})
    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
async def test_a_wrapped_call_keeps_both_streams_and_the_status(
    tmp_path: Path,
) -> None:
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "echo out; echo err 1>&2; exit 3",
            tool=Bash(exec_prefix=_OTHER_SESSION),
        ),
        timeout=10,
    )
    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err", 3)


@pytest.mark.asyncio
async def test_a_wrapped_command_cannot_reach_the_held_pipe(tmp_path: Path) -> None:
    """Input from the pipe, or a copy of it, would let the command free the guard."""
    result = await _run(
        tmp_path,
        'read -r line; echo "stdin_status=$?";'
        ' [ -e "/proc/$$/fd/9" ] && echo open || echo closed',
        limit_ms=3_000,
        tool=Bash(exec_prefix=_OTHER_SESSION),
    )
    assert result.stdout == "stdin_status=1\nclosed\n"


@pytest.mark.asyncio
async def test_a_bare_wait_does_not_wait_for_the_guard(tmp_path: Path) -> None:
    """The guard is no child of the shell, so ``wait`` ends with the command's jobs."""
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "sleep 0.02 & sleep 0.04 & wait; echo done",
            limit_ms=3_000,
            tool=Bash(exec_prefix=_OTHER_SESSION),
        ),
        timeout=10,
    )
    assert result.stdout == "done\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "limit_ms"), [("true", 5_000), ("sleep 30", 50)])
async def test_a_wrapped_call_leaves_no_descriptor_open(
    tmp_path: Path,
    command: str,
    limit_ms: int,
) -> None:
    """A held pipe left open is a guard never freed."""
    wrapper, _ = _recording_prefix(tmp_path)
    before = _open_descriptors()
    await _run(tmp_path, command, limit_ms=limit_ms, tool=Bash(exec_prefix=wrapper))
    # The loop closes the transport's own descriptors a moment after the call.
    for _ in range(5000):
        if _open_descriptors() == before:
            break
        await asyncio.sleep(0.001)
    assert _open_descriptors() == before


@pytest.mark.asyncio
async def test_a_wrapper_that_cannot_start_raises_and_leaks_no_descriptor(
    tmp_path: Path,
) -> None:
    before = _open_descriptors()
    with pytest.raises(FileNotFoundError):
        await _run(tmp_path, "true", tool=Bash(exec_prefix=(str(tmp_path / "none"),)))
    assert _open_descriptors() == before


@pytest.mark.asyncio
async def test_a_wrapped_timeout_stops_a_shell_this_process_cannot_signal(
    tmp_path: Path,
    started: list[int],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``killpg`` misses a shell in another session; closing the held pipe does not."""
    _stop_when(monkeypatch, written=(tmp_path / "shell.pid", tmp_path / "child.pid"))
    result = await asyncio.wait_for(
        _run(tmp_path, _SLEEPER, limit_ms=50, tool=Bash(exec_prefix=_OTHER_SESSION)),
        timeout=5,  # A stop that waited on the wrapper would take over five seconds.
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    assert "[timeout after" in result.stderr
    assert await _gone_within(pids)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_a_wrapped_cancel_stops_the_shell_and_its_jobs(
    tmp_path: Path,
    started: list[int],
) -> None:
    call = asyncio.create_task(
        _run(
            tmp_path,
            _SLEEPER,
            limit_ms=60_000,
            tool=Bash(exec_prefix=_OTHER_SESSION),
        ),
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)
    assert await _gone_within(pids)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [(), _OTHER_SESSION], ids=["bare", "wrapped"])
async def test_a_background_job_outlives_a_normal_end(
    tmp_path: Path,
    started: list[int],
    prefix: tuple[str, ...],
) -> None:
    """The guard fires only on an abnormal end, as nothing fires without a wrapper."""
    result = await _run(
        tmp_path,
        "sleep 60 </dev/null >/dev/null 2>&1 & echo $! > child.pid",
        tool=Bash(exec_prefix=prefix),
    )
    (child,) = await _written_pids(tmp_path, "child.pid")
    started.append(child)
    # A guard ends its group before it ends itself, so once the child is alone
    # there, no guard fired.
    assert await _alone_within(child)
    assert result.stdout == ""
    assert _is_running(child)


@pytest.mark.asyncio
async def test_a_shell_that_ignores_terminate_is_killed_after_the_grace(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bash, "_guard", functools.partial(_guard, kill_after_sec=0.02))
    _stop_when(monkeypatch, written=(tmp_path / "shell.pid", tmp_path / "child.pid"))
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            f"trap '' TERM; {_SLEEPER}",
            limit_ms=50,
            tool=Bash(exec_prefix=_OTHER_SESSION),
        ),
        timeout=5,
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    assert "[timeout after" in result.stderr
    assert await _gone_within(pids)


@pytest.mark.asyncio
async def test_a_wrapper_that_outlives_the_stop_is_killed_and_said_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(bash, "_stop", functools.partial(_stop, wait_sec=0.02))
    with caplog.at_level(logging.WARNING):
        result = await asyncio.wait_for(
            _run(
                tmp_path,
                "true",
                limit_ms=50,
                tool=Bash(exec_prefix=("/bin/sh", "-c", "exec sleep 60", "wrapper")),
            ),
            timeout=5,
        )
    assert "[timeout after" in result.stderr
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.name == "sagent.tools.bash"
    assert re.fullmatch(r"Killing wrapper pid=\d+", warning.getMessage())


@pytest.mark.asyncio
async def test_a_wrapper_this_process_cannot_kill_is_left_behind_not_awaited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    started: list[int],
) -> None:
    """A wrapper owned by another user answers the kill with ``EPERM``."""

    def refuse(self: asyncio.subprocess.Process) -> None:
        del self
        raise PermissionError("not permitted")

    monkeypatch.setattr(asyncio.subprocess.Process, "kill", refuse)
    monkeypatch.setattr(bash, "_stop", functools.partial(_stop, wait_sec=0.01))
    pidfile = tmp_path / "wrapper.pid"
    script = f"echo $$ > {shlex.quote(str(pidfile))}; exec sleep 60"
    with caplog.at_level(logging.WARNING):
        result = await asyncio.wait_for(
            _run(
                tmp_path,
                "echo before; sleep 30",
                limit_ms=20,
                tool=Bash(
                    exec_prefix=("/bin/sh", "-c", script, "wrapper"),
                    drain_sec=0.01,
                ),
            ),
            timeout=5,
        )
    (wrapper,) = await _written_pids(tmp_path, "wrapper.pid")
    started.append(wrapper)
    assert result.stdout == ""
    assert re.fullmatch(
        r"\[timeout after \d\.\ds; the shell did not stop, and its output is lost\]",
        result.stderr,
    )
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert messages[-1] == "Shell outlived its stop; output lost"
    # Ended and collected while the loop watching the child still runs.
    os.kill(wrapper, signal.SIGKILL)
    assert await _collected_within(wrapper)


def test_the_guard_is_the_script_the_stop_protocol_depends_on() -> None:
    """The shell enters its directory, holds the pipe on fd 9, starts a watcher
    outside its own children that ignores ``TERM`` and ends the group at end of
    file, and closes fd 9 for the command.
    """
    assert _guard() == (
        'cd -- "$2" 2>/dev/null || cd /\n'
        "exec 9<&0 </dev/null\n"
        "( ( trap '' TERM; read -r _ <&9 ||"
        " { kill -TERM 0; sleep 2; kill -KILL 0; } )"
        " >/dev/null 2>&1 & )\n"
        "exec 9<&-\n"
    )
    assert "sleep 0.25;" in _guard(kill_after_sec=0.25)


@pytest.mark.asyncio
async def test_the_stop_waits_at_most_five_seconds_for_a_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five seconds outlasts the guard's two-second grace and is still a bound."""
    waits: list[float | None] = []
    real_wait_for = asyncio.wait_for

    def recording_wait_for(
        awaitable: Awaitable[int],
        timeout: float | None,
    ) -> Awaitable[int]:
        waits.append(timeout)
        return real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)
    proc = await asyncio.create_subprocess_exec("true")
    held = _HeldPipe()
    output = asyncio.ensure_future(proc.communicate())
    await _stop(
        Child(process=proc, group=True, grace_sec=Bash().grace_sec),
        held=held,
        output=output,
    )
    assert waits == [5.0]
    assert held.reader.closed


def test_an_unwrapped_job_has_half_a_second_to_end_after_terminate() -> None:
    assert Bash().grace_sec == 0.5


def test_releasing_a_held_pipe_sends_one_line_and_then_end_of_file() -> None:
    held = _HeldPipe()
    held.release()
    try:
        assert held.reader.read() == b"\n"
    finally:
        held.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stderr",
    [asyncio.subprocess.PIPE, None],
    ids=["both", "stdout"],
)
async def test_a_drain_gives_up_on_a_process_that_does_not_end(
    stderr: int | None,
) -> None:
    """Only the streams it has are cut, and the process alone is left to wait for."""
    proc = await asyncio.create_subprocess_exec(
        "sleep",
        "60",
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr,
    )
    communicate = asyncio.ensure_future(proc.communicate())
    try:
        drained = await asyncio.wait_for(
            _drain(proc, communicate, grace_sec=0.01),
            timeout=2,
        )
    finally:
        proc.kill()
        await proc.wait()
    assert drained is None
    assert communicate.cancelled()


@pytest.mark.asyncio
async def test_a_drain_returns_what_a_process_wrote_once_its_pipes_are_cut(
    tmp_path: Path,
    started: list[int],
) -> None:
    """A job holding the pipe after the process ended costs only the grace."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "echo out; sleep 60 & echo $! > job.pid",
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    started.extend(await _written_pids(tmp_path, "job.pid"))
    communicate = asyncio.ensure_future(proc.communicate())
    drained = await asyncio.wait_for(
        _drain(proc, communicate, grace_sec=0.05),
        timeout=2,
    )
    assert drained == (b"out\n", b"")


@pytest.mark.asyncio
@pytest.mark.cli_python_subprocess
async def test_killing_the_calling_process_leaves_no_shell_behind(
    tmp_path: Path,
    started: list[int],
) -> None:
    """The pipe closes with the process that held it, and the guard ends the group."""
    script = tmp_path / "caller.py"
    script.write_text(_CALLER)
    caller = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        str(tmp_path),
        _SLEEPER,
        *_OTHER_SESSION,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
        started.extend(pids)
        caller.kill()
        await caller.wait()
        assert await _gone_within(pids, seconds=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            caller.kill()
        await caller.wait()


async def _run(
    cwd: Path,
    command: str,
    *,
    limit_ms: float = 10_000,
    tool: Bash | None = None,
) -> ShellCommandResult:
    """Run ``command`` with ``tool`` in a fresh context at ``cwd``."""
    return await (tool or Bash()).call(
        ToolContext(cwd=cwd),
        {"command": command, "timeout": limit_ms},
    )


# The wrapper writes its working directory and the arguments after its words,
# then runs those arguments unchanged.
def _recording_prefix(directory: Path) -> tuple[tuple[str, ...], Path]:
    """Return a wrapper command and the file that records what it was asked to run."""
    record = directory / "wrapper.args"
    script = (
        f"pwd > {shlex.quote(f'{record}.cwd')}; "
        f"printf '%s\\0' \"$@\" > {shlex.quote(str(record))}; "
        'exec "$@"'
    )
    return ("/bin/sh", "-c", script, "wrapper"), record


def _open_descriptors() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def _read_text(path: Path) -> str:
    """Return the file's text, or nothing while the shell has yet to write it."""
    return path.read_text().strip() if path.is_file() else ""


async def _written_pids(directory: Path, *names: str) -> list[int]:
    """Wait until the shell has written every named pid file, then read them."""
    async with asyncio.timeout(10):
        while True:
            texts = [_read_text(directory / name) for name in names]
            if all(texts):
                return [int(text) for text in texts]
            await asyncio.sleep(0.01)


async def _written_text(path: Path) -> str:
    """Wait until a job has written ``path``, then return its text."""
    async with asyncio.timeout(5):
        while True:
            text = _read_text(path)
            if text:
                return text
            await asyncio.sleep(0.01)


def _is_running(pid: int) -> bool:
    """Whether ``pid`` is alive; a zombie only waits for its parent to reap it."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rpartition(")")[2].split()[0] != "Z"


async def _gone_within(pids: Sequence[int], *, seconds: float = 3.0) -> bool:
    """Whether every process in ``pids`` has ended within ``seconds``."""
    deadline = time.monotonic() + seconds
    while any(_is_running(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


async def _collected_within(pid: int, *, seconds: float = 3.0) -> bool:
    """Whether the parent has collected ``pid`` within ``seconds``."""
    deadline = time.monotonic() + seconds
    while _is_listed(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _is_listed(pid: int) -> bool:
    """Whether ``pid`` still has an entry, which its parent's wait removes."""
    return Path(f"/proc/{pid}").exists()


# The timeout's clock starts with the shell, so on a busy machine it can run out
# before the command reached the state a test is about; the stop then waits.
def _stop_when(
    monkeypatch: pytest.MonkeyPatch,
    *,
    written: Sequence[Path] = (),
    exited: bool = False,
) -> None:
    """Hold each stop until ``written`` hold text and, if ``exited``, the shell ended."""
    monkeypatch.setattr(
        bash,
        "_stop",
        functools.partial(_held_stop, stop=bash._stop, written=written, exited=exited),
    )


async def _held_stop(
    child: Child,
    *,
    held: _HeldPipe | None,
    output: asyncio.Future[tuple[bytes, bytes]],
    stop: Callable[..., Awaitable[None]],
    written: Sequence[Path],
    exited: bool,
) -> None:
    for path in written:
        await _written_text(path)
    if exited:
        await child.exited()
    await stop(child, held=held, output=output)


async def _alone_within(pid: int, *, seconds: float = 3.0) -> bool:
    """Whether, within ``seconds``, no process but ``pid`` runs in its group."""
    group = _group_of(pid)
    deadline = time.monotonic() + seconds
    while _running_in(group) - {pid}:
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


def _group_of(pid: int) -> str:
    """Return the process group ``pid`` is in, as ``/proc`` spells it."""
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[2]


def _running_in(group: str) -> set[int]:
    """Return the processes of process group ``group`` that are not zombies."""
    running: set[int] = set()
    for entry in (e for e in Path("/proc").iterdir() if e.name.isdigit()):
        try:
            state, _, pgrp, *_ = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        if pgrp == group and state != "Z":
            running.add(int(entry.name))
    return running


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
