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
import subprocess
import sys
import time

from hypothesis import (
    given,
    strategies as st,
)

import pytest

from sagent.lib.posix.child import Child
from sagent.lib.posix.pids import gone_within, running
from sagent.session.types import BashResult, ToolResult
from sagent.tools import bash
from sagent.tools.bash import (
    Bash,
    _drain,
    _early_chars,
    _EarlyWriter,
    _final_text,
    _guard,
    _HeldPipe,
    _spill,
    _stop,
)
from sagent.tools.command_analyzer import Rewrite
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.read import Read
from sagent.tools.types import Stream, ToolContext, settled


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Awaitable, Callable, Collection, Iterator, Sequence

    from treekle import PlainTree


# The shell writes its pid and its child's into the directory it starts in, so a
# test can find both again and check that neither survives.
_SLEEPER: Final = "echo $$ > shell.pid; sleep 60 & echo $! > child.pid; wait"

# A wrapper that starts the shell in a process group of its own, outside the group
# the tool makes, which is where a wrapper that changes the user leaves it. Job
# control (``set -m``) makes the group, as macOS has no ``setsid`` program. The job
# runs in the background because bash 5 runs a last foreground command in its own
# process; ``set +m`` before the ``wait`` keeps bash 3.2 from writing the job's end
# into the stderr a test reads.
_OTHER_GROUP: Final = ("/bin/bash", "-c", 'set -m; "$@" & set +m; wait "$!"', "wrapper")

# Runs a Bash call in a process of its own, so a test can kill the caller.
_CALLER: Final = """\
import asyncio
import sys
from pathlib import Path

from sagent.tools.bash import Bash
from sagent.tools.types import ToolContext, settled

directory, command, *prefix = sys.argv[1:]
context = ToolContext(cwd=Path(directory))


async def main():
    await settled(
        Bash(exec_prefix=tuple(prefix)).call(
            context,
            {"command": command, "description": "Sleep", "timeout": 120_000},
        ),
    )


asyncio.run(main())
"""


# Times out a wrapped call beside a process of the caller's own group, then says
# whether that process outlived the guard's ``kill 0``.
_BYSTANDER: Final = """\
import asyncio
import subprocess
from pathlib import Path

from sagent.tools.bash import Bash
from sagent.tools.types import ToolContext, settled

bystander = subprocess.Popen(["sleep", "60"])


async def main():
    await settled(
        Bash(exec_prefix=("/usr/bin/env",)).call(
            ToolContext(cwd=Path.cwd()),
            {"command": "sleep 60", "description": "Sleep", "timeout": 100},
        ),
    )


asyncio.run(main())
print("alive" if bystander.poll() is None else "killed", flush=True)
bystander.kill()
"""


@pytest.mark.cli_python_subprocess
def test_the_guard_ends_only_the_shells_own_group(tmp_path: Path) -> None:
    """``kill 0`` reaches the caller's group unless the shell leads one of its own."""
    script = tmp_path / "caller.py"
    script.write_text(_BYSTANDER)
    # In a session of its own, so a guard that reached the caller ends that, not pytest.
    done = subprocess.run(  # noqa: S603 -- this interpreter, and a script of the test's.
        [sys.executable, str(script)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
        start_new_session=True,
        check=False,
    )

    assert (done.returncode, "".join(done.stdout)) == (0, "alive\n")


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
    # A tab stays a tab: the printer gives it the width every row's tab takes.
    assert Bash().summary({"command": "\nif x;\r\n\tthen y\n"}) == (
        "Bash\nif x;\n\tthen y"
    )
    # Only blank lines are dropped: indentation and the last word are the command's.
    assert Bash().summary({"command": "  echo X\n"}) == "Bash\n  echo X"


def test_schema_requires_a_command_and_its_description_and_bounds_the_timeout() -> None:
    """The description heads the call's row and names it in mined sessions."""
    schema = Bash.schema
    assert schema["required"] == ["command", "description"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    timeout = properties["timeout"]
    assert isinstance(timeout, Mapping)
    assert (timeout["minimum"], timeout["maximum"]) == (1, 600_000)
    assert "600000" in Bash.description


def test_a_command_with_no_timeout_is_killed_at_the_longest_allowed() -> None:
    """A hung command never runs on forever, holding what it touches."""
    properties = Bash.schema["properties"]
    assert isinstance(properties, Mapping)
    timeout = properties["timeout"]
    assert isinstance(timeout, Mapping)
    assert Bash.Arguments(command="x", description="x").timeout == timeout["maximum"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "true", "description": "x", "run_as_fully_detached": True},
        {"command": "true", "description": "x", "timeout": 600_001},
        {"command": "true", "description": "x", "timeout": 0},
        {"description": "x", "timeout": 5},
        {"command": "true"},
    ],
)
async def test_arguments_that_do_not_fit_are_refused(
    tmp_path: Path,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match=r"argument|must be"):
        await Bash().call(ToolContext(cwd=tmp_path), arguments).result


@pytest.mark.parametrize(
    "command",
    [
        "cat foo.py",
        "grep -rl pat d 2>/dev/null",
        "rm victim",
        "echo hi > victim",
        "sed -i 's/a/b/' foo.py",
        "git checkout .",
        "pytest -q",
    ],
)
def test_a_command_runs_beside_other_calls_whether_or_not_it_writes(
    tmp_path: Path,
    command: str,
) -> None:
    """Only Read, Write, and Edit keep an order; a shell's jobs run side by side."""
    assert Bash().serialize_key(ToolContext(cwd=tmp_path), {"command": command}) is None


@pytest.mark.parametrize(
    "command",
    [
        "cd /srv && head -20 f",
        "(cd /srv && head -20 f)",
        "pushd d",
        "source env.sh",
        'eval "$line"',
        "exec bash",
        "echo 'unclosed",
        3,
        None,
    ],
)
def test_a_command_that_may_move_the_shell_runs_alone(
    tmp_path: Path,
    command: PlainTree,
) -> None:
    """The calls after it start where it ends, so none may start beside it."""
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


def test_a_line_of_reads_that_drops_its_errors_becomes_each_peers_call() -> None:
    """``2>/dev/null`` writes no file, so the line splits rather than being refused."""
    tool = Bash(peers=(Grep(), List()))
    command = "grep -rl 'x y' d 2>/dev/null | head; ls d 2>/dev/null | head"
    assert tool.rewrite({"command": command}) == (
        Rewrite(
            tool=Grep(),
            arguments={
                "output_mode": "files_with_matches",
                "pattern": "x y",
                "path": "d",
                "keep_first": 10,
            },
        ),
        Rewrite(tool=List(), arguments={"path": "d", "max_results": 10}),
    )
    assert tool.rewrite({"command": "ls d 2> err"}) is None


def test_the_environment_never_appears_in_the_repr() -> None:
    shown = repr(Bash(exec_prefix=("/usr/bin/env",), env={"TOKEN": "marker-value"}))
    assert "marker-value" not in shown
    assert "/usr/bin/env" in shown


@pytest.mark.asyncio
async def test_a_command_reports_what_it_printed_and_how_it_exited(
    tmp_path: Path,
) -> None:
    result = await _run(tmp_path, "echo out; echo err >&2; exit 3")
    # Each stream holds what the command wrote on it; its exit is read after both.
    assert ("".join(result.stdout), "".join(result.stderr), _code(result)) == (
        "out\n",
        "err",
        3,
    )
    assert result.text() == "out\nerr\n[exit 3]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("columns", "env", "seen"),
    [
        (80, None, "80"),  # No frontend: the default.
        (75, None, "75"),
        (lambda: 61, None, "61"),  # Read when the command starts.
        (75, {"COLUMNS": "120", "PATH": os.environ["PATH"]}, "120"),  # Bash.env wins.
    ],
)
async def test_a_command_sees_the_columns_its_output_is_shown_in(
    tmp_path: Path,
    columns: int | Callable[[], int],
    env: dict[str, str] | None,
    seen: str,
) -> None:
    context = ToolContext(cwd=tmp_path, columns=columns)
    result = await settled(
        Bash(env=env).call(context, _arguments('printf %s "$COLUMNS"')),
    )
    assert "".join(result.stdout) == f"{seen}\n"


@pytest.mark.asyncio
async def test_columns_leave_the_rest_of_the_environment_inherited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGENT_MARKER", "kept")
    context = ToolContext(cwd=tmp_path, columns=75)
    result = await settled(
        Bash().call(context, _arguments('printf %s "$SAGENT_MARKER"')),
    )
    assert "".join(result.stdout) == "kept\n"


@pytest.mark.asyncio
async def test_a_silent_command_reports_nothing(tmp_path: Path) -> None:
    result = await _run(tmp_path, "true")
    assert ("".join(result.stdout), "".join(result.stderr), _code(result)) == (
        "",
        "",
        0,
    )


@pytest.mark.asyncio
async def test_the_command_runs_in_the_context_directory(tmp_path: Path) -> None:
    result = await _run(tmp_path, "pwd")
    assert "".join(result.stdout) == f"{tmp_path}\n"


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
    assert _code(await _run(tmp_path, command)) == code


@pytest.mark.asyncio
async def test_a_cd_moves_the_directory_later_calls_start_in(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    await settled(Bash().call(context, _arguments("cd sub")))
    assert context.cwd == tmp_path / "sub"
    result = await settled(Bash().call(context, _arguments("pwd")))
    assert "".join(result.stdout) == f"{tmp_path / 'sub'}\n"


@pytest.mark.asyncio
async def test_a_command_that_stays_put_keeps_a_cd_run_beside_it(
    tmp_path: Path,
) -> None:
    """Commands run side by side; only one that moved the shell sets the directory."""
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    waiting = asyncio.ensure_future(
        settled(
            Bash().call(
                context,
                _arguments("echo yes > started; until [ -e go ]; do sleep 0.01; done"),
            ),
        ),
    )
    assert await _written_text(tmp_path / "started") == "yes"

    await settled(Bash().call(context, _arguments("cd sub")))
    (tmp_path / "go").touch()
    _ = await waiting

    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
async def test_the_cwd_report_never_reaches_the_output(tmp_path: Path) -> None:
    """``printf`` adds no newline, so the report must not fuse with it."""
    (tmp_path / "sub").mkdir()
    context = ToolContext(cwd=tmp_path)
    result = await settled(Bash().call(context, _arguments("cd sub; printf hello")))
    assert "".join(result.stdout) == "hello\n"
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
    result = await settled(Bash().call(context, _arguments(command)))
    assert "__SAGENT_CWD_" not in "".join(result.stdout)
    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
async def test_a_trap_the_command_sets_still_runs_and_sees_its_status(
    tmp_path: Path,
) -> None:
    result = await _run(tmp_path, "trap 'echo \"bye rc=$?\"' EXIT; echo hi; exit 3")
    assert ("".join(result.stdout), _code(result)) == ("hi\nbye rc=3\n", 3)


@pytest.mark.asyncio
async def test_a_trap_in_a_subshell_reports_no_directory(tmp_path: Path) -> None:
    """A subshell's ``cd`` never moved the shell, so its exit must not say it did."""
    context = ToolContext(cwd=tmp_path)
    result = await settled(
        Bash().call(
            context,
            _arguments("(trap 'echo x' EXIT; cd /); echo y"),
        ),
    )
    assert "".join(result.stdout) == "x\ny\n"
    assert "".join(result.stderr) == ""
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
    result = await settled(Bash().call(context, _arguments("pwd")))
    assert "".join(result.stdout) == f"{tmp_path}\n"
    assert context.cwd == tmp_path


@pytest.mark.asyncio
async def test_a_reported_directory_that_is_gone_is_not_adopted(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path)
    await settled(
        Bash().call(context, _arguments('mkdir sub && cd sub && rmdir "$PWD"')),
    )
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
    assert "".join(inherited.stdout) == "leaked|\n"
    assert "".join(replaced.stdout) == "unset|yes\n"


@pytest.mark.asyncio
async def test_the_command_reads_no_input(tmp_path: Path) -> None:
    result = await _run(tmp_path, 'read -r line; echo "status=$?"')
    assert "".join(result.stdout) == "status=1\n"


@pytest.mark.asyncio
async def test_long_output_is_cut_on_a_line_and_says_how_much_was_omitted(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, max_result_chars=200)
    result = await settled(
        Bash().call(
            context,
            _arguments(
                "for i in $(seq 1 100); do echo line$i; done; echo boom >&2; exit 3",
            ),
        ),
    )
    assert "".join(result.stdout).startswith("line1\nline2\n")
    kept = sum(row.startswith("line") for row in "".join(result.stdout).splitlines())
    assert 0 < kept < 100
    assert f"\n... ({100 - kept} lines omitted)\n" in "".join(result.stdout)
    assert "".join(result.stdout).endswith("line100\n")
    assert "".join(result.stderr) == "boom"
    assert result.text().endswith("boom\n[exit 3]")
    # The cap covers the command's output; ``[exit N]`` is appended after it.
    assert len("".join(result.stdout)) + len("boom") <= context.max_result_chars


@pytest.mark.asyncio
async def test_output_past_bash_s_own_cap_is_cut_and_spilled_whole(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, spill_dir=tmp_path / "session")
    tool = Bash(max_result_chars=200)
    result = await settled(
        tool.call(
            context,
            _arguments("for i in $(seq 1 14); do echo line-$i-padding; done"),
        ),
    )
    assert len("".join(result.stdout)) <= tool.max_result_chars
    spilled = re.search(r"; full output: (\S+)\)", "".join(result.stdout))
    assert spilled is not None
    assert "".join(result.stdout).endswith("line-14-padding\n")
    whole = await asyncio.to_thread(Path(spilled[1]).read_text, encoding="utf-8")
    assert whole.splitlines() == [f"line-{i}-padding" for i in range(1, 15)]
    # A short name, as the model reads it in every cut result.
    assert re.fullmatch(r"bash-[0-9a-f]{8}\.txt", Path(spilled[1]).name)


def test_a_spill_makes_its_directories_and_shares_them_with_the_next(
    tmp_path: Path,
) -> None:
    """A session's spill directory starts missing and nested, then holds many."""
    first, second = tmp_path / "a" / "b" / "1.txt", tmp_path / "a" / "b" / "2.txt"
    _spill(first, ["x\n", "é\n"])
    _spill(second, ["y\n"])
    assert first.read_text(encoding="utf-8") == "x\né\n"
    assert second.read_text(encoding="utf-8") == "y\n"


@pytest.mark.asyncio
async def test_output_cut_with_nowhere_to_spill_names_no_file(tmp_path: Path) -> None:
    result = await settled(
        Bash(max_result_chars=200).call(
            ToolContext(cwd=tmp_path),
            _arguments("seq 1 1000"),
        ),
    )
    assert re.search(r"\.\.\. \(\d+ lines omitted\)", "".join(result.stdout))
    assert "".join(result.stdout).endswith("1000\n")
    assert len("".join(result.stdout)) <= 200


@pytest.mark.asyncio
async def test_stderr_keeps_a_quarter_so_stdout_cannot_be_crowded_out(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path, max_result_chars=400)
    result = await settled(
        Bash().call(
            context,
            _arguments("for i in $(seq 1 200); do echo err$i >&2; done; echo out"),
        ),
    )
    assert "".join(result.stdout) == "out\n"
    assert "".join(result.stderr).startswith("err1\nerr2\n")
    assert len("".join(result.stderr)) <= 100


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
    assert "".join(result.stdout) == "before\n"
    assert re.fullmatch(r"\[timeout after 0\.\ds\]", "".join(result.stderr))
    assert re.fullmatch(
        r"before\n\[timeout after 0\.\ds\]\n\[exit -?\d+\]",
        result.text(),
    )
    assert _code(result) not in (None, 0)


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
    assert "".join(result.stdout) == "started\n"
    assert "[timeout after" in "".join(result.stderr)
    assert await gone_within([child])


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
    assert "[timeout after" in "".join(result.stderr)
    assert await gone_within(pids)


@pytest.mark.asyncio
async def test_a_job_that_leaves_the_group_cannot_hang_the_call(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job in its own group escapes the kill; the drain gives up on its pipes."""
    _stop_when(monkeypatch, exited=True)
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "set -m; sleep 60 & set +m; echo $! > child.pid; echo started",
            limit_ms=1,
            tool=Bash(drain_sec=0.05, grace_sec=0.01),
        ),
        timeout=5,
    )
    started.extend(await _written_pids(tmp_path, "child.pid"))
    assert "".join(result.stdout) == "started\n"
    assert "[timeout after" in "".join(result.stderr)


@pytest.mark.asyncio
async def test_a_stopped_shell_reports_its_exit_status_without_waitid(
    tmp_path: Path,
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before 3.13, macOS has no ``os.waitid``; the stop must not need it.

    The job in a group of its own holds the pipes, so the status comes from the
    stop, not from the read of what the shell wrote.
    """
    monkeypatch.delattr(os, "waitid")
    _stop_when(monkeypatch, exited=True)
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "set -m; sleep 60 & set +m; echo $! > child.pid; echo started",
            limit_ms=1,
            tool=Bash(drain_sec=0.01, grace_sec=0.01),
        ),
        timeout=5,
    )
    started.extend(await _written_pids(tmp_path, "child.pid"))
    assert result.result is not None


@pytest.mark.asyncio
async def test_a_cancelled_call_stops_the_shell_and_its_jobs(
    tmp_path: Path,
    started: list[int],
) -> None:
    call = asyncio.create_task(_run(tmp_path, _SLEEPER, limit_ms=60_000))
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    call.cancel()
    assert (await asyncio.wait_for(call, timeout=5)).state == "cancelled"
    assert await gone_within(pids)


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
    assert (await asyncio.wait_for(call, timeout=5)).state == "cancelled"
    assert await gone_within(pids)


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
    assert "".join(result.stdout) == "hi\n"


@pytest.mark.asyncio
async def test_a_wrapped_shell_enters_the_directory_and_tracks_cd(
    tmp_path: Path,
) -> None:
    wrapper, record = _recording_prefix(tmp_path)
    (tmp_path / "sub").mkdir()
    tool = Bash(exec_prefix=wrapper, env={"KEPT": "yes"})
    context = ToolContext(cwd=tmp_path)
    shown = await settled(tool.call(context, _arguments('echo "$PWD $KEPT"')))
    await settled(tool.call(context, _arguments("cd sub")))
    assert "".join(shown.stdout) == f"{tmp_path} yes\n"
    assert context.cwd == tmp_path / "sub"
    assert _read_text(Path(f"{record}.cwd")) == "/"


@pytest.mark.asyncio
async def test_a_wrapped_shell_falls_back_to_the_root_when_its_directory_is_gone(
    tmp_path: Path,
) -> None:
    wrapper, _ = _recording_prefix(tmp_path)
    context = ToolContext(cwd=tmp_path / "gone")
    result = await settled(Bash(exec_prefix=wrapper).call(context, _arguments("pwd")))
    assert "".join(result.stdout) == "/\n"
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
    await settled(Bash(exec_prefix=wrapper).call(context, _arguments("cd sub")))
    assert context.cwd == tmp_path / "sub"


@pytest.mark.asyncio
async def test_a_wrapped_call_keeps_both_streams_and_the_status(
    tmp_path: Path,
) -> None:
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "echo out; echo err 1>&2; exit 3",
            tool=Bash(exec_prefix=_OTHER_GROUP),
        ),
        timeout=10,
    )
    assert ("".join(result.stdout), "".join(result.stderr), _code(result)) == (
        "out\n",
        "err",
        3,
    )
    assert result.text() == "out\nerr\n[exit 3]"


@pytest.mark.asyncio
async def test_a_wrapped_command_cannot_reach_the_held_pipe(tmp_path: Path) -> None:
    """Input from the pipe, or a copy of it, would let the command free the guard."""
    # A dup, not `[ -e /dev/fd/9 ]`: macOS fails that stat with EBADF while any
    # other process looks up fd 9, which printed "closed" for a leaked pipe.
    result = await _run(
        tmp_path,
        'read -r line; echo "stdin_status=$?";'
        " { true >&9; } 2>/dev/null && echo open || echo closed",
        limit_ms=3_000,
        tool=Bash(exec_prefix=_OTHER_GROUP),
    )
    assert "".join(result.stdout) == "stdin_status=1\nclosed\n"


@pytest.mark.asyncio
async def test_a_bare_wait_does_not_wait_for_the_guard(tmp_path: Path) -> None:
    """The guard is no child of the shell, so ``wait`` ends with the command's jobs."""
    result = await asyncio.wait_for(
        _run(
            tmp_path,
            "sleep 0.02 & sleep 0.04 & wait; echo done",
            limit_ms=3_000,
            tool=Bash(exec_prefix=_OTHER_GROUP),
        ),
        timeout=10,
    )
    assert "".join(result.stdout) == "done\n"


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
        # The loop closes the descriptors on its own schedule; nothing signals it.
        await asyncio.sleep(0.001)  # house-ignore[sleep] -- See above.
    assert _open_descriptors() == before


@pytest.mark.asyncio
async def test_a_wrapper_that_cannot_start_fails_and_leaks_no_descriptor(
    tmp_path: Path,
) -> None:
    before = _open_descriptors()
    tool = Bash(exec_prefix=(str(tmp_path / "none"),))
    with pytest.raises(FileNotFoundError):
        _ = await tool.call(ToolContext(cwd=tmp_path), _arguments("true")).result
    assert _open_descriptors() == before


@pytest.mark.asyncio
async def test_a_wrapped_timeout_stops_a_shell_this_process_cannot_signal(
    tmp_path: Path,
    started: list[int],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``killpg`` misses a shell in another group; closing the held pipe does not."""
    _stop_when(monkeypatch, written=(tmp_path / "shell.pid", tmp_path / "child.pid"))
    # A stop that signalled only the wrapper would leave the shell holding the
    # output, and wait out the drain.
    tool = Bash(exec_prefix=_OTHER_GROUP, drain_sec=30)
    result = await asyncio.wait_for(
        _run(tmp_path, _SLEEPER, limit_ms=50, tool=tool),
        timeout=5,  # A stop that waited on the wrapper would take over five seconds.
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    assert "[timeout after" in "".join(result.stderr)
    assert await gone_within(pids)
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
            # A stop that signalled only the wrapper would wait out this grace for
            # a shell it never reached.
            tool=Bash(exec_prefix=_OTHER_GROUP, grace_sec=30),
        ),
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    call.cancel()
    assert (await asyncio.wait_for(call, timeout=5)).state == "cancelled"
    assert await gone_within(pids)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [(), _OTHER_GROUP], ids=["bare", "wrapped"])
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
    assert "".join(result.stdout) == ""
    assert running(child)


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
            limit_ms=1,
            tool=Bash(exec_prefix=_OTHER_GROUP),
        ),
        timeout=5,
    )
    pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend(pids)
    assert "[timeout after" in "".join(result.stderr)
    assert await gone_within(pids)


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
    assert "[timeout after" in "".join(result.stderr)
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
    # What it printed before the stop may have been read by then.
    assert "".join(result.stdout) in ("", "before\n")
    assert re.fullmatch(
        r"\[timeout after \d\.\ds; the shell did not stop,"
        r" and the rest of its output is lost\]",
        "".join(result.stderr),
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
    assert drained is False
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
    assert drained is True
    assert communicate.result() == (b"out\n", b"")


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
        *_OTHER_GROUP,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        pids = await _written_pids(tmp_path, "shell.pid", "child.pid")
        started.extend(pids)
        caller.kill()
        await caller.wait()
        assert await gone_within(pids, seconds=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            caller.kill()
        await caller.wait()


@pytest.mark.asyncio
async def test_a_call_is_recorded_whole_and_the_shell_names_itself_bash(
    tmp_path: Path,
) -> None:
    assert await _run(tmp_path, 'echo "$0"') == ToolResult(
        call_id="",
        state="done",
        stdout=("bash\n",),
        result=BashResult(command=("bash", "-c", 'echo "$0"'), exit_code=0),
    )


@pytest.mark.asyncio
async def test_bytes_that_are_not_utf8_are_replaced_not_raised(tmp_path: Path) -> None:
    result = await _run(tmp_path, r"printf 'a \377\n'; printf 'b \376\nc' >&2")
    assert ("".join(result.stdout), "".join(result.stderr)) == (
        "a \ufffd\n",
        "b \ufffd\nc",
    )


@pytest.mark.asyncio
async def test_a_line_that_looks_like_a_directory_report_is_the_commands_own(
    tmp_path: Path,
) -> None:
    context = ToolContext(cwd=tmp_path)
    result = await settled(Bash().call(context, _arguments("echo None=/; echo x=/")))
    assert "".join(result.stdout) == "None=/\nx=/\n"
    assert context.cwd == tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [r"printf 'a\n\nb\n'", r"printf 'a\n\nb'; kill -9 $$"],
    ids=["reported", "killed-before-its-report"],
)
async def test_a_blank_line_inside_the_output_is_kept(
    tmp_path: Path,
    command: str,
) -> None:
    assert "".join((await _run(tmp_path, command)).stdout) == "a\n\nb\n"


@pytest.mark.asyncio
async def test_stderr_keeps_a_quarter_of_bash_s_own_cap(tmp_path: Path) -> None:
    """The context allows far more; Bash's own cap sets the quarter."""
    tool = Bash(max_result_chars=402)
    flood = await _run(
        tmp_path,
        "for i in $(seq 1 200); do echo err$i >&2; done",
        tool=tool,
    )
    assert "".join(flood.stderr).startswith("err1\nerr2\n")
    assert len("".join(flood.stderr)) <= 402 // 4
    long = await _run(tmp_path, "printf '%0300d' 0 >&2", tool=tool)
    # The line is cut with its newline: 301 characters, 100 kept.
    assert "".join(long.stderr) == "0" * 100 + "\n... (201 chars omitted)"


@pytest.mark.asyncio
async def test_a_cut_output_fills_the_cap_but_for_its_last_line_s_slack(
    tmp_path: Path,
) -> None:
    """Room is left for the longest omission line; the shorter one leaves a little."""
    result = await _run(
        tmp_path,
        "for i in $(seq 1 1000); do echo x; done; echo boom >&2",
        tool=Bash(max_result_chars=200),
    )
    assert re.search(r"\.\.\. \(\d+ lines omitted\)", "".join(result.stdout))
    assert "".join(result.stdout).endswith("x\n")
    assert 200 - 4 <= len("".join(result.stdout)) + len("".join(result.stderr)) <= 200


@pytest.mark.asyncio
async def test_a_timeout_keeps_what_the_command_said_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stop_when(monkeypatch, written=(tmp_path / "printed",))
    result = await _run(
        tmp_path,
        "echo oops >&2; echo yes > printed; sleep 30",
        limit_ms=60,
    )
    assert re.fullmatch(r"oops\n\[timeout after 0\.\ds\]", "".join(result.stderr))
    assert re.fullmatch(
        r"oops\n\[timeout after 0\.\ds\]\n\[exit -?\d+\]",
        result.text(),
    )


@pytest.mark.asyncio
async def test_a_job_slower_than_the_grace_is_killed_mid_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grace given is the grace kept: its handler never gets to write."""
    job = (
        "trap 'sleep 0.5; echo term > got.term; exit' TERM; echo yes > trapped;"
        " while :; do sleep 0.01; done"
    )
    _stop_when(monkeypatch, written=(tmp_path / "trapped",))
    await asyncio.wait_for(
        _run(tmp_path, f'sh -c "{job}" & wait', limit_ms=1, tool=Bash(grace_sec=0.02)),
        timeout=5,
    )
    assert not (tmp_path / "got.term").exists()


@pytest.mark.asyncio
async def test_what_a_command_prints_is_written_while_it_still_runs(
    tmp_path: Path,
) -> None:
    """A line is written once a later one shows it is neither the last nor past the cut."""
    waits = "until [ -e go ]; do sleep 0.01; done"
    streaming_result = Bash().call(
        ToolContext(cwd=tmp_path),
        _arguments(f"echo one; echo two; echo three; {waits}; echo end"),
    )

    assert await asyncio.wait_for(anext(streaming_result.stdout), timeout=5) == "one\n"
    assert not streaming_result.result.done()
    (tmp_path / "go").touch()

    assert [line async for line in streaming_result.stdout] == [
        "two\n",
        "three\n",
        "end\n",
    ]
    assert [line async for line in streaming_result.stderr] == []
    assert (await streaming_result.result).exit_code == 0


@pytest.mark.asyncio
async def test_a_line_longer_than_the_readers_buffer_is_read_whole(
    tmp_path: Path,
) -> None:
    result = await _run(tmp_path, "head -c 200000 /dev/zero | tr '\\0' x; echo")
    assert result.state == "done"
    assert result.text().startswith("xxxx")
    assert "chars omitted)" in result.text()


@pytest.mark.asyncio
async def test_arguments_that_do_not_fit_end_both_streams() -> None:
    """A call that never starts its command still ends what a reader waits on."""
    result = await settled(Bash().call(ToolContext(cwd=Path("/")), {}))
    assert (result.state, result.stdout) == ("failed", ())
    assert "Missing argument" in result.text()


async def _read_closed(stream: Stream[str]) -> list[str]:
    """Close ``stream`` and return its items."""
    stream.close()
    return [item async for item in stream]


@pytest.mark.asyncio
async def test_a_line_is_written_early_once_a_later_line_arrives() -> None:
    """Leading blank lines are never written; the newest line and trailing blanks wait."""
    stream = Stream[str]()
    writer = _EarlyWriter(stream=stream, early_chars=100)
    for line in ("", "  a", "", "b", "   "):
        writer.add_line(line)

    assert await _read_closed(stream) == ["a\n", "\n"]
    assert writer.lines == ["\n", "  a\n", "\n", "b\n", "   \n"]


@pytest.mark.asyncio
async def test_the_rest_of_the_final_text_is_written_at_the_end() -> None:
    stream = Stream[str]()
    writer = _EarlyWriter(stream=stream, early_chars=100)
    for line in ("a", "b", "c"):
        writer.add_line(line)

    writer.write_rest("a\nb\nc\n")

    assert await _read_closed(stream) == ["a\n", "b\n", "c\n"]


@pytest.mark.asyncio
async def test_a_line_past_early_chars_is_written_only_at_the_end() -> None:
    """Past ``early_chars`` a line may be truncated away, so it is not written early."""
    stream = Stream[str]()
    writer = _EarlyWriter(stream=stream, early_chars=6)
    for line in ("ab", "cd", "ef", "gh"):
        writer.add_line(line)
    early = stream._queue.qsize()

    writer.write_rest("ab\n... (2 lines omitted)\ngh\n")

    assert early == 1
    assert await _read_closed(stream) == ["ab\n", "... (2 lines omitted)\n", "gh\n"]


def test_a_long_output_is_written_early_only_as_far_as_truncation_keeps() -> None:
    """Truncation keeps at least two fifths of the room at the start."""
    context = ToolContext(cwd=Path("/"), max_result_chars=50_000)
    writer = _EarlyWriter(
        stream=Stream[str](),
        early_chars=_early_chars(2_000, note=""),
    )
    for _ in range(200):
        writer.add_line("x" * 19)

    final_text, truncated = _final_text(
        context,
        writer.lines,
        reserve=0,
        cap=2_000,
        note="",
    )

    assert truncated
    assert 400 < len("".join(writer._written)) <= 2_000 * 2 // 5
    assert final_text.startswith("".join(writer._written))


@given(
    lines=st.lists(st.text(alphabet=" \tab", max_size=40), max_size=200),
    cap=st.integers(min_value=600, max_value=3_000),
    stderr_share=st.floats(min_value=0, max_value=1),
    note=st.sampled_from(["", "; full output: /state/spill/bash-0123abcd.txt"]),
)
def test_the_final_text_always_starts_with_what_was_written_early(
    lines: list[str],
    cap: int,
    stderr_share: float,
    note: str,
) -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=50_000)
    writer = _EarlyWriter(
        stream=Stream[str](),
        early_chars=_early_chars(cap, note=note),
    )
    for line in lines:
        writer.add_line(line)
    # The diagnostics take at most a quarter of the cap, and a line of why it stopped.
    reserve = int(stderr_share * (cap // 4 + 160))

    final_text, _ = _final_text(
        context,
        writer.lines,
        reserve=reserve,
        cap=cap,
        note=note,
    )

    assert final_text.startswith("".join(writer._written))


def _arguments(command: str) -> dict[str, PlainTree]:
    """Return what the model passes to run ``command``."""
    return {"command": command, "description": "Run it"}


def _code(result: ToolResult[BashResult, str, str]) -> int | None:
    """Return the exit code in ``result``."""
    tool_result = result.result
    assert tool_result is not None
    return tool_result.exit_code


async def _run(
    cwd: Path,
    command: str,
    *,
    limit_ms: float = 10_000,
    tool: Bash | None = None,
) -> ToolResult[BashResult, str, str]:
    """Run ``command`` with ``tool`` in a fresh context at ``cwd``."""
    return await settled(
        (tool or Bash()).call(
            ToolContext(cwd=cwd),
            {**_arguments(command), "timeout": limit_ms},
        ),
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
    return len(list(Path("/dev/fd").iterdir()))


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
            # The shell's jobs are other processes; only their files say what they did.
            await asyncio.sleep(0.01)  # house-ignore[sleep] -- See above.


async def _written_text(path: Path) -> str:
    """Wait until a job has written ``path``, then return its text."""
    async with asyncio.timeout(5):
        while True:
            text = _read_text(path)
            if text:
                return text
            # The shell's jobs are other processes; only their files say what they did.
            await asyncio.sleep(0.01)  # house-ignore[sleep] -- See above.


async def _collected_within(pid: int, *, seconds: float = 3.0) -> bool:
    """Whether the parent has collected ``pid`` within ``seconds``."""
    deadline = time.monotonic() + seconds
    while _is_listed(pid):
        if time.monotonic() > deadline:
            return False
        # The shell's jobs are other processes; only their files say what they did.
        await asyncio.sleep(0.01)  # house-ignore[sleep] -- See above.
    return True


def _is_listed(pid: int) -> bool:
    """Whether ``pid`` still has an entry, which its parent's wait removes."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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
    group = os.getpgid(pid)
    deadline = time.monotonic() + seconds
    while await _running_in(group) - {pid}:
        if time.monotonic() > deadline:
            return False
        # The shell's jobs are other processes; only their files say what they did.
        await asyncio.sleep(0.005)  # house-ignore[sleep] -- See above.
    return True


# ``ps``, as macOS has no ``/proc`` to read a group's members from.
async def _running_in(group: int) -> set[int]:
    """Return the processes of process group ``group`` that are not zombies."""
    ps = await asyncio.create_subprocess_exec(
        "ps",
        "-A",
        "-o",
        "pid=",
        "-o",
        "pgid=",
        "-o",
        "stat=",
        stdout=asyncio.subprocess.PIPE,
    )
    out, _ = await ps.communicate()
    rows = (line.split() for line in out.decode().splitlines())
    return {
        int(pid)
        for pid, pgid, state in rows
        if int(pgid) == group and not state.startswith("Z")
    }


@pytest.mark.parametrize(("tool", "added"), [(Bash(), 10), (Bash(nice=None), 0)])
@pytest.mark.asyncio
async def test_commands_defer_to_the_desktop_unless_told_not_to(
    tmp_path: Path,
    tool: Bash,
    added: int,
) -> None:
    want = min(19, os.nice(0) + added)
    # The priority is set as the shell starts: wait for it, and fail on time.
    result = await settled(
        tool.call(
            ToolContext(cwd=tmp_path),
            {
                **_arguments(
                    f'until [ "$(nice)" = {want} ]; do sleep 0.01; done; echo ok',
                ),
                "timeout": 2_000,
            },
        ),
    )
    assert "".join(result.stdout) == "ok\n"


@pytest.mark.parametrize(("tool", "asked"), [(Bash(), [10]), (Bash(nice=None), [])])
@pytest.mark.asyncio
async def test_a_command_asks_the_kernel_to_run_below_this_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: Bash,
    asked: list[int],
) -> None:
    """Asked as the kernel sees it: a run already at the lowest shows no change."""
    lowered: list[int] = []

    def setpriority(which: int, who: int, priority: int) -> None:
        del which, who
        lowered.append(priority - os.nice(0))

    monkeypatch.setattr(os, "setpriority", setpriority)
    await _run(tmp_path, "true", tool=tool)
    assert lowered == asked


@pytest.mark.asyncio
async def test_output_that_fits_alone_is_cut_to_leave_room_for_stderr(
    tmp_path: Path,
) -> None:
    result = await _run(
        tmp_path,
        # 191 characters of output, under the cap alone but over it with stderr.
        "for i in $(seq 1 20); do echo line-$i--; done; echo errors-here >&2",
        tool=Bash(max_result_chars=200),
    )
    assert "lines omitted" in "".join(result.stdout)
    assert len("".join(result.stdout)) + len("".join(result.stderr)) <= 200


@pytest.mark.parametrize("last", ["y" * 20, "z" * 400])
@pytest.mark.asyncio
async def test_a_cut_output_keeps_its_end_and_counts_every_line_left_out(
    tmp_path: Path,
    last: str,
) -> None:
    """The start, the end a failure prints last, and how many lines are between."""
    result = await _run(
        tmp_path,
        f"for i in $(seq 1 100); do echo line-$i; done; echo {last}",
        tool=Bash(max_result_chars=200),
    )
    kept = re.findall(r"^line-\d+$", "".join(result.stdout), flags=re.MULTILINE)
    shown_last = last in "".join(result.stdout)
    assert "".join(result.stdout).count(" lines omitted)") == 1
    omitted = (
        "".join(result.stdout).partition(" lines omitted)")[0].rpartition("... (")[2]
    )
    assert "".join(result.stdout).startswith("line-1\n")
    assert int(omitted) == 101 - len(kept) - shown_last
    assert shown_last is (len(last) < 100)
    assert len("".join(result.stdout)) <= 200


@pytest.mark.asyncio
async def test_first_and_last_lines_too_long_to_show_still_count_the_lines(
    tmp_path: Path,
) -> None:
    result = await _run(
        tmp_path,
        f"echo {'x' * 400}; seq 1 50; echo {'z' * 400}",
        tool=Bash(max_result_chars=200),
    )
    assert "".join(result.stdout).startswith("xxx")
    assert re.search(r"\.\.\. \(5[12] lines omitted\)", "".join(result.stdout))
    assert len("".join(result.stdout)) <= 200


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
