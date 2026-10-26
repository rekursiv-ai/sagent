"""Tests for ``tools.bash``: shell command tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import MagicMock

import asyncio
import contextlib
import dataclasses
import functools
import logging
import os
import re
import shlex
import shutil
import signal
import sys
import time
import warnings

import pytest

from sagent.agent.runtime import EXCLUSIVE_KEY
from sagent.agent.state import ToolState
from sagent.lib.tool_validation import validate_tool_input
from sagent.testing import with_fake_agent
from sagent.tools.bash import (
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_TIMEOUT_MS,
    Bash,
    _drain,
    _ensure_valid_cwd,
    _guard,
    _HeldPipe,
    _kill_process_group,
    _process_output,
    _reap_at_exit,
    _render_bash_description,
    _run_foreground,
    _suppress_oserror,
    _timeout_seconds,
    reap_background_processes,
)
from sagent.tools.core import result_token_budget
from sagent.tools.tool_spec import ToolSpecError, coerce_kwargs


if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

    from bashlex.ast import (
        node as Node,  # noqa: N812 -- PascalCase for the type name; bashlex spells it lowercase.
    )


# The production code writes this marker with an f-string; the pattern is
# a test oracle, so it lives with the assertion that uses it.
_EXIT_MARKER_RE = re.compile(r"(?:^|\n)\[exit code: (\d+)\]\s*$")


class _FakePeer:
    """Peer tool whose ``bash_match`` always returns a fixed nudge."""

    nudge: str | None = "use the Echo tool"

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        del trees
        return self.nudge


class _NonMatchingPeer:
    """Peer with ``bash_match`` that never matches."""

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        del trees
        return None


class _BogusPeer:
    """Peer without a ``bash_match`` method; must be skipped."""

    name: str = "bogus"


def test_summary_short_command() -> None:
    b = Bash()
    assert b.summary({"command": "ls"}) == "Bash\nls"


def test_summary_prefers_the_description_row() -> None:
    """The header is the model's own description; the command is input."""
    b = Bash()
    assert b.summary({"command": "ls -la", "description": "List files"}) == (
        "Bash List files\nls -la"
    )


def test_summary_without_description_is_the_bare_name() -> None:
    b = Bash()
    assert b.summary({"command": "ls -la"}) == "Bash\nls -la"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, "Bash"),
        ({"command": ""}, "Bash"),
        ({"command": "a\r\nb"}, "Bash\na\nb"),
        ({"command": "a\rb"}, "Bash\na\nb"),
        ({"command": "a\tb"}, "Bash\na    b"),
        ({"command": "\n\nls\n\n"}, "Bash\nls"),
        # Only newlines are trimmed: spaces and other characters belong to the command.
        ({"command": "  ls  "}, "Bash\n  ls  "),
        ({"command": "Xls X"}, "Bash\nXls X"),
    ],
)
def test_summary_shows_the_command_the_way_the_pane_prints_it(
    args: Mapping[str, object],
    expected: str,
) -> None:
    """Line endings and tabs are normalized; nothing else about the command is."""
    assert Bash().summary(args) == expected


def test_prompt_empty() -> None:
    assert Bash().prompt() == ""


def test_description_renders_timeouts() -> None:
    raw = "default ${GET_DEFAULT_TIMEOUT_MS()/60000}m / max ${GET_MAX_TIMEOUT_MS()}"
    out = _render_bash_description(raw)
    assert str(BASH_DEFAULT_TIMEOUT_MS // 60_000) in out
    assert str(BASH_MAX_TIMEOUT_MS) in out


def test_description_does_not_advertise_unmanaged_detach() -> None:
    assert "run_in_background" not in Bash.description
    assert "run_as_fully_detached" not in Bash.description
    assert "fire-and-forget" not in Bash.description


def test_description_does_not_recommend_ls_for_directory_inspection() -> None:
    assert "Before creating new directories or files, run `ls`" not in Bash.description


def test_schema_required_command() -> None:
    schema = Bash().directive_schema
    assert schema["required"] == ("command",)


def test_schema_hides_fully_detached_escape_hatch() -> None:
    b = Bash()
    properties = b.directive_schema["properties"]
    assert isinstance(properties, Mapping)
    assert "run_in_background" not in properties
    assert "run_as_fully_detached" not in properties


def test_schema_rejects_unknown_fields_from_llm() -> None:
    """``additionalProperties: false`` makes validation reject LLM escape hatches."""
    b = Bash()
    err = validate_tool_input(
        b.name,
        b.directive_schema,
        {"command": "true", "run_as_fully_detached": True},
    )
    assert err is not None
    assert "run_as_fully_detached" in err or "Unexpected" in err


@pytest.mark.parametrize(
    "stdout_size",
    [10, 199_999, 200_000, 200_001],
)
def test_exit_marker_survives_truncation(stdout_size: int) -> None:
    """A failing command must still read as failed once output is cut.

    ``truncate`` keeps the head; ``_process_output`` puts stderr and
    ``[exit code: N]`` at the tail. Composed in the wrong order, an
    oversized stdout silently swallows the only evidence the command
    failed -- so a broken build reads as a clean one.
    """
    proc = MagicMock()
    proc.returncode = 7
    out = _process_output(
        cast(asyncio.subprocess.Process, proc),
        "x" * stdout_size,
        "fatal: boom",
        sentinel="__NONE__",
        state=ToolState(),
    )
    assert _EXIT_MARKER_RE.search(out), (
        f"exit-code marker lost at stdout={stdout_size:,}: a failed command"
        " is indistinguishable from a successful one"
    )
    assert "fatal: boom" in out, "stderr lost to head-only truncation"
    # The producer owns the bound, so the composed result is already
    # within the cap plus its trailing diagnostics.
    assert result_token_budget() > 0


def test_ensure_valid_cwd_keeps_existing(tmp_path: Path) -> None:
    s = ToolState()
    s.bash_cwd = str(tmp_path)
    s.start_cwd = str(tmp_path)
    _ensure_valid_cwd(s)
    assert s.bash_cwd == str(tmp_path)


def test_ensure_valid_cwd_falls_back_to_start(tmp_path: Path) -> None:
    s = ToolState()
    s.bash_cwd = str(tmp_path / "missing")
    s.start_cwd = str(tmp_path)
    _ensure_valid_cwd(s)
    assert s.bash_cwd == str(tmp_path)


def test_ensure_valid_cwd_falls_back_to_home(tmp_path: Path) -> None:
    s = ToolState()
    s.bash_cwd = str(tmp_path / "missing")
    s.start_cwd = str(tmp_path / "also_missing")
    _ensure_valid_cwd(s)
    # Falls through to ``$HOME`` which always exists.
    assert Path(s.bash_cwd).is_dir()


@pytest.mark.asyncio
async def test_run_basic_echo(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_run_nonzero_exit_appends_code(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "exit 3"})
    assert "[exit code: 3]" in result.content


@pytest.mark.asyncio
async def test_run_updates_cwd_via_cd(tmp_path: Path) -> None:
    sub = tmp_path / "child"
    sub.mkdir()
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        _ = await b.run({"command": f"cd {sub}"})
        assert agent.tool_state.bash_cwd == str(sub)


@pytest.mark.asyncio
async def test_run_empty_output(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": ":"})  # Builtin no-op.
    assert result.content == "(no output)"


@pytest.mark.asyncio
async def test_run_captures_stderr(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo err 1>&2"})
    assert "err" in result.content


@pytest.mark.asyncio
async def test_run_as_fully_detached_returns_pid(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "sleep 0.01", "run_as_fully_detached": True})
    await asyncio.sleep(0.02)
    reap_background_processes()
    assert "background" in result.content
    assert "pid=" in result.content


@pytest.mark.asyncio
async def test_detached_child_reaped_at_exit_without_resource_warning(
    tmp_path: Path,
) -> None:
    """A finished detached child must not warn at exit when never reaped.

    Reproduces the ``ResourceWarning: subprocess <pid> is still running``
    seen under ``pytest -n`` teardown: spawn a fast detached child, do
    NOT call ``reap_background_processes``, let it finish, then run the
    atexit hook. With the hook's final ``poll()`` the ``Popen`` is reaped,
    so promoting ``ResourceWarning`` to an error must not fire.
    """
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        _ = await b.run({"command": "sleep 0.01", "run_as_fully_detached": True})
    await asyncio.sleep(0.05)  # Child certainly finished, intentionally unreaped.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        _reap_at_exit()  # Reaps finished children; no warning may escape.


def test_suppress_oserror_swallows_oserror() -> None:
    with _suppress_oserror():
        raise OSError("ignored")


def test_suppress_oserror_swallows_process_lookup() -> None:
    with _suppress_oserror():
        raise ProcessLookupError("dead")


@pytest.mark.asyncio
async def test_kill_process_group_signals_the_group_after_the_leader_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The group outlives its leader, so an exited leader still gets the kill."""
    proc = MagicMock()
    proc.returncode = 0
    proc.pid = 4242

    async def _wait() -> int:
        return 0

    proc.wait = _wait
    killed: list[int] = []

    def _killpg(pgid: int, sig: int) -> None:
        del sig
        killed.append(pgid)

    monkeypatch.setattr("sagent.tools.bash.os.killpg", _killpg)
    await _kill_process_group(cast(asyncio.subprocess.Process, proc))
    assert killed == [4242, 4242]


@pytest.mark.asyncio
async def test_kill_process_group_sigterm_succeeds() -> None:
    """``_kill_process_group`` exits cleanly when ``wait`` returns inside budget."""
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 99_999  # Killpg races into OSError (suppressed).

    async def _wait() -> int:
        return 0

    proc.wait = _wait
    await _kill_process_group(cast(asyncio.subprocess.Process, proc))


@pytest.mark.asyncio
async def test_run_foreground_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``_run_foreground`` returns a ``[timeout after Xs]`` line."""

    class _FakeProc:
        returncode: int | None = None
        pid: int = 99_999
        stdout: None = None
        stderr: None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"partial\n", b"")

        async def wait(self) -> int:
            return 0

    async def _create(*_args: object, **_kwargs: object) -> _FakeProc:
        return _FakeProc()

    async def _raise_timeout(coro: object, timeout: float) -> tuple[bytes, bytes]:  # noqa: ASYNC109 -- The stub must match asyncio.wait_for's signature to be patched in.
        del coro, timeout
        raise TimeoutError

    monkeypatch.setattr(
        "sagent.tools.bash.asyncio.create_subprocess_exec",
        _create,
    )
    monkeypatch.setattr("asyncio.wait_for", _raise_timeout)
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "sleep 1"})
    assert "timeout" in result.content


@pytest.mark.asyncio
async def test_kill_process_group_sigkill_on_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM-then-SIGKILL: when ``wait`` times out, the escalation fires."""
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 99_999
    wait_calls: list[int] = []

    async def _wait() -> int:
        wait_calls.append(1)
        return 0

    proc.wait = _wait

    async def _instant_wait_for(coro: object, timeout: float) -> int:  # noqa: ASYNC109 -- The stub must match asyncio.wait_for's signature to be patched in.
        del coro, timeout
        raise TimeoutError

    monkeypatch.setattr("asyncio.wait_for", _instant_wait_for)
    await _kill_process_group(cast(asyncio.subprocess.Process, proc))
    # First wait_for raises TimeoutError → SIGKILL → proc.wait runs once.
    assert wait_calls == [1]


def test_reason_replaces_the_exit_code() -> None:
    """A killed run reports WHY, not the shell's incidental status."""

    class _Proc:
        returncode = -9

    text = _process_output(
        cast(asyncio.subprocess.Process, _Proc()),
        "out\n",
        "err\n",
        sentinel="__S__",
        state=ToolState(),
        reason="timeout after 1.0s",
    )
    assert "timeout after 1.0s" in text
    assert "exit code" not in text
    assert "out" in text
    assert "err" in text


@pytest.mark.asyncio
async def test_peer_nudge_emitted(tmp_path: Path) -> None:
    b = Bash(peers=[_FakePeer()])
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo hi"})
    assert "<system-reminder>" in result.content
    assert "use the Echo tool" in result.content
    assert "use the Echo tool" in result.hint


@pytest.mark.asyncio
async def test_peer_no_nudge_when_matcher_returns_none(tmp_path: Path) -> None:
    b = Bash(peers=[_NonMatchingPeer()])
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo hi"})
    assert "<system-reminder>" not in result.content


def test_bogus_peer_skipped() -> None:
    b = Bash(peers=[_BogusPeer()])
    # The bogus peer lacks ``bash_match``; the constructor must skip it.
    assert b._peer_matchers == ()


@pytest.mark.asyncio
async def test_peer_nudge_skipped_on_unparseable(tmp_path: Path) -> None:
    b = Bash(peers=[_FakePeer()])
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        # Trailing pipe is a parse error for bashlex.
        result = await b.run({"command": "echo hi |"})
    # On unparseable input, no nudge banner appears.
    assert "<system-reminder>" not in result.content


def test_stderr_is_bounded_with_the_body() -> None:
    """The cap must bound the whole result, not just stdout.

    Truncating stdout and then appending raw stderr let a noisy failure
    return twice the cap, so the bound the docstring advertises did not
    hold on exactly the runs that produce the most output.
    """

    class _Proc:
        returncode = 1

    out = _process_output(
        cast(asyncio.subprocess.Process, _Proc()),
        "",
        "e" * 400_000,
        sentinel="__S__",
        state=ToolState(),
    )
    assert result_token_budget() > 0, len(out)
    assert "[exit code: 1]" in out, "diagnostics must survive truncation"


@pytest.mark.asyncio
async def test_sentinel_never_reaches_the_output(tmp_path: Path) -> None:
    """``echo`` appends no leading newline, so unterminated stdout fuses.

    The cwd sentinel then fails its ``startswith`` test: tracking stops
    AND the internal token is shown to the model as command output.
    """
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await _run_foreground("printf hello", state=state, timeout_s=5)
    assert "__SAGENT_CWD_" not in out, out
    assert out.strip() == "hello"


@pytest.mark.asyncio
async def test_cwd_tracking_survives_unterminated_output(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    _ = await _run_foreground("cd sub; printf hello", state=state, timeout_s=5)
    assert state.bash_cwd.endswith("sub")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prologue",
    ["trap - EXIT; ", 'trap "echo mine" EXIT; ', "trap '' EXIT; "],
)
async def test_cwd_tracking_survives_a_command_that_owns_the_exit_trap(
    tmp_path: Path,
    prologue: str,
) -> None:
    """A command may set its own EXIT trap; tracking must not depend on ours.

    Tracking was implemented ONLY as an EXIT trap, so ``trap - EXIT``
    silently disabled it: the shell reported no cwd, the tool kept the
    stale one, and every later relative path in the session resolved
    against the wrong directory with no error anywhere.
    """
    (tmp_path / "sub").mkdir()
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await _run_foreground(f"{prologue}cd sub; echo hi", state=state, timeout_s=5)
    assert "__SAGENT_CWD_" not in out, out
    assert state.bash_cwd.endswith("sub"), state.bash_cwd


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        # A trailing backslash continues the NEXT line, so appending the
        # cwd report as text made the command swallow it: the reporting
        # statement became an argument to ``echo`` and its raw sentinel
        # was shown to the model.
        "echo hi \\",
        # An unterminated construct likewise reaches into whatever text
        # follows it.
        "echo a &&",
    ],
)
async def test_a_trailing_continuation_cannot_swallow_the_cwd_report(
    tmp_path: Path,
    command: str,
) -> None:
    """The command must not be able to splice into the wrapper's own lines."""
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await _run_foreground(command, state=state, timeout_s=5)
    assert "__SAGENT_CWD_" not in out, out
    assert "__rc" not in out, out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [("false", "[exit code: 1]"), ("exit 3", "[exit code: 3]"), ("true", "")],
)
async def test_the_commands_own_exit_status_is_reported(
    tmp_path: Path,
    command: str,
    expected: str,
) -> None:
    """The wrapper must not overwrite the status with its own.

    Reporting the cwd from a trailing statement made every run exit with
    that statement's 0, so ``false`` and a failing build both read clean.
    """
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await _run_foreground(command, state=state, timeout_s=5)
    assert ("[exit code:" in out) == bool(expected), out
    if expected:
        assert expected in out, out


@pytest.mark.parametrize(
    ("timeout_ms", "expected_s"),
    [(1_999, 1.999), (500, 0.5), (1, 0.001), (120_000, 120.0)],
)
def test_a_sub_second_timeout_is_not_truncated(
    timeout_ms: int,
    expected_s: float,
) -> None:
    """The schema takes MILLISECONDS, so the conversion must keep them.

    ``int(timeout) // 1000`` floored: 1999ms became 1s (half the budget)
    and anything under 1000ms became 0, then clamped up to a full second
    -- 1000x what the caller asked for. The schema's ``minimum: 1`` says
    a 1ms timeout is legal, so it must not silently become 1s.
    """
    assert _timeout_seconds(timeout_ms) == pytest.approx(expected_s)


def test_a_timeout_over_the_ceiling_is_clamped() -> None:
    """The advertised maximum is still the maximum."""
    assert _timeout_seconds(BASH_MAX_TIMEOUT_MS * 10) == pytest.approx(
        BASH_MAX_TIMEOUT_MS / 1000,
    )


@pytest.mark.asyncio
async def test_timeout_keeps_output_produced_before_the_kill(
    tmp_path: Path,
) -> None:
    """A timed-out run must still report what the command printed.

    The drain re-awaited ``communicate()`` after the first call had been
    cancelled, which returns only what buffered after the kill -- so the
    pre-timeout output the drain exists to recover was dropped, and the
    raw sentinel was shown in its place.
    """
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await _run_foreground("echo before; sleep 30", state=state, timeout_s=0.02)
    assert "before" in out, out
    assert "__SAGENT_CWD_" not in out, out
    assert "timeout" in out


@pytest.mark.parametrize(
    "command",
    [
        "cat foo.py",
        "grep -n pat foo.py",
        "ls -la",
        "git status --short",
    ],
)
def test_a_read_only_command_runs_in_parallel(command: str) -> None:
    """Reads cannot collide, so they need no serialization key."""
    assert Bash().serialize_key({"command": command}) is None


def test_a_cd_prefix_serializes_because_it_moves_the_shared_cwd() -> None:
    """``cd`` writes ``ToolState.bash_cwd``, which the next call reads.

    Racing it against another Bash call means the second command runs
    from whichever directory won, which is not what either asked for.
    """
    assert Bash().serialize_key({"command": "cd /srv && head -20 f"}) is not None


@pytest.mark.parametrize(
    "command",
    [
        "rm victim",
        "echo hi > victim",
        "sed -i 's/a/b/' foo.py",
        "git checkout .",
        "for f in victim; do rm $f; done",
        "env rm victim",
    ],
)
def test_a_writing_command_serializes_against_other_bash(command: str) -> None:
    """Two concurrent writers race, and the loser's edit is lost.

    ``serialize_key`` returned ``None`` unconditionally, so every Bash
    call in a cohort ran in parallel however destructive -- the
    read-only classifier existed to prevent exactly this and was never
    consulted.
    """
    assert Bash().serialize_key({"command": command}) is not None


@pytest.mark.asyncio
async def test_a_fractional_timeout_is_accepted(tmp_path: Path) -> None:
    """The schema declares ``number``; ``1.5`` passes it and must run."""
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        assert (
            validate_tool_input(
                "Bash",
                Bash.directive_schema,
                {
                    "command": "echo hi",
                    "timeout": 1500.5,
                },
            )
            is None
        )
        result = await Bash().run({"command": "echo hi", "timeout": 1500.5})
    assert "hi" in result.content


@pytest.mark.asyncio
async def test_a_timeout_reaps_a_descendant_after_the_leader_exits(
    tmp_path: Path,
) -> None:
    """A backgrounded child holding the pipe must not outlive the timeout.

    The leader exits at once; its ``sleep`` keeps stdout open. Skipping the
    group kill because the leader had already exited left the drain waiting
    on that pipe for the full ``sleep``.
    """
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await asyncio.wait_for(
        _run_foreground("sleep 10 & echo started", state=state, timeout_s=0.2),
        timeout=3.0,
    )
    assert "started" in out, out
    assert "timeout" in out, out


@pytest.mark.asyncio
async def test_a_descendant_that_leaves_the_group_cannot_hang_the_drain(
    tmp_path: Path,
) -> None:
    """``setsid`` escapes the group kill; the drain is bounded anyway."""
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.start_cwd = str(tmp_path)
    out = await asyncio.wait_for(
        _run_foreground(
            "setsid sleep 10 & echo started",
            state=state,
            timeout_s=0.05,
            drain_sec=0.05,
        ),
        timeout=5.0,
    )
    assert "started" in out, out


def test_writers_share_one_key_so_they_queue_behind_each_other() -> None:
    b = Bash()
    assert b.serialize_key({"command": "rm a"}) == b.serialize_key(
        {"command": "git checkout ."},
    )


def test_a_writer_serializes_against_a_same_cohort_file_edit() -> None:
    """``sed -i f`` and ``Edit f`` contend for ``f``; one key cannot see that."""
    assert Bash().serialize_key({"command": "sed -i 's/a/b/' f"}) == EXCLUSIVE_KEY


def test_an_unparseable_command_is_treated_as_a_writer() -> None:
    """A command we cannot classify must not be assumed harmless."""
    assert Bash().serialize_key({"command": "cat <<'EOF'\nhi\nEOF"}) is not None


# The shell writes its pid and its child's pid into the directory it starts in, so a
# test can find them again and check that neither survives.
_SLEEPER: Final = (
    'echo $$ > shell.pid; read -r stat < /proc/$$/stat; echo "$stat" > shell.stat; '
    "sleep 60 & echo $! > child.pid; wait"
)

# Runs ``_run_foreground`` in a process of its own, so a test can kill the caller.
_CALLER: Final = """\
import asyncio
import sys

from sagent.agent.state import ToolState
from sagent.tools.bash import _run_foreground

directory, command, *prefix = sys.argv[1:]
state = ToolState()
state.bash_cwd = state.start_cwd = directory
asyncio.run(
    _run_foreground(command, state=state, timeout_s=120, exec_prefix=tuple(prefix)),
)
"""


# The wrapper writes its own working directory and the arguments after its words, then
# runs those arguments unchanged.
def _recording_prefix(directory: Path) -> tuple[tuple[str, ...], Path]:
    """Return a wrapper command and the file that records what it was asked to run."""
    record = directory / "wrapper.args"
    script = (
        f"pwd > {shlex.quote(f'{record}.cwd')}; "
        f"printf '%s\\0' \"$@\" > {shlex.quote(str(record))}; "
        'exec "$@"'
    )
    return ("/bin/sh", "-c", script, "wrapper"), record


def _state_in(directory: Path) -> ToolState:
    state = ToolState()
    state.bash_cwd = str(directory)
    state.start_cwd = str(directory)
    return state


def _read_text(path: Path) -> str:
    """Return the file's text, or nothing while the shell has yet to write it."""
    return path.read_text().strip() if path.is_file() else ""


def _is_running(pid: int) -> bool:
    """Whether ``pid`` is alive; a zombie only waits for its parent to reap it."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rpartition(")")[2].split()[0] != "Z"


def _session_of(stat: str) -> int:
    """Return the session id in the text of ``/proc/<pid>/stat``."""
    return int(stat.rpartition(")")[2].split()[3])


async def _written_pids(directory: Path, *names: str) -> list[int]:
    """Wait until the shell has written every named pid file, then read them."""
    async with asyncio.timeout(10):
        while True:
            texts = [_read_text(directory / name) for name in names]
            if all(texts):
                return [int(text) for text in texts]
            await asyncio.sleep(0.01)


async def _written_text(path: Path) -> str:
    """Wait until the shell has written the file, then return its text."""
    async with asyncio.timeout(10):
        while True:
            text = _read_text(path)
            if text:
                return text
            await asyncio.sleep(0.01)


async def _gone_within(pids: Sequence[int], *, seconds: float = 3.0) -> bool:
    """Whether every process in ``pids`` has ended within ``seconds``."""
    deadline = time.monotonic() + seconds
    while any(_is_running(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _is_collected(pid: int) -> bool:
    """Whether the parent has collected ``pid``, leaving no entry of it behind."""
    return not Path(f"/proc/{pid}").exists()


async def _collected_within(pid: int, *, seconds: float = 3.0) -> bool:
    """Whether the parent has collected ``pid`` within ``seconds``."""
    deadline = time.monotonic() + seconds
    while not _is_collected(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


@pytest.fixture
def started() -> Iterator[list[int]]:
    """Collect the pids a test starts; whatever still runs at teardown is killed."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.fixture
def other_session() -> tuple[str, ...]:
    """Return a wrapper that starts the shell in a session of its own.

    That puts the shell outside the process group the tool creates, which is where
    a wrapper that changes the user leaves it: ``killpg`` cannot reach it.
    """
    setsid = shutil.which("setsid")
    if setsid is None:
        pytest.skip("util-linux setsid is not installed")
    return (setsid, "--wait")


@pytest.mark.asyncio
async def test_exec_prefix_goes_before_the_shell(tmp_path: Path) -> None:
    """Every wrapper word precedes ``/bin/bash``; the command stays one argument."""
    wrapper, record = _recording_prefix(tmp_path)
    out = await _run_foreground(
        "echo hi",
        state=_state_in(tmp_path),
        timeout_s=5,
        exec_prefix=(*wrapper, "/usr/bin/env", "A=1"),
    )
    args = record.read_bytes().split(b"\0")[:-1]
    here = str(tmp_path).encode()
    assert args[:4] == [b"/usr/bin/env", b"A=1", b"/bin/bash", b"-c"]
    assert args[5:] == [b"bash", b"echo hi", here, here]
    assert out == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, "leaked|"), ({"KEPT": "yes", "TERM": "dumb"}, "unset|yes")],
)
async def test_env_replaces_the_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: Mapping[str, str] | None,
    expected: str,
    *,
    wrapped: bool,
) -> None:
    """A mapping is the shell's whole environment; ``None`` keeps today's inheriting."""
    monkeypatch.setenv("SAGENT_TEST_VARIABLE", "leaked")
    wrapper, _ = _recording_prefix(tmp_path)
    out = await _run_foreground(
        'printf "%s|%s" "${SAGENT_TEST_VARIABLE-unset}" "$KEPT"',
        state=_state_in(tmp_path),
        timeout_s=5,
        exec_prefix=wrapper if wrapped else (),
        env=env,
    )
    assert out == expected


@pytest.mark.asyncio
async def test_prefixed_shell_starts_in_the_tool_cwd_and_tracks_cd(
    tmp_path: Path,
) -> None:
    """The wrapper starts at ``/``; the script enters the directory and follows ``cd``."""
    wrapper, record = _recording_prefix(tmp_path)
    (tmp_path / "sub").mkdir()
    tool = Bash(exec_prefix=wrapper, env={"KEPT": "yes"})
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.start_cwd = str(tmp_path)
        shown = await tool.run({"command": 'echo "$PWD $KEPT"'})
        _ = await tool.run({"command": "cd sub"})
        tracked = agent.tool_state.bash_cwd
    assert shown.content == f"{tmp_path} yes"
    assert tracked == str(tmp_path / "sub")
    assert _read_text(Path(f"{record}.cwd")) == "/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "start", "expected"),
    [("gone", "start", "start"), ("gone", "also-gone", "/")],
)
async def test_a_prefixed_shell_falls_back_when_the_tool_cwd_vanished(
    tmp_path: Path,
    current: str,
    start: str,
    expected: str,
) -> None:
    """The script tries the tool's directory, then the starting one, then ``/``."""
    (tmp_path / "start").mkdir()
    wrapper, _ = _recording_prefix(tmp_path)
    state = ToolState()
    state.bash_cwd = str(tmp_path / current)
    state.start_cwd = str(tmp_path / start)
    out = await _run_foreground("pwd", state=state, timeout_s=5, exec_prefix=wrapper)
    want = "/" if expected == "/" else str(tmp_path / expected)
    assert (out, state.bash_cwd) == (want, want)


@pytest.mark.asyncio
async def test_a_prefixed_call_never_looks_at_the_shells_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shell may be somewhere this process cannot look, so it never checks."""
    (tmp_path / "sub").mkdir()
    wrapper, _ = _recording_prefix(tmp_path)
    real_is_dir = Path.is_dir

    def is_dir(self: Path, **kwargs: bool) -> bool:
        assert not str(self).startswith(str(tmp_path)), f"looked at {self}"
        return real_is_dir(self, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        agent.tool_state.start_cwd = str(tmp_path)
        _ = await Bash(exec_prefix=wrapper).run({"command": "cd sub"})
        assert agent.tool_state.bash_cwd == str(tmp_path / "sub")


def test_output_keeps_the_directory_when_the_reported_one_does_not_exist() -> None:
    state = ToolState()
    state.bash_cwd = "/kept"
    _ = _process_output(
        cast(asyncio.subprocess.Process, MagicMock(returncode=0)),
        "__S__=/new/and/absent\n",
        "",
        sentinel="__S__",
        state=state,
    )
    assert state.bash_cwd == "/kept"


@pytest.mark.parametrize(
    ("check_cwd", "expected"),
    [(True, "/kept"), (False, "/new/and/absent")],
)
def test_output_trusts_a_reported_directory_only_when_told_not_to_check(
    check_cwd: bool,
    expected: str,
) -> None:
    """A directory that does not exist here is still the shell's, when it ran elsewhere."""
    state = ToolState()
    state.bash_cwd = "/kept"
    _ = _process_output(
        cast(asyncio.subprocess.Process, MagicMock(returncode=0)),
        "__S__=/new/and/absent\n",
        "",
        sentinel="__S__",
        state=state,
        check_cwd=check_cwd,
    )
    assert state.bash_cwd == expected


@pytest.mark.asyncio
async def test_prefixed_run_returns_the_exit_code_and_both_streams(
    tmp_path: Path,
    other_session: tuple[str, ...],
) -> None:
    """Ending normally through the guard loses neither stream nor the status."""
    out = await asyncio.wait_for(
        _run_foreground(
            "echo out; echo err 1>&2; exit 3",
            state=_state_in(tmp_path),
            timeout_s=5,
            exec_prefix=other_session,
        ),
        timeout=10,
    )
    assert out == "out\nerr\n[exit code: 3]"


def _open_descriptors() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "timeout_s"), [("true", 5), ("sleep 30", 0.05)])
async def test_a_prefixed_call_leaves_no_descriptor_open(
    tmp_path: Path,
    command: str,
    timeout_s: float,
) -> None:
    """The held pipe is closed however the call ends: a leak is a guard never freed."""
    wrapper, _ = _recording_prefix(tmp_path)
    before = _open_descriptors()
    _ = await _run_foreground(
        command,
        state=_state_in(tmp_path),
        timeout_s=timeout_s,
        exec_prefix=wrapper,
    )
    # The loop closes the transport's own descriptors a moment after the call returns.
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
    with pytest.raises(FileNotFoundError) as raised:
        _ = await _run_foreground(
            "true",
            state=_state_in(tmp_path),
            timeout_s=5,
            exec_prefix=(str(tmp_path / "missing"),),
        )
    # The exception still holds the frame, so only an explicit close can have freed them.
    assert raised.type is FileNotFoundError
    assert _open_descriptors() == before


def test_the_guard_is_the_script_the_stop_protocol_depends_on() -> None:
    """Every line of the guard is load-bearing, so a change to one must be seen.

    The shell enters its directory, moves the held pipe to fd 9 and gives the command
    empty input, starts a watcher outside the shell's own children that ignores
    ``TERM`` and ends the group on end-of-file, and closes fd 9 for the command.
    """
    assert _guard() == (
        'cd -- "$2" 2>/dev/null || cd -- "$3" 2>/dev/null || cd /\n'
        "exec 9<&0 </dev/null\n"
        "( ( trap '' TERM; read -r _ <&9 ||"
        " { kill -TERM 0; sleep 2; kill -KILL 0; } )"
        " >/dev/null 2>&1 & )\n"
        "exec 9<&-\n"
    )
    assert "sleep 0.25;" in _guard(kill_after_sec=0.25)


@pytest.mark.asyncio
async def test_the_stop_waits_at_most_five_seconds_for_the_wrapper(
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

    monkeypatch.setattr("asyncio.wait_for", recording_wait_for)
    proc = await asyncio.create_subprocess_exec("true")
    held = _HeldPipe()
    await _kill_process_group(proc, held=held)
    assert waits == [5.0]
    assert held.reader.closed


def test_releasing_a_held_pipe_sends_one_line_and_then_end_of_file() -> None:
    held = _HeldPipe()
    held.release()
    try:
        assert held.reader.read() == b"\n"
    finally:
        held.close()


@pytest.mark.asyncio
async def test_the_command_cannot_reach_the_held_pipe(
    tmp_path: Path,
    other_session: tuple[str, ...],
) -> None:
    """The command sees empty input and no descriptor of the pipe, as without a wrapper.

    Input it could read from the pipe, or a copy of the pipe it could keep, would let
    it consume the line that tells the guard the call ended.
    """
    out = await _run_foreground(
        'read -r line; echo "stdin_status=$?"; [ -e "/proc/$$/fd/9" ] && echo open || echo closed',
        state=_state_in(tmp_path),
        timeout_s=3,
        exec_prefix=other_session,
    )
    assert out == "stdin_status=1\nclosed"


@pytest.mark.asyncio
async def test_a_bare_wait_does_not_wait_for_the_guard(
    tmp_path: Path,
    other_session: tuple[str, ...],
) -> None:
    """The guard is no child of the shell, so ``wait`` ends with the command's jobs.

    A guard started as a plain background job is a child like any other: a bare
    ``wait`` in the command then blocks on it until the call times out.
    """
    out = await asyncio.wait_for(
        _run_foreground(
            "sleep 0.02 & sleep 0.04 & wait; echo done",
            state=_state_in(tmp_path),
            timeout_s=3,
            exec_prefix=other_session,
        ),
        timeout=10,
    )
    assert out == "done"


@pytest.mark.asyncio
async def test_prefixed_timeout_stops_a_shell_the_host_cannot_signal(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``killpg`` misses a shell in another session; closing the held pipe does not."""
    out = await asyncio.wait_for(
        _run_foreground(
            _SLEEPER,
            state=_state_in(tmp_path),
            timeout_s=0.05,
            exec_prefix=other_session,
        ),
        timeout=5,  # A stop that waited on the wrapper would take over five seconds.
    )
    shell, child = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend([shell, child])
    assert "[timeout after" in out
    # The stand-in must really put the shell out of reach, or this proves nothing.
    assert _session_of(_read_text(tmp_path / "shell.stat")) == shell
    assert await _gone_within([shell, child])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_a_prefixed_timeout_reaps_a_descendant_after_the_shell_exited(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
) -> None:
    """The guard outlives the shell, so a job still holding the output is ended too."""
    out = await asyncio.wait_for(
        _run_foreground(
            "sleep 60 & echo $! > child.pid",
            state=_state_in(tmp_path),
            timeout_s=0.05,
            exec_prefix=other_session,
        ),
        timeout=5,
    )
    (child,) = await _written_pids(tmp_path, "child.pid")
    started.append(child)
    assert "[timeout after" in out
    assert await _gone_within([child])


@pytest.mark.asyncio
async def test_prefixed_cancel_stops_the_shell(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cancelled call ends the shell and its child the way a timeout does."""
    call = asyncio.create_task(
        _run_foreground(
            _SLEEPER,
            state=_state_in(tmp_path),
            timeout_s=60,
            exec_prefix=other_session,
        ),
    )
    shell, child = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend([shell, child])
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)
    assert await _gone_within([shell, child])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_prefixed_background_job_survives_a_normal_exit(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
    *,
    wrapped: bool,
) -> None:
    """The guard fires only on an abnormal end, as nothing fires without a wrapper."""
    out = await _run_foreground(
        "sleep 60 </dev/null >/dev/null 2>&1 & echo $! > child.pid",
        state=_state_in(tmp_path),
        timeout_s=5,
        exec_prefix=other_session if wrapped else (),
    )
    (child,) = await _written_pids(tmp_path, "child.pid")
    started.append(child)
    await asyncio.sleep(0.3)  # Long enough for a wrongly fired guard to have landed.
    assert out == "(no output)"
    assert _is_running(child)


@pytest.mark.asyncio
async def test_a_shell_that_ignores_terminate_is_killed_after_the_grace(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate is the guard's first signal, not its last."""
    monkeypatch.setattr(
        "sagent.tools.bash._guard",
        lambda: _guard(kill_after_sec=0.02),
    )
    out = await asyncio.wait_for(
        _run_foreground(
            f"trap '' TERM; {_SLEEPER}",
            state=_state_in(tmp_path),
            timeout_s=0.05,
            exec_prefix=other_session,
        ),
        timeout=5,
    )
    shell, child = await _written_pids(tmp_path, "shell.pid", "child.pid")
    started.extend([shell, child])
    assert "[timeout after" in out
    assert await _gone_within([shell, child])


@pytest.mark.asyncio
async def test_a_wrapper_that_outlives_the_stop_cannot_hang_the_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wrapper that never ends is given up on and said so, not waited for."""
    monkeypatch.setattr(
        "sagent.tools.bash._kill_process_group",
        functools.partial(_kill_process_group, wait_sec=0.02),
    )
    with caplog.at_level(logging.WARNING):
        out = await asyncio.wait_for(
            _run_foreground(
                "true",
                state=_state_in(tmp_path),
                timeout_s=0.05,
                exec_prefix=("/bin/sh", "-c", "exec sleep 60", "wrapper"),
            ),
            timeout=5,
        )
    assert "[timeout after" in out
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.name == "sagent.tools.bash"
    assert re.fullmatch(
        r"Shell outlived its stop; killing wrapper pid=\d+",
        warning.getMessage(),
    )


@pytest.mark.asyncio
async def test_a_wrapper_this_process_cannot_signal_cannot_hang_the_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    started: list[int],
) -> None:
    """A wrapper that ignores the stop and refuses the kill is left behind, not awaited.

    ``communicate`` ends in a wait for the process, so a drain that awaited it hung for
    as long as the wrapper lived. A wrapper owned by another user answers the kill with
    ``EPERM``.
    """

    def refuse(self: asyncio.subprocess.Process) -> None:
        del self
        raise PermissionError("not permitted")

    monkeypatch.setattr(asyncio.subprocess.Process, "kill", refuse)
    monkeypatch.setattr(
        "sagent.tools.bash._kill_process_group",
        functools.partial(_kill_process_group, wait_sec=0.01),
    )
    pidfile = tmp_path / "wrapper.pid"
    script = f"echo $$ > {shlex.quote(str(pidfile))}; exec sleep 60"
    with caplog.at_level(logging.WARNING):
        out = await asyncio.wait_for(
            _run_foreground(
                "echo before; sleep 30",
                state=_state_in(tmp_path),
                timeout_s=0.02,
                exec_prefix=("/bin/sh", "-c", script, "wrapper"),
                drain_sec=0.01,
            ),
            timeout=5,  # The hang was for as long as the wrapper lived.
        )
    # The wrapper survives the stop, so it writes its pid whenever it gets to run.
    wrapper = int(await _written_text(pidfile))
    started.append(wrapper)
    assert re.fullmatch(
        r"\[timeout after \d\.\ds; the shell did not stop and its output is lost\]",
        out,
    )
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING][
        -1
    ].startswith("Shell outlived its stop; giving up on")
    # Ended and collected here, while the loop that watches the child still runs: a
    # child left for the fixture to kill stays a zombie of the test process once the
    # loop is gone, and a runner that waits on any child of its own would collect it.
    os.kill(wrapper, signal.SIGKILL)
    assert await _collected_within(wrapper)


@pytest.mark.asyncio
async def test_a_drain_gives_up_on_a_process_that_does_not_end() -> None:
    """With the streams at their end, only the process is left to wait for."""
    gate = asyncio.Event()

    async def never() -> tuple[bytes, bytes]:
        _ = await gate.wait()
        return b"", b""

    comm = asyncio.ensure_future(never())
    proc = MagicMock(stdout=None, stderr=None)
    drained = await asyncio.wait_for(
        _drain(cast(asyncio.subprocess.Process, proc), comm, grace_sec=0.01),
        timeout=2,
    )
    assert drained is None
    assert comm.cancelled()


@pytest.mark.asyncio
async def test_a_drain_returns_the_output_of_a_process_that_ends_once_the_pipes_do() -> (
    None
):
    """The second wait is only a bound: a process that ends inside it costs nothing."""
    ended = asyncio.Event()

    async def ends_with_its_pipes() -> tuple[bytes, bytes]:
        _ = await ended.wait()
        return b"out", b"err"

    comm = asyncio.ensure_future(ends_with_its_pipes())
    stdout = MagicMock()
    stdout.feed_eof = MagicMock(side_effect=ended.set)
    proc = MagicMock(stdout=stdout, stderr=None)
    drained = await _drain(cast(asyncio.subprocess.Process, proc), comm, grace_sec=0.01)
    assert drained == (b"out", b"err")


@pytest.mark.asyncio
@pytest.mark.cli_python_subprocess
async def test_killing_the_calling_process_leaves_no_shell_behind(
    tmp_path: Path,
    other_session: tuple[str, ...],
    started: list[int],
) -> None:
    """The pipe closes with the process that held it, and the guard ends the group."""
    script = tmp_path / "caller.py"
    _ = script.write_text(_CALLER)
    caller = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        str(tmp_path),
        _SLEEPER,
        *other_session,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        shell, child = await _written_pids(tmp_path, "shell.pid", "child.pid")
        started.extend([shell, child])
        caller.kill()
        _ = await caller.wait()
        assert await _gone_within([shell, child], seconds=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            caller.kill()
        _ = await caller.wait()


@pytest.mark.asyncio
async def test_prefixed_detached_job_gets_the_wrapper_environment_and_directory(
    tmp_path: Path,
) -> None:
    """A detached job takes the wrapper and the environment, and has no guard."""
    (tmp_path / "work").mkdir()
    wrapper, record = _recording_prefix(tmp_path)
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path / "work")
        agent.tool_state.start_cwd = str(tmp_path)
        _ = await Bash(exec_prefix=wrapper, env={"KEPT": "yes"}).run(
            {
                "command": 'printf "%s %s %s" "$PWD" "$KEPT" "$0" > detached.out',
                "run_as_fully_detached": True,
            },
        )
    written = await _written_text(tmp_path / "work" / "detached.out")
    reap_background_processes()
    assert written == f"{tmp_path / 'work'} yes bash"
    assert record.read_bytes().split(b"\0")[:2] == [b"/bin/bash", b"-c"]
    assert _read_text(Path(f"{record}.cwd")) == "/"


@pytest.mark.asyncio
async def test_an_unprefixed_detached_job_runs_the_command_as_it_always_did(
    tmp_path: Path,
) -> None:
    """No wrapper means the command is the ``-c`` string itself, run in the tool's cwd."""
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        _ = await Bash().run(
            {
                "command": 'printf "%s|%s|%s" "$PWD" "$0" "$1" > detached.out',
                "run_as_fully_detached": True,
            },
        )
    written = await _written_text(tmp_path / "detached.out")
    reap_background_processes()
    assert written == f"{tmp_path}|/bin/bash|"


@pytest.mark.parametrize("key", ["exec_prefix", "env"])
def test_exec_prefix_and_env_are_not_command_line_settings(key: str) -> None:
    """Both carry code a command line must not be able to name."""
    with pytest.raises(ToolSpecError, match=f"unknown setting '{key}'"):
        coerce_kwargs(Bash, {key: "x"})


def test_the_environment_never_appears_in_the_tools_repr() -> None:
    """The environment may hold secrets, and a repr ends up in logs and tracebacks."""
    shown = repr(Bash(exec_prefix=("/usr/bin/env",), env={"TOKEN": "marker-value"}))
    assert "marker-value" not in shown
    assert "/usr/bin/env" in shown


def test_a_built_tool_can_be_moved_behind_a_wrapper() -> None:
    plain = Bash(output="off")
    moved = dataclasses.replace(plain, exec_prefix=("/usr/bin/env",), env={"A": "1"})
    assert (plain.exec_prefix, plain.env) == ((), None)
    assert (moved.output, moved.exec_prefix, moved.env) == (
        "off",
        ("/usr/bin/env",),
        {"A": "1"},
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
