"""Tests for ``tools.bash``: shell command tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import asyncio
import re
import warnings

import pytest

from sagent.agent.state import ToolState
from sagent.lib.tool_validation import validate_tool_input
from sagent.testing import with_fake_agent
from sagent.tools.bash import (
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_TIMEOUT_MS,
    Bash,
    _ensure_valid_cwd,
    _kill_process_group,
    _process_output,
    _reap_at_exit,
    _render_bash_description,
    _run_foreground,
    _suppress_oserror,
    _timeout_seconds,
    reap_background_processes,
)
from sagent.tools.core import TOOL_RESULT_MAX_CHARS
from sagent.tools.lib.bash import Node


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
    assert isinstance(schema, Mapping)
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
    [10, TOOL_RESULT_MAX_CHARS - 1, TOOL_RESULT_MAX_CHARS, TOOL_RESULT_MAX_CHARS + 1],
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
        proc, "x" * stdout_size, "fatal: boom", sentinel="__NONE__", state=ToolState()
    )
    assert _EXIT_MARKER_RE.search(out), (
        f"exit-code marker lost at stdout={stdout_size:,}: a failed command"
        " is indistinguishable from a successful one"
    )
    assert "fatal: boom" in out, "stderr lost to head-only truncation"
    # The producer owns the bound, so the composed result is already
    # within the cap plus its trailing diagnostics.
    assert len(out) <= TOOL_RESULT_MAX_CHARS + 200


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
        result = await b.run({"command": ":"})  # builtin no-op
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
    await asyncio.sleep(0.05)  # child certainly finished, intentionally unreaped
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        _reap_at_exit()  # reaps finished children; no warning may escape


def test_suppress_oserror_swallows_oserror() -> None:
    with _suppress_oserror():
        raise OSError("ignored")


def test_suppress_oserror_swallows_process_lookup() -> None:
    with _suppress_oserror():
        raise ProcessLookupError("dead")


@pytest.mark.asyncio
async def test_kill_process_group_skips_completed() -> None:
    """``_kill_process_group`` no-ops when the process already exited."""
    proc = MagicMock()
    proc.returncode = 0
    # ``wait`` should never be called since we short-circuit.
    await _kill_process_group(proc)


@pytest.mark.asyncio
async def test_kill_process_group_sigterm_succeeds() -> None:
    """``_kill_process_group`` exits cleanly when ``wait`` returns inside budget."""
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 99_999  # killpg races into OSError (suppressed).

    async def _wait() -> int:
        return 0

    proc.wait = _wait
    await _kill_process_group(proc)


@pytest.mark.asyncio
async def test_run_foreground_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_run_foreground`` returns a ``[timeout after Xs]`` line."""

    class _FakeProc:
        returncode: int | None = None
        pid: int = 99_999

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"partial\n", b"")

        async def wait(self) -> int:
            return 0

    async def _create(*_args: object, **_kwargs: object) -> _FakeProc:
        return _FakeProc()

    async def _raise_timeout(coro: object, timeout: float) -> tuple[bytes, bytes]:  # noqa: ASYNC109 -- matches asyncio.wait_for signature
        del coro, timeout
        raise TimeoutError

    monkeypatch.setattr("sagent.tools.bash.asyncio.create_subprocess_exec", _create)
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

    async def _instant_wait_for(coro: object, timeout: float) -> int:  # noqa: ASYNC109 -- matches asyncio.wait_for signature
        del coro, timeout
        raise TimeoutError

    monkeypatch.setattr("asyncio.wait_for", _instant_wait_for)
    await _kill_process_group(proc)
    # First wait_for raises TimeoutError → SIGKILL → proc.wait runs once.
    assert wait_calls == [1]


def test_reason_replaces_the_exit_code() -> None:
    """A killed run reports WHY, not the shell's incidental status."""

    class _Proc:
        returncode = -9

    text = _process_output(
        cast("asyncio.subprocess.Process", _Proc()),
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
        cast("asyncio.subprocess.Process", _Proc()),
        "",
        "e" * (2 * TOOL_RESULT_MAX_CHARS),
        sentinel="__S__",
        state=ToolState(),
    )
    assert len(out) <= TOOL_RESULT_MAX_CHARS + 200, len(out)
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
    tmp_path: Path, prologue: str
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
    tmp_path: Path, command: str
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
    tmp_path: Path, command: str, expected: str
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
    timeout_ms: int, expected_s: float
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
        BASH_MAX_TIMEOUT_MS / 1000
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
    out = await _run_foreground("echo before; sleep 30", state=state, timeout_s=1)
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


def test_writers_share_one_key_so_they_queue_behind_each_other() -> None:
    b = Bash()
    assert b.serialize_key({"command": "rm a"}) == b.serialize_key(
        {"command": "git checkout ."}
    )


def test_an_unparseable_command_is_treated_as_a_writer() -> None:
    """A command we cannot classify must not be assumed harmless."""
    assert Bash().serialize_key({"command": "cat <<'EOF'\nhi\nEOF"}) is not None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
