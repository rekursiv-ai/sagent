"""Tests for ``tools.bash``: shell command tool."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sagent.agent.runtime import ToolResult
from sagent.testing import with_fake_agent
from sagent.tools.bash import (
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_TIMEOUT_MS,
    Bash,
    _ensure_valid_cwd,
    _kill_and_drain,
    _kill_process_group,
    _render_bash_description,
    _suppress_oserror,
    _trim_bash_output,
)
from sagent.tools.core import ToolState
from sagent.tools.lib.bash import Node


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
    assert b.summary({"command": "ls"}) == "Bash ls"


def test_summary_empty_command() -> None:
    b = Bash()
    assert b.summary({}) == "Bash"


def test_summary_truncates_long_command() -> None:
    b = Bash()
    out = b.summary({"command": "x" * 100})
    assert out.endswith("...")
    assert len(out) == len("Bash ") + 60


def test_summary_replaces_newlines_with_pilcrow() -> None:
    b = Bash()
    assert b.summary({"command": "a\nb\r\nc\rd\te"}) == "Bash a⏎b⏎c⏎d e"


def test_summary_result_off_by_default() -> None:
    b = Bash()
    r = ToolResult(call_id="", content="ok\n")
    assert b.summary_result(r) is None


def test_summary_result_line_count_when_emit_on() -> None:
    b = Bash()
    b.emit_tool_summary = True
    r = ToolResult(call_id="", content="line1\nline2\n")
    assert b.summary_result(r) == "2L"


def test_summary_result_with_exit_code() -> None:
    b = Bash()
    b.emit_tool_summary = True
    r = ToolResult(call_id="", content="oops\n[exit code: 7]\n")
    assert b.summary_result(r) == "1L · exit 7"


def test_summary_result_skipped_on_error() -> None:
    b = Bash()
    b.emit_tool_summary = True
    r = ToolResult(call_id="", content="boom", is_error=True)
    assert b.summary_result(r) is None


def test_summary_result_no_trailing_newline_counts_one() -> None:
    b = Bash()
    b.emit_tool_summary = True
    r = ToolResult(call_id="", content="only line")
    assert b.summary_result(r) == "1L"


def test_prompt_empty() -> None:
    assert Bash().prompt() == ""


def test_description_renders_timeouts() -> None:
    raw = "default ${GET_DEFAULT_TIMEOUT_MS()/60000}m / max ${GET_MAX_TIMEOUT_MS()}"
    out = _render_bash_description(raw)
    assert str(BASH_DEFAULT_TIMEOUT_MS // 60_000) in out
    assert str(BASH_MAX_TIMEOUT_MS) in out


def test_schema_required_command() -> None:
    b = Bash()
    schema = b.directive_schema
    assert isinstance(schema, dict) or hasattr(schema, "__getitem__")
    assert schema["required"] == ("command",)


def test_trim_bash_output_short_unchanged() -> None:
    assert _trim_bash_output(["a", "b"]) == "a\nb"


def test_trim_bash_output_long_drops_middle() -> None:
    lines = [f"l{i}" for i in range(700)]
    out = _trim_bash_output(lines)
    assert "lines omitted" in out
    assert "l0" in out
    assert "l699" in out


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
@pytest.mark.real_sleep
async def test_run_basic_echo(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_nonzero_exit_appends_code(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "exit 3"})
    assert "[exit code: 3]" in result.content


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_updates_cwd_via_cd(tmp_path: Path) -> None:
    sub = tmp_path / "child"
    sub.mkdir()
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        _ = await b.run({"command": f"cd {sub}"})
        assert agent.tool_state.bash_cwd == str(sub)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_empty_output(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": ":"})  # builtin no-op
    assert result.content == "(no output)"


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_captures_stderr(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo err 1>&2"})
    assert "err" in result.content


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_run_background_returns_pid(tmp_path: Path) -> None:
    b = Bash()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "sleep 0.01", "run_in_background": True})
    assert "background" in result.content
    assert "pid=" in result.content


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


@pytest.mark.asyncio
async def test_kill_and_drain_formats_reason() -> None:
    proc = MagicMock()
    proc.returncode = 0
    proc.pid = 12_345

    async def _comm() -> tuple[bytes, bytes]:
        return (b"out\n", b"err\n")

    proc.communicate = _comm
    text = await _kill_and_drain(proc, start=0.0, reason="timeout")
    assert "timeout" in text


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_peer_nudge_emitted(tmp_path: Path) -> None:
    b = Bash(peers=[_FakePeer()])
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await b.run({"command": "echo hi"})
    assert "<system-reminder>" in result.content
    assert "use the Echo tool" in result.content
    assert "use the Echo tool" in result.hint


@pytest.mark.asyncio
@pytest.mark.real_sleep
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
@pytest.mark.real_sleep
async def test_peer_nudge_skipped_on_unparseable(tmp_path: Path) -> None:
    b = Bash(peers=[_FakePeer()])
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        # Trailing pipe is a parse error for bashlex.
        result = await b.run({"command": "echo hi |"})
    # On unparseable input, no nudge banner appears.
    assert "<system-reminder>" not in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
