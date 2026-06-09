"""Tests for the structural-diff tripwire.

The diff is the load-bearing primitive that the v2.1-β startup canary
will act on. Wrong-positive risk: a real format change goes undetected
and materialization corrupts the session. Wrong-negative risk: the
tripwire flags a benign difference and we never enable materialization.
These tests pin both sides.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import pickle
import time

import pytest

from sagent.providers.anthropic_cli_session import (
    materialize_session,
    parse_jsonl_to_messages,
    session_jsonl_path,
)
from sagent.providers.anthropic_cli_session.tripwire import (
    DiffFinding,
    is_safe_to_enable,
    run_canary_against_live_cli,
    structural_diff,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def _msgs(*messages: object) -> list[ModelContextEvent]:
    """Cast helper to silence the variance complaint on heterogeneous lists."""
    return cast(list[ModelContextEvent], list(messages))


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_identical_messages_no_findings() -> None:
    """Two identical message lists yield zero findings + ``is_safe=True``."""
    a = _msgs(UserMessage(text="ping"), AssistantMessage(text="pong"))
    findings = structural_diff(a, a)
    assert findings == []
    assert is_safe_to_enable(findings) is True


def test_length_mismatch_reported() -> None:
    """A length difference surfaces as the first finding."""
    a = _msgs(UserMessage(text="a"), UserMessage(text="b"))
    b = _msgs(UserMessage(text="a"))
    findings = structural_diff(a, b)
    assert any("length mismatch" in f.detail for f in findings)
    assert is_safe_to_enable(findings) is False


def test_text_difference_in_user_message() -> None:
    """A text change in a UserMessage surfaces with location `[i].text`."""
    a = _msgs(UserMessage(text="hello"))
    b = _msgs(UserMessage(text="world"))
    findings = structural_diff(a, b)
    assert len(findings) == 1
    assert findings[0].location == "[0].text"


def test_type_mismatch_reported() -> None:
    """A different message type at the same position surfaces as a finding."""
    a = _msgs(UserMessage(text="hello"))
    b = _msgs(AssistantMessage(text="hello"))
    findings = structural_diff(a, b)
    assert len(findings) == 1
    assert "type mismatch" in findings[0].detail


def test_tool_call_args_difference_reported() -> None:
    """A diff in ``ToolCall.args`` surfaces with the nested location."""
    a = _msgs(
        AssistantMessage(
            tool_calls=(ToolCall(id="t1", name="Bash", args={"cmd": "ls"}),),
        ),
    )
    b = _msgs(
        AssistantMessage(
            tool_calls=(ToolCall(id="t1", name="Bash", args={"cmd": "pwd"}),),
        ),
    )
    findings = structural_diff(a, b)
    assert any(f.location == "[0].tool_calls[0].args" for f in findings)


def test_tool_result_content_difference_reported() -> None:
    """``ToolResult.content`` differences surface."""
    a = _msgs(ToolResult(call_id="t1", content="42 passed"))
    b = _msgs(ToolResult(call_id="t1", content="42 failed"))
    findings = structural_diff(a, b)
    assert any(f.location == "[0].content" for f in findings)


def test_thinking_blocks_not_compared() -> None:
    """Differences in ``thinking_blocks`` are intentionally ignored.

    The thinking ``signature`` is opaque/volatile and claude's
    line-splitting groups them differently from the materializer.
    Comparing them would generate noise without catching real drift.
    """
    a = _msgs(
        AssistantMessage(
            text="ok",
            thinking_blocks=(
                {"type": "thinking", "thinking": "A", "signature": "sigA"},
            ),
        )
    )
    b = _msgs(
        AssistantMessage(
            text="ok",
            thinking_blocks=(
                {"type": "thinking", "thinking": "B", "signature": "sigB"},
            ),
        )
    )
    assert structural_diff(a, b) == []


def test_materializer_round_trip_yields_no_findings(tmp_home: Path) -> None:
    """The canonical use case: tape → materialize → re-parse → diff vs tape.

    If this ever fails, the materializer is losing information that
    the tripwire considers meaningful.
    """
    original = _msgs(
        UserMessage(text="run the suite"),
        AssistantMessage(
            text="I will run pytest.",
            tool_calls=(
                ToolCall(id="toolu_001", name="Bash", args={"command": "pytest -q"}),
            ),
        ),
        ToolResult(call_id="toolu_001", content="42 passed"),
    )
    path, _ = materialize_session(
        ModelRequest(messages=original),
        session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        cwd=tmp_home,
    )
    reparsed = parse_jsonl_to_messages(path)
    findings = structural_diff(cast(list[object], list(original)), reparsed)
    assert findings == [], f"materializer round-trip drifted: {findings}"


def test_path_inputs_supported(tmp_home: Path) -> None:
    """``Path`` arguments are parsed transparently.

    The boot path uses this signature directly: pass a claude-written
    JSONL path and the materializer-written one, ask for the verdict.
    """
    original = _msgs(UserMessage(text="ping"))
    path, _ = materialize_session(
        ModelRequest(messages=original),
        session_id="11111111-2222-3333-4444-555555555555",
        cwd=tmp_home,
    )
    # Compare the file against the in-memory original.
    findings = structural_diff(path, cast(list[object], list(original)))
    assert findings == []


def test_canary_returns_unsafe_when_cli_missing(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``claude`` on PATH → ``is_safe=False`` + diagnostic finding.

    Boot path treats this as "tripwire unavailable, default off". The
    canary must NOT raise -- a missing CLI is an operational state,
    not a bug.
    """
    # Force `shutil.which("claude")` to return None.
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire.shutil.which",
        lambda _: None,
    )
    result = run_canary_against_live_cli(
        session_id="11111111-2222-3333-4444-555555555555",
        cwd=tmp_home,
        home=tmp_home,
    )
    assert result.is_safe is False
    assert any("not on PATH" in f.detail for f in result.findings)


def _write_canary_jsonl(
    path: Path, session_id: str, *, well_formed: bool = True
) -> None:
    """Write a synthetic ``claude --print`` session JSONL.

    Mirrors the format claude actually writes: one user entry + one
    assistant entry, linked by parentUuid. The well-formed shape
    passes our schema check; ``well_formed=False`` drops the required
    ``message`` field on the user entry to simulate format drift.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "sessionId": session_id,
        "cwd": str(path.parent),
        "gitBranch": "HEAD",
        "version": "2.1.168",
        "userType": "external",
        "entrypoint": "cli",
        "isSidechain": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    user_uuid = "0000aaaa-0000-0000-0000-000000000000"
    asst_uuid = "0000bbbb-0000-0000-0000-000000000000"
    user_entry: dict[str, object] = {
        **base,
        "type": "user",
        "parentUuid": None,
        "uuid": user_uuid,
    }
    if well_formed:
        user_entry["message"] = {"role": "user", "content": "ping"}
    asst_entry = {
        **base,
        "type": "assistant",
        "parentUuid": user_uuid,
        "uuid": asst_uuid,
        "requestId": "req_canary_test",
        "message": {
            "id": "msg_canary_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    path.write_text(json.dumps(user_entry) + "\n" + json.dumps(asst_entry) + "\n")


def test_canary_succeeds_on_well_formed_jsonl(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When claude's JSONL parses cleanly, ``is_safe=True``, no findings.

    We mock the subprocess spawn (no real ``claude`` CLI invoked) by
    pre-planting a well-formed JSONL at the path the canary expects,
    then short-circuiting ``_spawn_canary_claude`` to a no-op.
    """
    session_id = "aaaaaaaa-1111-2222-3333-444444444444"

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        # Pretend claude ran successfully -- pre-plant the file.
        path = session_jsonl_path(session_id, cwd=tmp_home, home=tmp_home)
        _write_canary_jsonl(path, session_id)
        return []

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
        cleanup=False,
    )
    assert result.is_safe is True, f"expected safe; got findings: {result.findings}"
    assert result.findings == []


def test_canary_flags_unknown_entry_type(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry type we don't know about → schema finding + is_safe=False.

    This is the bug class the schema check exists for: the CLI ships a
    new entry shape, our parser silently drops it, our materializer
    can't reproduce it, ``--resume`` against our output desyncs from
    what claude expects. Flag at canary time, not in production.
    """
    session_id = "bbbbbbbb-1111-2222-3333-444444444444"
    path = session_jsonl_path(session_id, cwd=tmp_home, home=tmp_home)

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        _write_canary_jsonl(path, session_id)
        # Append a novel entry shape.
        with path.open("a") as f:
            f.write(json.dumps({"type": "shiny-new-thing", "data": "??"}) + "\n")
        return []

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
        cleanup=False,
    )
    assert result.is_safe is False
    assert any(
        f.location.endswith(".type") and "shiny-new-thing" in f.detail
        for f in result.findings
    )


def test_canary_flags_missing_required_field(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claude entry missing ``message`` (required for user/assistant) → finding."""
    session_id = "cccccccc-1111-2222-3333-444444444444"
    path = session_jsonl_path(session_id, cwd=tmp_home, home=tmp_home)

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        _write_canary_jsonl(path, session_id, well_formed=False)
        return []

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
        cleanup=False,
    )
    assert result.is_safe is False
    assert any("message" in f.location for f in result.findings), result.findings


def test_canary_flags_missing_jsonl(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude exited successfully but wrote no file → drift finding.

    Possible if ``CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`` somehow leaked
    into the env, or if the operator's HOME diverges from where we
    expect. Surface as a finding rather than crash.
    """
    session_id = "dddddddd-1111-2222-3333-444444444444"

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        # Pretend claude ran, but never wrote the file.
        return []

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
    )
    assert result.is_safe is False
    assert any("did not write" in f.detail for f in result.findings)


def test_canary_cleanup_removes_jsonl(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cleanup=True`` (default) deletes the canary JSONL after the verdict.

    The canary's session ID is a fresh UUIDv4 that no other system
    will resume against, so the file is dead weight -- and on a
    monorepo with daily boots, it would accumulate. Cleanup keeps
    the operator's ``~/.claude/projects/`` directory tidy.
    """
    session_id = "eeeeeeee-1111-2222-3333-444444444444"
    path = session_jsonl_path(session_id, cwd=tmp_home, home=tmp_home)

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        _write_canary_jsonl(path, session_id)
        return []

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
        cleanup=True,
    )
    assert result.is_safe is True
    assert not path.exists(), "canary did not clean up its JSONL"


def test_canary_propagates_spawn_findings(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn-layer finding (timeout, exit code, etc.) is surfaced verbatim.

    The boot path reads ``result.findings[0].detail`` straight into
    its warning log; this test pins the propagation contract.
    """
    session_id = "ffffffff-1111-2222-3333-444444444444"

    async def _fake_spawn(**_kwargs: object) -> list[DiffFinding]:
        return [DiffFinding(location="<canary>", detail="simulated spawn failure")]

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli_session.tripwire._spawn_canary_claude",
        _fake_spawn,
    )
    result = run_canary_against_live_cli(
        session_id=session_id,
        cwd=tmp_home,
        home=tmp_home,
    )
    assert result.is_safe is False
    assert any("simulated spawn failure" in f.detail for f in result.findings)


def test_diff_finding_is_picklable() -> None:
    """``DiffFinding`` instances cross process boundaries cleanly.

    The boot path may report findings from a child process; ensure
    they serialize without surprises.
    """
    f = DiffFinding(location="[0].text", detail="x vs y")
    assert pickle.loads(pickle.dumps(f)) == f
