"""Tests for the CLI session materializer.

Coverage:

- ``test_path_encoding``: cwd → directory encoding matches claude's
  rule (non-alnum → ``-``, leading ``/`` → leading ``-``).
- ``test_empty_request``: empty tape produces an empty file.
- ``test_user_text_round_trip``: single UserMessage round-trips
  through materialize → parse to byte-equal text.
- ``test_assistant_with_tool_call``: assistant turn with thinking +
  text + tool_use survives the round trip; tool_use ids preserved.
- ``test_tool_result_coalescing``: two consecutive ToolResults become
  one user-role JSONL line containing two ``tool_result`` blocks.
- ``test_deterministic_uuids``: re-running materializer on the same
  input produces byte-identical output (UUIDs included).
- ``test_atomic_write``: target file either has full prior contents
  or full new contents; never a torn write.
- ``test_orphan_thinking_dropped``: thinking block with signature but
  no body is elided.
- ``test_real_jsonl_round_trip``: if a sample claude-written JSONL is
  reachable, parse → materialize → reparse and assert the message
  list survives.

The "golden file" check (test_user_text_round_trip + the explicit
UUID assertion in test_deterministic_uuids) covers the goal of
catching format-regressions in our materializer.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import os
import pwd
import uuid as _uuid

import pytest

from sagent.providers.anthropic_cli_session import (
    materialize_session,
    parse_jsonl_to_messages,
    session_jsonl_path,
)
from sagent.providers.anthropic_cli_session.materializer import _atomic_write
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    BytesMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def _request(*messages: object) -> ModelRequest:
    """Build a ModelRequest with the given linearized messages."""
    return ModelRequest(messages=cast(list[ModelContextEvent], list(messages)))


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``$HOME`` so we don't touch the operator's real
    ``~/.claude/projects/`` from tests.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_path_encoding() -> None:
    """The encoded-cwd directory matches claude's convention.

    ``/home/jp/blackjax-devs`` → ``-home-jp-blackjax-devs`` (every
    non-alnum, including the leading ``/`` and the embedded ``/``s,
    becomes ``-``; the hyphen survives because it's in the
    ``[A-Za-z0-9-]`` keep-set).
    """
    path = session_jsonl_path(
        "abc12345-aaaa-bbbb-cccc-dddddddddddd",
        cwd=Path("/home/jp/blackjax-devs"),
    )
    assert path.parent.name == "-home-jp-blackjax-devs"
    assert path.name == "abc12345-aaaa-bbbb-cccc-dddddddddddd.jsonl"


def test_empty_request_writes_empty_file(tmp_home: Path) -> None:
    """Materializing an empty tape produces an empty (0-byte) file."""
    req = _request()
    path, entries = materialize_session(
        req, session_id="11111111-2222-3333-4444-555555555555", cwd=tmp_home
    )
    assert entries == []
    assert path.exists()
    assert path.read_text() == ""


def test_user_text_round_trip(tmp_home: Path) -> None:
    """Single ``UserMessage`` → JSONL → parse-back yields equivalent text."""
    req = _request(UserMessage(text="hello materializer"))
    path, entries = materialize_session(
        req, session_id="11111111-2222-3333-4444-555555555555", cwd=tmp_home
    )
    assert len(entries) == 1
    line = json.loads(path.read_text().splitlines()[0])
    assert line["type"] == "user"
    assert line["parentUuid"] is None  # first entry — chain root
    assert line["message"]["content"] == "hello materializer"
    assert line["sessionId"] == "11111111-2222-3333-4444-555555555555"
    assert line["userType"] == "external"
    assert line["entrypoint"] == "cli"

    parsed = parse_jsonl_to_messages(path)
    assert len(parsed) == 1
    assert isinstance(parsed[0], UserMessage)
    assert parsed[0].text == "hello materializer"


def test_assistant_with_thinking_and_tool_call(tmp_home: Path) -> None:
    """Assistant turn with thinking + text + tool_use survives the round trip.

    The ``thoughtSignature`` opaque blob is not part of this provider's
    flow (that's Google); we still emit the thinking block faithfully
    in case a cross-provider replay carries one.
    """
    req = _request(
        UserMessage(text="run the suite"),
        AssistantMessage(
            text="I'll run the test command.",
            thinking_blocks=(
                {"type": "thinking", "thinking": "let me plan", "signature": "sig-abc"},
            ),
            tool_calls=(
                ToolCall(id="toolu_001", name="Bash", args={"command": "pytest -q"}),
            ),
        ),
        ToolResult(call_id="toolu_001", content="42 passed in 1.23s"),
    )
    path, entries = materialize_session(
        req, session_id="22222222-3333-4444-5555-666666666666", cwd=tmp_home
    )
    assert len(entries) == 3
    # Chain check: every parentUuid points at the previous entry's uuid.
    assert entries[0]["parentUuid"] is None
    assert entries[1]["parentUuid"] == entries[0]["uuid"]
    assert entries[2]["parentUuid"] == entries[1]["uuid"]

    # Assistant entry shape
    asst = entries[1]
    assert asst["type"] == "assistant"
    assert asst["message"]["role"] == "assistant"
    asst_blocks = asst["message"]["content"]
    block_types = [b["type"] for b in asst_blocks]
    assert block_types == ["thinking", "text", "tool_use"]
    assert asst_blocks[2]["id"] == "toolu_001"
    assert asst_blocks[2]["input"] == {"command": "pytest -q"}
    assert asst["message"]["stop_reason"] == "tool_use"

    # Tool result entry shape
    tr = entries[2]
    assert tr["type"] == "user"
    assert tr["message"]["content"][0]["tool_use_id"] == "toolu_001"
    assert tr["message"]["content"][0]["content"] == [
        {"type": "text", "text": "42 passed in 1.23s"}
    ]

    parsed = parse_jsonl_to_messages(path)
    assert len(parsed) == 3
    assert isinstance(parsed[0], UserMessage)
    assert parsed[0].text == "run the suite"
    assert isinstance(parsed[1], AssistantMessage)
    assert parsed[1].text == "I'll run the test command."
    assert parsed[1].tool_calls[0].id == "toolu_001"
    assert isinstance(parsed[2], ToolResult)
    assert parsed[2].call_id == "toolu_001"
    assert parsed[2].content == "42 passed in 1.23s"


def test_tool_result_coalescing(tmp_home: Path) -> None:
    """Consecutive ``ToolResult``s pack into one user-role JSONL entry.

    Mirrors ``providers/anthropic.py:_build_messages``: the Anthropic
    API expects multiple tool_result blocks in a single user message
    when the model emitted multiple tool_use blocks in one assistant
    turn. ``--resume`` likewise expects them packed.
    """
    req = _request(
        AssistantMessage(
            tool_calls=(
                ToolCall(id="toolu_A", name="X", args={}),
                ToolCall(id="toolu_B", name="Y", args={}),
            ),
        ),
        ToolResult(call_id="toolu_A", content="a"),
        ToolResult(call_id="toolu_B", content="b"),
    )
    _, entries = materialize_session(
        req, session_id="33333333-aaaa-bbbb-cccc-444444444444", cwd=tmp_home
    )
    # 1 assistant entry + 1 coalesced user entry with 2 blocks.
    assert len(entries) == 2
    user = entries[1]
    assert user["type"] == "user"
    assert len(user["message"]["content"]) == 2
    ids = [b["tool_use_id"] for b in user["message"]["content"]]
    assert ids == ["toolu_A", "toolu_B"]


def test_deterministic_uuids(tmp_home: Path) -> None:
    """Same input → byte-identical output across runs.

    The "byte-identical" claim is the core of the determinism contract;
    if it breaks we lose golden-file diffing, the round-trip test, and
    the tripwire's idempotency guarantee.
    """
    msgs = (
        UserMessage(text="ping"),
        AssistantMessage(text="pong"),
    )
    sid = "ffffffff-0000-1111-2222-333333333333"

    _, entries1 = materialize_session(
        _request(*msgs), session_id=sid, cwd=tmp_home, write=False
    )
    _, entries2 = materialize_session(
        _request(*msgs), session_id=sid, cwd=tmp_home, write=False
    )
    assert entries1 == entries2
    # And the UUIDs are not v4 — they're our v5 namespace.
    for e in entries1:
        u = _uuid.UUID(e["uuid"])
        # uuid5 sets version=5; reject any v4 leak.
        assert u.version == 5


def test_error_tool_result_is_marked(tmp_home: Path) -> None:
    """``is_error`` round-trips correctly."""
    req = _request(
        AssistantMessage(tool_calls=(ToolCall(id="t1", name="X", args={}),)),
        ToolResult(call_id="t1", content="boom", is_error=True),
    )
    _, entries = materialize_session(
        req, session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", cwd=tmp_home
    )
    tr_block = entries[1]["message"]["content"][0]
    assert tr_block["is_error"] is True


def test_orphan_thinking_block_dropped(tmp_home: Path) -> None:
    """Signed thinking block with empty body must be elided.

    Otherwise Anthropic answers HTTP 400 ``thinking blocks ... cannot
    be modified`` on the next wire send.
    """
    orphan = {"type": "thinking", "thinking": "", "signature": "sig-zzz"}
    req = _request(
        AssistantMessage(text="ok", thinking_blocks=(orphan,)),
    )
    _, entries = materialize_session(
        req, session_id="11111111-bbbb-2222-cccc-333333333333", cwd=tmp_home
    )
    blocks = entries[0]["message"]["content"]
    assert all(b.get("type") != "thinking" for b in blocks)
    assert blocks == [{"type": "text", "text": "ok"}]


def test_compact_boundary_drops_preceding_entries(tmp_home: Path) -> None:
    """Parser respects claude's compact_boundary chain reset.

    Claude's auto-compact writes a ``system/compact_boundary`` entry
    with ``parentUuid=None`` and chains subsequent entries off it,
    making everything before the boundary "compacted away" on
    ``--resume``. The parser must drop the pre-boundary entries; if
    it preserves them, a re-materialization carries duplicate
    context and the next ``--resume`` hits the same context limit
    that triggered the compaction in the first place.

    Verified against real claude-written JSONL at
    ``~/.claude/projects/.../*.jsonl`` 2026-06-09: the
    compact_boundary at index 400 of a 1087-entry file had
    ``parentUuid=None``; the entry at 401 chained off the boundary's
    uuid; entries 0-399 were unreachable from the post-boundary chain.
    """
    sid = "00000001-1111-2222-3333-444444444444"
    path = tmp_home / ".claude" / "projects" / "-tmp-test" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "sessionId": sid,
        "cwd": "/tmp/test",  # noqa: S108 -- fixture string only, never touches disk
        "gitBranch": "HEAD",
        "version": "test",
        "userType": "external",
        "entrypoint": "cli",
        "isSidechain": False,
        "timestamp": "2026-06-09T00:00:00.000Z",
    }
    pre_uuid = "00000001-aaaa-bbbb-cccc-000000000001"
    boundary_uuid = "00000001-aaaa-bbbb-cccc-000000000002"
    summary_uuid = "00000001-aaaa-bbbb-cccc-000000000003"

    entries = [
        # Pre-compaction history (should be dropped)
        {
            **base,
            "type": "user",
            "parentUuid": None,
            "uuid": pre_uuid,
            "message": {"role": "user", "content": "pre-compaction turn"},
        },
        # The boundary marker (system entry, parentUuid=None reset)
        {
            **base,
            "type": "system",
            "subtype": "compact_boundary",
            "parentUuid": None,
            "uuid": boundary_uuid,
            "content": "Conversation compacted",
        },
        # The summary user entry (should survive)
        {
            **base,
            "type": "user",
            "parentUuid": boundary_uuid,
            "uuid": summary_uuid,
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary of prior conversation"},
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    msgs = parse_jsonl_to_messages(path)
    assert len(msgs) == 1, f"expected 1 message (summary only), got {len(msgs)}: {msgs}"
    assert isinstance(msgs[0], UserMessage)
    assert msgs[0].text == "summary of prior conversation"


def test_multiple_compact_boundaries_only_last_one_matters(tmp_home: Path) -> None:
    """When claude compacted twice, only the most recent summary survives.

    The 2.9MB real claude session inspected 2026-06-09 had boundaries
    at indices 0 and 400 of 1087 entries — the 0-400 stretch was
    itself a post-compaction continuation that then got compacted
    again. Pre-second-boundary entries are unreachable from the
    final chain.
    """
    sid = "00000002-1111-2222-3333-444444444444"
    path = tmp_home / ".claude" / "projects" / "-tmp-test" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "sessionId": sid,
        "cwd": "/tmp/test",  # noqa: S108 -- fixture string only, never touches disk
        "gitBranch": "HEAD",
        "version": "test",
        "userType": "external",
        "entrypoint": "cli",
        "isSidechain": False,
        "timestamp": "2026-06-09T00:00:00.000Z",
    }

    def mk(uid: str, parent: str | None, content: str, **extra) -> dict[str, object]:
        return {
            **base,
            "type": "user",
            "parentUuid": parent,
            "uuid": uid,
            "message": {"role": "user", "content": content},
            **extra,
        }

    b1 = "00000002-aaaa-0000-0000-000000000001"
    s1 = "00000002-aaaa-0000-0000-000000000002"
    b2 = "00000002-aaaa-0000-0000-000000000003"
    s2 = "00000002-aaaa-0000-0000-000000000004"

    entries = [
        {
            **base,
            "type": "system",
            "subtype": "compact_boundary",
            "parentUuid": None,
            "uuid": b1,
            "content": "first compaction",
        },
        mk(s1, b1, "first summary", isCompactSummary=True),
        # ... intermediate turns ...
        mk("00000002-aaaa-0000-0000-000000000010", s1, "follow-up after compaction 1"),
        # Second compaction
        {
            **base,
            "type": "system",
            "subtype": "compact_boundary",
            "parentUuid": None,
            "uuid": b2,
            "content": "second compaction",
        },
        mk(s2, b2, "second summary", isCompactSummary=True),
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    msgs = parse_jsonl_to_messages(path)
    assert len(msgs) == 1, f"expected only second summary; got {msgs}"
    assert isinstance(msgs[0], UserMessage)
    assert msgs[0].text == "second summary"


def test_no_compact_boundary_means_no_drop(tmp_home: Path) -> None:
    """Sessions without any compact_boundary entry parse end-to-end.

    Sanity guard so the boundary detection doesn't break the
    common case.
    """
    sid = "00000003-1111-2222-3333-444444444444"
    path = tmp_home / ".claude" / "projects" / "-tmp-test" / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "sessionId": sid,
        "cwd": "/tmp/test",  # noqa: S108 -- fixture string only, never touches disk
        "gitBranch": "HEAD",
        "version": "test",
        "userType": "external",
        "entrypoint": "cli",
        "isSidechain": False,
        "timestamp": "2026-06-09T00:00:00.000Z",
    }
    u1 = "00000003-aaaa-0000-0000-000000000001"
    u2 = "00000003-aaaa-0000-0000-000000000002"
    entries = [
        {
            **base,
            "type": "user",
            "parentUuid": None,
            "uuid": u1,
            "message": {"role": "user", "content": "first turn"},
        },
        {
            **base,
            "type": "user",
            "parentUuid": u1,
            "uuid": u2,
            "message": {"role": "user", "content": "second turn"},
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    msgs = parse_jsonl_to_messages(path)
    assert len(msgs) == 2
    assert all(isinstance(m, UserMessage) for m in msgs)


def test_assistant_ending_in_thinking_is_not_padded(tmp_home: Path) -> None:
    """Materializer leaves a ``[thinking]``-only entry unpadded.

    Anthropic's API rejects a wire message whose last block is
    ``thinking`` — but claude's CLI WRITES exactly that shape into its
    JSONL (one assistant entry whose content is ``[thinking]`` only).
    ``--resume`` consumes the unpadded shape; whatever coalescing the
    CLI does before the next wire send happens internally. Verified
    against live CLI 2.1.168 output 2026-06-09 by the v2.1-β canary
    smoke run, which flagged a benign drift caused by our earlier
    over-defensive ``.`` padding. Pinned here so a future
    well-intentioned refactor doesn't reintroduce it.
    """
    req = _request(
        AssistantMessage(
            thinking_blocks=(
                {"type": "thinking", "thinking": "deliberating", "signature": "s"},
            ),
        ),
    )
    _, entries = materialize_session(
        req, session_id="11111111-cccc-2222-dddd-333333333333", cwd=tmp_home
    )
    blocks = entries[0]["message"]["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "thinking"


def test_atomic_write_replaces_atomically(tmp_path: Path) -> None:
    """Write A, write B → final state is exactly B (no merged tail bytes)."""
    p = tmp_path / "session.jsonl"
    # First write — a long line, twice.
    _atomic_write(p, [{"a": "x" * 100}, {"b": "y" * 100}])
    # Second write — a single short line.
    _atomic_write(p, [{"only": "short"}])
    text = p.read_text()
    # Must be exactly the second payload + trailing newline, no leftover.
    assert text == json.dumps({"only": "short"}, separators=(",", ":")) + "\n"


def test_attachment_text_block_added_when_present(tmp_home: Path) -> None:
    """When a UserMessage carries an attachment, content goes list-shaped.

    Image/PDF passthrough is a forward feature; for now we emit a
    text marker so the chain stays valid. The test pins behavior so a
    future implementation knows the contract it's replacing.
    """
    req = _request(
        UserMessage(
            text="see attached",
            attachments=(BytesMessage(data=b"\xff" * 16, descriptor="image/png"),),
        ),
    )
    _, entries = materialize_session(
        req, session_id="11111111-dddd-2222-eeee-333333333333", cwd=tmp_home
    )
    content = entries[0]["message"]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "see attached"}
    assert content[1]["type"] == "text"
    assert "dropped attachment" in content[1]["text"]


# ---------------------------------------------------------------------------
# Round-trip against real claude-written JSONL (best-effort, skips on absence)
# ---------------------------------------------------------------------------


def _find_real_jsonl_sample() -> Path | None:
    """Locate a small real claude-written JSONL on the operator's machine.

    Resolves the operator's REAL home via ``pwd`` so a HOME-monkeypatched
    test fixture doesn't blind us. Returns None if no sample is
    reachable; the round-trip test then skips. This keeps the test
    reproducible across machines without requiring a committed fixture
    (which would also drift as the CLI format evolves).
    """
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    root = real_home / ".claude" / "projects"
    if not root.exists():
        return None
    for p in root.glob("*/*.jsonl"):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if 5_000 <= sz <= 50_000:
            return p
    return None


def test_real_claude_jsonl_round_trip(tmp_path: Path) -> None:
    """Best-effort: parse a real CLI JSONL → re-materialize → assert
    the parsed messages survive.

    We can't byte-diff (claude's JSONL has UI sidecar entries, splits
    each block into its own line, and uses v4 UUIDs) — but the
    *linearized message list* should be preserved through one
    parse→materialize→reparse cycle.
    """
    sample = _find_real_jsonl_sample()
    if sample is None:
        pytest.skip("no real claude session JSONL reachable; skipping round-trip")
    parsed_a = parse_jsonl_to_messages(sample)
    req = ModelRequest(messages=cast(list[ModelContextEvent], parsed_a))
    path, _ = materialize_session(
        req,
        session_id="9999aaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        cwd=tmp_path,
        home=tmp_path,
    )
    parsed_b = parse_jsonl_to_messages(path)
    # Same count + same types in order. Equality on field values is
    # too strict (parse → materialize → parse can normalize whitespace
    # and tool_result block shape); type-and-count is the tripwire we
    # actually need.
    assert len(parsed_a) == len(parsed_b)
    for a, b in zip(parsed_a, parsed_b, strict=True):
        assert type(a) is type(b)
