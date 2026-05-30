"""Tests for ``agent.session_io``: v4 JSONL persistence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import dataclasses
import json

import pytest

from sagent.agent import (
    runtime as agent_runtime,
    session_io,
)
from sagent.agent.agent import Agent
from sagent.agent.context import resolve_context, validate_context
from sagent.agent.session_io import (
    PersistentAgentRecord,
    SessionMeta,
    append_context_repair,
    append_session,
    load_persistent_agents,
    load_session,
    rebuild_content_cache,
    repair_dangling_tool_calls,
    restore_model,
    restore_tool_state,
    serialize_tool_state,
)
from sagent.tools.core import ReadCacheEntry, ToolState
from sagent.types.model import ModelRequest, ModelResponse, Pricing
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    BytesMessage,
    ModelContextEvent,
    ModelServiceSuspended,
    SaveSession,
    ServiceErrorSnapshot,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
)


class _RuntimeModel:
    async def stream(
        self,
        history: list[ModelContextEvent],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_text, on_thinking
        return AssistantMessage(text="")


@dataclass(slots=True, kw_only=True)
class _NoopModel:
    model_id: str = "noop"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8_000
    max_image_bytes: int = 5 * 1024 * 1024

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def approx_text_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 256

    def approx_request_tokens(self, request: ModelRequest) -> int:
        del request
        return 1

    async def actual_text_tokens(self, text: str) -> int:
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        return ModelResponse(message=AssistantMessage(text=""))


def _records_from(entries: list[ModelContextEvent]) -> list[TapeRecord]:
    """Wrap each entry as a ``ReferrableTapeEvent`` with a synthetic ref."""
    return [
        ReferrableTapeEvent(ref=TapeRef(session_id="abc", ordinal=i), event=e)
        for i, e in enumerate(entries)
    ]


def _history_from_tape(tape: list[TapeRecord]) -> list[ModelContextEvent]:
    """Resolve a loaded tape to its provider-facing entries."""
    return resolve_context(tape).messages


def test_serialize_tool_state_empty_has_bash_cwd() -> None:
    blob = serialize_tool_state(ToolState())
    assert "bash_cwd" in blob


def test_serialize_tool_state_round_trip(tmp_path: Path) -> None:
    state = ToolState()
    state.bash_cwd = str(tmp_path)
    state.depth = 2
    state.additional_dirs = ["/tmp/a"]  # noqa: S108 -- placeholder
    state.read_cache["/tmp/x.txt"] = ReadCacheEntry(  # noqa: S108
        offset=0, limit=100, last_lines=10, mtime=1234.5
    )
    blob = serialize_tool_state(state)
    restored = ToolState()
    restore_tool_state(restored, blob)
    assert restored.bash_cwd == state.bash_cwd
    assert restored.depth == state.depth
    assert restored.additional_dirs == state.additional_dirs
    assert restored.read_cache["/tmp/x.txt"].mtime == 1234.5  # noqa: S108


def _round_trip(entry: ModelContextEvent, tmp_path: Path) -> ModelContextEvent:
    """Write ``entry`` to a fresh session and re-load the first record."""
    history = _round_trip_history([entry], tmp_path)
    assert len(history) == 1
    return history[0]


def _round_trip_history(
    entries: list[ModelContextEvent], tmp_path: Path
) -> list[ModelContextEvent]:
    """Write ``entries`` to a fresh session and return the reloaded history."""
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from(entries),
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    return _history_from_tape(tape)


def test_user_message_round_trip(tmp_path: Path) -> None:
    out = _round_trip(UserMessage(text="hello"), tmp_path)
    assert isinstance(out, UserMessage)
    assert out.text == "hello"


def test_agent_send_message_round_trip(tmp_path: Path) -> None:
    out = _round_trip(AgentSendMessage(source="reviewer", text="finding"), tmp_path)
    assert isinstance(out, AgentSendMessage)
    assert out.source == "reviewer"
    assert out.text == "finding"


def test_tool_result_splice_update_persists_through_reload(tmp_path: Path) -> None:
    """Splice updates of an existing ``ToolResult`` must survive session reload.

    Bug repro: when ``_run_tool_and_post`` posts a ``DetachedResult``
    that splices into a ``[detached]`` placeholder, the runtime mutates
    ``history[i]`` in memory. The persistence observer only writes new
    entries (``delta = history[persisted_len:]``), so the splice never
    reaches disk. On session resume the loader reconstructs history
    with the stale ``[detached]`` content, losing the real tool output.

    The fix: the persistence layer must support re-emitting an existing
    entry (same ``id``, updated content). The loader must dedupe by
    ``id`` keeping the LATEST occurrence so the spliced content wins.
    Append-only schema preserved.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")

    # Step 1: write a complete tool-use pair (assistant -> [detached]).
    assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="toolu_bash_1", name="Bash", args={}),),
    )
    original = ToolResult(call_id="toolu_bash_1", content="[detached]")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from([assistant, original]),
    )

    # Step 2: write an override that suppresses the placeholder and
    # injects the real bash output (simulating the splice).
    spliced = ToolResult(
        id=original.id,
        call_id="toolu_bash_1",
        content="hello world\n",
    )
    placeholder_records = _records_from([assistant, original])
    placeholder_ref = placeholder_records[1].ref
    parent_ref = placeholder_records[0].ref
    splice = ContextSplice(
        ref=TapeRef(session_id="abc", ordinal=2),
        mask=((placeholder_ref, placeholder_ref),),
        insert_after=parent_ref,
        payload=(spliced,),
        strategy="detached_splice",
        paired_externally=frozenset({"toolu_bash_1"}),
    )
    append_session(session_file, tape_delta=[splice])

    # Step 3: reload + resolve and assert the spliced content wins.
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    messages = resolve_context(tape).messages
    matching = [
        e for e in messages if isinstance(e, ToolResult) and e.call_id == "toolu_bash_1"
    ]
    assert len(matching) == 1
    assert matching[0].content == "hello world\n"


def test_user_message_with_attachment(tmp_path: Path) -> None:
    att = BytesMessage(data=b"\x89PNG", descriptor="image/png")
    out = _round_trip(UserMessage(text="see", attachments=(att,)), tmp_path)
    assert isinstance(out, UserMessage)
    assert out.text == "see"
    assert len(out.attachments) == 1
    assert out.attachments[0].data == b"\x89PNG"
    assert out.attachments[0].descriptor == "image/png"


def test_assistant_message_round_trip(tmp_path: Path) -> None:
    # Pair with a tool_result so the resume-time orphan repair doesn't
    # synthesize an ``[interrupted]`` entry.
    tc = ToolCall(id="c1", name="Echo", args={"msg": "hi"})
    asst = AssistantMessage(text="ok", tool_calls=(tc,))
    res = ToolResult(call_id="c1", content="done")
    history = _round_trip_history([asst, res], tmp_path)
    assert len(history) == 2
    out = history[0]
    assert isinstance(out, AssistantMessage)
    assert out.text == "ok"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "c1"
    assert out.tool_calls[0].name == "Echo"
    assert dict(out.tool_calls[0].args) == {"msg": "hi"}


def test_assistant_message_thinking_blocks_round_trip(tmp_path: Path) -> None:
    block = {"type": "thinking", "thinking": "step 1"}
    out = _round_trip(AssistantMessage(text="ok", thinking_blocks=(block,)), tmp_path)
    assert isinstance(out, AssistantMessage)
    assert out.thinking_blocks[0]["thinking"] == "step 1"


def test_tool_result_round_trip(tmp_path: Path) -> None:
    # Pair with the matching tool_use so the result isn't dropped as orphan.
    tc = ToolCall(id="c1", name="Echo", args={})
    asst = AssistantMessage(tool_calls=(tc,))
    res = ToolResult(
        call_id="c1", content="ran", diff="--- a\n+++ b\n", hint="hi", summary="1"
    )
    history = _round_trip_history([asst, res], tmp_path)
    assert len(history) == 2
    out = history[1]
    assert isinstance(out, ToolResult)
    assert out.call_id == "c1"
    assert out.content == "ran"
    assert out.diff == "--- a\n+++ b\n"


def test_load_session_missing_returns_none(tmp_path: Path) -> None:
    assert load_session(tmp_path, {}) is None


def test_append_context_repair_masks_current_view(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    old_records = _records_from(
        [
            UserMessage(text="x" * 1_000),
            UserMessage(text="y" * 1_000),
        ]
    )
    state = ToolState()
    state.invoked_skills.add("debug")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=old_records,
        tool_state_snapshot=serialize_tool_state(state),
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    assert (
        sum(
            len(m.text)
            for m in resolve_context(tape).messages
            if isinstance(m, UserMessage)
        )
        == 2_000
    )

    repair = append_context_repair(
        session_file,
        tape,
        payload=[UserMessage(text="slim repair")],
        strategy="manual_repair",
    )

    loaded_after = load_session(tmp_path, {})
    assert loaded_after is not None
    _, repaired_tape, repaired_state = loaded_after
    messages = resolve_context(repaired_tape).messages
    assert repair.strategy == "manual_repair"
    assert [m.text for m in messages if isinstance(m, UserMessage)] == ["slim repair"]
    assert repaired_state.invoked_skills == set()
    validate_context(messages)


def test_clear_barrier_drops_prior_history(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc", model_id="m", provider="P", auth="env")
    # Explicit refs to avoid collision between two ``_records_from`` calls.
    old_ref = TapeRef(session_id="abc", ordinal=0)
    splice_ref = TapeRef(session_id="abc", ordinal=1)
    new_ref = TapeRef(session_id="abc", ordinal=2)
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[ReferrableTapeEvent(ref=old_ref, event=UserMessage(text="old"))],
    )
    append_session(
        session_file,
        tape_delta=[
            ContextSplice(
                ref=splice_ref,
                mask=((old_ref, old_ref),),
                insert_after=None,
                payload=(),
                strategy="clear",
            )
        ],
    )
    append_session(
        session_file,
        tape_delta=[ReferrableTapeEvent(ref=new_ref, event=UserMessage(text="new"))],
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    texts = [e.text for e in history if isinstance(e, UserMessage)]
    assert texts == ["new"]


def test_meta_latest_wins(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        meta=SessionMeta(session_id="old", model_id="m").serialize(),
    )
    append_session(
        session_file,
        meta=SessionMeta(session_id="new", model_id="m").serialize(),
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    meta, _, _ = loaded
    assert meta.session_id == "new"


def test_tool_state_post_clear_wins(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    s1 = ToolState()
    s1.bash_cwd = "/old"
    s2 = ToolState()
    s2.bash_cwd = "/new"
    append_session(session_file, tool_state_snapshot=serialize_tool_state(s1))
    append_session(
        session_file,
        tape_delta=[
            ContextSplice(
                ref=TapeRef(session_id="abc", ordinal=999),
                mask=(),
                insert_after=None,
                payload=(),
                strategy="clear",
            )
        ],
    )
    append_session(session_file, tool_state_snapshot=serialize_tool_state(s2))
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, _, state = loaded
    assert state.bash_cwd == "/new"


def test_append_session_no_ops_on_empty_batch(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(session_file)
    assert not session_file.exists()


def test_load_session_orders_loaded_tape_by_ordinal(tmp_path: Path) -> None:
    """Persisted and synthetic refs load in canonical ordinal order."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 1},
            "type": "user",
            "text": "one",
        },
        {"kind": "history", "type": "user", "text": "synthetic"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "zero",
        },
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    assert [record.ref.ordinal for record in tape] == [0, 1, 2]
    user_messages: list[UserMessage] = []
    for entry in _history_from_tape(tape):
        assert isinstance(entry, UserMessage)
        user_messages.append(entry)
    assert [entry.text for entry in user_messages] == ["zero", "one", "synthetic"]


def test_legacy_override_with_gap_preserves_unmasked_ref(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    refs = [TapeRef(session_id="abc", ordinal=i) for i in range(4)]
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "first",
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 1},
            "type": "user",
            "text": "middle",
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 2},
            "type": "user",
            "text": "last",
        },
        {
            "kind": "context_override",
            "ref": {"session_id": "abc", "ordinal": 3},
            "suppresses": [
                {"session_id": "abc", "ordinal": 0},
                {"session_id": "abc", "ordinal": 2},
            ],
            "inject_after": None,
            "payload": [{"type": "user", "text": "replacement"}],
        },
    )

    loaded = load_session(tmp_path, {})

    assert loaded is not None
    _, tape, _ = loaded
    messages = resolve_context(tape).messages
    assert [entry.text for entry in messages if isinstance(entry, UserMessage)] == [
        "replacement",
        "middle",
    ]
    splice = tape[-1]
    assert isinstance(splice, ContextSplice)
    assert splice.mask == ((refs[0], refs[0]), (refs[2], refs[2]))


@pytest.mark.asyncio
async def test_persistence_skips_externally_replayed_records(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    persisted = ReferrableTapeEvent(
        ref=TapeRef(session_id="abc", ordinal=0),
        event=UserMessage(text="persisted"),
    )
    append_session(session_file, tape_delta=[persisted])
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.replay_tape([persisted])

    agent.runtime.publish(SaveSession())

    lines = session_file.read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines if '"kind": "history"' in line]
    assert texts.count("persisted") == 1


@pytest.mark.asyncio
async def test_save_session_skips_tool_state_when_unchanged(tmp_path: Path) -> None:
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.append_history(UserMessage(text="x"))
    agent.runtime.publish(SaveSession())
    session_file = tmp_path / "session.jsonl"
    size = session_file.stat().st_size

    for _ in range(5):
        agent.runtime.publish(SaveSession())

    assert session_file.stat().st_size == size


@pytest.mark.asyncio
async def test_save_session_writes_tool_state_when_changed(tmp_path: Path) -> None:
    agent = Agent(model=_NoopModel(), session_dir=tmp_path)
    agent.runtime.publish(SaveSession())
    lines_before = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()

    agent.tool_state.bash_cwd = "/changed"
    agent.runtime.publish(SaveSession())

    lines_after = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines_after) > len(lines_before)
    assert json.loads(lines_after[-1])["bash_cwd"] == "/changed"


@pytest.mark.asyncio
async def test_load_session_with_repair_preserves_meta_bash_cwd(
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(
        session_id="dangling",
        model_id="m",
        provider="P",
        auth="env",
        bash_cwd="/project",
    )
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="dangling", ordinal=0),
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="call_1", name="Bash", args={}),)
                ),
            ),
        ],
    )

    loaded = load_session(tmp_path, {})

    assert loaded is not None
    _, tape, state = loaded
    validate_context(resolve_context(tape).messages)
    assert state.bash_cwd == "/project"


def test_out_of_order_barrier_resets_prior_tool_state(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    old_state = ToolState()
    old_state.invoked_skills.add("debug")
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {"kind": "tool_state", **serialize_tool_state(old_state)},
        {
            "kind": "context_splice",
            "ref": {"session_id": "abc", "ordinal": 2},
            "mask": [
                [
                    {"session_id": "abc", "ordinal": 0},
                    {"session_id": "abc", "ordinal": 1},
                ]
            ],
            "insert_after": None,
            "payload": [{"type": "user", "text": "after"}],
            "strategy": "summary",
            "paired_externally": [],
        },
        {
            "kind": "history",
            "ref": {"session_id": "abc", "ordinal": 0},
            "type": "user",
            "text": "before",
        },
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, _, state = loaded
    assert state.invoked_skills == set()


def test_load_session_dangling_repair_resets_prior_tool_state(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="dangling", model_id="m", provider="P", auth="env")
    state = ToolState()
    state.invoked_skills.add("debug")
    state.bash_cwd = "/stale"
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="dangling", ordinal=0),
                event=AssistantMessage(
                    tool_calls=(ToolCall(id="call_1", name="Bash", args={}),)
                ),
            ),
        ],
        tool_state_snapshot=serialize_tool_state(state),
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, repaired_state = loaded
    validate_context(resolve_context(tape).messages)
    assert repaired_state.invoked_skills == set()
    assert repaired_state.bash_cwd == ToolState().bash_cwd


def test_load_session_repairs_orphan_tool_result(tmp_path: Path) -> None:
    """Disk-loaded ``ToolResult`` with no matching ``tool_use`` is hidden on resume.

    Regression: a session that crashed mid-tool or that imported a
    foreign history could persist a ``ToolResult`` whose ``call_id``
    has no preceding ``AssistantMessage.tool_calls`` match.
    ``repair_dangling_tool_calls`` drops the orphan from its returned
    list, but the loaded tape still contained the orphan as a
    ``ReferrableTapeEvent``; the resolved context surfaced it and the next
    provider call rejected the request with 400. The load-time tape
    repair must append a suppression override so the resolved view
    is wire-format-valid out of the box.
    """
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="orphan", model_id="m", provider="P", auth="env")
    # Persist: user -> orphan ToolResult -> user.
    user1 = UserMessage(text="kick off")
    orphan = ToolResult(call_id="ghost_1", content="dangling")
    user2 = UserMessage(text="continue")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=0), event=user1
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=1), event=orphan
            ),
            ReferrableTapeEvent(
                ref=TapeRef(session_id="orphan", ordinal=2), event=user2
            ),
        ],
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    resolved = resolve_context(tape).messages
    # The orphan must not surface in the resolved view; user messages survive.
    assert not any(
        isinstance(m, ToolResult) and m.call_id == "ghost_1" for m in resolved
    ), f"orphan ToolResult should be suppressed on load; resolved: {resolved}"
    users = [m for m in resolved if isinstance(m, UserMessage)]
    assert len(users) == 1
    assert users[0].text == f"{user1.text}\n\n{user2.text}"


def test_load_session_repairs_orphan_tool_result_from_splice_payload(
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(
        session_id="orphan-splice", model_id="m", provider="P", auth="env"
    )
    user = UserMessage(text="go")
    assistant = AssistantMessage(
        tool_calls=(ToolCall(id="kept", name="Echo", args={}),)
    )
    result = ToolResult(call_id="kept", content="done")
    summary = UserMessage(text="[summary]")
    orphan = ToolResult(call_id="ghost", content="late")
    refs = [TapeRef(session_id="orphan-splice", ordinal=i) for i in range(5)]
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=[
            ReferrableTapeEvent(ref=refs[0], event=user),
            ReferrableTapeEvent(ref=refs[1], event=assistant),
            ReferrableTapeEvent(ref=refs[2], event=result),
            ContextSplice(
                ref=refs[3],
                mask=((refs[0], refs[2]),),
                insert_after=None,
                payload=(summary,),
                strategy="summary",
            ),
            ContextSplice(
                ref=refs[4],
                mask=((refs[2], refs[2]),),
                insert_after=refs[1],
                payload=(orphan,),
                strategy="detached_splice",
                paired_externally=frozenset({"ghost"}),
            ),
        ],
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    resolved = resolve_context(tape).messages
    validate_context(resolved)
    assert resolved == [summary]
    assert any(
        isinstance(record, ContextSplice)
        and record.strategy == "orphan_tool_result_repair"
        and record.mask == ((refs[0], refs[4]),)
        for record in tape
    )
    runtime = agent_runtime.AgentRuntime(model=_RuntimeModel())
    runtime.replay_tape(tape)
    runtime.append_splice(
        mask=((runtime.tape[0].ref, runtime.tape[-1].ref),),
        insert_after=None,
        payload=(UserMessage(text="[next summary]"),),
        strategy="summary",
    )


def test_append_session_writes_meta_then_tape_delta(tmp_path: Path) -> None:
    """Within one batch: ``meta`` precedes any tape records."""
    session_file = tmp_path / "session.jsonl"
    meta = SessionMeta(session_id="abc")
    append_session(
        session_file,
        meta=meta.serialize(),
        tape_delta=_records_from([UserMessage(text="x")]),
    )
    lines = session_file.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    kinds = [r["kind"] for r in parsed]
    assert kinds == ["meta", "history"]


def _service_suspended(account: str | None = "default") -> ModelServiceSuspended:
    """Build a persisted service-suspension fixture."""
    return ModelServiceSuspended(
        provider="OpenAISubscription",
        auth="credentials",
        account=account,
        model_id="gpt-5.5",
        retry_at=1_800_000_000.0,
        delay_sec=14_868.0,
        server_supplied=True,
        error=ServiceErrorSnapshot(
            type_name="RateLimitError",
            message="limited",
            status=429,
            headers={"retry-after": "14868"},
            body='{"error":"rate_limit"}',
        ),
    )


def test_append_session_writes_model_service_suspended_event(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(session_file, runtime_events=[_service_suspended()])

    record = json.loads(session_file.read_text(encoding="utf-8"))
    assert record == {
        "kind": "runtime_event",
        "type": "model_service_suspended",
        "timestamp": record["timestamp"],
        "provider": "OpenAISubscription",
        "auth": "credentials",
        "account": "default",
        "model_id": "gpt-5.5",
        "retry_at": 1_800_000_000.0,
        "delay_sec": 14_868.0,
        "server_supplied": True,
        "error": {
            "type_name": "RateLimitError",
            "message": "limited",
            "status": 429,
            "headers": {"retry-after": "14868"},
            "body": '{"error":"rate_limit"}',
        },
    }


def test_load_session_decodes_model_service_suspended_event(tmp_path: Path) -> None:
    append_session(tmp_path / "session.jsonl", runtime_events=[_service_suspended()])

    loaded = load_session(tmp_path, {})

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (_service_suspended(),)


def test_model_service_suspended_account_none_round_trips(tmp_path: Path) -> None:
    append_session(
        tmp_path / "session.jsonl", runtime_events=[_service_suspended(None)]
    )

    loaded = load_session(tmp_path, {})

    assert loaded is not None
    meta, _, _ = loaded
    assert meta.runtime_events == (_service_suspended(None),)


def test_append_session_writes_persistent_agent_lifecycle(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        persistent_agents=[
            PersistentAgentRecord(
                label="fix-tools",
                run_id="run-1",
                session_dir=str(tmp_path / "session" / "fix-tools"),
                state="running",
                provider="OpenAISubscription",
                auth="credentials",
                account="default",
                model_id="gpt-5.5",
                tools=("Read", "Edit"),
                system="system text",
                notify_on_asleep=True,
            )
        ],
    )

    record = json.loads(session_file.read_text(encoding="utf-8"))
    assert record == {
        "kind": "persistent_agent",
        "label": "fix-tools",
        "run_id": "run-1",
        "session_dir": str(tmp_path / "session" / "fix-tools"),
        "state": "running",
        "provider": "OpenAISubscription",
        "auth": "credentials",
        "account": "default",
        "model_id": "gpt-5.5",
        "tools": ["Read", "Edit"],
        "system": "system text",
        "notify_on_asleep": True,
        "max_tool_call_rounds": None,
        "max_request_tokens": None,
        "max_response_tokens": None,
        "thinking": None,
        "thinking_state": None,
        "effort": None,
        "cache_ttl": "5m",
        "service_tier": None,
        "max_budget_usd": None,
        "persistent_retry": False,
        "provider_args": {},
        "timestamp": record["timestamp"],
    }


def test_load_persistent_agents_returns_latest_running_records(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    running = PersistentAgentRecord(
        label="fix-tools",
        run_id="run-1",
        session_dir=str(tmp_path / "children" / "run-1"),
        state="running",
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        tools=("Read",),
        system="system text",
        notify_on_asleep=True,
    )
    append_session(session_file, persistent_agents=[running])
    append_session(
        session_file,
        persistent_agents=[dataclasses.replace(running, state="cancelled")],
    )
    still_running = dataclasses.replace(running, run_id="run-2", label="fix-compact")
    append_session(session_file, persistent_agents=[still_running])

    assert load_persistent_agents(tmp_path) == [still_running]


def test_load_persistent_agents_ignores_all_terminal_states(tmp_path: Path) -> None:
    session_file = tmp_path / "session.jsonl"
    running = PersistentAgentRecord(
        label="fix-tools",
        run_id="run-1",
        session_dir=str(tmp_path / "children" / "run-1"),
        state="running",
        provider="OpenAISubscription",
        auth="credentials",
        account="default",
        model_id="gpt-5.5",
        tools=("Read",),
        system="system text",
        notify_on_asleep=True,
    )
    append_session(session_file, persistent_agents=[running])
    for state in ("completed", "failed", "cancelled"):
        append_session(
            session_file,
            persistent_agents=[dataclasses.replace(running, state=state)],
        )

    assert load_persistent_agents(tmp_path) == []


def test_load_persistent_agents_rejects_empty_session_dir(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": "",
            "state": "running",
            "provider": "OpenAISubscription",
            "auth": "credentials",
            "account": None,
            "model_id": "gpt-5.5",
            "tools": ["Read"],
            "system": "system text",
            "notify_on_asleep": True,
        },
    )

    assert load_persistent_agents(tmp_path) == []


def test_persistent_agent_legacy_notify_on_asleep_defaults_true(tmp_path: Path) -> None:
    session_dir = tmp_path / "children" / "run-1"
    _write_jsonl(
        tmp_path / "session.jsonl",
        {
            "kind": "persistent_agent",
            "label": "fix-tools",
            "run_id": "run-1",
            "session_dir": str(session_dir),
            "state": "running",
            "provider": "OpenAISubscription",
            "auth": "credentials",
            "account": None,
            "model_id": "gpt-5.5",
            "tools": ["Read"],
            "system": "system text",
        },
    )

    records = load_persistent_agents(tmp_path)

    assert len(records) == 1
    assert records[0].notify_on_asleep is True
    assert records[0].account is None


def test_session_meta_round_trip() -> None:
    src = SessionMeta(
        session_id="x", model_id="y", provider="P", auth="env", status="busy"
    )
    blob = src.serialize()
    back = SessionMeta.deserialize(blob)
    assert back.session_id == "x"
    assert back.model_id == "y"
    assert back.provider == "P"
    assert back.status == "busy"


def _write_jsonl(path: Path, *records: object) -> None:
    """Write each record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            _ = f.write(json.dumps(r) + "\n")


def test_load_session_skips_unknown_history_type(tmp_path: Path) -> None:
    """Unknown history ``type`` falls through ``_entry_from_json`` → None."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "abc"},
        {"kind": "history", "type": "mystery", "text": "x"},
    )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    # Unknown ``type`` was silently dropped.
    assert history == []


def test_load_session_drops_attachment_with_bad_mime_or_data(tmp_path: Path) -> None:
    """``_att_from_json`` drops attachments missing or malformed fields."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {"kind": "meta", "session_id": "x"},
        {
            "kind": "history",
            "type": "user",
            "text": "with bad atts",
            "attachments": [
                "not a dict",
                {"mime": 5, "data": "abc"},
                {"mime": "image/png", "data": "!!! not base64 !!!"},
                {"mime": "image/png", "data": "aGVsbG8="},  # valid
            ],
        },
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    # Only the valid base64 attachment survives.
    assert len(entry.attachments) == 1
    assert entry.attachments[0].data == b"hello"


def test_load_session_drops_non_list_attachments_and_thinking(
    tmp_path: Path,
) -> None:
    """Non-list ``attachments`` / ``thinking_blocks`` parse to empty."""
    session_file = tmp_path / "session.jsonl"
    _write_jsonl(
        session_file,
        {
            "kind": "history",
            "type": "user",
            "text": "u",
            "attachments": "not-a-list",
        },
        {
            "kind": "history",
            "type": "assistant",
            "text": "a",
            "thinking_blocks": "not-a-list",
        },
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    user = history[0]
    asst = history[1]
    assert isinstance(user, UserMessage)
    assert user.attachments == ()
    assert isinstance(asst, AssistantMessage)
    assert asst.thinking_blocks == ()


def test_load_session_drops_non_dict_tool_calls(tmp_path: Path) -> None:
    """``tool_calls`` items that aren't dicts are skipped."""
    session_file = tmp_path / "session.jsonl"
    bad_tc: object = "not a dict"
    good_tc: dict[str, object] = {"id": "c1", "name": "echo", "args": {}}
    record: dict[str, object] = {
        "kind": "history",
        "type": "assistant",
        "text": "x",
        "tool_calls": [bad_tc, good_tc],
    }
    _write_jsonl(session_file, record)
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    asst = history[0]
    assert isinstance(asst, AssistantMessage)
    assert len(asst.tool_calls) == 1
    assert asst.tool_calls[0].id == "c1"


def test_load_session_skips_blank_lines_and_non_dict_records(tmp_path: Path) -> None:
    """Blank lines and non-dict JSON records are skipped without error."""
    session_file = tmp_path / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with session_file.open("w", encoding="utf-8") as f:
        _ = f.write("\n")  # blank line → ``continue``
        _ = f.write("   \n")  # whitespace-only line
        _ = f.write("42\n")  # not a dict
        _ = f.write(
            json.dumps({"kind": "history", "type": "user", "text": "hi"}) + "\n"
        )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "hi"


def test_load_session_preserves_and_skips_corrupt_lines(tmp_path: Path) -> None:
    """A corrupt JSON line triggers backup and skip; valid lines load."""
    session_file = tmp_path / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with session_file.open("w", encoding="utf-8") as f:
        _ = f.write("{not valid json\n")
        _ = f.write(
            json.dumps({"kind": "history", "type": "user", "text": "after"}) + "\n"
        )

    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, tape, _ = loaded
    history = _history_from_tape(tape)
    assert len(history) == 1
    entry = history[0]
    assert isinstance(entry, UserMessage)
    assert entry.text == "after"
    # A `*.corrupt-*` sibling was created to preserve the original bytes.
    backups = list(tmp_path.glob("session.jsonl.corrupt-*"))
    assert len(backups) == 1


def test_load_session_returns_none_when_file_unreadable(tmp_path: Path) -> None:
    """OSError while reading the session file returns None."""
    session_file = tmp_path / "session.jsonl"
    session_file.touch()

    def _boom(self: Path, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        raise OSError("permission denied")

    with patch.object(Path, "open", _boom):
        assert load_session(tmp_path, {}) is None


def test_load_session_uses_meta_bash_cwd_when_no_snapshot(tmp_path: Path) -> None:
    """``meta.bash_cwd`` seeds ``ToolState`` when no tool_state record exists."""
    session_file = tmp_path / "session.jsonl"
    append_session(
        session_file,
        meta=SessionMeta(session_id="x", bash_cwd="/from/meta").serialize(),
    )
    loaded = load_session(tmp_path, {})
    assert loaded is not None
    _, _, state = loaded
    assert state.bash_cwd == "/from/meta"


def test_preserve_corrupt_session_swallows_oserror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``_preserve_corrupt_session`` logs and returns if write fails."""
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("garbage", encoding="utf-8")

    def _boom(self: Path, data: object) -> int:
        del self, data
        raise OSError("disk full")

    with caplog.at_level("ERROR"), patch.object(Path, "write_bytes", _boom):
        session_io._preserve_corrupt_session(session_file)

    assert "Could not preserve corrupt session file" in caplog.text


def test_restore_tool_state_drops_bad_read_cache_entries() -> None:
    """``restore_tool_state`` ignores non-dict and missing-path read_cache rows."""
    state = ToolState()
    snapshot: dict[str, object] = {
        "bash_cwd": "/x",
        "depth": 0,
        "additional_dirs": [],
        "recent_files": ["", 42, "/real/path"],  # blank, non-str, valid
        "read_cache": [
            "not-a-dict",  # skipped
            {"path": "", "offset": 0},  # skipped (empty path)
            {
                "path": "/p/x.txt",
                "offset": 0,
                "limit": 100,
                "last_lines": 10,
                "mtime": 1.0,
            },
        ],
    }
    restore_tool_state(state, snapshot)
    assert "/p/x.txt" in state.read_cache
    # Only "/real/path" survives the recent_files filter.
    assert any("/real/path" in p for p in state.recent_files)


def test_restore_model_returns_none_when_missing_provider_or_model() -> None:
    """``restore_model`` short-circuits to None for missing fields."""
    assert restore_model(SessionMeta()) is None
    assert restore_model(SessionMeta(provider="P")) is None
    assert restore_model(SessionMeta(model_id="m")) is None


def test_restore_model_returns_none_on_attribute_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider lookup failure (AttributeError) returns None."""
    meta = SessionMeta(provider="DoesNotExist", model_id="m", auth="env")
    with caplog.at_level("WARNING"):
        assert restore_model(meta) is None
    assert "Failed to restore model" in caplog.text


def test_restore_model_success_path() -> None:
    """A working provider builds a model and spec."""

    class _FakeModel:
        model_id: str = "fake-m"

    class _FakeProvider:
        def model(self, model_id: str) -> _FakeModel:
            del model_id
            return _FakeModel()

    class _FakeBuilder:
        def build_provider(
            self, provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            del provider, auth, account
            return _FakeProvider()

    meta = SessionMeta(provider="Fake", model_id="fake-m", auth="env", account="me")
    with patch.object(session_io, "providers_lib", _FakeBuilder()):
        result = restore_model(meta)
    assert result is not None
    _, spec = result
    assert spec.provider == "Fake"
    assert spec.model_id == "fake-m"
    assert spec.account == "me"


def test_repair_synthesizes_missing_tool_result() -> None:
    """C2: orphan tool_use gets a synthetic ``[interrupted]`` placeholder."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    history: list[ModelContextEvent] = [
        UserMessage(text="do X"),
        asst,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert len(repaired) == 3
    last = repaired[-1]
    assert isinstance(last, ToolResult)
    assert last.call_id == "c1"
    assert last.content == "[interrupted]"
    assert last.is_error is True


def test_repair_is_idempotent() -> None:
    """C2: re-running the repair pass over its own output is a no-op."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    history: list[ModelContextEvent] = [
        UserMessage(text="do X"),
        asst,
    ]
    repaired = repair_dangling_tool_calls(history)
    again = repair_dangling_tool_calls(repaired)
    assert [type(x) for x in again] == [type(x) for x in repaired]
    assert len(again) == len(repaired)


def test_repair_drops_orphan_tool_result_with_no_call() -> None:
    """C2: dangling ToolResult lacking a parent AssistantMessage is dropped."""
    orphan = ToolResult(call_id="ghost", content="leftover")
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        orphan,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert len(repaired) == 1
    assert isinstance(repaired[0], UserMessage)


def test_repair_preserves_matching_tool_result_pair() -> None:
    """C2: existing tool_use + tool_result pair stays intact."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="c1", name="N", args={}),))
    res = ToolResult(call_id="c1", content="OK")
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        asst,
        res,
    ]
    repaired = repair_dangling_tool_calls(history)
    assert repaired == history


def test_rebuild_content_cache_from_read(tmp_path: Path) -> None:
    """L5: Read tool result seeds _content_cache so post-resume reads are clean."""
    f = tmp_path / "data.txt"
    body = "hello world\n"
    _ = f.write_text(body)
    asst = AssistantMessage(
        tool_calls=(ToolCall(id="c1", name="Read", args={"file_path": str(f)}),),
    )
    result_text = f"     1\t{body}"
    res = ToolResult(call_id="c1", content=result_text)
    state = ToolState()
    rebuild_content_cache([asst, res], state)
    assert state.has_been_read(str(f))


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
