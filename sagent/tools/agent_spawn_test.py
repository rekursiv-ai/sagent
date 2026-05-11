"""Smoke tests for the v3 ``tools.AgentSpawn`` surface.

Covers the directive-to-args fall-through, model resolution, and the
child-event observer forwarder. End-to-end spawn-and-run is covered by
``agent/agent_test.py`` (and integration-level tests).
"""

from __future__ import annotations

from typing import Any, cast

import dataclasses
import time

import pytest

from sagent.custom_types import (
    ChildDoneEvent,
    ChildEvent,
    Event,
    JsonMessage,
    Message,
    MultipartMessage,
    RecoverableErrorEvent,
    TextChunkEvent,
    TextMessage,
    ThinkingEvent,
    ToolLabelEvent,
    ToolResultEvent,
)
from sagent.lib.json import json_freeze
from sagent.tools.agent_spawn import (
    AgentSpawn,
    ChildStats,
    _build_forwarder,
    _ChildForwarder,
    _pick_field,
)
from sagent.tools.core import (
    ToolState,
    max_depth_var,
    tool_state_context,
)


@dataclasses.dataclass(slots=True, kw_only=True)
class _ParentStub:
    """Minimal parent surface for ``_ChildForwarder`` tests."""

    published: list[Event] = dataclasses.field(default_factory=list)
    active_children: dict[str, ChildStats] = dataclasses.field(default_factory=dict)

    def publish(self, event: Event) -> None:
        self.published.append(event)


class TestAgentSpawnSchema:
    def test_name_and_id(self) -> None:
        spawn = AgentSpawn()
        assert spawn.name == "AgentSpawn"
        assert spawn.tool_id == "application/x-tool-agentspawn"

    def test_required_prompt(self) -> None:
        schema = cast(dict[str, Any], dict(AgentSpawn().directive_schema))
        assert "prompt" in schema["required"]


class TestChildForwarder:
    def _build(self, verbosity: int = 1) -> tuple[_ParentStub, _ChildForwarder]:
        parent = _ParentStub()
        forwarder = _build_forwarder("Agent_0", verbosity, cast(Any, parent))
        assert forwarder is not None
        return parent, forwarder

    def test_text_chunk_forwards_at_verbosity_1(self) -> None:
        parent, fwd = self._build(verbosity=1)
        fwd(TextChunkEvent("hi"))
        assert any(isinstance(e, ChildEvent) for e in parent.published)

    def test_thinking_dropped_at_verbosity_1(self) -> None:
        parent, fwd = self._build(verbosity=1)
        fwd(ThinkingEvent("planning"))
        assert parent.published == []

    def test_thinking_forwarded_at_verbosity_2(self) -> None:
        parent, fwd = self._build(verbosity=2)
        fwd(ThinkingEvent("planning"))
        assert len(parent.published) == 1
        ce = parent.published[0]
        assert isinstance(ce, ChildEvent)
        assert isinstance(ce.inner, ThinkingEvent)

    def test_error_always_forwarded(self) -> None:
        parent, fwd = self._build(verbosity=0)
        fwd(RecoverableErrorEvent(msg=TextMessage("boom", "text/x-error")))
        assert any(isinstance(e, ChildEvent) for e in parent.published)

    def test_error_tool_result_always_forwarded(self) -> None:
        parent, fwd = self._build(verbosity=0)
        result = MultipartMessage(
            (
                TextMessage("qid_1", "text/x-queue-id"),
                TextMessage("oops", "text/x-error"),
            ),
            "multipart/x-tool-result",
        )
        fwd(ToolResultEvent(result))
        assert any(isinstance(e, ChildEvent) for e in parent.published)

    def test_tool_label_forwarded_at_verbosity_1(self) -> None:
        parent, fwd = self._build(verbosity=1)
        fwd(ToolLabelEvent("Bash ls"))
        assert any(isinstance(e, ChildEvent) for e in parent.published)

    def test_emit_done_publishes_child_done_and_clears_active(self) -> None:
        parent = _ParentStub()
        stats = ChildStats(label="Agent_0", start=time.monotonic())
        parent.active_children["Agent_0"] = stats
        forwarder = _ChildForwarder(
            parent_agent=cast(Any, parent),
            forward_set=frozenset(),
            stats=stats,
            label="Agent_0",
        )
        forwarder.emit_done()
        assert any(isinstance(e, ChildDoneEvent) for e in parent.published)
        assert "Agent_0" not in parent.active_children


class TestModelResolution:
    def test_pick_field_prefers_llm_arg(self) -> None:
        assert _pick_field("llm", "factory", "spec") == "llm"

    def test_pick_field_falls_through_to_spec(self) -> None:
        assert _pick_field(None, None, "spec") == "spec"

    def test_resolve_model_returns_error_when_no_spec(self) -> None:
        spawn = AgentSpawn()
        result = spawn._resolve_model(
            provider="P",
            auth=None,
            model_id=None,
            account=None,
            parent_agent=None,
        )
        assert isinstance(result, TextMessage) or (
            isinstance(result, MultipartMessage) and False
        )


def _ms(directive: object) -> Message:
    """Tiny helper to build a tool-use message."""
    return MultipartMessage(
        (
            JsonMessage(
                json_freeze(cast(dict[str, Any], directive)),
                "application/x-tool-agentspawn",
            ),
        ),
        "multipart/x-tool-call",
    )


@pytest.mark.anyio
class TestAgentSpawnRun:
    async def test_max_depth_exceeded_returns_error(self) -> None:
        state = ToolState()
        state.depth = 5
        spawn = AgentSpawn()
        with tool_state_context(state):
            depth_token = max_depth_var.set(2)
            try:
                msg = _ms({"prompt": "do something"})
                result = await spawn.run(msg)
            finally:
                max_depth_var.reset(depth_token)
        assert result.descriptor == "text/x-error"
        assert "max_depth" in str(result.content)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
