"""Tests for the runtime event vocabulary."""

from __future__ import annotations

import pytest

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import Response
from sagent.runtime.tape import ContextSplice, TapeRef
from sagent.runtime.types import (
    CompactComplete,
    DetachedResult,
    Markers,
    ModelContextEvent,
    labeled_agent_send_text,
    wire_role,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ShellCommandResult,
    SystemMessage,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)


def test_agent_text_is_labeled_with_its_sender() -> None:
    entry = AgentToAgentMessage(sender="reviewer", content="finding")
    assert labeled_agent_send_text(entry) == "[from reviewer]: finding"


def test_agent_text_is_labeled_once() -> None:
    entry = AgentToAgentMessage(sender="reviewer", content="[from reviewer]: x")
    assert labeled_agent_send_text(entry) == "[from reviewer]: x"


def test_another_senders_label_is_not_mistaken_for_this_one() -> None:
    entry = AgentToAgentMessage(sender="b", content="[from a]: x")
    assert labeled_agent_send_text(entry) == "[from b]: [from a]: x"


def test_agent_message_without_content_is_just_the_label() -> None:
    assert labeled_agent_send_text(AgentToAgentMessage(sender="a")) == "[from a]: "


def test_compact_complete_takes_every_field_from_the_override() -> None:
    override = ContextSplice(
        ref=TapeRef(session_id="s", ordinal=4),
        mask=(),
        insert_after=None,
        payload=(UserMessage(content="summary"), _turn("ack")),
        strategy="summary",
        token_before=900,
        token_after=40,
        fallback_reason="summary failed",
        preserved_tail_count=3,
    )
    event = CompactComplete.from_override(override)
    assert event == CompactComplete(
        token_before=900,
        token_after=40,
        payload_entries=2,
        fallback_reason="summary failed",
        preserved_tail_count=3,
    )


def test_detached_result_is_keyed_by_its_results_call() -> None:
    result = ShellCommandResult(call_id="c7", stdout="ok")
    assert DetachedResult(result=result).call_id == "c7"


@pytest.mark.parametrize(
    ("entry", "role"),
    [
        (UserMessage(content="u"), "user"),
        (AgentToAgentMessage(sender="a", content="a"), "user"),
        (
            Response(
                records=(),
                tokens=TokenCount(),
                spend=TokenCost(),
                stop_reason="",
            ),
            "assistant",
        ),
        (UncategorizedToolResult(call_id="c"), None),
        (SystemMessage(content="n"), None),
        (TurnContext(cwd="/x"), None),
    ],
)
def test_wire_role(entry: ModelContextEvent, role: str | None) -> None:
    assert wire_role(entry) == role


def test_the_system_note_names_the_tool_deliveries_come_under() -> None:
    assert f"`{Markers().delivery}` foreground" in Markers().note
    assert Markers().arrived not in Markers().note


def _turn(text: str) -> Response:
    return Response(
        records=(AssistantMessage(content=text),),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
