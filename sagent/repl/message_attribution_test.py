"""Worker provenance must survive batching, persistence and REPL rendering."""

from __future__ import annotations

import io

from rich.console import Console

import pytest

from sagent.agent.runtime_test import (
    _runtime_for_alternation_tests,
    make_agent,
    run_with_quit,
)
from sagent.agent.session_io import _entry_from_json, _entry_to_json
from sagent.repl.console_pane import ConsolePrinter
from sagent.repl.render import RecordingPrinter, make_render_observer
from sagent.repl.replay import replay_messages
from sagent.repl.replay_test import _agent
from sagent.request_materialization import materialize_messages
from sagent.types.runtime import (
    AgentSendMessage,
    AgentSendQueuedMessage,
    AssistantMessage,
    ModelCallStarted,
    RuntimeEvent,
    UserMessage,
)
from sagent.types.tape import coalesce_roles


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_batched_workers_keep_agent_presentation(saved: bool, replay: bool) -> None:
    messages = [
        AgentSendMessage(source="review-a", text="Ready for review."),
        AgentSendMessage(source="review-b", text="Ready for review."),
    ]
    batch = coalesce_roles(messages)[0]
    assert isinstance(batch, (UserMessage, AgentSendMessage))
    if saved:
        restored = _entry_from_json(_entry_to_json(batch))
        assert isinstance(restored, (UserMessage, AgentSendMessage))
        batch = restored
    printer = RecordingPrinter()
    if replay:
        replay_messages(_agent(history=[batch]), printer)
    else:
        make_render_observer(printer)(batch)
    assert printer.user_bars == []
    assert printer.agent_bars == [
        ("review-a", "Ready for review."),
        ("review-b", "Ready for review."),
    ]


def test_human_quoting_a_sender_marker_stays_human() -> None:
    printer = RecordingPrinter()
    text = "[from review-a]: Ready for review."
    make_render_observer(printer)(UserMessage(text=text))
    assert printer.user_bars == [text]
    assert printer.agent_bars == []


def test_mixed_batch_keeps_human_and_worker_presentation() -> None:
    batch = coalesce_roles(
        [
            UserMessage(text="Please review this."),
            AgentSendMessage(source="review-a", text="Ready for review."),
            UserMessage(text="Thanks."),
        ],
    )[0]
    assert isinstance(batch, (UserMessage, AgentSendMessage))
    printer = RecordingPrinter()
    make_render_observer(printer)(batch)
    assert printer.user_bars == ["Please review this.", "Thanks."]
    assert printer.agent_bars == [("review-a", "Ready for review.")]


def test_hidden_parts_stay_hidden_when_batched_with_visible_messages() -> None:
    batch = coalesce_roles(
        [
            AgentSendMessage(source="private", text="hidden payload", hidden=True),
            AgentSendMessage(source="review-a", text="Ready for review."),
        ],
    )[0]
    assert isinstance(batch, (UserMessage, AgentSendMessage))
    printer = RecordingPrinter()
    make_render_observer(printer)(batch)
    assert printer.user_bars == []
    assert printer.agent_bars == [("review-a", "Ready for review.")]
    assert "hidden payload" in batch.text  # Still available to the model.


@pytest.mark.parametrize("width", [40, 80, 120])
def test_batched_workers_in_plain_terminal_output(width: int) -> None:
    batch = coalesce_roles(
        [
            AgentSendMessage(source="review-a", text="Ready for review."),
            AgentSendMessage(source="review-b", text="Ready for review."),
        ],
    )[0]
    assert isinstance(batch, (UserMessage, AgentSendMessage))
    output = io.StringIO()
    printer = ConsolePrinter(Console(file=output, width=width, color_system=None))
    make_render_observer(printer)(batch)
    text = output.getvalue()
    assert ">" not in text
    assert "\x1b" not in text
    assert text.count("Ready for review.") == 2
    assert "[from review-a]" in text
    assert "[from review-b]" in text


def test_provenance_does_not_change_provider_text_or_attachments() -> None:
    messages = [
        AgentSendMessage(source="review-a", text="Ready for review."),
        AgentSendMessage(source="review-b", text="Ready for review."),
    ]
    batch = coalesce_roles(messages)[0]
    assert isinstance(batch, UserMessage)
    with_parts = materialize_messages([batch])
    without_parts = materialize_messages([UserMessage(text=batch.text)])
    assert [
        (m.text, m.attachments) for m in with_parts if isinstance(m, UserMessage)
    ] == [(m.text, m.attachments) for m in without_parts if isinstance(m, UserMessage)]


def test_legacy_user_message_does_not_infer_senders_from_labels() -> None:
    text = "[from review-a]: Ready for review.\n\n[from review-b]: Ready for review."
    message = _entry_from_json({"type": "user", "text": text})
    assert isinstance(message, UserMessage)
    printer = RecordingPrinter()
    make_render_observer(printer)(message)
    assert printer.user_bars == [text]
    assert printer.agent_bars == []


def test_batches_preserve_sources_across_repeated_runtime_merges() -> None:
    runtime = _runtime_for_alternation_tests()
    runtime.append_history(AgentSendMessage(source="review-a", text="First."))
    runtime._append_or_coalesce_user(UserMessage(text="Human reply."))
    combined = runtime._append_or_coalesce_user(
        AgentSendMessage(source="review-b", text="Second."),
    )
    printer = RecordingPrinter()
    make_render_observer(printer)(combined)
    assert printer.user_bars == ["Human reply."]
    assert printer.agent_bars == [("review-a", "First."), ("review-b", "Second.")]


def test_child_batch_uses_the_same_sender_presentation() -> None:
    batch = coalesce_roles(
        [
            AgentSendMessage(source="review-a", text="First."),
            AgentSendMessage(source="review-b", text="Second."),
        ],
    )[0]
    assert isinstance(batch, UserMessage)
    output = io.StringIO()
    printer = ConsolePrinter(Console(file=output, width=100, color_system=None))
    printer.write_child_block("coordinator", [batch])
    text = output.getvalue()
    assert ">" not in text
    assert "[from review-a]: First." in text
    assert "[from review-b]: Second." in text


@pytest.mark.parametrize("human_first", [False, True])
def test_runtime_hidden_part_does_not_hide_a_visible_part(human_first: bool) -> None:
    runtime = _runtime_for_alternation_tests()
    messages = [
        UserMessage(text="Hidden.", hidden=True),
        AgentSendMessage(source="review-a", text="Visible."),
    ]
    if not human_first:
        messages.reverse()
    runtime.append_history(messages[0])
    combined = runtime._append_or_coalesce_user(messages[1])
    printer = RecordingPrinter()
    make_render_observer(printer)(combined)
    assert printer.user_bars == []
    assert printer.agent_bars == [("review-a", "Visible.")]


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_runtime_deliveries_keep_senders_and_model_batch(queued: bool) -> None:
    runtime, collector = make_agent(
        [AssistantMessage(text="Working."), AssistantMessage(text="Received.")],
        model_delay_sec=0.01,
    )
    printer = RecordingPrinter()
    runtime.observers.append(make_render_observer(printer))
    sent = False

    def deliver(event: RuntimeEvent) -> None:
        nonlocal sent
        if isinstance(event, ModelCallStarted) and not sent:
            sent = True
            for source in ("review-a", "review-b"):
                message = (
                    AgentSendQueuedMessage(source=source, text="Ready for review.")
                    if queued
                    else AgentSendMessage(source=source, text="Ready for review.")
                )
                runtime.inbox.push_back(message)

    runtime.observers.append(deliver)
    runtime.inbox.push_back(UserMessage(text="Start."))
    await run_with_quit(runtime)
    assert printer.user_bars == ["Start."]
    assert printer.agent_bars == [
        ("review-a", "Ready for review."),
        ("review-b", "Ready for review."),
    ]
    batch = runtime.context().messages[2]
    assert isinstance(batch, UserMessage)
    assert batch.text == (
        "[from review-a]: Ready for review.\n\n[from review-b]: Ready for review."
    )
    assert sum(isinstance(event, ModelCallStarted) for event in collector.events) == 2
