"""Regression coverage for interleaved child replies and response boundaries."""

from __future__ import annotations

import pytest

from sagent.repl.render import (
    _STREAM_BUF_FLUSH_CHARS,
    RecordingPrinter,
    make_render_observer,
    strict_observer,
)
from sagent.types.runtime import (
    AssistantMessage,
    ChildDoneEvent,
    ChildEvent,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    NoticeMessage,
    ToolLabel,
    ToolResult,
)


def _replies(printer: RecordingPrinter) -> list[tuple[str, str]]:
    return [
        (label, item.text)
        for label, items in printer.child_blocks
        for item in items
        if isinstance(item, AssistantMessage)
    ]


@pytest.mark.parametrize("seed", range(5))
def test_interleaved_words_remain_whole_replies(seed: int) -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    texts = {
        "reader": "This project supports research.",
        "reviewer": "Check café 科学 ✅.",
    }
    offsets = dict.fromkeys(texts, 0)
    with strict_observer():
        while pending := [
            label for label, text in texts.items() if offsets[label] < len(text)
        ]:
            label = pending[(sum(offsets.values()) + seed) // (seed + 1) % len(pending)]
            offset = offsets[label]
            observer(
                ChildEvent(
                    label=label,
                    inner=ModelResponsePartial(text=texts[label][offset]),
                ),
            )
            offsets[label] += 1
        # Speaker switches alone must not emit isolated characters.
        assert _replies(printer) == []
        for label, text in texts.items():
            observer(
                ChildEvent(
                    label=label,
                    inner=ModelResponseComplete(message=AssistantMessage(text=text)),
                ),
            )
    assert _replies(printer) == list(texts.items())


@pytest.mark.parametrize("second", ["Second reply.", "First reply."])
def test_persistent_worker_replies_finish_separately(second: str) -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    for text in ["First reply.", second]:
        observer(ChildEvent(label="reader", inner=ModelResponsePartial(text=text)))
        observer(
            ChildEvent(
                label="reader",
                inner=ModelResponseComplete(message=AssistantMessage(text=text)),
            ),
        )
    assert _replies(printer) == [("reader", "First reply."), ("reader", second)]
    assert observer._child_text == {}


def test_fast_reply_is_visible_while_other_worker_is_paused() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(ChildEvent(label="slow", inner=ModelResponsePartial(text="Checking ")))
    observer(ChildEvent(label="fast", inner=ModelResponsePartial(text="Ready.\n\n")))
    observer(
        ChildEvent(
            label="fast",
            inner=ModelResponseComplete(message=AssistantMessage(text="Ready.\n\n")),
        ),
    )
    assert _replies(printer) == [("fast", "Ready.")]
    assert observer._child_text["slow"] == "Checking "
    observer(ChildEvent(label="slow", inner=ModelResponsePartial(text="the README.")))
    observer(
        ChildEvent(
            label="slow",
            inner=ModelResponseComplete(
                message=AssistantMessage(text="Checking the README."),
            ),
        ),
    )
    assert _replies(printer) == [("fast", "Ready."), ("slow", "Checking the README.")]


def test_other_workers_tool_error_does_not_split_pending_reply() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(ChildEvent(label="reader", inner=ModelResponsePartial(text="Reviewing ")))
    error = ToolResult(call_id="read-1", content="missing file", is_error=True)
    observer(ChildEvent(label="reviewer", inner=error))
    assert printer.child_blocks == [("reviewer", [error])]
    observer(
        ChildEvent(label="reader", inner=ModelResponsePartial(text="the local copy.")),
    )
    observer(
        ChildEvent(
            label="reader",
            inner=ModelResponseComplete(
                message=AssistantMessage(text="Reviewing the local copy."),
            ),
        ),
    )
    assert _replies(printer) == [("reader", "Reviewing the local copy.")]


def test_same_worker_tool_label_preserves_order() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(
        ChildEvent(
            label="reader",
            inner=ModelResponsePartial(text="I will read the file."),
        ),
    )
    label = ToolLabel(call_id="read-1", text="Read")
    observer(ChildEvent(label="reader", inner=label))
    assert _replies(printer) == [("reader", "I will read the file.")]
    assert printer.child_blocks[0][0] == "reader"
    assert printer.child_blocks[0][1][-1] is label


@pytest.mark.parametrize("ending", ["error", "cancel", "exit"])
def test_interruption_preserves_received_text_and_releases_buffer(ending: str) -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(
        ChildEvent(label="reader", inner=ModelResponsePartial(text="Unfinished reply")),
    )
    match ending:
        case "error":
            observer(
                ChildEvent(
                    label="reader",
                    inner=ModelResponseError(RuntimeError("connection lost")),
                ),
            )
        case "cancel":
            observer(ChildEvent(label="reader", inner=ModelResponseCancelled()))
            assert any(
                isinstance(item, NoticeMessage) and item.text == "[interrupted]"
                for _, items in printer.child_blocks
                for item in items
            )
        case _:
            observer(ChildDoneEvent(label="reader", elapsed=1.0, tokens=0, cost=0.0))
    assert _replies(printer) == [("reader", "Unfinished reply")]
    assert observer._child_text == {}


def test_nested_completion_flushes_reply() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    for inner in [
        ModelResponsePartial(text="Nested reply."),
        ModelResponseComplete(message=AssistantMessage(text="Nested reply.")),
    ]:
        observer(
            ChildEvent(
                label="group/reviewer",
                inner=ChildEvent(label="reviewer", inner=inner),
            ),
        )
    assert _replies(printer) == [("group/reviewer", "Nested reply.")]


def test_code_fence_and_unicode_survive_other_worker_activity() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(
        ChildEvent(
            label="coder",
            inner=ModelResponsePartial(text="```python\n    print('ca"),
        ),
    )
    observer(
        ChildEvent(label="reviewer", inner=ModelResponsePartial(text="科学 ✅\n\n")),
    )
    observer(
        ChildEvent(
            label="reviewer",
            inner=ModelResponseComplete(message=AssistantMessage(text="科学 ✅\n\n")),
        ),
    )
    observer(
        ChildEvent(label="coder", inner=ModelResponsePartial(text="fé')\n```\n\n")),
    )
    observer(
        ChildEvent(
            label="coder",
            inner=ModelResponseComplete(
                message=AssistantMessage(text="```python\n    print('café')\n```\n\n"),
            ),
        ),
    )
    assert _replies(printer) == [
        ("reviewer", "科学 ✅"),
        ("coder", "```python\n    print('café')\n```"),
    ]


@pytest.mark.parametrize("chunk_size", [97, 70009])
def test_large_unclosed_fence_stays_bounded_and_keeps_exact_content(
    chunk_size: int,
) -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    text = "```text\n" + "x" * 70000
    for offset in range(0, len(text), chunk_size):
        observer(
            ChildEvent(
                label="coder",
                inner=ModelResponsePartial(text=text[offset : offset + chunk_size]),
            ),
        )
        assert len(observer._child_text.get("coder", "")) <= _STREAM_BUF_FLUSH_CHARS
    observer(
        ChildEvent(label="reviewer", inner=ModelResponsePartial(text="Ready.\n\n")),
    )
    observer(
        ChildEvent(
            label="reviewer",
            inner=ModelResponseComplete(message=AssistantMessage(text="Ready.\n\n")),
        ),
    )
    observer(
        ChildEvent(
            label="coder",
            inner=ModelResponseComplete(message=AssistantMessage(text=text)),
        ),
    )
    assert (
        "".join(reply for label, reply in _replies(printer) if label == "coder") == text
    )
    assert ("reviewer", "Ready.") in _replies(printer)


def test_assembled_completion_does_not_repeat_rendered_paragraph() -> None:
    printer = RecordingPrinter()
    observer = make_render_observer(printer)
    observer(ChildEvent(label="reader", inner=ModelResponsePartial(text="Ready.\n\n")))
    observer(
        ChildEvent(
            label="reader",
            inner=ModelResponseComplete(message=AssistantMessage(text="Ready.\n\n")),
        ),
    )
    assert _replies(printer) == [("reader", "Ready.")]
