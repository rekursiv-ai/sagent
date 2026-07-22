"""Tests for ``repl.input_queues``: single-message panes + coalesce."""

from __future__ import annotations

from typing import cast

from sagent.agent.agent import Agent
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.types.runtime import (
    BytesMessage,
    UserDeferredMessage,
    UserMessage,
)


def test_stage_queue_into_empty_pane_sets_single_message() -> None:
    queues = InputQueues()
    queues.stage_queue("alpha")
    assert queues.queue == QueuedInputBlock(text="alpha")
    assert queues.deferred is None


def test_stage_queue_coalesces_append_after_existing() -> None:
    r"""Spec: queue becomes ``existing + "\n\n" + input`` (FIFO)."""
    queues = InputQueues()
    queues.stage_queue("alpha")
    queues.stage_queue("beta")
    assert queues.queue is not None
    assert queues.queue.text == "alpha\n\nbeta"


def test_stage_deferred_coalesces_append_after_existing() -> None:
    queues = InputQueues()
    queues.stage_deferred("one")
    queues.stage_deferred("two")
    assert queues.deferred is not None
    assert queues.deferred.text == "one\n\ntwo"


def test_panes_are_independent() -> None:
    queues = InputQueues()
    queues.stage_queue("q")
    queues.stage_deferred("d")
    assert queues.queue is not None
    assert queues.deferred is not None
    assert queues.queue.text == "q"
    assert queues.deferred.text == "d"


def test_coalesce_concatenates_attachments() -> None:
    a1 = BytesMessage(data=b"1", descriptor="image/png")
    a2 = BytesMessage(data=b"2", descriptor="image/png")
    queues = InputQueues()
    queues.stage_queue("first", (a1,))
    queues.stage_queue("second", (a2,))
    assert queues.queue is not None
    assert queues.queue.attachments == (a1, a2)


def test_render_lines_deferred_above_queue_with_prefix() -> None:
    """Spec: deferred pane above queue; ``[deferred]`` prefix; queue bare."""
    queues = InputQueues(
        queue=QueuedInputBlock(text="q-msg"),
        deferred=QueuedInputBlock(text="d-msg"),
    )
    assert queues.render_lines() == ["[deferred] d-msg", "q-msg"]


def test_render_lines_queue_only_has_no_prefix() -> None:
    queues = InputQueues(queue=QueuedInputBlock(text="just queued"))
    assert queues.render_lines() == ["just queued"]


def test_render_lines_empty_is_empty() -> None:
    assert InputQueues().render_lines() == []


def test_has_any() -> None:
    assert not InputQueues().has_any()
    assert InputQueues(queue=QueuedInputBlock(text="x")).has_any()
    assert InputQueues(deferred=QueuedInputBlock(text="x")).has_any()


def test_clear_empties_both_panes() -> None:
    queues = InputQueues(
        queue=QueuedInputBlock(text="q"), deferred=QueuedInputBlock(text="d")
    )
    queues.clear()
    assert not queues.has_any()


def test_peek_tail_preview_prefers_queue() -> None:
    queues = InputQueues(
        queue=QueuedInputBlock(text="now"),
        deferred=QueuedInputBlock(text="later"),
    )
    assert queues.peek_tail_preview() == "now"
    # Read-only.
    assert queues.queue is not None
    assert queues.deferred is not None


def test_peek_tail_preview_falls_back_to_deferred() -> None:
    queues = InputQueues(deferred=QueuedInputBlock(text="later"))
    assert queues.peek_tail_preview() == "later"


def test_peek_tail_preview_empty() -> None:
    assert InputQueues().peek_tail_preview() == ""


def test_pop_queue_message_clears_pane_and_preserves_attachments() -> None:
    attachment = BytesMessage(data=b"img", descriptor="image/png")
    queues = InputQueues(queue=QueuedInputBlock(text="see", attachments=(attachment,)))
    message = queues.pop_queue_message()
    assert isinstance(message, UserMessage)
    assert message.text == "see"
    assert message.attachments == (attachment,)
    assert queues.queue is None


def test_pop_queue_message_none_when_empty() -> None:
    assert InputQueues().pop_queue_message() is None


def test_commit_queue_pushes_user_message() -> None:
    attachment = BytesMessage(data=b"img", descriptor="image/png")
    queues = InputQueues(queue=QueuedInputBlock(text="see", attachments=(attachment,)))
    agent = _FakeAgent()
    assert queues.commit_queue(cast(Agent, agent))
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "see"
    assert pushed.attachments == (attachment,)
    assert queues.queue is None


def test_commit_queue_false_when_empty() -> None:
    agent = _FakeAgent()
    assert not InputQueues().commit_queue(cast(Agent, agent))
    assert agent.runtime.inbox.items == []


def test_commit_deferred_on_idle_pushes_deferred_message() -> None:
    attachment = BytesMessage(data=b"pdf", descriptor="application/pdf")
    queues = InputQueues(
        deferred=QueuedInputBlock(text="read", attachments=(attachment,))
    )
    agent = _FakeAgent()
    assert queues.commit_deferred_on_idle(cast(Agent, agent))
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserDeferredMessage)
    assert pushed.text == "read"
    assert pushed.attachments == (attachment,)
    assert queues.deferred is None


def test_commit_deferred_false_when_empty() -> None:
    agent = _FakeAgent()
    assert not InputQueues().commit_deferred_on_idle(cast(Agent, agent))


class _FakeInbox:
    def __init__(self) -> None:
        self.items: list[object] = []

    def push_back(self, item: object) -> None:
        self.items.append(item)


class _FakeRuntime:
    def __init__(self) -> None:
        self.inbox = _FakeInbox()


class _FakeAgent:
    def __init__(self) -> None:
        self.runtime = _FakeRuntime()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
