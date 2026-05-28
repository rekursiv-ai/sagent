"""Tests for ``repl.input_queues`` lane behavior."""

from __future__ import annotations

from typing import cast

from sagent.agent.agent import Agent
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.types.runtime import (
    BytesMessage,
    UserMessage,
    UserQueuedMessage,
)


def test_pop_tail_preview_prefers_urgent() -> None:
    queues = InputQueues(
        urgent=[QueuedInputBlock(text="now")], deferred=[QueuedInputBlock(text="later")]
    )
    assert queues.pop_tail_preview() == "now"


def test_restore_from_snapshot_preserves_urgent_prefix() -> None:
    queues = InputQueues()
    queues.restore_from_snapshot(
        (QueuedInputBlock(text="now"), QueuedInputBlock(text="later")),
        urgent_count=1,
    )
    assert queues.urgent == [QueuedInputBlock(text="now")]
    assert queues.deferred == [QueuedInputBlock(text="later")]


def test_replace_from_navigation_preserves_urgent_edit_lane() -> None:
    queues = InputQueues()
    queues.replace_from_navigation(
        (QueuedInputBlock(text="now"), QueuedInputBlock(text="later")),
        "edited now",
        edit_mode=True,
        urgent_count=1,
    )
    assert queues.urgent == [QueuedInputBlock(text="edited now")]
    assert queues.deferred == []


def test_commit_urgent_preserves_attachments() -> None:
    attachment = BytesMessage(data=b"img", descriptor="image/png")
    queues = InputQueues(
        urgent=[QueuedInputBlock(text="see", attachments=(attachment,))],
    )
    agent = _FakeAgent()
    assert queues.commit_urgent(cast(Agent, agent))
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "see"
    assert pushed.attachments == (attachment,)


def test_commit_deferred_preserves_attachments() -> None:
    attachment = BytesMessage(data=b"pdf", descriptor="application/pdf")
    queues = InputQueues(
        deferred=[QueuedInputBlock(text="read", attachments=(attachment,))],
    )
    agent = _FakeAgent()
    assert queues.commit_deferred_on_idle(cast(Agent, agent))
    pushed = agent.runtime.inbox.items[0]
    assert isinstance(pushed, UserQueuedMessage)
    assert pushed.text == "read"
    assert pushed.attachments == (attachment,)


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
