"""Tests for ``repl.input_queues`` lane behavior."""

from __future__ import annotations

from typing import cast

from sagent.agent.agent import Agent
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.types.runtime import (
    BytesMessage,
    UserDeferredMessage,
    UserMessage,
)


def test_peek_tail_preview_prefers_urgent() -> None:
    queues = InputQueues(
        urgent=[QueuedInputBlock(text="now")], deferred=[QueuedInputBlock(text="later")]
    )
    assert queues.peek_tail_preview() == "now"
    # Peek is read-only: blocks survive.
    assert queues.urgent == [QueuedInputBlock(text="now")]
    assert queues.deferred == [QueuedInputBlock(text="later")]


def test_restore_from_snapshot_preserves_urgent_prefix() -> None:
    queues = InputQueues()
    queues.restore_from_snapshot(
        (QueuedInputBlock(text="now"), QueuedInputBlock(text="later")),
        urgent_count=1,
    )
    assert queues.urgent == [QueuedInputBlock(text="now")]
    assert queues.deferred == [QueuedInputBlock(text="later")]


def test_replace_from_navigation_preserves_urgent_edit_lane() -> None:
    """Edit-mode commit replaces only the lifted head block; the deferred
    tail of the snapshot survives the commit (F34).
    """
    queues = InputQueues()
    queues.replace_from_navigation(
        (QueuedInputBlock(text="now"), QueuedInputBlock(text="later")),
        "edited now",
        edit_mode=True,
        urgent_count=1,
    )
    assert queues.urgent == [QueuedInputBlock(text="edited now")]
    assert queues.deferred == [QueuedInputBlock(text="later")]


def test_replace_from_navigation_edit_mode_preserves_deferred_tail() -> None:
    """F34: ``edit_mode=True`` must not silently discard the deferred tail.

    Snapshot has 1 urgent + 2 deferred. The user lifts everything,
    edits, presses Enter. The committed text replaces the head urgent
    block; the deferred tail returns to the deferred queue intact.
    Without the fix the deferred entries are dropped.
    """
    queues = InputQueues()
    queues.replace_from_navigation(
        (
            QueuedInputBlock(text="u1"),
            QueuedInputBlock(text="d1"),
            QueuedInputBlock(text="d2"),
        ),
        "edited",
        edit_mode=True,
        urgent_count=1,
    )
    assert [b.text for b in queues.urgent] == ["edited"]
    assert [b.text for b in queues.deferred] == ["d1", "d2"], (
        f"deferred tail must survive edit-mode commit; got {queues.deferred!r}"
    )


def test_replace_from_navigation_edit_mode_urgent_only_keeps_urgent_tail() -> None:
    """Multiple urgent + no deferred: edit-mode keeps the urgent tail."""
    queues = InputQueues()
    queues.replace_from_navigation(
        (
            QueuedInputBlock(text="u1"),
            QueuedInputBlock(text="u2"),
        ),
        "edited",
        edit_mode=True,
        urgent_count=2,
    )
    assert [b.text for b in queues.urgent] == ["edited", "u2"]
    assert queues.deferred == []


def test_replace_from_navigation_edit_mode_deferred_only_keeps_deferred_tail() -> None:
    """Pure-deferred snapshot under edit-mode: tail preserved on deferred."""
    queues = InputQueues()
    queues.replace_from_navigation(
        (
            QueuedInputBlock(text="d1"),
            QueuedInputBlock(text="d2"),
        ),
        "edited",
        edit_mode=True,
        urgent_count=0,
    )
    assert queues.urgent == []
    assert [b.text for b in queues.deferred] == ["edited", "d2"]


def test_replace_from_navigation_urgent_lane_keeps_committed_block_urgent() -> None:
    """Enter-during-navigation (``lane="urgent"``) stages the committed
    text on the urgent lane even when ``urgent_count == 0``.

    Without an explicit lane the helper used to hardcode ``stage_deferred``,
    which meant a user who navigated up, edited, and pressed Enter saw
    their input parked behind the deferred queue's ``AgentIdle`` drain
    instead of dispatching urgent-style at the next chat-safe boundary.
    """
    queues = InputQueues()
    queues.replace_from_navigation(
        (QueuedInputBlock(text="history-1"),),
        "edited",
        edit_mode=True,
        urgent_count=0,
        lane="urgent",
    )
    assert [b.text for b in queues.urgent] == ["edited"]
    assert queues.deferred == []


def test_replace_from_navigation_non_edit_preserves_selected_block_attachments() -> (
    None
):
    """Non-edit commit (cursor past head; Enter/Tab) must carry the
    snapshot block's attachments forward.

    The user typed text, attached an image, navigated past the head
    via Up, then pressed Enter or Tab. The old code restored the
    snapshot then re-staged ``text`` *without* attachments -- silently
    dropping the image. The committed block must keep the attachments
    from the snapshot's head so the user's payload survives a navigate
    -> commit gesture.
    """
    attachment = BytesMessage(data=b"x", descriptor="image/png")
    queues = InputQueues()
    queues.replace_from_navigation(
        (QueuedInputBlock(text="with image", attachments=(attachment,)),),
        "edited text",
        edit_mode=False,
        urgent_count=1,
        lane="urgent",
    )
    assert queues.urgent, "expected one staged urgent block"
    committed = queues.urgent[-1]
    assert committed.text == "edited text"
    assert committed.attachments == (attachment,), (
        f"non-edit commit must preserve snapshot attachments;"
        f" got {committed.attachments!r}"
    )


def test_replace_from_navigation_deferred_lane_is_default() -> None:
    """Tab-during-navigation (``lane="deferred"``, the default) keeps
    the historical Tab semantics: text lands on the deferred lane.
    """
    queues = InputQueues()
    queues.replace_from_navigation(
        (QueuedInputBlock(text="snap"),),
        "tabbed",
        edit_mode=False,
        urgent_count=0,
    )
    assert queues.urgent == [QueuedInputBlock(text="snap")] or queues.urgent == []
    # Tab fallthrough lands in deferred.
    assert any(b.text == "tabbed" for b in queues.deferred)


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
    assert isinstance(pushed, UserDeferredMessage)
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
