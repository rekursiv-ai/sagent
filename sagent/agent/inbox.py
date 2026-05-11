"""Inbox: a flat FIFO buffer with bulk-drain semantics.

See ``docs/private/execution_model.md`` for the role this plays in the
round loop. Three operations matter:

- ``send(msg, source=...)`` -- append; wake any blocked drain.
- ``drain()`` -- block until at least one item is present (or until a
  user-source item lands if ``block_until_user`` is in effect), then
  return everything in arrival order and clear.
- ``requeue_front(items)`` -- restore items at the head, preserving
  their relative order, after a ``/halt`` rollback.

Every item carries a free-form ``source`` tag used by callers for
prompt formatting and queue introspection (e.g. ``peek_by_source`` for
the editable-tail UX). The inbox doesn't interpret tags except for
``"user"``, which gates ``block_until_user``.
"""

from __future__ import annotations

from collections.abc import Iterator

import asyncio
import dataclasses

from sagent.custom_types import Message


USER_SOURCE: str = "user"
TOOLS_SOURCE: str = "tools"
QUIT_SOURCE: str = "quit"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InboxItem:
    """A single inbox entry tagged with its source.

    Attributes:
      source: Free-form tag, e.g. ``"user"``, ``"Agent_0"``, ``"tools"``,
        ``"bg_<qid>"``. Used for prompt formatting and selective lookup.
      msg: The message payload.

    """

    source: str
    msg: Message


class Inbox:
    """Per-agent message buffer with bulk-drain + user-gated blocking.

    Not thread-safe; assumes the single asyncio event loop sagent runs on.
    """

    def __init__(self) -> None:
        self._items: list[InboxItem] = []
        self._event: asyncio.Event = asyncio.Event()
        self._user_block: bool = False
        # User-source item count at the moment ``block_until_user`` was
        # most recently armed. ``drain`` only unblocks when the current
        # user-source count exceeds this baseline (i.e. a NEW user item
        # arrived) or a QUIT sentinel lands. Without the baseline, a
        # halt-requeued user item would immediately satisfy the gate.
        self._user_block_baseline: int = 0

    def send(self, msg: Message, *, source: str) -> None:
        """Append an item; wake any blocked drain (subject to user-gate).

        Args:
          msg: Payload.
          source: Source tag (free-form; ``"user"`` is gate-relevant).

        """
        self._items.append(InboxItem(source=source, msg=msg))
        self._event.set()

    async def drain(self) -> list[InboxItem]:
        """Block until items are drainable; return all items in order.

        If ``block_until_user()`` has been called and no user-source
        item (and no QUIT control sentinel) is present, blocks past
        non-user arrivals (they accumulate in the buffer) until a
        user-source item lands. Then returns everything queued and
        clears the user-gate. QUIT always unblocks so ``shutdown`` can
        bring the loop down regardless of the gate.

        Returns:
          items: All buffered items, oldest first.

        """
        while True:
            await self._event.wait()
            if self._user_block and not self._has_user_or_quit():
                self._event.clear()
                continue
            break
        items = list(self._items)
        self._items.clear()
        self._event.clear()
        if self._user_block:
            self._user_block = False
        return items

    def drain_nowait(self) -> list[InboxItem]:
        """Return all currently-buffered items immediately (may be empty).

        Does not honor ``block_until_user``; caller is responsible for
        deciding when this is safe (typically after a ``drain()`` has
        already returned).
        """
        items = list(self._items)
        self._items.clear()
        self._event.clear()
        return items

    def requeue_front(self, items: list[InboxItem]) -> None:
        """Restore items at the head, preserving their relative order.

        Used by the round loop's ``/halt`` rollback path so drained
        items go back into the buffer for the next round.
        """
        if not items:
            return
        self._items = items + self._items
        self._event.set()

    def block_until_user(self) -> None:
        r"""Gate ``drain()`` until a NEW user-source item arrives.

        Snapshots the current user-source count as the gating baseline.
        ``drain`` will block until at least one additional user-source
        item has been ``send``\ ed (or a ``QUIT`` sentinel lands).
        Without the baseline, a halt-requeued user item would
        immediately satisfy the gate -- the spec requires the user to
        explicitly redirect before the agent resumes.

        Idempotent; cleared automatically when the gating drain returns.
        Non-user arrivals continue to accumulate in the buffer; they
        just don't wake the gated drain.
        """
        self._user_block = True
        self._user_block_baseline = sum(
            1 for item in self._items if item.source == USER_SOURCE
        )

    def peek_by_source(self, source: str) -> InboxItem | None:
        """Return the most recently appended item with matching source, or None."""
        for item in reversed(self._items):
            if item.source == source:
                return item
        return None

    def pop_by_source(self, source: str) -> InboxItem | None:
        """Remove and return the most recent item with matching source.

        Used by the REPL's edit-back UX (Up arrow lifts the queued
        user message into the buffer for editing). "Most recent" is
        the correct semantics there -- the user wants to edit what
        they JUST typed, not their oldest pending message.
        """
        for idx in range(len(self._items) - 1, -1, -1):
            if self._items[idx].source == source:
                popped = self._items.pop(idx)
                if not self._items:
                    self._event.clear()
                return popped
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[InboxItem]:
        return iter(self._items)

    def _has_user_or_quit(self) -> bool:
        """True when the gate should release: new user arrival or QUIT."""
        user_count = 0
        for item in self._items:
            if item.source == QUIT_SOURCE:
                return True
            if item.source == USER_SOURCE:
                user_count += 1
        return user_count > self._user_block_baseline
