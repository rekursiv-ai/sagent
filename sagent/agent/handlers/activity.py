"""``ActivityTracker`` + ``ActivityHandler``: drive the bottom toolbar.

The tracker is a small dataclass on the Agent that handlers update at
specific lifecycle moments:

- ``current_call_start``: monotonic time when the latest
  ``text/x-model-call`` was received; ``0.0`` between calls.
- ``elapsed_seconds``: cumulative time spent on model calls.
- ``live_response_chars``: char count from streaming chunks during
  the in-flight call; reset to 0 at call start.
- ``active``: True iff a call is in flight.

Token estimate: ``live_response_tokens`` = ``live_response_chars // 4``
(matches v1's heuristic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import asyncio
import dataclasses

from sagent.agent.handlers.base import InlineHandler


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


@dataclasses.dataclass(kw_only=True, slots=True)
class ActivityTracker:
    """Lifecycle counters for the bottom toolbar / status bar.

    Attributes:
      elapsed_seconds: Cumulative time spent on model calls.
      current_call_start: Monotonic time of the active call, or 0.0.
      live_response_chars: Streaming chars accumulated this call.
      active: True iff a model call is currently in flight.

    """

    elapsed_seconds: float = 0.0
    current_call_start: float = 0.0
    live_response_chars: int = 0
    active: bool = False
    num_tool_call_rounds: int = 0
    truncation_recoveries: int = 0
    last_stop_reason: str | None = None

    @property
    def live_response_tokens(self) -> int:
        """Estimated output tokens streamed so far (chars/4 heuristic)."""
        return self.live_response_chars // 4


class ActivityHandler(InlineHandler):
    """Maintain ``Agent.activity`` from inbox lifecycle events.

    Subscribes to:

    - ``text/x-model-call``: mark a call as active; record start time.
    - ``text/plain``: accumulate streamed chars.
    - ``text/x-stream-end``: end of streamed text for this call.
    - ``multipart/x-model-message``: model call complete -- accumulate
      elapsed, reset live counters.

    The ordering inside ``core_handlers`` puts ``ActivityHandler``
    before the spawned ``ModelCallHandler`` so ``active`` flips True
    before the spawn fires.
    """

    descriptors: tuple[str, ...] = (
        "text/x-model-call",
        "text/plain",
        "text/x-stream-end",
        "multipart/x-model-message",
    )

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Update the activity tracker based on the descriptor."""
        tracker = agent.activity
        if msg.descriptor == "text/x-model-call":
            tracker.active = True
            tracker.current_call_start = asyncio.get_running_loop().time()
            tracker.live_response_chars = 0
            return
        if msg.descriptor == "text/plain":
            tracker.live_response_chars += len(str(msg.content))
            return
        if msg.descriptor == "text/x-stream-end":
            return
        if msg.descriptor == "multipart/x-model-message":
            if tracker.active:
                tracker.elapsed_seconds += (
                    asyncio.get_running_loop().time() - tracker.current_call_start
                )
            tracker.active = False
            tracker.current_call_start = 0.0
            tracker.live_response_chars = 0
            return
