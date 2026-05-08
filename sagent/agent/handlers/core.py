"""Core single-purpose inline handlers + the standard handler-set factory.

Each handler in this module owns one piece of mutable agent state and
fires on one descriptor. They share a common shape ("inline, mutate,
maybe post a follow-up message") so collecting them in one file keeps
the per-class noise low. The substantial multi-class handler bundles
(model calls, tool dispatch, compaction) live in their own modules.

:func:`core_handlers` returns the standard handler set wired in
dispatch order. Order matters: ``HistoryHandler`` runs first so
follow-up handlers see the message already in history;
``ActivityHandler`` precedes ``ModelCallHandler`` so ``active`` flips
True before the spawned model task is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from sagent.agent.handlers.activity import ActivityHandler
from sagent.agent.handlers.base import Handler, InlineHandler
from sagent.agent.handlers.compact import (
    BudgetWatcher,
    CompactHandler,
    UncompactHandler,
)
from sagent.agent.handlers.model import (
    ModelCallHandler,
    ModelResponseHandler,
)
from sagent.agent.handlers.tools import (
    ToolBatchHandler,
    ToolBatchResultHandler,
)
from sagent.custom_types import TextMessage, TokenCount


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Compactor, Message


# -- HistoryHandler ----------------------------------------------------


class HistoryHandler(InlineHandler):
    """Sole writer of ``Agent.history``.

    Subscribes to every history-bearing descriptor and appends the
    message verbatim. Other handlers subscribe to the same descriptors
    for routing follow-ups; ``HistoryHandler`` must register first so
    the message is in history before downstream handlers act on it.
    """

    descriptors: tuple[str, ...] = (
        "text/x-user-message",
        "multipart/x-model-message",
        "multipart/x-tool-result",
    )

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        agent.history.append(msg)


# -- UserMessageHandler ------------------------------------------------


class UserMessageHandler(InlineHandler):
    """Trigger a model call when a user message arrives.

    Coalesces back-to-back user messages: if a ``text/x-model-call``
    is already queued or the ``model_lock`` is held, the in-flight
    call already sees the appended history, so posting another would
    just produce a redundant request. Mirrors v1's drain-and-batch
    on a per-message basis.
    """

    descriptors: tuple[str, ...] = ("text/x-user-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        if _has_pending_model_call(agent):
            return
        agent.inbox.put(TextMessage("", "text/x-model-call", parent_id=msg.id))


def _has_pending_model_call(agent: Agent) -> bool:
    """Return True if a model call is already queued or being processed."""
    if agent.model_lock.locked():
        return True
    return any(queued.descriptor == "text/x-model-call" for queued in agent.inbox)


# -- ClearHandler ------------------------------------------------------


class ClearHandler(InlineHandler):
    """Wipe history + file-tracking caches on ``text/x-clear-request``.

    Resets ``cost_tracker.last_request`` and ``ToolState`` file-tracking
    caches so the Edit/Write "must read first" invariant starts fresh.
    Cumulative cost totals survive (cost is a session metric, not a
    context metric). Saves the cleared session so a crash before the
    next model request leaves disk state matching memory.
    """

    descriptors: tuple[str, ...] = ("text/x-clear-request",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        agent.history.clear()
        agent.cost_tracker.last_request = TokenCount()
        agent.activity.num_tool_call_rounds = 0
        agent.tool_state.reset_file_tracking()
        agent._save_session()  # noqa: SLF001 -- handler is agent's intimate helper


# -- AbortHandler ------------------------------------------------------


class AbortHandler(InlineHandler):
    """Cancel in-flight spawned tasks; signal sync tools via ``abort_event``.

    Cancels every task in the agent's in-flight registry; sync tool
    threads polling ``tool_state.abort_event`` see the set flag and
    unwind. Surviving queued messages are processed normally -- abort
    is for in-flight work, not the entire session.
    """

    descriptors: tuple[str, ...] = ("text/x-abort",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        agent.tool_state.abort_event.set()
        for task in list(agent.tasks.values()):
            if not task.done():
                _ = task.cancel()


# -- StatsHandler ------------------------------------------------------


class StatsHandler(InlineHandler):
    """Refresh ``tool_state.stats`` after each model response.

    The Diagnostics tool reads ``tool_state.stats`` synchronously when
    invoked. Refreshing the snapshot after every model response keeps
    the published view aligned with the most recent counters.
    """

    descriptors: tuple[str, ...] = ("multipart/x-model-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        c = agent.cost_tracker
        agent.tool_state.stats = {
            "num_tool_call_rounds": agent.activity.num_tool_call_rounds,
            "total_input_tokens": c.total.input_tokens,
            "total_output_tokens": c.total.output_tokens,
            "input_tokens": c.last_request.input_tokens,
            "cache_creation_tokens": c.total.cache_creation_tokens,
            "cache_read_tokens": c.total.cache_read_tokens,
            "total_cost_usd": c.total_cost_usd,
            "max_request_tokens": agent.max_request_tokens,
            "max_response_tokens": agent.max_response_tokens,
        }


# -- SessionSaveHandler -----------------------------------------------


class SessionSaveHandler(InlineHandler):
    """Persist ``session.jsonl`` after every model response.

    Delegates to ``agent._save_session()``; that helper builds the full
    meta (model spec, status, tokens, cost, compaction, etc.) so resumed
    sessions can restore the model and continue.
    """

    descriptors: tuple[str, ...] = ("multipart/x-model-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        agent._save_session()  # noqa: SLF001 -- handler is agent's intimate helper


# -- Standard handler-set factory -------------------------------------


def core_handlers(
    *,
    compactor: Compactor | None = None,
) -> list[Handler]:
    """Build the standard core handler set.

    Order matters:
      - ``HistoryHandler`` first: every history-bearing descriptor also
        routes to a follow-up handler that expects the message to
        already be in history.
      - ``ActivityHandler`` before ``ModelCallHandler``: ``active``
        must flip True before the spawned model task is created.
      - ``BudgetWatcher`` before ``ModelCallHandler``: pre-flight
        compaction has to land before the spawned model call task.

    Args:
      compactor: Optional compactor; enables ``BudgetWatcher`` and
          ``CompactHandler`` to do real work. With no compactor, those
          handlers are no-ops.

    Returns:
      handlers: List of handler instances ready to register on Agent.

    """
    del compactor  # only needed to advertise compaction is an option
    return [
        HistoryHandler(),
        ActivityHandler(),
        UserMessageHandler(),
        BudgetWatcher(),
        ModelCallHandler(),
        ModelResponseHandler(),
        ToolBatchHandler(),
        ToolBatchResultHandler(),
        CompactHandler(),
        UncompactHandler(),
        ClearHandler(),
        AbortHandler(),
        StatsHandler(),
        SessionSaveHandler(),
    ]
