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
    RecompactHandler,
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
from sagent.lib.descriptors import QUIT_SENTINEL
from sagent.tools.core import agent_registry


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


# -- HistoryHandler ----------------------------------------------------


class HistoryHandler(InlineHandler):
    """Sole writer of ``Agent.history``.

    Subscribes to every history-bearing descriptor and appends the
    message verbatim. Other handlers subscribe to the same descriptors
    for routing follow-ups; ``HistoryHandler`` must register first so
    the message is in history before downstream handlers act on it.

    Mid-call deferral: when a ``text/x-user-message`` arrives while
    ``model_lock`` is held, the in-flight call has already snapshotted
    history into a ``ModelRequest``. Appending the user message to
    ``history`` now would mean the next queued model call (which
    builds *after* the assistant response is appended) sees an
    assistant terminal, which Anthropic's claude-opus-4-7+1m rejects
    ("assistant message prefill not supported"). Instead, park the
    user message on ``agent.history_pending``. The drain runs on
    ``text/x-call-ended`` -- emitted by ``ModelCallHandler``'s
    ``finally`` after lock release on every path (success, cancel,
    error) -- so messages parked during a *cancelled* call are also
    drained, not stranded.
    """

    descriptors: tuple[str, ...] = (
        "text/x-user-message",
        "multipart/x-model-message",
        "multipart/x-tool-result",
        "text/x-call-ended",
    )

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        if msg.descriptor == "text/x-call-ended":
            # The in-flight call has ended (success, cancel, or error);
            # drain any user messages parked while ``model_lock`` was
            # held. Emit a single ``text/x-model-call`` covering all
            # drained messages -- emitting one per parked message
            # would produce byte-identical redundant requests because
            # each spawned ``ModelCallHandler`` snapshots history
            # *after* the drain. ``parent_id`` follows the convention
            # used by ``UserMessageHandler``: the id of the user
            # message that triggered the call (here, the most recent
            # parked message).
            if agent.history_pending:
                drained = list(agent.history_pending)
                agent.history_pending.clear()
                agent.history.extend(drained)
                agent.inbox.put(
                    TextMessage("", "text/x-model-call", parent_id=drained[-1].id),
                )
            return
        if msg.descriptor == "text/x-user-message" and agent.model_lock.locked():
            agent.history_pending.append(msg)
            return
        agent.history.append(msg)


# -- UserMessageHandler ------------------------------------------------


class UserMessageHandler(InlineHandler):
    """Trigger a model call when a user message arrives.

    Coalesces back-to-back user messages. If a ``text/x-model-call``
    is already queued, posting another would produce a redundant
    request. If ``model_lock`` is held, ``HistoryHandler`` has parked
    this message on ``agent.history_pending`` and will post a fresh
    ``text/x-model-call`` when the in-flight response drains pending
    -- posting one here would race that drain.
    """

    descriptors: tuple[str, ...] = ("text/x-user-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        if _has_pending_model_call(agent):
            return
        agent.inbox.put(TextMessage("", "text/x-model-call", parent_id=msg.id))


def _has_pending_model_call(agent: Agent) -> bool:
    """Return True iff a model call is already queued or in flight.

    Returns True when either a ``text/x-model-call`` is queued in the
    inbox or ``model_lock`` is held by an in-flight call. A user
    message arriving mid-call is *not* stranded by this check:
    ``HistoryHandler`` parks it on ``agent.history_pending`` and
    drains the queue itself when the in-flight call ends (the
    ``text/x-call-ended`` event from ``ModelCallHandler``'s
    ``finally``). The drain posts a single ``text/x-model-call``
    that combines all deferred messages into one turn. Without the
    lock check, ``UserMessageHandler`` would enqueue a second model
    call whose handler races to build its request *after* the
    in-flight response is appended to history, producing an
    assistant-terminal request that Anthropic's claude-opus-4-7+1m
    rejects.
    """
    if agent.model_lock.locked():
        return True
    return any(queued.descriptor == "text/x-model-call" for queued in agent.inbox)


# -- ClearHandler ------------------------------------------------------


class ClearHandler(InlineHandler):
    """Wipe history + file-tracking caches on ``text/x-clear-request``.

    Resets ``cost_tracker.last_request`` and ``ToolState`` file-tracking
    caches so the Edit/Write "must read first" invariant starts fresh.
    Cumulative cost totals survive (cost is a session metric, not a
    context metric).

    Persists a ``kind: clear`` barrier marker to ``session.jsonl`` --
    append-only, never truncating. Bytes preceding the barrier remain
    on disk for forensics; the loader's live view starts after the
    most recent barrier. Recovery from an accidental ``/clear`` is to
    open the file, delete the offending barrier line, and restart.
    """

    descriptors: tuple[str, ...] = ("text/x-clear-request",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        agent.history.clear()
        agent.history_pending.clear()
        agent.cost_tracker.last_request = TokenCount()
        agent.activity.num_tool_call_rounds = 0
        agent.tool_state.reset_file_tracking()
        agent.save_session(clear=True)


# -- AbortHandler ------------------------------------------------------


def _cancel_local_tasks(agent: Agent) -> None:
    """Cancel every spawned task currently in ``agent.tasks``.

    Hidden background tasks (REPL pump, daemons) live in
    ``agent.background_tasks`` and are untouched here -- bash-style
    foreground/background separation. Cancellation propagates through
    ``await``s natively; subprocess-backed tools (Bash) kill their
    process groups in their own ``CancelledError`` handlers.
    """
    for task in list(agent.tasks.values()):
        if not task.done():
            _ = task.cancel()


def _drop_pending_user_input(agent: Agent) -> None:
    """Drop typed-ahead user input, keep ``text/x-quit`` sentinels.

    Both queues represent user input the agent has not yet acted on:
    the inbox holds messages awaiting dispatch; ``history_pending``
    holds messages parked by ``HistoryHandler`` while ``model_lock``
    is held. ``/abort`` should erase both, matching the verb's
    "throw away typed-ahead" semantics. The ``text/x-quit`` sentinel
    survives so the dispatch loop still terminates cleanly.
    """
    kept = [m for m in agent.inbox.drain() if m.descriptor == QUIT_SENTINEL]
    for m in kept:
        _ = agent.inbox.put(m)
    agent.history_pending.clear()


def _cancel_visible_background(agent: Agent) -> None:
    """Cancel every visible (non-hidden) background task on this agent."""
    for qid, job in list(agent.background_tasks.items()):
        if job.hidden:
            continue
        _ = job.task.cancel()
        agent.background_tasks.pop(qid, None)


class BreakHandler(InlineHandler):
    """Cancel the in-flight step (``/break``; Ctrl+Z analog).

    The inbox queue and background tasks survive: ``/break`` is the
    "interrupt this step, keep planned follow-ups" verb. Messages
    parked in ``history_pending`` (typed *during* the in-flight call,
    not after it) do not survive -- the cancel path posts
    ``text/x-stream-end`` + ``text/x-interrupted``, never
    ``multipart/x-model-message``, so ``HistoryHandler``'s drain
    trigger never fires; preserving them would silently reorder them
    behind whatever the user types next, across the interrupt
    boundary. ``content`` selects the scope:

    - ``""``: this agent only.
    - ``"<label>"``: forward to that agent (cross-agent reach via inbox).
    - ``"all"``: act locally and forward ``""`` to every other
      registered agent so each cancels its own step. The fan-out is
      one-hop -- forwarded messages arrive with ``content=""`` and
      do not re-broadcast.
    """

    descriptors: tuple[str, ...] = ("text/x-break",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        target = str(msg.content).strip()
        if target in ("", agent.name):
            _cancel_local_tasks(agent)
            agent.history_pending.clear()
            return
        if target == "all":
            _cancel_local_tasks(agent)
            agent.history_pending.clear()
            for other in list(agent_registry.values()):
                if other is agent:
                    continue
                _ = other.inbox.put_left(TextMessage("", "text/x-break"))
            return
        other = agent_registry.get(target)
        if other is None:
            agent.inbox.put(
                TextMessage(f"[/break] unknown agent: {target}", "text/x-error"),
            )
            return
        _ = other.inbox.put_left(TextMessage("", "text/x-break"))


class AbortHandler(InlineHandler):
    """Cancel step + drain queue (``/abort``; Ctrl+C analog).

    More aggressive than ``/break``: also drops typed-ahead commands
    and model-call triggers from the inbox. The ``text/x-quit``
    sentinel is preserved so a queued quit still terminates the
    loop.

    ``content`` selects the scope:

    - ``""``: this agent only; foreground + queue.
    - ``"bg"``: foreground + queue + visible background tasks
      (this is what the ``"all"`` fan-out forwards to recipients).
    - ``"<label>"``: forward to that agent.
    - ``"all"``: act locally with bg-kill, then forward ``"bg"`` to
      every other registered agent. Forwarded messages do not
      re-broadcast.
    """

    descriptors: tuple[str, ...] = ("text/x-abort",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        target = str(msg.content).strip()
        if target in ("", agent.name):
            _cancel_local_tasks(agent)
            _drop_pending_user_input(agent)
            return
        if target == "bg":
            _cancel_local_tasks(agent)
            _drop_pending_user_input(agent)
            _cancel_visible_background(agent)
            return
        if target == "all":
            _cancel_local_tasks(agent)
            _drop_pending_user_input(agent)
            _cancel_visible_background(agent)
            for other in list(agent_registry.values()):
                if other is agent:
                    continue
                _ = other.inbox.put_left(TextMessage("bg", "text/x-abort"))
            return
        other = agent_registry.get(target)
        if other is None:
            agent.inbox.put(
                TextMessage(f"[/abort] unknown agent: {target}", "text/x-error"),
            )
            return
        _ = other.inbox.put_left(TextMessage("", "text/x-abort"))


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

    Delegates to ``agent.save_session()``; that helper builds the full
    meta (model spec, status, tokens, cost, compaction, etc.) so resumed
    sessions can restore the model and continue.
    """

    descriptors: tuple[str, ...] = ("multipart/x-model-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        agent.save_session()


# -- Standard handler-set factory -------------------------------------


def core_handlers() -> list[Handler]:
    """Build the standard core handler set.

    Order matters:
      - ``HistoryHandler`` first: every history-bearing descriptor also
        routes to a follow-up handler that expects the message to
        already be in history.
      - ``ActivityHandler`` before ``ModelCallHandler``: ``active``
        must flip True before the spawned model task is created.
      - ``BudgetWatcher`` before ``ModelCallHandler``: pre-flight
        compaction has to land before the spawned model call task.

    Compaction handlers read ``agent.compactor`` directly and no-op
    when it's ``None``; the agent owns that wiring.

    Returns:
      handlers: List of handler instances ready to register on Agent.

    """
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
        RecompactHandler(),
        ClearHandler(),
        BreakHandler(),
        AbortHandler(),
        StatsHandler(),
        SessionSaveHandler(),
    ]
