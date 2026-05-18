r"""AgentRuntime: inbox-driven event loop.

One loop, one pipe, one match block. Everything is a ``RuntimeEvent``.

This module is the canonical sagent runtime, locked per
``docs/private/agent_v4_contract.md``. It owns the history dataclasses
(``UserMessage``, ``AssistantMessage``, ``ToolResult``, ``ToolCall``,
``BytesMessage``, ``SessionMessage``), the ``RuntimeEvent`` union, the
dispatch loop, the gate logic, the detach machinery, the AWAIT
mechanism, and the minimal ``Tool`` / ``Model`` / ``Compactor``
protocols. Adapters and wrappers in ``agent/agent.py`` present richer
protocols inward; the runtime sees only its minimal protocols.

Architecture
~~~~~~~~~~~~

``AgentRuntime.run_forever`` drains a ``GatedDeque[RuntimeEvent]`` in
a ``while True`` loop. Each drain returns a batch of events. A
``match`` dispatches each event, mutating instance state:
``running_tools``, ``cohort``, ``model_call``, ``compact_task``.
After the batch, a gate check fires the model if conditions are met.
That's the entire engine.

``run(msg)`` is a thin convenience wrapper for tests and child agents.

History
~~~~~~~

``list[UserMessage | AssistantMessage | ToolResult]``. Three types.
Match on type, not string descriptors.

Tool task lifecycle
~~~~~~~~~~~~~~~~~~~

Two registries with orthogonal semantics:

- ``cohort: set[str]`` names call IDs the model gate waits on.
  Membership = blocking the next model call.
- ``detached: dict[str, Task]`` names tasks whose completion splices
  into an existing ``[detached]`` placeholder via ``DetachedResult``
  rather than appending a fresh ``ToolResult``. Membership = in-place
  mutation of history at completion.

Both names describe the *role*, not the *origin*. ``cohort`` is not
``Undetached`` because most entries arrive from a fresh
``ModelResponseComplete`` and were never detached; ``Undetach`` is
just one of several ways to populate the set. Likewise ``detached``
is not ``Stubbed`` because not all entries got there via the
stub-and-let-finish path -- mid-stream-detach spawns straight into
``detached`` with no prior ``Detach`` event. Naming by role lets the
same set absorb every spawn path without renaming.

A call ID can be in either, both, or neither::

    cohort  detached  state
    yes     no        fresh spawn, gating, completion appends new entry
    no      yes       background work, completion splices placeholder
    yes     yes       re-gated after detach (``Undetach``)
    no      no        not running

Only one cohort generation is alive at a time (the gate requires
``not self.cohort`` to fire the next ``model_call``, so no new
``ModelResponseComplete`` can populate a second generation while the
current one is non-empty). ``detached`` has no such restriction:
tasks from any number of prior rounds can run in parallel.

Spawn paths:

- Regular ``ModelResponseComplete`` (no pending mid-stream user):
  cohort + ``running_tools``.
- ``ModelResponseComplete`` with mid-stream user pending: bypass
  cohort entirely -- task goes straight into ``detached`` with a
  ``[detached]`` placeholder appended. The gate then fires for the
  coalesced user content.

Transitions:

- ``Detach`` / implicit stub on ``UserMessage`` / ``Compact`` /
  ``Clear``: move cohort entry to detached, append ``[detached]``
  placeholder.
- ``Undetach``: re-add to cohort. Stays in detached -- so completion
  still splices the existing placeholder rather than appending a
  phantom result.
- ``Kill``: drop cohort + ``running_tools`` entry, cancel task.

Completion routing is decided in ``_run_tool_and_post`` by whether
``call.id in self.detached`` at finish time. If yes, the task posts
``DetachedResult`` (splice). Otherwise it posts a fresh ``ToolResult``
(append).

User-message dispatch timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provider APIs require a ``tool_result`` for every ``tool_use``, so
history must stay linear (no orphaned ``tool_use`` blocks). The
runtime maintains this invariant by stubbing unfinished tools with
``[detached]`` placeholders before appending user-side content that
would otherwise leave the branch open. Within that constraint, a
``UserMessage`` arriving while the agent is busy can be released at
one of four trigger points, trading responsiveness against
preservation of in-flight work:

- **Pre-MRC (immediate cancel).** Cancel ``model_call``, discard
  the partial stream, append user, gate fires now. Maximum
  responsiveness, partial response lost. Reached via ``Halt``
  followed by a fresh ``UserMessage``.
- **At ``ModelResponseComplete`` with tool detach.** Let the stream
  complete; if it produced tool_calls, detach them to background;
  gate fires for the coalesced user. Default for ``UserMessage``
  arriving while ``model_call`` is in flight (held in
  ``_mid_stream_queue``). No discarded model work; tools run
  asynchronously and splice in later.
- **At ``CohortComplete``.** Let stream complete *and* tools drain;
  insert user between tool results and the next round's model call.
  User waits for tools; in-flight work preserved fully. No surface
  today.
- **At ``ModelIdle``.** Let the entire round chain complete
  (model -> tools -> model -> ... -> final no-tool response), then
  dispatch. Most conservative. Reached by ``UserQueuedMessage``,
  buffered in the local ``queued`` list and drained only when
  ``not self._should_call_model()`` -- i.e. the gate would not fire
  naturally, meaning history's tail is an ``AssistantMessage`` with
  no tool_calls. Coalesces with ``\n\n`` joins.

``UserMessage`` arriving mid-cohort (model_call is None, cohort
non-empty) preempts: tools are stubbed to ``detached``, the user
message is appended, gate fires. This is the original "type to
redirect" path and is orthogonal to the four trigger points above
(which apply during streaming, not during cohort execution).

Provider role-alternation: Anthropic Messages enforces strict
``user``/``assistant`` alternation and returns 400 on consecutive
same-role turns; Gemini ``generateContent`` and OpenAI Chat
Completions are permissive. The runtime maintains strict alternation
regardless -- it's the lowest common denominator, and the gate's
invariants (``_should_call_model``, the mid-stream buffer drain,
the queued coalesce) all depend on the tail-of-history check
distinguishing user/tool entries from assistant entries.

Model call gate
~~~~~~~~~~~~~~~

Fires when all true:

- ``cohort`` is empty (all tools done or detached).
- ``model_call`` is None (no model streaming in flight).
- ``compact_task`` is None (no compaction in progress).
- ``_should_call_model()`` (history ends with user/tool content).

Five verbs on tools::

    halt              cancel model, gate on user (tools keep running)
    kill <id|all>     cancel task(s), remove from cohort
    detach <id|all>   stub + let finish, remove from cohort
    undetach <id|all> re-gate model on detached tool
    clear             cancel model, detach all, wipe history, gate

GatedDeque
~~~~~~~~~~

``push_back(item)`` adds to the back. ``push_front(*items)`` adds to
the front; an ``Await(types)`` among the items sets a gate that makes
``drain()`` block until an item matching those types (or ``Quit``)
arrives. Used by ``Halt`` / ``Clear`` / ``ModelResponseError`` to
wait for user input before resuming.

Published events
~~~~~~~~~~~~~~~~

The runtime publishes ``RuntimeEvent`` items to observers at specific
points. Events published (in addition to passing through the match):

- ``ModelResponsePartial`` -- each streaming text chunk.
- ``ModelResponseThinking`` -- each streaming thinking chunk.
- ``ModelResponseComplete`` -- full response (before tool spawn).
- ``ModelResponseCancelled`` -- model call cancelled mid-stream.
- ``ModelResponseError`` -- model call failed.
- ``UserMessage`` -- user message appended to history.
- ``ToolResult`` -- tool completed (in cohort). Carries optional
  ``diff`` / ``hint`` / ``summary`` / ``attachments`` for observer
  rendering; no separate event types.
- ``DetachedResult`` -- detached tool completed.
- ``CompactComplete`` -- compaction finished.
- ``ModelIdle`` -- model responded with no tool calls.
- ``CohortComplete`` -- all tool results arrived (natural completion
  only; not on preempt/halt/kill/detach).
- ``SaveSession`` -- end of each loop iteration; observer persists.
- ``ModelSwitch`` -- observer swaps ``runtime.model``.

Publish-only events (never enter the inbox; fanned out by wrappers
via ``runtime.publish(...)``):

- ``ToolLabel`` -- pre-execution label for a tool call (REPL).
- ``ChildEvent`` -- wrapped event from a child agent (AgentSpawn).
- ``ChildDoneEvent`` -- child agent completed; carries totals.

Composition
~~~~~~~~~~~

No stubs, no hooks, no callbacks. Extension is via observers on
published events. If something needs to happen outside the runtime
(persist session, swap model, track cost, enforce budget), it's an
event that an observer handles. If you're tempted to add a method
stub or a callback parameter, publish an event instead.

Why the runtime owns Model and Tool references: the alternative is a
generic runtime that only processes events, with the Agent calling
the model externally and pushing results into the inbox. That moves
the gate logic, task cancellation, and cohort tracking into the
Agent -- which is the hard part. Keeping Model/Tool in the runtime
keeps all state transitions in one match block, visible in one read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import asyncio
import contextlib
import contextvars
import dataclasses
import logging

from sagent.types.exceptions import (
    log_exception_or_warning,
    log_task_exception,
)

# Engine consumes a subset of the types. No re-export shim -- callers
# must import message types and event vocabulary from
# ``sagent.types.history`` / ``sagent.types.runtime`` directly.
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.runtime import (
    Clear,
    CohortComplete,
    CohortStarted,
    Compact,
    CompactComplete,
    CompactFailed,
    CompactStarted,
    Detach,
    DetachedResult,
    Halt,
    HistoryEntryUpdated,
    Kill,
    ModelCallStarted,
    ModelIdle,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelSwitch,
    ModelSwitchRejected,
    Quit,
    Recompact,
    RuntimeEvent,
    SaveSession,
    ToolResultPartial,
    Undetach,
    UserQueuedMessage,
)


# ``current_call_id_var`` is set by ``_run_tool_and_post`` before
# invoking the tool, so wrappers (in the Agent layer) and streaming
# tools can correlate ``ToolLabel`` / ``ToolResultPartial`` /
# ``runtime.publish`` events with the originating ``ToolCall.id``.
# Tools spawned by ``asyncio.create_task`` inherit the parent
# context, so the var resolves correctly inside the spawned task.
current_call_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_call_id", default=""
)

logger = logging.getLogger(__name__)


def _type_names(types: Sequence[type[object]]) -> tuple[str, ...]:
    """Return class names for compact debug logging."""
    return tuple(t.__name__ for t in types)


def _item_names(items: Sequence[object]) -> tuple[str, ...]:
    """Return event class names for compact debug logging."""
    return tuple(type(item).__name__ for item in items)


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Await:
    """Drain gate: block until an item matching ``types`` arrives."""

    types: tuple[type, ...]
    """Event classes that satisfy the gate (``Quit`` always does)."""


AWAIT_USER = Await((UserMessage, UserQueuedMessage, Quit))


class GatedDeque[T]:
    """Async item queue with drain gating.

    The gate counts items matching its types that are already in the
    queue at arm-time as ``_gate_baseline``. ``drain`` releases only
    when the running count exceeds the baseline (a NEW item arrived
    after arming) -- or when any ``Quit`` is seen, which always
    releases regardless of baseline.

    Why the baseline matters: under ``Halt``, the runtime arms
    ``AWAIT_USER`` then requeues drained items. If a pending
    ``UserMessage`` were already in the deque (the user typed multiple
    redirects while busy), it would otherwise satisfy the gate
    immediately and the ``/halt`` semantic ("wait for a fresh
    redirect") would be lost.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._gate: tuple[type, ...] | None = None
        self._gate_baseline: int = 0

    @property
    def gate_armed(self) -> bool:
        """True while a drain gate is set and not yet released.

        Surfaces inbox-level "waiting for X" state to the runtime's
        gate-check section so the model isn't fired against stale
        history while an ``Await`` (e.g. ``AWAIT_USER`` armed by
        ``Halt`` / ``ModelResponseError``) is still pending.
        """
        return self._gate is not None

    def push_back(self, item: T) -> None:
        """Add to back of queue.

        Args:
          item: Item to enqueue at the tail.

        """
        self._queue.put_nowait(item)

    def push_front(self, *items: T | Await) -> None:
        """Add to front of queue. Await sets the drain gate.

        Args:
          items: Items to push at the head, in argument order. An
              ``Await`` arms the gate; non-``Await`` items are queued.

        """
        old: list[T] = []
        while not self._queue.empty():
            try:
                old.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for item in items:
            if isinstance(item, Await):
                self._gate = item.types
                # Snapshot how many pre-existing items already satisfy
                # the gate (excluding ``Quit`` which always releases).
                self._gate_baseline = sum(
                    1
                    for i in old
                    if isinstance(i, item.types) and not isinstance(i, Quit)
                )
                logger.debug(
                    "runtime inbox gate armed: gate=%s baseline=%d queued=%s",
                    _type_names(item.types),
                    self._gate_baseline,
                    _item_names(old),
                )
            else:
                self._queue.put_nowait(item)
        for item in old:
            self._queue.put_nowait(item)

    async def drain(self) -> list[T]:
        """Block until items are available, then return all queued items.

        When a gate is set, keeps draining until a NEW item matching
        the gate types arrives (count exceeds the baseline captured at
        arm time) or until any ``Quit`` is observed.

        Returns:
          items: All queued items in arrival order.

        """
        first = await self._queue.get()
        items = [first]
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if self._gate is not None:
            gate = self._gate
            baseline = self._gate_baseline
            if not self._gate_satisfied(items, gate, baseline):
                logger.debug(
                    "runtime inbox gate waiting: gate=%s baseline=%d buffered=%s",
                    _type_names(gate),
                    baseline,
                    _item_names(items),
                )
            while not self._gate_satisfied(items, gate, baseline):
                items.append(await self._queue.get())
                while not self._queue.empty():
                    try:
                        items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            logger.debug(
                "runtime inbox gate released: gate=%s baseline=%d buffered=%s",
                _type_names(gate),
                baseline,
                _item_names(items),
            )
            self._gate = None
            self._gate_baseline = 0
        return items

    @classmethod
    def _gate_satisfied(
        cls,
        items: Sequence[object],
        gate: tuple[type, ...],
        baseline: int,
    ) -> bool:
        """Return True when ``items`` satisfies the gate beyond ``baseline``."""
        if any(isinstance(i, Quit) for i in items):
            return True
        count = sum(1 for i in items if isinstance(i, gate) and not isinstance(i, Quit))
        return count > baseline


class Tool(Protocol):
    """Minimal tool interface.

    ``run`` returns a fully-formed ``ToolResult``. The runtime stamps
    ``call_id`` from the originating ``ToolCall`` if the result has an
    empty one, so most tools construct ``ToolResult(call_id="", ...)``
    and let the runtime fill it in.
    """

    @property
    def name(self) -> str:
        """Return the tool's stable identifier.

        Returns:
          name: Lookup key in ``AgentRuntime.tools_map``.

        """
        ...

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the tool against ``args``.

        Args:
          args: Parsed directive arguments.

        Returns:
          result: ``ToolResult`` (``call_id`` may be empty; runtime fills).

        """
        ...


class Model(Protocol):
    """Minimal model interface for the runtime."""

    async def stream(
        self,
        history: list[HistoryEntry],
        system: str,
        tools: list[Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        """Stream a model response.

        Args:
          history: Conversation history.
          system: System prompt.
          tools: Available tools.
          on_text: Callback for each streamed text chunk.
          on_thinking: Callback for each streamed thinking chunk.

        Returns:
          message: Complete assistant response.

        """
        ...


class Compactor(Protocol):
    """Minimal compactor interface."""

    async def compact(
        self,
        history: list[HistoryEntry],
        model: Model,
        args: str = "",
    ) -> list[HistoryEntry]:
        """Summarize history into a shorter sequence.

        Args:
          history: Full conversation history.
          model: Model to use for summarization.
          args: Custom compaction instructions.

        Returns:
          summary: Compacted history.

        """
        ...


class AgentRuntime:
    """Inbox-driven event loop agent.

    Args:
      model: Model implementation used for ``stream``.
      system: System prompt threaded through every model call.
      tools: Tools registered for dispatch; must have unique ``name``.
      compactor: Optional compactor invoked on ``Compact`` / ``Recompact``.

    """

    def __init__(
        self,
        *,
        model: Model,
        system: str = "",
        tools: list[Tool] | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        self.model = model
        self.system = system
        self.tools_map: dict[str, Tool] = {}
        for t in tools or []:
            if t.name in self.tools_map:
                raise ValueError(f"Duplicate tool name: {t.name!r}")
            self.tools_map[t.name] = t
        self.compactor = compactor
        self.history: list[HistoryEntry] = []
        self.inbox: GatedDeque[RuntimeEvent] = GatedDeque()
        # Task[None]: tools post results to inbox, not via return value.
        self.detached: dict[str, asyncio.Task[None]] = {}
        self.observers: list[Callable[[RuntimeEvent], None]] = []
        # Lifted from run_forever locals for observer/REPL visibility.
        # Task[None]: tools/model post results to inbox, not via return.
        self.running_tools: dict[str, asyncio.Task[None]] = {}
        self.cohort: set[str] = set()
        # ``_cohort_seen`` tracks "we started a cohort and want to publish
        # ``CohortComplete`` when it naturally drains." Reset by every
        # stop-cohort path so a preempted cohort doesn't fire complete.
        self._cohort_seen: bool = False
        self.model_call: asyncio.Task[None] | None = None
        self.compact_task: asyncio.Task[None] | None = None
        # Buffered ModelSwitch awaiting a safe moment (no in-flight
        # model call, no compaction). Applied at the end of each
        # iteration once that condition holds.
        self._pending_switch: ModelSwitch | None = None
        # Mid-stream UserMessages: buffered while ``model_call`` is in
        # flight. On ``ModelResponseComplete`` the buffer is coalesced
        # into one ``UserMessage`` appended after the assistant response;
        # any tool calls the response carried are detached so the
        # follow-up round for the user's input fires immediately
        # ("type to redirect" UX during streaming, matching the
        # mid-cohort behavior of stub-and-detach).
        self._mid_stream_queue: list[UserMessage] = []

    def publish(self, event: RuntimeEvent) -> None:
        """Fan out an event to all observers.

        Args:
          event: Event to deliver to every observer; exceptions are logged.

        """
        for obs in self.observers:
            try:
                obs(event)
            except Exception:
                logger.exception("observer raised on %s", type(event).__name__)

    async def run_forever(self) -> None:
        """Drain inbox, dispatch, repeat. The entire engine."""
        awaiting_user = False
        queued: list[UserQueuedMessage] = []

        while True:
            try:
                self._collect_detached()
                items = await self.inbox.drain()

                for item_idx, item in enumerate(items):
                    match item:
                        case Quit():
                            logger.debug(
                                "runtime quit: model_call=%s compact_task=%s "
                                "running_tools=%d cohort=%d detached=%d",
                                self.model_call is not None,
                                self.compact_task is not None,
                                len(self.running_tools),
                                len(self.cohort),
                                len(self.detached),
                            )
                            if self.model_call:
                                self.model_call.cancel()
                            if self.compact_task:
                                self.compact_task.cancel()
                            for t in self.running_tools.values():
                                t.cancel()
                            return

                        case Halt():
                            logger.debug(
                                "runtime halt: model_call=%s compact_task=%s "
                                "running_tools=%d cohort=%d detached=%d "
                                "mid_stream=%d",
                                self.model_call is not None,
                                self.compact_task is not None,
                                len(self.running_tools),
                                len(self.cohort),
                                len(self.detached),
                                len(self._mid_stream_queue),
                            )
                            if self.model_call:
                                self.model_call.cancel()
                                self.model_call = None
                            # Halt intentionally leaves the cohort intact:
                            # running tools keep going and their results land
                            # via the normal cohort gate after the user resumes.
                            # Do NOT call ``_stop_all_tools`` here -- that's
                            # for hard preempts (Clear / Compact / mid-cohort
                            # UserMessage / Detach / Kill), not soft Halt.
                            # Preserve any mid-stream typed content: commit
                            # it to history (alternation-coalesce handles
                            # back-to-back UserMessages) and publish the
                            # coalesced bar -- the preview drops, the bar
                            # appears, single UI transition.
                            coalesced = self._drain_mid_stream_queue()
                            if coalesced is not None:
                                self.publish(coalesced)
                            self.inbox.push_front(
                                AWAIT_USER,
                                *items[item_idx + 1 :],
                            )
                            awaiting_user = True
                            break

                        case Clear():
                            if self.model_call:
                                self.model_call.cancel()
                                self.model_call = None
                            self._stop_all_tools(mode="detach")
                            queued.clear()
                            self._mid_stream_queue.clear()
                            self.history.clear()
                            self.inbox.push_front(
                                AWAIT_USER,
                                *items[item_idx + 1 :],
                            )
                            awaiting_user = True
                            break

                        case ModelResponseError(exception=exc):
                            logger.debug(
                                "runtime model response error: error=%s "
                                "model_call=%s compact_task=%s running_tools=%d "
                                "cohort=%d detached=%d",
                                type(exc).__name__,
                                self.model_call is not None,
                                self.compact_task is not None,
                                len(self.running_tools),
                                len(self.cohort),
                                len(self.detached),
                            )
                            self.model_call = None
                            self._append_or_coalesce_user(
                                UserMessage(
                                    text=f"[Error: {type(exc).__name__}: {exc}]",
                                ),
                            )
                            self.publish(item)
                            # Preserve mid-stream content past the error;
                            # commit + publish the coalesced bar so the
                            # pending preview transitions to a permanent
                            # entry.
                            coalesced = self._drain_mid_stream_queue()
                            if coalesced is not None:
                                self.publish(coalesced)
                            self.inbox.push_front(
                                AWAIT_USER,
                                *items[item_idx + 1 :],
                            )
                            awaiting_user = True
                            break

                        case Kill(call_id=cid):
                            if cid is None:
                                logger.debug(
                                    "runtime kill all tools: running_tools=%d "
                                    "cohort=%d detached=%d",
                                    len(self.running_tools),
                                    len(self.cohort),
                                    len(self.detached),
                                )
                                self._stop_all_tools(mode="kill")
                            elif cid in self.running_tools:
                                logger.debug("runtime kill tool: call_id=%s", cid)
                                self._stop_tool(
                                    cid,
                                    self.running_tools.pop(cid),
                                    mode="kill",
                                )
                                self.cohort.discard(cid)
                            else:
                                logger.debug(
                                    "runtime kill missed tool: call_id=%s", cid
                                )

                        case Detach(call_id=cid):
                            if cid is None:
                                logger.debug(
                                    "runtime detach all tools: running_tools=%d "
                                    "cohort=%d detached=%d",
                                    len(self.running_tools),
                                    len(self.cohort),
                                    len(self.detached),
                                )
                                self._stop_all_tools(mode="detach")
                            elif cid in self.running_tools:
                                logger.debug("runtime detach tool: call_id=%s", cid)
                                self._stop_tool(
                                    cid,
                                    self.running_tools.pop(cid),
                                    mode="detach",
                                )
                                self.cohort.discard(cid)
                            else:
                                logger.debug(
                                    "runtime detach missed tool: call_id=%s", cid
                                )

                        case Undetach(call_id=cid):
                            if cid is None:
                                for did in self.detached:
                                    self.cohort.add(did)
                            elif cid in self.detached:
                                self.cohort.add(cid)

                        case Compact(args=args) | Recompact(args=args):
                            if self.compact_task and not self.compact_task.done():
                                logger.debug(
                                    "runtime compact ignored while compact_task active",
                                )
                                continue
                            logger.debug(
                                "runtime compact start: kind=%s model_call=%s "
                                "running_tools=%d cohort=%d detached=%d args=%r",
                                type(item).__name__,
                                self.model_call is not None,
                                len(self.running_tools),
                                len(self.cohort),
                                len(self.detached),
                                args,
                            )
                            if self.model_call:
                                self.model_call.cancel()
                                self.model_call = None
                            self._stop_all_tools(mode="detach")
                            queued.clear()
                            # Capture buffered mid-stream input into the snapshot
                            # the compactor will see; publish the coalesced
                            # bar so the pending preview transitions to a
                            # committed entry.
                            coalesced = self._drain_mid_stream_queue()
                            if coalesced is not None:
                                self.publish(coalesced)
                            self.compact_task = asyncio.create_task(
                                self._compact_and_post(args),
                            )
                            self.compact_task.add_done_callback(
                                log_task_exception(logger, "compaction task crashed"),
                            )
                            self.publish(CompactStarted())

                        case CompactComplete(
                            summary=summary,
                            snapshot_len=n,
                        ):
                            self.compact_task = None
                            new_items = self.history[n:]
                            self.history.clear()
                            self.history.extend(summary)
                            self.history.extend(new_items)
                            self.publish(item)

                        case CompactFailed(exception=exc):
                            self.compact_task = None
                            self._append_or_coalesce_user(
                                UserMessage(
                                    text=(
                                        f"[Compaction error:"
                                        f" {type(exc).__name__}: {exc}]"
                                    ),
                                ),
                            )
                            self.publish(item)

                        case UserMessage():
                            if self.model_call is not None:
                                # Mid-stream: buffer only. The ``queued_input_pane``
                                # in the REPL renders ``pending_mid_stream()`` as
                                # a dim preview while the buffer is non-empty,
                                # so the user has immediate visual feedback
                                # without a duplicate bar in console. The bar
                                # appears on drain (ModelResponseComplete /
                                # Halt / ModelResponseError / Compact) when
                                # the coalesced UserMessage is published --
                                # at which point the preview drops because the
                                # buffer is empty. One UI surface at a time.
                                self._mid_stream_queue.append(item)
                            else:
                                # Mid-cohort or idle: preempt and append.
                                # Coalesce on the alternation-invariant helper
                                # so two same-batch Enters (or a post-halt
                                # follow-up) don't stack as consecutive user
                                # turns in history.
                                self._stop_all_tools(mode="detach")
                                self._append_or_coalesce_user(item)
                                self.publish(item)
                            awaiting_user = False

                        case UserQueuedMessage():
                            if awaiting_user:
                                coalesced = UserMessage(
                                    text=item.text,
                                    attachments=item.attachments,
                                )
                                self._append_or_coalesce_user(coalesced)
                                self.publish(coalesced)
                                awaiting_user = False
                            else:
                                queued.append(item)

                        case ModelResponsePartial():
                            self.publish(item)

                        case ModelResponseThinking():
                            self.publish(item)

                        case ModelResponseComplete(message=msg):
                            self.model_call = None
                            self.history.append(msg)
                            self.publish(item)
                            if self._mid_stream_queue:
                                # User typed mid-stream. Cut their content in
                                # line: relegate any tool calls to background
                                # (placeholder + detached task; the result
                                # splices in via ``DetachedResult`` when the
                                # tool finishes), then append the coalesced
                                # user content so the gate fires for it next.
                                # No ``CohortStarted`` / ``ModelIdle`` here:
                                # this round did not idle (a follow-up is
                                # about to fire) and no cohort gates the model.
                                for tc in msg.tool_calls:
                                    self.history.append(
                                        ToolResult(
                                            call_id=tc.id,
                                            parent_id=msg.id,
                                            content="[detached]",
                                        ),
                                    )
                                    detached_task = asyncio.create_task(
                                        self._run_tool_and_post(tc, parent_id=msg.id),
                                    )
                                    detached_task.add_done_callback(
                                        log_task_exception(
                                            logger,
                                            f"detached tool {tc.name!r} crashed",
                                        ),
                                    )
                                    self.detached[tc.id] = detached_task
                                # Commit mid-stream input to history and
                                # publish the coalesced bar -- the pending
                                # preview drops as the buffer empties and
                                # the bar appears in console, single UI
                                # transition.
                                coalesced = self._drain_mid_stream_queue()
                                if coalesced is not None:
                                    self.publish(coalesced)
                            elif msg.tool_calls:
                                self._cohort_seen = True
                                self.publish(CohortStarted())
                                logger.debug(
                                    "runtime cohort start: parent_id=%s tools=%s",
                                    msg.id,
                                    [(tc.id, tc.name) for tc in msg.tool_calls],
                                )
                                for tc in msg.tool_calls:
                                    self.cohort.add(tc.id)
                                    tool_task = asyncio.create_task(
                                        self._run_tool_and_post(tc, parent_id=msg.id),
                                    )
                                    tool_task.add_done_callback(
                                        log_task_exception(
                                            logger,
                                            f"cohort tool {tc.name!r} crashed",
                                        ),
                                    )
                                    self.running_tools[tc.id] = tool_task
                            else:
                                self.publish(ModelIdle())

                        case ModelResponseCancelled():
                            self.model_call = None
                            self.publish(item)

                        case ToolResultPartial():
                            self.publish(item)

                        case ToolResult(call_id=cid) if cid in self.cohort:
                            self.cohort.discard(cid)
                            self.running_tools.pop(cid, None)
                            self.history.append(item)
                            self.publish(item)

                        case ToolResult(call_id=cid) if cid in self.detached:
                            # In-batch race: the tool task completed before
                            # ``_stop_all_tools`` reclassified its call_id as
                            # detached. The ``ToolResult`` was pushed as a
                            # regular result (because the task ran
                            # ``self.detached`` check before the stub), but a
                            # peer item in the same drain batch (e.g. a
                            # tool-pushed ``UserMessage``) triggered a preempt
                            # that cleared the cohort. Without this case the
                            # result would fall through to ``_`` and leave
                            # the assistant's ``tool_use`` paired only with
                            # the ``[detached]`` placeholder forever. Splice
                            # the real content into the placeholder exactly
                            # like ``DetachedResult`` does.
                            del self.detached[cid]
                            for i, prior in enumerate(self.history):
                                if (
                                    isinstance(prior, ToolResult)
                                    and prior.call_id == cid
                                ):
                                    self.history[i] = dataclasses.replace(
                                        prior,
                                        content=item.content,
                                        is_error=item.is_error,
                                    )
                                    self.publish(
                                        HistoryEntryUpdated(entry=self.history[i]),
                                    )
                                    break
                            self.publish(item)

                        case DetachedResult():
                            self.cohort.discard(item.call_id)
                            # Splice into the existing placeholder so the
                            # model sees the real result in the slot it
                            # already expects, without duplicating the
                            # full content into a phantom user message.
                            # Both ``[detached]`` (preempt) and ``[Running
                            # in background: ...]`` (explicit-bg)
                            # placeholders match by ``call_id``.
                            spliced = False
                            for i, prior in enumerate(self.history):
                                if (
                                    isinstance(prior, ToolResult)
                                    and prior.call_id == item.call_id
                                ):
                                    self.history[i] = dataclasses.replace(
                                        prior,
                                        content=item.content,
                                        is_error=item.is_error,
                                    )
                                    self.publish(
                                        HistoryEntryUpdated(entry=self.history[i]),
                                    )
                                    spliced = True
                                    break
                            if not spliced:
                                # No placeholder to splice into (rare: result
                                # arrived before the stub was inserted). Fall
                                # back to a user message so the content isn't
                                # silently dropped.
                                self.history.append(
                                    UserMessage(
                                        text=(
                                            f"[Tool {item.call_id} completed]\n"
                                            f"{item.content}"
                                        ),
                                    ),
                                )
                            elif isinstance(self.history[-1], AssistantMessage):
                                # Splice landed after the preempted round
                                # already responded. History tail is an
                                # ``AssistantMessage``, so the end-of-loop
                                # model-call gate won't fire on its own.
                                # Append a terse notification (the real
                                # content is already in its proper slot
                                # above) so the model wakes and can react
                                # to the now-real tool result.
                                self.history.append(
                                    UserMessage(
                                        text=(
                                            f"[Detached tool {item.call_id} completed]"
                                        ),
                                    ),
                                )
                            self.publish(item)

                        case ModelSwitch():
                            # Buffer the switch; applied below once the
                            # in-flight model call / compaction (if any)
                            # completes. The OLD model finishes recording
                            # its cost before the swap lands.
                            self._pending_switch = item

                        case _:
                            pass

                if (
                    self._pending_switch is not None
                    and self.model_call is None
                    and self.compact_task is None
                ):
                    pending = self._pending_switch
                    self._pending_switch = None
                    # ``apply`` runs slash-handler-supplied code
                    # (``Agent.swap_model``), the only synchronous user-facing
                    # raiser in the per-iteration gates. Isolate so a
                    # rejected swap (e.g. new model's window < current
                    # budget) doesn't skip the remaining gates -- the
                    # model-call gate must still fire on this iteration
                    # or the next drain blocks with no live state to wake
                    # it. ``publish`` only on success: observers treat
                    # ``ModelSwitch`` as "the swap landed."
                    try:
                        pending.apply()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 -- surface swap errors without halting the engine
                        log_exception_or_warning(
                            logger, f"model swap rejected ({pending.label})", exc
                        )
                        self.publish(
                            ModelSwitchRejected(
                                exception=exc,
                                label=pending.label,
                            ),
                        )
                    else:
                        self.publish(pending)

                if not self.cohort and self._cohort_seen:
                    logger.debug(
                        "runtime cohort complete: running_tools=%d detached=%d",
                        len(self.running_tools),
                        len(self.detached),
                    )
                    self.publish(CohortComplete())
                    self._cohort_seen = False

                # ``UserQueuedMessage`` drains at ``ModelIdle``, not
                # ``CohortComplete``. The ``not self._should_call_model()``
                # check distinguishes "round chain ended" (history tail is
                # ``AssistantMessage`` with no tool_calls, i.e. the gate
                # would not fire on its own) from "between rounds" (history
                # tail is ``ToolResult``, the gate would fire next round).
                # Under the former, draining is correct; under the latter,
                # we'd be cutting the queued content into a chain the user
                # didn't intend to interrupt.
                if (
                    not self.cohort
                    and self.model_call is None
                    and self.compact_task is None
                    and not self._should_call_model()
                    and queued
                ):
                    coalesced = UserMessage(
                        text="\n\n".join(q.text for q in queued),
                        attachments=sum(
                            (q.attachments for q in queued),
                            (),
                        ),
                    )
                    self._append_or_coalesce_user(coalesced)
                    queued.clear()
                    # Publish so observers (renderers, persistence, the
                    # REPL's ``make_queued_input_clearer``) see the
                    # commit. Without this, the user bar never renders
                    # in ``console_pane`` and ``queued_input`` is never
                    # cleared.
                    self.publish(coalesced)

                if (
                    not self.cohort
                    and self.model_call is None
                    and self.compact_task is None
                    and not self.inbox.gate_armed
                    and self._should_call_model()
                ):
                    logger.debug(
                        "runtime model call start: history=%d detached=%d queued=%d",
                        len(self.history),
                        len(self.detached),
                        len(queued),
                    )
                    # ``inbox.gate_armed`` blocks firing while ``AWAIT_USER``
                    # is pending (armed by ``Halt`` / ``ModelResponseError``).
                    # Without this guard the model would fire on the stale
                    # ``UserMessage`` still at history.tail, treating the
                    # cancellation as a retry rather than waiting for the
                    # user's next input.
                    self.model_call = asyncio.create_task(
                        self._stream_and_post(),
                    )
                    self.model_call.add_done_callback(
                        log_task_exception(logger, "model-call task crashed"),
                    )
                    self.publish(ModelCallStarted())

                self.publish(SaveSession())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- master catch around the dispatch body; a sync raise (e.g. `pending.apply` for a queued ModelSwitch) must not tear down the engine
                log_exception_or_warning(logger, "dispatch loop iteration raised", exc)

    async def run(self, msg: UserMessage) -> list[HistoryEntry]:
        """Process one user message to completion.

        Convenience wrapper for tests and AgentSpawn. Sends the
        message, runs until the model is idle, returns history.

        Args:
          msg: User message to process.

        Returns:
          history: Conversation history after processing.

        """
        self.inbox.push_back(msg)

        done = asyncio.Event()
        turns_before = len(self.history)

        def _watch(event: RuntimeEvent) -> None:
            if isinstance(event, ModelIdle):
                done.set()

        self.observers.append(_watch)
        task = asyncio.create_task(self.run_forever())
        task.add_done_callback(
            log_task_exception(logger, "run_forever driver task crashed"),
        )

        await done.wait()
        self.inbox.push_back(Quit())
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.observers.remove(_watch)
        return self.history[turns_before:]

    def pending_mid_stream(self) -> Sequence[UserMessage]:
        """Snapshot of mid-stream ``UserMessage`` items awaiting drain.

        Buffered while ``model_call`` is in flight; drained into history
        on ``ModelResponseComplete`` (or ``Halt`` / ``ModelResponseError``
        / ``Compact``). Public surface for UI surfaces (``queue_pane``)
        that need to show "your Enter mid-stream is queued, waiting for
        the model to finish."

        Returns:
          snapshot: Read-only tuple of the buffered messages in arrival
              order. Returns an empty tuple when nothing is buffered.

        """
        return tuple(self._mid_stream_queue)

    def _should_call_model(self) -> bool:
        """Return True when history ends with content the model should answer."""
        if not self.history:
            return False
        return isinstance(self.history[-1], (UserMessage, ToolResult))

    def _append_or_coalesce_user(self, item: UserMessage) -> None:
        r"""Append ``item`` to history; coalesce with tail if also ``UserMessage``.

        Anthropic-style chat APIs require user/assistant turn alternation.
        Several runtime paths produce a ``UserMessage`` while the previous
        history entry is already a ``UserMessage`` (no assistant turn
        between -- the model was cancelled, errored, or simply never
        ran). Appending naively would generate back-to-back user turns
        and the next model call would 400.

        Sites that need this:

        - Idle/mid-cohort ``UserMessage`` branch (two rapid Enters land
          in one drain batch; the gate only sets ``model_call`` AFTER
          the per-item loop).
        - ``ModelResponseError`` synthesizing ``[Error: ...]`` (model
          never produced an assistant turn for the preceding user input).
        - Halt-then-fresh-user (cancellation drops any partial assistant
          content; only the user's prior message is in history).
        - ``_drain_mid_stream_queue`` when invoked from a halt/compact/
          error path (no assistant turn between the prior user input
          and the buffered mid-stream content).
        - ``UserQueuedMessage`` idle drain (Tab-staged content lands as
          a coalesced ``UserMessage``; history tail can already be a
          ``UserMessage`` from a same-batch race).

        Coalesce semantics: text joins with ``\n\n``; attachments
        concatenate in arrival order. The tail entry's ``id`` is
        preserved so downstream consumers keyed on ids remain stable.
        """
        if self.history and isinstance(self.history[-1], UserMessage):
            tail = self.history[-1]
            self.history[-1] = dataclasses.replace(
                tail,
                text=f"{tail.text}\n\n{item.text}",
                attachments=tail.attachments + item.attachments,
            )
        else:
            self.history.append(item)

    def _stop_tool(
        self,
        cid: str,
        task: asyncio.Task[None],
        *,
        mode: Literal["detach", "kill"],
    ) -> None:
        """Transition one cohort tool out of the cohort, pairing its tool_use.

        Always appends a ``ToolResult`` placeholder so history alternation
        stays well-formed; always routes the task into ``self.detached``
        so any late-arriving content splices into the placeholder via
        the ``DetachedResult`` path (the cohort gate no longer matches
        once cid leaves the cohort).

        ``mode="detach"`` lets the task complete naturally
        (``[detached]`` placeholder); ``mode="kill"`` cancels the task
        (``[cancelled]``, ``is_error=True``). The closed mode set is
        deliberate: any soft path that wants tools to keep running
        without leaving the cohort (Halt) must NOT call this.
        """
        placeholder, is_error = (
            ("[cancelled]", True) if mode == "kill" else ("[detached]", False)
        )
        self.history.append(
            ToolResult(call_id=cid, content=placeholder, is_error=is_error),
        )
        self.detached[cid] = task
        if mode == "kill":
            task.cancel()

    def _stop_all_tools(self, *, mode: Literal["detach", "kill"]) -> None:
        """Run :meth:`_stop_tool` for every cohort tool; reset cohort state.

        The single choke point for the "preempt the cohort" semantic
        (Clear, Compact, mid-cohort UserMessage, Detach-all, Kill-all).
        Halt deliberately does NOT preempt the cohort; do not call from
        ``Halt`` -- results land via the normal cohort gate after Halt.
        """
        for cid, task in list(self.running_tools.items()):
            self._stop_tool(cid, task, mode=mode)
        self.running_tools.clear()
        self.cohort.clear()
        self._cohort_seen = False

    def _drain_mid_stream_queue(self) -> UserMessage | None:
        r"""Append a coalesced ``UserMessage`` for any buffered mid-stream input.

        Multiple ``UserMessage`` items received while ``model_call`` was in
        flight collapse into a single entry with ``\n\n``-joined text
        and attachments concatenated in arrival order -- mirroring
        :class:`UserQueuedMessage` coalescing semantics.

        Returns:
          appended: The coalesced ``UserMessage`` (so callers can publish
              it) when the buffer was non-empty; ``None`` otherwise.
              ``self.history`` is mutated in place.

        """
        if not self._mid_stream_queue:
            return None
        coalesced = UserMessage(
            text="\n\n".join(q.text for q in self._mid_stream_queue),
            attachments=sum(
                (q.attachments for q in self._mid_stream_queue),
                (),
            ),
        )
        self._mid_stream_queue.clear()
        self._append_or_coalesce_user(coalesced)
        return coalesced

    def _collect_detached(self) -> None:
        """Clean up detached tasks that completed or were cancelled.

        Completed tasks already posted their result as DetachedResult
        from ``_run_tool_and_post`` and removed themselves from
        ``self.detached``. This handles cancelled tasks that never
        posted.
        """
        for cid in [c for c, t in self.detached.items() if t.done()]:
            del self.detached[cid]

    async def _stream_and_post(self) -> None:
        """Stream a model response, posting chunks and the final message."""
        chars = 0

        def on_text(text: str) -> None:
            nonlocal chars
            chars += len(text)
            self.inbox.push_back(ModelResponsePartial(text))

        def on_thinking(text: str) -> None:
            self.inbox.push_back(ModelResponseThinking(text))

        try:
            response = await self.model.stream(
                self.history,
                self.system,
                list(self.tools_map.values()),
                on_text,
                on_thinking,
            )
            self.inbox.push_back(ModelResponseComplete(message=response))
        except asyncio.CancelledError:
            # Publish directly (not via inbox): the Halt handler has
            # already armed ``AWAIT_USER``, so an inbox push would gate
            # this event behind the next ``UserMessage`` -- and the
            # render/activity observers would miss it. They need to
            # flush the streaming buffer and stop the spinner NOW.
            self.publish(
                ModelResponseCancelled(output_chars_estimate=chars),
            )
        except Exception as exc:  # noqa: BLE001 -- log_exception_or_warning routes UserFacingError to warning, others to exception; intentional catch-all for model-call failures
            log_exception_or_warning(logger, "model call failed", exc)
            self.inbox.push_back(ModelResponseError(exc))

    async def _run_tool_and_post(
        self,
        call: ToolCall,
        *,
        parent_id: int = -1,
    ) -> None:
        """Run one tool invocation, post the ``ToolResult`` to the inbox.

        Tool authors return a fully-formed ``ToolResult``; the runtime
        stamps ``call_id`` (and ``parent_id`` when unset) from the
        originating assistant message. Exceptions auto-convert to
        ``is_error=True``. If the tool was detached mid-flight, the
        runtime emits ``DetachedResult`` instead of ``ToolResult`` so
        the late completion arrives as context rather than being
        silently dropped by the cohort gate.
        """
        tool = self.tools_map.get(call.name)
        if tool is None:
            logger.debug(
                "runtime tool unknown: call_id=%s tool=%s parent_id=%s",
                call.id,
                call.name,
                parent_id,
            )
            self.inbox.push_back(
                ToolResult(
                    call_id=call.id,
                    parent_id=parent_id,
                    content=f"Unknown tool: {call.name}",
                    is_error=True,
                ),
            )
            return
        # Expose ``call.id`` via ContextVar so wrappers + streaming tools
        # can publish ``ToolLabel`` / ``ToolResultPartial`` correlated with
        # the originating call.
        call_token = current_call_id_var.set(call.id)
        try:
            logger.debug(
                "runtime tool start: call_id=%s tool=%s parent_id=%s",
                call.id,
                call.name,
                parent_id,
            )
            result = await tool.run(call.args)
            replacements: dict[str, object] = {}
            if not result.call_id:
                replacements["call_id"] = call.id
            if result.parent_id == -1 and parent_id != -1:
                replacements["parent_id"] = parent_id
            if replacements:
                result = dataclasses.replace(result, **replacements)
        except asyncio.CancelledError:
            logger.debug(
                "runtime tool cancelled: call_id=%s tool=%s parent_id=%s",
                call.id,
                call.name,
                parent_id,
            )
            # Synthesize a result and fall through to the post block so
            # the cohort/detached splice gates always pair the assistant's
            # tool_use with a tool_result. Without this, any asyncio
            # cancellation (Kill, future paths) orphans the tool_use and
            # the next provider call fails with HTTP 400.
            result = ToolResult(
                call_id=call.id,
                parent_id=parent_id,
                content="[cancelled]",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 -- surface arbitrary tool errors
            logger.debug(
                "runtime tool failed: call_id=%s tool=%s parent_id=%s error=%s",
                call.id,
                call.name,
                parent_id,
                type(exc).__name__,
            )
            result = ToolResult(
                call_id=call.id,
                parent_id=parent_id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        finally:
            current_call_id_var.reset(call_token)
        if call.id in self.detached:
            logger.debug(
                "runtime tool complete detached: call_id=%s tool=%s is_error=%s",
                call.id,
                call.name,
                result.is_error,
            )
            del self.detached[call.id]
            self.inbox.push_back(
                DetachedResult(
                    call_id=call.id,
                    content=result.content,
                    is_error=result.is_error,
                ),
            )
        else:
            logger.debug(
                "runtime tool complete cohort: call_id=%s tool=%s is_error=%s",
                call.id,
                call.name,
                result.is_error,
            )
            self.inbox.push_back(result)

    async def _compact_and_post(self, args: str) -> None:
        """Run compaction and post the result."""
        if self.compactor is None:
            return
        snapshot_len = len(self.history)
        try:
            summary = await self.compactor.compact(
                list(self.history),
                self.model,
                args,
            )
            self.inbox.push_back(
                CompactComplete(summary=summary, snapshot_len=snapshot_len),
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 -- compaction calls the model; catch-all routes UserFacingError to warning, others to exception
            log_exception_or_warning(logger, "compaction failed", exc)
            self.inbox.push_back(
                CompactFailed(exception=exc, snapshot_len=snapshot_len),
            )
