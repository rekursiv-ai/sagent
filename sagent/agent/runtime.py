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

# Engine consumes a subset of the types. No re-export shim -- callers
# must import message types and event vocabulary from
# ``sagent.types.history`` / ``sagent.types.runtime`` directly.
from sagent.agent.context import (
    InvalidContextError,
    ResolvedContext,
    resolve_context,
    validate_context,
)
from sagent.types.exceptions import (
    log_exception_or_warning,
    log_task_exception,
)
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.runtime import (
    AgentIdle,
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
from sagent.types.tape import (
    ContextClear,
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
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


def _sanitize_for_send(
    entries: Sequence[HistoryEntry],
) -> tuple[HistoryEntry, ...]:
    """Return a wire-format-valid version of ``entries``.

    Stronger than :func:`_insert_synth_trs_in_payload`: also drops
    orphan ``ToolResult`` entries (no preceding ``AssistantMessage``
    declaring the ``call_id``) and duplicate ``call_id`` entries.
    Inserts synthetic ``[interrupted]`` ``ToolResult`` records for any
    ``AssistantMessage.tool_calls`` id that lacks a matching
    ``ToolResult`` before the next ``AssistantMessage`` or non-``TR``
    entry.

    Used by the runtime gate's rescue path when structural repair
    can't pair tool_use / tool_result across overrides (e.g.
    accumulated microcompact debt from a session predating the
    cache-warm gate fix).

    Idempotent.
    """
    out: list[HistoryEntry] = []
    pending: list[str] = []
    seen: set[str] = set()

    def _flush_pending() -> None:
        for cid in pending:
            out.append(
                ToolResult(
                    call_id=cid,
                    content="[interrupted]",
                    is_error=True,
                ),
            )
            seen.add(cid)
        pending.clear()

    for entry in entries:
        if isinstance(entry, AssistantMessage):
            _flush_pending()
            out.append(entry)
            pending.extend(tc.id for tc in entry.tool_calls)
        elif isinstance(entry, ToolResult):
            if entry.call_id in seen:
                continue
            if entry.call_id not in pending:
                continue  # orphan: drop
            out.append(entry)
            pending.remove(entry.call_id)
            seen.add(entry.call_id)
        else:
            _flush_pending()
            out.append(entry)
    _flush_pending()
    return tuple(out)


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

    def empty(self) -> bool:
        """True iff no items are queued. Snapshot at call time.

        Used by ``AgentRuntime._fully_drained`` to decide whether to
        publish ``AgentIdle`` before the next blocking ``drain()``.
        Safe against TOCTOU: callers must read this and act on it
        within the same synchronous block (no intervening ``await``),
        which asyncio's cooperative scheduling guarantees no other
        coroutine will run during.
        """
        return self._queue.empty()

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
    """Minimal compactor interface (tape-native).

    Compactors emit a :class:`ContextOverride` that the runtime appends
    to its tape. The runtime supplies a ``mint_ref`` factory so the
    compactor can mint fresh ``TapeRef`` values without seeing the rest
    of the runtime.
    """

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[HistoryEntry],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        args: str = "",
    ) -> ContextOverride:
        """Produce a barrier override from the current tape/context.

        Args:
          tape: Append-only session tape.
          context: Resolved provider-facing context (resolver's
              ``messages`` view of ``tape``).
          model: Model used for summarization.
          mint_ref: Factory returning fresh ``TapeRef`` values.
          args: Custom compaction instructions.

        Returns:
          override: Barrier ``ContextOverride`` with the summary payload.

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
        session_id: str = "",
    ) -> None:
        self.model = model
        self.system = system
        self.tools_map: dict[str, Tool] = {}
        for t in tools or []:
            if t.name in self.tools_map:
                raise ValueError(f"Duplicate tool name: {t.name!r}")
            self.tools_map[t.name] = t
        self.compactor = compactor
        self.session_id = session_id
        self.tape: list[TapeRecord] = []
        self._next_ordinal: int = 0
        self._cached_resolved: ResolvedContext | None = None
        # ``_tape_by_ref`` enables O(1) ref -> record lookups used by
        # detached-splice and coalesce site conversions. ``_placeholder_refs``
        # / ``_parent_assistant_refs`` cache the call_id -> ref mappings the
        # same sites need.
        self._tape_by_ref: dict[TapeRef, TapeRecord] = {}
        self._placeholder_refs: dict[str, TapeRef] = {}
        self._parent_assistant_refs: dict[str, TapeRef] = {}
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
        # Detached-tool results whose splice failed (no placeholder
        # survived, typically post-``Clear``) AND a fresh cohort is in
        # flight. Held here until ``CohortComplete`` so the user-facing
        # notification doesn't interleave between the cohort's
        # ``tool_use`` and its forthcoming ``tool_result``. Drained at
        # the same site that publishes ``CohortComplete``.
        self._pending_detached_user: list[UserMessage] = []
        # ``AgentIdle`` is edge-triggered: published once when the
        # runtime is about to block on an empty inbox with no work
        # in flight, then suppressed until the next ``drain()`` returns
        # items. Initialized to ``True`` so cold start (drained but
        # never having processed anything) does NOT publish -- the
        # signal is "transitioned from working to idle", not "exists
        # in idle state".
        self._was_idle: bool = True

    def _fully_drained(self) -> bool:
        """True iff the agent has no work to do and no gate is armed.

        Sources of work checked:

        * ``inbox`` non-empty -- items waiting to be drained.
        * ``model_call`` set -- LLM call in flight.
        * ``compact_task`` set -- compaction in progress.
        * ``cohort`` non-empty -- tool batch in progress.
        * ``running_tools`` non-empty -- individual tool tasks live.
        * ``detached`` non-empty -- backgrounded tools whose results
          will land later. Treated as work-in-progress: the agent has
          unfinished business even if it can accept new input.
        * ``_mid_stream_queue`` non-empty -- buffered ``UserMessage``
          received while the model was streaming.
        * ``inbox.gate_armed`` -- the inbox is waiting for a specific
          event type (e.g. ``AWAIT_USER`` after ``Halt`` /
          ``ModelResponseError``). Semantically "parked on a particular
          event," not "idle."

        Local ``run_forever`` state (``awaiting_user``, ``queued``) is
        not consulted directly: ``awaiting_user`` correlates with
        ``inbox.gate_armed``, and ``queued`` correlates with
        ``model_call`` being set (the only case a queued list can
        survive across iterations).

        Snapshot at call time. The caller must read this and act on it
        within the same synchronous block (no intervening ``await``),
        which asyncio's cooperative scheduling guarantees no other
        coroutine will run during.
        """
        return (
            self.inbox.empty()
            and self.model_call is None
            and self.compact_task is None
            and not self.cohort
            and not self.running_tools
            and not self.detached
            and not self._mid_stream_queue
            and not self.inbox.gate_armed
        )

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

    def context(self) -> ResolvedContext:
        """Return the provider-facing context resolved from the tape.

        Memoized against ``len(self.tape)`` so repeated calls within one
        gate iteration walk the tape once.

        Returns:
          resolved: Resolved messages, origins, slot anchors, version,
              and a discontinuity flag (always ``False`` here; send-path
              callers pass their own ``prior`` to :func:`resolve_context`
              when they need it).

        """
        if self._cached_resolved is None or self._cached_resolved.version != len(
            self.tape
        ):
            self._cached_resolved = resolve_context(self.tape)
        return self._cached_resolved

    def append_history(self, entry: HistoryEntry) -> TapeRef:
        """Append a ``HistoryRecord`` for ``entry`` to the tape.

        Args:
          entry: Provider-facing message to record.

        Returns:
          ref: ``TapeRef`` minted for the new record.

        """
        ref = self.mint_ref()
        record = HistoryRecord(ref=ref, entry=entry)
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)
        return ref

    def append_override(
        self,
        *,
        suppresses: tuple[TapeRef, ...] = (),
        inject_after: TapeRef | None = None,
        payload: tuple[HistoryEntry, ...] = (),
        strategy: str = "",
        barrier: bool = False,
        token_before: int = 0,
        token_after: int = 0,
        fallback_reason: str = "",
        preserved_tail_count: int = 0,
        paired_externally: frozenset[str] = frozenset(),
    ) -> TapeRef:
        """Append a ``ContextOverride`` to the tape.

        Args:
          suppresses: Earlier refs hidden when this override is visible.
          inject_after: Anchor ref whose visible record the payload renders
              after; ``None`` injects at the head of the visible slice.
          payload: Provider-facing messages this override injects.
          strategy: Name of the producing strategy.
          barrier: When true, stops the resolver walk.
          token_before: Token count of the suppressed slice (best-effort).
          token_after: Token count of the injected payload (best-effort).
          fallback_reason: Reason the producer fell back to non-summary
              payload, if any.
          preserved_tail_count: Number of tail entries preserved verbatim
              in fallback mode.
          paired_externally: Call ids whose pair lives outside this
              payload (typically a ``HistoryRecord`` or a sibling
              override).

        Returns:
          ref: ``TapeRef`` minted for the new override.

        """
        ref = self.mint_ref()
        record = ContextOverride(
            ref=ref,
            suppresses=suppresses,
            inject_after=inject_after,
            payload=payload,
            strategy=strategy,
            barrier=barrier,
            token_before=token_before,
            token_after=token_after,
            fallback_reason=fallback_reason,
            preserved_tail_count=preserved_tail_count,
            paired_externally=paired_externally,
        )
        self.tape.append(record)
        self._cached_resolved = None
        self._tape_by_ref[ref] = record
        return ref

    def append_clear(self) -> TapeRef:
        """Append a ``ContextClear`` (barrier) to the tape.

        Returns:
          ref: ``TapeRef`` minted for the clear record.

        """
        ref = self.mint_ref()
        record = ContextClear(ref=ref)
        self.tape.append(record)
        self._cached_resolved = None
        self._tape_by_ref[ref] = record
        self._placeholder_refs.clear()
        self._parent_assistant_refs.clear()
        return ref

    def replay_tape(self, records: Sequence[TapeRecord]) -> None:
        """Bulk-append pre-built tape records and advance the ordinal cursor.

        Used by session resume to load persisted tape records while
        preserving their original ``TapeRef`` identities. New appends
        continue from ``max(record.ref.ordinal) + 1``.

        Args:
          records: Records to append in their existing order.

        """
        if not records:
            return
        self.tape.extend(records)
        self._next_ordinal = max(r.ref.ordinal for r in records) + 1
        self._cached_resolved = None
        for record in records:
            self._index_record(record)

    def mint_ref(self) -> TapeRef:
        """Mint the next ``TapeRef`` and advance the ordinal counter.

        Used by compactors that build ``ContextOverride`` instances
        directly. The runtime is the sole appender; minting separately
        lets the compactor stamp records without seeing the runtime.

        Returns:
          ref: Freshly minted ``TapeRef`` tagged with this runtime's
              ``session_id``.

        """
        ref = TapeRef(session_id=self.session_id, ordinal=self._next_ordinal)
        self._next_ordinal += 1
        return ref

    def adopt_record(self, record: TapeRecord) -> None:
        """Append a pre-built tape record (with a runtime-minted ref).

        Used by compactors that build ``ContextOverride`` instances
        directly via ``mint_ref()``. The runtime updates its side
        tables (cache invalidation, ref index, call_id anchors) and
        appends the record to the tape.

        Args:
          record: Pre-built tape record whose ``ref`` was minted via
              :meth:`mint_ref` on this runtime.

        """
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)

    def _index_record(self, record: TapeRecord) -> None:
        """Cache ref->record and call_id->anchor mappings after append."""
        self._tape_by_ref[record.ref] = record
        if isinstance(record, HistoryRecord):
            entry = record.entry
            if isinstance(entry, AssistantMessage):
                for tc in entry.tool_calls:
                    self._parent_assistant_refs[tc.id] = record.ref
            elif isinstance(entry, ToolResult):
                self._placeholder_refs[entry.call_id] = record.ref

    def _splice_detached_result(
        self,
        call_id: str,
        content: str,
        is_error: bool,
    ) -> ToolResult | None:
        """Replace a placeholder ``ToolResult`` with the real detached result.

        Appends an override that suppresses the placeholder and injects
        the real ``ToolResult`` at the parent assistant's anchor, keeping
        provider-valid tool_use/tool_result pairing.

        Args:
          call_id: Tool-call id whose placeholder should be replaced.
          content: Real result text.
          is_error: True when the underlying tool signalled failure.

        Returns:
          spliced: The injected ``ToolResult`` instance, or ``None`` when
              no placeholder survives in the resolved context (e.g. a
              compaction barrier hid it before the result arrived).

        """
        placeholder_ref = self._placeholder_refs.get(call_id)
        parent_ref = self._parent_assistant_refs.get(call_id)
        if placeholder_ref is None or parent_ref is None:
            return None
        placeholder_record = self._tape_by_ref.get(placeholder_ref)
        if not isinstance(placeholder_record, HistoryRecord):
            return None
        prior = placeholder_record.entry
        if not isinstance(prior, ToolResult):
            return None
        real = dataclasses.replace(prior, content=content, is_error=is_error)
        self.append_override(
            suppresses=(placeholder_ref,),
            inject_after=parent_ref,
            payload=(real,),
            strategy="detached_splice",
            paired_externally=frozenset({call_id}),
        )
        # Keep the call_id -> placeholder mapping pointing at the original
        # record; a second splice for the same call_id should still find
        # and suppress the same placeholder. The override's ref is not
        # an anchor candidate, so we don't reindex it.
        return real

    async def run_forever(self) -> None:
        """Drain inbox, dispatch, repeat. The entire engine."""
        awaiting_user = False
        queued: list[UserQueuedMessage] = []

        while True:
            try:
                self._collect_detached()
                # AgentIdle publish: edge-triggered, fires when we are
                # about to block on an empty inbox with no work in
                # flight. Must be entirely synchronous up to the next
                # ``await`` so no other coroutine can mutate the work
                # sources between the predicate check and the publish.
                # ``_was_idle`` resets to ``False`` after ``drain()``
                # returns items (i.e. we just consumed work).
                if self._fully_drained() and not self._was_idle:
                    self.publish(AgentIdle())
                    self._was_idle = True
                items = await self.inbox.drain()
                self._was_idle = False

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
                            # Cancel and forget every detached task. Clear
                            # is a hard reset; without this, surviving
                            # background tasks eventually fire
                            # ``DetachedResult`` into the wiped session
                            # (splice fails, fallback appends an orphan
                            # ``UserMessage``). After ``self.detached``
                            # is empty the cancelled tasks' completions
                            # match neither cohort nor detached
                            # membership at ``_run_tool_and_post`` and
                            # are silently dropped at the default
                            # inbox case.
                            for task in self.detached.values():
                                task.cancel()
                            self.detached.clear()
                            self._pending_detached_user.clear()
                            queued.clear()
                            self._mid_stream_queue.clear()
                            self.append_clear()
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

                        case CompactComplete():
                            # The compaction task already appended its
                            # override(s) to the tape; this handler just
                            # clears bookkeeping and republishes.
                            self.compact_task = None
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
                            self.append_history(msg)
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
                                    self.append_history(
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
                            self.append_history(item)
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
                            self._splice_detached_result(
                                cid,
                                item.content,
                                item.is_error,
                            )
                            self.publish(item)

                        case DetachedResult():
                            self.cohort.discard(item.call_id)
                            # Splice the real result into the placeholder's
                            # slot via a tape override so the model sees
                            # the real result where it already expects one.
                            # ``[detached]`` (preempt) and ``[Running in
                            # background: ...]`` (explicit-bg) placeholders
                            # both match by ``call_id``.
                            spliced_entry = self._splice_detached_result(
                                item.call_id,
                                item.content,
                                item.is_error,
                            )
                            spliced = spliced_entry is not None
                            if not spliced:
                                # No placeholder to splice into (rare:
                                # ``Clear`` / ``Compact`` wiped it before
                                # the detached task finished). Surface
                                # the content as a ``UserMessage`` so
                                # it isn't silently dropped. If a fresh
                                # cohort is in flight, defer to
                                # ``CohortComplete`` -- appending now
                                # would interleave between the cohort's
                                # ``tool_use`` and its forthcoming
                                # ``tool_result``.
                                fallback = UserMessage(
                                    text=(
                                        f"[Tool {item.call_id} completed]\n"
                                        f"{item.content}"
                                    ),
                                )
                                if self.cohort:
                                    self._pending_detached_user.append(fallback)
                                else:
                                    self._append_or_coalesce_user(fallback)
                            elif (
                                isinstance(
                                    self.context().messages[-1], AssistantMessage
                                )
                                and not self.cohort
                            ):
                                # Splice landed after the preempted round
                                # already idled with text. History tail
                                # is an ``AssistantMessage`` with no
                                # pending cohort, so the end-of-loop
                                # gate won't fire on its own. Append a
                                # terse notification (real content lives
                                # in its proper slot above) to wake the
                                # model. The ``not self.cohort`` guard
                                # is essential: with an in-flight cohort
                                # the tail assistant carries an
                                # unanswered ``tool_use`` and appending
                                # here interleaves a ``UserMessage``
                                # between ``tool_use`` and its
                                # forthcoming ``tool_result``
                                # (Anthropic rejects with HTTP 400).
                                # ``CohortComplete`` will wake the model
                                # with the spliced content already
                                # visible in its proper slot.
                                self.append_history(
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
                    # Flush any ``DetachedResult`` fallbacks that landed
                    # mid-cohort. Tail is now a ``ToolResult`` (the
                    # cohort just settled), so appending here is safe.
                    # ``_append_or_coalesce_user`` collapses consecutive
                    # entries when the model gate fires next.
                    for pending_user in self._pending_detached_user:
                        self._append_or_coalesce_user(pending_user)
                        self.publish(pending_user)
                    self._pending_detached_user.clear()

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
                        len(self.context().messages),
                        len(self.detached),
                        len(queued),
                    )
                    # ``inbox.gate_armed`` blocks firing while ``AWAIT_USER``
                    # is pending (armed by ``Halt`` / ``ModelResponseError``).
                    # Without this guard the model would fire on the stale
                    # ``UserMessage`` still at history.tail, treating the
                    # cancellation as a retry rather than waiting for the
                    # user's next input.
                    self._assert_alternation_invariant()
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
        tape_cursor = len(self.tape)

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
        return [
            r.entry for r in self.tape[tape_cursor:] if isinstance(r, HistoryRecord)
        ]

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
        messages = self.context().messages
        if not messages:
            return False
        return isinstance(messages[-1], (UserMessage, ToolResult))

    def _assert_alternation_invariant(self) -> None:
        """Repair HR-level orphans, validate; rescue if still broken.

        The model-call gate's last guard before firing the provider:

        1. Repair HistoryRecord-level orphans
           (:meth:`_repair_history_record_orphans`): pair unmatched
           ``tool_use`` ids with synthetic ``[interrupted]``
           ``ToolResult`` records; suppress orphan ``ToolResult`` from
           a ``HistoryRecord`` origin.
        2. If validation still fails -- typically a legacy session
           reconstructed via :meth:`ContextOverride.replay` with
           invalid payloads predating the construct-time invariant --
           rescue: append a single barrier override that suppresses
           every visible tape ref and re-injects a fully sanitized
           payload. Structural attribution is lost for the rescued
           section but the session stays live.

        Forward producers can no longer emit overrides with invalid
        payloads (the validator in
        :meth:`ContextOverride.__post_init__` rejects them at
        construct), so phase 1 of the prior implementation has been
        deleted.

        Raises:
          InvalidContextError: When even the rescue path can't produce
              a wire-format-valid context (should not happen in
              practice; rescue's sanitizer is total).

        """
        self._repair_history_record_orphans()
        try:
            validate_context(self.context().messages)
            return
        except InvalidContextError as exc:
            logger.warning(
                "context invariant still violated after HR-level repair; "
                "applying rescue barrier: %s",
                exc,
            )
        self._rescue_context()
        validate_context(self.context().messages)

    def _rescue_context(self) -> None:
        """Append a barrier override carrying a fully sanitized payload.

        Last-resort repair, primarily for legacy sessions whose
        :meth:`ContextOverride.replay` reconstructions carry payloads
        the construct-time invariant would reject. Suppresses every
        currently-visible tape ref and re-injects the result of
        :func:`_sanitize_for_send` over the current resolved messages.
        Structural attribution is lost for the rescued section but the
        resolved view is guaranteed wire-format-valid by construction.

        The synthesized override declares every preserved ``call_id``
        as ``paired_externally`` so the payload validator accepts it
        even when the sanitized sequence contains tool_use/tool_result
        across positions the local in-payload check would flag.
        """
        resolved = self.context()
        sanitized = _sanitize_for_send(resolved.messages)
        suppresses = tuple(set(resolved.origins))
        paired_externally = frozenset(
            m.call_id for m in sanitized if isinstance(m, ToolResult)
        )
        self.append_override(
            suppresses=suppresses,
            inject_after=None,
            payload=sanitized,
            strategy="context_rescue",
            barrier=True,
            paired_externally=paired_externally,
        )

    def _repair_history_record_orphans(self) -> None:
        """Pair / drop HistoryRecord-origin orphans in the resolved context.

        For each ``AssistantMessage`` with unpaired ``tool_calls`` from
        a ``HistoryRecord`` origin, append an override injecting synth
        ``[interrupted]`` ``ToolResult`` records in its slot suffix.
        For each orphan ``ToolResult`` from a ``HistoryRecord`` origin,
        append a suppression override.

        Overrides from forward producers carry ``paired_externally``
        declarations covering their own payload pairing, so anything
        the resolver flags as orphan after this method runs originates
        either in a legacy ``replay()`` payload (handled by rescue) or
        in a producer bug (which the construct-time validator should
        have caught). Idempotent: a second call on a repaired context
        is a no-op.
        """
        resolved = self.context()
        messages = resolved.messages
        origins = resolved.origins
        unmatched_per_am: list[tuple[TapeRef | None, list[str]]] = []
        pending: dict[str, int] = {}
        am_repair_anchor: TapeRef | None = None
        seen_results: set[str] = set()
        hr_orphan_refs: list[TapeRef] = []

        def _flush_pending() -> None:
            if pending:
                unmatched_per_am.append((am_repair_anchor, list(pending)))
            pending.clear()

        for idx, entry in enumerate(messages):
            origin = origins[idx]
            origin_record = self._tape_by_ref.get(origin)
            origin_is_hr = isinstance(origin_record, HistoryRecord)
            if isinstance(entry, AssistantMessage):
                _flush_pending()
                am_repair_anchor = origin if origin_is_hr else None
                pending.update({tc.id: idx for tc in entry.tool_calls})
            elif isinstance(entry, ToolResult):
                if entry.call_id in seen_results or entry.call_id not in pending:
                    if origin_is_hr:
                        hr_orphan_refs.append(origin)
                    # Non-HR orphans get cleared by the rescue path.
                    continue
                del pending[entry.call_id]
                seen_results.add(entry.call_id)
            else:
                _flush_pending()
                am_repair_anchor = None
        _flush_pending()

        for anchor, missing_ids in unmatched_per_am:
            if anchor is None:
                # Origin is an override; producer was supposed to
                # declare ``paired_externally``. Rescue handles it.
                continue
            payload = tuple(
                ToolResult(
                    call_id=cid,
                    content="[interrupted]",
                    is_error=True,
                )
                for cid in missing_ids
            )
            # The synth TRs have no matching AM in their payload; the
            # matching AMs live in the ``HistoryRecord`` at ``anchor``.
            self.append_override(
                suppresses=(),
                inject_after=anchor,
                payload=payload,
                strategy="orphan_tool_use_repair",
                paired_externally=frozenset(missing_ids),
            )
        if hr_orphan_refs:
            self.append_override(
                suppresses=tuple(hr_orphan_refs),
                inject_after=None,
                payload=(),
                strategy="orphan_tool_result_repair",
            )

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
        resolved = self.context()
        messages = resolved.messages
        if not messages or not isinstance(messages[-1], UserMessage):
            self.append_history(item)
            return
        tail = messages[-1]
        combined = dataclasses.replace(
            tail,
            text=f"{tail.text}\n\n{item.text}",
            attachments=tail.attachments + item.attachments,
        )
        tail_origin = resolved.origins[-1]
        anchor = resolved.slot_anchors[-1]
        prior_record = self._tape_by_ref.get(tail_origin)
        if isinstance(prior_record, ContextOverride):
            # Stacking a coalesce on a prior coalesce: subsume the prior
            # override's suppression set so previously-hidden records do
            # not re-emerge once the prior override is itself hidden.
            suppresses: tuple[TapeRef, ...] = (tail_origin, *prior_record.suppresses)
        else:
            suppresses = (tail_origin,)
        self.append_override(
            suppresses=suppresses,
            inject_after=anchor,
            payload=(combined,),
            strategy="user_coalesce",
        )

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
        self.append_history(
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
              The tape is mutated via ``_append_or_coalesce_user``.

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
                self.context().messages,
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
        """Run compaction; the compactor's overrides land in the tape.

        The compactor returns one :class:`ContextOverride` whose ref
        was minted via the ``mint_ref`` factory the runtime provided.
        The runtime is the sole appender: it stores the returned
        override on the tape and publishes ``CompactComplete``.
        """
        if self.compactor is None:
            return
        tape_len = len(self.tape)
        try:
            override = await self.compactor.compact(
                self.tape,
                self.context().messages,
                self.model,
                self.mint_ref,
                args,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 -- compaction calls the model; catch-all routes UserFacingError to warning, others to exception
            log_exception_or_warning(logger, "compaction failed", exc)
            self.inbox.push_back(
                CompactFailed(exception=exc, tape_len=tape_len),
            )
            return
        self.adopt_record(override)
        self.inbox.push_back(
            CompactComplete(
                records=(override,),
                fallback_reason=override.fallback_reason,
                preserved_tail_count=override.preserved_tail_count,
            ),
        )
