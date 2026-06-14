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

``list[UserMessage | AgentSendMessage | AssistantMessage | ToolResult]``
(the ``ModelContextEvent`` union). Match on type, not string descriptors.

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
one of several trigger points, trading responsiveness against
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
- **Control/error drains.** ``Halt``, ``ModelResponseError``, and
  ``Compact`` also drain runtime-owned ``_mid_stream_queue`` so
  already-submitted non-REPL messages are not stranded behind control
  flow. REPL-local urgent/deferred queues are handled by the REPL
  observer/keybindings before they enter runtime history.

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
concentrates the inbox-arm dispatch logic in one ``match`` block.
Tape mutations and event publications that happen on the model and
tool worker tasks (``_compact_and_post``'s ``adopt_record``,
``_stream_and_post``'s ``ModelResponseCancelled`` publish,
``_run_tool_and_post``'s result routing) intentionally live outside
that ``match`` block; the inbox arm dispatches and the worker arms
drive the next events back through the inbox.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, Protocol

import asyncio
import contextlib
import contextvars
import dataclasses
import logging

# Engine consumes a subset of the runtime event vocabulary. No re-export shim.
from sagent.agent.context import (
    InvalidContextError,
    ResolvedContext,
    alive_splices,
    resolve_context,
    validate_context,
)
from sagent.types.exceptions import (
    log_exception_or_warning,
    log_task_exception,
)
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DETACHED_ARRIVAL_SUFFIX,
    DETACHED_ARRIVED_MIMIC_PREFIX,
    DETACHED_ARRIVED_TOOL,
    DETACHED_PLACEHOLDER,
    AgentIdle,
    AgentSendDeferredMessage,
    AgentSendMessage,
    AgentSendQueuedMessage,
    AssistantMessage,
    Clear,
    ClearComplete,
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
    LazyEvent,
    ModelCallStarted,
    ModelContextEvent,
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
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultPartial,
    Undetach,
    UserDeferredMessage,
    UserMessage,
    UserQueuedMessage,
    wire_role,
)
from sagent.types.tape import (
    ContextSplice,
    InvalidPayloadError,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
    full_tape_mask,
    mask_contains_ref,
    mask_ranges_overlap,
    merge_mask_ranges,
    splice_safe_repair,
    unpaired_call_ids,
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


# ``cli_publish_var`` is set by ``_AgentModel.stream`` before invoking
# a CLI provider whose tool loop happens inside its subprocess. The
# MCP bridge (``providers.lib.mcp_bridge``) reads this var when it
# receives a ``call_tool`` request and synthesises a ``ToolLabel``
# event so the REPL renderer surfaces CLI tool calls the same way it
# does API-driven ones. API providers don't read the var; the bridge
# only exists for CLI transports. Default ``None`` makes the lookup
# safe in non-CLI code paths.
cli_publish_var: contextvars.ContextVar[Callable[[RuntimeEvent], None] | None] = (
    contextvars.ContextVar("cli_publish", default=None)
)

logger = logging.getLogger(__name__)


def widen_barrier_mask(
    override: ContextSplice,
    tape: Sequence[TapeRecord],
) -> ContextSplice:
    """Rewrite a compaction barrier mask to cover current tape refs.

    The widening preserves any ordinal gaps between the producer's
    original mask ranges, per ``session_id``: a sparse producer mask of
    the form ``((r5, r5), (r10, r10))`` stays sparse over ``r6..r9``
    even though tape refs would otherwise be absorbed. Each range
    contributed by the widening stays within one ``session_id`` so the
    result satisfies ``_validate_mask_disjoint`` for legacy resumed
    tapes that span multiple session id namespaces.

    Args:
      override: Compactor-produced barrier splice. Today's only
          producer (compaction barriers via ``full_tape_mask``) emits a
          mask that is contiguous per ``session_id``; sparse masks are
          still accepted and their gaps preserved -- see
          ``test_widen_barrier_mask_preserves_mask_gaps``.
      tape: Runtime tape at adopt time.

    Returns:
      widened: ``override`` with its mask widened to current tape refs
      while preserving gaps between the producer's original mask
      ranges, per ``session_id``.

    """
    # Mask ranges are single-session by construction (``MaskRange``), so the
    # per-session widening below cannot mis-attribute a cross-session range --
    # Issue#313 deleted the runtime guard that used to assert this here.
    mask = _widen_mask_ranges(override.mask, tape)
    if mask == override.mask:
        return override
    return dataclasses.replace(override, mask=mask)


def _widen_mask_ranges(
    mask: tuple[MaskRange, ...],
    tape: Sequence[TapeRecord],
) -> tuple[MaskRange, ...]:
    """Return ``mask`` widened over ``tape`` without filling existing gaps.

    Tape refs are partitioned by ``session_id`` before widening; each emitted
    :class:`MaskRange` is single-session by construction.
    """
    if not tape:
        return mask
    ordinals_by_session: dict[str, list[int]] = {}
    for record in tape:
        ordinals_by_session.setdefault(record.ref.session_id, []).append(
            record.ref.ordinal
        )
    preserved_gaps_by_session = _preserved_mask_gaps_by_session(mask)
    widened: list[MaskRange] = []
    for sid, ordinals in ordinals_by_session.items():
        ordinals.sort()
        preserved_gaps = preserved_gaps_by_session.get(sid, ())
        start: int | None = None
        previous: int | None = None
        for ordinal in ordinals:
            if any(lo < ordinal < hi for lo, hi in preserved_gaps):
                if start is not None and previous is not None:
                    widened.append(MaskRange(session_id=sid, lo=start, hi=previous))
                    start = None
                continue
            if start is None:
                start = ordinal
            previous = ordinal
        if start is not None and previous is not None:
            widened.append(MaskRange(session_id=sid, lo=start, hi=previous))
    return tuple(widened)


def _preserved_mask_gaps_by_session(
    mask: tuple[MaskRange, ...],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return ordinal gaps between sorted mask ranges, per session_id."""
    per_session: dict[str, list[MaskRange]] = {}
    for r in mask:
        per_session.setdefault(r.session_id, []).append(r)
    return {
        sid: tuple(
            (left.hi, right.lo)
            for left, right in pairwise(sorted(ranges, key=lambda r: r.lo))
            if left.hi + 1 < right.lo
        )
        for sid, ranges in per_session.items()
    }


def _sanitize_for_send(
    entries: Sequence[ModelContextEvent],
) -> tuple[ModelContextEvent, ...]:
    """Return a wire-format-valid version of ``entries`` for the rescue path.

    Used by the runtime gate when structural repair can't pair
    tool_use / tool_result across overrides (e.g. accumulated microcompact
    debt from a session predating the cache-warm gate fix).

    A thin wrapper over the canonical :func:`tape.splice_safe_repair`, so the
    rescue path shares one pair/dedup/synthesize/coalesce policy with the
    compaction repair and cannot drift from it (the H2/F1 disease). The repair
    detail lives at that canonical site. Idempotent.
    """
    return splice_safe_repair(entries)


def _mimic_index_of(cid: str) -> int | None:
    """Return the ``N`` of a ``DetachedArrived:mimic:N`` id, or ``None``."""
    if not cid.startswith(DETACHED_ARRIVED_MIMIC_PREFIX):
        return None
    suffix = cid.removeprefix(DETACHED_ARRIVED_MIMIC_PREFIX)
    return int(suffix) if suffix.isdigit() else None


def _mimic_indices(entry: object) -> list[int]:
    """Return numeric indices of any ``DetachedArrived:mimic:N`` id in ``entry``.

    Covers both sides of the mimic id namespace: an ``AssistantMessage``
    contributes its ``tool_calls`` ids, a ``ToolResult`` its ``call_id``.
    Scanning only one side would under-seed the counter when the tape
    preserves a lone partner -- e.g. ``_commit_pairing`` splices a mimic
    ``ToolResult`` whose parent ``AssistantMessage`` was compacted away.
    """
    if isinstance(entry, AssistantMessage):
        ids = [tc.id for tc in entry.tool_calls]
    elif isinstance(entry, ToolResult):
        ids = [entry.call_id]
    else:
        return []
    return [n for cid in ids if (n := _mimic_index_of(cid)) is not None]


def _max_mimic_index(records: Sequence[TapeRecord]) -> int:
    """Return the largest ``DetachedArrived:mimic:N`` index in ``records``.

    Scans both ``ReferrableTapeEvent`` events and ``ContextSplice`` payloads,
    on both sides of the id namespace (see :func:`_mimic_indices`). Masked
    (dead) splice payloads are scanned too: undelete can resurrect them, so
    their ids must still reserve namespace. A splice's ``paired_externally``
    ids are also scanned: ``ContextSplice.replay`` (legacy on-disk load)
    bypasses ``_validate_payload``, so a persisted record can declare a mimic
    id whose local pair is absent from ``payload`` -- reading only ``payload``
    would then under-seed. Returns ``-1`` when no mimic id is present, so the
    caller can seed its counter to ``result + 1`` and start at ``0`` on a clean
    tape.

    Args:
      records: Loaded tape records to scan.

    Returns:
      max_index: Largest mimic index found, or ``-1`` when none exist.

    """
    found = [-1]
    for record in records:
        if isinstance(record, ReferrableTapeEvent):
            found.extend(_mimic_indices(record.event))
        else:
            for entry in record.payload:
                found.extend(_mimic_indices(entry))
            found.extend(
                n
                for cid in record.paired_externally
                if (n := _mimic_index_of(cid)) is not None
            )
    return max(found)


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


@dataclass(frozen=True, slots=True, kw_only=True)
class _PendingCommit:
    """One deferred tape commit, flushed at the next gate (see ``_flush_pending``).

    The single mechanism behind every "hold this, commit it on the next real
    turn" need -- a mimicked-tool error pairing, a completed detached tool's
    forward delivery, and ride-along system context all share it. ``kind``
    selects the placement:

    - ``"pairing"``: ``result`` answers an already-emitted ``tool_use`` whose
      result was deferred (a mimicked ``DetachedArrived``). Committed adjacent
      to its parent ``AssistantMessage`` so it pairs in-slot, never stranded.
    - ``"forward"``: ``result`` is a completed detached tool's real output,
      delivered as a synthetic ``DetachedArrived`` pair (the original stub
      stays). A user-role separator precedes it after an assistant tail.
    - ``"ride_along"``: ``user`` is system-injected context (e.g. a reminder)
      coalesced onto the user side.
    """

    kind: Literal["pairing", "forward", "ride_along"]
    result: ToolResult | None = None
    user: UserMessage | None = None


AWAIT_USER = Await(
    (
        UserMessage,
        UserQueuedMessage,
        UserDeferredMessage,
        AgentSendMessage,
        AgentSendQueuedMessage,
        AgentSendDeferredMessage,
        Quit,
    )
)


# Recovery gate for an unrepairable model-call context: like ``AWAIT_USER`` but
# also admits the tape-mutating control verbs. A broken tape is most directly
# fixed by ``Clear`` / ``Compact`` / ``Recompact``, and ``Halt`` / ``Kill``
# must still reach the loop; ``AWAIT_USER`` alone buffers all of these behind
# the gate, stranding the very recovery actions the user reaches for.
AWAIT_RECOVERY = Await((*AWAIT_USER.types, Clear, Compact, Recompact, Halt, Kill))


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

    def drain_nowait(self) -> list[T]:
        """Return all currently queued items without waiting."""
        items: list[T] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def push_front(self, *items: T | Await) -> None:
        """Add to front of queue. Await sets the drain gate.

        Args:
          items: Items to push at the head, in argument order. An
              ``Await`` arms the gate; non-``Await`` items are queued.
              **Precondition**: when an ``Await`` is present it must be
              the first argument. The gate baseline (count of items that
              already satisfy the gate) snapshots from the pre-existing
              queue only -- items queued before the ``Await`` in this same
              call would otherwise go uncounted, so the gate would release
              on the first NEW gate-type item even though one already
              arrived in the same ``push_front`` call.

        """
        await_idx = next(
            (i for i, item in enumerate(items) if isinstance(item, Await)), -1
        )
        assert await_idx <= 0, (
            f"GatedDeque.push_front precondition violated: Await must be the "
            f"first argument when present; got Await at index {await_idx} with "
            f"{await_idx} non-Await item(s) ahead of it"
        )
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


def _message_from_queued(
    item: UserQueuedMessage
    | UserDeferredMessage
    | AgentSendQueuedMessage
    | AgentSendDeferredMessage,
) -> UserMessage | AgentSendMessage:
    if isinstance(item, (AgentSendQueuedMessage, AgentSendDeferredMessage)):
        return AgentSendMessage(
            source=item.source,
            text=item.text,
            attachments=item.attachments,
        )
    return UserMessage(text=item.text, attachments=item.attachments)


def _coalesce_user_side(
    items: Sequence[UserMessage | AgentSendMessage],
) -> UserMessage | AgentSendMessage:
    first = items[0]
    text = "\n\n".join(item.text for item in items)
    attachments = sum((item.attachments for item in items), ())
    if isinstance(first, AgentSendMessage):
        return AgentSendMessage(
            source=first.source,
            text=text,
            attachments=attachments,
        )
    return UserMessage(text=text, attachments=attachments)


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

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Return a serialization key, or ``None`` for unrestricted parallelism.

        Within one cohort the runtime groups calls sharing a non-``None``
        key and runs them in a single coroutine, sequentially, in
        submission order. This is how same-file Read/Edit/Write avoid
        racing each other's read-modify-write: they return the resolved
        file path here. Tools that never contend for a shared resource
        return ``None`` to run fully in parallel.

        Args:
          args: Parsed directive arguments for this call.

        Returns:
          key: Stable string shared by calls that must serialize, or
              ``None`` to run fully in parallel.

        """
        ...


class Model(Protocol):
    """Minimal model interface for the runtime.

    The runtime hands the model only ``history`` and the streaming
    callbacks. ``system`` and ``tools`` were historical args every
    production wrapper either discarded or recomputed against its own
    state, so the runtime no longer forwards them; consumers that need
    a system prompt or tool list must capture them at construction.
    """

    async def stream(
        self,
        history: list[ModelContextEvent],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        """Stream a model response.

        Args:
          history: Conversation history.
          on_text: Callback for each streamed text chunk.
          on_thinking: Callback for each streamed thinking chunk.

        Returns:
          message: Complete assistant response.

        """
        ...


class Compactor(Protocol):
    """Minimal compactor interface (tape-native).

    Compactors emit a :class:`ContextSplice` that the runtime appends
    to its tape. The runtime supplies a ``mint_ref`` factory so the
    compactor can mint fresh ``TapeRef`` values without seeing the rest
    of the runtime.

    The Protocol is deliberately lean: the runtime only knows about
    ``compact`` (driven by ``Compact`` / ``Recompact`` events). The
    Agent layer's :class:`Compactor` is richer (``should_compact`` for
    the proactive gate, ``maintain`` for periodic upkeep) and reaches
    its inner compactor via a bridge rather than the runtime Protocol.
    """

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Produce a barrier splice from the current tape/context.

        Producers whose payload preserves an ``AssistantMessage`` with
        ``tool_calls`` whose matching ``ToolResult`` lives outside the
        payload (e.g. fallback mode keeping the parent assistant so a
        still-running tool can splice in later) must declare those ids
        in :attr:`ContextSplice.paired_externally`. Same in reverse for
        payloads carrying a ``ToolResult`` whose matching AM is
        external. ``ContextSplice.__post_init__`` enforces both
        directions strictly; declaring an id ``paired_externally``
        while *both* sides also appear locally is rejected as misuse.
        :func:`sagent.types.tape.unpaired_call_ids`
        computes the field honestly from a finished payload.

        Args:
          tape: Append-only session tape.
          context: Resolved provider-facing context (resolver's
              ``messages`` view of ``tape``).
          model: Model used for summarization.
          mint_ref: Factory returning fresh ``TapeRef`` values.
          custom_instructions: Optional free-form guidance from the
              caller (e.g. user-typed hint after ``/compact <text>``).

        Returns:
          splice: Barrier ``ContextSplice`` with the summary payload.

        """
        ...


class AgentRuntime:
    """Inbox-driven event loop agent.

    Args:
      model: Model implementation used for ``stream``.
      tools: Tools registered for dispatch; must have unique ``name``.
      compactor: Optional compactor invoked on ``Compact`` / ``Recompact``.

    """

    def __init__(
        self,
        *,
        model: Model,
        tools: list[Tool] | None = None,
        compactor: Compactor | None = None,
        session_id: str = "",
        preempt_in_flight: bool = False,
        coalesce_inbox: bool = True,
    ) -> None:
        self.model = model
        # When True, mid-stream UserMessage / AgentSendMessage attempts a
        # provider-side cancel (``model.cancel_in_flight()``) before
        # buffering. Only useful with providers that drive their own
        # tool loop opaquely (e.g. AnthropicCLI / GoogleCLI), where
        # ``_stop_all_tools`` has no cohort entries to act on. Default
        # off: changes user-observable timing (the in-flight model
        # response is truncated to a ModelResponseError) and only the
        # caller knows whether that tradeoff is desired for this agent.
        self._preempt_in_flight = preempt_in_flight
        # When True (default — the sagent-design behaviour), consecutive
        # same-source ``UserMessage`` / ``AgentSendMessage`` items
        # arriving without an intervening assistant turn are coalesced
        # into a single history entry (see :meth:`_append_or_coalesce_user`).
        # This satisfies Anthropic's user/assistant alternation rule when
        # the model errored or was cancelled mid-turn.
        #
        # Set to False for chat-channel use cases where each peer
        # ``AgentSendMessage`` is a deliberate discrete event that the
        # recipient should process as a separate turn (e.g. a hard STOP
        # arriving after a prior delegation message must be visible AS a
        # distinct inbound, not concatenated to the tail of the prior
        # one). When False, the coalesce path is replaced by an explicit
        # synthetic assistant-turn boundary so the API alternation rule
        # still holds.
        self._coalesce_inbox = coalesce_inbox
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
        # ``_parent_assistant_refs[call_id]`` maps a tool call to the ref of
        # the ``AssistantMessage`` that requested it; ``_parent_id_for_call``
        # reads it so a detached tool's ``[detached]`` stub is stamped with
        # the right ``parent_id``. ``_placeholder_refs[call_id]`` tracks the
        # ref carrying that call's ``ToolResult``. ``_index_record`` keeps both
        # current; barriers evict masked entries via
        # ``_invalidate_masked_anchors`` / ``_clear_detached_anchors``.
        self._placeholder_refs: dict[str, TapeRef] = {}
        self._parent_assistant_refs: dict[str, TapeRef] = {}
        self.inbox: GatedDeque[RuntimeEvent] = GatedDeque()
        # Task[None]: tools post results to inbox, not via return value.
        self.detached: dict[str, asyncio.Task[None]] = {}
        self.observers: list[Callable[[RuntimeEvent], None]] = []
        self.before_tool_spawn: (
            Callable[[AssistantMessage], RuntimeEvent | None] | None
        ) = None
        # Reports whether the Agent layer has live ``background: true`` /
        # ``delay`` tool jobs (its ``_bg`` registry, invisible to the
        # runtime). Consulted by ``_fully_drained`` so ``AgentIdle`` does not
        # fire -- and a one-shot ``Agent.run`` does not reap -- while a
        # backgrounded tool is still producing a result to deliver forward.
        #
        # This is a synchronous *query* into the Agent layer, not an event.
        # The "publish an event instead" rule (module header) governs
        # extension/notification -- things that happened, fanned out to
        # observers. ``_fully_drained`` needs a boolean answer inline at
        # predicate-evaluation time; an async one-directional event cannot
        # supply that. Same shape and rationale as ``before_tool_spawn``.
        self.has_pending_background: Callable[[], bool] | None = None
        # Lifted from run_forever locals for observer/REPL visibility.
        # Task[None]: tools/model post results to inbox, not via return.
        self.running_tools: dict[str, asyncio.Task[None]] = {}
        self.cohort: set[str] = set()
        # ``_cohort_seen`` tracks "we started a cohort and want to publish
        # ``CohortComplete`` when it naturally drains." Reset by every
        # stop-cohort path so a preempted cohort doesn't fire complete.
        self._cohort_seen: bool = False
        self.model_call: asyncio.Task[None] | None = None
        self._model_call_generation: int = 0
        self.compact_task: asyncio.Task[None] | None = None
        # Bumped by every Halt/Clear that publishes CompactFailed for an
        # in-flight compaction, and at compact spawn. ``_compact_and_post``
        # captures its generation at spawn time and refuses to push the
        # terminal CompactComplete or CompactFailed when its generation
        # is no longer current -- preventing double terminal events when
        # a Halt cancels a compactor that is between its model await and
        # the synchronous CompactComplete push.
        self._compact_generation: int = 0
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
        self._mid_stream_queue: list[UserMessage | AgentSendMessage] = []
        # The single deferred-commit queue: tape entries held until the next
        # real turn, then committed (in-slot) by ``_flush_pending``. Covers
        # forward detached-tool deliveries, mimicked-tool error pairings, and
        # ride-along system context -- one mechanism, one flush site, so no
        # per-feature special case can re-derive commit timing/placement
        # (which is how the earlier split caused stranding / lone-round bugs).
        self._pending_commits: list[_PendingCommit] = []
        # Original ``call_id``s whose detached result has already been
        # forward-delivered (or is queued to be). The forward-delivery
        # invariant is "at most one ``DetachedArrived`` pair per original
        # ``call_id``, ever": two deliveries for one id would resolve to two
        # ``AssistantMessage``s sharing the ``f"{id}:detached"`` tool_call id,
        # which ``validate_context`` rejects and ``_rescue_context`` cannot
        # repair (wedging the gate). ``_defer_detached_forward`` consults this
        # set so any second producer -- the in-batch ``ToolResult`` race and a
        # later ``DetachedResult``, a re-spawned task, any future path -- is a
        # no-op rather than a duplicate. Never cleared except by ``Clear``
        # (which wipes history, so the ids no longer resolve).
        self._forwarded_call_ids: set[str] = set()
        # Monotonic counter minting unique ids for model-forged
        # ``DetachedArrived`` calls (``_sanitize_forged_arrivals``). A forged
        # call's model-chosen id can collide with a real arrival id; rewriting
        # into the ``DETACHED_ARRIVED_MIMIC_PREFIX`` namespace keyed off this
        # counter keeps every forgery's id globally unique.
        self._mimic_counter: int = 0
        # ``AgentIdle`` is edge-triggered: published once when the
        # runtime is about to block on an empty inbox with no work
        # in flight, then suppressed until the next ``drain()`` returns
        # items. Initialized to ``False`` so a cold-start REPL where
        # the user pre-staged a deferred block before any work ever
        # ran sees an initial ``AgentIdle`` and flushes the queue.
        # The signal is "agent is now idle and accepting input"; that
        # is true at the very first ``run_forever`` iteration too.
        self._was_idle: bool = False
        # Wall-clock seconds at which a prior ``ModelServiceSuspended``
        # scheduled the next retry. Read once by the model loop before
        # the first ``send_with_retry`` and cleared; subsequent sends
        # use live backoff. Populated by ``Agent.resume`` from the
        # latest persisted suspension event.
        self.resume_retry_at: float | None = None
        self.service_suspended_until: float | None = None

    @property
    def _engine_quiescent(self) -> bool:
        """True when no streaming, compaction, or open tool batch is in flight.

        The one place the "engine has nothing actively running that a gate
        must wait behind" cluster is defined. Every per-iteration gate that
        used to spell out ``not self.cohort and self.model_call is None and
        self.compact_task is None`` reads this instead, so a future term can
        never be added at one gate and forgotten at another -- the scattered
        re-derivation that produced the deferral and wedge seam bugs.

        Deliberately excludes ``inbox.gate_armed`` (a gate that must also
        respect an armed ``AWAIT_USER`` composes via :attr:`_ready_to_advance`)
        and ``detached`` / ``running_tools`` (backgrounded work does not block
        the model-call gate -- it fires on ``not self.cohort``).

        Snapshot at call time; read within one synchronous block.
        """
        return not self.cohort and self.model_call is None and self.compact_task is None

    @property
    def _ready_to_advance(self) -> bool:
        """True when the engine is quiescent *and* no ``AWAIT_USER`` is armed.

        :attr:`_engine_quiescent` plus ``not inbox.gate_armed``: the predicate
        the gates that advance the conversation (the deferred-commit flush and
        the model-call gate) require, since both must hold while a ``Halt`` /
        ``Clear`` / ``ModelResponseError`` has parked the inbox on
        ``AWAIT_USER`` waiting for the user to resume.

        Snapshot at call time; read within one synchronous block.
        """
        return self._engine_quiescent and not self.inbox.gate_armed

    @property
    def is_idle(self) -> bool:
        """True iff a freshly pushed user-side message would drain now.

        Companion to :meth:`_fully_drained` but with a different
        contract -- they are intentionally NOT equivalent:

        * :meth:`_fully_drained` answers "should the runtime publish
          ``AgentIdle``?" -- the strict invariant guarding the
          edge-triggered publish at the top of every ``run_forever``
          iteration. It excludes detached background work because the
          agent is not fully done until those land.

        * ``is_idle`` answers "can a REPL keybinding push a queued
          block to the inbox now and expect the gate to fire it?".
          Detached background tools are irrelevant to this -- the
          model-call gate fires on ``not self.cohort``, not
          ``not self.detached`` -- so the predicate omits ``detached``.
          ``inbox.empty()`` and ``_should_call_model()`` are also
          omitted: both flip on the very push the caller is about to
          make.

        Adding ``detached`` to this predicate would break REPL
        responsiveness: every time a tool ran in the background, Tab
        and Up-Enter would refuse to dispatch.

        Built on :attr:`_ready_to_advance` (quiescent + no armed
        ``AWAIT_USER``) plus the two extra "no half-consumed work" terms
        the model-call gate does not need but a REPL push does:
        ``running_tools`` (a serialized group still executing) and
        ``_mid_stream_queue`` (buffered streaming input).

        Snapshot at call time. The caller must read this and act on it
        within the same synchronous block (no intervening ``await``);
        asyncio's cooperative scheduling guarantees no other coroutine
        runs during.
        """
        return (
            self._ready_to_advance
            and not self.running_tools
            and not self._mid_stream_queue
        )

    @property
    def accepts_user_dispatch(self) -> bool:
        """True iff a directly-pushed user message would make progress now.

        The REPL's dispatch-vs-stage predicate. Dispatch (push a
        ``UserMessage`` straight to the inbox) whenever the runtime can
        act on it without discarding in-flight model work -- i.e. when no
        model call is streaming and no compaction is running. This covers:

        * **Idle** -- a pushed message fires the model gate immediately.
        * **Awaiting user** -- an ``AWAIT_USER`` / ``AWAIT_RECOVERY`` gate
          is parked (after ``Halt`` / ``Clear`` / ``ModelResponseError``);
          the message releases it and resumes the loop. This wins over
          stale task fields: immediately after ``Halt`` the gate can be
          armed before the cancelled ``model_call`` slot has cleared.
        * **Mid-cohort** -- tools are running but no model is streaming
          (``model_call is None``). The ``UserMessage`` handler preempts:
          it detaches the running cohort to the background and fires a
          fresh round for the user's input ("type to redirect"). Staging
          instead would never reach that handler -- the queue pane only
          commits at the next ``ModelResponseComplete`` /
          ``AgentIdle``, so the tools would run to completion with no
          detach.

        Only two states STAGE (return False): a streaming ``model_call``
        (the mid-stream buffer + the ``before_tool_spawn`` pop at
        ``ModelResponseComplete`` perform the detach there, preserving the
        partial stream) and an in-flight ``compact_task`` (compaction is
        not cleanly preemptible; the staged message commits after).

        Exception -- a pending inbox item while ``model_call`` is set
        forces dispatch. This closes the ``Halt`` race: between Ctrl+C
        (which queues ``Halt`` but does not drain it) and the runtime
        cancelling the model call, ``model_call`` is still set. Staging
        then would orphan the message -- the imminent ``AWAIT_USER`` arm
        suppresses ``AgentIdle``, the only edge that commits a staged
        queue block, so it would never reach the model. A non-empty inbox
        means the runtime is mid-transition: dispatch so the message lands
        in the inbox and is processed in the same drain.

        Snapshot at call time; read within one synchronous block.
        """
        if self.inbox.gate_armed:
            return True
        if self.model_call is None and self.compact_task is None:
            return True
        return self.model_call is not None and not self.inbox.empty()

    @property
    def accepts_deferred_dispatch(self) -> bool:
        """True iff a directly-pushed DEFERRED message would dispatch now.

        Tab's dispatch-vs-stage predicate, distinct from
        :attr:`accepts_user_dispatch` (Enter's) in exactly one state:
        **mid-cohort**. Tab means "defer until the current round chain
        goes idle," so while a cohort runs Tab must STAGE into the deferred
        pane, never dispatch -- dispatching a ``UserDeferredMessage``
        mid-cohort would preempt the very work the user chose not to
        interrupt. Enter, by contrast, dispatches mid-cohort to redirect.

        So Tab dispatches only when no tool work is actively running that
        the defer should wait behind: ``accepts_user_dispatch`` (idle,
        awaiting-user, halt-race) AND an empty cohort / no running tools.
        Streaming and compaction already stage via
        ``accepts_user_dispatch`` returning False.

        Snapshot at call time; read within one synchronous block.
        """
        return self.accepts_user_dispatch and not self.cohort and not self.running_tools

    def _fully_drained(self) -> bool:
        """True iff the agent has no work to do and no gate is armed.

        Companion to :attr:`is_idle` -- this is the strict
        ``AgentIdle``-publish gate; ``is_idle`` is the looser
        "REPL can push input now" predicate. See ``is_idle``'s
        docstring for the contract difference.

        Sources of work checked:

        * ``inbox`` non-empty -- items waiting to be drained.
        * ``model_call`` set -- LLM call in flight.
        * ``compact_task`` set -- compaction in progress.
        * ``cohort`` non-empty -- tool batch in progress.
        * ``running_tools`` non-empty -- individual tool tasks live.
        * ``detached`` non-empty -- backgrounded tools whose results
          will land later. Treated as work-in-progress: the agent has
          unfinished business even if it can accept new input.
        * ``has_pending_background()`` true -- the Agent layer has live
          ``background: true`` / ``delay`` tool jobs in its ``_bg``
          registry. The runtime cannot see ``_bg`` directly, so the Agent
          supplies this callback. Without it, ``AgentIdle`` fires while a
          backgrounded tool is still running and a one-shot ``Agent.run``
          reaps the live job before its forward result lands.
        * ``_mid_stream_queue`` non-empty -- buffered ``UserMessage``
          received while the model was streaming.
        * ``inbox.gate_armed`` -- the inbox is waiting for a specific
          event type (e.g. ``AWAIT_USER`` after ``Halt`` /
          ``ModelResponseError``). Semantically "parked on a particular
          event," not "idle."
        * ``_should_call_model()`` -- history tail wants a model turn.
          The end-of-iteration gate will fire one this pass; we are
          about to be busy, not idle. **This is the only source that
          resolves the tape** (via ``self.context()``). The cost is
          one cached lookup in the hot path because the gate sections
          below call ``self.context()`` already; a future change to
          ``_should_call_model`` that bypasses the cache (or to the
          tape resolver that invalidates per call) would shift this
          predicate from O(1) to O(tape).

        Local ``run_forever`` state (``awaiting_user``, ``queued``) is
        not consulted directly: ``awaiting_user`` correlates with
        ``inbox.gate_armed``, and ``queued`` correlates with
        ``model_call`` being set (the only case a queued list can
        survive across iterations).

        Built on :attr:`is_idle` (the REPL-push predicate, itself
        :attr:`_ready_to_advance` + no half-consumed tool/stream work),
        adding the terms that distinguish "fully done" from "can accept
        input": an empty inbox, no backgrounded ``detached`` work, no live
        Agent-layer background tool (``has_pending_background``), and a tail
        that does not itself want a model turn.

        Snapshot at call time. The caller must read this and act on it
        within the same synchronous block (no intervening ``await``),
        which asyncio's cooperative scheduling guarantees no other
        coroutine will run during.
        """
        return (
            self.is_idle
            and self.inbox.empty()
            and not self.detached
            and not (
                self.has_pending_background is not None
                and self.has_pending_background()
            )
            and not self._should_call_model()
        )

    def publish(self, event: RuntimeEvent) -> None:
        """Fan out an event to all observers.

        Args:
          event: Event to deliver to every observer; exceptions are logged.

        """
        for obs in tuple(self.observers):
            try:
                obs(event)
            except Exception:
                logger.exception("observer raised on %s", type(event).__name__)

    def discard_detached(self, call_id: str) -> asyncio.Task[None] | None:
        """Unregister a detached tool task; return the dropped task, if any.

        The runtime owns ``self.detached`` so outside callers (Agent
        cancel verbs, test scaffolds) go through this method rather than
        reaching into the dict. Cancelling the returned task remains the
        caller's responsibility -- this method only severs the runtime's
        reference so a subsequent ``DetachedResult`` for the same
        ``call_id`` no longer fires the late-splice path.

        Args:
          call_id: Provider call id whose detached task should be dropped.

        Returns:
          task: The previously-registered task, or ``None`` when no task
              was registered under ``call_id``.

        """
        return self.detached.pop(call_id, None)

    def context(self) -> ResolvedContext:
        """Return the provider-facing context resolved from the tape.

        Memoized against ``len(self.tape)`` so repeated calls within one
        gate iteration walk the tape once.

        Returns:
          resolved: Resolved messages, origins, slot anchors, and version.

        """
        if self._cached_resolved is None or self._cached_resolved.version != len(
            self.tape
        ):
            self._cached_resolved = resolve_context(self.tape)
        return self._cached_resolved

    def append_history(self, entry: TapeEvent) -> TapeRef:
        """Append a ``ReferrableTapeEvent`` for ``entry`` to the tape.

        Args:
          entry: Provider-facing message to record.

        Returns:
          ref: ``TapeRef`` minted for the new record.

        """
        ref = self.mint_ref()
        record = ReferrableTapeEvent(ref=ref, event=entry)
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)
        return ref

    def append_splice(
        self,
        *,
        mask: tuple[MaskRange, ...] = (),
        insert_after: TapeRef | None = None,
        payload: tuple[ModelContextEvent, ...] = (),
        strategy: str = "",
        token_before: int = 0,
        token_after: int = 0,
        fallback_reason: str = "",
        preserved_tail_count: int = 0,
        paired_externally: frozenset[str] = frozenset(),
    ) -> TapeRef:
        """Append a ``ContextSplice`` to the tape.

        Validates that ``mask`` does not overlap any existing splice's
        mask (each tape ref has at most one editor for its lifetime).
        To re-edit, mask the editing splice's own ref. Producers must
        also avoid placing ``insert_after`` inside the splice's own
        ``mask`` (validated below).

        Args:
          mask: Inclusive ``(from, to)`` ranges of tape refs to mask.
          insert_after: Anchor ref after which ``payload`` renders;
              ``None`` injects at the head of the visible view.
          payload: Provider-facing messages this splice injects.
          strategy: Name of the producing strategy.
          token_before: Token count of the masked view portion.
          token_after: Token count of the injected payload.
          fallback_reason: Reason the producer fell back to non-summary
              payload, if any.
          preserved_tail_count: Number of tail entries preserved verbatim
              in fallback mode.
          paired_externally: Call ids whose pair lives outside this
              payload (typically a ``ReferrableTapeEvent``).

        Returns:
          ref: ``TapeRef`` minted for the new splice.

        Raises:
          InvalidSpliceError: When ``mask`` overlaps any existing alive
              splice's mask, or ``insert_after`` falls inside ``mask``.

        """
        self._validate_no_alive_mask_overlap(mask)
        # ``mask_contains_ref`` compares ``session_id`` before ordinal: a raw
        # ordinal compare false-rejects an ``insert_after`` anchor from a
        # different session whose ordinal happens to fall in this mask's range
        # (multi-session resumed/legacy tapes).
        if insert_after is not None and mask_contains_ref(mask, insert_after):
            raise InvalidSpliceError(
                f"insert_after {insert_after} falls inside this splice's own mask",
            )
        ref = self.mint_ref()
        record = ContextSplice(
            ref=ref,
            mask=mask,
            insert_after=insert_after,
            payload=payload,
            strategy=strategy,
            token_before=token_before,
            token_after=token_after,
            fallback_reason=fallback_reason,
            preserved_tail_count=preserved_tail_count,
            paired_externally=paired_externally,
        )
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)
        self._invalidate_masked_anchors(record)
        return ref

    def _validate_no_alive_mask_overlap(self, new_mask: tuple[MaskRange, ...]) -> None:
        """Reject ``new_mask`` if it shares any position with a currently
        alive splice's mask.

        A splice is alive iff its record ref isn't covered by another
        alive splice's mask. The rule is: each tape ref has at most one
        editor at a time. The new splice may absorb (mask) existing
        alive splices — their masking responsibilities lapse — but it
        may not double-claim a position from an alive splice it isn't
        absorbing.

        Args:
          new_mask: Inclusive ``(from, to)`` ranges this splice claims.

        Raises:
          InvalidSpliceError: On overlap with an alive splice's mask
              where the alive splice's ref is NOT in ``new_mask``.

        """
        alive = alive_splices(self.tape)
        for alive_ref in alive:
            splice = self._tape_by_ref.get(alive_ref)
            if not isinstance(splice, ContextSplice):
                continue
            if mask_contains_ref(new_mask, alive_ref):
                continue  # being absorbed by this splice
            if mask_ranges_overlap(new_mask, splice.mask):
                raise InvalidSpliceError(
                    f"new splice mask overlaps alive splice {alive_ref};"
                    f" either don't claim already-claimed positions or extend"
                    f" the mask to include the splice's own ref to absorb it",
                )

    def append_clear(self) -> TapeRef:
        """Append a barrier splice that masks the full tape prefix.

        Empty payload masking every prior tape ref. Also resets the
        per-call_id state (placeholder refs, parent-AM refs) so
        subsequent splices don't try to anchor at refs the barrier has
        masked.

        Returns:
          ref: ``TapeRef`` minted for the barrier splice.

        """
        if not self.tape:
            # Nothing to mask; mint a no-op marker splice.
            ref = self.mint_ref()
            record = ContextSplice(
                ref=ref,
                mask=(),
                insert_after=None,
                payload=(),
                strategy="clear",
            )
            self.tape.append(record)
            self._cached_resolved = None
            self._index_record(record)
            self._clear_detached_anchors()
            return ref
        ref = self.append_splice(
            mask=full_tape_mask(self.tape),
            insert_after=None,
            payload=(),
            strategy="clear",
        )
        self._clear_detached_anchors()
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
        self._mimic_counter = max(self._mimic_counter, _max_mimic_index(records) + 1)
        self._cached_resolved = None
        for record in records:
            self._index_record(record)

    def mint_ref(self) -> TapeRef:
        """Mint the next ``TapeRef`` and advance the ordinal counter.

        Used by compactors that build ``ContextSplice`` instances
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

        Used by compactors that build ``ContextSplice`` instances
        directly via ``mint_ref()``. The runtime updates its side
        tables (cache invalidation, ref index, call_id anchors) and
        appends the record to the tape.

        Args:
          record: Pre-built tape record whose ``ref`` was minted via
              :meth:`mint_ref` on this runtime.

        Raises:
          InvalidSpliceError: When ``record`` is a ``ContextSplice``
              whose mask overlaps another alive splice's mask without
              absorbing it.

        """
        if isinstance(record, ContextSplice):
            self._validate_no_alive_mask_overlap(record.mask)
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)
        if isinstance(record, ContextSplice):
            self._invalidate_masked_anchors(record)

    def _clear_detached_anchors(self) -> None:
        """Forget per-call detached splice anchors invalidated by barriers."""
        self._placeholder_refs.clear()
        self._parent_assistant_refs.clear()

    def _invalidate_masked_anchors(self, splice: ContextSplice) -> None:
        """Drop per-call anchors whose ref is masked by ``splice``.

        A compaction barrier masks prior tape refs; any call_id anchor
        whose ref is now masked must be evicted unless the splice
        payload preserves it (``_index_record`` already overwrote
        preserved anchors with the splice's own ref, so anchors still
        pointing at masked refs after indexing are the unrecovered
        ones). Without this eviction, a late ``DetachedResult`` would
        try to splice into a masked placeholder and the splice would
        either fail visibility checks or, worse, anchor on a ref the
        resolver no longer renders.
        """
        if not splice.mask:
            return
        for cid, ref in list(self._parent_assistant_refs.items()):
            if mask_contains_ref(splice.mask, ref):
                del self._parent_assistant_refs[cid]
        for cid, ref in list(self._placeholder_refs.items()):
            if mask_contains_ref(splice.mask, ref):
                del self._placeholder_refs[cid]

    def _index_record(self, record: TapeRecord) -> None:
        """Cache ref->record and call_id->anchor mappings after append.

        ``ContextSplice`` payloads also contribute anchors: a compactor
        that preserves an ``AssistantMessage(tool_calls=...)`` paired
        externally (e.g. fallback mode keeping the parent assistant)
        registers its ``tool_calls`` ids against the splice ref so
        ``_parent_id_for_call`` still resolves the right ``parent_id`` for a
        detached tool's stub after the barrier.
        """
        self._tape_by_ref[record.ref] = record
        if isinstance(record, ReferrableTapeEvent):
            event = record.event
            if isinstance(event, AssistantMessage):
                for tc in event.tool_calls:
                    self._parent_assistant_refs[tc.id] = record.ref
            elif isinstance(event, ToolResult):
                self._placeholder_refs[event.call_id] = record.ref
            return
        assert isinstance(record, ContextSplice)
        for entry in record.payload:
            if isinstance(entry, AssistantMessage):
                for tc in entry.tool_calls:
                    self._parent_assistant_refs[tc.id] = record.ref
            elif isinstance(entry, ToolResult):
                self._placeholder_refs[entry.call_id] = record.ref

    def _defer_detached_forward(self, result: ToolResult) -> None:
        """Queue a completed detached tool's real result for forward delivery.

        The original ``tool_use`` keeps its permanent ``[detached]`` stub; the
        real result is committed later as a synthetic ``DetachedArrived`` pair
        (``_flush_pending``), preserving full ``ToolResult`` structure. See
        ``docs/private/design_detached_tool_results.md``.

        Idempotent per ``call_id``: a second delivery for an id already
        forwarded (or already queued) is dropped, so no two
        ``DetachedArrived`` pairs ever share an arrival id (the wedge in
        ``Issue#294``).

        A ``PENDING`` stub (``[detached]`` / ``[Running in background]``) is
        never forwarded and never marks the id delivered: it is not the tool's
        real output, and forwarding it would mark the id consumed and suppress
        the genuine result that arrives later via the background task's own
        ``DetachedResult`` (the regression in ``Issue#294``'s review). The real
        result then forwards normally. Keyed on ``ToolResult.kind``, never on
        ``content`` text, so a real output resembling a stub is never dropped.

        A ``CANCELLED`` result whose call is ALREADY answered in-slot by a
        terminal result (a cohort ``Kill``: ``_stop_tool`` wrote the
        ``CANCELLED`` answer, then the cancelled task's unwind posts a second
        ``CANCELLED``) is not forwarded -- the model already has the
        cancellation, and a forward pair would tell it the same thing twice
        (``f43f811c9`` review). A background job's cancellation, whose in-slot
        answer is a ``PENDING`` running-stub, still forwards (it is the only
        delivery of that outcome).
        """
        if result.kind is ToolResultKind.PENDING:
            logger.debug(
                "runtime detached forward skipped (PENDING stub, not real result):"
                " call_id=%s",
                result.call_id,
            )
            return
        if result.kind is ToolResultKind.CANCELLED and self._inslot_result_is_terminal(
            result.call_id
        ):
            logger.debug(
                "runtime detached forward skipped (cancellation already answered"
                " in-slot): call_id=%s",
                result.call_id,
            )
            return
        if result.call_id in self._forwarded_call_ids:
            logger.debug(
                "runtime detached forward suppressed (already delivered): call_id=%s",
                result.call_id,
            )
            return
        self._forwarded_call_ids.add(result.call_id)
        self._pending_commits.append(_PendingCommit(kind="forward", result=result))

    def _inslot_result_is_terminal(self, call_id: str) -> bool:
        """True when ``call_id``'s in-slot ``ToolResult`` is a final answer.

        Reads the call's existing placeholder/result record (O(1) via
        ``_placeholder_refs``). ``CANCELLED`` / ``FINAL`` are terminal (the
        model already has the answer); ``PENDING`` is not (a real result is
        still pending forward delivery). ``_index_record`` registers
        ``_placeholder_refs`` from both plain history records AND
        ``ContextSplice`` payloads (a result preserved across compaction), so
        both shapes are inspected -- otherwise a terminal result compacted into
        a splice would read as non-terminal and a duplicate cancellation would
        forward (``77bf1d67f`` review C3).
        """
        ref = self._placeholder_refs.get(call_id)
        if ref is None:
            return False
        record = self._tape_by_ref.get(ref)
        if isinstance(record, ReferrableTapeEvent):
            event = record.event
            return (
                isinstance(event, ToolResult)
                and event.kind is not ToolResultKind.PENDING
            )
        if isinstance(record, ContextSplice):
            return any(
                isinstance(entry, ToolResult)
                and entry.call_id == call_id
                and entry.kind is not ToolResultKind.PENDING
                for entry in record.payload
            )
        return False

    def _defer_pairing(self, result: ToolResult) -> None:
        """Queue an error ``ToolResult`` pairing an already-emitted ``tool_use``.

        Used for a mimicked ``DetachedArrived`` call: its error is committed
        in-slot (adjacent to the parent assistant) on the next real turn, so
        the bogus call spends no round of its own.
        """
        self._pending_commits.append(_PendingCommit(kind="pairing", result=result))

    def _defer_ride_along(self, user: UserMessage) -> None:
        """Queue system-injected user-side context to ride the next real turn."""
        self._pending_commits.append(_PendingCommit(kind="ride_along", user=user))

    def _flush_pending(self) -> None:
        """Commit every queued :class:`_PendingCommit`, in arrival order, in-slot.

        The single drain for all deferred tape commits. Called at the model
        gate once a round is confirmed firing (so nothing here wakes the model
        on its own) and immediately before ``_assert_alternation_invariant``,
        so the committed context is what the provider receives.

        Placement is by ``kind``:

        - ``pairing``: insert the result immediately after its parent
          ``AssistantMessage`` (``insert_after``), so a mimicked ``tool_use``
          pairs in-slot and can never be stranded by intervening turns.
        - ``forward``: append the synthetic ``DetachedArrived`` pair (a
          user-role separator first when the tail is an assistant turn).
        - ``ride_along``: coalesce onto the user side.
        """
        pending, self._pending_commits = self._pending_commits, []
        for commit in pending:
            if commit.kind == "pairing":
                assert commit.result is not None
                self._commit_pairing(commit.result)
            elif commit.kind == "forward":
                assert commit.result is not None
                self._append_detached_arrival(commit.result)
            else:
                assert commit.user is not None
                self.publish(self._append_or_coalesce_user(commit.user))

    def _has_waking_commit(self) -> bool:
        """True when a queued commit should itself drive a round.

        Only ``forward`` (a completed detached tool's real result) wakes the
        model: a finished tool is worth surfacing promptly. ``pairing`` /
        ``ride_along`` are non-urgent and must ride a round driven by real
        content, never fire one alone.
        """
        return any(c.kind == "forward" for c in self._pending_commits)

    def _commit_pairing(self, result: ToolResult) -> None:
        """Insert ``result`` immediately after the ``AssistantMessage`` that
        emitted its ``call_id``, pairing the dangling ``tool_use`` in-slot.

        Slot placement (not tail append) is load-bearing: an intervening user
        turn or a forward detached delivery must not strand the ``tool_use``
        and trip orphan-repair into an ``[interrupted]`` substitution.
        """
        parent_ref = self._parent_assistant_refs.get(result.call_id)
        if parent_ref is None or parent_ref not in set(self.context().origins):
            # Parent assistant gone (compaction/Clear); nothing to pair. The
            # call is no longer visible, so dropping the result is correct.
            return
        self.append_splice(
            mask=(),
            insert_after=parent_ref,
            payload=(result,),
            strategy="lazy_pairing",
            paired_externally=frozenset({result.call_id}),
        )
        self.publish(result)

    def _append_detached_arrival(self, result: ToolResult) -> None:
        """Append the synthetic ``DetachedArrived`` tool pair for ``result``.

        A user-role separator precedes the synthetic ``AssistantMessage`` when
        the history tail is itself an assistant turn, since the provider
        forbids assistant->assistant. The separator names the arriving call so
        the model has a forward, in-band signal that the awaited tool finished.
        """
        messages = self.context().messages
        if messages and isinstance(messages[-1], AssistantMessage):
            self._append_or_coalesce_user(
                UserMessage(text=f"[detached tool {result.call_id} completed]"),
            )
        arrival_id = f"{result.call_id}{DETACHED_ARRIVAL_SUFFIX}"
        self.append_history(
            AssistantMessage(
                tool_calls=(
                    ToolCall(id=arrival_id, name=DETACHED_ARRIVED_TOOL, args={}),
                ),
            ),
        )
        self.append_history(dataclasses.replace(result, call_id=arrival_id))

    async def run_forever(self) -> None:
        """Drain inbox, dispatch, repeat. The entire engine."""
        awaiting_user = False
        queued: list[UserQueuedMessage | AgentSendQueuedMessage] = []
        deferred: list[UserDeferredMessage | AgentSendDeferredMessage] = []

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
                            self._cancel_model_and_compaction()
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
                            else:
                                self.inbox.push_front(
                                    AWAIT_USER,
                                    *items[item_idx + 1 :],
                                )
                                awaiting_user = True
                                break

                        case Clear():
                            self._cancel_model_and_compaction()
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
                            self._pending_commits.clear()
                            self._forwarded_call_ids.clear()
                            queued.clear()
                            deferred.clear()
                            self._mid_stream_queue.clear()
                            self.append_clear()
                            self.inbox.push_front(
                                AWAIT_USER,
                                *items[item_idx + 1 :],
                            )
                            # Publish *after* arming ``AWAIT_USER`` so an
                            # observer that responds by pushing user input
                            # (the REPL committer flushing deferred Tab input
                            # on ``ClearComplete``) lands as a genuinely-new
                            # post-arm item. Publishing first would let that
                            # push count toward the gate baseline, so it would
                            # satisfy the baseline without exceeding it and the
                            # gate would never release -- re-wedging.
                            self.publish(ClearComplete())
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
                            # The error is not conversation: the published
                            # ``ModelResponseError`` already renders the banner
                            # and halts. Synthesizing a ``[Error: ...]``
                            # ``UserMessage`` here only polluted model context
                            # (the model never needs to see its own transport
                            # failures) and coalesced into the user's next bar
                            # (Issue#316 #6). Drop it; the gate still parks on
                            # ``AWAIT_USER`` below when no mid-stream content
                            # remains.
                            coalesced = self._drain_mid_stream_queue()
                            if coalesced is not None:
                                self.publish(coalesced)
                            self.publish(item)
                            if coalesced is None:
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
                                # Already-detached tools are still owned by the
                                # runtime; kill discards the registry entry and
                                # cancels the live task so the late-splice path
                                # cannot fire.
                                task = self.discard_detached(cid)
                                if task is not None:
                                    logger.debug(
                                        "runtime kill detached tool: call_id=%s",
                                        cid,
                                    )
                                    if not task.done():
                                        _ = task.cancel()
                                else:
                                    logger.debug(
                                        "runtime kill missed tool: call_id=%s",
                                        cid,
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
                                self._model_call_generation += 1
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
                            started = CompactStarted()
                            self.append_history(started)
                            self.publish(started)

                        case CompactComplete():
                            # The compaction task already appended its
                            # override(s) to the tape; this handler just
                            # clears bookkeeping and republishes.
                            self.compact_task = None
                            self.append_history(item)
                            self.publish(item)

                        case CompactFailed():
                            self.compact_task = None
                            # The ``CompactFailed`` event (appended + published
                            # below) is taped and rendered as a dim line. The
                            # former ``[Compaction error: ...]`` ``UserMessage``
                            # only polluted model context (Issue#316 #6); drop
                            # it.
                            self.append_history(item)
                            self.publish(item)

                        case UserMessage():
                            if self.model_call is not None:
                                # Mid-stream: buffer only. The REPL input pane
                                # renders ``pending_mid_stream()`` as
                                # a dim preview while the buffer is non-empty,
                                # so the user has immediate visual feedback
                                # without a duplicate bar in console. The bar
                                # appears on drain (ModelResponseComplete /
                                # Halt / ModelResponseError / Compact) when
                                # the coalesced UserMessage is published --
                                # at which point the preview drops because the
                                # buffer is empty. One UI surface at a time.
                                # If ``preempt_in_flight`` is enabled AND the
                                # message is flagged ``urgent`` (the historical
                                # default, preserved for tests/internal
                                # callers that don't specify), additionally
                                # SIGINT the provider so a CLI-driven opaque
                                # turn aborts immediately rather than waiting
                                # for natural completion. ``urgent=False`` lets
                                # ingress layers (e.g. plugin web UIs) opt
                                # operator messages into queue-by-default so
                                # back-to-back operator typing doesn't waste
                                # the recipient's in-flight compute on routine
                                # follow-ups; see AgentSendMessage handler
                                # below for the symmetric peer case.
                                if self._preempt_in_flight and item.urgent:
                                    cancel = getattr(
                                        self.model,
                                        "cancel_in_flight",
                                        None,
                                    )
                                    if callable(cancel):
                                        try:
                                            cancel()
                                        except Exception:
                                            logger.exception(
                                                "preempt_in_flight: "
                                                "model.cancel_in_flight() raised",
                                            )
                                self._mid_stream_queue.append(item)
                            else:
                                # Mid-cohort or idle: preempt and append.
                                # Coalesce on the alternation-invariant helper
                                # so two same-batch Enters (or a post-halt
                                # follow-up) don't stack as consecutive user
                                # turns in history.
                                self._stop_all_tools(mode="detach")
                                committed = self._append_or_coalesce_user(item)
                                self.publish(committed)
                            awaiting_user = False

                        case AgentSendMessage():
                            if self.model_call is not None:
                                # Provider-side cancel for CLI-driven models
                                # whose tool loop is opaque (no cohort to
                                # detach). When enabled, SIGINT the
                                # subprocess so the in-flight call resolves
                                # as ModelResponseError; the buffered
                                # message drains on the next gate firing.
                                # No-op for providers without
                                # ``cancel_in_flight`` or when the runtime
                                # was not opted into preempt-in-flight.
                                #
                                # Peer messages preempt only when the
                                # sender flagged the message as ``urgent``
                                # (default False). Routine peer traffic
                                # (acks, status updates, FYIs) queues
                                # cleanly without interrupting the
                                # recipient's current turn; only genuine
                                # interrupt-class messages (TL STOPs,
                                # pivot directives) pay the preempt cost.
                                # Empirically 2026-06-04: ~72% of TL's
                                # preempts were routine peer messages
                                # that shouldn't have interrupted at
                                # all -- this gate eliminates that
                                # wasted compute.
                                if self._preempt_in_flight and item.urgent:
                                    cancel = getattr(
                                        self.model,
                                        "cancel_in_flight",
                                        None,
                                    )
                                    if callable(cancel):
                                        try:
                                            cancel()
                                        except Exception:
                                            logger.exception(
                                                "preempt_in_flight: "
                                                "model.cancel_in_flight() raised",
                                            )
                                self._mid_stream_queue.append(item)
                            else:
                                self._stop_all_tools(mode="detach")
                                committed = self._append_or_coalesce_user(item)
                                self.publish(committed)
                            awaiting_user = False

                        case UserQueuedMessage() | AgentSendQueuedMessage():
                            if awaiting_user:
                                committed = self._append_or_coalesce_user(
                                    _message_from_queued(item)
                                )
                                self.publish(committed)
                                awaiting_user = False
                            else:
                                queued.append(item)

                        case UserDeferredMessage() | AgentSendDeferredMessage():
                            deferred.append(item)

                        case ModelResponsePartial():
                            self.publish(item)

                        case ModelResponseThinking():
                            self.publish(item)

                        case ModelResponseComplete(message=msg, generation=generation):
                            if generation not in (-1, self._model_call_generation):
                                logger.debug(
                                    "runtime stale model response ignored: "
                                    "generation=%d current=%d",
                                    generation,
                                    self._model_call_generation,
                                )
                                continue
                            self.model_call = None
                            # Neutralize any model-forged ``DetachedArrived``
                            # call at the entry boundary, before the message
                            # reaches history / cohort / pairing. Every
                            # downstream consumer then sees an id that cannot
                            # collide with a real arrival id (``Issue#297``).
                            msg = self._sanitize_forged_arrivals(msg)
                            before_tool_spawn = None
                            if self.before_tool_spawn is not None:
                                before_tool_spawn = self.before_tool_spawn(msg)
                            if before_tool_spawn is not None and not isinstance(
                                before_tool_spawn, UserMessage
                            ):
                                self.inbox.push_front(
                                    before_tool_spawn,
                                    *items[item_idx + 1 :],
                                )
                                break
                            self.append_history(msg)
                            self.publish(item)
                            if isinstance(before_tool_spawn, UserMessage):
                                self._relegate_tool_calls_to_background(msg)
                                committed = self._append_or_coalesce_user(
                                    before_tool_spawn
                                )
                                self.publish(committed)
                            elif self._mid_stream_queue:
                                # User typed mid-stream. Cut their content in
                                # line: relegate any tool calls to background
                                # (placeholder + detached task; the result
                                # splices in via ``DetachedResult`` when the
                                # tool finishes), then append the coalesced
                                # user content so the gate fires for it next.
                                # No ``CohortStarted`` / ``ModelIdle`` here:
                                # this round did not idle (a follow-up is
                                # about to fire) and no cohort gates the model.
                                self._relegate_tool_calls_to_background(msg)
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
                                for group in self._partition_cohort(msg.tool_calls):
                                    names = "+".join(tc.name for tc in group)
                                    tool_task = asyncio.create_task(
                                        self._run_tool_group_and_post(
                                            group, parent_id=msg.id
                                        ),
                                    )
                                    tool_task.add_done_callback(
                                        log_task_exception(
                                            logger,
                                            f"cohort tool {names!r} crashed",
                                        ),
                                    )
                                    # Every call in a serialized group shares
                                    # the one task; detach/kill cancels the
                                    # whole group via any member's id.
                                    for tc in group:
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
                            # that cleared the cohort. The original ``tool_use``
                            # is already answered by its ``[detached]`` stub;
                            # deliver the real content forward, exactly like
                            # ``DetachedResult``.
                            del self.detached[cid]
                            self._defer_detached_forward(item)
                            self.publish(item)

                        case DetachedResult():
                            self.cohort.discard(item.call_id)
                            # Deliver the real result as NEW forward context
                            # (a synthetic ``DetachedArrived`` tool pair), never
                            # by back-patching the ``[detached]`` stub slot. The
                            # stub stays the honest answer to the original call;
                            # the result arrives forward so nothing the model
                            # already read is silently rewritten. Compaction
                            # cannot drop it -- delivery keys off the inbox, not
                            # a tape anchor. See
                            # ``docs/private/design_detached_tool_results.md``.
                            self._defer_detached_forward(item.result)
                            self.publish(item)

                        case LazyEvent(payload=payload):
                            # Defer the payload to the next real turn via the
                            # one pending-commit queue. A ``ToolResult`` payload
                            # answers an already-emitted ``tool_use`` (a mimicked
                            # ``DetachedArrived``): settle that call's cohort /
                            # running-tools membership now -- otherwise the
                            # cohort never empties, the model-call gate
                            # (``not self.cohort``) never fires, and the deferred
                            # pairing never flushes (deadlock). A ``UserMessage``
                            # payload is ride-along system context.
                            if isinstance(payload, ToolResult):
                                self.cohort.discard(payload.call_id)
                                self.running_tools.pop(payload.call_id, None)
                                self._defer_pairing(payload)
                            else:
                                self._defer_ride_along(payload)

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
                    self._engine_quiescent
                    and not self._should_call_model()
                    and (queued or deferred)
                ):
                    if queued:
                        for committed in self._commit_queued_user_side(queued):
                            self.publish(committed)
                        queued.clear()
                    elif deferred:
                        for committed in self._commit_queued_user_side(deferred):
                            self.publish(committed)
                        deferred.clear()

                # The one deferred-commit flush. ``_flush_pending`` commits each
                # queued entry in-slot. ``waking`` entries (a completed detached
                # tool's forward delivery) *should* surface promptly, so they
                # flush whenever the tail is appendable; their ``ToolResult``
                # tail then drives the round below so the model observes them.
                # Non-waking entries (a mimicked-tool error pairing, ride-along
                # context) must not fire a round of their own, so they wait for
                # a round already being driven by real content
                # (``_should_call_model``). A gate-armed ``AWAIT_USER`` (Halt)
                # blocks both until the user resumes.
                if (
                    self._pending_commits
                    and self._ready_to_advance
                    and (self._should_call_model() or self._has_waking_commit())
                ):
                    self._flush_pending()

                if self._ready_to_advance and self._should_call_model():
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
                    try:
                        self._assert_alternation_invariant()
                    except (
                        InvalidContextError,
                        InvalidPayloadError,
                        InvalidSpliceError,
                    ) as exc:
                        # The gate cannot fire against an invalid context that
                        # even rescue could not repair (rescue's own
                        # ``append_splice`` raises ``InvalidPayloadError`` /
                        # ``InvalidSpliceError`` when its sanitized payload is
                        # still malformed). Surfacing the failure
                        # (rather than letting the master catch swallow it) is
                        # mandatory: a silent re-raise every iteration wedges
                        # the loop on the same broken tape with no UI feedback
                        # -- the user sees a frozen prompt (``Issue#294``). Arm
                        # ``AWAIT_USER`` so the loop stops re-validating the
                        # same tape until the user acts, and skip the model
                        # spawn below: firing the provider on the unrepaired
                        # context is the very failure this guard prevents.
                        log_exception_or_warning(
                            logger, "model-call gate context unrepairable", exc
                        )
                        # Arm ``AWAIT_RECOVERY`` BEFORE publishing: an observer
                        # that reacts to ``ModelResponseError`` by pushing a
                        # recovery verb (``Clear`` / ``Compact``) must land a
                        # genuinely-new post-arm item. Publishing first would let
                        # that push count toward the gate baseline, so it would
                        # satisfy the baseline without exceeding it and the gate
                        # would never release -- stranding the recovery action
                        # (same hazard the ``Clear`` arm documents). The batch is
                        # fully processed here (this gate runs after the per-item
                        # loop), so there are no items to requeue. ``AWAIT_RECOVERY``
                        # admits the control verbs (``Clear`` / ``Compact`` /
                        # ``Recompact``) that repair a broken tape -- which a bare
                        # ``AWAIT_USER`` would strand -- plus ``Halt`` / ``Kill``
                        # so no control input is buffered behind the gate.
                        self.inbox.push_front(AWAIT_RECOVERY)
                        self.publish(ModelResponseError(exc))
                        awaiting_user = True
                    else:
                        self._model_call_generation += 1
                        model_call: asyncio.Task[None] = asyncio.create_task(
                            self._stream_and_post(self._model_call_generation),
                        )
                        self.model_call = model_call
                        model_call.add_done_callback(
                            log_task_exception(logger, "model-call task crashed"),
                        )
                        self.publish(ModelCallStarted())

                self.publish(SaveSession())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- master catch around the dispatch body; a sync raise (e.g. `pending.apply` for a queued ModelSwitch) must not tear down the engine
                log_exception_or_warning(logger, "dispatch loop iteration raised", exc)

    async def run(self, msg: UserMessage) -> list[TapeEvent]:
        """Process one user message to completion.

        Convenience wrapper for tests and AgentSpawn. Sends the
        message, runs until the model is idle, returns history. If the
        engine's ``run_forever`` task crashed under us, its exception is
        re-raised after ``Quit()`` shuts the driver down -- callers that
        expect a clean result see the crash rather than a silent
        truncation past a wedged dispatch loop.

        Args:
          msg: User message to process.

        Returns:
          history: Conversation history after processing.

        Raises:
          Exception: Re-raises any exception the engine task captured,
              including ``ModelResponseError`` payloads that escaped
              into ``run_forever``.

        """
        context_cursor = len(self.context().messages)
        done = asyncio.Event()

        def _watch(event: RuntimeEvent) -> None:
            if isinstance(event, (ModelIdle, ModelResponseError)):
                done.set()

        self.observers.append(_watch)
        task = asyncio.create_task(self.run_forever())
        self.inbox.push_back(msg)
        task.add_done_callback(
            log_task_exception(logger, "run_forever driver task crashed"),
        )

        try:
            await done.wait()
            return list(self.context().messages[context_cursor:])
        finally:
            if _watch in self.observers:
                self.observers.remove(_watch)
            self.inbox.push_back(Quit())
            with contextlib.suppress(asyncio.CancelledError):
                await task
            # Surface engine crashes; without this, ``done.wait()`` returns
            # when ``run_forever`` exits via exception and the caller sees a
            # clean partial history instead of the real failure.
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    raise exc

    def pending_mid_stream(self) -> Sequence[UserMessage | AgentSendMessage]:
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
        return isinstance(messages[-1], (UserMessage, AgentSendMessage, ToolResult))

    def _assert_alternation_invariant(self) -> None:
        """Repair HR-level orphans, validate; rescue if still broken.

        The model-call gate's last guard before firing the provider:

        1. Repair ReferrableTapeEvent-level orphans
           (:meth:`_repair_history_record_orphans`): pair unmatched
           ``tool_use`` ids with synthetic ``[interrupted]``
           ``ToolResult`` records; suppress orphan ``ToolResult`` from
           a ``ReferrableTapeEvent`` origin.
        2. If validation still fails -- typically a legacy session
           reconstructed via :meth:`ContextSplice.replay` with
           invalid payloads predating the construct-time invariant --
           rescue: append a single barrier override that suppresses
           every visible tape ref and re-injects a fully sanitized
           payload. Structural attribution is lost for the rescued
           section but the session stays live.

        Forward producers can no longer emit overrides with invalid
        payloads (the validator in
        :meth:`ContextSplice.__post_init__` rejects them at
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
        :meth:`ContextSplice.replay` reconstructions carry payloads
        the construct-time invariant would reject. Suppresses every
        currently-visible tape ref and re-injects the result of
        :func:`_sanitize_for_send` over the current resolved messages.
        Structural attribution is lost for the rescued section but the
        resolved view is guaranteed wire-format-valid by construction.

        ``paired_externally`` is computed from the sanitized payload
        via :func:`unpaired_call_ids` -- declaring only those call ids
        whose pair is genuinely missing locally keeps the declaration
        honest against the strict validator.
        """
        resolved = self.context()
        sanitized = _sanitize_for_send(resolved.messages)
        if not self.tape:
            return
        self.append_splice(
            mask=full_tape_mask(self.tape),
            insert_after=None,
            payload=tuple(sanitized),
            strategy="context_rescue",
            paired_externally=unpaired_call_ids(sanitized),
        )

    def _repair_history_record_orphans(self) -> None:
        """Pair / drop ReferrableTapeEvent-origin orphans in the resolved context.

        For each ``AssistantMessage`` with unpaired ``tool_calls`` from
        a ``ReferrableTapeEvent`` origin, append an override injecting synth
        ``[interrupted]`` ``ToolResult`` records in its slot suffix.
        For each orphan ``ToolResult`` from a ``ReferrableTapeEvent`` origin,
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
            origin_is_hr = isinstance(origin_record, ReferrableTapeEvent)
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
            # matching AMs live in the ``ReferrableTapeEvent`` at ``anchor``.
            # Empty mask: pure injection, doesn't claim any positions.
            self.append_splice(
                mask=(),
                insert_after=anchor,
                payload=payload,
                strategy="orphan_tool_use_repair",
                paired_externally=frozenset(missing_ids),
            )
        for orphan in hr_orphan_refs:
            # Pure deletion: mask this single HR ref, empty payload.
            self.append_splice(
                mask=(
                    MaskRange(
                        session_id=orphan.session_id,
                        lo=orphan.ordinal,
                        hi=orphan.ordinal,
                    ),
                ),
                insert_after=None,
                payload=(),
                strategy="orphan_tool_result_repair",
            )

    def _commit_queued_user_side(
        self,
        items: Sequence[
            UserQueuedMessage
            | UserDeferredMessage
            | AgentSendQueuedMessage
            | AgentSendDeferredMessage
        ],
    ) -> list[UserMessage | AgentSendMessage]:
        committed: list[UserMessage | AgentSendMessage] = []
        group: list[UserMessage | AgentSendMessage] = []
        for item in items:
            message = _message_from_queued(item)
            if group and type(message) is not type(group[-1]):
                committed.append(
                    self._append_or_coalesce_user(_coalesce_user_side(group))
                )
                group = []
            group.append(message)
        if group:
            committed.append(self._append_or_coalesce_user(_coalesce_user_side(group)))
        return committed

    def _append_or_coalesce_user(
        self, item: UserMessage | AgentSendMessage
    ) -> UserMessage | AgentSendMessage:
        r"""Append ``item`` to history; return the committed entry.

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

        When the owning :class:`AgentRuntime` was constructed with
        ``coalesce_inbox=False`` (e.g. chat-channel runtimes where each
        peer ``AgentSendMessage`` is a deliberate discrete event), the
        coalesce path is replaced by injecting a synthetic empty
        :class:`AssistantMessage` between the tail user-side message
        and the new item, then appending the item as-is. The synthetic
        assistant turn satisfies the API alternation rule without
        hiding the new item inside the prior one — important when the
        new item is e.g. a hard STOP that must be visible to the
        recipient as a distinct inbound, not concatenated onto a tail
        that the model has already started reasoning past.
        """
        resolved = self.context()
        messages = resolved.messages
        if not messages or wire_role(messages[-1]) != wire_role(item):
            self.append_history(item)
            return item
        if not self._coalesce_inbox:
            # Discrete-inbound mode: inject a synthetic empty
            # assistant turn so each peer message remains a distinct
            # history entry. The synthetic turn carries an explicit
            # marker so downstream consumers (UI, audit, retrying
            # providers) can recognise it as runtime-synthesized
            # rather than a real model response.
            self.append_history(
                AssistantMessage(text="(runtime: discrete-inbound boundary)"),
            )
            self.append_history(item)
            return item
        tail = messages[-1]
        assert isinstance(tail, (UserMessage, AgentSendMessage))
        text = f"{tail.text}\n\n{item.text}"
        attachments = tail.attachments + item.attachments
        # Same-type merges preserve the tail's id (downstream consumers key
        # on it). Cross-type merges adopt the agent type when either side
        # is an ``AgentSendMessage`` so the source attribution survives;
        # the merged entry then gets a fresh id (the prior tail's identity
        # is no longer applicable to the changed type).
        if isinstance(tail, AgentSendMessage):
            combined: UserMessage | AgentSendMessage = dataclasses.replace(
                tail,
                text=text,
                attachments=attachments,
            )
        elif isinstance(item, AgentSendMessage):
            combined = dataclasses.replace(
                item,
                text=text,
                attachments=attachments,
            )
        else:
            combined = dataclasses.replace(
                tail,
                text=text,
                attachments=attachments,
            )
        tail_origin = resolved.origins[-1]
        # When the visible tail is itself a coalesce splice's payload,
        # the new splice must absorb both that splice AND the original
        # ref(s) the splice was masking — otherwise killing the splice
        # under undelete semantics would resurrect the originally-
        # masked content (causing both the merged tail AND the
        # originals to render side-by-side).
        sid = tail_origin.session_id
        prior_record = self._tape_by_ref.get(tail_origin)
        # Absorb the prior coalesce splice's OWN same-session mask ranges
        # verbatim (preserving any gaps -- a sparse mask must stay sparse, or
        # we delete records the prior splice intentionally left visible), then
        # add a range for the prior splice's own ref so undelete of the new
        # splice cannot resurrect the originally-masked content. Only ranges in
        # ``sid`` are absorbed: ``prior_record``'s mask may carry ranges from
        # other sessions (resumed/legacy tapes), and the new ranges are built in
        # ``sid``. Each ``MaskRange`` is single-session by construction, so a
        # plain ``session_id`` match selects the right ones.
        prior_ranges: tuple[MaskRange, ...] = ()
        if isinstance(prior_record, ContextSplice):
            # Each range is single-session by construction, so a simple
            # session match suffices (Issue#313 deleted the former
            # both-endpoint cross-session filter).
            prior_ranges = tuple(r for r in prior_record.mask if r.session_id == sid)
        own_range = MaskRange(
            session_id=sid, lo=tail_origin.ordinal, hi=tail_origin.ordinal
        )
        mask = merge_mask_ranges((*prior_ranges, own_range))
        # Anchor: the same-session tape ref immediately before the earliest
        # masked position, or None for head insertion. The scan is
        # session-scoped because the mask is built in ``sid``: a foreign-session
        # record whose ordinal happens to exceed the low must not terminate the
        # scan early and mis-anchor the splice (multi-session tapes).
        lo_ord = min(r.lo for r in mask)
        anchor: TapeRef | None = None
        for record in self.tape:
            if record.ref.session_id != sid:
                continue
            if record.ref.ordinal >= lo_ord:
                break
            anchor = record.ref
        self.append_splice(
            mask=mask,
            insert_after=anchor,
            payload=(combined,),
            strategy="user_coalesce",
        )
        return combined

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
        placeholder, is_error, kind = (
            (CANCELLED_PLACEHOLDER, True, ToolResultKind.CANCELLED)
            if mode == "kill"
            else (DETACHED_PLACEHOLDER, False, ToolResultKind.PENDING)
        )
        # Carry parent_id through; every other placeholder/result append site
        # (1666, 1698, _run_tool_and_post) does so. Without it, downstream
        # consumers that key on parent_id (UI grouping, splice routing) see
        # the synth result as orphan even though the cohort gate matched.
        self.append_history(
            ToolResult(
                call_id=cid,
                parent_id=self._parent_id_for_call(cid),
                content=placeholder,
                is_error=is_error,
                kind=kind,
            ),
        )
        self.detached[cid] = task
        if mode == "kill":
            task.cancel()

    def _parent_id_for_call(self, call_id: str) -> int:
        """Return the originating ``AssistantMessage.id`` for ``call_id``, or -1."""
        parent_ref = self._parent_assistant_refs.get(call_id)
        if parent_ref is None:
            return -1
        record = self._tape_by_ref.get(parent_ref)
        if not isinstance(record, ReferrableTapeEvent):
            return -1
        event = record.event
        if not isinstance(event, AssistantMessage):
            return -1
        return event.id

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

    def _cancel_model_and_compaction(self) -> None:
        """Cancel any in-flight model call and compaction, bumping generations.

        The shared preempt prologue of the hard-reset control events
        (``Halt`` / ``Clear``): cancel the streaming ``model_call`` and bump
        ``_model_call_generation`` so its late ``ModelResponseComplete`` is
        ignored as stale; cancel an in-flight ``compact_task`` and bump
        ``_compact_generation`` so ``_compact_and_post`` refuses to push a
        terminal event, publishing ``CompactFailed`` in its place. Idempotent
        when neither is live. Does NOT touch the cohort -- ``Halt`` preserves
        running tools, and ``Clear`` calls ``_stop_all_tools`` separately.
        """
        if self.model_call:
            self.model_call.cancel()
            self.model_call = None
            self._model_call_generation += 1
        if self.compact_task is not None:
            compact_task = self.compact_task
            if not compact_task.done():
                self._compact_generation += 1
                compact_task.cancel()
                self.publish(
                    CompactFailed(
                        exception=asyncio.CancelledError(),
                        tape_len=len(self.tape),
                    ),
                )
            self.compact_task = None

    def _sanitize_forged_arrivals(self, msg: AssistantMessage) -> AssistantMessage:
        """Rewrite model-forged ``DetachedArrived`` call ids into a safe namespace.

        The ``DetachedArrived`` name and the ``:detached`` arrival-id scheme
        belong to the runtime's synthetic forward-delivery turns. A model can
        emit its own ``DetachedArrived`` call carrying any id, including one
        equal to a real arrival id -- two ``AssistantMessage``s would then share
        a tool_call id, breaking wire validity and stranding one pairing
        (``Issue#297``). Rewriting each forged call's id into the
        ``DETACHED_ARRIVED_MIMIC_PREFIX`` namespace (keyed off a monotonic
        counter) restores global id uniqueness at the single entry boundary, so
        no downstream consumer -- history, cohort, ``_parent_assistant_refs``,
        the mimic pairing -- can be confused. Returns ``msg`` unchanged when it
        carries no forged ``DetachedArrived`` call (the common case).
        """
        if not any(tc.name == DETACHED_ARRIVED_TOOL for tc in msg.tool_calls):
            return msg
        # A model can also emit a non-forged tool call whose id already lies in
        # the mimic namespace; advance the counter past any such collision so
        # the rewrite never creates a duplicate id (which would fail
        # ``AssistantMessage`` validation and lose the whole turn).
        used_ids = {tc.id for tc in msg.tool_calls if tc.name != DETACHED_ARRIVED_TOOL}
        rewritten: list[ToolCall] = []
        for tc in msg.tool_calls:
            if tc.name != DETACHED_ARRIVED_TOOL:
                rewritten.append(tc)
                continue
            safe_id = f"{DETACHED_ARRIVED_MIMIC_PREFIX}{self._mimic_counter}"
            self._mimic_counter += 1
            while safe_id in used_ids:
                safe_id = f"{DETACHED_ARRIVED_MIMIC_PREFIX}{self._mimic_counter}"
                self._mimic_counter += 1
            used_ids.add(safe_id)
            logger.debug(
                "runtime forged DetachedArrived rewritten: %s -> %s",
                tc.id,
                safe_id,
            )
            rewritten.append(dataclasses.replace(tc, id=safe_id))
        return dataclasses.replace(msg, tool_calls=tuple(rewritten))

    def _relegate_tool_calls_to_background(self, msg: AssistantMessage) -> None:
        """Stub every tool call in ``msg`` and spawn it as a detached task.

        The shared "the model produced tool calls but a user redirect cuts in
        line" path: append a ``[detached]`` placeholder answering each
        ``tool_use`` (so history stays wire-valid) and spawn the tool into
        ``self.detached``. Its real result later arrives forward via
        ``DetachedResult`` rather than gating the next model call. Used by
        both ``ModelResponseComplete`` redirect branches -- an external
        ``before_tool_spawn`` ``UserMessage`` and a buffered mid-stream user
        turn -- which differ only in how they commit the user content, not in
        how they background the tools.
        """
        for tc in msg.tool_calls:
            self.append_history(
                ToolResult(
                    call_id=tc.id,
                    parent_id=msg.id,
                    content=DETACHED_PLACEHOLDER,
                    kind=ToolResultKind.PENDING,
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

    def _drain_mid_stream_queue(self) -> UserMessage | AgentSendMessage | None:
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
        coalesced = _coalesce_user_side(self._mid_stream_queue)
        self._mid_stream_queue.clear()
        return self._append_or_coalesce_user(coalesced)

    def _collect_detached(self) -> None:
        """Clean up detached tasks that completed or were cancelled.

        Completed tasks already posted their result as DetachedResult
        from ``_run_tool_and_post`` and removed themselves from
        ``self.detached``. This handles cancelled tasks that never
        posted.
        """
        for cid in [c for c, t in self.detached.items() if t.done()]:
            del self.detached[cid]

    async def _stream_and_post(self, generation: int) -> None:
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
                on_text,
                on_thinking,
            )
            self.inbox.push_back(
                ModelResponseComplete(message=response, generation=generation),
            )
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

    def _partition_cohort(self, calls: Sequence[ToolCall]) -> list[list[ToolCall]]:
        """Split a cohort into serialized groups, preserving submission order.

        Calls sharing a non-``None`` ``serialize_key`` (e.g. same-file
        Read/Edit/Write) collapse into one group run sequentially in
        submission order; every other call becomes a singleton group run
        in parallel. Iterating ``calls`` once and appending keeps each
        group in original order, which is what "sequential in submission
        order" requires.

        Args:
          calls: The cohort's tool calls in submission order.

        Returns:
          groups: List of call groups; same-key calls coalesced, others
              singleton. Group order follows first appearance.

        """
        groups: list[list[ToolCall]] = []
        by_key: dict[str, list[ToolCall]] = {}
        for call in calls:
            tool = self.tools_map.get(call.name)
            key = tool.serialize_key(call.args) if tool is not None else None
            if key is None:
                groups.append([call])
                continue
            group = by_key.get(key)
            if group is None:
                group = by_key[key] = []
                groups.append(group)
            group.append(call)
        return groups

    async def _run_tool_group_and_post(
        self,
        group: Sequence[ToolCall],
        *,
        parent_id: int = -1,
    ) -> None:
        """Run a serialized group of calls one at a time, in order.

        Each call's ``ToolResult`` is posted as it completes (via
        ``_run_tool_and_post``), so the cohort gate and tool_use/
        tool_result pairing still see one result per ``call_id``. A
        single coroutine awaiting each call in turn makes ordering
        deterministic -- no reliance on lock fairness. Cancellation
        (detach/kill) interrupts the in-flight call and skips the rest;
        their stubs are posted by the detach/preempt machinery.

        Args:
          group: Calls to run sequentially in submission order.
          parent_id: Originating assistant message id.

        """
        for call in group:
            await self._run_tool_and_post(call, parent_id=parent_id)

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
        if call.name == DETACHED_ARRIVED_TOOL:
            # ``DetachedArrived`` is not a real tool: the runtime synthesizes
            # turns with this name to deliver completed detached results. A
            # model can copy that pattern and emit a real ``DetachedArrived``
            # call (seen live). Pair it with an error result, but deliver that
            # result LAZILY: held until the next real turn so the bogus call
            # spends no model round of its own (there is no urgency to tell the
            # model it cannot do a thing it cannot do). ``hidden`` keeps the
            # correction out of the human's view; the model still sees it.
            logger.debug(
                "runtime detached-arrived mimic: call_id=%s parent_id=%s",
                call.id,
                parent_id,
            )
            self.inbox.push_back(
                LazyEvent(
                    payload=ToolResult(
                        call_id=call.id,
                        parent_id=parent_id,
                        # Terse on purpose; the model self-corrects. This short
                        # form is unproven, though. The longer form below is the
                        # one observed to work live -- revert to it if the short
                        # form ever fails:
                        #   f"{DETACHED_ARRIVED_TOOL} is not a tool you can call; "
                        #   "it is a runtime marker for a completed detached "
                        #   "tool's result. Detached results arrive automatically "
                        #   "-- do not call it."
                        content=f"{DETACHED_ARRIVED_TOOL} is not callable; detached "
                        "results arrive automatically.",
                        is_error=True,
                        hidden=True,
                    ),
                ),
            )
            return
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
                content=CANCELLED_PLACEHOLDER,
                is_error=True,
                kind=ToolResultKind.CANCELLED,
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
                DetachedResult(result=dataclasses.replace(result, call_id=call.id)),
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

        The compactor returns one :class:`ContextSplice` whose ref
        was minted via the ``mint_ref`` factory the runtime provided.
        The runtime is the sole appender: it stores the returned
        override on the tape and publishes ``CompactComplete``.

        Captures ``_compact_generation`` at spawn time and refuses to
        push terminal events when the generation has moved past it --
        a Halt/Clear bumping the generation and publishing CompactFailed
        must not be followed by a stale CompactComplete from a
        compactor task whose synchronous tail beat the cancellation.
        """
        generation = self._compact_generation
        if self.compactor is None:
            if generation == self._compact_generation:
                self.inbox.push_back(CompactComplete())
            return
        tape_len = len(self.tape)
        try:
            override = await self.compactor.compact(
                self.tape,
                self.context().messages,
                self.model,
                self.mint_ref,
                custom_instructions=args or None,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 -- compaction calls the model; catch-all routes UserFacingError to warning, others to exception
            log_exception_or_warning(logger, "compaction failed", exc)
            if generation == self._compact_generation:
                self.inbox.push_back(
                    CompactFailed(exception=exc, tape_len=tape_len),
                )
            return
        if generation != self._compact_generation:
            return
        override = widen_barrier_mask(override, self.tape)
        self.adopt_record(override)
        self.inbox.push_back(CompactComplete.from_override(override))
