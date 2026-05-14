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
from typing import Protocol

import asyncio
import contextlib
import contextvars
import dataclasses
import itertools
import logging
import time

from sagent.custom_exceptions import log_exception_or_warning


_id_counter: itertools.count[int] = itertools.count()


def reset_id_counter(start: int) -> None:
    """Reset the SessionMessage id counter.

    Used by session resume: after loading history with persisted ids
    1..N, callers reset the counter to N+1 so newly created messages
    don't collide.

    Args:
      start: First id the counter will yield next.

    """
    global _id_counter  # noqa: PLW0603 -- module-level counter requires global statement
    _id_counter = itertools.count(start)


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


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class BytesMessage:
    """Binary payload (image, PDF)."""

    data: bytes
    """Raw bytes of the payload."""

    descriptor: str
    """MIME-style content type (e.g. ``image/png``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionMessage:
    """Common fields for all history entries."""

    id: int = dataclasses.field(default_factory=lambda: next(_id_counter))
    """Monotonically increasing per-session message id."""

    parent_id: int = -1
    """Id of the message this one responds to, or ``-1``."""

    timestamp: float = dataclasses.field(default_factory=time.monotonic)
    """Monotonic clock seconds when the message was created."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    """Provider-assigned call id (e.g. ``toolu_01...``)."""

    name: str
    """Tool name the model wants to invoke."""

    args: Mapping[str, object]
    """Parsed directive arguments."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(SessionMessage):
    """User or system text the model should see."""

    text: str
    """Plain-text content."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads sent alongside the text."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage(SessionMessage):
    """Model response: text and/or tool calls."""

    text: str = ""
    """User-visible response text."""

    thinking_blocks: tuple[Mapping[str, object], ...] = ()
    """Provider thinking blocks (opaque dicts)."""

    tool_calls: tuple[ToolCall, ...] = ()
    """Tool invocations requested by the model."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(SessionMessage):
    """Result of one tool invocation."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    content: str
    """Result text shown to the model."""

    is_error: bool = False
    """True when the tool raised or signalled failure."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads produced by the tool."""

    diff: str = ""
    """Unified-diff fragment for renderers (e.g. Edit/Write)."""

    diff_file_path: str = ""
    """Absolute path the ``diff`` applies to."""

    hint: str = ""
    """Optional follow-up nudge surfaced to the model."""

    summary: str = ""
    """Optional short post-execution receipt line."""


type HistoryEntry = UserMessage | AssistantMessage | ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class UserQueuedMessage:
    """User context that doesn't preempt. Waits for cohort to finish."""

    text: str
    """Plain-text content to merge into the next ``UserMessage``."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads to merge alongside ``text``."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Quit:
    """Shut down the agent."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Halt:
    """Cancel model call, wait for user. Tools keep running."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Clear:
    """Detach tools, wipe history, wait for user."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Kill:
    """Cancel one or all tool tasks."""

    call_id: str | None = None
    """Specific call to cancel, or ``None`` to cancel all."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Detach:
    """Stub one or all tools, let them finish in background."""

    call_id: str | None = None
    """Specific call to detach, or ``None`` to detach all."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Undetach:
    """Re-gate model on a detached tool's completion."""

    call_id: str | None = None
    """Specific call to re-gate on, or ``None`` for all."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitch:
    """Queue a model swap; the runtime applies it once safe.

    The slash handler builds the new model synchronously and packages
    the swap as ``apply``. The runtime defers the call until
    ``model_call`` and ``compact_task`` are both ``None`` so the
    in-flight response finishes against the OLD model (cost
    attribution, retry state, etc. stay self-consistent) and only the
    NEXT call uses the new model.
    """

    apply: Callable[[], None]
    """Closure that performs the swap (typically ``agent.swap_model``)."""

    label: str = ""
    """Optional human-readable label shown to renderers (e.g. ``old -> new``)."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelCallStarted:
    """Model streaming call has been spawned."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponsePartial:
    """Streaming text chunk from the model."""

    text: str
    """Newly arrived text chunk."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseThinking:
    """Streaming thinking chunk from the model."""

    text: str
    """Newly arrived thinking chunk."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseComplete:
    """Model finished streaming."""

    message: AssistantMessage
    """Final assembled ``AssistantMessage``."""

    input_tokens: int = 0
    """Input token count reported by the provider."""

    output_tokens: int = 0
    """Output token count reported by the provider."""

    cache_creation_tokens: int = 0
    """Tokens spent creating cache breakpoints."""

    cache_read_tokens: int = 0
    """Tokens served from cache."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseCancelled:
    """Model call was cancelled mid-stream."""

    output_chars_estimate: int = 0
    """Approximate chars streamed before cancel."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseError:
    """Unrecoverable failure (creds expired, retries exhausted)."""

    exception: BaseException
    """The raised exception surfaced to observers."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelIdle:
    """Model finished with no tool calls."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CohortStarted:
    """Tool cohort has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultPartial:
    """Streaming chunk from a tool (e.g. Bash long output)."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Newly arrived output chunk."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachedResult:
    """A previously-detached tool completed."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    content: str
    """Result text from the completed tool."""

    is_error: bool = False
    """True when the tool raised or signalled failure."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEntryUpdated:
    """An existing history entry was mutated in-place (splice).

    Publish-only. Emitted when ``DetachedResult`` (or its in-batch
    sibling) splices real tool output into a ``[detached]`` placeholder
    via ``dataclasses.replace``. Same ``id``, new ``content``.

    Without this event the persistence observer never learns of the
    update -- its delta-based append (``history[persisted_len:]``)
    can't see in-place mutations, so resumed sessions would load the
    stale placeholder. The persistence observer listens for this event
    and re-emits the entry; the loader dedupes by ``id`` so last-write-
    wins.
    """

    entry: HistoryEntry
    """The updated history entry (carries id + new content)."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CohortComplete:
    """All tool results for the current cohort have arrived."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Compact:
    """Trigger context compaction."""

    args: str = ""
    """Free-form compaction instructions for the compactor."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Recompact:
    """Reload last pre-compact transcript and re-run compaction."""

    args: str = ""
    """Free-form compaction instructions for the compactor."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CompactStarted:
    """Compaction task has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactComplete:
    """Compaction finished; splice summary into history."""

    summary: list[HistoryEntry]
    """Compactor output to install at the head of history."""

    snapshot_len: int
    """History length captured before compaction; entries appended after
    that index are preserved post-splice."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class SaveSession:
    """Signals observers to persist session state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusChanged:
    """Agent ``status`` field was updated.

    Publish-only: emitted by ``Agent.status`` setter. Renderers update
    the terminal title, persistence observers re-flush ``meta`` so the
    new status survives a crash even without a history delta.
    """

    text: str
    """New status string."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolLabel:
    """Pre-execution label for a tool call (REPL rendering).

    Publish-only: tool wrappers fan this out via ``runtime.publish``
    before invoking the inner tool. Never enters the inbox, never
    hits the match block.
    """

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Short label rendered above the call (e.g. ``Read(path)``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvent:
    """Wrapped event from a child agent (AgentSpawn forwarding).

    Publish-only: AgentSpawn's observer wraps each child event and
    fans it out on the parent's runtime. Never enters the inbox.
    """

    label: str
    """Child agent's display label."""

    inner: RuntimeEvent
    """The forwarded child event."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildDoneEvent:
    """Child agent completed; carries totals for the gutter.

    Publish-only: AgentSpawn emits one at completion. Never enters
    the inbox.
    """

    label: str
    """Child agent's display label."""

    elapsed: float
    """Wall-clock seconds the child ran."""

    tokens: int
    """Total tokens (input + output) the child consumed."""

    cost: float
    """Total USD cost attributable to the child."""


type RuntimeEvent = (
    Quit
    | Halt
    | Clear
    | Kill
    | Detach
    | Undetach
    | UserMessage
    | UserQueuedMessage
    | ModelSwitch
    | ModelCallStarted
    | ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelIdle
    | CohortStarted
    | ToolResultPartial
    | ToolResult
    | DetachedResult
    | HistoryEntryUpdated
    | CohortComplete
    | Compact
    | Recompact
    | CompactStarted
    | CompactComplete
    | SaveSession
    | StatusChanged
    | ToolLabel
    | ChildEvent
    | ChildDoneEvent
)


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Await:
    """Drain gate: block until an item matching ``types`` arrives."""

    types: tuple[type, ...]
    """Event classes that satisfy the gate (``Quit`` always does)."""


AWAIT_USER = Await((UserMessage, Quit))


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
            while not self._gate_satisfied(items, gate, baseline):
                items.append(await self._queue.get())
                while not self._queue.empty():
                    try:
                        items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
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
        cohort_seen: bool = False
        queued: list[UserQueuedMessage] = []

        while True:
            self._collect_detached()
            items = await self.inbox.drain()

            for item_idx, item in enumerate(items):
                match item:
                    case Quit():
                        if self.model_call:
                            self.model_call.cancel()
                        if self.compact_task:
                            self.compact_task.cancel()
                        for t in self.running_tools.values():
                            t.cancel()
                        return

                    case Halt():
                        if self.model_call:
                            self.model_call.cancel()
                            self.model_call = None
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
                        break

                    case Clear():
                        if self.model_call:
                            self.model_call.cancel()
                            self.model_call = None
                        self._stub_running_tools_and_let_finish()
                        self.running_tools = {}
                        self.cohort.clear()
                        cohort_seen = False
                        queued.clear()
                        self._mid_stream_queue.clear()
                        self.history.clear()
                        self.inbox.push_front(
                            AWAIT_USER,
                            *items[item_idx + 1 :],
                        )
                        break

                    case ModelResponseError(exception=exc):
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
                        break

                    case Kill(call_id=cid):
                        if cid is None:
                            for t in self.running_tools.values():
                                t.cancel()
                            self.running_tools = {}
                            self.cohort.clear()
                            cohort_seen = False
                        elif cid in self.running_tools:
                            self.running_tools.pop(cid).cancel()
                            self.cohort.discard(cid)

                    case Detach(call_id=cid):
                        if cid is None:
                            self._stub_running_tools_and_let_finish()
                            self.running_tools = {}
                            self.cohort.clear()
                            cohort_seen = False
                        elif cid in self.running_tools:
                            task = self.running_tools.pop(cid)
                            self.cohort.discard(cid)
                            self.history.append(
                                ToolResult(
                                    call_id=cid,
                                    content="[detached]",
                                ),
                            )
                            self.detached[cid] = task

                    case Undetach(call_id=cid):
                        if cid is None:
                            for did in self.detached:
                                self.cohort.add(did)
                        elif cid in self.detached:
                            self.cohort.add(cid)

                    case Compact(args=args) | Recompact(args=args):
                        if self.compact_task and not self.compact_task.done():
                            continue
                        if self.model_call:
                            self.model_call.cancel()
                            self.model_call = None
                        self._stub_running_tools_and_let_finish()
                        self.running_tools = {}
                        self.cohort.clear()
                        cohort_seen = False
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
                            self._stub_running_tools_and_let_finish()
                            self.running_tools = {}
                            self.cohort.clear()
                            cohort_seen = False
                            self._append_or_coalesce_user(item)
                            self.publish(item)

                    case UserQueuedMessage():
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
                                self.detached[tc.id] = asyncio.create_task(
                                    self._run_tool_and_post(tc, parent_id=msg.id),
                                )
                            # Commit mid-stream input to history and
                            # publish the coalesced bar -- the pending
                            # preview drops as the buffer empties and
                            # the bar appears in console, single UI
                            # transition.
                            coalesced = self._drain_mid_stream_queue()
                            if coalesced is not None:
                                self.publish(coalesced)
                        elif msg.tool_calls:
                            cohort_seen = True
                            self.publish(CohortStarted())
                            for tc in msg.tool_calls:
                                self.cohort.add(tc.id)
                                self.running_tools[tc.id] = asyncio.create_task(
                                    self._run_tool_and_post(tc, parent_id=msg.id),
                                )
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
                        # ``_stub_running_tools_and_let_finish`` reclassified
                        # its call_id as detached. The ``ToolResult`` was
                        # pushed as a regular result (because the task
                        # ran ``self.detached`` check before the stub),
                        # but a peer item in the same drain batch (e.g.
                        # a tool-pushed ``UserMessage``) triggered a
                        # preempt that cleared the cohort. Without this
                        # case the result would fall through to ``_`` and
                        # leave the assistant's ``tool_use`` paired only
                        # with the ``[detached]`` placeholder forever.
                        # Splice the real content into the placeholder
                        # exactly like ``DetachedResult`` does.
                        del self.detached[cid]
                        for i, prior in enumerate(self.history):
                            if isinstance(prior, ToolResult) and prior.call_id == cid:
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
                                    text=(f"[Detached tool {item.call_id} completed]"),
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
                pending.apply()
                self.publish(pending)

            if not self.cohort and cohort_seen:
                self.publish(CohortComplete())
                cohort_seen = False

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
                # ``inbox.gate_armed`` blocks firing while ``AWAIT_USER``
                # is pending (armed by ``Halt`` / ``ModelResponseError``).
                # Without this guard the model would fire on the stale
                # ``UserMessage`` still at history.tail, treating the
                # cancellation as a retry rather than waiting for the
                # user's next input.
                self.model_call = asyncio.create_task(
                    self._stream_and_post(),
                )
                self.publish(ModelCallStarted())

            self.publish(SaveSession())

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

    def _stub_running_tools_and_let_finish(self) -> None:
        """Stub all running tools with placeholders; route results via detached.

        Appends ``[detached]`` placeholders for EVERY ``running_tools``
        entry, including tasks that have already completed and pushed
        their ``ToolResult`` to the inbox but whose result has not yet
        drained. The earlier ``if not task.done()`` guard skipped done
        tasks under the false assumption that their results were
        already in history -- they're in the inbox, which the caller
        then prevents from landing properly by clearing ``cohort`` and
        ``running_tools``. The dropped result would leave the
        assistant's ``tool_use`` block in history with no adjacent
        ``tool_result``, breaking Anthropic alternation.

        Moving the call_id into ``self.detached`` reroutes any
        late-arriving ``ToolResult`` (whether already-in-inbox or
        still-running) through the ``DetachedResult`` splice path so
        the real content replaces the ``[detached]`` placeholder when
        it arrives.
        """
        for cid, task in self.running_tools.items():
            self.history.append(
                ToolResult(call_id=cid, content="[detached]"),
            )
            self.detached[cid] = task

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
            result = await tool.run(call.args)
            replacements: dict[str, object] = {}
            if not result.call_id:
                replacements["call_id"] = call.id
            if result.parent_id == -1 and parent_id != -1:
                replacements["parent_id"] = parent_id
            if replacements:
                result = dataclasses.replace(result, **replacements)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 -- surface arbitrary tool errors
            result = ToolResult(
                call_id=call.id,
                parent_id=parent_id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        finally:
            current_call_id_var.reset(call_token)
        if call.id in self.detached:
            del self.detached[call.id]
            self.inbox.push_back(
                DetachedResult(
                    call_id=call.id,
                    content=result.content,
                    is_error=result.is_error,
                ),
            )
        else:
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
                UserMessage(
                    text=f"[Compaction error: {type(exc).__name__}: {exc}]",
                ),
            )
