"""AgentRuntime: inbox-driven event loop.

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

Cohort
~~~~~~

A tool cohort is ``set[str]`` of pending call IDs, populated when
``ModelResponseComplete`` carries tool calls. Model fires only when
the cohort is empty. No Cohort class.

Rebase model
~~~~~~~~~~~~

Provider APIs require a ``tool_result`` for every ``tool_use``. When
the user types mid-cohort, we stub unfinished tools with
``"[detached]"`` placeholders to close out the branch, then append
the user message. History stays linear. Detached tools post
``DetachedResult`` when they complete; these arrive as ``UserMessage``
context in the next round.

``UserQueuedMessage`` is a non-preempting variant: it buffers text
and coalesces into one ``UserMessage`` after the cohort completes,
without stubbing tools.

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

Agent composes the runtime::

    class Agent:
        def __init__(self, model, tools, ...):
            self.runtime = AgentRuntime(model=model, tools=tools)
            self.runtime.observers.append(self._on_event)

        async def run_forever(self):
            await self.runtime.run_forever()

        def halt(self):
            self.runtime.inbox.push_back(Halt())

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
            while not _gate_satisfied(items, gate, baseline):
                items.append(await self._queue.get())
                while not self._queue.empty():
                    try:
                        items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            self._gate = None
            self._gate_baseline = 0
        return items


def _gate_satisfied(
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
                        self.history.clear()
                        self.inbox.push_front(
                            AWAIT_USER,
                            *items[item_idx + 1 :],
                        )
                        break

                    case ModelResponseError(exception=exc):
                        self.model_call = None
                        self.history.append(
                            UserMessage(
                                text=f"[Error: {type(exc).__name__}: {exc}]",
                            ),
                        )
                        self.publish(item)
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
                        self._stub_running_tools_and_let_finish()
                        self.running_tools = {}
                        self.cohort.clear()
                        cohort_seen = False
                        self.history.append(item)
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
                        if msg.tool_calls:
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

                    case DetachedResult():
                        self.cohort.discard(item.call_id)
                        # Splice into the existing placeholder so the
                        # model sees the real result in the slot it
                        # already expects, not a phantom user message
                        # that triggers an extra round. Both
                        # ``[detached]`` (preempt) and ``[Running in
                        # background: ...]`` (explicit-bg) placeholders
                        # match by ``call_id``.
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

            if not self.cohort and queued:
                self.history.append(
                    UserMessage(
                        text="\n\n".join(q.text for q in queued),
                        attachments=sum(
                            (q.attachments for q in queued),
                            (),
                        ),
                    ),
                )
                queued.clear()

            if (
                not self.cohort
                and self.model_call is None
                and self.compact_task is None
                and self._should_call_model()
            ):
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

    def _should_call_model(self) -> bool:
        """Return True when history ends with content the model should answer."""
        if not self.history:
            return False
        return isinstance(self.history[-1], (UserMessage, ToolResult))

    def _stub_running_tools_and_let_finish(self) -> None:
        """Stub unfinished tools with placeholders. Tasks keep running."""
        for cid, task in self.running_tools.items():
            if not task.done():
                self.history.append(
                    ToolResult(call_id=cid, content="[detached]"),
                )
                self.detached[cid] = task

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
        except Exception as exc:
            logger.exception("model call failed")
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
        except Exception as exc:
            logger.exception("compaction failed")
            self.inbox.push_back(
                UserMessage(
                    text=f"[Compaction error: {type(exc).__name__}: {exc}]",
                ),
            )
