"""The agent's event loop: one inbox, one ``match``, one model-call gate.

``Runtime.run_forever`` drains its inbox, dispatches each event, then calls the
model when the context ends in something to answer and nothing is in flight: no
open cohort, no model call, no compaction, and no wait for the user.

Tool calls run in one of two registries:

- ``cohort``: the calls the next model call waits on.
- ``detached``: calls already answered by a stub; each one's real result is
  delivered later as a new ``BackgroundTask`` foreground turn, so nothing the model has
  read is rewritten.

A call goes to ``detached`` when the model asks for ``background`` or
``delay``, when it runs past ``detach_after_sec``, or when the user interrupts
its cohort: a message, ``Compact``, or ``Clear``. ``Kill`` cancels calls in
either. A call the model waits on holds what it touches (its ``serialize_key``),
and a later call contending for that waits; detached, it lets go, as nothing
waits on it any more. A killed call holds on until it has stopped.

What the user sends while the model streams is answered next: the reply's calls
start, detached, and their results arrive later.

``Halt`` stops the model and waits for the user's next message. Meanwhile the
inbox holds every event back except the verbs, ``Halt``, ``Kill``, ``Clear``,
and ``Compact``, and the compaction they start.

A failed model call stops the turn one of two ways. A failure that may pass
(``RetryableError``, under ``persistent_retry``) suspends it: ``ModelCallSuspended``,
then the call again at the advised time. Any other ends it: ``ModelResponseError``,
then a wait for the user. Either way, the user's message, a new model, or a
``Retry`` (as after a sign-in) calls again at once.

The history is a tape that only grows (``tape.py``); ``context.py`` resolves
what the model sees of it. Providers need user and model turns to alternate,
and each call answered before the next turn: the runtime merges adjacent
user-side turns, and repairs at the model-call gate what it cannot keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import asyncio
import dataclasses
import itertools
import logging
import time

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.exceptions import ContextOverflowError, RetryableError
from sagent.model.types import Request, Response, TextChunk

# Engine consumes a subset of the runtime event vocabulary. No re-export shim.
from sagent.runtime.context import (
    InvalidContextError,
    ResolvedContext,
    alive_splices,
    resolve_context,
    validate_context,
)
from sagent.runtime.exceptions import (
    describe,
    log_exception_or_warning,
    log_task_exception,
)
from sagent.runtime.tape import (
    ContextSplice,
    InvalidPayloadError,
    InvalidSpliceError,
    MaskRange,
    ReferrableTapeEvent,
    TapeEvent,
    TapeRecord,
    TapeRef,
    calls,
    coalesce_roles,
    full_tape_mask,
    mask_contains_ref,
    mask_ranges_overlap,
    splice_safe_repair,
    unpaired_call_ids,
)
from sagent.runtime.types import (
    AgentIdle,
    AgentSendDeferredMessage,
    AgentSendQueuedMessage,
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
    Job,
    Kill,
    LazyEvent,
    Markers,
    ModelCallStarted,
    ModelCallSuspended,
    ModelContextEvent,
    ModelIdle,
    ModelResponseCancelled,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    Nudge,
    Quit,
    Retry,
    RuntimeEvent,
    SaveSession,
    Scheduled,
    UserDeferredMessage,
    UserQueuedMessage,
)
from sagent.session.types import (
    AgentStatusResult,
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    ShellCommandResult,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
    WebFetchResult,
    WebSearchResults,
)
from sagent.tools.background_task import (
    BackgroundTask,
    background_delay,
    backgroundable,
    offered,
    without_controls,
)
from sagent.tools.bash import Bash
from sagent.tools.command_analyzer import Rewrite, drop_cd, moves
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Awaitable,
        Callable,
        Iterator,
        Mapping,
        Sequence,
    )

    from sagent.providers.types import Model
    from sagent.session.types import SessionRecord
    from sagent.tools.types import Tool


logger = logging.getLogger(__name__)


def widen_barrier_mask(
    override: ContextSplice,
    tape: Sequence[TapeRecord],
) -> ContextSplice:
    """Return ``override`` with its mask widened over ``tape``, keeping its gaps.

    A sparse mask such as ``((r5, r5), (r10, r10))`` stays sparse over ``r6..r9``.

    Args:
      override: A compactor's barrier splice.
      tape: The tape it is adopted onto.

    Returns:
      widened: ``override``, masking every record of ``tape`` outside its gaps.

    """
    mask = _widen_mask_ranges(override.mask, tape)
    if mask == override.mask:
        return override
    return dataclasses.replace(override, mask=mask)


@dataclass(frozen=True, slots=True, kw_only=True)
class Await:
    """Drain gate: block until an item matching ``types`` arrives."""

    types: tuple[type, ...]
    """Event classes that satisfy the gate (``Quit`` always does)."""

    passes: tuple[type, ...] = ()
    """Event classes handed over at once, the gate still set."""

    @classmethod
    def user(cls) -> Await:
        """Return the wait for the user's next message.

        A verb acts meanwhile, and so does the compaction it starts; only the
        user's message lets the model answer again.

        Returns:
          gate: Released by a user-side message or ``Quit``.

        """
        return cls(
            types=(
                UserMessage,
                UserQueuedMessage,
                UserDeferredMessage,
                AgentToAgentMessage,
                AgentSendQueuedMessage,
                AgentSendDeferredMessage,
                Nudge,
                Quit,
            ),
            passes=(Halt, Kill, Clear, Compact, CompactComplete, CompactFailed),
        )

    @classmethod
    def failure(cls) -> Await:
        """Return the wait after a failed model call ended the turn.

        Returns:
          gate: Released as ``user`` is, and by a ``Retry``: a new model or a
            sign-in may get past the failure.

        """
        user = cls.user()
        return cls(types=(*user.types, Retry), passes=user.passes)

    @classmethod
    def recovery(cls) -> Await:
        """Return the wait after a context the gate cannot repair.

        Returns:
          gate: Released also by a verb that can repair the context.

        """
        return cls(types=(*cls.user().types, Clear, Compact, Halt, Kill))


class GatedDeque[T]:
    """An async queue whose drain can wait for a kind of item.

    A gate holds the queue until an item of its ``types`` arrives that was not
    queued when the gate was set, or a ``Quit``. Counting only new ones keeps a
    ``Halt`` waiting for a fresh message when the user typed several while the
    agent was busy.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._gate: tuple[type, ...] | None = None
        self._gate_baseline: int = 0
        self._passes: tuple[type, ...] = ()

    @property
    def gate_armed(self) -> bool:
        """True while a gate holds the queue."""
        return self._gate is not None

    def empty(self) -> bool:
        """Return whether nothing is queued.

        Returns:
          empty: Whether the queue is empty now.

        """
        return self._queue.empty()

    def push_back(self, item: T) -> None:
        """Add to back of queue.

        Args:
          item: Item to enqueue at the tail.

        """
        self._queue.put_nowait(item)

    def drain_nowait(self) -> list[T]:
        """Return all currently queued items without waiting.

        Returns:
          items: All items currently queued at call time.

        """
        # One task owns the queue, so ``empty()`` cannot go stale before the get.
        items: list[T] = []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
        return items

    def push_front(self, *items: T | Await) -> None:
        """Queue ``items`` at the front, in order; an ``Await`` among them sets the gate.

        Args:
          items: What to queue. An ``Await`` must come first: the gate counts
            only the items queued before the call.

        Raises:
          ValueError: An ``Await`` is not first.

        """
        await_idx = next(
            (i for i, item in enumerate(items) if isinstance(item, Await)),
            -1,
        )
        if await_idx > 0:
            raise ValueError(
                f"GatedDeque.push_front precondition violated: Await must be the "
                f"first argument when present; got Await at index {await_idx} with "
                f"{await_idx} non-Await item(s) ahead of it",
            )
        old = self.drain_nowait()
        for item in items:
            if isinstance(item, Await):
                self._gate = item.types
                self._passes = item.passes
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
        arm time) or until any ``Quit`` is observed. Items the gate
        passes are returned as they arrive, the gate still set, and the
        rest keep their order behind it.

        Returns:
          items: All queued items in arrival order.

        """
        first = await self._queue.get()
        items = [first, *self.drain_nowait()]
        if self._gate is None:
            return items
        gate = self._gate
        baseline = self._gate_baseline
        while not self._gate_satisfied(items, gate, baseline):
            passing = [i for i in items if isinstance(i, self._passes)]
            if passing:
                for item in items:
                    if not isinstance(item, self._passes):
                        self._queue.put_nowait(item)
                return passing
            items.append(await self._queue.get())
            items.extend(self.drain_nowait())
        logger.debug(
            "runtime inbox gate released: gate=%s baseline=%d buffered=%s",
            _type_names(gate),
            baseline,
            _item_names(items),
        )
        # Arming sets the baseline and the passes afresh; unarmed, neither is read.
        self._gate = None
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


@runtime_checkable
class Compactor(Protocol):
    """What replaces the context with a splice the runtime appends to its tape."""

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Return a barrier splice for the current context.

        A payload keeping a call whose result lies outside it, or a result
        whose call does, names that id in ``paired_externally``;
        :func:`~sagent.runtime.tape.unpaired_call_ids` computes it.

        Args:
          tape: The tape.
          context: What the model sees of it.
          model: The model to summarize with.
          mint_ref: Returns a fresh ``TapeRef``.
          custom_instructions: What the summary must keep, such as the hint
            after ``/compact``.

        Returns:
          splice: The barrier splice.

        """
        ...


@runtime_checkable
class Pruning(Protocol):
    """What drops old tool results before a model call, when that costs less."""

    def prune(self) -> bool:
        """Drop what pays to drop, now; return whether anything was."""
        ...


@runtime_checkable
class Sizing(Protocol):
    """What says when the context is due to shrink, and how long a reply may run."""

    def should_run(self) -> bool:
        """Return whether to shrink the context before the next model call."""
        ...

    def reply_room(self) -> int:
        """Return the most the next reply may hold; ``0`` for the model's own limit."""
        ...


class Runtime:
    """Inbox-driven event loop agent.

    Args:
      model: What answers; assign ``runtime.model`` to switch.
      tools: What the model may call; names must differ.
      compactor: What ``Compact`` runs; ``None`` compacts nothing.
      session_id: Namespace of this runtime's tape refs.
      tool_context: Where tools run, and what the model has read.
      markers: What the runtime writes for the calls it answers itself.

    """

    def __init__(
        self,
        *,
        model: Model,
        tools: Sequence[Tool] | None = None,
        compactor: Compactor | None = None,
        session_id: str = "",
        tool_context: ToolContext | None = None,
        markers: Markers | None = None,
    ) -> None:
        self._model: Model = model
        self.markers: Markers = markers if markers is not None else Markers()
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            if t.name in self.tools:
                raise ValueError(f"Duplicate tool name: {t.name!r}")
            self.tools[t.name] = t
        # The system prompt's blocks, sent with every model call.
        self.system: tuple[str, ...] = ()
        # The tools the model may call; ``None`` for all. Every tool is still
        # offered, so agents with different gates send the same cached prefix.
        self.allowed: frozenset[str] | None = None
        self.compactor = compactor
        # Whether the compactor wants to run before the next model call.
        self.compact_due: Callable[[], bool] | None = None
        # The most the next reply may hold, as the window leaves room; ``0`` or
        # ``None`` asks the model's own limit.
        self.reply_room: Callable[[], int] | None = None
        # Automatic compactions since the last model reply; the circuit breaker.
        self._auto_compactions = 0
        # The conversation vendors are told of, and this agent's place in it, as
        # ``Request`` names them; ``session_id`` namespaces tape refs and is empty
        # for a live agent.
        self.conversation = ""
        self.agent = ""
        self.parent = ""
        self.parent_turn = 0
        self.root_prompt = ""
        self.name = "/root"
        self.turn_started_ms = 0
        # Calls made this turn.
        self._steps = 0
        self.compactions = 0
        self.turns = 0
        # What started the running compaction, or the last one until the next
        # main call reads it.
        self._compaction: Literal["", "auto", "manual"] = ""
        self.session_id = session_id
        self.tape: list[TapeRecord] = []
        self._next_ordinal: int = 0
        self._cached_resolved: ResolvedContext | None = None
        self._tape_by_ref: dict[TapeRef, TapeRecord] = {}
        # By call id: the record holding its latest result, and the model turn
        # that made it, which a deferred pairing is inserted after.
        self._placeholder_refs: dict[str, TapeRef] = {}
        self._parent_assistant_refs: dict[str, TapeRef] = {}
        self.inbox: GatedDeque[RuntimeEvent] = GatedDeque()
        self.detached: dict[str, asyncio.Task[None]] = {}
        self.observers: list[Callable[[RuntimeEvent], None]] = []
        # Asked with each reply before its calls run: a message in answer is
        # answered next, as one typed mid-stream is.
        self.before_tool_spawn: Callable[[Response], UserMessage | None] | None = None
        # Asked once per model round, as it is confirmed to fire, for the notes
        # the Agent layer has to add first. Each is its own entry, no one's turn,
        # so it can never fire a round of its own.
        self.context_updates: (
            Callable[[], Sequence[SystemMessage | TurnContext]] | None
        ) = None
        # Asked before each model round, ahead of compaction: drops what the
        # context no longer needs when that pays.
        self.prune: Callable[[], bool] | None = None
        self.running_tools: dict[str, asyncio.Task[None]] = {}
        self.cohort: set[str] = set()
        # A cohort started and not preempted, so its draining is published.
        self._cohort_seen: bool = False
        self.model_call: asyncio.Task[None] | None = None
        # Bumped as a model call starts or is cancelled, and as a compaction is
        # cancelled: the late reply or completion of an older one is dropped.
        self._model_call_generation: int = 0
        self.compact_task: asyncio.Task[None] | None = None
        self._compact_generation: int = 0
        # What the user sent while the model streamed; it joins the context once
        # the reply ends, and the reply's calls run on detached.
        self._mid_stream_queue: list[UserMessage | AgentToAgentMessage] = []
        # What waits for the next model round to be committed (``_flush_pending``).
        self._pending_commits: list[_PendingCommit] = []
        # Calls whose result was delivered forward, or is queued to be: a second
        # delivery would give two arrival turns one call id.
        self._forwarded_call_ids: set[str] = set()
        # The next ``<markers.mimic>N`` for a ``DetachedArrived`` call the model
        # makes itself; past every one the tape holds.
        self._mimic_counter: int = 0
        # Wall-clock seconds before which a rate-limited model call waits.
        self.service_suspended_until: float | None = None
        # The wake-up armed for the end of that suspension.
        self._retry_timer: asyncio.TimerHandle | None = None
        # Retryable failures in a row; what the next wait backs off from.
        self._retries = 0
        # Whether a failed call ended the turn, so it waits; a new model, or a
        # ``Retry``, may get past the failure, and so calls again.
        self._failed = False
        # The model the call in flight asked; a failure of one switched away from
        # says nothing of the model now in use.
        self._called_model: Model = model
        # Whether a long rate limit is waited out rather than handed to the user.
        self.persistent_retry = False
        # Where tools run, and what the model has read; one per agent.
        self.tool_context = tool_context or ToolContext(cwd=Path.cwd())
        # Each running call, and when it started, for ``jobs``.
        self._started: dict[str, tuple[ToolCall, float]] = {}
        # Bash calls that are other tools' work but no one call: the refusal.
        self._refusals: dict[str, str] = {}
        # Characters the model call in flight has streamed so far.
        self._streamed_chars = 0
        # ``Scheduled`` work waiting for a safe point.
        self._scheduled: list[Scheduled] = []
        # Whether a call may ask to run in the background, or after a delay.
        self.allow_background = True
        # When each delayed background call may start, until it does.
        self._wake_at: dict[str, float] = {}
        # Callers waiting on a background call's result instead of its delivery.
        self._foreground_waiters: dict[str, list[asyncio.Future[AnyToolResult]]] = {}
        # How long a call the model waits on may run before it is answered by a stub
        # and left to finish, its result delivered later; ``None`` waits for each.
        self.detach_after_sec: float | None = None
        # What each group of calls the model waits on touches, in the order they
        # started: a later call contending for any of it waits.
        self._holding: list[_Hold] = []
        # Each such group's hold, by its task, so detaching the group lets go.
        self._holds: dict[asyncio.Task[None], _Hold] = {}

    # Background work does not hold the model back: the gate waits on the cohort only.
    @property
    def _engine_quiescent(self) -> bool:
        """True when no streaming, compaction, or open tool batch is in flight."""
        return not self.cohort and self.model_call is None and self.compact_task is None

    @property
    def _ready_to_advance(self) -> bool:
        """True when the engine is quiescent and the inbox waits for nothing."""
        return self._engine_quiescent and not self.inbox.gate_armed

    @property
    def model(self) -> Model:
        """What answers; another assigned retries the turn a failure stopped."""
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        changed = model is not self._model
        self._model = model
        if changed:
            self.retry()

    @property
    def stopped(self) -> bool:
        """True while a failed model call holds the turn: suspended, or failed."""
        return self._failed or self.service_suspended_until is not None

    @property
    def is_idle(self) -> bool:
        """True iff a user-side message pushed now would be answered now.

        Background work does not count: it never holds the model back.
        """
        return (
            self._ready_to_advance
            and not self.running_tools
            and not self._mid_stream_queue
        )

    @property
    def accepts_user_dispatch(self) -> bool:
        """True iff a user message pushed now makes progress, rather than being staged.

        It does unless a model call streams or a compaction runs. While tools
        run, the message interrupts them. While the runtime waits for the user,
        it is what the runtime waits for. And while a ``Halt`` is queued but
        not yet taken, the stream about to stop must not strand it.
        """
        if self.inbox.gate_armed:
            return True
        if self.model_call is None and self.compact_task is None:
            return True
        return self.model_call is not None and not self.inbox.empty()

    @property
    def accepts_deferred_dispatch(self) -> bool:
        """True iff a deferred message pushed now makes progress.

        As :attr:`accepts_user_dispatch`, but never while tools run: a
        deferred message waits for them rather than interrupting them.
        """
        return self.accepts_user_dispatch and not self.cohort and not self.running_tools

    # A queued pairing or ride-along does not count: it waits for a round that real
    # content drives, so waiting on it would wedge the loop. A forward delivery does:
    # its call is already taped, so idling first would persist an unanswered call.
    def _fully_drained(self) -> bool:
        """Return True iff the agent has no work to do and no gate is armed."""
        return (
            self.is_idle
            and self.inbox.empty()
            and not self.detached
            and not self._has_waking_commit()
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

    def push(self, event: RuntimeEvent) -> None:
        """Queue ``event``; the loop takes it at its next drain.

        Args:
          event: An input, such as a ``UserMessage``, ``Halt``, or ``Compact``.

        """
        self.inbox.push_back(event)

    def halt(self) -> None:
        """Stop the model call; tool calls keep running."""
        self.push(Halt())

    # Only while stopped: a ``Retry`` queued otherwise reads as input waiting
    # behind the stream (``accepts_user_dispatch``) and as a wake to a watcher.
    def retry(self) -> None:
        """Call the model now for the turn a failure stopped, as after a sign-in."""
        if self.stopped:
            self.push(Retry())

    def _end_suspension(self) -> None:
        """Let the next model call go now, its backoff started afresh."""
        self._retries = 0
        self.service_suspended_until = None
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    async def schedule(self, work: Callable[[], Awaitable[None]]) -> None:
        """Run ``work`` at the next safe point, between turns, and wait for it.

        While a ``Halt``, ``Clear``, or model error has the runtime waiting for
        the user, the work waits too.

        Args:
          work: What to run once nothing is streaming, compacting, or gating.

        """
        done = asyncio.Future[None]()
        self.push(Scheduled(work=work, done=done))
        await done

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        """Return every event published from the first ``__anext__`` on."""
        return self._events()

    @property
    def jobs(self) -> Mapping[str, Job]:
        """Tool calls still running, by call id."""
        now = time.time()
        return {
            cid: _RunningJob(
                call=call,
                started=started,
                state="sleeping"
                if self._wake_at.get(cid, now) > now
                else "detached"
                if cid in self.detached and cid not in self.cohort
                else "running",
                runtime=self,
            )
            for cid, (call, started) in self._started.items()
        }

    async def foreground(self, call_id: str) -> AnyToolResult | None:
        """Wait for a detached call's result, and take it instead of its delivery.

        Args:
          call_id: The call, as ``jobs`` names it.

        Returns:
          result: Its result, once it has one; ``None`` for no such call.

        """
        for commit in self._pending_commits:
            entry = commit.entry
            if commit.kind == "forward" and entry.call_id == call_id:
                self._pending_commits.remove(commit)
                return entry
        delivered = f"{call_id}{self.markers.arrival}"
        for entry in self.context().messages:
            if isinstance(entry, ToolResult) and entry.call_id == delivered:
                return dataclasses.replace(entry, call_id=call_id)
        if call_id not in self.detached:
            return None
        waiter = asyncio.Future[AnyToolResult]()
        waiters = self._foreground_waiters.setdefault(call_id, [])
        waiters.append(waiter)
        try:
            return await waiter
        finally:
            waiters.remove(waiter)
            if not waiters and self._foreground_waiters.get(call_id) is waiters:
                del self._foreground_waiters[call_id]

    # A resumed context's detached calls ran in a process that has ended, so no
    # result will follow; and a clean quit's cancellations are never committed.
    def settle_detached(self) -> None:
        """Deliver a cancellation for each call answered only by a stub, running nowhere."""
        pending: list[str] = []
        answered: set[str] = set()
        for entry in self.context().messages:
            if not isinstance(entry, ToolResult):
                continue
            if entry.stub == "pending":
                pending.append(entry.call_id)
            else:
                answered.add(entry.call_id.removesuffix(self.markers.arrival))
        for cid in pending:
            if cid in answered or cid in self.detached or cid in self.running_tools:
                continue
            self._append_detached_arrival(
                UncategorizedToolResult(
                    call_id=cid,
                    content=self.markers.cancelled,
                    is_error=True,
                    stub="cancelled",
                ),
            )

    # A resumed window may end with something the last process never answered: a
    # message, a result, a job it left running. That is as stale as a halted turn's,
    # so it waits for the user's next message, as after a halt; any other wake-up,
    # such as closing a model the resume replaced, must not answer it.
    def hold_for_user(self) -> None:
        """Wait for the user's next message if the history ends with one to answer."""
        if self._should_call_model():
            self.inbox.push_front(Await.user())

    async def _events(self) -> AsyncIterator[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        observer = queue.put_nowait
        self.observers.append(observer)
        try:
            while True:
                yield await queue.get()
        finally:
            self.observers.remove(observer)

    async def _run_scheduled(self, scheduled: Scheduled) -> None:
        """Run one ``Scheduled`` work, resolving its future either way."""
        # A caller that stopped waiting no longer wants the work done.
        if scheduled.done.cancelled():
            return
        try:
            await scheduled.work()
        except asyncio.CancelledError:
            # Only the loop's own cancellation goes on up. The work's own -- it
            # awaited something that was cancelled -- ended the loop with it,
            # and the caller waited forever on a future nothing would resolve.
            loop_task = asyncio.current_task()
            assert isinstance(loop_task, asyncio.Task)
            if loop_task.cancelling():
                raise
            _ = scheduled.done.cancel()
        except Exception as exc:  # noqa: BLE001 -- The caller awaiting ``done`` is told; the loop keeps dispatching.
            if not scheduled.done.done():
                scheduled.done.set_exception(exc)
        else:
            if not scheduled.done.done():
                scheduled.done.set_result(None)

    def context(self) -> ResolvedContext:
        """Return what the model sees of the tape, resolved once per tape length.

        Returns:
          resolved: The messages, the ref each came from, and the tape's length.

        """
        if self._cached_resolved is None or self._cached_resolved.version != len(
            self.tape,
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
        discards_content: bool = False,
    ) -> TapeRef:
        """Append a ``ContextSplice`` to the tape.

        A ref has one alive editor at a time: to edit what a splice masks,
        mask the splice's own ref, which absorbs it.

        Args:
          mask: The ranges of tape refs to hide.
          insert_after: Where ``payload`` renders; ``None`` for the head.
          payload: What the splice shows.
          strategy: What produced it.
          token_before: Tokens of what it hides.
          token_after: Tokens of ``payload``.
          fallback_reason: Why a summary fell back to keeping history, if it did.
          preserved_tail_count: How many recent entries that fallback kept.
          paired_externally: Call ids whose partner lies outside ``payload``.
          discards_content: Whether it replaces what it absorbs, as compaction
            does, rather than carrying it forward.

        Returns:
          ref: ``TapeRef`` minted for the new splice.

        Raises:
          InvalidSpliceError: When ``mask`` overlaps any existing alive
              splice's mask, or ``insert_after`` falls inside ``mask``.

        """
        self._validate_no_alive_mask_overlap(mask)
        if not discards_content:
            self._validate_absorbed_payloads_carried(mask, payload)
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

    def _alive_splices(self) -> list[ContextSplice]:
        """Return the splices on the tape that no alive splice masks."""
        alive = alive_splices(self.tape)
        return [r for r in self.tape if isinstance(r, ContextSplice) and r.ref in alive]

    # A new splice may absorb an alive one, masking its ref, but not claim a position
    # one it leaves alive still masks.
    def _validate_no_alive_mask_overlap(self, new_mask: tuple[MaskRange, ...]) -> None:
        """Reject ``new_mask`` if it shares any position with a live splice's mask."""
        for splice in self._alive_splices():
            alive_ref = splice.ref
            if mask_contains_ref(new_mask, alive_ref):
                continue  # Being absorbed by this splice.
            if mask_ranges_overlap(new_mask, splice.mask):
                raise InvalidSpliceError(
                    f"new splice mask overlaps alive splice {alive_ref};"
                    f" either don't claim already-claimed positions or extend"
                    f" the mask to include the splice's own ref to absorb it",
                )

    # Absorbing a splice stops its payload rendering, so a splice that merges rather
    # than replaces must carry that payload. Checked by text: a merge concatenates,
    # so an absorbed entry survives as part of a longer one.
    def _validate_absorbed_payloads_carried(
        self,
        new_mask: tuple[MaskRange, ...],
        payload: tuple[ModelContextEvent, ...],
    ) -> None:
        """Reject a mask that absorbs a splice whose payload is not carried."""
        if not payload:
            return
        for splice in self._alive_splices():
            alive_ref = splice.ref
            if not mask_contains_ref(new_mask, alive_ref):
                continue
            carried = "\n".join(_entry_text(e) for e in payload)
            missing = [
                text
                for entry in splice.payload
                if (text := _entry_text(entry)) and text not in carried
            ]
            if missing:
                raise InvalidSpliceError(
                    f"new splice absorbs {alive_ref} but its payload drops"
                    f" {len(missing)} entry(s) of the absorbed content"
                    f" (first: {missing[0][:60]!r}); carry them forward, or"
                    f" pass discards_content=True to replace deliberately",
                )

    def append_clear(self) -> TapeRef:
        """Append a splice hiding the whole tape, and forget every call's anchors.

        Returns:
          ref: The splice's ref.

        """
        ref = self.append_splice(
            mask=full_tape_mask(self.tape),
            insert_after=None,
            payload=(),
            strategy="clear",
        )
        self._clear_detached_anchors()
        return ref

    def mint_ref(self) -> TapeRef:
        """Return the next ``TapeRef``, for a record a compactor builds.

        Returns:
          ref: A ref no record holds.

        """
        ref = TapeRef(session_id=self.session_id, ordinal=self._next_ordinal)
        self._next_ordinal += 1
        return ref

    def adopt_record(
        self,
        record: TapeRecord,
        *,
        discards_content: bool = False,
    ) -> None:
        """Append a record built with :meth:`mint_ref`, checked as :meth:`append_splice` checks.

        Args:
          record: The record.
          discards_content: As :meth:`append_splice` takes it.

        Raises:
          InvalidSpliceError: A splice claims what an alive splice masks, or
            absorbs one without carrying its payload.
          InvalidContextError: ``record``'s ref is already on the tape.

        """
        # A second claimant of one ref makes every ``context()`` raise, and the
        # dispatch loop catches that raise above its only ``await``, spinning.
        if record.ref in self._tape_by_ref:
            raise InvalidContextError(
                f"duplicate tape ref {record.ref}: two records claim one position",
            )
        if isinstance(record, ContextSplice):
            self._validate_no_alive_mask_overlap(record.mask)
            if not discards_content:
                self._validate_absorbed_payloads_carried(record.mask, record.payload)
        self.tape.append(record)
        self._cached_resolved = None
        self._index_record(record)
        if isinstance(record, ContextSplice):
            self._invalidate_masked_anchors(record)

    def _clear_detached_anchors(self) -> None:
        """Forget every call's anchors."""
        self._placeholder_refs.clear()
        self._parent_assistant_refs.clear()

    # Runs after ``_index_record``, which points an anchor the payload keeps at the
    # splice, so an anchor still masked is one the splice dropped.
    def _invalidate_masked_anchors(self, splice: ContextSplice) -> None:
        """Drop per-call anchors whose ref is masked by ``splice``."""
        if not splice.mask:
            return
        for cid, ref in list(self._parent_assistant_refs.items()):
            if mask_contains_ref(splice.mask, ref):
                del self._parent_assistant_refs[cid]
        for cid, ref in list(self._placeholder_refs.items()):
            if mask_contains_ref(splice.mask, ref):
                del self._placeholder_refs[cid]

    # A splice's payload anchors too: a compactor that keeps a model turn paired
    # externally registers its calls against the splice, so a deferred pairing still
    # finds its anchor after the barrier. Every record is indexed, a resumed one
    # included, so here every mimic id the context holds is reserved.
    def _index_record(self, record: TapeRecord) -> None:
        """Cache ref->record and call_id->anchor mappings after append."""
        self._tape_by_ref[record.ref] = record
        entries = (
            (record.event,)
            if isinstance(record, ReferrableTapeEvent)
            else record.payload
        )
        for entry in entries:
            ids: tuple[str, ...] = ()
            if isinstance(entry, Response):
                ids = tuple(tc.call_id for tc in calls(entry.records))
                for cid in ids:
                    self._parent_assistant_refs[cid] = record.ref
            elif isinstance(entry, ToolResult):
                ids = (entry.call_id,)
                self._placeholder_refs[entry.call_id] = record.ref
            for cid in ids:
                n = _mimic_index_of(cid, prefix=self.markers.mimic)
                if n is not None:
                    self._mimic_counter = max(self._mimic_counter, n + 1)

    # A caller of :meth:`foreground` takes the result as its own answer, so delivering
    # it forward as well would tell the model twice.
    def _deliver_detached(self, result: AnyToolResult) -> None:
        """Hand a detached call's result to whoever waits for it, else deliver it."""
        waiting = [
            w for w in self._foreground_waiters.get(result.call_id, ()) if not w.done()
        ]
        if not waiting or result.stub == "pending":
            self._defer_detached_forward(result)
            return
        for waiter in waiting:
            waiter.set_result(result)
        self._forwarded_call_ids.add(result.call_id)

    # Delivered once per call. A stub is never delivered: it is not the tool's output,
    # and marking its call delivered would drop the real result. A cancellation whose
    # call is already answered cancelled, as a cohort ``Kill`` answers it, is not
    # delivered again; a background job's, answered only by its stub, is.
    def _defer_detached_forward(self, result: AnyToolResult) -> None:
        """Queue a completed detached tool's real result for forward delivery."""
        if result.stub == "pending":
            logger.debug(
                "runtime detached forward skipped (pending stub, not real result):"
                " call_id=%s",
                result.call_id,
            )
            return
        if result.stub == "cancelled" and self._inslot_result_is_terminal(
            result.call_id,
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
        self._pending_commits.append(_PendingCommit(kind="forward", entry=result))

    # The record may be a splice: a compaction can keep the result in its payload.
    def _inslot_result_is_terminal(self, call_id: str) -> bool:
        """Return True when ``call_id``'s in-slot ``ToolResult`` is a final answer."""
        ref = self._placeholder_refs.get(call_id)
        if ref is None:
            return False
        record = self._tape_by_ref[ref]
        entries = (
            (record.event,)
            if isinstance(record, ReferrableTapeEvent)
            else record.payload
        )
        return any(
            isinstance(entry, ToolResult)
            and entry.call_id == call_id
            and entry.stub != "pending"
            for entry in entries
        )

    # A ``DetachedArrived`` call the model made itself is answered beside it on the
    # next real round, so it spends no round of its own.
    def _defer_pairing(self, result: AnyToolResult) -> None:
        """Queue an error ``ToolResult`` pairing an already-emitted ``tool_use``."""
        self._pending_commits.append(_PendingCommit(kind="pairing", entry=result))

    # Runs at the model gate before the context is checked, so what it commits is
    # what the model is sent.
    def _flush_pending(self) -> None:
        """Commit every queued :class:`_PendingCommit`, in arrival order, in-slot."""
        pending = self._pending_commits
        self._pending_commits = []
        for commit in pending:
            if commit.kind == "pairing":
                self._commit_pairing(commit.entry)
            else:
                self._append_detached_arrival(commit.entry)

    # A finished tool is worth a round; a pairing is not.
    def _has_waking_commit(self) -> bool:
        """Return True when a queued commit should itself drive a round."""
        return any(c.kind == "forward" for c in self._pending_commits)

    # Beside its call, not at the tail: a turn in between would leave the call
    # unanswered, and the gate would answer it ``[interrupted]``.
    def _commit_pairing(self, result: AnyToolResult) -> None:
        """Insert ``result`` after its ``AssistantMessage``, pairing the tool-use."""
        parent_ref = self._parent_assistant_refs.get(result.call_id)
        if parent_ref is None or parent_ref not in set(self.context().origins):
            # The call is gone with its turn, so its answer goes too.
            return
        self.append_splice(
            mask=(),
            insert_after=parent_ref,
            payload=(result,),
            strategy="lazy_pairing",
            paired_externally=frozenset({result.call_id}),
        )
        self.publish(result)

    # After a model turn, a user turn naming the call comes first: two model turns
    # may not be adjacent.
    def _append_detached_arrival(self, result: AnyToolResult) -> None:
        """Append the synthetic ``BackgroundTask`` foreground pair for ``result``."""
        messages = self.context().messages
        at = _last_turn(messages)
        if at is not None and isinstance(messages[at], Response):
            self.append_history(
                UserMessage(content=f"[detached tool {result.call_id} completed]"),
            )
        arrival_id = f"{result.call_id}{self.markers.arrival}"
        self.append_history(
            Response(
                records=(
                    ToolCall(
                        call_id=arrival_id,
                        name=self.markers.delivery,
                        arguments={"operation": "foreground", "id": result.call_id},
                    ),
                ),
                tokens=TokenCount(),
                spend=TokenCost(),
                stop_reason="tool_use",
            ),
        )
        self.append_history(dataclasses.replace(result, call_id=arrival_id))

    def _context_validation_error(self) -> Exception | None:
        """Return an unrepaired context error, if the model-call gate finds one."""
        try:
            self._assert_alternation_invariant()
        except (
            InvalidContextError,
            InvalidPayloadError,
            InvalidSpliceError,
        ) as exc:
            return exc
        return None

    async def run_forever(self) -> None:
        """Drain inbox, dispatch, repeat. The entire engine."""
        awaiting_user = False
        queued: list[UserQueuedMessage | AgentSendQueuedMessage] = []
        deferred: list[UserDeferredMessage | AgentSendDeferredMessage] = []

        while True:
            self._collect_detached()
            if self._fully_drained():
                self.publish(AgentIdle())
            items = await self.inbox.drain()
            failed_reply = False

            for item_idx, item in enumerate(items):
                # One event that fails to dispatch is logged; the rest of
                # its batch still is.
                try:
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
                            # Background work too, or it runs on after shutdown.
                            self._cancel_detached()
                            # Work no loop will reach: its callers stop waiting.
                            for later in (
                                *self._scheduled,
                                *items[item_idx + 1 :],
                                *self.inbox.drain_nowait(),
                            ):
                                if isinstance(later, Scheduled):
                                    _ = later.done.cancel()
                            if self._retry_timer is not None:
                                self._retry_timer.cancel()
                            return

                        case Halt():
                            self.publish(item)
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
                            # The user took the turn over: what a failure waited
                            # for no longer calls the model.
                            self._end_suspension()
                            self._failed = False
                            # Tools keep running. What the user typed during the
                            # stream is what the runtime would wait for; held for
                            # the reply's calls, it detaches them, as a message does.
                            if self._mid_stream_queue:
                                self._stop_all_tools(mode="detach")
                            if not self._drain_mid_stream_queue():
                                self.inbox.push_front(
                                    Await.user(),
                                    *items[item_idx + 1 :],
                                )
                                awaiting_user = True
                                break

                        case Clear():
                            self._cancel_model_and_compaction()
                            self._stop_all_tools(mode="detach")
                            # With no call registered, a late result matches
                            # nothing and is dropped.
                            self._cancel_detached()
                            self._pending_commits.clear()
                            self._forwarded_call_ids.clear()
                            queued.clear()
                            deferred.clear()
                            self._mid_stream_queue.clear()
                            self._refusals.clear()
                            # A new conversation does not inherit the old
                            # one's failure, or its wait for a rate limit.
                            self._end_suspension()
                            self._failed = False
                            self.append_clear()
                            self.inbox.push_front(
                                Await.user(),
                                # A tool that finished as the user cleared
                                # posted its result behind the ``Clear``, or
                                # waits behind a Halt; delivering it would
                                # carry the wiped session's work into the
                                # fresh one.
                                *(
                                    i
                                    for i in (
                                        *items[item_idx + 1 :],
                                        *self.inbox.drain_nowait(),
                                    )
                                    if not isinstance(i, DetachedResult)
                                ),
                            )
                            # After the gate: input an observer pushes in reply
                            # must count as new, or it never releases the gate.
                            self.publish(ClearComplete())
                            awaiting_user = True
                            break

                        case ModelResponseError(exception=exc):
                            # A failure of a model switched away from says nothing
                            # of the one in use: the gate asks it, at once.
                            if self._called_model is not self.model:
                                self.model_call = None
                                _ = self._drain_mid_stream_queue()
                                continue
                            # A request too long for the window is compaction's job: compact,
                            # and let the gate call again, under the same breaker.
                            if (
                                isinstance(exc, ContextOverflowError)
                                and self.compactor is not None
                                and self._auto_compactions < 2
                            ):
                                logger.warning(
                                    "model context overflowed; compacting: %s",
                                    exc,
                                )
                                self.model_call = None
                                self._auto_compactions += 1
                                self.inbox.push_front(
                                    Compact(trigger="auto"),
                                    *items[item_idx + 1 :],
                                )
                                break
                            # A long rate limit or outage, for an agent that waits them
                            # out: hold the model call until the advised time, backing
                            # off to 5 minutes, then call again; 60 in a row end the turn.
                            if (
                                isinstance(exc, RetryableError)
                                and self.persistent_retry
                                and self._retries < 60
                            ):
                                self.model_call = None
                                delay = max(
                                    exc.retry_after_sec,
                                    min(5.0 * (1 << self._retries), 300.0),
                                )
                                self._retries += 1
                                self.service_suspended_until = time.time() + delay
                                self.publish(
                                    ModelCallSuspended(
                                        exception=exc,
                                        until=self.service_suspended_until,
                                        attempt=self._retries,
                                    ),
                                )
                                # Input typed during the failed stream is part
                                # of what the retry asks.
                                _ = self._drain_mid_stream_queue()
                                # Any input wakes the loop; ``SaveSession`` changes nothing else.
                                self._retry_timer = (
                                    asyncio.get_running_loop().call_later(
                                        delay,
                                        self.inbox.push_back,
                                        SaveSession(),
                                    )
                                )
                                continue
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
                            self._end_suspension()
                            # The error is shown, never put in the context.
                            committed = self._drain_mid_stream_queue()
                            self.publish(item)
                            if not committed:
                                self._failed = True
                                self.inbox.push_front(
                                    Await.failure(),
                                    *items[item_idx + 1 :],
                                )
                                awaiting_user = True
                                break

                        # Released from a failure's wait, or ending a suspension's:
                        # the gate calls the model for the turn.
                        case Retry():
                            self._end_suspension()
                            self._failed = False
                            awaiting_user = False

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
                                self._cancel_detached()
                            elif cid in self.running_tools:
                                logger.debug("runtime kill tool: call_id=%s", cid)
                                self._stop_tool(
                                    cid,
                                    self.running_tools.pop(cid),
                                    mode="kill",
                                )
                                self.cohort.discard(cid)
                            else:
                                # Its own late result then matches nothing; the
                                # model is told it was cancelled instead.
                                task = self.detached.pop(cid, None)
                                if task is not None:
                                    logger.debug(
                                        "runtime kill detached tool: call_id=%s",
                                        cid,
                                    )
                                    self._kill_detached(cid, task)
                                else:
                                    logger.debug(
                                        "runtime kill missed tool: call_id=%s",
                                        cid,
                                    )

                        case Detach(call_id=cid) if cid in self.cohort:
                            task = self.running_tools[cid]
                            for member, held in list(self.running_tools.items()):
                                if held is task:
                                    logger.debug(
                                        "runtime tool detached after %ss: call_id=%s",
                                        self.detach_after_sec,
                                        member,
                                    )
                                    self._stop_tool(
                                        member,
                                        self.running_tools.pop(member),
                                        mode="detach",
                                    )
                                    self.cohort.discard(member)
                            if not self.cohort:
                                self._cohort_seen = False

                        case Compact(args=args, trigger=trigger):
                            if self.compact_task and not self.compact_task.done():
                                logger.debug(
                                    "runtime compact ignored while compact_task active",
                                )
                                continue
                            self._compaction = trigger
                            logger.debug(
                                "runtime compact start: model_call=%s "
                                "running_tools=%d cohort=%d detached=%d args=%r",
                                self.model_call is not None,
                                len(self.running_tools),
                                len(self.cohort),
                                len(self.detached),
                                args,
                            )
                            self._cancel_model()
                            self._stop_all_tools(mode="detach")
                            # What the user typed goes into what is compacted;
                            # only ``Clear`` discards it.
                            for staged in queued:
                                self._commit_user(_message_from_queued(staged))
                            queued.clear()
                            _ = self._drain_mid_stream_queue()
                            self.compact_task = asyncio.create_task(
                                self._compact_and_post(args),
                            )
                            self.compact_task.add_done_callback(
                                log_task_exception(
                                    logger,
                                    "compaction task crashed",
                                ),
                            )
                            started = CompactStarted()
                            self.append_history(started)
                            self.publish(started)

                        case CompactComplete():
                            self.compact_task = None
                            self.compactions += 1
                            self.append_history(item)
                            self.publish(item)

                        case CompactFailed():
                            self.compact_task = None
                            self._compaction = ""
                            self.append_history(item)
                            self.publish(item)

                        # Input held for the running calls goes first. The user
                        # writing during a suspension asks to try now.
                        case UserMessage():
                            self._end_suspension()
                            if self.model_call is not None:
                                self._mid_stream_queue.append(item)
                            else:
                                self._stop_all_tools(mode="detach")
                                _ = self._drain_mid_stream_queue()
                                self._commit_user(item)
                            awaiting_user = False

                        case AgentToAgentMessage():
                            if self.model_call is not None:
                                self._mid_stream_queue.append(item)
                            else:
                                self._stop_all_tools(mode="detach")
                                _ = self._drain_mid_stream_queue()
                                self._commit_user(item)
                            awaiting_user = False

                        # A nudge to a busy agent is moot: it is nudged again
                        # at its next idle if it still owes.
                        case Nudge(content=content):
                            if self.model_call is None:
                                self.append_history(
                                    SystemMessage(content=content, subtype="nudge"),
                                )
                            awaiting_user = False

                        # Input that ends the wait for the user is the user
                        # resuming; held for idle, the halted turn would be
                        # sent again without it.
                        case (
                            UserQueuedMessage()
                            | AgentSendQueuedMessage()
                            | UserDeferredMessage()
                            | AgentSendDeferredMessage()
                        ) if awaiting_user:
                            self._commit_user(_message_from_queued(item))
                            awaiting_user = False

                        case UserQueuedMessage() | AgentSendQueuedMessage():
                            queued.append(item)

                        case UserDeferredMessage() | AgentSendDeferredMessage():
                            deferred.append(item)

                        case ModelResponsePartial():
                            self.publish(item)

                        case ModelResponseThinking():
                            self.publish(item)

                        case ModelResponseComplete(
                            response=msg,
                            generation=generation,
                        ):
                            if generation not in (-1, self._model_call_generation):
                                logger.debug(
                                    "runtime stale model response ignored: "
                                    "generation=%d current=%d",
                                    generation,
                                    self._model_call_generation,
                                )
                                continue
                            self._auto_compactions = 0
                            self._retries = 0
                            self.model_call = None
                            msg = self._rewrite_calls(
                                self._sanitize_forged_arrivals(msg),
                            )
                            self._gate_calls(msg)
                            cut_in = (
                                self.before_tool_spawn(msg)
                                if self.before_tool_spawn is not None
                                else None
                            )
                            self.append_history(msg)
                            # Observers see the turn as recorded, rewrites included.
                            self.publish(dataclasses.replace(item, response=msg))
                            if cut_in is not None:
                                self._mid_stream_queue.append(cut_in)
                            if calls(msg.records):
                                # What the user sent as the reply streamed is answered
                                # next; the reply's calls run on, detached, and no
                                # wait on them begins.
                                cut = bool(self._mid_stream_queue)
                                self._start_cohort(
                                    self._start_background(calls(msg.records)),
                                    waited=not cut,
                                )
                                if cut:
                                    self._stop_all_tools(mode="detach")
                                    _ = self._drain_mid_stream_queue()
                            elif not self._drain_mid_stream_queue():
                                self.publish(ModelIdle())

                        case ModelResponseCancelled():
                            self.model_call = None
                            self.publish(item)

                        case (
                            ShellCommandResult()
                            | FileReadResult()
                            | FileWriteResult()
                            | FileEditResult()
                            | WebSearchResults()
                            | WebFetchResult()
                            | AgentStatusResult()
                            | UncategorizedToolResult()
                        ) if item.call_id in self.cohort:
                            cid = item.call_id
                            self.cohort.discard(cid)
                            self.running_tools.pop(cid, None)
                            self.append_history(item)
                            self.publish(item)

                        case (
                            ShellCommandResult()
                            | FileReadResult()
                            | FileWriteResult()
                            | FileEditResult()
                            | WebSearchResults()
                            | WebFetchResult()
                            | AgentStatusResult()
                            | UncategorizedToolResult()
                        ) if item.call_id in self.detached:
                            cid = item.call_id
                            # Posted before its call was detached, or refused:
                            # its call is answered by a stub, so it goes forward.
                            del self.detached[cid]
                            self._deliver_detached(item)
                            self.publish(item)

                        case DetachedResult():
                            self._deliver_detached(item.result)
                            self.publish(item)

                        case LazyEvent(payload=payload):
                            # The call leaves the cohort now, or the round its
                            # answer waits for never comes.
                            self.cohort.discard(payload.call_id)
                            self.running_tools.pop(payload.call_id, None)
                            self._defer_pairing(payload)

                        case Scheduled():
                            self._scheduled.append(item)

                        case _:
                            pass
                except Exception as exc:  # noqa: BLE001 -- One event's failure must not drop the events after it.
                    log_exception_or_warning(
                        logger,
                        f"dispatching {type(item).__name__} raised",
                        exc,
                    )
                    # A reply not taken is a failed model call: asked again at
                    # once, it would fail again, so the runtime waits.
                    if isinstance(item, ModelResponseComplete):
                        self.inbox.push_front(
                            ModelResponseError(exception=exc),
                            *items[item_idx + 1 :],
                        )
                        failed_reply = True
                        break

            if failed_reply:
                continue

            # A failure here leaves the batch taken; the next pass goes on.
            try:
                if self._scheduled and self._engine_quiescent:
                    scheduled = self._scheduled
                    self._scheduled = []
                    for work in scheduled:
                        await self._run_scheduled(work)

                if not self.cohort and self._cohort_seen:
                    logger.debug(
                        "runtime cohort complete: running_tools=%d detached=%d",
                        len(self.running_tools),
                        len(self.detached),
                    )
                    self.publish(CohortComplete())
                    self._cohort_seen = False

                # What the user sent as the reply streamed, if no ending of the reply
                # took it in already.
                if (
                    self._mid_stream_queue
                    and self.model_call is None
                    and not self.cohort
                ):
                    _ = self._drain_mid_stream_queue()

                # Staged input waits until the model is done, not merely between
                # rounds: a tool result still to answer means it is not.
                if (
                    self._engine_quiescent
                    and not self._should_call_model()
                    and (queued or deferred)
                ):
                    # Queued first; deferred waits for the idle after it.
                    staged = queued or deferred
                    for waiting in staged:
                        self._commit_user(_message_from_queued(waiting))
                    staged.clear()

                # A delivery drives its own round; the rest ride one that content
                # drives.
                if (
                    self._pending_commits
                    and self._ready_to_advance
                    and (self._should_call_model() or self._has_waking_commit())
                ):
                    self._flush_pending()

                # Prune first: a drop that brings the context back under the
                # compaction trigger spares the compaction.
                if (
                    self.prune is not None
                    and self._ready_to_advance
                    and self._should_call_model()
                ):
                    _ = self.prune()
                # Compact first when the compactor says the next request would not fit.
                # Twice with no model call between means compacting does not help, so
                # the call goes ahead and the provider's error says why.
                compact_first = (
                    self._ready_to_advance
                    and self._should_call_model()
                    and self.compactor is not None
                    and self.compact_due is not None
                    and self._auto_compactions < 2
                    and self.compact_due()
                )
                if compact_first:
                    self._auto_compactions += 1
                    self.inbox.push_back(Compact(trigger="auto"))
                elif (
                    self._ready_to_advance
                    and self._should_call_model()
                    and time.time() >= (self.service_suspended_until or 0.0)
                ):
                    if self.context_updates is not None:
                        for note in self.context_updates():
                            self.append_history(note)
                    logger.debug(
                        "runtime model call start: history=%d detached=%d queued=%d",
                        len(self.context().messages),
                        len(self.detached),
                        len(queued),
                    )
                    exc = self._context_validation_error()
                    if exc is not None:
                        # Raised, it would repeat every pass on the same tape;
                        # the user is told, and the runtime waits for them.
                        log_exception_or_warning(
                            logger,
                            "model-call gate context unrepairable",
                            exc,
                        )
                        # The gate first, as for ``Clear``.
                        self.inbox.push_front(Await.recovery())
                        self.publish(ModelResponseError(exception=exc))
                        awaiting_user = True
                    else:
                        # Whatever released a failure's wait, the turn goes on.
                        self._failed = False
                        model_call: asyncio.Task[None] = asyncio.create_task(
                            self._stream_and_post(self._next_generation()),
                        )
                        self.model_call = model_call
                        model_call.add_done_callback(
                            log_task_exception(logger, "model-call task crashed"),
                        )
                        self.publish(ModelCallStarted())

                self.publish(SaveSession())
            except Exception as exc:  # noqa: BLE001 -- The loop must outlive one pass's failure.
                log_exception_or_warning(logger, "dispatch loop iteration raised", exc)

    def pending_mid_stream(self) -> Sequence[UserMessage | AgentToAgentMessage]:
        """Return what the user sent while the model streams, in order.

        Returns:
          pending: The messages; they join the context once the reply ends, its
            calls detached.

        """
        return tuple(self._mid_stream_queue)

    # A note rides a round that real content drives and drives none itself, so the entry
    # that decides is the last turn before any trailing notes; a nudge alone drives one.
    def _should_call_model(self) -> bool:
        """Return True when history ends with content the model should answer."""
        messages = self.context().messages
        at = _last_turn(messages)
        trailing = messages[0 if at is None else at + 1 :]
        if any(isinstance(m, SystemMessage) and m.subtype == "nudge" for m in trailing):
            return True
        return at is not None and isinstance(
            messages[at],
            (UserMessage, AgentToAgentMessage, ToolResult),
        )

    # Splices are valid one by one, but together they can still leave a call or a
    # result unpaired; then the whole context is rebuilt repaired.
    def _assert_alternation_invariant(self) -> None:
        """Repair unpaired recorded calls and results; rebuild the context if still invalid."""
        self._repair_history_record_orphans()
        try:
            validate_context(self.context().messages)
            return
        except InvalidContextError as exc:
            logger.warning(
                "context invariant still violated after repair; "
                "applying rescue barrier: %s",
                exc,
            )
        self._rescue_context()
        validate_context(self.context().messages)

    def _rescue_context(self) -> None:
        """Append a splice hiding the whole tape and showing the context repaired."""
        resolved = self.context()
        sanitized = splice_safe_repair(resolved.messages)
        self.append_splice(
            mask=full_tape_mask(self.tape),
            insert_after=None,
            payload=tuple(sanitized),
            strategy="context_rescue",
            paired_externally=unpaired_call_ids(sanitized),
            # The repair drops what cannot be paired.
            discards_content=True,
        )

    # Only recorded events are repaired here: a call answered ``[interrupted]`` beside
    # its turn, a result with no call hidden. An unpaired entry a splice shows is left
    # to the rebuild.
    def _repair_history_record_orphans(self) -> None:
        """Pair each recorded call left unanswered; hide each recorded result with no call."""
        resolved = self.context()
        messages = resolved.messages
        origins = resolved.origins
        unmatched_per_am: list[tuple[TapeRef | None, list[str]]] = []
        # An ordered set.
        pending: dict[str, None] = {}
        am_repair_anchor: TapeRef | None = None
        hr_orphan_refs: list[TapeRef] = []

        for idx, entry in enumerate(messages):
            # No one's turn: a note neither opens nor closes a tool pair.
            if isinstance(entry, (SystemMessage, TurnContext)):
                continue
            origin = origins[idx]
            origin_record = self._tape_by_ref.get(origin)
            origin_is_hr = isinstance(origin_record, ReferrableTapeEvent)
            if isinstance(entry, Response):
                _flush_unmatched(pending, unmatched_per_am, am_repair_anchor)
                am_repair_anchor = origin if origin_is_hr else None
                pending.update(dict.fromkeys(tc.call_id for tc in calls(entry.records)))
            elif isinstance(entry, ToolResult):
                if entry.call_id not in pending:
                    if origin_is_hr:
                        hr_orphan_refs.append(origin)
                    continue
                del pending[entry.call_id]
            else:
                _flush_unmatched(pending, unmatched_per_am, am_repair_anchor)
                am_repair_anchor = None
        _flush_unmatched(pending, unmatched_per_am, am_repair_anchor)

        for anchor, missing_ids in unmatched_per_am:
            if anchor is None:
                continue
            payload = tuple(
                UncategorizedToolResult(
                    call_id=cid,
                    content="[interrupted]",
                    is_error=True,
                )
                for cid in missing_ids
            )
            # Their calls are in the recorded turn at ``anchor``.
            self.append_splice(
                mask=(),
                insert_after=anchor,
                payload=payload,
                strategy="orphan_tool_use_repair",
                paired_externally=frozenset(missing_ids),
            )
        for orphan in hr_orphan_refs:
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

    # Each message stays its own record, as it was sent and shown; the request merges
    # adjacent ones, as providers need.
    def _commit_user(self, message: UserMessage | AgentToAgentMessage) -> None:
        """Append ``message`` to history and publish it."""
        self.turns += 1
        self._steps = 0
        self.turn_started_ms = int(time.time() * 1_000)
        self.append_history(message)
        self.publish(message)

    # The call is answered now, by a stub, and its task registered as detached, so
    # its own result, if it comes, is delivered forward.
    def _stop_tool(
        self,
        cid: str,
        task: asyncio.Task[None],
        *,
        mode: Literal["detach", "kill"],
    ) -> None:
        """Answer a cohort call ``[detached]`` and let it run, or ``[cancelled]`` and cancel it."""
        stub = (
            UncategorizedToolResult(
                call_id=cid,
                content=self.markers.cancelled,
                is_error=True,
                stub="cancelled",
            )
            if mode == "kill"
            else UncategorizedToolResult(
                call_id=cid,
                content=self.markers.detached,
                stub="pending",
            )
        )
        self.append_history(stub)
        self.publish(stub)
        self.detached[cid] = task
        if mode == "kill":
            task.cancel()
        else:
            self._release(self._holds.get(task))

    def _stop_all_tools(self, *, mode: Literal["detach", "kill"]) -> None:
        """Run :meth:`_stop_tool` for every cohort tool; reset cohort state."""
        for cid, task in list(self.running_tools.items()):
            self._stop_tool(cid, task, mode=mode)
        self.running_tools.clear()
        self.cohort.clear()
        self._cohort_seen = False

    def _cancel_model(self) -> None:
        """Cancel any model call; its late reply is then dropped."""
        if self.model_call:
            self.model_call.cancel()
            self.model_call = None
            _ = self._next_generation()

    def _next_generation(self) -> int:
        """Return a new model call generation; a reply from an older one is dropped."""
        self._model_call_generation += 1
        return self._model_call_generation

    def _cancel_model_and_compaction(self) -> None:
        """Cancel any model call and compaction; their late results are then dropped."""
        self._cancel_model()
        if self.compact_task is not None:
            compact_task = self.compact_task
            if not compact_task.done():
                self._compact_generation += 1
                self._compaction = ""
                compact_task.cancel()
                self.publish(
                    CompactFailed(
                        exception=asyncio.CancelledError(),
                        tape_len=len(self.tape),
                    ),
                )
            self.compact_task = None

    # A call the model makes may copy a delivery, id and all, where a provider lets it
    # name its calls: two turns would then share a call id. The old ``arrived`` name
    # is renamed too, as a journal that holds it may still be copied.
    def _sanitize_forged_arrivals(self, msg: Response) -> Response:
        """Rewrite model-forged delivery call ids into a safe namespace."""
        found = calls(msg.records)
        if not any(_forged(tc, markers=self.markers) for tc in found):
            return msg
        # Another call of the same reply may already hold a renamed id.
        used_ids = {tc.call_id for tc in found if not _forged(tc, markers=self.markers)}
        rewritten: list[AssistantMessage | Thinking | ToolCall] = []
        for record in msg.records:
            if not isinstance(record, ToolCall) or not _forged(
                record,
                markers=self.markers,
            ):
                rewritten.append(record)
                continue
            safe_id = f"{self.markers.mimic}{self._mimic_counter}"
            self._mimic_counter += 1
            while safe_id in used_ids:
                safe_id = f"{self.markers.mimic}{self._mimic_counter}"
                self._mimic_counter += 1
            logger.debug(
                "runtime forged delivery id rewritten: %s -> %s",
                record.call_id,
                safe_id,
            )
            rewritten.append(dataclasses.replace(record, call_id=safe_id))
        return dataclasses.replace(msg, records=tuple(rewritten))

    # The model's history should show the tool it should have used: a Bash call that
    # is exactly other tools' jobs becomes their calls before it is recorded, so
    # later turns copy the right tools. The first keeps the call's id; the rest get
    # ``<id>_<n>``, unused in the turn. A command that is other tools' work but not
    # as such calls stays a Bash call, answered with the refusal instead of run.
    #
    # A leading ``cd`` to where the shell starts anyway is dropped, so the model
    # stops copying it; only until a call that may move the shell, since the
    # calls after it may start somewhere else.
    def _rewrite_calls(self, msg: Response) -> Response:
        """Return ``msg`` with each Bash call that is other tools' job rewritten."""
        records: list[AssistantMessage | Thinking | ToolCall] = []
        used = {r.call_id for r in calls(msg.records)}
        # The reply's calls run side by side and share the shell's directory: one
        # that moves it may land before any other, whatever its place in the reply.
        cwd = str(self.tool_context.cwd)
        settled = not any(
            isinstance(self.tools.get(c.name), Bash)
            and isinstance(command := c.arguments.get("command"), str)
            and moves(drop_cd(command, cwd=cwd))
            for c in calls(msg.records)
        )
        for given in msg.records:
            bash = self.tools.get(given.name) if isinstance(given, ToolCall) else None
            if not isinstance(given, ToolCall) or not isinstance(bash, Bash):
                records.append(given)
                continue
            record = given
            command = given.arguments.get("command")
            # A call run later may start after the next reply moved the shell.
            later = given.arguments.get("background") is True or bool(
                given.arguments.get("delay"),
            )
            if isinstance(command, str) and settled and not later:
                record = dataclasses.replace(
                    given,
                    arguments={
                        **given.arguments,
                        "command": drop_cd(command, cwd=cwd),
                    },
                )
            rewritten = bash.rewrite(record.arguments)
            if isinstance(rewritten, (Rewrite, tuple)):
                # Each rewritten call keeps any request to run in the background.
                timing = {
                    k: v
                    for k, v in record.arguments.items()
                    if k in ("background", "delay")
                }
                ids = _fresh_ids(record.call_id, used=used)
                records.extend(
                    dataclasses.replace(
                        record,
                        call_id=next(ids),
                        name=each.tool.name,
                        arguments={**each.arguments, **timing},
                    )
                    for each in (
                        (rewritten,) if isinstance(rewritten, Rewrite) else rewritten
                    )
                )
                continue
            if isinstance(rewritten, str):
                self._refusals[record.call_id] = rewritten
            records.append(record)
        return dataclasses.replace(msg, records=tuple(records))

    # Every tool is offered, so the gate holds here, and again in the context. It is
    # the gate the model was told when it called: observers of the reply may close it
    # for the next round, as the round cap does, and must not refuse this one's calls.
    def _gate_calls(self, msg: Response) -> None:
        """Refuse each call in ``msg`` of a tool ``allowed`` withholds now."""
        if self.allowed is None:
            return
        for tc in calls(msg.records):
            if tc.name in self.tools and tc.name not in self.allowed:
                self._refusals.setdefault(
                    tc.call_id,
                    f"{tc.name} is not available to this agent.",
                )

    # ``waited`` false starts them only to detach them at once: no wait begins, so
    # ``CohortStarted`` is not told.
    def _start_cohort(self, cohort: Sequence[ToolCall], *, waited: bool = True) -> None:
        """Run ``cohort``, the calls the next model round waits on."""
        if not cohort:
            return
        self._cohort_seen = True
        if waited:
            self.publish(CohortStarted())
        logger.debug(
            "runtime cohort start: tools=%s",
            [(tc.call_id, tc.name) for tc in cohort],
        )
        for tc in cohort:
            self.cohort.add(tc.call_id)
        for group in self._partition_cohort(cohort):
            names = "+".join(tc.name for tc in group)
            hold = self._hold(group)
            tool_task = asyncio.create_task(
                self._run_tool_group_and_post(group, hold=hold),
            )
            tool_task.add_done_callback(
                log_task_exception(logger, f"cohort tool {names!r} crashed"),
            )
            if hold is not None:
                self._holds[tool_task] = hold
                tool_task.add_done_callback(self._holds.pop)
            # A ``Kill`` of any member stops the group's task.
            for tc in group:
                self.running_tools[tc.call_id] = tool_task
            self._detach_later(group, tool_task)

    # A tool that manages jobs, such as waiting for one, is waited on to its end.
    def _detach_later(
        self,
        group: Sequence[ToolCall],
        task: asyncio.Task[None],
    ) -> None:
        """Detach ``group`` if its task outlives ``detach_after_sec``."""
        tools = [self.tools.get(tc.name) for tc in group]
        if (
            self.detach_after_sec is None
            or not self.allow_background
            or not all(t is None or backgroundable(t) for t in tools)
        ):
            return
        timer = asyncio.get_running_loop().call_later(
            self.detach_after_sec,
            self.push,
            Detach(call_id=group[0].call_id),
        )
        task.add_done_callback(lambda _: timer.cancel())

    # Each runs as its own task: the call was promised to run apart.
    def _start_background(self, tool_calls: Sequence[ToolCall]) -> list[ToolCall]:
        """Start each call that asks to run in the background; return the others."""
        foreground: list[ToolCall] = []
        for tc in tool_calls:
            tool = self.tools.get(tc.name)
            delay = None
            if tool is not None and self.allow_background and backgroundable(tool):
                try:
                    delay = background_delay(tc.arguments)
                except (TypeError, ValueError) as exc:
                    self._refusals[tc.call_id] = str(exc)
            if delay is None:
                foreground.append(tc)
                continue
            stub = UncategorizedToolResult(
                call_id=tc.call_id,
                content=f"{self.markers.running}{tc.name} as job {tc.call_id}]",
                stub="pending",
            )
            self.append_history(stub)
            self.publish(stub)
            task = asyncio.create_task(
                self._run_tool_group_and_post([tc], wake_at=time.time() + delay),
            )
            task.add_done_callback(
                log_task_exception(logger, f"background tool {tc.name!r} crashed"),
            )
            self.detached[tc.call_id] = task
        return foreground

    def _drain_mid_stream_queue(self) -> bool:
        """Commit and publish any buffered mid-stream input; return whether any was."""
        buffered = self._mid_stream_queue
        self._mid_stream_queue = []
        for message in buffered:
            self._commit_user(message)
        return bool(buffered)

    # With no call registered, a cancelled task's late result matches nothing and is
    # dropped; each call is told it was cancelled instead.
    def _cancel_detached(self) -> None:
        """Cancel and forget every detached task, reporting each cancellation."""
        killed = dict(self.detached)
        self.detached.clear()
        for cid, task in killed.items():
            self._kill_detached(cid, task)

    # The call was answered with a stub saying it still runs, so the model is told it
    # was killed. A finished task already delivered its own result, or will.
    def _kill_detached(self, cid: str, task: asyncio.Task[None]) -> None:
        """Cancel a detached call's task and deliver its cancellation."""
        if task.done():
            return
        _ = task.cancel()
        self._deliver_detached(
            UncategorizedToolResult(
                call_id=cid,
                content=self.markers.cancelled,
                is_error=True,
                stub="cancelled",
            ),
        )

    # A finished task removes itself; one cancelled before it ran a step does not.
    def _collect_detached(self) -> None:
        """Clean up detached tasks that completed or were cancelled."""
        for cid in [c for c, t in self.detached.items() if t.done()]:
            del self.detached[cid]

    async def _stream_and_post(self, generation: int) -> None:
        """Stream a model response, posting chunks and the final response."""
        self._streamed_chars = 0
        try:
            response = await self._reply()
            self.inbox.push_back(
                ModelResponseComplete(response=response, generation=generation),
            )
        except asyncio.CancelledError:
            # Published, not queued: the Halt that cancelled the stream holds
            # the inbox until the user types, and the display must stop now.
            self.publish(
                ModelResponseCancelled(output_chars_estimate=self._streamed_chars),
            )
        except Exception as exc:  # noqa: BLE001 -- Model-call failures from adapters must be routed into the runtime error record.
            log_exception_or_warning(logger, "model call failed", exc)
            self.inbox.push_back(ModelResponseError(exception=exc))

    def request(self, *, after: Sequence[SessionRecord] = ()) -> Request:
        """Return the request the next model call sends, with ``after`` appended.

        Its prefix is the last call's, byte for byte, so a side call such as a
        summary that appends to it reads the cache that call wrote.

        Args:
          after: Records to add after the context.

        Returns:
          request: The system prompt, the context and ``after``, and every tool.

        """
        return self.tagged(
            Request(
                # The runtime's note opens the first block: it never changes, so it
                # caches with the prompt under one mark. It tells the model to call
                # the tool deliveries are made under, so without that tool it would
                # send the model to a tool it does not have.
                system=_noted(self.system, note=self.markers.note)
                if self.markers.delivery in self.tools
                else self.system,
                records=[
                    *(
                        record
                        for entry in coalesce_roles(self.context().messages)
                        for record in (
                            entry.records if isinstance(entry, Response) else (entry,)
                        )
                    ),
                    *after,
                ],
                # By name, so the same tools are the same bytes whatever order
                # they came in.
                tools=tuple(
                    offered(t) if self.allow_background else t
                    for t in sorted(self.tools.values(), key=lambda t: t.name)
                ),
                allowed=self.allowed,
                max_response_tokens=self.reply_room()
                if self.reply_room is not None
                else 0,
            ),
        )

    def tagged(self, request: Request) -> Request:
        """Return ``request`` naming this conversation, its window, turn, and compaction.

        Args:
          request: A call this agent makes.

        Returns:
          tagged: ``request``, told whose conversation it is part of, as the
            turn's next call.

        """
        return dataclasses.replace(
            request,
            compaction=self._compaction,
            session=self.conversation,
            agent=self.agent or self.conversation,
            parent=self.parent,
            parent_turn=self.parent_turn,
            root_prompt=self.root_prompt,
            name=self.name,
            window=self.compactions,
            turn=self.turns,
            turn_started_ms=self.turn_started_ms,
            step=self._steps,
        )

    def sending(self, request: Request) -> Request:
        """Return ``request`` tagged as it is sent, and count it among the turn's calls.

        Args:
          request: A call this agent is about to send.

        Returns:
          sent: ``request``, tagged.

        """
        sent = self.tagged(request)
        self._steps += 1
        return sent

    async def _reply(self) -> Response:
        """Ask the model about the context; post its words as they arrive."""
        response = None
        request = self.sending(self.request())
        # Only the first call after a compaction is told of it.
        self._compaction = ""
        self._called_model = self.model
        async for event in self._called_model.stream(request):
            if isinstance(event, TextChunk):
                self._streamed_chars += len(event.text)
                self.inbox.push_back(
                    ModelResponseThinking(text=event.text, record=event.record)
                    if event.thinking
                    else ModelResponsePartial(text=event.text, record=event.record),
                )
            else:
                response = event
        if response is None:
            raise RuntimeError("The model ended its stream without a reply.")
        ids = [r.call_id for r in response.records if isinstance(r, ToolCall)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"The reply repeats a tool call id: {sorted(ids)}.")
        return response

    # Calls sharing a ``serialize_key``, such as Edits of one file, run one after
    # another; the rest run in parallel. A ``"*"`` key, a Bash writer's, contends with
    # every keyed call, so then all of them form one group, in submission order.
    def _partition_cohort(self, calls: Sequence[ToolCall]) -> list[list[ToolCall]]:
        """Split a cohort into serialized groups, preserving submission order."""
        keys = [self._serialize_key(call) for call in calls]
        exclusive = "*" in keys
        groups: list[list[ToolCall]] = []
        by_key: dict[str, list[ToolCall]] = {}
        for call, key in zip(calls, keys, strict=True):
            if key is None:
                groups.append([call])
                continue
            slot = "*" if exclusive else key
            group = by_key.get(slot)
            if group is None:
                group = by_key[slot] = [call]
                groups.append(group)
            else:
                group.append(call)
        return groups

    def _serialize_key(self, call: ToolCall) -> str | None:
        """Return the tool's serialization key for ``call``; unknown tools have none."""
        tool = self.tools.get(call.name)
        if tool is None:
            return None
        return tool.serialize_key(
            self.tool_context,
            without_controls(call.arguments)
            if backgroundable(tool)
            else call.arguments,
        )

    # One coroutine awaits each call in turn, so order never rests on lock fairness.
    # Released at the end, so a killed call holds on until it has stopped.
    async def _run_tool_group_and_post(
        self,
        group: Sequence[ToolCall],
        *,
        wake_at: float | None = None,
        hold: _Hold | None = None,
    ) -> None:
        """Run a serialized group of calls one at a time, in order, from ``wake_at``."""
        try:
            if hold is not None and not await self._wait_for(hold, group):
                return
            for index, call in enumerate(group):
                if await self._run_tool_and_post(call, wake_at=wake_at):
                    continue
                # Cancelling one call of a group stops the resource they share,
                # so the next Edit of the file the user just stopped never starts.
                for skipped in group[index + 1 :]:
                    self._post_cancelled(skipped)
                return
        finally:
            self._release(hold)

    def _hold(self, group: Sequence[ToolCall]) -> _Hold | None:
        """Register what ``group`` touches, behind every earlier holder of any of it."""
        keys = frozenset(k for k in map(self._serialize_key, group) if k is not None)
        if not keys:
            return None
        hold = _Hold(
            keys=keys,
            after=tuple(h for h in self._holding if _contend(keys, h.keys)),
        )
        self._holding.append(hold)
        return hold

    def _release(self, hold: _Hold | None) -> None:
        """Let the calls waiting on ``hold`` go; once, however often it is asked."""
        if hold is None or hold.released.is_set():
            return
        self._holding.remove(hold)
        hold.released.set()

    async def _wait_for(self, hold: _Hold, group: Sequence[ToolCall]) -> bool:
        """Wait out each earlier holder; ``False``, each call answered, if cancelled."""
        try:
            for earlier in hold.after:
                _ = await earlier.released.wait()
        except asyncio.CancelledError:
            for call in group:
                self._post_cancelled(call)
            return False
        return True

    def _post_cancelled(self, call: ToolCall) -> None:
        """Post a ``[cancelled]`` result for a call that never ran."""
        logger.debug("runtime group skipped after cancel: call_id=%s", call.call_id)
        self.inbox.push_back(
            UncategorizedToolResult(
                call_id=call.call_id,
                content=self.markers.cancelled,
                is_error=True,
                stub="cancelled",
            ),
        )

    # A failure or a cancellation is still a result, so every call is answered; a call
    # detached meanwhile is delivered forward.
    async def _run_tool_and_post(
        self,
        call: ToolCall,
        *,
        wake_at: float | None = None,
    ) -> bool:
        """Run one tool invocation from ``wake_at``, post the ``ToolResult`` to the inbox."""
        if call.name == self.markers.arrived:
            # No tool has this name; the model copied a delivery. Its answer waits
            # for the next real round, so it spends no round of its own.
            logger.debug("runtime detached-arrived mimic: call_id=%s", call.call_id)
            self.inbox.push_back(
                LazyEvent(
                    payload=UncategorizedToolResult(
                        call_id=call.call_id,
                        content=f"{self.markers.arrived} is not callable; detached "
                        "results arrive automatically.",
                        is_error=True,
                    ),
                ),
            )
            return True
        tool = self.tools.get(call.name)
        refused = self._refusals.pop(call.call_id, None)
        if tool is None or refused is not None:
            logger.debug(
                "runtime tool not run: call_id=%s tool=%s refused=%s",
                call.call_id,
                call.name,
                refused is not None,
            )
            self.inbox.push_back(
                UncategorizedToolResult(
                    call_id=call.call_id,
                    content=refused
                    if refused is not None
                    else f"Unknown tool: {call.name}",
                    is_error=True,
                ),
            )
            return True
        # Listed from here to the post below, with no ``await`` outside the
        # ``try``: a call refused, or cancelled before its task ran a step, is
        # never listed, and every listed one is released.
        self._started[call.call_id] = (call, time.time())
        if wake_at is not None:
            self._wake_at[call.call_id] = wake_at
        completed = True
        result: AnyToolResult
        try:
            logger.debug(
                "runtime tool start: call_id=%s tool=%s",
                call.call_id,
                call.name,
            )
            # Asleep inside the ``try``, so a job cancelled before it starts is
            # still answered, as cancelled.
            if wake_at is not None and (wait_sec := wake_at - time.time()) > 0:
                await asyncio.sleep(wait_sec)
            arguments = (
                without_controls(call.arguments)
                if backgroundable(tool)
                else call.arguments
            )
            outcome = (
                tool.manage(self, arguments)
                if isinstance(tool, BackgroundTask)
                else tool.call(self.tool_context, arguments)
            )
            result = dataclasses.replace(await outcome, call_id=call.call_id)
        except asyncio.CancelledError:
            logger.debug(
                "runtime tool cancelled: call_id=%s tool=%s",
                call.call_id,
                call.name,
            )
            completed = False
            result = UncategorizedToolResult(
                call_id=call.call_id,
                content=self.markers.cancelled,
                is_error=True,
                stub="cancelled",
            )
        except Exception as exc:  # noqa: BLE001 -- Tool failures must be recorded without stopping the dispatch engine.
            logger.debug(
                "runtime tool failed: call_id=%s tool=%s error=%s",
                call.call_id,
                call.name,
                type(exc).__name__,
            )
            result = UncategorizedToolResult(
                call_id=call.call_id,
                content=describe(exc),
                is_error=True,
            )
        if call.call_id in self.detached:
            logger.debug(
                "runtime tool complete detached: call_id=%s tool=%s is_error=%s",
                call.call_id,
                call.name,
                result.is_error,
            )
            del self.detached[call.call_id]
            self.inbox.push_back(DetachedResult(result=result))
        else:
            logger.debug(
                "runtime tool complete cohort: call_id=%s tool=%s is_error=%s",
                call.call_id,
                call.name,
                result.is_error,
            )
            self.inbox.push_back(result)
        self._started.pop(call.call_id, None)
        self._wake_at.pop(call.call_id, None)
        return completed

    # A compaction cancelled by ``Halt`` or ``Clear`` has its failure published
    # already, so its late outcome is dropped.
    async def _compact_and_post(self, args: str) -> None:
        """Run the compactor and adopt its splice."""
        generation = self._compact_generation
        if self.compactor is None:
            self.inbox.push_back(CompactComplete())
            return
        tape_len = len(self.tape)
        tape_snapshot = tuple(self.tape)
        context_snapshot = self.context().messages
        try:
            override = await self.compactor.compact(
                tape_snapshot,
                context_snapshot,
                self.model,
                self.mint_ref,
                custom_instructions=args or None,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 -- Compaction adapter failures must be routed into the runtime error record.
            log_exception_or_warning(logger, "compaction failed", exc)
            if generation == self._compact_generation:
                self.inbox.push_back(
                    CompactFailed(exception=exc, tape_len=tape_len),
                )
            return
        if generation != self._compact_generation:
            return
        override = widen_barrier_mask(override, tape_snapshot)
        self.adopt_record(override, discards_content=True)
        self.inbox.push_back(CompactComplete.from_override(override))


def _widen_mask_ranges(
    mask: tuple[MaskRange, ...],
    tape: Sequence[TapeRecord],
) -> tuple[MaskRange, ...]:
    """Return ``mask`` widened over ``tape`` without filling existing gaps."""
    if not tape:
        return mask
    ordinals_by_session: dict[str, list[int]] = {}
    for record in tape:
        ordinals_by_session.setdefault(record.ref.session_id, []).append(
            record.ref.ordinal,
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


def _noted(system: tuple[str, ...], *, note: str) -> tuple[str, ...]:
    """Return ``system`` with ``note`` opening its first block."""
    if not system:
        return (note,)
    first, *rest = system
    return (f"{note}\n\n{first}", *rest)


def _entry_text(entry: ModelContextEvent) -> str:
    """Return the provider-visible text of one context entry."""
    if isinstance(entry, ToolResult):
        return entry.text()
    if isinstance(entry, TurnContext):
        return entry.text()
    if isinstance(entry, Response):
        return "".join(
            r.content or "" for r in entry.records if isinstance(r, AssistantMessage)
        )
    return entry.content or ""


# ``isdecimal``, not ``isdigit``: ``"²".isdigit()`` is true, and ``int`` refuses it.
def _mimic_index_of(cid: str, *, prefix: str) -> int | None:
    """Return the ``N`` of a ``<prefix>N`` id, or ``None``."""
    if not cid.startswith(prefix):
        return None
    suffix = cid.removeprefix(prefix)
    return int(suffix) if suffix.isdecimal() else None


def _type_names(types: Sequence[type[object]]) -> tuple[str, ...]:
    """Return class names for compact debug logging."""
    return tuple(t.__name__ for t in types)


def _item_names(items: Sequence[object]) -> tuple[str, ...]:
    """Return event class names for compact debug logging."""
    return tuple(type(item).__name__ for item in items)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RunningJob:
    """A tool call the runtime is running, as ``Runtime.jobs`` lists it."""

    call: ToolCall
    started: float
    state: Literal["running", "sleeping", "detached"]
    runtime: Runtime

    def cancel(self) -> None:
        """Ask the runtime to cancel the call."""
        self.runtime.push(Kill(call_id=self.call.call_id))


@dataclass(frozen=True, slots=True, kw_only=True)
class _PendingCommit:
    """What waits for the next model round to be committed.

    - ``"pairing"``: the answer to a ``DetachedArrived`` call the model made,
      inserted beside the call.
    - ``"forward"``: a detached call's result, delivered as a new arrival turn.
    """

    kind: Literal["pairing", "forward"]
    entry: AnyToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class _Hold:
    """What a running group of the model's calls touches."""

    keys: frozenset[str]
    after: tuple[_Hold, ...]
    """The earlier holders of any of it, which the group waits out first."""

    released: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


# ``"*"`` is a call whose resources are unknown, so it contends with every keyed call.
def _contend(keys: frozenset[str], held: frozenset[str]) -> bool:
    """Return whether calls touching ``keys`` must wait for ones touching ``held``."""
    return bool(keys & held) or "*" in keys or "*" in held


def _message_from_queued(
    item: UserQueuedMessage
    | UserDeferredMessage
    | AgentSendQueuedMessage
    | AgentSendDeferredMessage,
) -> UserMessage | AgentToAgentMessage:
    if isinstance(item, (AgentSendQueuedMessage, AgentSendDeferredMessage)):
        return AgentToAgentMessage(
            sender=item.sender,
            content=item.text,
            attachments=item.attachments,
        )
    return UserMessage(content=item.text, attachments=item.attachments)


def _forged(call: ToolCall, *, markers: Markers) -> bool:
    """Return whether the model made ``call`` as a delivery: under its id, or the old name."""
    return call.name == markers.arrived or call.call_id.endswith(markers.arrival)


def _fresh_ids(call_id: str, *, used: set[str]) -> Iterator[str]:
    """Yield ``call_id``, then ``<call_id>_<n>`` for each ``n`` not yet ``used``; mark each."""
    yield call_id
    for n in itertools.count(1):
        candidate = f"{call_id}_{n}"
        if candidate not in used:
            used.add(candidate)
            yield candidate


def _last_turn(messages: Sequence[ModelContextEvent]) -> int | None:
    """Return the index of the last entry that is someone's turn, not a note."""
    return next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if not isinstance(messages[i], (SystemMessage, TurnContext))
        ),
        None,
    )


def _flush_unmatched(
    pending: dict[str, None],
    unmatched: list[tuple[TapeRef | None, list[str]]],
    anchor: TapeRef | None,
) -> None:
    """Move pending tool ids into the repair list and clear the pending map."""
    if pending:
        unmatched.append((anchor, list(pending)))
    pending.clear()
