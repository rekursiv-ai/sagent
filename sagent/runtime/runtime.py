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
either.

A reply's calls run as a shell line. Calls sharing a ``serialize_key`` run as
``a ; b``, a group; other calls run beside them, as ``&``. A ``"*"`` call, whose
resources are unknown, runs after every keyed call before it, and every call
after it waits for it. Each call waits for those before it to stop, detached or
not; ``Kill`` stops one call, ``Detach`` the whole group. A call run in the
background is a trailing ``&``: it waits for what is before it, and nothing
waits for it. A later reply is a new line: it waits only for a killed call still
stopping.

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
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

import asyncio
import dataclasses
import itertools
import logging
import time

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.exceptions import ContextOverflowError, RetryableError
from sagent.model.token_accounting import TokenAccounting
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
    AgentToAgentMessage,
    AssistantMessage,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
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
from sagent.tools.types import ToolContext, settled


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

# How long ``Quit`` waits for the tools it cancelled to stop.
_QUIT_WAIT_SEC: Final = 10.0


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
        self.token_accounting = TokenAccounting()
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
        # Asked just before each model call, ``True`` for a turn's first: it may
        # assign another model, as a login whose quota reset.
        self.before_call: Callable[[bool], Awaitable[None]] | None = None
        # Asked when the model in use fails a call: it may assign another that can
        # get past the failure, which is then called at once; or say when, in
        # seconds since the epoch, a wait for it should end instead.
        self.after_failure: (
            Callable[[BaseException], Awaitable[float | None]] | None
        ) = None
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
        # Input sent to wait until the model is done: the queued is committed then,
        # the deferred at the idle after that.
        self._queued: list[UserQueuedMessage | AgentSendQueuedMessage] = []
        self._deferred: list[UserDeferredMessage | AgentSendDeferredMessage] = []
        # Whether the runtime waits for the user; input sent to wait is then
        # committed at once.
        self._awaiting_user = False  # pragma: no mutate -- None is as false.
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
        self._failed = False  # pragma: no mutate -- read for truth; None is as false.
        self.recheck_quota = False
        # The model the call in flight asked; a failure of one switched away from
        # says nothing of the model now in use.
        self._called_model: Model = model
        # Whether a long rate limit is waited out rather than handed to the user.
        self.persistent_retry = False
        # Where tools run, and what the model has read; one per agent.
        self.tool_context = tool_context or ToolContext(cwd=Path.cwd())
        # Each call the runtime runs, until its task ends.
        self._calls: dict[str, _Call] = {}
        # Bash calls that are other tools' work but no one call: the refusal.
        self._refusals: dict[str, str] = {}
        # Characters the model call in flight has streamed so far.
        self._streamed_chars = 0
        # ``Scheduled`` work waiting for a safe point.
        self._scheduled: list[Scheduled] = []
        # Whether a call may ask to run in the background, or after a delay.
        self.allow_background = True
        # Callers waiting on a background call's result instead of its delivery.
        self._foreground_waiters: dict[str, list[asyncio.Future[ToolResult]]] = {}
        # How long a call the model waits on may run before it is answered by a stub
        # and left to finish, its result delivered later; ``None`` waits for each.
        self.detach_after_sec: float | None = None
        # Each killed call still stopping, and its key: its process may still be
        # writing, so a later reply's call contending with it waits.
        self._stopping: dict[asyncio.Task[None], str | None] = {}
        # Each waited-on call's group, by call id: a detach of one detaches the
        # group.
        self._groups: dict[str, tuple[ToolCall, ...]] = {}

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
        """Tool calls not yet stopped, by call id: waited on, waiting to start, or apart."""
        return {
            cid: _RunningJob(
                call=c.call,
                started=c.started,
                state="running"
                if cid in self.cohort
                else "sleeping"
                if c.waiting
                else "detached",
                runtime=self,
            )
            for cid, c in self._calls.items()
        }

    async def foreground(self, call_id: str) -> ToolResult | None:
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
        waiter = asyncio.Future[ToolResult]()
        waiters = self._foreground_waiters.setdefault(call_id, [])
        waiters.append(waiter)
        try:
            return await waiter
        finally:
            waiters.remove(waiter)
            if not waiters and self._foreground_waiters.get(call_id) is waiters:
                del self._foreground_waiters[call_id]

    def detach[T](self, task: asyncio.Future[T] | None) -> None:
        """Answer the waited-on call ``task`` runs with a stub, and let it run on.

        As when it outlives ``detach_after_sec``: its real result arrives later.
        Nothing happens when background runs are off, or no waited call runs there.

        Args:
          task: The task running the call's tool, as ``asyncio.current_task()``
            in it; ``None``, outside any task, runs none.

        """
        if not self.allow_background:
            return
        for cid in self.running_tools:
            if self._calls[cid].result_future is task and cid in self.cohort:
                self.push(Detach(call_id=cid))
                return

    # Counting turns of the loop instead would not do: how many pass between a tool's
    # end and its result reaching the inbox is the runtime's own business.
    def push_after[T](
        self,
        task: asyncio.Future[T] | None,
        event: RuntimeEvent,
    ) -> None:
        """Queue ``event`` right behind the result of the call ``task`` runs.

        A tool that asks the runtime to act on its own agent, as to compact it,
        asks so: taken ahead of the result, the event would detach the call.

        Args:
          task: The task running the call's tool, as ``asyncio.current_task()``
            in it.
          event: What to queue; at once, when ``task`` runs no call of this
            runtime, as a call made by another agent.

        """
        for call in self._calls.values():
            if task is not None and call.result_future is task:
                call.events_after_result.append(event)
                return
        self.push(event)

    # A resumed context's detached calls ran in a process that has ended, so no
    # result will follow; and a clean quit's cancellations are never committed.
    def settle_detached(self) -> None:
        """Deliver a cancellation for each call answered only by a stub, running nowhere."""
        pending: list[str] = []
        answered: set[str] = set()
        for entry in self.context().messages:
            if not isinstance(entry, ToolResult):
                continue
            if entry.state == "pending":
                pending.append(entry.call_id)
            else:
                answered.add(entry.call_id.removesuffix(self.markers.arrival))
        for cid in pending:
            if cid in answered or cid in self.detached or cid in self.running_tools:
                continue
            self._append_detached_arrival(
                ToolResult(
                    call_id=cid,
                    state="cancelled",
                    shown=self.markers.cancelled,
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
                n = _mimic_index(cid, prefix=self.markers.mimic)
                if n is not None:
                    self._mimic_counter = max(self._mimic_counter, n + 1)

    # A caller of :meth:`foreground` takes the result as its own answer, so delivering
    # it forward as well would tell the model twice.
    def _deliver_detached(self, result: ToolResult) -> None:
        """Hand a detached call's result to whoever waits for it, else deliver it."""
        waiting = [
            w for w in self._foreground_waiters.get(result.call_id, ()) if not w.done()
        ]
        if not waiting or result.state == "pending":
            self._defer_detached_forward(result)
            return
        for waiter in waiting:
            waiter.set_result(result)
        self._forwarded_call_ids.add(result.call_id)

    # Delivered once per call. A stub is never delivered: it is not the tool's output,
    # and marking its call delivered would drop the real result. A cancellation whose
    # call is already answered cancelled, as a cohort ``Kill`` answers it, is not
    # delivered again; a background job's, answered only by its stub, is.
    def _defer_detached_forward(self, result: ToolResult) -> None:
        """Queue a completed detached tool's real result for forward delivery."""
        if result.state == "pending":
            logger.debug(
                "runtime detached forward skipped (pending stub, not real result):"
                " call_id=%s",
                result.call_id,
            )
            return
        if result.state == "cancelled" and self._inslot_result_is_terminal(
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
            and entry.state != "pending"
            for entry in entries
        )

    # A ``DetachedArrived`` call the model made itself is answered beside it on the
    # next real round, so it spends no round of its own.
    def _defer_pairing(self, result: ToolResult) -> None:
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
    def _commit_pairing(self, result: ToolResult) -> None:
        """Insert ``result`` after its ``AssistantMessage``, pairing the tool-use."""
        parent_ref = self._parent_assistant_refs.get(result.call_id)
        if parent_ref is None:
            # The call is gone with its turn, so its answer goes too.
            return
        self.append_splice(
            insert_after=parent_ref,
            payload=(result,),
            strategy="lazy_pairing",
            paired_externally=frozenset({result.call_id}),
        )
        self.publish(result)

    # After a model turn, a user turn naming the call comes first: two model turns
    # may not be adjacent.
    def _append_detached_arrival(self, result: ToolResult) -> None:
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
        while True:
            self._collect_detached()
            if self._fully_drained():
                self.publish(AgentIdle())
            items = await self.inbox.drain()
            failed_reply = False  # pragma: no mutate -- None is as false.

            for item_idx, item in enumerate(items):
                # One event that fails to dispatch is logged; the rest of
                # its batch still is.
                try:
                    match item:
                        case Quit():
                            await self._on_quit(items[item_idx + 1 :])
                            return

                        case Halt():
                            if self._on_halt(item, later=items[item_idx + 1 :]):
                                break

                        case Clear():
                            self._on_clear(items[item_idx + 1 :])
                            break

                        case ModelResponseError():
                            if await self._on_model_response_error(
                                item,
                                later=items[item_idx + 1 :],
                            ):
                                break

                        # Released from a failure's wait, or ending a suspension's:
                        # the gate calls the model for the turn.
                        case Retry():
                            self.recheck_quota = True
                            self._end_suspension()
                            self._failed = False  # pragma: no mutate -- read for truth; None is as false.
                            self._awaiting_user = False  # pragma: no mutate -- read for truth; None is as false.

                        case Kill():
                            self._on_kill(item)

                        # The group goes as one, whichever of its calls is named.
                        case Detach(call_id=cid) if cid in self._groups and any(
                            c.call_id in self.cohort for c in self._groups[cid]
                        ):
                            self._on_detach(item)

                        case Compact():
                            await self._on_compact(item)

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
                            self.recheck_quota = True
                            self._end_suspension()
                            if self.model_call is not None:
                                self._mid_stream_queue.append(item)
                            else:
                                self._stop_all_tools(mode="detach")
                                _ = self._drain_mid_stream_queue()
                                self._commit_user(item)
                            self._awaiting_user = False  # pragma: no mutate -- read for truth; None is as false.

                        case AgentToAgentMessage():
                            if self.model_call is not None:
                                self._mid_stream_queue.append(item)
                            else:
                                self._stop_all_tools(mode="detach")
                                _ = self._drain_mid_stream_queue()
                                self._commit_user(item)
                            self._awaiting_user = False  # pragma: no mutate -- read for truth; None is as false.

                        # A nudge to a busy agent is moot: it is nudged again
                        # at its next idle if it still owes.
                        case Nudge(content=content):
                            if self.model_call is None:
                                self.append_history(
                                    SystemMessage(content=content, subtype="nudge"),
                                )
                            self._awaiting_user = False  # pragma: no mutate -- read for truth; None is as false.

                        # Input that ends the wait for the user is the user
                        # resuming; held for idle, the halted turn would be
                        # sent again without it.
                        case (
                            UserQueuedMessage()
                            | AgentSendQueuedMessage()
                            | UserDeferredMessage()
                            | AgentSendDeferredMessage()
                        ) if self._awaiting_user:
                            self._commit_user(_message_from_queued(item))
                            self._awaiting_user = False  # pragma: no mutate -- read for truth; None is as false.

                        case UserQueuedMessage() | AgentSendQueuedMessage():
                            self._queued.append(item)

                        case UserDeferredMessage() | AgentSendDeferredMessage():
                            self._deferred.append(item)

                        case ModelResponsePartial():
                            self.publish(item)

                        case ModelResponseThinking():
                            self.publish(item)

                        case ModelResponseComplete():
                            self._on_model_response_complete(item)

                        case ModelResponseCancelled():
                            self.model_call = None
                            self.publish(item)

                        case ToolResult() if item.call_id in self.cohort:
                            cid = item.call_id
                            self.cohort.discard(cid)
                            del self.running_tools[cid]
                            self.append_history(item)
                            self.publish(item)

                        case ToolResult() if item.call_id in self.detached:
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

                        case _:  # pragma: no mutate -- basedpyright asks for the default; without it nothing else matches either.
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
                await self._advance()
            except Exception as exc:  # noqa: BLE001 -- The loop must outlive one pass's failure.
                log_exception_or_warning(logger, "dispatch loop iteration raised", exc)

    async def _on_quit(self, later: Sequence[RuntimeEvent]) -> None:
        """Cancel everything in flight; wait for the cancelled tools to stop."""
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
        stopped = [
            *self.running_tools.values(),
            *self.detached.values(),
            *self._stopping,
        ]
        for t in self.running_tools.values():
            t.cancel()
        # Background work too, or it runs on after shutdown.
        self._cancel_detached()
        # Work no loop will reach: its callers stop waiting.
        for event in (*self._scheduled, *later, *self.inbox.drain_nowait()):
            if isinstance(event, Scheduled):
                _ = event.done.cancel()
        if self._retry_timer is not None:
            self._retry_timer.cancel()
        # A cancelled tool takes a moment to stop, as a shell does. The loop ends
        # when they have all stopped, or after ``_QUIT_WAIT_SEC``: a tool that
        # ignores its cancel must not keep the process from exiting.
        if stopped:
            _, still_running = await asyncio.wait(stopped, timeout=_QUIT_WAIT_SEC)
            if still_running:
                logger.warning(
                    "runtime quit: %d tool(s) did not stop",
                    len(still_running),
                )

    def _on_halt(self, item: Halt, *, later: Sequence[RuntimeEvent]) -> bool:
        """Cancel the model call and any compaction; return whether ``later`` now waits for the user."""
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
        # The user took the turn over: what a failure waited for no longer calls
        # the model.
        self._end_suspension()
        self._failed = False  # pragma: no mutate -- read for truth; None is as false.
        # Tools keep running. What the user typed during the stream is what the
        # runtime would wait for.
        if self._drain_mid_stream_queue():
            return False
        self.inbox.push_front(Await.user(), *later)
        self._awaiting_user = True
        return True

    def _on_clear(self, later: Sequence[RuntimeEvent]) -> None:
        """Stop everything, hide the tape, and wait for the user."""
        self._cancel_model_and_compaction()
        self._stop_all_tools(mode="detach")
        # With no call registered, a late result matches nothing and is dropped.
        self._cancel_detached()
        self._pending_commits.clear()
        self._forwarded_call_ids.clear()
        self._queued.clear()
        self._deferred.clear()
        self._mid_stream_queue.clear()
        self._refusals.clear()
        # A new conversation does not inherit the old one's failure, or its wait
        # for a rate limit.
        self._end_suspension()
        self._failed = False  # pragma: no mutate -- read for truth; None is as false.
        self.append_clear()
        self.inbox.push_front(
            Await.user(),
            # A tool that finished as the user cleared posted its result behind
            # the ``Clear``, or waits behind a Halt; delivering it would carry the
            # wiped session's work into the fresh one.
            *(
                i
                for i in (*later, *self.inbox.drain_nowait())
                if not isinstance(i, DetachedResult)
            ),
        )
        # After the gate: input an observer pushes in reply must count as new, or
        # it never releases the gate.
        self.publish(ClearComplete())
        self._awaiting_user = True

    async def _on_model_response_error(
        self,
        item: ModelResponseError,
        *,
        later: Sequence[RuntimeEvent],
    ) -> bool:
        """Compact, suspend, end the turn, or drop the failure; return whether ``later`` was put back."""
        exc = item.exception
        wake = (
            await self._after_failure(exc) if self._called_model is self.model else None
        )
        # A failure of a model switched away from says nothing of the one in use:
        # the gate asks it, at once.
        if self._called_model is not self.model:
            self.model_call = None
            _ = self._drain_mid_stream_queue()
            return False
        # A request too long for the window is compaction's job: compact,
        # and let the gate call again, under the same breaker.
        if (
            isinstance(exc, ContextOverflowError)
            and self.compactor is not None
            and self._auto_compactions < 2
        ):
            logger.warning("model context overflowed; compacting: %s", exc)
            self.model_call = None
            self._auto_compactions += 1
            self.inbox.push_front(Compact(trigger="auto"), *later)
            return True
        # A long rate limit or outage, for an agent that waits them out: hold the
        # model call until the advised time, backing off to 5 minutes, then call
        # again; 60 in a row end the turn.
        if (
            isinstance(exc, RetryableError)
            and self.persistent_retry
            and self._retries < 60
        ):
            self.model_call = None
            delay = (
                max(exc.retry_after_sec, min(5.0 * (1 << self._retries), 300.0))
                if wake is None
                else max(wake - time.time(), 0.0)
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
            # Input typed during the failed stream is part of what the retry asks.
            _ = self._drain_mid_stream_queue()
            # Any input wakes the loop; ``SaveSession`` changes nothing else.
            self._retry_timer = asyncio.get_running_loop().call_later(
                delay,
                self.inbox.push_back,
                SaveSession(),
            )
            return False
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
            self.inbox.push_front(Await.failure(), *later)
            self._awaiting_user = True
            return True
        return False

    def _on_kill(self, item: Kill) -> None:
        """Cancel the call ``item`` names, or every call."""
        cid = item.call_id
        if cid is None:
            logger.debug(
                "runtime kill all tools: running_tools=%d cohort=%d detached=%d",
                len(self.running_tools),
                len(self.cohort),
                len(self.detached),
            )
            self._stop_all_tools(mode="kill")
            self._cancel_detached()
        elif cid in self.running_tools:
            logger.debug("runtime kill tool: call_id=%s", cid)
            self._stop_tool(cid, self.running_tools.pop(cid), mode="kill")
            self.cohort.discard(cid)
        else:
            # Its own late result then matches nothing; the model is told it was
            # cancelled instead.
            task = self.detached.pop(cid, None)
            if task is not None:
                logger.debug("runtime kill detached tool: call_id=%s", cid)
                self._kill_detached(cid, task)
            else:
                logger.debug("runtime kill missed tool: call_id=%s", cid)

    def _on_detach(self, item: Detach) -> None:
        """Answer each waited-on call of the group with a stub, and let it run on."""
        for member in self._groups[item.call_id]:
            mid = member.call_id
            if mid not in self.cohort:
                continue
            logger.debug(
                "runtime tool detached after %ss: call_id=%s",
                self.detach_after_sec,
                mid,
            )
            self._stop_tool(mid, self.running_tools.pop(mid), mode="detach")
            self.cohort.discard(mid)
        if not self.cohort:
            self._cohort_seen = False  # pragma: no mutate -- None is as false.

    async def _on_compact(self, item: Compact) -> None:
        """Start a compaction, unless one runs; what the user typed goes into it."""
        if self.compact_task and not self.compact_task.done():
            logger.debug("runtime compact ignored while compact_task active")
            return
        self._compaction = item.trigger
        logger.debug(
            "runtime compact start: model_call=%s "
            "running_tools=%d cohort=%d detached=%d args=%r",
            self.model_call is not None,
            len(self.running_tools),
            len(self.cohort),
            len(self.detached),
            item.args,
        )
        self._cancel_model()
        self._stop_all_tools(mode="detach")
        # A compaction calls the model too: not on a spent login.
        await self._before_call(turn_start=False)
        # What the user typed goes into what is compacted; only ``Clear`` discards it.
        for staged in self._queued:
            self._commit_user(_message_from_queued(staged))
        self._queued.clear()
        _ = self._drain_mid_stream_queue()
        self.compact_task = asyncio.create_task(self._compact_and_post(item.args))
        self.compact_task.add_done_callback(
            log_task_exception(logger, "compaction task crashed"),
        )
        started = CompactStarted()
        self.append_history(started)
        self.publish(started)

    def _on_model_response_complete(self, item: ModelResponseComplete) -> None:
        """Record the reply and start its calls, unless a newer call replaced its own."""
        if item.generation not in (-1, self._model_call_generation):
            logger.debug(
                "runtime stale model response ignored: generation=%d current=%d",
                item.generation,
                self._model_call_generation,
            )
            return
        self._auto_compactions = 0
        self._retries = 0
        self.model_call = None
        msg = self._rewrite_calls(self._sanitize_forged_arrivals(item.response))
        self._gate_calls(msg)
        cut_in = (
            self.before_tool_spawn(msg) if self.before_tool_spawn is not None else None
        )
        self.append_history(msg)
        # Observers see the turn as recorded, rewrites included.
        self.publish(dataclasses.replace(item, response=msg))
        if cut_in is not None:
            self._mid_stream_queue.append(cut_in)
        if calls(msg.records):
            # What the user sent as the reply streamed is answered next; the reply's
            # calls run on, detached, and no wait on them begins.
            cut = bool(self._mid_stream_queue)
            self._start_calls(calls(msg.records), waited=not cut)
            if cut:
                self._stop_all_tools(mode="detach")
                _ = self._drain_mid_stream_queue()
        elif not self._drain_mid_stream_queue():
            self.publish(ModelIdle())

    async def _advance(self) -> None:
        """Run what a batch leaves due: scheduled work, staged input, a model call."""
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
            self._cohort_seen = False  # pragma: no mutate -- None is as false.

        # What the user sent as the reply streamed, if no ending of the reply
        # took it in already.
        if self._mid_stream_queue and self.model_call is None and not self.cohort:
            _ = self._drain_mid_stream_queue()

        # Staged input waits until the model is done, not merely between
        # rounds: a tool result still to answer means it is not.
        if (
            self._engine_quiescent
            and not self._should_call_model()
            and (self._queued or self._deferred)
        ):
            # Queued first; deferred waits for the idle after it.
            staged = self._queued or self._deferred
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
            and (
                self.service_suspended_until is None
                or time.time() >= self.service_suspended_until
            )
        ):
            await self._call_model()

        self.publish(SaveSession())

    async def _call_model(self) -> None:
        """Start a model call; a context past repair waits for the user instead."""
        if self.context_updates is not None:
            for note in self.context_updates():
                self.append_history(note)
        logger.debug(
            "runtime model call start: history=%d detached=%d queued=%d",
            len(self.context().messages),
            len(self.detached),
            len(self._queued),
        )
        exc = self._context_validation_error()
        if exc is not None:
            # Raised, it would repeat every pass on the same tape; the user is told,
            # and the runtime waits for them.
            log_exception_or_warning(
                logger,
                "model-call gate context unrepairable",
                exc,
            )
            # The gate first, as for ``Clear``.
            self.inbox.push_front(Await.recovery())
            self.publish(ModelResponseError(exception=exc))
            self._awaiting_user = True
        else:
            # Whatever released a failure's wait, or ended a suspension's, the turn
            # goes on, on the model asked now.
            self._failed = False  # pragma: no mutate -- None is as false.
            self.service_suspended_until = None
            await self._before_call(turn_start=self._steps == 0)
            self._called_model = self.model
            model_call: asyncio.Task[None] = asyncio.create_task(
                self._stream_and_post(self._next_generation()),
            )
            self.model_call = model_call
            model_call.add_done_callback(
                log_task_exception(logger, "model-call task crashed"),
            )
            self.publish(ModelCallStarted())

    # The hooks are advisory: one that raises is logged, and the runtime goes on
    # as without it, or the turn would hang with no call in flight to end it.
    async def _before_call(self, *, turn_start: bool) -> None:
        """Ask ``before_call`` about the next call; a failure of its own is logged."""
        if self.before_call is None:
            return
        try:
            await self.before_call(turn_start)
        except Exception as exc:  # noqa: BLE001 -- An advisory hook must not stop the call.
            log_exception_or_warning(logger, "before_call raised", exc)

    async def _after_failure(self, exc: BaseException) -> float | None:
        """Ask ``after_failure`` about ``exc``; a failure of its own is logged, as ``None``."""
        if self.after_failure is None:
            return None
        try:
            return await self.after_failure(exc)
        except Exception as raised:  # noqa: BLE001 -- An advisory hook must not hang the turn.
            log_exception_or_warning(logger, "after_failure raised", raised)
            return None

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
        am_repair_anchor: TapeRef | None = None  # pragma: no mutate -- unread till set.
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
        _flush_unmatched(pending, unmatched_per_am, am_repair_anchor)

        for anchor, missing_ids in unmatched_per_am:
            if anchor is None:
                continue
            payload = tuple(
                ToolResult(call_id=cid, state="failed", stderr=("[interrupted]",))
                for cid in missing_ids
            )
            # Their calls are in the recorded turn at ``anchor``.
            self.append_splice(
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
        match mode:
            case "kill":
                state, shown = "cancelled", self.markers.cancelled
            case "detach":
                state, shown = "pending", self.markers.detached
        stub = ToolResult(call_id=cid, state=state, shown=shown)
        self.append_history(stub)
        self.publish(stub)
        self.detached[cid] = task
        if mode == "kill":
            self._stop(cid, task)

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

    # The line the module docstring describes. Each call is its own task, waiting
    # on the tasks of the earlier calls it contends with, so a ``Kill`` stops one
    # call and the next starts once it has stopped. ``waited`` false starts the
    # cohort only to detach it at once: no wait begins, so ``CohortStarted`` is not
    # told.
    def _start_calls(self, tool_calls: Sequence[ToolCall], *, waited: bool) -> None:
        """Run a reply's calls, each after the earlier ones it contends with."""
        delays = {
            tc.call_id: delay
            for tc in tool_calls
            if (delay := self._background_delay(tc)) is not None
        }
        cohort = [tc for tc in tool_calls if tc.call_id not in delays]
        if cohort:
            self._cohort_seen = True
            if waited:
                self.publish(CohortStarted())
            logger.debug(
                "runtime cohort start: tools=%s",
                [(tc.call_id, tc.name) for tc in cohort],
            )
            self.cohort.update(tc.call_id for tc in cohort)
        now = time.time()
        waited_on: list[tuple[str | None, asyncio.Task[None]]] = []
        tasks: dict[str, asyncio.Task[None]] = {}
        groups: list[list[ToolCall]] = []
        by_key: dict[str, list[ToolCall]] = {}
        for tc in tool_calls:
            cid = tc.call_id
            key = self._serialize_key(tc)
            delay = delays.get(cid)
            # A call answered without running waits for nothing.
            after = (
                []
                if cid in self._refusals or tc.name not in self.tools
                else [
                    *(t for k, t in waited_on if _contend(key, k)),
                    *(t for t, k in self._stopping.items() if _contend(key, k)),
                ]
            )
            task = asyncio.create_task(
                self._run_queued(
                    tc,
                    after=after,
                    wake_at=None if delay is None else now + delay,
                ),
            )
            task.add_done_callback(
                log_task_exception(
                    logger,
                    f"{'cohort' if delay is None else 'background'} tool"
                    f" {tc.name!r} crashed",
                ),
            )
            task.add_done_callback(lambda _, cid=cid: self._forget(cid))
            self._calls[cid] = _Call(call=tc, started=now, key=key)
            tasks[cid] = task
            if delay is not None:
                self._start_background(tc, task)
                continue
            self.running_tools[cid] = task
            waited_on.append((key, task))
            if key == "*":
                groups.extend(by_key.values())
                by_key = {}
            if key is None or key == "*":
                groups.append([tc])
            else:
                by_key.setdefault(key, []).append(tc)
        groups.extend(by_key.values())
        for members in groups:
            self._start_group(tuple(members), [tasks[m.call_id] for m in members])

    def _forget(self, call_id: str) -> None:
        """Drop what the runtime knows of ``call_id``, once its task has ended."""
        del self._calls[call_id]

    # A group is what detaches as one: the calls of one key between ``"*"`` calls.
    def _start_group(
        self,
        group: tuple[ToolCall, ...],
        tasks: Sequence[asyncio.Task[None]],
    ) -> None:
        """Know ``group`` by each of its calls until they have all stopped."""
        for tc in group:
            self._groups[tc.call_id] = group
        done = asyncio.gather(*tasks, return_exceptions=True)
        done.add_done_callback(lambda _: self._end_group(group))
        self._detach_later(group, done)

    def _end_group(self, group: tuple[ToolCall, ...]) -> None:
        """Forget ``group``, once every call in it has stopped."""
        for tc in group:
            # A later reply may reuse the id while this group runs on, detached.
            if self._groups.get(tc.call_id) is group:
                del self._groups[tc.call_id]

    # A tool that manages jobs, such as waiting for one, is waited on to its end.
    def _detach_later(
        self,
        group: tuple[ToolCall, ...],
        done: asyncio.Future[list[BaseException | None]],
    ) -> None:
        """Detach ``group`` if it outlives ``detach_after_sec``."""
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
        done.add_done_callback(lambda _: timer.cancel())

    def _background_delay(self, call: ToolCall) -> float | None:
        """Return the delay ``call`` asks to run in the background after, or ``None``."""
        tool = self.tools.get(call.name)
        if tool is None or not self.allow_background or not backgroundable(tool):
            return None
        try:
            return background_delay(call.arguments)
        except (TypeError, ValueError) as exc:
            self._refusals[call.call_id] = str(exc)
            return None

    def _start_background(self, call: ToolCall, task: asyncio.Task[None]) -> None:
        """Answer ``call``, which ``task`` runs apart, with a stub."""
        stub = ToolResult(
            call_id=call.call_id,
            state="pending",
            shown=f"{self.markers.running}{call.name} as job {call.call_id}]",
        )
        self.append_history(stub)
        self.publish(stub)
        self.detached[call.call_id] = task

    def _stop(self, call_id: str, task: asyncio.Task[None]) -> None:
        """Cancel ``task``; until it stops, a later contending call waits for it."""
        # A task already done, its result still queued, has nothing to stop.
        if task.cancel():
            self._stopping[task] = self._calls[call_id].key
            task.add_done_callback(lambda _: self._stopping.pop(task, None))

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
        self._stop(cid, task)
        self._deliver_detached(
            ToolResult(call_id=cid, state="cancelled", shown=self.markers.cancelled),
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

    def request(
        self,
        *,
        after: Sequence[SessionRecord] = (),
        max_response_tokens: int | None = None,
    ) -> Request:
        """Return the request the next model call sends, with ``after`` appended.

        Its prefix is the last call's, byte for byte, so a side call such as a
        summary that appends to it reads the cache that call wrote.

        Args:
          after: Records to add after the context.
          max_response_tokens: Explicit cap; ``None`` asks ``reply_room``, while
            ``0`` bypasses it for sizing without recursion.

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
                max_response_tokens=max_response_tokens
                if max_response_tokens is not None
                else self.reply_room()
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
        request = self.sending(
            dataclasses.replace(self.request(), recheck_quota=self.recheck_quota),
        )
        self.recheck_quota = False
        snapshot = self.token_accounting.capture(request, model=self._called_model)
        # Only the first call after a compaction is told of it.
        self._compaction = ""
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
        return dataclasses.replace(
            response,
            measurement=self.token_accounting.observe(snapshot, tokens=response.tokens),
        )

    def _serialize_key(self, call: ToolCall) -> str | None:
        """Return the serialization key, refusing calls whose scheduling fails."""
        if call.call_id in self._refusals:
            return None
        tool = self.tools.get(call.name)
        if tool is None:
            return None
        try:
            return tool.serialize_key(
                self.tool_context,
                without_controls(call.arguments)
                if backgroundable(tool)
                else call.arguments,
            )
        except Exception as exc:  # noqa: BLE001 -- Scheduling failures must answer the call without stopping dispatch.
            self._refusals[call.call_id] = describe(exc)
            return None

    # It waits on tasks, not a lock, so order never rests on lock fairness. A call
    # killed before it starts was answered by whatever killed it.
    async def _run_queued(
        self,
        call: ToolCall,
        *,
        after: Sequence[asyncio.Task[None]],
        wake_at: float | None,
    ) -> None:
        """Run ``call`` once ``after`` has stopped and ``wake_at`` has passed."""
        if after:
            _ = await asyncio.wait(after)
        if wake_at is not None:
            await asyncio.sleep(
                wake_at - time.time(),
            )  # house-ignore[sleep] -- The schedule.
        self._calls[call.call_id].waiting = False
        await self._run_tool_and_post(call)

    # A failure or a cancellation is still a result, so every call is answered; a call
    # detached meanwhile is delivered forward.
    async def _run_tool_and_post(
        self,
        call: ToolCall,
    ) -> None:
        """Run one tool invocation, post the ``ToolResult`` to the inbox."""
        if call.name == self.markers.arrived:
            # No tool has this name; the model copied a delivery. Its answer waits
            # for the next real round, so it spends no round of its own.
            logger.debug("runtime detached-arrived mimic: call_id=%s", call.call_id)
            self.inbox.push_back(
                LazyEvent(
                    payload=ToolResult(
                        call_id=call.call_id,
                        state="failed",
                        stderr=(
                            (
                                f"{self.markers.arrived} is not callable; detached "
                                "results arrive automatically."
                            ),
                        ),
                    ),
                ),
            )
            return
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
                ToolResult(
                    call_id=call.call_id,
                    state="failed",
                    stderr=(
                        refused
                        if refused is not None
                        else f"Unknown tool: {call.name}",
                    ),
                ),
            )
            return
        result: ToolResult
        try:
            logger.debug(
                "runtime tool start: call_id=%s tool=%s",
                call.call_id,
                call.name,
            )
            arguments = (
                without_controls(call.arguments)
                if backgroundable(tool)
                else call.arguments
            )
            if isinstance(tool, BackgroundTask):
                result = dataclasses.replace(
                    await tool.manage(self, arguments),
                    call_id=call.call_id,
                )
            else:
                streaming_result = tool.call(self.tool_context, arguments)
                self._calls[call.call_id].result_future = streaming_result.result
                # A cancel here stops the tool; ``settled`` still returns its output.
                result = await settled(streaming_result, call_id=call.call_id)
        except asyncio.CancelledError:
            result = ToolResult(call_id=call.call_id, state="cancelled")
        except Exception as exc:  # noqa: BLE001 -- Tool failures must be recorded without stopping the dispatch engine.
            result = ToolResult(
                call_id=call.call_id,
                state="failed",
                stderr=(describe(exc),),
            )
        if result.state == "failed":
            # A failure made here ends ``stderr`` with ``describe(error)``; a record
            # made elsewhere may hold none. Only the error's type is logged: the
            # message may hold the tool's output.
            logger.debug(
                "runtime tool failed: call_id=%s tool=%s error=%s",
                call.call_id,
                call.name,
                str(result.stderr[-1]).partition(":")[0] if result.stderr else "",
            )
        if result.state == "cancelled":
            logger.debug(
                "runtime tool cancelled: call_id=%s tool=%s",
                call.call_id,
                call.name,
            )
            result = dataclasses.replace(result, shown=self.markers.cancelled)
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
        for event in self._calls[call.call_id].events_after_result:
            self.inbox.push_back(event)

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
def _mimic_index(cid: str, *, prefix: str) -> int | None:
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
    entry: ToolResult


@dataclass(slots=True, kw_only=True)
class _Call:
    """A call the runtime runs, as ``Runtime.jobs`` lists it."""

    call: ToolCall
    started: float
    """When it was asked for, in seconds since the epoch."""

    key: str | None
    """What it touches, its ``serialize_key``."""

    waiting: bool = True
    """Whether it still waits for the calls before it, or for its delay."""

    result_future: asyncio.Future[object] | None = None
    """The future of the tool's result, set once the tool starts. For most tools it
    is the task the tool runs in, so the tool sees it as ``asyncio.current_task()``."""

    events_after_result: list[RuntimeEvent] = dataclasses.field(
        default_factory=list[RuntimeEvent],
    )
    """Events to queue right after the call's result; see ``Runtime.push_after``."""


# ``"*"`` is a call whose resources are unknown: it waits for every keyed call before
# it, and every call after it waits for it. An unkeyed call touches nothing another
# does, so nothing waits for it.
def _contend(key: str | None, earlier: str | None) -> bool:
    """Return whether a call keyed ``key`` must wait for an earlier one keyed ``earlier``."""
    return earlier is not None and (key in (earlier, "*") or earlier == "*")


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
