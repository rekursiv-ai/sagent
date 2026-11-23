"""The agent's event loop: one inbox, one ``match``, one model-call gate.

Laws:

1. Every request ends once, told as its ``...Completed``; ``Runtime.ask`` hands
   whoever asked it how.
2. What waits before it is acted on waits in one list, each item with the
   published state it waits for.
3. Whether the model and the agent are idle is worked out from state
   (``model_idle``, ``agent_idle``), never kept.
4. The inbox gate is the only wait for the user.
5. The tape only grows (``tape.py``); ``context.py`` resolves what the model sees.

``Runtime.run_forever`` drains the inbox, dispatches each event, then calls the
model when the context ends in something to answer and nothing is at work: no
reply streaming, no compaction, no call the model waits on, no wait for the user.

A reply's calls run as a shell line. Calls sharing a ``serialize_key`` run as
``a ; b``, a group; other calls run beside them, as ``&``. A ``"*"`` call, whose
resources are unknown, runs after every keyed call before it, and every call
after it waits for it. A call run in the background is a trailing ``&``. A later
reply waits only for a killed call still stopping.

The model waits on a reply's calls unless one asks to run apart (``background``,
``delay``). A waited call is detached, answered with a stub and left running, at
its tool's first wake, at ``ToolDetach`` (its whole group), or when the user
interrupts: a message, ``ModelHalt``, ``SessionCompact``, ``SessionClear``. A
call's stub is its first ``ToolMessage``; what it writes later, at each wake,
and its result are more ``ToolMessage``s of its id, which the request sends as
the user's side (``session.types.as_sent``), so nothing the model has read is
rewritten. Later wakes tell an idle model what a call wrote since it last looked;
a call's ``wake_every`` wakes it on that period whatever it wrote. ``ToolHalt``
cancels calls.

What the user sends while the model streams is answered next: the reply's calls
start detached. ``ModelHalt`` stops the reply, detaches the waited calls, and
waits for the user. A failed model call goes on by itself where it can: on a
model switched to meanwhile, after compacting a request too long, or after a
``ModelResponseSuspended`` wait under ``persistent_retry``. Any other failure
stops the turn (``ModelIdle`` says how), which waits for the user or a
``ModelRetry``.

Providers need user and model turns to alternate, and each call answered before
the next turn: the request merges adjacent user-side turns, and the model-call
gate repairs what it cannot keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

import asyncio
import dataclasses
import itertools
import logging
import math
import time

from sagent.model.exceptions import (
    ContextOverflowError,
    DisconnectedError,
    RetryableError,
)
from sagent.model.token_accounting import TokenAccounting, bound, stamped
from sagent.model.types import Request, Response, TextChunk

# Engine consumes a subset of the runtime event vocabulary. No re-export shim.
from sagent.runtime.context import Tape, stubbed, widen_barrier_mask
from sagent.runtime.exceptions import (
    UserFacingError,
    describe,
    log_exception_or_warning,
    log_task_exception,
)
from sagent.runtime.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    calls,
    full_tape_mask,
    sent_records,
    unpaired_call_ids,
    unreadable,
)
from sagent.runtime.types import (
    AgentMessageCompleted,
    Job,
    ModelContextEvent,
    ModelHalt,
    ModelHaltCompleted,
    ModelMessage,
    ModelMessageCompleted,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelResponseSuspended,
    ModelResponseTextStreamed,
    ModelResponseThinkingStreamed,
    ModelRetry,
    ModelRetryCompleted,
    RuntimeEvent,
    SessionClear,
    SessionClearCompleted,
    SessionCompact,
    SessionCompactCompleted,
    SessionCompacted,
    SessionCompactStarted,
    SessionContextUpdate,
    SessionContextUpdateCompleted,
    SessionPrune,
    SessionPruneCompleted,
    SessionQuit,
    SessionQuitCompleted,
    SessionSaveDue,
    Stubs,
    SystemMessageCompleted,
    ToolCallCompleted,
    ToolCohortCompleted,
    ToolCohortStarted,
    ToolDetach,
    ToolDetachCompleted,
    ToolHalt,
    ToolHaltCompleted,
    ToolMessageCompleted,
    ToolWakeSchedules,
    UserMessageCompleted,
    WakeSchedule,
)
from sagent.session.types import (
    AgentIdle,
    AgentMessage,
    AssistantMessage,
    BackgroundTaskResult,
    Cancelled,
    ContextCompaction,
    Failed,
    ModelIdle,
    Outcome,
    SessionContext,
    Succeeded,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
    now_timestamp,
)
from sagent.tools.background_task import (
    BackgroundTask,
    background_delay,
    backgroundable,
    controls,
    offered,
    wake_period,
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

# How long ``SessionQuit`` waits for the tools it cancelled to stop.
_QUIT_WAIT_SEC: Final = 10.0

# How long before the prompt cache lapses a wake comes, to reach the model in time.
_CACHE_MARGIN_SEC: Final = 30.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Await:
    """Drain gate: hold the inbox until an item of ``types`` arrives.

    ``SessionQuit`` always releases it; items of ``passes`` are handed over at
    once, the gate still set.
    """

    types: tuple[type, ...]
    passes: tuple[type, ...] = ()

    @classmethod
    def user(cls) -> Await:
        """Return the wait for the user's next message.

        A verb acts meanwhile, and so does the compaction it starts; only the
        user's message lets the model answer again.

        Returns:
          gate: Released by a user-side message or ``SessionQuit``.

        """
        return cls(
            types=(UserMessage, AgentMessage, SystemMessage, SessionQuit),
            passes=(
                ModelHalt,
                ToolHalt,
                SessionClear,
                SessionCompact,
                SessionCompactCompleted,
            ),
        )

    @classmethod
    def failure(cls) -> Await:
        """Return the wait after a failed model call ended the turn.

        Returns:
          gate: Released as ``user`` is, and by a ``ModelRetry``: a new model or a
            sign-in may get past the failure.

        """
        user = cls.user()
        return cls(types=(*user.types, ModelRetry), passes=user.passes)

    @classmethod
    def recovery(cls) -> Await:
        """Return the wait after a context the gate cannot repair.

        Returns:
          gate: Released also by a verb that can repair the context.

        """
        return cls(
            types=(
                *cls.user().types,
                SessionClear,
                SessionCompact,
                ModelHalt,
                ToolHalt,
            ),
        )


class GatedDeque[T]:
    """An async queue whose drain can wait for a kind of item.

    A gate holds the queue until an item of its ``types`` arrives that was not
    queued when the gate was set, or a ``SessionQuit``. Counting only new ones keeps a
    ``ModelHalt`` waiting for a fresh message when the user typed several while the
    agent was busy.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self.awaiting: Await | None = None
        self._gate_baseline: int = 0

    @property
    def gate_armed(self) -> bool:
        """True while a gate holds the queue."""
        return self.awaiting is not None

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

    def push_front(self, *items: T, gate: Await | None = None) -> None:
        """Queue ``items`` at the front, in order; ``gate``, if given, holds the drain.

        Args:
          items: What to queue; they count as new to ``gate``.
          gate: What the drain waits for; only items queued before the call
            count against it.

        """
        old = self.drain_nowait()
        if gate is not None:
            self.awaiting = gate
            self._gate_baseline = sum(
                1
                for i in old
                if isinstance(i, gate.types) and not isinstance(i, SessionQuit)
            )
            logger.debug(
                "runtime inbox gate armed: gate=%s baseline=%d queued=%s",
                _type_names(gate.types),
                self._gate_baseline,
                _type_names([type(i) for i in old]),
            )
        for item in (*items, *old):
            self._queue.put_nowait(item)

    async def drain(self) -> list[T]:
        """Block until items are available, then return all queued items.

        When a gate is set, keeps draining until a NEW item matching
        the gate types arrives (count exceeds the baseline captured at
        arm time) or until any ``SessionQuit`` is observed. Items the gate
        passes are returned as they arrive, the gate still set, and the
        rest keep their order behind it.

        Returns:
          items: All queued items in arrival order.

        """
        first = await self._queue.get()
        items = [first, *self.drain_nowait()]
        if self.awaiting is None:
            return items
        gate = self.awaiting.types
        passes = self.awaiting.passes
        baseline = self._gate_baseline
        while not self._gate_satisfied(items, gate, baseline):
            passing = [i for i in items if isinstance(i, passes)]
            if passing:
                for item in items:
                    if not isinstance(item, passes):
                        self._queue.put_nowait(item)
                return passing
            items.append(await self._queue.get())
            items.extend(self.drain_nowait())
        logger.debug(
            "runtime inbox gate released: gate=%s baseline=%d buffered=%s",
            _type_names(gate),
            baseline,
            _type_names([type(i) for i in items]),
        )
        # Arming sets the baseline afresh; unarmed, it is not read.
        self.awaiting = None
        return items

    @classmethod
    def _gate_satisfied(
        cls,
        items: Sequence[object],
        gate: tuple[type, ...],
        baseline: int,
    ) -> bool:
        """Return True when ``items`` satisfies the gate beyond ``baseline``."""
        if any(isinstance(i, SessionQuit) for i in items):
            return True
        count = sum(
            1 for i in items if isinstance(i, gate) and not isinstance(i, SessionQuit)
        )
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

        A replacement keeping a call whose result lies outside it, or a result
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
      compactor: What ``SessionCompact`` runs; ``None`` compacts nothing.
      session_id: Namespace of this runtime's tape refs.
      tool_context: Where tools run, and what the model has read.
      stubs: What the runtime writes for the calls it answers itself.

    """

    def __init__(
        self,
        *,
        model: Model,
        tools: Sequence[Tool] | None = None,
        compactor: Compactor | None = None,
        session_id: str = "",
        tool_context: ToolContext | None = None,
        stubs: Stubs | None = None,
    ) -> None:
        self._model: Model = model
        self.token_accounting = TokenAccounting()
        self.stubs: Stubs = stubs if stubs is not None else Stubs()
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            if t.name in self.tools:
                raise ValueError(f"Duplicate tool name: {t.name!r}")
            self.tools[t.name] = t
        self.compactor = compactor
        # Where tools run, and what the model has read; one per agent.
        self.tool_context = tool_context or ToolContext(cwd=Path.cwd())

        # Settings. ``allowed`` gates calls, not the offer: every tool is still
        # offered, so agents with different gates send the same cached prefix.
        self.system: tuple[str, ...] = ()
        self.allowed: frozenset[str] | None = None
        self.allow_background = True
        self.persistent_retry = False
        self.recheck_quota = False
        self.wake_schedules = ToolWakeSchedules()

        # Hooks the Agent layer sets; see where each is asked.
        self.observers: list[Callable[[RuntimeEvent], None]] = []
        self.compact_due: Callable[[], bool] | None = None
        self.reply_room: Callable[[], int] | None = None
        self.prune: Callable[[], bool] | None = None
        self.result_note: Callable[[ToolMessage], str] | None = None
        self.before_tool_spawn: Callable[[Response], UserMessage | None] | None = None
        self.context_updates: (
            Callable[[], Sequence[SystemMessage | AgentMessage | SessionContextUpdate]]
            | None
        ) = None
        self.before_call: Callable[[bool], Awaitable[None]] | None = None
        self.after_failure: (
            Callable[[BaseException], Awaitable[Mapping[str, datetime]]] | None
        ) = None

        # What each request names its conversation and this agent's place in it.
        self.conversation = ""
        self.agent = ""
        self.parent = ""
        self.parent_turn = 0
        self.root_prompt = ""
        self.name = "/root"
        self.turns = 0
        self.turn_started_ms = 0
        self.compactions = 0
        self._steps = 0
        self._compaction_trigger: Literal["", "auto", "manual"] = ""

        # What the runtime recorded, and what the model sees of it.
        self.tape = Tape(session_id)

        # The inbox; its gate is the only wait for the user. ``_awaited`` is what
        # it waited for before this drain, until a message or retry takes it; each
        # drain sets it before anything reads it.
        self.inbox: GatedDeque[RuntimeEvent] = GatedDeque()
        self._awaited: Await | None
        # Each request taken in and not yet ended, and where it stands: held for a
        # state, taken into the turn, or neither yet. Kept in the order each came to
        # where it stands.
        self._requests: list[_Taken] = []
        # Whether a model call was told to start and neither ``AgentIdle`` nor a
        # stop was told since: the turn a halt or a clear stops. ``AgentIdle`` is
        # told only with the inbox empty, so state alone cannot say it.
        self.turn_under_way = False  # pragma: no mutate[truthiness]
        self.has_quit = False  # pragma: no mutate[truthiness]

        self.model_call: (
            asyncio.Task[ModelMessage | ModelResponseCompleted | None] | None
        ) = None
        # The model the call in flight asked: a failure of one switched away from
        # says nothing of the model now in use.
        self._called_model: Model = model
        self._streamed_chars = 0
        # What a model call or a compaction had queued when the runtime stopped it:
        # its ending was told as it was stopped, so each is dropped when its turn comes.
        self._abandoned: list[
            ModelMessage | ModelResponseCompleted | SessionCompactCompleted
        ] = []
        # When the last model call was sent, by ``time.monotonic``: the prompt cache
        # stays warm from then.
        self._last_request: float | None = None
        # Automatic compactions since the last reply; two in a row stop trying.
        self._auto_compactions = 0
        # Whether the last compaction failed. A reply leaves the context it could not
        # compact, so none is tried again on its own until the user writes, retries,
        # or clears, or one asked for succeeds; each reply retried it before
        # (2026-11-22).
        self._compaction_failed = False  # pragma: no mutate[truthiness]

        self.compact_task: asyncio.Task[SessionCompactCompleted | None] | None = None
        self._compacting: SessionCompact | None = None

        # Wall-clock seconds before which a rate-limited model call waits.
        self.service_suspended_until: float | None = None
        self._retry_timer: asyncio.TimerHandle | None = None
        self._retries = 0

        # Each call the runtime runs, until its task has ended and its result,
        # or what stands for it, is taken in.
        self._calls: dict[str, _Call] = {}

    @property
    def model(self) -> Model:
        """What answers; another assigned retries the turn a failure stopped."""
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        changed = model is not self._model
        # Each model has its own prompt cache: another starts with it cold.
        if _cache_key(model) != _cache_key(self._model):
            self._last_request = None
        self._model = model
        if changed:
            self.retry()

    @property
    def running_tools(self) -> Mapping[str, asyncio.Task[None]]:
        """The calls the model waits on, by id."""
        return {cid: c.task for cid, c in self._calls.items() if c.status == "waited"}

    # A finished task's result is in the inbox, or was never posted: a call
    # killed before it ran was answered by whatever killed it. A killed call
    # still stopping runs on.
    @property
    def detached(self) -> Mapping[str, asyncio.Task[None]]:
        """The calls answered by a stub that run on, by id."""
        return {
            cid: c.task
            for cid, c in self._calls.items()
            if c.status != "waited" and not c.task.done()
        }

    # A compaction is no turn: one asked for at rest stops none.
    @property
    def model_idle(self) -> bool:
        """True when the model has no turn to work on, or the user holds it."""
        return (
            self.model_call is None
            and not self.running_tools
            and (self.inbox.gate_armed or not self._should_call_model())
        )

    # A detached call still owing its result counts: its call is already taped,
    # so idling first would persist an unanswered call.
    @property
    def agent_idle(self) -> bool:
        """True when the model is idle, and no call owes it a result unless the user holds the turn."""
        return self.model_idle and (
            self.inbox.gate_armed
            or not (self._owes_a_result() or self._holding(ModelResponseStarted))
        )

    @property
    def stopped(self) -> bool:
        """True while a failed model call holds the turn: suspended, or failed."""
        return (
            self.inbox.awaiting == Await.failure()
            or self.service_suspended_until is not None
        )

    # Background work does not hold the model back: only the calls it waits on.
    @property
    def at_work(self) -> bool:
        """True while a reply streams, a compaction runs, or a waited-on call runs."""
        return bool(
            self.model_call is not None
            or self.compact_task is not None
            or self.running_tools,
        )

    # A call that runs until it is stopped has no result to wait for: waiting on it
    # would leave the agent never idle. It wakes the model when it has news.
    def _owes_a_result(self) -> bool:
        """Return whether a call run apart is still to deliver its result."""
        return any(not _runs_until_stopped(self._calls[cid]) for cid in self.detached)

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
          event: An input, such as a ``UserMessage``, ``ModelHalt``, or ``SessionCompact``.

        """
        self.inbox.push_back(event)

    def halt(self, *, sender: str | None = None) -> None:
        """Stop the model call; the calls it waited on are detached and keep running.

        Args:
          sender: Who asks; ``None`` for the session's user.

        """
        self.push(ModelHalt(sender=sender))

    # Only while stopped: a ``ModelRetry`` queued otherwise wakes a watcher for
    # nothing.
    def retry(self) -> None:
        """Call the model now for the turn a failure stopped, as after a sign-in."""
        if self.stopped:
            self.push(ModelRetry())

    @overload
    async def ask(self, request: SessionCompact) -> Outcome[SessionCompacted]: ...

    @overload
    async def ask(
        self,
        request: SessionPrune | SessionClear | UserMessage,
    ) -> Outcome[None]: ...

    async def ask(
        self,
        request: SessionCompact | SessionPrune | SessionClear | UserMessage,
    ) -> Outcome[SessionCompacted] | Outcome[None]:
        """Push ``request`` and wait for it to end.

        Args:
          request: What to ask of the runtime.

        Returns:
          outcome: How it ended; ``Cancelled``, at once, from a runtime that quit.

        """
        return await self.submit(request)

    # The request itself pairs it with its ending: another's of the same kind, equal
    # to it field for field, may end first.
    def submit(
        self,
        request: SessionCompact | SessionPrune | SessionClear | UserMessage,
    ) -> asyncio.Future[Outcome[SessionCompacted] | Outcome[None]]:
        """Push ``request`` now; return what resolves to how it ends.

        Args:
          request: What to ask of the runtime.

        Returns:
          ended: Resolved as the request ends, just before its ending is told to
            observers; already ``Cancelled`` from a runtime that quit.

        """
        ended = asyncio.Future[Outcome[SessionCompacted] | Outcome[None]]()
        if self.has_quit:
            ended.set_result(Cancelled())
            return ended
        self._requests.append(_Taken(request=request, ended=ended))
        self.push(request)
        return ended

    # What is asked of the runtime ends once, told as its ``...Completed``.
    def _open_if_asked(self, event: RuntimeEvent) -> None:
        """Note ``event`` as owed one ending if it is something asked of the runtime."""
        if isinstance(
            event,
            SessionCompact
            | SessionClear
            | SessionPrune
            | UserMessage
            | AgentMessage
            | SystemMessage,
        ):
            _ = self._opened(event)

    # Requests equal field for field are still two: each is found as itself.
    def _taken(self, request: _Request) -> _Taken | None:
        """Return where ``request`` stands, if it is taken in and not ended."""
        return next((e for e in self._requests if e.request is request), None)

    def _opened(self, request: _Request) -> _Taken:
        """Note ``request`` as taken in and owed one ending, unless it already is."""
        entry = self._taken(request)
        if entry is None:
            entry = _Taken(request=request)
            self._requests.append(entry)
        return entry

    def _stand(
        self,
        request: _Held,
        stage: Literal["held", "in_turn"],
        until: _Until | None = None,
    ) -> None:
        """Put ``request`` where ``stage`` says, after the others standing there."""
        entry = self._taken(request)
        if entry is None:
            entry = _Taken(request=request)
        # One pushed twice is one request held twice: acted on twice, ended once,
        # where the first push ends.
        elif entry.stage != "open":
            entry = _Taken(request=request, owed=False)  # pragma: no mutate[truthiness]
        else:
            self._requests.remove(entry)
        entry.stage, entry.until = stage, until
        self._requests.append(entry)

    # Every caller ends a request it took from ``_requests``, so one is always found.
    def _end(self, request: _Request, ending: _Ending) -> None:
        """End ``request`` as ``ending`` says: to whoever asked it, then to every observer."""
        self._ended(next(e for e in self._requests if e.request is request), ending)

    # A request pushed twice stands twice: the hold that was acted on ends, not the
    # first entry that names the same request.
    def _ended(self, entry: _Taken, ending: _Ending) -> None:
        """End the request ``entry`` holds, as :meth:`_end` does."""
        self._requests.remove(entry)
        if not entry.owed:
            return
        if entry.ended is not None and not entry.ended.done():
            entry.ended.set_result(ending.outcome)
        self.publish(ending)

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        """Return every event published from the first ``__anext__`` on."""
        return self._events()

    async def _events(self) -> AsyncIterator[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        observer = queue.put_nowait
        self.observers.append(observer)
        try:
            while True:
                yield await queue.get()
        finally:
            self.observers.remove(observer)

    @property
    def jobs(self) -> Mapping[str, Job]:
        """Tool calls not yet stopped, by call id: waited on, waiting to start, or apart."""
        return {
            cid: _RunningJob(
                call=c.call,
                status="running"
                if c.status == "waited"
                else "sleeping"
                if c.sleeping
                else "detached",
                runs_until_stopped=_runs_until_stopped(c),
                runtime=self,
            )
            for cid, c in self._calls.items()
            if not c.task.done()
        }

    async def foreground(self, call_id: str) -> ToolMessage | None:
        """Wait for a detached call's result, or its next wake, whichever comes first.

        Waiting no longer than a wake keeps the cache warm, and shows the model a
        long call as it goes; the call runs on.

        Args:
          call_id: The call, as ``jobs`` names it.

        Returns:
          result: Its result, taken instead of its delivery; else, at its next
            wake, what it wrote since the model last looked; ``None`` for no
            such call.

        """
        for held in self._requests:
            if (
                held.stage == "held"
                and isinstance(entry := held.request, ToolMessage)
                and entry.call_id == call_id
            ):
                # Taken here, it is answered here: no round will end it.
                self._ended(held, _message_ending(entry, Succeeded(value=None)))
                return entry
        for entry in self.tape.context().messages:
            if (
                isinstance(entry, ToolMessage)
                and entry.call_id == call_id
                and entry.state != "pending"
            ):
                return entry
        call = self._calls.get(call_id)
        if call is None or call_id not in self.detached or call.status == "answered":
            return None
        waiter = asyncio.Future[ToolMessage]()
        call.foreground_waiters.append(waiter)
        try:
            return await self._result_or_wake(call, waiter)
        finally:
            call.foreground_waiters.remove(waiter)

    async def _result_or_wake(
        self,
        call: _Call,
        waiter: asyncio.Future[ToolMessage],
    ) -> ToolMessage:
        """Return ``waiter``'s result, or at a wake what the call wrote since last looked."""
        watch = call.watch
        schedule = None if watch is None else self._schedule(watch)
        if watch is None or schedule is None:
            return await waiter
        while True:
            now = time.monotonic()
            next_wake = schedule.delay_sec_until_next_wake(
                now - watch.started,
                cache_left_sec=self._cache_bound(watch, now),
            )
            done, _ = await asyncio.wait((waiter,), timeout=next_wake.delay_sec)
            if done:
                return waiter.result()
            if next_wake.fires_even_without_new_output or watch.fresh:
                return ToolMessage(
                    call_id=call.call.call_id,
                    state="done",
                    stdout=(self._progress(call.call.call_id, watch),),
                    result=BackgroundTaskResult(
                        operation="foreground",
                        job_id=call.call.call_id,
                    ),
                )

    def detach[T](self, task: asyncio.Future[T] | None) -> None:
        """Answer the waited-on call ``task`` runs with a stub, and let it run on.

        As when a wake finds it running: its real result arrives later.
        Nothing happens when background runs are off, or no waited call runs there.

        Args:
          task: The task running the call's tool, as ``asyncio.current_task()``
            in it; ``None``, outside any task, runs none.

        """
        cid = self._call_run_by(task)
        if cid is not None and cid in self.running_tools and self.allow_background:
            self.push(ToolDetach(call_id=cid))

    def wake[T](self, task: asyncio.Future[T] | None) -> None:
        """Tell the model what the call ``task`` runs wrote since it last looked.

        A call the model waits on is detached, as :meth:`detach` does: its stub
        shows what it wrote. Of a call run apart the model is told once it is
        free: after its reply, or, while the runtime waits for the user, with
        their next message. Nothing happens when ``task`` runs no call here.

        Args:
          task: The task running the call's tool, as ``asyncio.current_task()``
            in it; ``None``, outside any task, runs none.

        """
        cid = self._call_run_by(task)
        if cid is None:
            return
        if cid in self.running_tools:
            self.detach(task)
        elif (watch := self._calls[cid].watch) is not None:
            self._owe(watch)

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
        cid = self._call_run_by(task)
        if cid is None:
            self.push(event)
        else:
            self._calls[cid].events_after_result.append(event)

    # A call still waiting its turn has no task yet, so ``None`` names none.
    def _call_run_by[T](self, task: asyncio.Future[T] | None) -> str | None:
        """Return the id of the call whose tool ``task`` runs, if it runs one here."""
        if task is None:
            return None
        return next(
            (cid for cid, call in self._calls.items() if call.result_future is task),
            None,
        )

    # A resumed context's detached calls ran in a process that has ended, so no
    # result will follow; and a clean quit's cancellations are never committed.
    def settle_detached(self) -> None:
        """Deliver a cancellation for each call answered only by a stub, running nowhere."""
        pending: list[str] = []
        answered: set[str] = set()
        for entry in self.tape.context().messages:
            if not isinstance(entry, ToolMessage):
                continue
            if entry.state == "pending":
                pending.append(entry.call_id)
            else:
                answered.add(entry.call_id)
        for cid in dict.fromkeys(pending):
            if cid in answered or cid in self.detached or cid in self.running_tools:
                continue
            self._append_late(
                ToolMessage(
                    call_id=cid,
                    timestamp=now_timestamp(),
                    state="cancelled",
                    shown=self.stubs.cancelled,
                ),
            )

    # A resumed window may end with something the last process never answered: a
    # message, a result, a job it left running. That is as stale as a halted turn's,
    # so it waits for the user's next message, as after a halt; any other wake-up,
    # such as closing a model the resume replaced, must not answer it.
    def hold_for_user(self) -> None:
        """Wait for the user's next message if the history ends with one to answer."""
        if self._should_call_model():
            self.inbox.push_front(gate=Await.user())

    # Calls are named by id, which no compaction or repair since moves. An output the
    # tape holds but the context no longer shows was taken by a compaction or a
    # clear since the prune was asked: it is gone already, as the prune wanted.
    def _on_prune(self, item: SessionPrune) -> None:
        """Replace the outputs of the calls ``item`` names with a stub, and tell how it went."""
        chosen: list[int] = []
        found: set[str | None] = set()
        for i, entry in enumerate(self.tape.context().messages):
            if isinstance(entry, ToolMessage) and entry.call_id in item.call_ids:
                chosen.append(i)
                found.add(entry.call_id)
        taped = {
            record.event.call_id
            for record in self.tape
            if isinstance(record, ReferrableTapeEvent)
            and isinstance(record.event, ToolMessage)
        }
        missing = [cid for cid in item.call_ids if cid not in found | taped]
        if missing:
            exc = ValueError(f"No output of call {missing[0]} is in the context.")
            self._end(item, SessionPruneCompleted(outcome=Failed(exception=exc)))
            return
        self.drop(chosen)
        self._end(item, SessionPruneCompleted(outcome=Succeeded(value=None)))

    def drop(self, indices: Sequence[int]) -> None:
        """Replace the context's tool outputs at ``indices`` with ``stubs.pruned``, in one splice.

        Args:
          indices: Positions in ``context().messages``, each a tool's result or a
            wake's news.

        Raises:
          TypeError: An index is no tool output.

        """
        if not indices:
            return
        context = stubbed(self.tape.context().messages, indices, stub=self.stubs.pruned)
        _ = self.tape.append_splice(
            mask=full_tape_mask(self.tape),
            replacement=tuple(context),
            strategy="prune",
            paired_externally=unpaired_call_ids(context),
            discards_content=True,
        )
        self.tool_context.files.forget()

    def _holding(self, until: _Until) -> bool:
        """Return whether anything taken in waits for ``until``."""
        return any(e.stage == "held" and e.until == until for e in self._requests)

    # Only ``_stand`` holds, and it holds no clear: a clear acts at once.
    def _take(self, *states: _Until) -> list[_Taken]:
        """Mark what waits for any of ``states`` as acted on, and return it, in the order it came."""
        taken = [e for e in self._requests if e.stage == "held" and e.until in states]
        for entry in taken:
            entry.stage, entry.until = "open", None
        return taken

    async def _release(self, *states: _Until) -> bool:
        """Act on what waits for any of ``states``, in order; return whether any did."""
        released = self._take(*states)
        for entry in released:
            match entry.request:
                case SessionPrune() as prune:
                    self._on_prune(prune)
                case SessionCompact() as compact:
                    await self._on_compact(compact)
                case _:
                    self._commit_due(entry)
        return bool(released)

    def pending_mid_stream(self) -> Sequence[UserMessage | AgentMessage]:
        """Return what the user sent while the model streams, in order.

        Returns:
          pending: The messages; they join the context once the reply ends, its
            calls detached.

        """
        return tuple(
            request
            for e in self._requests
            if e.stage == "held"
            and e.until == ModelResponseCompleted
            and isinstance(request := e.request, (UserMessage, AgentMessage))
        )

    async def run_forever(self) -> None:
        """Drain the inbox, dispatch each event, then advance; until ``SessionQuit``."""
        while True:
            if (
                self.agent_idle
                and not self.at_work
                and not self.inbox.gate_armed
                and self.inbox.empty()
            ):
                self.turn_under_way = False  # pragma: no mutate[truthiness]
                self.publish(AgentIdle())
            self._awaited = self.inbox.awaiting
            items = await self.inbox.drain()
            reply_failed = False  # pragma: no mutate[truthiness]
            for at, item in enumerate(items):
                later = items[at + 1 :]
                # One event that fails to dispatch is logged; the rest of its batch
                # still is.
                try:
                    if await self._dispatch(item, later=later):
                        break
                except Exception as exc:  # noqa: BLE001 -- One event's failure must not drop the events after it.
                    log_exception_or_warning(
                        logger,
                        f"dispatching {type(item).__name__} raised",
                        exc,
                    )
                    # A reply not taken is a failed model call: asked again at
                    # once, it would fail again, so the runtime waits.
                    if isinstance(item, ModelMessage):
                        self.publish(
                            ModelMessageCompleted(outcome=Failed(exception=exc)),
                        )
                        self.inbox.push_front(
                            ModelResponseCompleted(outcome=Failed(exception=exc)),
                            *later,
                        )
                        reply_failed = True
                        break
            if self.has_quit:
                return
            if reply_failed:
                continue
            # A failure here leaves the batch taken; the next pass goes on.
            try:
                await self._advance()
            except Exception as exc:  # noqa: BLE001 -- The loop must outlive one pass's failure.
                log_exception_or_warning(logger, "dispatch loop iteration raised", exc)

    async def _dispatch(
        self,
        item: RuntimeEvent,
        *,
        later: Sequence[RuntimeEvent],
    ) -> bool:
        """Act on ``item``; return whether the rest of its batch was put back, or the loop ends."""
        self._open_if_asked(item)
        match item:
            case SessionQuit():
                await self._on_quit(later)
                return True
            case ModelHalt():
                return await self._on_halt(item, later=later)
            case SessionClear():
                self._on_clear(item, later=later)
                return True
            # What a call or a compaction queued before it was stopped is its ending
            # no longer (``_cancel_model``, ``_cancel_model_and_compaction``).
            case (
                ModelMessage() | ModelResponseCompleted() | SessionCompactCompleted()
            ) if self._was_abandoned(item):
                if isinstance(item, ModelMessage):
                    self.publish(ModelMessageCompleted(outcome=Cancelled()))
            case ModelResponseCompleted(outcome=Failed(exception=raised)):
                return await self._on_model_response_error(
                    raised,
                    ended=item,
                    later=later,
                )
            case ModelMessage():
                return await self._on_model_message(item, later=later)
            case ModelResponseTextStreamed() | ModelResponseThinkingStreamed():
                self.publish(item)
            case ModelRetry():
                self._on_retry()
            case ToolHalt():
                self._on_kill(item)
            case ToolDetach():
                self._on_detach(item)
            case ToolMessage():
                self._take_result(item)
            case SessionContextUpdate():
                self._update_context(item)
            case SessionCompact():
                await self._on_compact(item)
            case SessionCompactCompleted():
                await self._on_compacted(item)
            case SessionPrune():
                self._stand(item, "held", ToolCohortCompleted)
            case UserMessage() | AgentMessage() | SystemMessage():
                self._on_message(item)
            case _:  # pragma: no mutate -- basedpyright asks for the default; without it nothing else matches either.
                pass
        return False

    # A message sent to wait is held for what it names, unless it ends a wait for the
    # user: that is the user resuming, and held for idle the halted turn would be
    # sent again without it.
    def _on_message(self, item: UserMessage | AgentMessage | SystemMessage) -> None:
        """Commit ``item``, hold it until what it waits for, or drop a moot nudge."""
        if item.due_after is not None and self._awaited is None:
            self._stand(item, "held", item.due_after)
            return
        self._awaited = None
        if item.due_after is not None:
            self._commit_due(self._opened(item))
            return
        # The user writing during a suspension asks to try now, as after a failed
        # compaction.
        if isinstance(item, UserMessage):
            self.recheck_quota = True
            self._compaction_failed = False
            self._end_suspension()
        if isinstance(item, SystemMessage):
            # A nudge to a busy agent is moot: it is told again at its next idle if
            # it still owes. Waiting on calls is busy too, and a note there would
            # come between a call and its result, which no provider takes.
            if self.model_call is None and not self.running_tools:
                self._note(item)
            else:
                self._end(item, SystemMessageCompleted(outcome=Cancelled()))
        elif self.model_call is not None:
            self._stand(item, "held", ModelResponseCompleted)
        else:
            self._stop_all_tools(mode="detach")
            self._commit_user(item)

    # Released from a failure's wait, or ending a suspension's: the gate calls the
    # model for the turn. Told before that call: with no turn stopped, none follows.
    def _on_retry(self) -> None:
        """Let the next model call go now, and tell whether a stopped turn goes on."""
        released = (
            self._awaited == Await.failure() or self.service_suspended_until is not None
        )
        self._awaited = None
        self.recheck_quota = True
        self._compaction_failed = False
        self._end_suspension()
        self.publish(
            ModelRetryCompleted(
                outcome=Succeeded(value=None) if released else Cancelled(),
            ),
        )

    def _end_suspension(self) -> None:
        """Let the next model call go now, its backoff started afresh."""
        self._retries = 0
        self.service_suspended_until = None
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    async def _on_quit(self, later: Sequence[RuntimeEvent]) -> None:
        """Cancel everything in flight; wait for the cancelled tools to stop."""
        # Whatever was told to have started is told to have ended, here cancelled.
        self._cancel_model_and_compaction()
        if self.running_tools:
            self.publish(ToolCohortCompleted(outcome=Cancelled()))
        # Background work too, or it runs on after shutdown.
        stopped = [c.task for c in self._calls.values() if not c.task.done()]
        for task in stopped:
            task.cancel()
        # Work no loop will reach: its callers stop waiting. Each request still open,
        # taken in or not, ends cancelled.
        self.has_quit = True
        for event in (*later, *self.inbox.drain_nowait()):
            self._open_if_asked(event)
        for entry in tuple(self._requests):
            self._ended(entry, self._cancelled(entry.request))
        if self._retry_timer is not None:
            self._retry_timer.cancel()
        self.publish(SessionQuitCompleted(outcome=Succeeded(value=None)))
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

    # The turn a halt stops may have its reply done, and only what its calls owe
    # still to come; an agent at rest, or waiting for the user, has none.
    async def _on_halt(self, item: ModelHalt, *, later: Sequence[RuntimeEvent]) -> bool:
        """Stop the turn: cancel the model call and any compaction, detach the waited calls; return whether ``later`` now waits for the user."""
        self.publish(item)
        under_way = self.turn_under_way
        halted = ModelHaltCompleted(
            outcome=Succeeded(value=None) if under_way or self.at_work else Cancelled(),
        )
        self._cancel_model_and_compaction()
        # Waited on, a call's result would be held behind the wait for the user, so
        # the agent would look busy until the user wrote. Detached, the calls run on
        # and their results arrive after the user's next message.
        self._stop_all_tools(mode="detach")
        self._end_suspension()
        # What the user typed during the stream is what the runtime would wait for:
        # the turn it was typed into stopped first.
        if self._holding(ModelResponseCompleted):
            self._end_turn(Cancelled())
            _ = await self._release(ModelResponseCompleted)
            self.publish(halted)
            return False
        self.inbox.push_front(*later, gate=Await.user())
        # After the gate, as a clear's end is told.
        if under_way:
            self._end_turn(Cancelled())
        # Taken in, with no call for it begun: no turn's end will tell of it.
        self._end_messages(Cancelled())
        self.publish(halted)
        return True

    def _on_clear(self, item: SessionClear, *, later: Sequence[RuntimeEvent]) -> None:
        """Stop everything, hide the tape, and wait for the user."""
        under_way = self.turn_under_way
        self._cancel_model_and_compaction()
        self._stop_all_tools(mode="detach")
        self._cancel_detached()
        # No result of the wiped session's calls is taken in.
        for call in tuple(self._calls.values()):
            call.status = "answered"
            self._forget(call)
        # What waited for the model, or for rest, is dropped: no turn takes it in.
        dropped = self._take(
            ModelResponseStarted,
            ModelResponseCompleted,
            ModelIdle,
            AgentIdle,
        )
        # A new conversation does not inherit the old one's failure, its wait for a
        # rate limit, or its failed compaction.
        self._end_suspension()
        self._compaction_failed = False
        self.tape.append_clear()
        # A tool that finished as the user cleared posted its result behind the
        # ``SessionClear``; delivering it would carry the wiped session's work into
        # the fresh one.
        self.inbox.push_front(
            *(
                i
                for i in (*later, *self.inbox.drain_nowait())
                if not isinstance(i, ToolMessage)
            ),
            gate=Await.user(),
        )
        # After the gate: input an observer pushes in reply must count as new, or
        # it never releases the gate.
        if under_way:
            self._end_turn(Cancelled())
        self._end_messages(Cancelled())
        for entry in dropped:
            self._ended(entry, self._cancelled(entry.request))
        self._end(item, SessionClearCompleted(outcome=Succeeded(value=None)))

    # A turn that stopped short is under way no more; one the model finished is,
    # until ``AgentIdle``: what its calls owe is still to come.
    def _end_turn(self, outcome: Outcome[None]) -> None:
        """Tell that the model's work on the turn stopped, and that each message it took in ended so."""
        if not isinstance(outcome, Succeeded):
            self.turn_under_way = False  # pragma: no mutate[truthiness]
        self.publish(ModelIdle(outcome=outcome))
        self._end_messages(outcome)

    # Whoever sent a message and waits takes its end from here, whatever became of
    # it: a turn's end tells only of what a turn took in.
    def _end_messages(self, outcome: Outcome[None]) -> None:
        """End each message taken in since the last turn ended as ``outcome``."""
        taken = [
            message
            for e in self._requests
            if e.stage == "in_turn"
            and isinstance(
                message := e.request,
                UserMessage | AgentMessage | SystemMessage | ToolMessage,
            )
        ]
        for message in taken:
            self._end(message, _message_ending(message, outcome))

    def _cancelled(self, request: _Request) -> _Ending:
        """Return the ending of ``request`` that says it was stopped first."""
        match request:
            case SessionCompact():
                return SessionCompactCompleted(
                    outcome=Cancelled(),
                    tape_len_before=len(self.tape),
                )
            case SessionClear():
                return SessionClearCompleted(outcome=Cancelled())
            case SessionPrune():
                return SessionPruneCompleted(outcome=Cancelled())
            case _:
                return _message_ending(request, Cancelled())

    async def _on_model_response_error(
        self,
        exc: BaseException,
        *,
        ended: ModelResponseCompleted | None,
        later: Sequence[RuntimeEvent],
    ) -> bool:
        """Tell the call's end, unless it was told (``ended`` is ``None``); then call again, compact, suspend, or stop the turn; return whether ``later`` was put back."""
        resets = (
            await self._after_failure(exc)
            if self._called_model is self.model
            else dict[str, datetime]()
        )
        # Told whatever the turn does next: a call's end is never the turn's.
        self.model_call = None
        if ended is not None:
            self.publish(ended)
        # A failure of a model switched away from says nothing of the one in use:
        # the gate asks it, at once.
        if self._called_model is not self.model:
            _ = await self._release(ModelResponseCompleted)
            return False
        # A request too long for the window is compaction's job: compact, and let
        # the gate call again, under the same breaker. Started here, not queued:
        # the gate would call the model first, for the compaction to cancel.
        if (
            isinstance(exc, ContextOverflowError)
            and self.compactor is not None
            and self._auto_compactions < 2
            and not self._compaction_failed
        ):
            logger.warning("model context overflowed; compacting: %s", exc)
            self._auto_compactions += 1
            await self._on_compact(SessionCompact(trigger="auto"))
            return False
        # A long rate limit or outage, for an agent that waits them out: hold the
        # model call until the advised time, backing off to 5 minutes, then call
        # again; 60 in a row end the turn.
        if (
            isinstance(exc, RetryableError)
            and self.persistent_retry
            and self._retries < 60
        ):
            delay = (
                max(min(resets.values()).timestamp() - time.time(), 0.0)
                if resets
                else max(exc.retry_after_sec, min(5.0 * (1 << self._retries), 300.0))
            )
            self._retries += 1
            self.service_suspended_until = time.time() + delay
            self.publish(
                ModelResponseSuspended(
                    exception=exc,
                    retry_at=datetime.fromtimestamp(self.service_suspended_until, UTC),
                    failures_in_a_row=self._retries,
                    provider=self.model.provider.name if resets else "",
                    reset_at_by_login=resets,
                ),
            )
            # Input typed during the failed stream is part of what the retry asks.
            _ = await self._release(ModelResponseCompleted)
            # Any input wakes the loop; ``SessionSaveDue`` changes nothing else.
            self._retry_timer = asyncio.get_running_loop().call_later(
                delay,
                self.inbox.push_back,
                SessionSaveDue(),
            )
            return False
        self._end_suspension()
        # The error is shown, never put in the context. What the user typed as the
        # call ran is answered next: the turn it was typed into stopped first.
        if self._holding(ModelResponseCompleted):
            self._end_turn(Failed(exception=exc))
            _ = await self._release(ModelResponseCompleted)
            return False
        self.inbox.push_front(*later, gate=Await.failure())
        # After the gate, as a clear's end is told.
        self._end_turn(Failed(exception=exc))
        return True

    def _on_kill(self, item: ToolHalt) -> None:
        """Cancel the call ``item`` names, or every call, and tell whether any was."""
        cid = item.call_id
        running = {*self.running_tools, *self.detached}
        stopped = bool(running) if cid is None else cid in running
        if cid is None:
            self._stop_all_tools(mode="kill")
            self._cancel_detached()
        elif cid in self.running_tools:
            _ = self._stop_tool(cid, mode="kill")
            # A wait a kill ends ends cancelled, as ``_stop_all_tools`` ends it.
            self._cohort_left(Cancelled())
        elif cid in self.detached:
            self._kill_detached(self._calls[cid])
        self.publish(
            ToolHaltCompleted(
                outcome=Succeeded(value=None) if stopped else Cancelled(),
            ),
        )

    # The group goes as one, whichever of its calls is named: one finished already
    # still names its group.
    def _on_detach(self, item: ToolDetach) -> None:
        """Answer each waited-on call of the named call's group with a stub, and let it run on."""
        group = next(
            (
                c.group
                for c in self._calls.values()
                if any(m.call_id == item.call_id for m in c.group)
            ),
            (),
        )
        stubs = {
            m.call_id: self._stop_tool(m.call_id, mode="detach")
            for m in group
            if m.call_id in self.running_tools
        }
        if stubs:
            self._cohort_left(Cancelled())
        named = stubs.get(item.call_id)
        self.publish(
            ToolDetachCompleted(
                outcome=Cancelled() if named is None else Succeeded(value=named),
            ),
        )

    # A result is taken once. A waited-on call's is read whole; a detached call's
    # goes forward, to a caller of ``foreground`` or as a delivery. What came for a
    # call answered already (cancelled, or cleared) is dropped.
    # A pending record that ends a call is a relayed call's stub, which runs on
    # elsewhere; a call's news while it runs is asked for by ``wake``.
    def _take_result(self, result: ToolMessage) -> None:
        """Take in a call's result: into the turn, or forward."""
        call = self._calls.get(result.call_id)
        if call is None or call.status == "answered" or result.state == "pending":
            return
        waited_on = call.status == "waited"
        call.status = "answered"
        if waited_on:
            # The model waits on it: it is due now, whatever its tool said.
            result = dataclasses.replace(result, due_after=None)
            self.tape.append_history(result)
            self._stand(result, "in_turn")
        else:
            self._deliver(call, result)
        self.publish(result)
        self.publish(
            ToolCallCompleted(
                outcome=Cancelled()
                if result.state == "cancelled"
                else Succeeded(value=result),
            ),
        )
        if waited_on:
            self._cohort_left(Succeeded(value=None))
        self._forget(call)

    # A caller of :meth:`foreground` takes the result as its own answer, so delivering
    # it forward as well would tell the model twice.
    def _deliver(self, call: _Call, result: ToolMessage) -> None:
        """Hand a detached call's result to whoever waits for it, else hold it for the next round."""
        result = self._unseen(call, result)
        waiting = [w for w in call.foreground_waiters if not w.done()]
        for waiter in waiting:
            waiter.set_result(result)
        # A cancellation the context already answers, as a ``ToolHalt`` of a
        # waited-on call does, is not told again.
        cid = call.call.call_id
        told = result.state == "cancelled" and any(
            isinstance(e, ToolMessage) and e.call_id == cid and e.state != "pending"
            for e in self.tape.context().messages
        )
        if not waiting and not told:
            self._stand(result, "held", result.due_after or ModelResponseStarted)

    # Its stub answered the call already: the request sends this as the user's side
    # (``as_sent``), so it may follow the model's turn directly.
    def _append_late(self, message: ToolMessage) -> None:
        """Append a call's later message, its news or its result, and tell it."""
        self.tape.append_history(message)
        self._stand(message, "in_turn")

    # A timer runs it between the loop's passes, so it reads the runtime and asks;
    # only the loop changes it. While the runtime waits for the user it asks
    # nothing, nor for a note while the model works, a compaction runs, or calls
    # are waited on: what the call wrote waits for the next wake, which shows it
    # as well. A call whose tool wakes no one, or that can never run apart, was
    # never armed (``_run_tool_and_post``); it is detached only with its group.
    def _wake(self, cid: str, watch: _Watch, forced: bool) -> None:
        """Ask to detach the armed calls waited on, or to tell the model what ``cid`` wrote; arm its next wake."""
        # A model call since this wake was armed warmed the cache again.
        forced = forced and watch.cached in (None, self._last_request)
        if (forced or watch.fresh) and not self.inbox.gate_armed:
            if cid in self.running_tools:
                for waited, c in self._calls.items():
                    if (
                        c.status == "waited"
                        and c.watch is not None
                        and c.watch.timer is not None
                    ):
                        self.push(ToolDetach(call_id=waited))
            elif not self.at_work:
                self._owe(watch)
        self._arm(cid, watch)

    # What it wrote is read when the model is told, not here: a model call that
    # starts first loses none of it.
    def _owe(self, watch: _Watch) -> None:
        """Have the loop tell the model what ``watch``'s call wrote, once the model is free."""
        watch.waking = True
        # Any input makes the loop pass; ``SessionSaveDue`` changes nothing else.
        self.push(SessionSaveDue())

    def _arm(self, cid: str, watch: _Watch) -> None:
        """Set the next wake of the call ``cid``, if its schedule has one."""
        schedule = self._schedule(watch)
        if schedule is None:
            return
        now = time.monotonic()
        cache_left = self._cache_bound(watch, now)
        next_wake = schedule.delay_sec_until_next_wake(
            now - watch.started,
            cache_left_sec=cache_left,
        )
        forced = next_wake.fires_even_without_new_output
        watch.cached = (
            self._last_request if forced and cache_left < schedule.max_gap_sec else None
        )
        watch.timer = asyncio.get_running_loop().call_later(
            next_wake.delay_sec,
            self._wake,
            cid,
            watch,
            forced,
        )

    def _schedule(self, watch: _Watch) -> WakeSchedule | None:
        """Return when ``watch``'s call wakes the model: as the call said, else as its tool's schedule says."""
        if watch.every_sec is None:
            return self.wake_schedules.per_tool.get(
                watch.name,
                self.wake_schedules.default,
            )
        # The longest gap is the first, so every gap is the period, and every wake
        # is due whatever the call wrote.
        return WakeSchedule(first_gap_sec=watch.every_sec, max_gap_sec=watch.every_sec)

    # A call that said how often it is looked at is looked at then, whatever the cache.
    def _cache_bound(self, watch: _Watch, now: float) -> float:
        """Return how long a wake of ``watch``'s call may wait for the prompt cache's sake."""
        return math.inf if watch.every_sec is not None else self._cache_left(now)

    # Past its lifetime the cache is cold already, and no wake can keep it.
    def _cache_left(self, now: float) -> float:
        """Return how long until a wake must come to keep the prompt cache warm."""
        ttl = self.model.settings.cache_ttl_sec
        if not ttl or self._last_request is None:
            return math.inf
        left = self._last_request + ttl - _CACHE_MARGIN_SEC - now
        return left if left > 0 else math.inf

    def _look(self, watch: _Watch) -> str:
        """Return what the call wrote that the model has not read, its ends if long."""
        unseen = watch.items[watch.seen :]
        watch.seen = len(watch.items)
        text = "".join(item for item in unseen if isinstance(item, str))
        view, _ = self.tool_context.view(text.splitlines(keepends=True))
        return view.rstrip("\n")

    def _progress(self, cid: str, watch: _Watch) -> str:
        """Return a long call's news: how long it has run, and what it wrote since."""
        elapsed = time.monotonic() - watch.started
        news = self._look(watch) or "(no new output)"
        return f"[{watch.name} {cid} still running after {elapsed:.0f}s]\n{news}"

    # The model read the rest at its wakes, and the record keeps all of it. A view
    # the tool gave, such as a child's reply, is the tool's to say, and stays.
    def _unseen(self, call: _Call, result: ToolMessage) -> ToolMessage:
        """Return a detached call's result reading what the model has not seen of it."""
        watch = call.watch
        if (
            watch is None
            or not watch.seen
            or result.state != "done"
            or result.shown is not None
        ):
            return result
        tail = self._look(watch) or "(no new output)"
        shown = dataclasses.replace(result, shown=None, stdout=(tail,), stderr=())
        return dataclasses.replace(result, shown=shown.text())

    # While one is under way, started and its ending not yet told, every reader that
    # shows it takes the next ending for its: one asked for meanwhile is held, and
    # ends after it has run. The runtime's own is its duplicate: the one under way
    # does its work, so it ends at once, and no reader is told.
    async def _on_compact(self, item: SessionCompact) -> None:
        """Start a compaction, or hold it for after the one under way; what the user typed goes into it."""
        self._opened(item)
        if self._compacting is not None:
            if item.trigger == "auto":
                entry = self._opened(item)
                self._requests.remove(entry)
                if entry.ended is not None and not entry.ended.done():
                    entry.ended.set_result(Cancelled())
            else:
                self._stand(item, "held", SessionCompactCompleted)
            return
        self._compacting = item
        self._compaction_trigger = item.trigger
        self._cancel_model()
        self._stop_all_tools(mode="detach")
        # A compaction calls the model too: not on a spent login.
        await self._before_call(turn_start=False)
        # What the user typed goes into what is compacted; only ``SessionClear`` discards it.
        _ = await self._release(ModelIdle, ModelResponseCompleted)
        self.compact_task = asyncio.create_task(self._compact_and_post(item))
        self.compact_task.add_done_callback(
            log_task_exception(logger, "compaction task crashed"),
        )
        started = SessionCompactStarted()
        self.tape.append_history(started)
        self.publish(started)

    def _was_abandoned(
        self,
        item: ModelMessage | ModelResponseCompleted | SessionCompactCompleted,
    ) -> bool:
        """Return whether ``item`` is what a stopped call or compaction queued; forget it."""
        for at, abandoned in enumerate(self._abandoned):
            if abandoned is item:
                del self._abandoned[at]
                return True
        return False

    async def _on_compacted(self, item: SessionCompactCompleted) -> None:
        """Take in the ending the compaction under way queued; start the next held."""
        request = self._compacting
        if request is None:
            return
        self._end_compaction(request, item)
        # The first held starts; the rest are held again behind it.
        _ = await self._release(SessionCompactCompleted)

    # One the runtime stopped is not taped: the tape is as its ending says it stood.
    def _end_compaction(
        self,
        request: SessionCompact,
        ending: SessionCompactCompleted,
    ) -> None:
        """Take in how the compaction under way, ``request``, ended, and tell it."""
        self.compact_task = None
        self._compacting = None
        if isinstance(ending.outcome, Succeeded):
            self.compactions += 1
        else:
            self._compaction_trigger = ""
        if not isinstance(ending.outcome, Cancelled):
            self._compaction_failed = isinstance(ending.outcome, Failed)
        if not isinstance(ending.outcome, Cancelled):
            self.tape.append_history(ending)
        self._end(request, ending)

    # The call ended as its reply was taken in. A failure to start what the reply
    # asked for is the turn's, not a second ending of the call.
    async def _on_model_message(
        self,
        item: ModelMessage,
        *,
        later: Sequence[RuntimeEvent],
    ) -> bool:
        """Record the reply and start its calls; return whether ``later`` was put back, as a failure to start them may."""
        reply, refusals = self._take_reply(item)
        try:
            await self._start_what_it_asked(reply, refusals)
        except Exception as exc:  # noqa: BLE001 -- The turn stops on it, as on a failed call.
            log_exception_or_warning(logger, "starting a reply's calls raised", exc)
            return await self._on_model_response_error(exc, ended=None, later=later)
        return False

    def _take_reply(self, item: ModelMessage) -> tuple[Response, dict[str, str]]:
        """Record the reply and tell that its call ended; return it as recorded, and the calls it refuses."""
        self._auto_compactions = 0
        self._retries = 0
        self.model_call = None
        rewritten, refusals = self._rewrite_calls(item.response)
        msg = _stamped(rewritten)
        self._gate_calls(msg, refusals)
        cut_in = (
            self.before_tool_spawn(msg) if self.before_tool_spawn is not None else None
        )
        self.tape.append_history(msg)
        # Observers see the turn as recorded, rewrites included.
        self.publish(ModelResponseCompleted(outcome=Succeeded(value=msg)))
        if cut_in is not None:
            self._stand(cut_in, "held", ModelResponseCompleted)
        return msg, refusals

    # What the user sent as the reply streamed is answered next; the reply's calls
    # run on, detached, and no wait on them begins.
    async def _start_what_it_asked(
        self,
        msg: Response,
        refusals: dict[str, str],
    ) -> None:
        """Start the reply's calls; with none, the model finished its turn."""
        if calls(msg.records):
            cut = self._holding(ModelResponseCompleted)
            self._start_calls(calls(msg.records), refusals=refusals, detached=cut)
            if cut:
                _ = await self._release(ModelResponseCompleted)
        elif not await self._release(ModelResponseCompleted):
            self._end_turn(Succeeded(value=None))

    async def _advance(self) -> None:
        """Act on what the batch left due: prunes, held messages, deliveries, wakes, then a model call."""
        if self.at_work:
            self.publish(SessionSaveDue())
            return
        _ = await self._release(ToolCohortCompleted)
        # A held message waits until the model is done, not merely between
        # rounds: a tool result still to answer means it is not. Those due at
        # ``ModelIdle`` go first; those due at ``AgentIdle`` wait for every call
        # that still owes its result, and for the round a finished call's result
        # is about to start.
        if (
            not self._should_call_model()
            and not await self._release(ModelIdle)
            and not (self._owes_a_result() or self._holding(ModelResponseStarted))
        ):
            _ = await self._release(AgentIdle)
        if not self.inbox.gate_armed:
            self._tell_wakes()
            # A delivery drives its own round; it is committed at the model gate,
            # so what it commits is what the model is sent.
            _ = await self._release(ModelResponseStarted)
            if self._should_call_model():
                await self._start_round()
        self.publish(SessionSaveDue())

    # What a call is owed a telling of rides the round about to start, or starts one.
    def _tell_wakes(self) -> None:
        """Note what each call a wake asked about wrote since the model last looked."""
        for cid, call in self._calls.items():
            watch = call.watch
            if watch is not None and watch.waking and call.status != "answered":
                watch.waking = False  # pragma: no mutate[truthiness]
                news = ToolMessage(
                    call_id=cid,
                    timestamp=now_timestamp(),
                    state="pending",
                    stdout=(self._progress(cid, watch),),
                )
                self._append_late(news)
                self.publish(news)

    # Prune first: a drop that brings the context back under the compaction trigger
    # spares the compaction, and a note of what may be pruned puts it off a round,
    # so the model can answer it. Compact when the compactor says the request would
    # not fit; twice with no model call between means compacting does not help, so
    # the call goes ahead and the provider's error says why.
    async def _start_round(self) -> None:
        """Prune, compact, or call the model, whichever the context is due."""
        pruned = self.prune is not None and self.prune()
        if (
            not pruned
            and self.compactor is not None
            and self.compact_due is not None
            and self._auto_compactions < 2
            and not self._compaction_failed
            and self.compact_due()
        ):
            self._auto_compactions += 1
            self.inbox.push_back(SessionCompact(trigger="auto"))
        elif (
            self.service_suspended_until is None
            or time.time() >= self.service_suspended_until
        ):
            await self._call_model()

    async def _call_model(self) -> None:
        """Start a model call; a context past repair waits for the user instead."""
        if self.context_updates is not None:
            for note in self.context_updates():
                match note:
                    case SessionContextUpdate():
                        self._update_context(note)
                    case AgentMessage():
                        self.tape.append_history(note)
                        self.publish(note)
                    case SystemMessage():
                        self.tape.append_history(note)
        exc = self.tape.repair()
        if exc is not None:
            # Raised, it would repeat every pass on the same tape; the user is told,
            # and the runtime waits for them. No call was made, so none ends: the
            # turn stops.
            log_exception_or_warning(
                logger,
                "model-call gate context unrepairable",
                exc,
            )
            self.inbox.push_front(gate=Await.recovery())
            self._end_turn(Failed(exception=exc))
            return
        # Whatever released a failure's wait, or ended a suspension's, the turn goes
        # on, on the model asked now.
        self.service_suspended_until = None
        await self._before_call(turn_start=self._steps == 0)
        self._called_model = self.model
        self._streamed_chars = 0
        self.model_call = asyncio.create_task(self._stream_and_post())
        self.model_call.add_done_callback(
            log_task_exception(logger, "model-call task crashed"),
        )
        self.turn_under_way = True
        self.publish(ModelResponseStarted())

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

    async def _after_failure(self, exc: BaseException) -> Mapping[str, datetime]:
        """Ask ``after_failure`` about ``exc``; a failure of its own is logged, as no resets."""
        none: dict[str, datetime] = {}
        if self.after_failure is None:
            return none
        try:
            return await self.after_failure(exc)
        except Exception as raised:  # noqa: BLE001 -- An advisory hook must not hang the turn.
            log_exception_or_warning(logger, "after_failure raised", raised)
            return none

    # A note rides a round that real content drives and drives none itself, so the entry
    # that decides is the last turn before any trailing notes; a nudge alone drives one.
    def _should_call_model(self) -> bool:
        """Return True when history ends with content the model should answer."""
        messages = self.tape.context().messages
        at = _last_turn(messages)
        trailing = messages[0 if at is None else at + 1 :]
        if any(isinstance(m, SystemMessage) and m.subtype == "nudge" for m in trailing):
            return True
        return at is not None and isinstance(
            messages[at],
            (UserMessage, AgentMessage, ToolMessage),
        )

    def _note(self, note: SystemMessage) -> None:
        """Append the harness's message for the model to history, and publish it."""
        self.tape.append_history(note)
        self._stand(note, "in_turn")
        self.publish(note)

    def _update_context(self, update: SessionContextUpdate) -> None:
        """Put ``update``'s context in force, and tell it."""
        self.tape.append_history(update.context)
        self.publish(
            SessionContextUpdateCompleted(outcome=Succeeded(value=update.context)),
        )

    # Each message stays its own record, as it was sent and shown; the request merges
    # adjacent ones, as providers need.
    def _commit_user(self, message: UserMessage | AgentMessage) -> None:
        """Append ``message`` to history and publish it."""
        self.turns += 1
        self._steps = 0
        self.turn_started_ms = int(time.time() * 1_000)
        # A call's later message that came first, held for the next round, is told
        # first: the model reads them in the order they happened. Only a call's later
        # message waits for the next round (``_deliver``).
        for earlier in self._take(ModelResponseStarted):
            self._commit_due(earlier)
        self.tape.append_history(message)
        self._stand(message, "in_turn")
        self.publish(message)

    # The record says what was said, not how long it was held. It ends as the
    # message sent does: whoever sent that one holds it, not the record.
    def _commit_due(self, entry: _Taken) -> None:
        """Commit the message ``entry`` held as one sent with no wait."""
        match entry.request:
            case ToolMessage() | UserMessage() | AgentMessage() | SystemMessage():
                taken = dataclasses.replace(entry.request, due_after=None)
            case _:
                raise TypeError(f"{type(entry.request).__name__} is no message")
        entry.request = taken
        match taken:
            case ToolMessage():
                self._append_late(taken)
            case SystemMessage():
                self._note(taken)
            case UserMessage() | AgentMessage():
                self._commit_user(taken)

    # A detached call's stub holds what it wrote so far. The call leaves the cohort
    # before its stub is published: an observer reading the runtime as it hears the
    # stub sees the call apart, not still waited on.
    def _stop_tool(self, cid: str, *, mode: Literal["detach", "kill"]) -> ToolMessage:
        """Answer a waited-on call ``[detached]`` and let it run, or ``[cancelled]`` and cancel it; return the stub."""
        call = self._calls[cid]
        call.status = "apart"
        match mode:
            case "kill":
                state, shown = "cancelled", self.stubs.cancelled
            case "detach":
                wrote = "" if call.watch is None else self._look(call.watch)
                state = "pending"
                shown = (
                    f"{self.stubs.detached}\n{wrote}" if wrote else self.stubs.detached
                )
        stub = ToolMessage(
            call_id=cid,
            timestamp=now_timestamp(),
            state=state,
            shown=shown,
        )
        self.tape.append_history(stub)
        self.publish(stub)
        if mode == "kill":
            self._stop(call)
        return stub

    def _stop_all_tools(self, *, mode: Literal["detach", "kill"]) -> None:
        """Run :meth:`_stop_tool` for every waited-on call; the wait on them ends cancelled."""
        stopped = list(self.running_tools)
        for cid in stopped:
            _ = self._stop_tool(cid, mode=mode)
        if stopped:
            self._cohort_left(Cancelled())

    def _cohort_left(self, outcome: Outcome[None]) -> None:
        """Tell that the wait on the reply's calls ended, once none is waited on."""
        if not self.running_tools:
            self.publish(ToolCohortCompleted(outcome=outcome))

    # Told here, as the call is cancelled, so its ending comes before whatever the
    # cancelling leads to: its task takes the cancel only once the loop next turns,
    # and a call started meanwhile would be told first. One already done returned
    # what it queued, a reply or a failure, which is taken later or behind a gate:
    # the call ends here instead, and that is dropped when its turn comes.
    def _cancel_model(self) -> None:
        """Cancel any model call; it ends cancelled, once, whatever it had queued."""
        call = self.model_call
        if call is None:
            return
        queued = _queued_by(call)
        if call.cancel():
            self.publish(
                ModelResponseCompleted(
                    outcome=Cancelled(),
                    chars_streamed_before_cancel=self._streamed_chars,
                ),
            )
        elif queued is not None:
            self._abandoned.append(queued)
            self.publish(ModelResponseCompleted(outcome=Cancelled()))
        self.model_call = None

    # One whose work is done ends as it did: its ending, queued and not yet taken, is
    # taken here, and dropped when its turn in the inbox comes.
    def _cancel_model_and_compaction(self) -> None:
        """Cancel any model call and compaction; their late results are then dropped."""
        self._cancel_model()
        queued = _queued_by(self.compact_task)
        # A cancel does nothing to one that finished.
        if self.compact_task is not None:
            _ = self.compact_task.cancel()
        request = self._compacting
        cancelled = SessionCompactCompleted(
            outcome=Cancelled(),
            tape_len_before=len(self.tape),
        )
        if queued is not None:
            self._abandoned.append(queued)
        if request is not None:
            self._end_compaction(request, queued if queued is not None else cancelled)
        # Those held for it are stopped with it: none of them started. Only a
        # compaction waits for another.
        for held in self._take(SessionCompactCompleted):
            self._ended(held, cancelled)

    # The model's history should show the tool it should have used: a Bash call that
    # is exactly other tools' jobs becomes their calls before it is recorded, so
    # later turns copy the right tools. The first keeps the call's id; the rest get
    # ``<id>_<n>``, unused in the turn. A command that is other tools' work but not
    # as such calls stays a Bash call, answered with the refusal instead of run.
    #
    # A leading ``cd`` to where the shell starts anyway is dropped, so the model
    # stops copying it; only until a call that may move the shell, since the
    # calls after it may start somewhere else.
    #
    # A provider may number its calls anew in each reply. The call's id names it
    # here, and names each later message of a call run apart, so a call under an id
    # an earlier call has gets ``<id>_<n>``.
    def _rewrite_calls(self, msg: Response) -> tuple[Response, dict[str, str]]:
        """Return ``msg`` with repeated ids replaced and other tools' Bash calls rewritten, and the refusals by call id."""
        records: list[AssistantMessage | Thinking | ToolCall] = []
        refusals: dict[str, str] = {}
        in_context = {
            tc.call_id
            for entry in self.tape.context().messages
            if isinstance(entry, Response)
            for tc in calls(entry.records)
        }
        taken = {*self._calls, *in_context}
        used = taken | {r.call_id for r in calls(msg.records)}
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
            if not isinstance(given, ToolCall):
                records.append(given)
                continue
            record = given
            if given.call_id in taken:
                # The first id ``_fresh_ids`` yields is the taken one itself.
                record = dataclasses.replace(
                    given,
                    call_id=next(
                        itertools.islice(
                            _fresh_ids(given.call_id, used=used),
                            1,
                            None,
                        ),
                    ),
                )
                logger.debug(
                    "runtime repeated call id replaced: %s -> %s",  # pragma: no mutate[diagnostic-text]
                    given.call_id,
                    record.call_id,
                )
            bash = self.tools.get(record.name)
            if not isinstance(bash, Bash):
                records.append(record)
                continue
            command = record.arguments.get("command")
            # A call run later may start after the next reply moved the shell.
            later = record.arguments.get("background") is True or bool(
                record.arguments.get("delay"),
            )
            if isinstance(command, str) and settled and not later:
                record = dataclasses.replace(
                    record,
                    arguments={
                        **record.arguments,
                        "command": drop_cd(command, cwd=cwd),
                    },
                )
            rewritten = bash.rewrite(record.arguments)
            if isinstance(rewritten, (Rewrite, tuple)):
                # Each rewritten call keeps any request to run in the background.
                timing = controls(record.arguments)
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
                refusals[record.call_id] = rewritten
            records.append(record)
        return dataclasses.replace(msg, records=tuple(records)), refusals

    # Every tool is offered, so the gate holds here, and again in the context. It is
    # the gate the model was told when it called: observers of the reply may close it
    # for the next round, as the round cap does, and must not refuse this one's calls.
    def _gate_calls(self, msg: Response, refusals: dict[str, str]) -> None:
        """Refuse each call in ``msg`` of a tool ``allowed`` withholds now."""
        if self.allowed is None:
            return
        for tc in calls(msg.records):
            if tc.name in self.tools and tc.name not in self.allowed:
                refusals.setdefault(
                    tc.call_id,
                    f"{tc.name} is not available to this agent.",
                )

    # The line the module docstring describes. Each call is its own task, waiting
    # on the tasks of the earlier calls it contends with, so a ``ToolHalt`` stops one
    # call and the next starts once it has stopped. ``detached`` starts the calls
    # answered by stubs: no wait begins, so ``ToolCohortStarted`` is not told.
    def _start_calls(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        refusals: dict[str, str],
        detached: bool,
    ) -> None:
        """Run a reply's calls, each after the earlier ones it contends with."""
        now = time.time()
        started: list[_Call] = []
        groups: list[list[_Call]] = []
        by_key: dict[str, list[_Call]] = {}
        for tc in tool_calls:
            cid = tc.call_id
            delay = self._background_delay(tc, refusals)
            key = self._serialize_key(tc, refusals)
            # A call answered without running waits for nothing.
            after = (
                []
                if cid in refusals or tc.name not in self.tools
                else [
                    *(
                        c.task
                        for c in started
                        if not c.background and _contend(key, c.key)
                    ),
                    *(
                        c.task
                        for c in self._calls.values()
                        if c.stopping and _contend(key, c.key)
                    ),
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
            call = _Call(
                call=tc,
                key=key,
                task=task,
                background=delay is not None,
                status="waited" if delay is None and not detached else "apart",
                refusal=refusals.get(cid),
            )
            task.add_done_callback(lambda _, call=call: self._forget(call))
            self._calls[cid] = call
            if call.background:
                self._answer(
                    call,
                    f"{self.stubs.background_prefix}{tc.name} as job {cid}]",
                )
                continue
            started.append(call)
            # A ``"*"`` call ends every group before it, and is one of its own: the
            # next ends it in turn.
            if key == "*":
                groups.extend(by_key.values())
                by_key = {}
            if key is None:
                groups.append([call])
            else:
                by_key.setdefault(key, []).append(call)
        groups.extend(by_key.values())
        for members in groups:
            group = tuple(c.call for c in members)
            for c in members:
                c.group = group
        if detached:
            for call in started:
                self._answer(call, self.stubs.detached)
        # Told once the calls are registered: an observer reading the runtime as it
        # hears the start, as the status line does, sees them running.
        elif started:
            self.publish(ToolCohortStarted())

    def _answer(self, call: _Call, stub: str) -> None:
        """Answer ``call``, which runs apart, with ``stub``."""
        result = ToolMessage(
            call_id=call.call.call_id,
            timestamp=now_timestamp(),
            state="pending",
            shown=stub,
        )
        self.tape.append_history(result)
        self.publish(result)

    def _forget(self, call: _Call) -> None:
        """Drop ``call`` once its task has ended and its result is taken in."""
        cid = call.call.call_id
        if (
            call.status == "answered"
            and call.task.done()
            and self._calls.get(cid) is call
        ):
            del self._calls[cid]

    def _background_delay(
        self,
        call: ToolCall,
        refusals: dict[str, str],
    ) -> float | None:
        """Return the delay ``call`` asks to run in the background after, or ``None``."""
        tool = self.tools.get(call.name)
        if tool is None or not self.allow_background or not backgroundable(tool):
            return None
        try:
            _ = wake_period(call.arguments)
            return background_delay(call.arguments)
        except (TypeError, ValueError) as exc:
            refusals[call.call_id] = str(exc)
            return None

    def _serialize_key(self, call: ToolCall, refusals: dict[str, str]) -> str | None:
        """Return the serialization key, refusing calls whose scheduling fails."""
        if call.call_id in refusals:
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
            refusals[call.call_id] = describe(exc)
            return None

    def _stop(self, call: _Call) -> None:
        """Cancel ``call``'s task; until it stops, a later contending call waits for it."""
        # A task already done, its result still queued, has nothing to stop.
        _ = call.task.cancel()

    def _cancel_detached(self) -> None:
        """Cancel every detached call, delivering each one's cancellation."""
        for cid in tuple(self.detached):
            self._kill_detached(self._calls[cid])

    # The call was answered with a stub saying it still runs, so the model is told it
    # was killed. A finished task already delivered its own result, or will.
    def _kill_detached(self, call: _Call) -> None:
        """Cancel a detached call's task and deliver its cancellation."""
        if call.task.done() or call.status == "answered" or call.stopping:
            return
        self._stop(call)
        self._deliver(
            call,
            ToolMessage(
                call_id=call.call.call_id,
                timestamp=now_timestamp(),
                state="cancelled",
                shown=self.stubs.cancelled,
            ),
        )
        call.status = "answered"

    # Queues how the call ended and returns what it queued: the reply, or the
    # failure. A call cancelled returns ``None``.
    async def _stream_and_post(self) -> ModelMessage | ModelResponseCompleted | None:
        """Stream a model response; queue how the call ended, and return it."""
        queued: ModelMessage | ModelResponseCompleted
        try:
            response = await self._reply()
        except asyncio.CancelledError:
            # A call ``_cancel_model`` cancelled was told to have ended there, and is
            # the runtime's call no longer.
            if not _cut_off(self.model_call):
                return None
            # No one stopped this one: its stream ended cancelled of itself. That is
            # a call that failed, and a failure frees the model and waits.
            queued = ModelResponseCompleted(
                outcome=Failed(
                    exception=DisconnectedError(
                        "The reply was cut off before it finished.",
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- Model-call failures from adapters must be routed into the runtime error record.
            log_exception_or_warning(logger, "model call failed", exc)
            queued = ModelResponseCompleted(outcome=Failed(exception=exc))
        else:
            queued = ModelMessage(response=response)
        self.inbox.push_back(queued)
        return queued

    def request(
        self,
        *,
        context: Sequence[ModelContextEvent] | None = None,
        after: Sequence[SessionRecord] = (),
        max_response_tokens: int | None = None,
    ) -> Request:
        """Return the request the next model call sends, with ``after`` appended.

        Its prefix is the last call's, byte for byte, so a side call such as a
        summary that appends to it reads the cache that call wrote.

        Args:
          context: What to send in place of the context, such as a prefix of
            it a summary reads; ``None`` sends the context.
          after: Records to add after the context.
          max_response_tokens: Explicit cap; ``None`` asks ``reply_room``, while
            ``0`` bypasses it for sizing without recursion.

        Returns:
          request: The system prompt, the context and ``after``, and every tool;
            each tool result ends with its ``result_note``, which is never kept.

        """
        records = sent_records(
            self.tape.context().messages if context is None else context,
            keep_senders_apart=self.model.capability.keeps_user_turns_apart,
            provider=self.model.provider.name,
        )
        note = self.result_note
        return self.tagged(
            Request(
                system=self.system,
                records=[
                    *(
                        r
                        if note is None or not isinstance(r, ToolMessage)
                        else _noted(r, note(r))
                        for r in records
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
            compaction=self._compaction_trigger,
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
          sent: ``request``, tagged, without sealed reasoning whose prefix changed.

        """
        sent = bound(self.tagged(request))
        self._steps += 1
        return sent

    async def _reply(self) -> Response:
        """Ask the model about the context; post its words as they arrive."""
        response = None
        # Left out of the request, it would leave the model answering as if
        # nothing came before; a compaction that failed to rewrite it says so.
        provider = self._called_model.provider.name
        sealed = {
            e.provider
            for e in self.tape.context().messages
            if isinstance(e, ContextCompaction) and unreadable(e, provider=provider)
        }
        if sealed:
            # pragma: no mutate[diagnostic-text] start
            names = ", ".join(sorted(sealed))
            raise UserFacingError(
                f"The context holds a summary only {names} can read, and"
                f" compacting it for {provider} failed. Switch back to {names},"
                " or clear the context.",
            )
            # pragma: no mutate end
        request = self.sending(
            dataclasses.replace(self.request(), recheck_quota=self.recheck_quota),
        )
        self.recheck_quota = False
        # Only this request starts with the agent's tools and system prompt; a
        # compaction's, with its own prompt and none, warms nothing this reuses.
        self._last_request = time.monotonic()
        snapshot = self.token_accounting.capture(request, model=self._called_model)
        # Only the first call after a compaction is told of it.
        self._compaction_trigger = ""
        async for event in self._called_model.stream(request):
            if isinstance(event, TextChunk):
                self._streamed_chars += len(event.text)
                self.inbox.push_back(
                    ModelResponseThinkingStreamed(
                        text=event.text,
                        response_record_index=event.record,
                    )
                    if event.thinking
                    else ModelResponseTextStreamed(
                        text=event.text,
                        response_record_index=event.record,
                    ),
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
            records=stamped(request, records=response.records),
            measurement=self.token_accounting.observe(snapshot, tokens=response.tokens),
        )

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
        self._calls[call.call_id].sleeping = False
        await self._run_tool_and_post(self._calls[call.call_id])

    # A failure or a cancellation is still a result, so every call is answered; the
    # loop decides where it goes, by what the call is when it arrives.
    async def _run_tool_and_post(self, running: _Call) -> None:
        """Run one tool invocation, post the ``ToolMessage`` to the inbox."""
        call = running.call
        tool = self.tools.get(call.name)
        if tool is None or running.refusal is not None:
            self.inbox.push_back(
                ToolMessage(
                    call_id=call.call_id,
                    timestamp=now_timestamp(),
                    state="failed",
                    stderr=(
                        running.refusal
                        if running.refusal is not None
                        else f"Unknown tool: {call.name}",
                    ),
                ),
            )
            return
        result: ToolMessage
        try:
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
                running.result_future = streaming_result.result
                takes_controls = self.allow_background and backgroundable(tool)
                watch = _Watch(
                    name=call.name,
                    started=time.monotonic(),
                    runs_until_stopped=streaming_result.runs_until_stopped,
                    every_sec=wake_period(call.arguments) if takes_controls else None,
                )
                running.watch = watch
                # A job the model asked for apart wakes it only if the call said
                # when; a tool that manages jobs never does.
                if takes_controls and (
                    not running.background or watch.every_sec is not None
                ):
                    self._arm(call.call_id, watch)
                try:
                    # A cancel here stops the tool; ``settled`` still returns its
                    # output.
                    result = await settled(
                        streaming_result,
                        call_id=call.call_id,
                        on_item=watch.add,
                    )
                finally:
                    watch.end()
                # The tool says when its result is due; a relayed record may say so.
                if result.due_after is None:
                    result = dataclasses.replace(
                        result,
                        due_after=streaming_result.due_after,
                    )
        except asyncio.CancelledError:
            result = ToolMessage(call_id=call.call_id, state="cancelled")
        except Exception as exc:  # noqa: BLE001 -- Tool failures must be recorded without stopping the dispatch engine.
            result = ToolMessage(
                call_id=call.call_id,
                state="failed",
                stderr=(describe(exc),),
            )
        if result.state == "cancelled":
            result = dataclasses.replace(result, shown=self.stubs.cancelled)
        self.inbox.push_back(dataclasses.replace(result, timestamp=now_timestamp()))
        for event in running.events_after_result:
            self.inbox.push_back(event)

    # Queues how the compaction ended and returns what it queued, as a model call
    # does. One a ``ModelHalt``, ``SessionClear`` or ``SessionQuit`` stopped was told
    # to have ended there, and returns ``None``: the runtime waits for it no more.
    async def _compact_and_post(
        self,
        request: SessionCompact,
    ) -> SessionCompactCompleted | None:
        """Run the compactor and adopt its splice; queue how ``request`` ended."""
        queued = await self._compacted(request)
        if queued is not None:
            self.inbox.push_back(queued)
        return queued

    # What the tape refuses of the splice is a failure as what the compactor raises
    # is: either way the tape is as it was.
    async def _compacted(
        self,
        request: SessionCompact,
    ) -> SessionCompactCompleted | None:
        """Run the compactor and adopt its splice; return how it went, ``None`` if stopped."""
        if self.compactor is None:
            return SessionCompactCompleted(outcome=Succeeded(value=SessionCompacted()))
        tape_len = len(self.tape)
        tape_snapshot = tuple(self.tape)
        try:
            override = await self.compactor.compact(
                tape_snapshot,
                self.tape.context().messages,
                self.model,
                self.tape.mint_ref,
                custom_instructions=request.summary_must_keep or None,
            )
            if self.compact_task is not asyncio.current_task():
                return None
            override = widen_barrier_mask(override, tape_snapshot)
            self.tape.adopt_record(override, discards_content=True)
        except asyncio.CancelledError:
            # As a model call's: stopped by the runtime it was told there; cut off
            # by no one, it failed.
            if not _cut_off(self.compact_task):
                return None
            return SessionCompactCompleted(
                outcome=Failed(
                    exception=DisconnectedError(
                        "The summary was cut off before it finished.",
                    ),
                ),
                tape_len_before=tape_len,
            )
        except Exception as exc:  # noqa: BLE001 -- Compaction adapter failures must be routed into the runtime error record.
            log_exception_or_warning(logger, "compaction failed", exc)
            if self.compact_task is not asyncio.current_task():
                return None
            return SessionCompactCompleted(
                outcome=Failed(exception=exc),
                tape_len_before=tape_len,
            )
        return SessionCompactCompleted(
            outcome=Succeeded(value=SessionCompacted.from_splice(override)),
        )


# A task that crashed, or was cancelled, queued nothing.
def _queued_by[T](task: asyncio.Task[T | None] | None) -> T | None:
    """Return what ``task`` queued as it ended; ``None`` while it runs, or for none."""
    if task is None or not task.done() or task.cancelled() or task.exception():
        return None
    return task.result()


# The runtime clears its hold on a task before it cancels it, and whoever else
# cancels a task (a loop shutting down) leaves it marked as being cancelled.
def _cut_off(held: asyncio.Task[object] | None) -> bool:
    """Return whether the running task, the one ``held``, ended cancelled of itself."""
    task = asyncio.current_task()
    return task is not None and held is task and not task.cancelling()


def _type_names(types: Sequence[type[object]]) -> tuple[str, ...]:
    """Return class names for compact debug logging."""
    return tuple(t.__name__ for t in types)


def _message_ending(message: _Message, outcome: Outcome[None]) -> _Ending:
    """Return the ending of ``message``, as ``outcome`` says it went."""
    match message:
        case UserMessage():
            return UserMessageCompleted(outcome=outcome)
        case AgentMessage():
            return AgentMessageCompleted(outcome=outcome)
        case SystemMessage():
            return SystemMessageCompleted(outcome=outcome)
        case ToolMessage():
            return ToolMessageCompleted(outcome=outcome)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RunningJob:
    """A tool call the runtime is running, as ``Runtime.jobs`` lists it."""

    call: ToolCall
    status: Literal["running", "sleeping", "detached"]
    runs_until_stopped: bool
    runtime: Runtime

    # Every call the runtime runs was stamped as its reply was taken in.
    @property
    def elapsed_sec(self) -> float:
        """How long since the model asked for the call."""
        asked = self.call.timestamp
        return 0.0 if asked is None else (datetime.now(UTC) - asked).total_seconds()

    def cancel(self) -> None:
        """Ask the runtime to cancel the call."""
        self.runtime.push(ToolHalt(call_id=self.call.call_id))


type _Message = UserMessage | AgentMessage | SystemMessage | ToolMessage

# What may wait, held for a moment to come; a clear never waits.
type _Held = SessionCompact | SessionPrune | _Message

type _Request = _Held | SessionClear


# What a held request waits for, named by the state told when it comes, as
# ``due_after`` names its own: the reply streaming now to end; the model to be done;
# the agent to be done; the next model round, for a detached call's result; nothing
# at work, the cohort done, for a prune; the compaction under way to end.
type _Until = type[
    ModelResponseCompleted
    | ModelIdle
    | AgentIdle
    | ModelResponseStarted
    | ToolCohortCompleted
    | SessionCompactCompleted
]

type _Ending = (
    SessionCompactCompleted
    | SessionClearCompleted
    | SessionPruneCompleted
    | UserMessageCompleted
    | AgentMessageCompleted
    | SystemMessageCompleted
    | ToolMessageCompleted
)


# ``stage`` is where it stands: ``held`` until the state ``until`` is told,
# ``in_turn`` (it ends as the turn does), or ``open`` (acted on, or yet to be).
# Each is itself, as its request is: ``list.remove`` must take this one, not one
# whose request is equal field for field.
@dataclass(slots=True, kw_only=True, eq=False)
class _Taken:
    """A request taken in and not yet ended, what whoever asked it awaits, and where it stands."""

    request: _Request
    ended: asyncio.Future[Outcome[SessionCompacted] | Outcome[None]] | None = None
    stage: Literal["open", "held", "in_turn"] = "open"
    until: _Until | None = None
    owed: bool = True
    """Whether its ending is still to be told: a second hold of one request owes none."""


# ``result_future`` is the task the tool runs in, for most tools, so the tool sees it
# as ``asyncio.current_task()``. ``group`` is what detaches as one: the calls of one
# key between ``"*"`` calls. ``status`` is where the call stands for the model: it
# waits on it; a stub answered it and it runs apart; or its result is taken in, or
# stands taken (a kill, a clear). ``sleeping`` is the task's own: not yet started.
@dataclass(slots=True, kw_only=True)
class _Call:
    """A call the runtime runs, and where it stands."""

    call: ToolCall
    key: str | None
    task: asyncio.Task[None]
    background: bool
    status: Literal["waited", "apart", "answered"]
    refusal: str | None
    group: tuple[ToolCall, ...] = ()
    sleeping: bool = True
    result_future: asyncio.Future[object] | None = None
    watch: _Watch | None = None
    foreground_waiters: list[asyncio.Future[ToolMessage]] = dataclasses.field(
        default_factory=list[asyncio.Future[ToolMessage]],
    )
    events_after_result: list[RuntimeEvent] = dataclasses.field(
        default_factory=list[RuntimeEvent],
    )

    # A later call that contends with it waits until it has stopped.
    @property
    def stopping(self) -> bool:
        """Whether the call was cancelled and its task has not yet stopped."""
        return not self.task.done() and self.task.cancelling() > 0


# ``seen`` counts the ``items`` the model has read; ``cached`` is the model call
# whose cache deadline set the armed wake; ``waking`` says a wake asked to tell the
# model what the call wrote.
@dataclass(slots=True, kw_only=True)
class _Watch:
    """What a running call wrote, how much of it the model has read, and its next wake."""

    name: str
    started: float
    runs_until_stopped: bool = False
    every_sec: float | None = None
    items: list[object] = dataclasses.field(default_factory=list[object])
    seen: int = 0
    timer: asyncio.TimerHandle | None = None
    cached: float | None = None
    waking: bool = False

    @property
    def fresh(self) -> bool:
        """Whether the call wrote something the model has not read."""
        return len(self.items) > self.seen

    def add(self, stream: Literal["stdout", "stderr"], item: object) -> None:
        """Keep ``item``, which the call wrote on ``stream``."""
        del stream
        self.items.append(item)

    def end(self) -> None:
        """Cancel the next wake: the tool ended."""
        if self.timer is not None:
            self.timer.cancel()


# A call that has yet to start has said nothing of itself.
def _runs_until_stopped(call: _Call) -> bool:
    """Return whether the call's tool said it ends only when it is stopped."""
    return call.watch is not None and call.watch.runs_until_stopped


# Through ``shown``, so the result's attachments are sent as before.
def _noted(result: ToolMessage, note: str) -> ToolMessage:
    """Return ``result`` reading ``note`` on a line after its text; as it is, for none."""
    return (
        dataclasses.replace(result, shown=f"{result.text()}\n{note}")
        if note
        else result
    )


def _cache_key(model: Model) -> tuple[str, str]:
    """Return what a prompt cache belongs to: the provider and the model."""
    return model.provider.name, model.capability.model_id


def _stamped(response: Response) -> Response:
    """Return ``response`` with each call it makes stamped now."""
    when = now_timestamp()
    return dataclasses.replace(
        response,
        records=tuple(
            dataclasses.replace(r, timestamp=when)
            if isinstance(r, ToolCall) and r.timestamp is None
            else r
            for r in response.records
        ),
    )


# ``"*"`` is a call whose resources are unknown: it waits for every keyed call before
# it, and every call after it waits for it. An unkeyed call touches nothing another
# does, so nothing waits for it.
def _contend(key: str | None, earlier: str | None) -> bool:
    """Return whether a call keyed ``key`` must wait for an earlier one keyed ``earlier``."""
    return earlier is not None and (key in (earlier, "*") or earlier == "*")


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
            if not isinstance(messages[i], (SystemMessage, SessionContext))
        ),
        None,
    )
