"""Agent: actor-model with one foreground slot, observer fan-out, mailbox.

Three primitives, one role each (see ``docs/private/agent_refactor.md``
and ``docs/private/execution_model.md``):

- **Mailbox** (``self.inbox``): external work to do later. ``Inbox`` of
  source-tagged messages with bulk-drain semantics.
- **Foreground slot** (``self.work``): the one in-flight strategy task
  (a round body, compaction, or clear). Cancel via ``self.work.cancel()``
  (sync, foreign-task safe).
- **Observers** (``self.observers``): synchronous fan-out callables that
  receive ``Event`` values for rendering and forwarding.

One round = one drain + one model call + (maybe) one cohort spawn. The
round loop never directly awaits tools; cohorts run fire-and-forget and
emit their consolidated ``multipart/x-tool-batch-result`` back to the
inbox when ready (or via ``force_close``-induced decay-to-background).
See ``execution_model.md`` §1.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from pathlib import Path
from typing import Literal, cast

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import logging
import time
import uuid

from sagent.agent.cohort import Cohort, CohortMember
from sagent.agent.compaction import (
    CompactionState,
    estimate_total_tokens,
    post_compact_enrich,
)
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.dispatch import (
    add_tool_input_batch_hint,
    conditional_rules_for_request,
    invoke_tool,
    tc_directive,
    tc_tool_id,
    tool_call_label,
)
from sagent.agent.inbox import (
    QUIT_SOURCE,
    TOOLS_SOURCE,
    USER_SOURCE,
    Inbox,
    InboxItem,
)
from sagent.agent.retry import send_with_retry
from sagent.agent.session_io import (
    SessionMeta,
    append_session,
    load_message,
    load_session,
    rebuild_tool_state_from_messages,
    repair_dangling_tool_calls,
    restore_model,
    text_from_msg,
)
from sagent.custom_exceptions import ModelTerminationError
from sagent.custom_types import (
    Compactor,
    ContextBudget,
    ErrorEvent,
    Event,
    InterruptedEvent,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    MultipartMessage,
    StatusUpdateEvent,
    StreamEndEvent,
    TextChunkEvent,
    TextMessage,
    ThinkingEvent,
    TokenCount,
    Tool,
    ToolLabelEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UserBarEvent,
    reset_id_counter,
)
from sagent.lib.compaction import write_pre_compact_transcript
from sagent.lib.descriptors import (
    QUIT_SENTINEL,
    has_error,
)
from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.lib.message import (
    get_queue_id,
    get_tool_name,
    response_tool_calls,
)
from sagent.providers.lib.stop_reason import BENIGN_STOP_REASONS
from sagent.sessions import parse_jsonl
from sagent.tools.background_task import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
)
from sagent.tools.core import (
    ToolState,
    agent_counter_var,
    agent_label_var,
    agent_registry,
    changed_files_context,
    cost_root_var,
    current_agent_var,
    tool_state_context,
    tool_state_var,
)
from sagent.tools.result_storage import (
    ReplacementState,
    enforce_message_budget,
    inject_empty_marker,
    persist_result,
)


logger = logging.getLogger(__name__)


SystemPrompt = str | dict[str, str | Callable[[], str]]
SystemPromptArg = SystemPrompt | Callable[[], SystemPrompt]


ERROR_MAX_TOOL_CALL_ROUNDS = "error:max_tool_call_rounds"

MAX_OVERFLOW_RECOVERY = 3
MAX_TRUNCATION_RECOVERY = 3
MAX_COMPACT_FAILURES = 3

_RECOVERABLE_TRUNCATION = frozenset(
    {"max_tokens", "model_context_window_exceeded"},
)
_TRUNCATION_NUDGE = (
    "Output token limit hit. Resume directly - no apology, no recap of "
    "what you were doing. Pick up mid-thought if that is where the cut "
    "happened. Break remaining work into smaller pieces."
)

_MAX_UNSAVED_EVENTS = 1000


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PendingOp:
    """One queued strategy mutation set by a tool, drained at top of next round."""

    kind: Literal["compact", "recompact", "clear"]
    args: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _CostState:
    """Opaque cost-lifecycle token; ``token=None`` means we did not open."""

    token: contextvars.Token[CostTracker | None] | None


@dataclasses.dataclass(kw_only=True, slots=True)
class ActivityTracker:
    """Lifecycle counters for the bottom toolbar.

    Attributes:
      elapsed_seconds: Cumulative wall-clock time spent on model calls.
      current_call_start: Monotonic time of the active call, or 0.0.
      live_response_chars: Streaming chars accumulated this call.
      active: True iff a model call is currently in flight.
      num_tool_call_rounds: Cumulative tool-call rounds for this session.
      truncation_recoveries: Consecutive max-tokens truncation recoveries.

    """

    elapsed_seconds: float = 0.0
    current_call_start: float = 0.0
    live_response_chars: int = 0
    active: bool = False
    num_tool_call_rounds: int = 0
    truncation_recoveries: int = 0


class Agent:
    """Conversation agent: mailbox + one foreground slot + observer fan-out."""

    supports_microcompaction: bool = False

    def __init__(
        self,
        *,
        model: Model,
        model_spec: ModelSpec | None = None,
        system: SystemPromptArg = "",
        tools: list[Tool] | None = None,
        compactor: Compactor | None = None,
        session_dir: str | Path | None = None,
        budget: ContextBudget | None = None,
        max_attempts: int = 5,
        name: str = "Agent",
        description: str = "An AI agent.",
        max_tool_call_rounds: int | None = None,
        thinking: str | None = "adaptive",
        effort: str | None = None,
        max_budget_usd: float | None = None,
        persistent_retry: bool = False,
        track_changed_files: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        self.directive_schema: JSON = json_freeze(
            {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question.",
                    },
                },
                "required": ["prompt"],
            },
        )
        self.model = model
        self.model_spec = model_spec
        self._system_factory: Callable[[], SystemPrompt] = cast(
            "Callable[[], SystemPrompt]",
            system if callable(system) else (lambda s=system: s),
        )
        self.tools_map: dict[str, Tool] = {}
        for t in tools or []:
            if t.tool_id in self.tools_map:
                raise ValueError(f"Duplicate tool: {t.tool_id!r}")
            self.tools_map[t.tool_id] = t
        self.compactor = compactor
        if budget is not None:
            _validate_budget(budget, model)
        self._budget = budget
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self.max_tool_call_rounds = max_tool_call_rounds
        self._thinking = thinking
        if effort is not None and not model.supports_effort:
            raise ValueError(
                f"Model {model.model_id!r} does not support effort"
                f" (got effort={effort!r}).",
            )
        self._effort = effort
        self._cache_ttl: str = "5m"
        self.persistent_retry = persistent_retry
        self._track_changed_files = track_changed_files
        self.cost_tracker = CostTracker()
        self._max_budget_usd = max_budget_usd
        self.inbox: Inbox = Inbox()
        self.history: list[Message] = []
        self.tool_state = ToolState()
        self.activity = ActivityTracker()
        self.compaction_state = CompactionState()
        self.replacement_state = ReplacementState(
            persist_threshold=self.budget.persist_threshold,
            message_budget=self.budget.message_budget_chars,
        )
        self.observers: list[Callable[[Event], None]] = []
        self.work: asyncio.Task[object] | None = None
        self.background: dict[str, BackgroundTaskEntry] = {}
        self.active_children: dict[str, object] = {}
        # Cohorts spawned by past rounds that haven't yet emitted their
        # consolidated tool-result bundle. Force-closed at the top of the
        # next round when their results haven't naturally landed yet.
        self._active_cohorts: list[Cohort] = []
        self._next_op: PendingOp | None = None
        self._shutting_down: bool = False
        # When True, the round body's ``except CancelledError`` performs
        # /halt semantics (expunge zombie response, requeue items, set
        # ``block_until_user``). Plain ``cancel()`` does not set this.
        self._halt_requested: bool = False
        # Optional yield-queue for the consumer-facing event subset
        # (UserBarEvent, ToolResultEvent, TurnCompleteEvent,
        # InterruptedEvent). Set only while ``run()`` is driving rounds;
        # outside ``run()`` those events go to ``self.publish``.
        self._yield_queue: asyncio.Queue[Event | object] | None = None
        self._session_id = str(uuid.uuid4())[:8]
        self._status: str = ""
        self._event_log: list[dict[str, object]] = []
        self._persisted_idx: int = 0
        # Lifecycle bookkeeping for ``last_run_*`` deltas. Set at
        # ``_open_cost_lifecycle``, cleared at ``_close_cost_lifecycle``.
        self._run_start_tokens: TokenCount | None = None
        self._run_start_cost_usd: float = 0.0
        self._last_run_tokens: TokenCount = TokenCount()
        self._last_run_cost_usd: float = 0.0
        self._persistent: bool = False
        self.session_dir: Path | None = None
        if session_dir is not None:
            self.session_dir = Path(session_dir)
            self.replacement_state.storage_dir = self.session_dir
            self._load_session()

    # -- Properties / config surface ----------------------------------

    @property
    def budget(self) -> ContextBudget:
        """Context budget; auto-derived from the model when unset."""
        return (
            self._budget
            if self._budget is not None
            else ContextBudget.from_model(self.model)
        )

    @property
    def max_request_tokens(self) -> int:
        """Maximum input tokens this agent will send."""
        return self.budget.max_request_tokens

    @max_request_tokens.setter
    def max_request_tokens(self, value: int) -> None:
        if value > self.model.max_request_tokens:
            raise ValueError(
                f"max_request_tokens={value:,} exceeds model's"
                f" {self.model.max_request_tokens:,}",
            )
        self._budget = dataclasses.replace(self.budget, max_request_tokens=value)

    @property
    def max_response_tokens(self) -> int:
        """Maximum output tokens reserved for the model's response."""
        return self.budget.max_response_tokens

    @max_response_tokens.setter
    def max_response_tokens(self, value: int) -> None:
        if value > self.model.max_response_tokens:
            raise ValueError(
                f"max_response_tokens={value:,} exceeds model's"
                f" {self.model.max_response_tokens:,}",
            )
        self._budget = dataclasses.replace(self.budget, max_response_tokens=value)

    def reset_budget(self) -> None:
        """Clear explicit budget so it auto-derives from the model on next access."""
        self._budget = None

    @property
    def thinking(self) -> str | None:
        """Thinking mode for this agent (e.g. ``"adaptive"``)."""
        return self._thinking

    @thinking.setter
    def thinking(self, value: str | None) -> None:
        self._thinking = value

    @property
    def effort(self) -> str | None:
        """Effort level for this agent (provider-specific)."""
        return self._effort

    @effort.setter
    def effort(self, value: str | None) -> None:
        if value is not None and not self.model.supports_effort:
            raise ValueError(f"Model {self.model.model_id!r} does not support effort.")
        self._effort = value

    @property
    def cache_ttl(self) -> str:
        """Prompt-cache TTL for outgoing requests (``"5m"`` or ``"1h"``)."""
        return self._cache_ttl

    @cache_ttl.setter
    def cache_ttl(self, value: str) -> None:
        if value not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {value!r}")
        self._cache_ttl = value

    @property
    def session_id(self) -> str:
        """Stable 8-char ID for this agent's session log."""
        return self._session_id

    @property
    def status(self) -> str:
        """Session status for the terminal titlebar."""
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value
        self.save_session()
        self.publish(StatusUpdateEvent(value))

    @property
    def messages(self) -> list[Message]:
        """Alias for ``history`` (v1/v2 compatibility)."""
        return self.history

    @property
    def tools(self) -> list[Tool]:
        """Tools available to this agent (order preserved)."""
        return list(self.tools_map.values())

    @property
    def system(self) -> SystemPrompt:
        """The system prompt this agent was constructed with."""
        return self._system_factory()

    @property
    def active(self) -> bool:
        """True while the foreground slot is occupied or a model call is in flight."""
        return self.work is not None or self.activity.active

    @property
    def total_cost_usd(self) -> float:
        """Cumulative subtree USD cost; thin alias for ``cost_tracker.total_cost_usd``."""
        return self.cost_tracker.total_cost_usd

    @property
    def total_tokens(self) -> TokenCount:
        """Cumulative subtree token counts; thin alias for ``cost_tracker.total``."""
        return self.cost_tracker.total

    @property
    def last_run_tokens(self) -> TokenCount:
        """Token counts for the current/most-recent run.

        While a run is active, returns the running delta
        (``cost_tracker.total - _run_start_tokens``). After the run
        ends, returns the final delta captured at lifecycle close.
        """
        if self._run_start_tokens is not None:
            return self.cost_tracker.total - self._run_start_tokens
        return self._last_run_tokens

    @property
    def last_run_cost_usd(self) -> float:
        """USD cost for the current/most-recent run."""
        if self._run_start_tokens is not None:
            return self.cost_tracker.total_cost_usd - self._run_start_cost_usd
        return self._last_run_cost_usd

    @property
    def total_active_elapsed_seconds(self) -> float:
        """Cumulative wall-clock time spent active across the session."""
        if self.activity.active and self.activity.current_call_start > 0:
            return self.activity.elapsed_seconds + (
                asyncio.get_running_loop().time() - self.activity.current_call_start
            )
        return self.activity.elapsed_seconds

    @property
    def request_start_time(self) -> float:
        """Event-loop timestamp when the current model call started."""
        return self.activity.current_call_start

    @property
    def live_model_response_tokens(self) -> int:
        """Live output-token estimate for the in-flight model call."""
        return self.activity.live_response_chars // self.budget.chars_per_token

    @property
    def num_tool_call_rounds(self) -> int:
        """Cumulative tool-call rounds for this session."""
        return self.activity.num_tool_call_rounds

    # -- Tool-shaped helpers (Agent-as-Tool ergonomics) ---------------

    def summary(self, msg: Message) -> str:
        """Return the agent's name (Tool-protocol shape)."""
        del msg
        return self.name

    def prompt(self) -> str:
        """Per-request system-prompt contribution (none)."""
        return ""

    def system_prompt(self) -> str:
        """Assemble the current system prompt from sections + tool contributions.

        Returns:
          text: Concatenated system prompt for one model request.

        """
        system = self._system_factory()
        parts: list[str] = []
        if isinstance(system, str):
            if system:
                parts.append(system)
        else:
            for value in system.values():
                content = value if isinstance(value, str) else value()
                if content:
                    parts.append(content)
        if self._track_changed_files:
            diff = changed_files_context()
            if diff:
                parts.append(diff)
        for tool in self.tools_map.values():
            section = tool.prompt()
            if section:
                parts.append(section)
        return "\n\n".join(parts)

    def swap_model(self, model: Model, *, spec: ModelSpec | None = None) -> None:
        """Replace the active model and (optionally) its spec.

        Args:
          model: New model instance.
          spec: Recipe that produced ``model``; ``None`` clears it.

        Raises:
          ValueError: Explicit budget exceeds the new model's limits.

        """
        if self._budget is not None:
            _validate_budget(self._budget, model)
        self.model = model
        self.model_spec = spec
        self.save_session()

    # -- Foreground slot + cancel verbs -------------------------------

    async def _start_foreground[T](self, coro: Coroutine[object, object, T]) -> T:
        """Cancel current foreground; spawn ``coro`` as the new foreground task.

        Awaits the spawned task and returns its result. The caller's task is
        NEVER the foreground -- ``coro`` runs in a fresh task that becomes
        ``self.work``. Cancelling ``self.work`` cancels the strategy work, not
        the caller. The caller's own ``await`` observes the cancel as a
        ``CancelledError`` raised at its await point.

        Used by the public strategy methods (``compact``, ``recompact``,
        ``clear``) to claim the foreground slot. The round loop also writes
        to ``self.work`` directly when running a round body so /halt and
        /kill can address them by the same handle.
        """
        self.cancel()
        if self.work is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.work
        task = asyncio.create_task(coro)
        self.work = task
        try:
            return await task
        finally:
            if self.work is task:
                self.work = None

    def cancel(self) -> None:
        """Cancel the current foreground task. Sync; safe from any task.

        Does NOT set halt semantics. The round body treats a bare cancel
        as a plain interrupt (publish ``InterruptedEvent``, account
        partial cost) and propagates.
        """
        if self.work is not None and not self.work.done():
            _ = self.work.cancel()

    def halt(self) -> None:
        """``/halt``: cancel + arm halt semantics in the round body.

        Sets ``_halt_requested``; the round body's ``except CancelledError``
        will expunge any zombie response, cancel the just-spawned cohort,
        ``requeue_front`` drained items, and ``block_until_user``.

        Composes freely with :meth:`kill_tool` / :meth:`kill_all_tools`.
        """
        self._halt_requested = True
        self.cancel()

    def kill_tool(self, qid: str) -> bool:
        """``/kill <qid>``: cancel one outstanding tool task by queue-id.

        Searches active cohorts and ``self.background``. Cohort members
        produce a ``[Cancelled by user]`` ``tool_result`` via the cohort's
        natural emission; bg tasks produce a ``[Background tool cancelled:
        ...]`` inbox message via the shared :meth:`_post_bg_completion`
        callback.

        Args:
          qid: Queue id of the task to cancel.

        Returns:
          killed: True if a matching task was found and cancelled.

        """
        for cohort in self._active_cohorts:
            for member in cohort.members:
                if member.tool_use_id == qid and not member.task.done():
                    _ = member.task.cancel()
                    return True
        job = self.background.get(qid)
        if job is not None and not job.task.done() and not job.hidden:
            _ = job.task.cancel()
            return True
        return False

    def kill_all_tools(self) -> int:
        """``/kill all``: cancel every outstanding tool task.

        Returns:
          count: Number of tasks cancelled.

        """
        count = 0
        for cohort in list(self._active_cohorts):
            for member in cohort.members:
                if not member.task.done():
                    _ = member.task.cancel()
                    count += 1
        for job in list(self.background.values()):
            if job.hidden:
                continue
            if not job.task.done():
                _ = job.task.cancel()
                count += 1
        return count

    def shutdown(self, *, force: bool = False) -> None:
        """End ``serve_forever`` cleanly.

        Args:
          force: When True, also cancel foreground + visible bg jobs.

        """
        self._shutting_down = True
        if force:
            self.cancel()
            for qid, job in list(self.background.items()):
                if job.hidden:
                    continue
                if job.kind == "persistent_subagent":
                    child = agent_registry.get(qid.removeprefix("persistent:"))
                    if isinstance(child, Agent):
                        child.shutdown(force=True)
                else:
                    _ = job.task.cancel()
        self.inbox.send(TextMessage("", QUIT_SENTINEL), source=QUIT_SOURCE)

    # -- Observer fan-out ---------------------------------------------

    def publish(self, event: Event) -> None:
        """Synchronously dispatch ``event`` to every observer.

        Args:
          event: Event to publish.

        """
        for obs in self.observers:
            try:
                obs(event)
            except Exception as e:  # noqa: BLE001 -- dispatch safety net
                logger.warning(
                    "observer raised on %s: %s: %s",
                    type(event).__name__,
                    type(e).__name__,
                    e,
                )
                logger.debug("observer traceback", exc_info=True)

    def _emit_yielded(self, event: Event) -> None:
        r"""Emit one consumer-facing event (UserBar / ToolResult / TurnComplete /
        InterruptedEvent).

        When ``run()`` is driving rounds, the event is pushed onto its
        yield queue so the consumer's ``async for`` sees it. Otherwise
        the event falls back to ``publish()`` so ``serve_forever``-style
        observers still see it. This split mirrors v2's ``run()`` shape
        where structural turn events were ``yield``\ ed (not published)
        and avoids double-counting in tools that wire both an observer
        and an ``async for`` consumer.
        """
        if self._yield_queue is not None:
            self._yield_queue.put_nowait(event)
        else:
            self.publish(event)

    # -- Persistent driver loop ---------------------------------------

    async def serve_forever(self) -> None:
        """Drive the agent until ``shutdown`` is called.

        Sets up the per-agent ContextVars (``current_agent_var``,
        ``cost_root_var``, ``agent_label_var``, ``tool_state_var``) and
        installs the agent in ``agent_registry``. Drains ``self.inbox``
        in bulk and runs one round body per drain (spec §1). Returns on
        QUIT_SENTINEL or ``shutdown``.
        """
        agent_token = current_agent_var.set(self)
        cost_state = self._open_cost_lifecycle()
        counter_token = agent_counter_var.set(itertools.count())
        label = agent_label_var.get("") or self.name
        label_token = agent_label_var.set(label)
        agent_registry[label] = self
        parent_state = tool_state_var.get(None)
        self.tool_state.depth = 0 if parent_state is None else parent_state.depth + 1
        try:
            with tool_state_context(self.tool_state):
                await self._round_loop()
        finally:
            self._close_cost_lifecycle(cost_state)
            if not self._persistent:
                _ = agent_registry.pop(label, None)
            agent_label_var.reset(label_token)
            agent_counter_var.reset(counter_token)
            current_agent_var.reset(agent_token)

    async def _round_loop(self) -> None:
        """Round loop body (ContextVars already installed).

        Each iteration:
          1. Block on ``inbox.drain()``.
          2. If shutdown or QUIT_SENTINEL: return.
          3. Spawn a round-body task as ``self.work``; await it. The
             round body internally handles cancellation (/halt semantics
             and the bare-cancel ``InterruptedEvent`` path).
          4. Catch other exceptions; publish ``ErrorEvent`` and continue.
        """
        while not self._shutting_down:
            items = await self.inbox.drain()
            if self._shutting_down:
                return
            quit_idx = next(
                (
                    idx
                    for idx, item in enumerate(items)
                    if item.msg.descriptor == QUIT_SENTINEL
                ),
                None,
            )
            if quit_idx is not None:
                items = items[:quit_idx]
                if items:
                    await self._run_round(items)
                return
            await self._run_round(items)

    async def _run_round(self, items: list[InboxItem]) -> None:
        """Spawn ``_round_body(items)`` as ``self.work`` and await it."""
        task = asyncio.create_task(self._round_body(items))
        self.work = task
        try:
            await task
        except asyncio.CancelledError:
            # ``_round_body`` handles its own cancel paths; if a
            # propagating CancelledError reaches us it means the
            # cancellation came from outside the round and we should
            # uncancel so the next ``inbox.drain()`` is fresh.
            cur = asyncio.current_task()
            if cur is not None:
                _ = cur.uncancel()
        except (AssertionError, TypeError, AttributeError, NameError):
            raise
        except Exception as e:
            # Operational errors (provider failures, retries exhausted,
            # budget caps, model refusal) are scoped to the active turn.
            # Keep the mailbox alive so the user can retry, switch models,
            # or quit. Bug-class exceptions are re-raised above.
            logger.exception(
                "turn failed on %s: %s", self.model.model_id, type(e).__name__
            )
            self.publish(ErrorEvent(f"turn failed: {type(e).__name__}: {e}"))
            self.log_event(
                "turn_failed",
                error_type=type(e).__name__,
                error=str(e),
            )
        finally:
            if self.work is task:
                self.work = None

    # -- Public strategy methods --------------------------------------

    async def run(self, msg: Message) -> AsyncGenerator[Event, None]:
        """Process one inbound message; drive rounds until the agent is idle.

        Convenience entrypoint used by tests and non-``serve_forever``
        callers. Sends ``msg`` into the inbox tagged as ``user`` and
        spins rounds until there's no inbox work, no active cohorts,
        and no pending op.

        Args:
          msg: Inbound message (typically ``text/x-user-message``).

        Yields:
          event: Observer-shaped events emitted across one or more rounds.

        """
        cost_state = self._open_cost_lifecycle()
        events: asyncio.Queue[Event | object] = asyncio.Queue()
        DONE = _SENTINEL_DONE
        prior_queue = self._yield_queue
        self._yield_queue = events
        self.inbox.send(msg, source=USER_SOURCE)

        async def _driver() -> None:
            try:
                while True:
                    if (
                        len(self.inbox) == 0
                        and not self._active_cohorts
                        and self._next_op is None
                    ):
                        return
                    if len(self.inbox) == 0:
                        # Active cohort outstanding: block until it emits.
                        round_items = await self.inbox.drain()
                    else:
                        round_items = self.inbox.drain_nowait()
                    await self._run_round(round_items)
            finally:
                events.put_nowait(DONE)

        drive_task = asyncio.create_task(_driver())
        try:
            while True:
                ev = await events.get()
                if ev is DONE:
                    break
                yield cast("Event", ev)
            await drive_task
        except asyncio.CancelledError:
            self.cancel()
            _ = drive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drive_task
            raise
        finally:
            self._yield_queue = prior_queue
            self._close_cost_lifecycle(cost_state)

    async def compact(self, args: str = "") -> None:
        """Preempt in-flight work and run compaction.

        Args:
          args: Optional custom compaction instructions.

        """
        await self._start_foreground(
            self._do_compact(args, count_failures=False, write_transcript=True),
        )

    async def recompact(self, args: str = "") -> None:
        """Preempt; reload most recent pre-compact transcript and re-run compaction.

        Args:
          args: Optional custom compaction instructions for the retry.

        """
        await self._start_foreground(self._do_recompact(args))

    async def clear(self) -> None:
        """Preempt and wipe history + file-tracking caches."""
        await self._start_foreground(self._do_clear_async())

    # -- Persistence (auto-called from strategy methods) --------------

    def save_session(self, *, clear: bool = False) -> None:
        """Append meta + new history + events to ``session_dir/session.jsonl``.

        Args:
          clear: When True, write a barrier line and re-persist the entire
              current ``history`` (used after wholesale-replacement paths).

        """
        if self.session_dir is None:
            return
        path = self.session_dir / "session.jsonl"
        delta = list(self.history) if clear else self.history[self._persisted_idx :]
        append_session(
            path,
            meta=self._build_session_meta().serialize(),
            messages_delta=delta,
            events=self._event_log,
            clear=clear,
        )
        self._persisted_idx = len(self.history)
        self._event_log.clear()

    def log_event(self, event: str, **data: object) -> None:
        """Record a structured event for ``session.jsonl``.

        Args:
          event: Event name.
          **data: Arbitrary key/value payload.

        """
        entry = {
            "ts": time.time(),
            "session": self.session_id,
            "agent": self.name,
            "event": event,
            **data,
        }
        self._event_log.append(entry)
        if len(self._event_log) > _MAX_UNSAVED_EVENTS:
            if self.session_dir is not None:
                try:
                    self.save_session()
                except OSError as save_err:
                    self.publish(
                        ErrorEvent(
                            text=(
                                f"event log save failed: "
                                f"{type(save_err).__name__}: {save_err}; "
                                f"truncating in memory"
                            ),
                        ),
                    )
                    logger.debug("event log save failed", exc_info=True)
                    self._event_log = self._event_log[-_MAX_UNSAVED_EVENTS:]
            else:
                self._event_log = self._event_log[-_MAX_UNSAVED_EVENTS:]
        logger.debug("%s: %s", event, data)

    # -- Round body ---------------------------------------------------

    async def _round_body(self, items: list[InboxItem]) -> None:
        """One round per spec §1: drain → merge → model call → maybe cohort.

        At the top: force-close any unsettled prior cohorts (their
        bundles must join this request -- every ``tool_use`` in history
        needs a paired ``tool_result``); drain post-emission arrivals;
        run the pending op (``compact`` / ``recompact`` / ``clear``);
        compact if needed. Then merge items, call the model, and on
        tool calls spawn a cohort fire-and-forget. ``/halt`` rolls back
        merge + zombie response and arms ``block_until_user``.
        """
        for cohort in list(self._active_cohorts):
            if not cohort.emitted:
                cohort.force_close()
        items = items + self.inbox.drain_nowait()

        if self._next_op is not None:
            op, self._next_op = self._next_op, None
            if op.kind == "compact":
                _ = await self._do_compact(op.args)
            elif op.kind == "recompact":
                await self._do_recompact(op.args)
            else:
                self._do_clear_sync()
                return

        cap = self.max_tool_call_rounds
        if cap is not None and self.activity.num_tool_call_rounds >= cap:
            self.history.append(
                MultipartMessage(
                    (
                        TextMessage(
                            f"Tool-call-round limit reached ({cap} rounds). "
                            f"[{ERROR_MAX_TOOL_CALL_ROUNDS}]",
                            "text/x-error",
                        ),
                    ),
                    "multipart/x-model-message",
                ),
            )
            self._emit_yielded(TurnCompleteEvent())
            return

        # Compaction position is load-bearing: it MUST run before the
        # merge step below so the ``merge_start_idx`` rollback in the
        # /halt CancelledError handler stays valid. Reversing the order
        # (merge then compact) lets the compactor rewrite ``self.history``
        # underneath us; ``del self.history[merge_start_idx:]`` would
        # then point into stale positions. If you ever need to move this
        # after merge for latency reasons, also switch the halt path to
        # reference-based rollback (capture merged Message objects, then
        # ``self.history.remove(m)`` each on cancel) or drop the rollback
        # entirely (let merged items stay in history; user redirect joins
        # them on the next round).
        await self._maybe_compact()
        merge_start_idx = len(self.history)
        if not self._merge_items_into_history(items):
            return

        response: ModelResponse | None = None
        try:
            while True:
                response = await self._call_model()
                self.history.append(response.content)
                self._guard_stop_reason(response)
                tools = response_tool_calls(response.content)
                if tools:
                    self.activity.num_tool_call_rounds += 1
                    _ = self._spawn_cohort(tools)
                    break
                if self._is_truncated_no_tools(response) and (
                    self._post_truncation_nudge()
                ):
                    response = None  # consumed; clear so /halt doesn't expunge
                    continue
                self._emit_yielded(TurnCompleteEvent())
                break
        except asyncio.CancelledError:
            if self._halt_requested:
                self._halt_requested = False
                # ``_spawn_cohort`` runs synchronously after the response
                # commit; a CancelledError that left ``response`` set
                # provably arrived before the cohort spawn, so no cohort
                # cleanup is needed here.
                if (
                    response is not None
                    and self.history
                    and self.history[-1] is response.content
                ):
                    _ = self.history.pop()
                del self.history[merge_start_idx:]
                self.inbox.requeue_front(items)
                self.inbox.block_until_user()
                self._emit_yielded(InterruptedEvent())
                self._account_cancelled()
                return
            self._emit_yielded(InterruptedEvent())
            self._account_cancelled()
            raise
        finally:
            self.save_session()

    def _merge_items_into_history(self, items: list[InboxItem]) -> bool:
        """Drain ``items`` into history; publish ``UserBarEvent`` for user content.

        Cohort bundles (``multipart/x-tool-batch-result``, source
        ``tools``) are unpacked into individual ``multipart/x-tool-result``
        history entries so each pairs with its ``tool_use`` parent.
        User / peer / bg items merge into one ``text/x-user-message``
        per spec §3 (single-item verbatim; multi-item tagged).

        Returns True iff the model should be called this round.
        """
        added_tool_results = False
        user_pieces: list[str] = []
        bg_pieces: list[str] = []
        peer_pieces: list[tuple[str, str]] = []
        for item in items:
            msg = item.msg
            if msg.descriptor == QUIT_SENTINEL or item.source == QUIT_SOURCE:
                continue
            if (
                item.source == TOOLS_SOURCE
                and msg.descriptor == "multipart/x-tool-batch-result"
            ):
                for part in cast("tuple[Message, ...]", msg.content):
                    if part.descriptor == "multipart/x-tool-result":
                        self.history.append(part)
                        self._emit_yielded(ToolResultEvent(part))
                        added_tool_results = True
                continue
            text = _user_text(msg)
            if not text and msg.descriptor != "text/x-user-message":
                continue
            if item.source == USER_SOURCE:
                user_pieces.append(text)
            elif item.source.startswith("bg_"):
                bg_pieces.append(text)
            else:
                peer_pieces.append((item.source, text))

        if user_pieces:
            self._emit_yielded(UserBarEvent(user_pieces[-1]))

        bundle = _format_user_bundle(user_pieces, peer_pieces, bg_pieces)
        if bundle is not None:
            self.history.append(TextMessage(bundle, "text/x-user-message"))
            return True
        return added_tool_results

    # -- Model call ---------------------------------------------------

    async def _call_model(self) -> ModelResponse:
        """Send one model request with retry, overflow recovery, thinking fallback.

        Returns:
          response: Completed model response.

        """
        self.activity.active = True
        self.activity.current_call_start = asyncio.get_running_loop().time()
        self.activity.live_response_chars = 0
        try:
            return await self._call_model_inner()
        finally:
            elapsed = (
                asyncio.get_running_loop().time() - self.activity.current_call_start
            )
            self.activity.elapsed_seconds += max(0.0, elapsed)
            self.activity.active = False
            self.activity.current_call_start = 0.0
            self.activity.live_response_chars = 0

    async def _call_model_inner(self) -> ModelResponse:
        """The actual send-with-retry loop; assumes ``activity`` already bracketed."""
        thinking_buffer: list[str] = []
        streamed = [0]

        def _on_text(chunk: str) -> None:
            if thinking_buffer:
                self.publish(ThinkingEvent("".join(thinking_buffer)))
                thinking_buffer.clear()
            streamed[0] += len(chunk)
            self.activity.live_response_chars += len(chunk)
            self.publish(TextChunkEvent(chunk))

        def _on_thinking(chunk: str) -> None:
            thinking_buffer.append(chunk)

        request = self._build_request()
        last_err: Exception | None = None
        response: ModelResponse | None = None
        for attempt in range(MAX_OVERFLOW_RECOVERY + 1):
            try:
                response = await send_with_retry(
                    self.model,
                    request,
                    on_text=_on_text,
                    on_thinking=_on_thinking,
                    max_attempts=self.max_attempts,
                    persistent_retry=self.persistent_retry,
                    log_event=self.log_event,
                    on_discarded_response=self._record_response,
                )
                break
            except asyncio.CancelledError:
                self._account_cancelled_partial(request, streamed[0])
                self.publish(StreamEndEvent())
                raise
            except Exception as e:
                if _is_thinking_unsupported(e) and request.thinking is not None:
                    logger.info(
                        "Model %r rejected thinking=%r; falling back to None",
                        self.model.model_id,
                        request.thinking,
                    )
                    self._thinking = None
                    request = dataclasses.replace(request, thinking=None)
                    continue
                if not self.model.is_context_overflow(e):
                    raise
                last_err = e
                if attempt >= MAX_OVERFLOW_RECOVERY:
                    break
                logger.info("Context overflow recovery attempt %d", attempt)
                _ = await self._do_compact("")
                request = dataclasses.replace(
                    request,
                    messages=[wrap_errors_for_llm(m) for m in self.history],
                    system=self.system_prompt(),
                )
        if response is None:
            raise RuntimeError(
                "context overflow recovery failed after "
                f"{MAX_OVERFLOW_RECOVERY} compactions",
            ) from last_err
        if thinking_buffer:
            self.publish(ThinkingEvent("".join(thinking_buffer)))
            thinking_buffer.clear()
        self.publish(StreamEndEvent())
        if streamed[0] == 0:
            for part in _text_parts(response.content):
                self.publish(TextChunkEvent(part))
        if streamed[0] == 0:
            for thinking in _thinking_parts(response.content):
                self.publish(ThinkingEvent(thinking))
        self._record_response(response)
        return response

    def _build_request(self) -> ModelRequest:
        """Snapshot history + system + tools into a ``ModelRequest``."""
        tools_list = list(self.tools_map.values())
        if tools_list and "application/x-tool-backgroundtask" in self.tools_map:
            tools_list = [
                t
                if t.tool_id == "application/x-tool-backgroundtask"
                else cast("Tool", BackgroundAwareTool(t))
                for t in tools_list
            ]
        return ModelRequest(
            messages=[wrap_errors_for_llm(m) for m in self.history],
            system=self.system_prompt(),
            tools=tools_list or None,
            max_response_tokens=self.max_response_tokens,
            thinking=self.thinking if self.model.supports_thinking else None,
            effort=self.effort if self.model.supports_effort else None,
            cache_ttl=self.cache_ttl,
        )

    def _guard_stop_reason(self, response: ModelResponse) -> None:
        """Raise on ``model_refusal`` or unrecognized non-benign stop reasons."""
        sr = response.stop_reason
        if sr == "model_refusal":
            raise RuntimeError(
                "Model refused to respond (content filter or usage policy).",
            )
        if sr and sr not in BENIGN_STOP_REASONS and sr not in _RECOVERABLE_TRUNCATION:
            raise ModelTerminationError(response)
        if sr not in _RECOVERABLE_TRUNCATION:
            self.activity.truncation_recoveries = 0

    def _is_truncated_no_tools(self, response: ModelResponse) -> bool:
        """True if the response was cut off mid-stream and has no tool calls."""
        if response.stop_reason not in _RECOVERABLE_TRUNCATION:
            return False
        return not response_tool_calls(response.content)

    def _post_truncation_nudge(self) -> bool:
        """Append a recovery nudge to history; return True if injected."""
        attempts = self.activity.truncation_recoveries
        if attempts >= MAX_TRUNCATION_RECOVERY:
            self.history.append(
                TextMessage("[truncation recovery exhausted]", "text/x-error"),
            )
            return False
        self.activity.truncation_recoveries = attempts + 1
        self.history.append(
            TextMessage(_TRUNCATION_NUDGE, "text/x-user-message"),
        )
        return True

    # -- Cohort spawn -------------------------------------------------

    def _spawn_cohort(self, calls: list[Message]) -> Cohort:
        """Build a cohort over ``calls``; fire-and-forget; register in active list.

        Every call becomes a Cohort member in emission order. Bg calls
        (``background:true`` directive, or ``delay>0``) are added with
        ``bg=True`` -- the cohort emits a ``[Running in background:
        <tool>]`` placeholder for them and doesn't wait. Read / Edit /
        Write are chained via :meth:`_file_op_in_chain`; everything else
        fans out. The cohort's ``on_emit`` runs :meth:`_finalize_results`
        on the bundle and sends a ``multipart/x-tool-batch-result`` to
        the inbox tagged ``tools``.
        """
        self.tool_state.bash_parse_cache.clear()
        for call in calls:
            self.log_event(
                "tool_call",
                tool=tc_tool_id(call),
                tool_id=get_queue_id(call),
                input=tc_directive(call),
            )

        cohort: Cohort | None = None  # bound below; captured by _on_emit

        def _on_emit(results: list[Message]) -> None:
            if cohort is not None:
                with contextlib.suppress(ValueError):
                    self._active_cohorts.remove(cohort)
            try:
                final = self._finalize_results(calls, results)
            except Exception:
                logger.exception("cohort finalize raised; emitting error bundle")
                final = [
                    MultipartMessage(
                        (
                            TextMessage(get_queue_id(c), "text/x-queue-id"),
                            TextMessage("cohort finalize failed", "text/x-error"),
                        ),
                        "multipart/x-tool-result",
                        parent_id=c.id,
                    )
                    for c in calls
                ]
            self.inbox.send(
                MultipartMessage(tuple(final), "multipart/x-tool-batch-result"),
                source=TOOLS_SOURCE,
            )

        cohort = Cohort(on_emit=_on_emit, on_promote_to_bg=self._promote_to_bg)
        self._active_cohorts.append(cohort)

        # Tool labels render in the model's emission order alongside its reasoning.
        for call in calls:
            tool = self.tools_map.get(tc_tool_id(call))
            self.publish(ToolLabelEvent(tool_call_label(tool, call)))

        prev: asyncio.Task[Message] | None = None
        has_fg = False
        for call in calls:
            qid = get_queue_id(call)
            name = get_tool_name(call)
            directive = tc_directive(call)
            is_bg = bool_val(directive.get("background"), False) or (
                int_val(directive.get("delay"), 0) > 0
            )
            if is_bg:
                _ = self._spawn_bg(call)  # creates task, registers post-completion
                task = self.background[qid].task
                cohort.add_member(
                    CohortMember(tool_use_id=qid, tool_name=name, task=task),
                    bg=True,
                )
            else:
                has_fg = True
                if _is_file_op(call):
                    task = asyncio.create_task(self._file_op_in_chain(prev, call))
                    prev = task
                else:
                    task = asyncio.create_task(self._invoke_tool_safe(call))
                cohort.add_member(
                    CohortMember(tool_use_id=qid, tool_name=name, task=task),
                )

        if not has_fg:
            cohort.force_close()  # no fg to wait for; emit placeholders now
        return cohort

    def _finalize_results(
        self,
        calls: list[Message],
        cohort_results: list[Message],
    ) -> list[Message]:
        """Postprocess + budget + batch-hint a cohort's settled results."""
        finalized = self._postprocess_results(calls, cohort_results)
        for r in finalized:
            self.log_event(
                "tool_result",
                tool_id=get_queue_id(r),
                is_error=has_error(r),
                content_len=len(text_from_msg(r)),
            )
        tool_names = {get_queue_id(c): tc_tool_id(c) for c in calls}
        budgeted = enforce_message_budget(
            finalized,
            tool_names,
            self.replacement_state,
        )
        return add_tool_input_batch_hint(budgeted)

    async def _file_op_in_chain(
        self,
        prev: asyncio.Task[Message] | None,
        call: Message,
    ) -> Message:
        """Run ``call`` after ``prev`` settles (file ops in emission order).

        Failures or cancellations of ``prev`` do not block ``call`` --
        the chain provides ordering, not dependency. But if *this* task
        is itself being cancelled (e.g. ``/kill <qid>``), propagate.
        """
        if prev is not None:
            try:
                await prev
            except asyncio.CancelledError:
                cur = asyncio.current_task()
                if cur is not None and cur.cancelling():
                    raise
            except Exception:  # noqa: BLE001 -- prev's failure is its problem
                logger.debug("prev file-op raised", exc_info=True)
        return await self._invoke_tool_safe(call)

    def _promote_to_bg(self, member: CohortMember) -> None:
        """Move a cohort member's task into ``self.background`` on decay.

        Called by ``Cohort.force_close`` for members still running when
        the cohort was interrupted. The task continues; its eventual
        result lands in the inbox via the shared
        :meth:`_post_bg_completion` done-callback.
        """
        qid = member.tool_use_id
        self.background[qid] = BackgroundTaskEntry(
            task=member.task,
            tool_name=member.tool_name,
            queue_id=qid,
            started=time.time(),
            kind="tool",
        )
        member.task.add_done_callback(
            functools.partial(self._post_bg_completion, qid, member.tool_name),
        )

    def _post_bg_completion(
        self,
        qid: str,
        tool_name: str,
        task: asyncio.Task[Message],
    ) -> None:
        """Done-callback for any bg task; post completion to the inbox.

        Shared by :meth:`_spawn_bg` (explicitly-backgrounded tools) and
        :meth:`_promote_to_bg` (cohort-decay promotions). Pops the
        ``background`` entry first; if already gone, the
        ``BackgroundTask.foreground`` op already consumed the task and
        we skip the inbox send to avoid duplication.
        """
        if self.background.pop(qid, None) is None:
            return
        source = f"bg_{qid}"
        if task.cancelled():
            text = f"[Background tool cancelled: {tool_name} ({qid})]"
        elif (exc := task.exception()) is not None:
            text = (
                f"[Background tool failed: {tool_name} ({qid})]\n\n"
                f"{type(exc).__name__}: {exc}"
            )
        else:
            result = task.result()
            framing = "failed" if has_error(result) else "completed"
            text = (
                f"[Background tool {framing}: {tool_name} ({qid})]"
                f"\n\n{text_from_msg(result)}"
            )
        self.inbox.send(TextMessage(text, "text/x-user-message"), source=source)

    async def _invoke_tool_safe(self, call: Message) -> Message:
        """Run one tool call; convert any exception to a structured error result.

        Streaming tools yield intermediate events onto a queue; a sibling
        drainer task publishes them as ``TextChunkEvent`` in real time so
        the REPL sees live progress, not a batch at tool completion.
        """
        events: asyncio.Queue[Message | None] = asyncio.Queue()
        drainer = asyncio.create_task(self._drain_tool_events(events))
        try:
            try:
                result = await invoke_tool(self.tools_map, call, events)
            except Exception as e:  # noqa: BLE001 -- dispatch safety net
                logger.debug("Tool %s raised", get_tool_name(call), exc_info=True)
                result = MultipartMessage(
                    (
                        TextMessage(get_queue_id(call), "text/x-queue-id"),
                        TextMessage(f"{type(e).__name__}: {e}", "text/x-error"),
                    ),
                    "multipart/x-tool-result",
                    parent_id=call.id,
                )
            return result
        finally:
            events.put_nowait(None)  # sentinel: drainer exits cleanly
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(drainer)

    async def _drain_tool_events(
        self,
        events: asyncio.Queue[Message | None],
    ) -> None:
        """Forward intermediate streaming-tool yields as ``TextChunkEvent``."""
        while True:
            event = await events.get()
            if event is None:
                return
            if isinstance(event, TextMessage) and event.descriptor == "text/plain":
                self.publish(TextChunkEvent(event.content))

    def _postprocess_results(
        self,
        calls: list[Message],
        results: list[Message],
    ) -> list[Message]:
        """Inject empty-output marker, persist oversized, append conditional rules."""
        seen_rules: set[Path] = set()
        out = list(results)
        # Build a qid->call map so we tolerate cohort results arriving
        # in a different order from ``calls`` (e.g. a Cancelled member
        # that emitted instantly).
        by_qid = {get_queue_id(c): c for c in calls}
        for i, r in enumerate(results):
            qid = get_queue_id(r)
            call = by_qid.get(qid)
            if call is None:
                continue
            text = text_from_msg(r)
            content = inject_empty_marker(get_tool_name(call), text)
            r_parts = cast("tuple[Message, ...]", r.content)
            if not has_error(r):
                preview = persist_result(
                    qid,
                    tc_tool_id(call),
                    content,
                    self.replacement_state,
                )
                if preview is not None:
                    content = preview
                reminder = conditional_rules_for_request(
                    call,
                    self.tool_state,
                    seen_rules,
                )
                if reminder:
                    content = content.rstrip() + "\n\n" + reminder
            if content != text:
                non_text = tuple(
                    p
                    for p in r_parts
                    if p.descriptor not in ("text/plain", "text/x-error")
                )
                out[i] = dataclasses.replace(
                    r,
                    content=(TextMessage(content, "text/plain"), *non_text),
                )
        return out

    def _spawn_bg(self, call: Message) -> Message:
        """Spawn a background worker for ``call``; return the inbox placeholder.

        The task runs :meth:`_invoke_tool_safe` (optionally preceded by a
        sleep). Completion routes through :meth:`_post_bg_completion`, the
        same done-callback used by cohort-decay promotions — one inbox-send
        path, one bg-cleanup path.
        """
        qid = get_queue_id(call)
        tool_name = get_tool_name(call)
        delay = int_val(tc_directive(call).get("delay"), 0)
        if delay > 0:

            async def _delayed() -> Message:
                await asyncio.sleep(delay)
                return await self._invoke_tool_safe(call)

            task = asyncio.create_task(_delayed())
        else:
            task = asyncio.create_task(self._invoke_tool_safe(call))
        self.background[qid] = BackgroundTaskEntry(
            task=task,
            tool_name=tool_name,
            queue_id=qid,
            started=time.time(),
            delay_sec=delay,
            kind="tool",
        )
        task.add_done_callback(
            functools.partial(self._post_bg_completion, qid, tool_name),
        )
        return MultipartMessage(
            (
                TextMessage(qid, "text/x-queue-id"),
                TextMessage(
                    f"[Running in background: {tool_name}]",
                    "text/plain",
                ),
            ),
            "multipart/x-tool-result",
            parent_id=call.id,
        )

    # -- Cost accounting ----------------------------------------------

    def _record_response(self, response: ModelResponse) -> None:
        """Write one response through to the active root tracker.

        Per-response writes flow to ``cost_root_var.get(self.cost_tracker)``:
        sub-agents inherit the root agent's tracker via ContextVar copy
        and write through; the root agent (or a persistent sub-agent
        that shadowed the var) writes to its own tracker. Single store,
        always live -- no separate ledger, no fold step.

        Raises:
          RuntimeError: When this agent's ``max_budget_usd`` cap is hit.
              The cap is checked against the destination tracker, so any
              agent in the chain can independently raise.

        """
        target = cost_root_var.get(None) or self.cost_tracker
        target.record(response, model_id=self.model.model_id)
        if (
            self._max_budget_usd is not None
            and target.total_cost_usd >= self._max_budget_usd
        ):
            raise RuntimeError(
                f"Budget exhausted: ${target.total_cost_usd:.2f}"
                f" >= ${self._max_budget_usd:.2f}",
            )

    def _open_cost_lifecycle(self) -> _CostState:
        """Install ``self.cost_tracker`` as the subtree's cost target.

        Sync sub-agents (running in their parent's task) inherit the
        parent's tracker via ContextVar copy; their open is a no-op.
        Root agents and persistent sub-agents (which run in their own
        task and want independent accounting) shadow the var with their
        own tracker. Returns an opaque token consumed by
        :meth:`_close_cost_lifecycle`; ``token=None`` means we did not
        open and close should no-op.
        """
        if not self._persistent and cost_root_var.get(None) is not None:
            return _CostState(token=None)
        self._run_start_tokens = self.cost_tracker.total
        self._run_start_cost_usd = self.cost_tracker.total_cost_usd
        return _CostState(token=cost_root_var.set(self.cost_tracker))

    def _close_cost_lifecycle(self, state: _CostState) -> None:
        """Capture the run's delta into ``last_run_*`` and reset the ContextVar."""
        if state.token is None:
            return
        if self._run_start_tokens is not None:
            self._last_run_tokens = self.cost_tracker.total - self._run_start_tokens
            self._last_run_cost_usd = (
                self.cost_tracker.total_cost_usd - self._run_start_cost_usd
            )
            self._run_start_tokens = None
            self._run_start_cost_usd = 0.0
        with contextlib.suppress(ValueError):
            cost_root_var.reset(state.token)

    def _account_cancelled(self) -> None:
        """Best-effort cost record when cancel arrived outside a model call."""
        with contextlib.suppress(RuntimeError):
            self._record_response(
                ModelResponse(
                    content=TextMessage("", "text/plain"),
                    tokens=TokenCount(),
                ),
            )

    def _account_cancelled_partial(
        self,
        request: ModelRequest,
        chars_streamed: int,
    ) -> None:
        """Record an estimated cost for a cancelled in-flight model call."""
        estimated_input = max(
            self.cost_tracker.last_request.input_tokens,
            estimate_total_tokens(request.system or "", request.messages, self.model),
        )
        estimated_output = (
            max(1, chars_streamed // self.budget.chars_per_token)
            if chars_streamed > 0
            else 0
        )
        pricing = self.model.pricing
        estimated_cost = (
            estimated_input * pricing.request + estimated_output * pricing.response
        ) / 1_000_000
        partial = ModelResponse(
            content=TextMessage("", "text/plain"),
            tokens=TokenCount(
                input_tokens=estimated_input,
                output_tokens=estimated_output,
            ),
            total_cost=estimated_cost,
            input_cost=estimated_input * pricing.request / 1_000_000,
            output_cost=estimated_output * pricing.response / 1_000_000,
        )
        with contextlib.suppress(RuntimeError):
            self._record_response(partial)

    # -- Compaction ---------------------------------------------------

    async def _maybe_compact(self) -> None:
        """Microcompact + threshold-driven full compaction. Caller holds foreground."""
        if self.compactor is None:
            return
        self.compactor.maintain(
            self.history,
            self.tools_map,
            read_cache=self.tool_state.read_cache,
            last_response_time=self.cost_tracker.last_response_time,
        )
        system = self.system_prompt()
        input_tokens = max(
            self.cost_tracker.last_request.input_tokens,
            estimate_total_tokens(system, self.history, self.model),
        )
        if await self.compactor.should_compact(
            input_tokens=input_tokens,
            max_request_tokens=self.max_request_tokens,
            max_response_tokens=self.max_response_tokens,
        ):
            _ = await self._do_compact("")
            return
        if input_tokens > (
            self.max_request_tokens
            - self.max_response_tokens
            - self.budget.buffer_tokens
        ):
            _ = await self._do_compact("")

    async def _do_compact(
        self,
        args: str = "",
        *,
        count_failures: bool = True,
        write_transcript: bool = True,
    ) -> bool:
        """Run one compaction round; return True on success."""
        if self.compactor is None:
            return False
        if self.compaction_state.compacting:
            return False
        if self.compaction_state.compact_failures >= MAX_COMPACT_FAILURES:
            return False
        self.compaction_state.compacting = True
        try:
            transcript_path = None
            if self.session_dir is not None:
                transcript_path = (
                    self.session_dir
                    / f"pre_compact_{self.compaction_state.compact_count}.jsonl"
                )
                if write_transcript:
                    write_pre_compact_transcript(transcript_path, self.history)
            prior_pointers = list(self.compaction_state.summary_pointers)
            result = await self.compactor.compact(
                messages=self.history,
                model=self.model,
                transcript_path=transcript_path,
                custom_instructions=args or None,
                summary_pointers=prior_pointers or None,
            )
            self.history.clear()
            self.history.extend(result)
        except Exception as e:  # noqa: BLE001 -- compactor errors are heterogeneous
            if count_failures:
                self.compaction_state.compact_failures += 1
            self.publish(
                ErrorEvent(
                    text=(
                        f"compaction failed "
                        f"({self.compaction_state.compact_failures}/"
                        f"{MAX_COMPACT_FAILURES}): "
                        f"{type(e).__name__}: {e}"
                    ),
                ),
            )
            logger.debug("compaction failed", exc_info=True)
            return False
        finally:
            self.compaction_state.compacting = False

        used = estimate_total_tokens(self.system_prompt(), self.history, self.model)
        await post_compact_enrich(
            result=result,
            messages=self.history,
            state=self.compaction_state,
            session_dir=self.session_dir,
            tool_state=self.tool_state,
            budget=self.budget,
            tools=self.tools_map,
            background_tasks=self.background,
            estimate_tokens=self.max_request_tokens - used,
            headroom=self.max_response_tokens + self.budget.buffer_tokens,
        )
        self.compaction_state.compact_count += 1
        self.cost_tracker.last_request = TokenCount()
        self.save_session(clear=True)
        return True

    async def _do_recompact(self, args: str = "") -> None:
        """Reload pre-compact transcript and re-run compaction with rollback."""
        if self.compactor is None or self.session_dir is None:
            return
        count = self.compaction_state.compact_count
        if count == 0:
            return
        path = self.session_dir / f"pre_compact_{count - 1}.jsonl"
        if not path.exists():
            return
        try:
            raw_text = path.read_text(encoding="utf-8")
            loaded = [load_message(rec) for rec in parse_jsonl(raw_text)]
        except (OSError, KeyError, AssertionError, TypeError) as e:
            self.publish(
                ErrorEvent(
                    text=f"recompact transcript load failed: {type(e).__name__}: {e}",
                ),
            )
            logger.debug("recompact transcript load failed", exc_info=True)
            return
        if not loaded:
            return
        saved = list(self.history)
        self.history.clear()
        self.history.extend(loaded)
        rollback = True
        try:
            ok = await self._do_compact(
                args,
                count_failures=False,
                write_transcript=False,
            )
            rollback = not ok
        finally:
            if rollback:
                self.history.clear()
                self.history.extend(saved)

    async def _do_clear_async(self) -> None:
        """Async wrapper so ``clear`` matches the public-strategy shape."""
        self._do_clear_sync()

    def _do_clear_sync(self) -> None:
        """Synchronous clear; assumes the caller already holds foreground."""
        self.history.clear()
        self.cost_tracker.last_request = TokenCount()
        self.activity.num_tool_call_rounds = 0
        self.tool_state.reset_file_tracking()
        self.save_session(clear=True)

    # -- Session persistence ------------------------------------------

    def _build_session_meta(self) -> SessionMeta:
        """Snapshot agent state into a serializable ``SessionMeta``."""
        spec = self.model_spec
        return SessionMeta(
            session_id=self.session_id,
            model_id=self.model.model_id,
            provider=spec.provider if spec else "",
            auth=spec.auth if spec else "",
            account=spec.account or "" if spec else "",
            name=self.name,
            status=self.status,
            tokens=type(self.cost_tracker.total)(
                input_tokens=self.cost_tracker.total.input_tokens,
                output_tokens=self.cost_tracker.total.output_tokens,
                cache_creation_tokens=self.cost_tracker.total.cache_creation_tokens,
                cache_read_tokens=self.cost_tracker.total.cache_read_tokens,
            ),
            total_cost_usd=self.cost_tracker.total_cost_usd,
            num_tool_call_rounds=self.activity.num_tool_call_rounds,
            compact_count=self.compaction_state.compact_count,
            summary_pointers=self.compaction_state.summary_pointers,
            bash_cwd=self.tool_state.bash_cwd,
            total_active_elapsed_seconds=self.activity.elapsed_seconds,
        )

    def _load_session(self) -> None:
        """Restore meta + history from ``session_dir/session.jsonl``."""
        if self.session_dir is None:
            return
        result = load_session(
            self.session_dir,
            defaults={
                "session_id": self.session_id,
                "model_id": self.model.model_id,
                "name": self.name,
                "bash_cwd": self.tool_state.bash_cwd,
            },
        )
        if result is None:
            return
        meta, messages = result
        if messages:
            reset_id_counter(max(m.id for m in messages) + 1)
        repaired = repair_dangling_tool_calls(messages)
        repair_inserted = len(repaired) > len(messages)
        self.history = repaired
        self._persisted_idx = len(messages)
        rebuild_tool_state_from_messages(self.history, self.tool_state)
        if meta:
            m = SessionMeta.deserialize(meta)
            self._session_id = m.session_id or self.session_id
            self._status = m.status
            self.cost_tracker.restore(
                total_cost_usd=m.total_cost_usd,
                total=type(self.cost_tracker.total)(
                    input_tokens=m.tokens.input_tokens,
                    output_tokens=m.tokens.output_tokens,
                    cache_creation_tokens=m.tokens.cache_creation_tokens,
                    cache_read_tokens=m.tokens.cache_read_tokens,
                ),
            )
            self.activity.num_tool_call_rounds = m.num_tool_call_rounds
            self.activity.elapsed_seconds = m.total_active_elapsed_seconds
            self.compaction_state.compact_count = m.compact_count
            self.compaction_state.summary_pointers = list(m.summary_pointers)
            if m.bash_cwd:
                self.tool_state.bash_cwd = m.bash_cwd
            if m.provider and m.model_id:
                restored = restore_model(m)
                if restored is not None:
                    self.model, self.model_spec = restored
        self.cost_tracker.last_response_time = time.time()
        logger.info(
            "Resumed session %s (%d messages)",
            self.session_id,
            len(self.history),
        )
        if repair_inserted:
            self.save_session(clear=True)


# -- Module helpers --------------------------------------------------------


_SENTINEL_DONE: object = object()


def _validate_budget(budget: ContextBudget, model: Model) -> None:
    """Raise if ``budget`` exceeds the model's context limits."""
    if budget.max_request_tokens > model.max_request_tokens:
        raise ValueError(
            f"budget.max_request_tokens={budget.max_request_tokens:,}"
            f" exceeds model's {model.max_request_tokens:,}",
        )
    if budget.max_response_tokens > model.max_response_tokens:
        raise ValueError(
            f"budget.max_response_tokens={budget.max_response_tokens:,}"
            f" exceeds model's {model.max_response_tokens:,}",
        )


def _user_text(msg: Message) -> str:
    """Extract a renderable user-message body from ``msg``."""
    if isinstance(msg, TextMessage):
        return msg.content
    if isinstance(msg, MultipartMessage):
        for part in msg.content:
            if isinstance(part, TextMessage) and part.descriptor == "text/plain":
                return part.content
    return ""


def _format_user_bundle(
    user_pieces: list[str],
    peer_pieces: list[tuple[str, str]],
    bg_pieces: list[str],
) -> str | None:
    """Build the merged user-message body per spec §3.

    Single-item user-only: verbatim. Otherwise each part tagged
    ``[user]`` / ``[from <peer>]``; bg pieces append verbatim
    (already framed with ``[Background tool ...]``).
    """
    total = len(user_pieces) + len(peer_pieces) + len(bg_pieces)
    if total == 0:
        return None
    if total == 1 and user_pieces:
        return user_pieces[0]
    parts: list[str] = [f"[user] {t}" for t in user_pieces]
    parts.extend(f"[from {src}] {t}" for src, t in peer_pieces)
    parts.extend(bg_pieces)
    return "\n\n".join(parts)


_FILE_OP_TOOL_IDS: frozenset[str] = frozenset(
    {
        "application/x-tool-read",
        "application/x-tool-edit",
        "application/x-tool-write",
    },
)


def _is_file_op(call: Message) -> bool:
    """Return True iff ``call`` targets a Read / Edit / Write tool."""
    return tc_tool_id(call) in _FILE_OP_TOOL_IDS


def _text_parts(response_msg: Message) -> list[str]:
    """Extract ``text/plain`` content from a ``multipart/x-model-message``."""
    if not isinstance(response_msg, MultipartMessage):
        return []
    return [
        str(part.content)
        for part in response_msg.content
        if part.descriptor == "text/plain"
    ]


def _thinking_parts(response_msg: Message) -> list[str]:
    """Extract thinking text from each thinking part in ``response_msg``."""
    if not isinstance(response_msg, MultipartMessage):
        return []
    out: list[str] = []
    for part in response_msg.content:
        if part.descriptor == "text/x-thinking" and isinstance(part, TextMessage):
            if part.content:
                out.append(part.content)
        elif part.descriptor == "application/x-thinking-structured":
            content = cast(Mapping[str, object], part.content)
            text = str(content.get("thinking", ""))
            if text:
                out.append(text)
    return out


def wrap_errors_for_llm(tr: Message) -> Message:
    """Wrap ``text/x-error`` parts in ``<tool_use_error>`` for the LLM transcript."""
    if tr.descriptor != "multipart/x-tool-result":
        return tr
    parts = cast("tuple[Message, ...]", tr.content)
    new_parts: list[Message] = []
    changed = False
    for p in parts:
        if p.descriptor == "text/x-error":
            new_parts.append(
                dataclasses.replace(
                    p,
                    content=f"<tool_use_error>{p.content}</tool_use_error>",
                ),
            )
            changed = True
        else:
            new_parts.append(p)
    if not changed:
        return tr
    return dataclasses.replace(tr, content=tuple(new_parts))


def _is_thinking_unsupported(exc: BaseException) -> bool:
    """True if ``exc`` is an Anthropic 400 saying thinking isn't supported."""
    msg = str(exc).lower()
    return "thinking is not supported" in msg or (
        "thinking" in msg and "not supported" in msg
    )
