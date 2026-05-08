"""Agent v2: deque + handler registry + dispatch loop.

The agent IS its deque. Empty deque + no in-flight tasks = idle.
Messages drive everything: user input, model calls, tool batches,
streaming chunks, abort, quit. Handlers register against descriptors
and post follow-up messages to advance the conversation.

See ``docs/private/inbox_refactor.md`` for the full design.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from pathlib import Path
from typing import cast

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import time
import uuid

from sagent.agent.compaction import CompactionState
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.handlers import Handler, core_handlers
from sagent.agent.handlers.activity import ActivityTracker
from sagent.agent.retry import (
    RateLimitError,
    RetriesExhaustedError,
)
from sagent.agent.session_io import (
    SessionMeta,
    append_session,
    load_session,
    rebuild_tool_state_from_messages,
    repair_dangling_tool_calls,
    restore_model,
)
from sagent.custom_types import (
    Compactor,
    ContextBudget,
    JsonMessage,
    Message,
    Model,
    ModelSpec,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
    reset_id_counter,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.descriptors import QUIT_SENTINEL
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import response_text
from sagent.tools.background_task import BackgroundTaskEntry
from sagent.tools.core import (
    CostLedger,
    ToolState,
    agent_counter_var,
    agent_label_var,
    agent_registry,
    changed_files_context,
    cost_ledger_var,
    current_agent_var,
    tool_state_context,
    tool_state_var,
)
from sagent.tools.result_storage import ReplacementState


logger = logging.getLogger(__name__)


SystemPrompt = str | dict[str, str | Callable[[], str]]
SystemPromptArg = SystemPrompt | Callable[[], SystemPrompt]


_MAX_UNSAVED_EVENTS = 1000

# Hard cap on shutdown-drain wait. Cooperative tasks cancel promptly;
# uncancellable work (sync tools in ``asyncio.to_thread``) cannot be
# stopped from the event loop, so the runner falls through after this.
_DRAIN_TIMEOUT = 2.0

# Error code returned in the final assistant message when the agent's
# tool-call-round cap was reached. Callers (AgentSpawn, orchestrator)
# match on this string in the message content.
ERROR_MAX_TOOL_CALL_ROUNDS = "error:max_tool_call_rounds"


class Agent:
    """Conversation agent: deque + handler registry + dispatch loop.

    Public surface mirrors v1 ``Agent`` so the ``Tool``, ``AgentSpawn``,
    ``AgentSelf``, and ``Advisor`` consumers use the same API. The
    runtime model is different (deque + handlers, not a send loop),
    but the contract is preserved.
    """

    # Tool-shaped attribute (not full Tool conformance -- see
    # Addendum A in docs/private/inbox_refactor.md). Agent results
    # are the final summary of a subagent's work; dropping one on
    # cache-cold would hide the whole subtask's output. Always keep
    # the last message.
    supports_microcompaction: bool = False

    def __init__(
        self,
        *,
        model: Model,
        model_spec: ModelSpec | None = None,
        system: SystemPromptArg = "",
        tools: list[Tool] | None = None,
        handlers: list[Handler] | None = None,
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
        # Normalize the ``system`` arg to a callable factory so
        # ``system_prompt`` has a single code path. Static inputs
        # (str / dict) are wrapped in a constant lambda; callable
        # inputs are used as-is. The cast is needed because ty's
        # ``callable(...)`` narrowing does not exclude str/dict from
        # the union -- a known limitation, see the ``Top[(...) -> object]``
        # diagnostic.
        self._system_factory: Callable[[], SystemPrompt] = cast(
            "Callable[[], SystemPrompt]",
            system if callable(system) else (lambda s=system: s),
        )
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            if t.tool_id in self._tools:
                raise ValueError(f"Duplicate tool: {t.tool_id!r}")
            self._tools[t.tool_id] = t
        self.compactor = compactor
        if budget is not None:
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
        self.cost_tracker = CostTracker(max_budget_usd=max_budget_usd)
        self.inbox: Deque[Message] = Deque()
        self.history: list[Message] = []
        self.tool_state = ToolState()
        self.activity = ActivityTracker()
        # Model calls hold this lock for the full duration of the
        # provider request. Multiple ``text/x-model-call`` messages
        # arriving back-to-back (e.g. user types while a call is in
        # flight) serialize via the lock instead of issuing concurrent
        # provider requests.
        self.model_lock: asyncio.Lock = asyncio.Lock()
        self.compaction_state = CompactionState()
        self.replacement_state = ReplacementState(
            persist_threshold=self.budget.persist_threshold,
            message_budget=self.budget.message_budget_chars,
        )
        self.background_tasks: dict[str, BackgroundTaskEntry] = {}
        # ``AgentSpawn`` populates this with per-child ``ChildStats`` so
        # the parent's toolbar can render running-child summaries.
        self.active_children: dict[str, object] = {}
        self._handlers: dict[str, list[Handler]] = {}
        self._wildcards: list[Handler] = []
        self.tasks: dict[int, asyncio.Task[None]] = {}
        self._session_id = str(uuid.uuid4())[:8]
        self._status: str = ""
        self._event_log: list[dict[str, object]] = []
        # Cursor: index up to which ``self.history`` has been persisted
        # to ``session.jsonl``. Saves append ``self.history[idx:]``;
        # wholesale-replacement paths (``/clear``, compact, repair on
        # load) call ``_save_session(clear=True)`` to write a barrier
        # marker and re-persist the current history fresh.
        self._persisted_idx: int = 0
        # Spawn-tree cost tracking (parity with v1).
        self._tracker_snap_cost_usd: float = 0.0
        self._tracker_snap_tokens: TokenCount = TokenCount()
        self._active_cost_ledger: CostLedger | None = None
        # Snapshot of the most recent completed run's totals.
        self._last_run_tokens: TokenCount = TokenCount()
        self._last_run_cost_usd: float = 0.0
        self._persistent: bool = False
        self.session_dir: Path | None = None
        if session_dir is not None:
            self.session_dir = Path(session_dir)
            self.replacement_state.storage_dir = self.session_dir
            self._load_session()
        default = core_handlers()
        for h in handlers if handlers is not None else default:
            self.register(h)

    # -- Properties / config surface ----------------------------------

    @property
    def budget(self) -> ContextBudget:
        """Context budget; auto-derived from model when unset."""
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

    @property
    def thinking(self) -> str | None:
        """Thinking mode for this agent (e.g. ``"adaptive"``). ``None`` disables."""
        return self._thinking

    def set_thinking(self, value: str | None) -> None:
        """Update the thinking mode for subsequent model calls.

        Used by ``ModelCallHandler`` when the API rejects the current
        mode (e.g. Haiku 4.5 doesn't accept ``"adaptive"``) so the
        rest of the session degrades gracefully instead of looping
        on the same 400.
        """
        self._thinking = value

    @property
    def effort(self) -> str | None:
        """Effort level for this agent (provider-specific; may be ``None``)."""
        return self._effort

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
        """Session status for terminal titlebar (empty until set)."""
        return self._status

    def set_status(self, status: str) -> None:
        """Set the session status, persist, and post a status-update message.

        Args:
          status: New status string for the terminal titlebar.

        """
        self._status = status
        self._save_session()
        self.inbox.put(TextMessage(status, "text/x-status-update"))

    @property
    def messages(self) -> list[Message]:
        """Alias for ``history`` (v1 compatibility)."""
        return self.history

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self.history = value

    @property
    def tools(self) -> list[Tool]:
        """Tools available to this agent (order preserved).

        Returns a list (v1 contract) so consumers like ``AgentSpawn``
        and ``advisor`` can iterate without dict-vs-list contortions.
        Handlers needing dict semantics use ``agent._tools`` directly.
        """
        return list(self._tools.values())

    @property
    def system(self) -> SystemPrompt:
        """The system prompt this agent was constructed with.

        Returns the unwrapped value: a string for static prompts or
        the dict for sectioned prompts. Callable system arguments
        evaluate at access (rare; mostly used by the CLI to rebuild
        per-model-id prompts on ``/model`` swaps).
        """
        return self._system_factory()

    @property
    def cost_ledger(self) -> CostLedger | None:
        """The active subtree-wide ledger, or ``None`` outside a call."""
        return cost_ledger_var.get()

    @property
    def active(self) -> bool:
        """True while a model call or tool batch is in flight."""
        return bool(self.tasks) or self.activity.active

    @property
    def total_cost_usd(self) -> float:
        """Cumulative subtree USD cost for this session."""
        if self.activity.active and self._active_cost_ledger is not None:
            return self._tracker_snap_cost_usd + self._active_cost_ledger.total_cost_usd
        return self.cost_tracker.total_cost_usd

    @property
    def total_tokens(self) -> TokenCount:
        """Cumulative subtree token counts across the session."""
        if self.activity.active and self._active_cost_ledger is not None:
            return self._tracker_snap_tokens + self._active_cost_ledger.tokens
        return self.cost_tracker.total

    @property
    def last_run_tokens(self) -> TokenCount:
        """Token counts for the active root run, including descendants.

        While a root run is active, returns the running ledger total
        (matches v1). Outside a run, returns the most recent ledger
        snapshot.
        """
        if self._active_cost_ledger is not None:
            return self._active_cost_ledger.tokens
        return self._last_run_tokens

    @property
    def last_run_cost_usd(self) -> float:
        """Cost for the active root run, including descendants."""
        if self._active_cost_ledger is not None:
            return self._active_cost_ledger.total_cost_usd
        return self._last_run_cost_usd

    @property
    def total_active_elapsed_seconds(self) -> float:
        """Cumulative wall-clock seconds spent active across the session.

        Live during an active model call: sums historical elapsed with
        ``(now - current_call_start)`` so the toolbar ticks in real time.
        """
        if self.activity.active and self.activity.current_call_start > 0:
            return self.activity.elapsed_seconds + (
                asyncio.get_running_loop().time() - self.activity.current_call_start
            )
        return self.activity.elapsed_seconds

    @property
    def request_start_time(self) -> float:
        """Event-loop timestamp when the current model call started.

        Returns the start of the in-flight call, or ``0.0`` between calls.
        """
        return self.activity.current_call_start

    @property
    def live_model_response_tokens(self) -> int:
        """Live output-token estimate (chars / chars_per_token) for the current call."""
        return self.activity.live_response_chars // self.budget.chars_per_token

    @property
    def num_tool_call_rounds(self) -> int:
        """Cumulative tool-call rounds for this session."""
        return self.activity.num_tool_call_rounds

    # -- Tool contract -------------------------------------------------

    def summary(self, msg: Message) -> str:
        """Return a help description for this agent.

        Args:
          msg: The requesting message (unused).

        Returns:
          description: The agent name.

        """
        del msg
        return self.name

    def prompt(self) -> str:
        """Return per-request system-prompt contribution.

        Returns:
          text: Empty string (no contribution).

        """
        return ""

    def system_prompt(self) -> str:
        """Assemble the system prompt from sections + tool contributions.

        ``self._system_factory`` is always a callable -- the
        constructor wraps static inputs. The factory is re-evaluated
        on every call so post-``/model`` swaps regenerate sections
        that depend on the current ``model.model_id``.
        """
        system = self._system_factory()
        parts: list[str]
        if isinstance(system, str):
            parts = [system] if system else []
        else:
            parts = []
            for value in system.values():
                content = value if isinstance(value, str) else value()
                if content:
                    parts.append(content)
        if self._track_changed_files:
            diff = changed_files_context()
            if diff:
                parts.append(diff)
        for tool in self._tools.values():
            section = tool.prompt()
            if section:
                parts.append(section)
        return "\n\n".join(parts)

    def swap_model(self, model: Model, *, spec: ModelSpec | None = None) -> None:
        """Replace the active model and (optionally) its spec.

        Args:
          model: New model instance.
          spec: Recipe that produced ``model``; ``None`` clears it.

        """
        self.model = model
        self.model_spec = spec

    def register(self, handler: Handler) -> None:
        """Add a handler to the registry.

        Args:
          handler: Handler instance to register. ``descriptors=()``
              registers it as a wildcard (fires after specific handlers).

        """
        if not handler.descriptors:
            self._wildcards.append(handler)
            return
        for descriptor in handler.descriptors:
            self._handlers.setdefault(descriptor, []).append(handler)

    # -- Dispatch loop -------------------------------------------------

    async def run_loop(self) -> None:
        """Run the dispatch loop until ``text/x-quit``.

        Sets up the per-agent ContextVars (``current_agent_var``,
        ``cost_ledger_var``, ``agent_label_var``, ``agent_registry``
        entry, ``tool_state_context``) so tools dispatched via the
        loop see the correct state.
        """
        agent_token = current_agent_var.set(self)
        # The root run owns the UI snapshot; children inherit this
        # ContextVar ledger.
        ledger = cost_ledger_var.get()
        ledger_token = None
        if ledger is None:
            ledger = CostLedger()
            self._active_cost_ledger = ledger
            ledger_token = cost_ledger_var.set(ledger)
            self._tracker_snap_cost_usd = self.cost_tracker.total_cost_usd
            self._tracker_snap_tokens = self.cost_tracker.total
        counter_token = agent_counter_var.set(itertools.count())
        label = agent_label_var.get("") or self.name
        label_token = agent_label_var.set(label)
        agent_registry[label] = self
        # Set the agent's tree depth so AgentSpawn's ``max_depth``
        # guard knows how deep this agent sits (parent + 1, or 0 for
        # a root run).
        parent_state = tool_state_var.get(None)
        self.tool_state.depth = 0 if parent_state is None else parent_state.depth + 1
        # Reset the abort flag so a previously-cancelled session
        # doesn't poison subsequent runs.
        self.tool_state.abort_event.clear()
        try:
            with tool_state_context(self.tool_state):
                await self._dispatch_loop()
        finally:
            # Fold the active ledger into the persistent tracker so
            # cumulative totals survive across run_loop boundaries.
            if self._active_cost_ledger is not None:
                self._last_run_tokens = self._active_cost_ledger.tokens
                self._last_run_cost_usd = self._active_cost_ledger.total_cost_usd
                self.cost_tracker.fold(
                    snapshot_cost_usd=self._tracker_snap_cost_usd,
                    snapshot_tokens=self._tracker_snap_tokens,
                    run_ledger=self._active_cost_ledger,
                )
                self._active_cost_ledger = None
            if not self._persistent:
                agent_registry.pop(label, None)
            agent_label_var.reset(label_token)
            agent_counter_var.reset(counter_token)
            current_agent_var.reset(agent_token)
            if ledger_token is not None:
                cost_ledger_var.reset(ledger_token)

    # ``run`` is a higher-level interface than the Tool protocol --
    # it returns a streaming + awaitable handle, not a flat Message.
    # AgentSpawn (in ``tools/agent_spawn.py``) is the actual ``Tool``
    # that wraps an Agent and exposes a Tool-conforming ``run(msg) ->
    # Message``. ``run_loop`` is the bare dispatch loop the REPL
    # drives directly. ``run_forever`` is the external-prompt-feed
    # loop used by Slack adapters and persistent subagents.
    def run(self, directive: JSON) -> RunHandle:
        """Run the agent on a prompt, returning a streaming + awaitable handle.

        Mirrors v1's ``Agent.run``. Streaming callers iterate events;
        batch callers ``await`` the handle for the final message.

        Args:
          directive: JSON mapping with at least a ``"prompt"`` key.

        Returns:
          handle: Awaitable and async-iterable run handle.

        """
        prompt = str(directive.get("prompt", ""))
        if not prompt:
            raise ValueError("No prompt provided.")
        events: asyncio.Queue[Message | None] = asyncio.Queue()
        self.inbox.put(TextMessage(prompt, "text/x-user-message"))
        cost_before = _subtree_cost(self)

        async def _drive() -> Message:
            # Forward inbox traffic onto ``events`` via a wildcard
            # handler so callers iterating the handle see streaming
            # chunks, tool labels, tool results, and errors. Posted
            # by ``_DriveForwarder`` registered for the duration of
            # this call only.
            forwarder = _DriveForwarder(events)
            self.register(forwarder)
            try:
                loop_task = asyncio.create_task(self.run_loop())
                try:
                    await self._wait_for_idle()
                finally:
                    self.inbox.put(TextMessage("", QUIT_SENTINEL))
                    await loop_task
                return _last_assistant_message(self.history)
            finally:
                self._unregister_wildcard(forwarder)
                # Mirror v1's ``application/x-done`` request-boundary
                # sentinel: emit before terminating so callers like
                # ``AgentSpawn._ChildForwarder`` can build a per-call
                # summary from the per-call token totals + cost.
                cost_delta = max(0.0, _subtree_cost(self) - cost_before)
                events.put_nowait(
                    JsonMessage(
                        json_freeze(
                            {
                                "input_tokens": (
                                    self.cost_tracker.last_request.input_tokens
                                ),
                                "output_tokens": (self.cost_tracker.call_output_tokens),
                                "cost_usd": cost_delta,
                            },
                        ),
                        "application/x-done",
                    ),
                )
                events.put_nowait(None)

        task = asyncio.create_task(_drive())
        return RunHandle(self, task, events)

    async def run_forever(self) -> AsyncGenerator[Message | None, None]:
        """Drive the agent forever, yielding display events.

        Mirrors v1's ``run_forever`` contract: drains inbox into
        coalesced prompts, calls ``run`` per prompt, yields the
        streamed events and a ``None`` boundary marker after each.
        Returns when ``QUIT_SENTINEL`` arrives.

        Used by Slack adapters and multi-agent orchestrators that
        feed prompts via ``inbox.put`` and consume events via
        ``async for``.

        Yields:
          event: Streaming event, or ``None`` at request boundaries.

        """
        while True:
            items = await self.inbox.get_all()
            if any(m.descriptor == QUIT_SENTINEL for m in items):
                self._save_session()
                return
            prompts: list[str] = []
            for m in items:
                if m.descriptor == "text/x-clear-request":
                    # Re-enqueue so ClearHandler runs on the next loop tick.
                    self.inbox.put_left(m)
                else:
                    prompts.append(str(m.content))
            if not prompts:
                continue
            prompt = "\n\n".join(prompts)
            handle = self.run(json_freeze({"prompt": prompt}))
            try:
                async for event in handle:
                    yield event
                _ = await handle
            except asyncio.CancelledError:
                pass
            except (RateLimitError, RetriesExhaustedError) as e:
                logger.warning("run_forever: %s", e)
                yield TextMessage(str(e), "text/x-error")
            except Exception as e:  # noqa: BLE001 -- run_forever safety net
                logger.debug("run_forever: run raised", exc_info=True)
                yield TextMessage(f"⚠ {type(e).__name__}: {e}", "text/x-error")
            yield None

    async def _dispatch_loop(self) -> None:
        """The bare dispatch loop, no context setup."""
        while True:
            msg = await self.inbox.get()
            if msg.descriptor == QUIT_SENTINEL:
                await self._drain_tasks()
                return
            for h in self._handlers.get(msg.descriptor, ()):
                await self._invoke(h, msg)
            for h in self._wildcards:
                await self._invoke(h, msg)

    async def _wait_for_idle(self) -> None:
        """Block until the inbox is empty and no spawned tasks are in flight."""
        while True:
            await self.inbox.wait_for_empty()
            if not self.tasks:
                return
            _ = await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def _invoke(self, handler: Handler, msg: Message) -> None:
        """Run one handler against one message, dispatching by spawn mode."""
        if handler.spawn:
            task = asyncio.create_task(self._wrap_spawn(handler, msg))
            # Key by ``id(task)`` rather than ``msg.id`` so multiple
            # spawned handlers fanned out from the same descriptor each
            # get their own slot. Otherwise the second create_task
            # clobbers the first and ``_wait_for_idle`` returns early.
            self.tasks[id(task)] = task
            return
        try:
            await handler.handle(self, msg)
        except Exception as e:  # noqa: BLE001 -- surface as inbox error
            self.inbox.put(_error_message(msg, e))

    async def _wrap_spawn(self, handler: Handler, msg: Message) -> None:
        """Run a spawned handler; surface exceptions; auto-deregister."""
        try:
            await handler.handle(self, msg)
        except Exception as e:  # noqa: BLE001 -- surface as inbox error
            self.inbox.put(_error_message(msg, e))
        finally:
            task = asyncio.current_task()
            if task is not None:
                _ = self.tasks.pop(id(task), None)

    async def _drain_tasks(self) -> None:
        """Cancel and await in-flight spawned tasks; called on shutdown.

        Cancels each task before awaiting so cooperative tasks can
        unwind promptly on quit. Tasks blocked on uncancellable work
        (sync tools in ``asyncio.to_thread``) cannot be forced to stop;
        we wait up to ``_DRAIN_TIMEOUT`` and then proceed -- the
        process-level shutdown will close their executor.
        """
        if not self.tasks:
            return
        for task in list(self.tasks.values()):
            if not task.done():
                _ = task.cancel()
        try:
            async with asyncio.timeout(_DRAIN_TIMEOUT):
                _ = await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        except (TimeoutError, asyncio.CancelledError):
            logger.debug("Drain timed out; %d tasks still running.", len(self.tasks))

    def _unregister_wildcard(self, handler: Handler) -> None:
        """Remove ``handler`` from the wildcard list (best-effort)."""
        with contextlib.suppress(ValueError):
            self._wildcards.remove(handler)

    # -- Event log -----------------------------------------------------

    def log_event(self, event: str, **data: object) -> None:
        """Record a structured event for session.jsonl.

        Caps in-memory growth: when the buffer exceeds
        ``_MAX_UNSAVED_EVENTS``, flush via ``_save_session`` if a
        session_dir is configured, else truncate.
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
                    self._save_session()
                except OSError as save_err:
                    logger.warning(
                        "Event log save failed (%s); truncating in memory.",
                        save_err,
                    )
                    self._event_log = self._event_log[-_MAX_UNSAVED_EVENTS:]
            else:
                self._event_log = self._event_log[-_MAX_UNSAVED_EVENTS:]
        logger.debug("%s: %s", event, data)

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
            tokens=TokenCount(
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

    def _save_session(self, *, clear: bool = False) -> None:
        """Append meta + new messages + events to ``session_dir/session.jsonl``.

        Normal saves append ``self.history[self._persisted_idx:]`` --
        the messages added since the last save. Wholesale-replacement
        paths (``/clear``, compaction, uncompact, repair-on-load that
        inserted messages) pass ``clear=True``: a ``kind: clear``
        barrier line is appended first, then the entire current
        ``self.history`` is re-persisted. The loader treats the barrier
        as a reset, so the live history view starts after the most
        recent barrier; everything before remains in the file for
        forensics and accidental-clear recovery.
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

    def _load_session(self) -> None:
        """Restore meta + history from ``session_dir/session.jsonl``.

        On resume: rebuilds tool-state file caches from the loaded
        history, repairs any dangling tool calls, and restores the
        model+spec when the meta has the necessary recipe fields.
        """
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
        # Repair may insert synthesized ``[interrupted]`` results
        # mid-history. Those inserts aren't on disk yet; flag for a
        # ``clear``-barrier repersist below once the agent state is
        # otherwise consistent.
        repair_inserted = len(repaired) != len(messages)
        self.history = repaired
        # Disk view matches what we just loaded; cursor follows.
        self._persisted_idx = len(messages)
        rebuild_tool_state_from_messages(self.history, self.tool_state)
        if meta:
            m = SessionMeta.deserialize(meta)
            self._session_id = m.session_id or self.session_id
            self._status = m.status
            self.cost_tracker.restore(
                total_cost_usd=m.total_cost_usd,
                total=TokenCount(
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
            # Repair added messages that aren't on disk; persist via a
            # clear barrier so the file's live view matches memory.
            self._save_session(clear=True)


class _DriveForwarder:
    """Wildcard handler: forward display-relevant messages to ``events``.

    Used by ``Agent.run`` to expose streaming events to callers
    iterating the ``RunHandle``. Registered once per ``run`` call,
    unregistered when the run completes. Only display-relevant
    descriptors are forwarded; control flow (``text/x-model-call``,
    ``multipart/x-tool-batch``) stays inboard.
    """

    descriptors: tuple[str, ...] = ()
    spawn: bool = False

    _DISPLAY_DESCRIPTORS = frozenset(
        {
            "text/plain",
            "text/x-thinking",
            "text/x-tool-label",
            "text/x-error",
            "text/x-interrupted",
            "text/x-user-injected",
            "text/x-status-update",
            "multipart/x-tool-result",
            "multipart/x-child-event",
        },
    )

    def __init__(self, events: asyncio.Queue[Message | None]) -> None:
        self._events = events

    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent
        if msg.descriptor in self._DISPLAY_DESCRIPTORS:
            self._events.put_nowait(msg)


class RunHandle:
    """Dual-interface handle: awaitable for the result, async-iterable for events.

    Streaming callers iterate events; batch callers await the result.
    Both may be used on the same handle (iterate first, then await).
    """

    __slots__ = ("_agent", "_drained", "_events", "_task")

    def __init__(
        self,
        agent: Agent,
        task: asyncio.Task[Message],
        events: asyncio.Queue[Message | None],
    ) -> None:
        self._agent = agent
        self._task = task
        self._events = events
        self._drained = False

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncGenerator[Message, None]:
        """Yield streaming events until the run completes."""
        while True:
            event = await self._events.get()
            if event is None:
                self._drained = True
                break
            yield event

    def __await__(self) -> Generator[object, None, Message]:
        return self._await_result().__await__()

    async def _await_result(self) -> Message:
        """Drain unconsumed events, then return the task result."""
        if not self._drained:
            while True:
                event = await self._events.get()
                if event is None:
                    self._drained = True
                    break
        return await self._task


def _subtree_cost(agent: Agent) -> float:
    """Return the active subtree cost: the inherited ledger if present, else the tracker.

    Used by ``Agent.run`` to bracket a single-run cost delta. For the
    root run, the run's own ledger isn't installed yet at ``run`` entry
    so we fall back to ``cost_tracker.total_cost_usd`` (which equals
    snapshot + previously-folded ledgers). For nested children, the
    parent's ledger is already in the ContextVar; reading it gives a
    subtree-wide measurement that includes any descendants the child
    spawns.
    """
    ledger = cost_ledger_var.get()
    if ledger is not None:
        return ledger.total_cost_usd
    return agent.cost_tracker.total_cost_usd


def _error_message(origin: Message, exc: BaseException) -> Message:
    """Wrap an exception as a user-visible ``text/x-error`` message.

    Goes through the inbox spine so ``RenderError`` (REPL) and any
    other ``text/x-error`` subscribers see it. The descriptor is
    deliberately the user-facing one rather than ``application/json``
    so silent drops never happen on handler exceptions.
    """
    return TextMessage(
        f"{type(exc).__name__}: {exc}",
        "text/x-error",
        parent_id=origin.id,
    )


def _last_assistant_message(history: list[Message]) -> Message:
    """Return the last assistant response as a flat ``text/plain`` message.

    Tool contract (Agent-as-Tool, AgentSpawn callers, ``orchestrator``):
    a ``Tool.run`` returns a single Message whose textual content goes
    into the parent's history as the tool result. We extract the
    last ``multipart/x-model-message``'s text and return a flat
    ``TextMessage`` so that contract is preserved across v1 -> v2.

    Round-limit / refusal sentinels: when the multipart's only text
    content is a ``text/x-error`` part (synthesized by
    :func:`ModelCallHandler._post_round_limit_response`), we surface
    it as a ``text/x-error`` ``TextMessage`` so callers can match on
    descriptor + content (parity with v1's ``ERROR_MAX_TOOL_CALL_ROUNDS``).
    """
    for msg in reversed(history):
        if msg.descriptor == "multipart/x-model-message":
            text = response_text(msg)
            if text:
                return TextMessage(text, "text/plain")
            if isinstance(msg, MultipartMessage):
                for part in msg.content:
                    if part.descriptor == "text/x-error":
                        return TextMessage(str(part.content), "text/x-error")
            return TextMessage("", "text/plain")
    return TextMessage("", "text/plain")
