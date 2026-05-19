"""Agent: composition over :class:`AgentRuntime`.

Owns three wrappers and a small set of observers:

- :class:`_AgentModel` bridges the rich provider ``Model`` interface
  (``buffer`` / ``stream`` returning ``ModelResponse``) to the
  runtime's lean ``stream(history, system, tools, on_text,
  on_thinking) -> AssistantMessage`` protocol. Runs the retry loop
  with persistent-retry and overflow recovery. Records cost
  out-of-band on :attr:`Agent.cost_tracker`.

- :class:`_AgentTool` bridges a rich ``Tool`` (with metadata,
  ``summary`` / ``summary_result`` / ``prompt``) to the runtime's
  minimal ``run(args) -> ToolResult`` protocol. Emits ``ToolLabel``
  before running, validates the directive against the tool's
  ``directive_schema``, postprocesses the result (empty-marker,
  oversized persist).

- :class:`_AgentCompactor` bridges the rich ``SummaryCompactor``
  interface to the runtime's lean ``compact(history, model, args)``
  protocol. Writes ``pre_compact_<N>.jsonl`` transcripts when a
  session dir is set; runs the post-compact enrich pipeline
  (file reattach, status injection, tool restore).

Observers track cost, activity (call timing, streamed chars), tool
registry (cohort id → tool name), persistence (``SaveSession`` →
session.jsonl append), and budget caps (max tool-call rounds, max
budget USD).

Public surface includes:
``halt`` / ``kill_tool`` / ``kill_all_tools`` / ``shutdown`` /
``compact`` / ``recompact`` / ``swap_model`` / ``serve_forever`` /
``run``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from pathlib import Path

import asyncio
import contextlib
import contextvars
import dataclasses
import itertools
import logging
import time
import uuid

from sagent import providers, types
from sagent.agent import runtime as agent_runtime
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    split_bg_args,
)
from sagent.agent.compaction import (
    CompactionState,
    estimate_total_tokens,
    post_compact_enrich,
)
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.result_storage import post_process_result
from sagent.agent.retry import send_with_retry
from sagent.agent.session_io import (
    SessionMeta,
    rebuild_content_cache,
    restore_model,
)
from sagent.agent.state import (
    ToolState,
    agent_counter_var,
    agent_label_var,
    agent_registry,
    cost_root_var,
    current_agent_var,
    tool_state_var,
    unique_registry_label,
)
from sagent.lib import last_models
from sagent.lib.compaction import write_pre_compact_transcript
from sagent.lib.json import JSON


logger = logging.getLogger(__name__)

SystemPromptArg = str | Callable[[], str]
"""System-prompt spec. ``str`` is literal; ``Callable[[], str]`` is
re-invoked per request so cwd-aware sections stay live after ``cd``."""

ERROR_MAX_TOOL_CALL_ROUNDS = "error:max_tool_call_rounds"
MAX_OVERFLOW_RECOVERY = 3


@dataclasses.dataclass(kw_only=True, slots=True)
class ActivityTracker:
    """Lifecycle counters for the status pane."""

    elapsed_seconds: float = 0.0
    """Cumulative wall-clock seconds spent in active model calls."""

    current_call_start: float = 0.0
    """Event-loop time when the current model call started (``0`` when idle)."""

    current_compact_start: float = 0.0
    """Event-loop time when the current compaction started (``0`` when idle)."""

    live_response_chars: int = 0
    """Characters streamed so far in the current response."""

    active: bool = False
    """True between ``types.runtime.ModelCallStarted`` and ``types.runtime.ModelResponseComplete`` /
    ``types.runtime.ModelIdle`` for the current call."""

    num_tool_call_rounds: int = 0
    """Cumulative count of responses that included tool calls."""


class Agent:
    """Conversation agent: composes :class:`agent_runtime.AgentRuntime` with wrappers.

    Args:
      model: Rich provider model the agent calls.
      model_spec: Optional spec recording how the model was built.
      system: Static system prompt string or a no-arg factory rebuilt
          each request.
      tools: Rich tools advertised to the model.
      compactor: Rich compactor used on ``types.runtime.Compact`` / ``types.runtime.Recompact`` and
          on overflow recovery.
      session_dir: Directory for session persistence and pre-compact
          transcripts; ``None`` disables both.
      budget: Context budget; defaults to ``types.model.ContextBudget.from_model``.
      max_attempts: Retry attempts inside ``send_with_retry``.
      name: Human-readable agent label.
      description: Agent description for parent agents and the UI.
      max_tool_call_rounds: Cap on tool-call rounds before the agent
          forces ``types.runtime.ModelResponseError``.
      thinking: Extended-thinking mode; passed through when supported.
      effort: Effort hint; passed through when supported.
      max_budget_usd: Hard USD cap; ``record_response`` raises when hit.
      persistent_retry: Enable persistent-mode backoff for 429/529.

    """

    def __init__(
        self,
        *,
        model: types.model.Model,
        model_spec: types.model.ModelSpec | None = None,
        system: SystemPromptArg = "",
        tools: list[types.tools.Tool] | None = None,
        compactor: types.compactor.Compactor | None = None,
        session_dir: str | Path | None = None,
        budget: types.model.ContextBudget | None = None,
        max_attempts: int = 5,
        name: str = "Agent",
        description: str = "An AI agent.",
        max_tool_call_rounds: int | None = None,
        thinking: str | None = "adaptive",
        effort: str | None = None,
        max_budget_usd: float | None = None,
        persistent_retry: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.model = model
        self.model_spec = model_spec
        if model_spec is not None:
            last_models.record(model_spec.provider, model_spec.model_id)
        self._system_spec: SystemPromptArg = system
        self._tools_list: list[types.tools.Tool] = list(tools or [])
        self.compactor = compactor
        if budget is None:
            budget = types.model.ContextBudget.from_model(model)
        self._budget = budget
        self.max_attempts = max_attempts
        self.max_tool_call_rounds = max_tool_call_rounds
        self._thinking = thinking
        self._effort = effort
        self._cache_ttl: str = "5m"
        self._service_tier: str | None = None
        self.persistent_retry = persistent_retry
        self._max_budget_usd = max_budget_usd

        self.cost_tracker = CostTracker()
        self.activity = ActivityTracker()
        self.tool_state = ToolState()
        self.compaction_state = CompactionState()
        self._bg: dict[str, BackgroundTaskEntry] = {}
        self.observers: list[Callable[[types.runtime.RuntimeEvent], None]] = []
        # ``_tool_registry`` maps cohort call_id → (tool_name, started_at)
        # so ``background`` can synthesize ``BackgroundTaskEntry`` rows
        # for detached cohort members.
        self._tool_registry: dict[str, tuple[str, float]] = {}
        self.session_dir: Path | None = (
            Path(session_dir) if session_dir is not None else None
        )
        self._session_id = uuid.uuid4().hex[:8]
        self._status: str = ""
        self._persistent: bool = False
        self._shutting_down: bool = False

        # Wrappers. ``_agent_compactor`` is None when no rich compactor
        # was supplied (the runtime is satisfied with a None compactor).
        self._agent_model = _AgentModel(model, self)
        # ``_tools_map`` holds the RAW rich tool keyed by name so
        # isinstance / Protocol checks (CompactRestorable, Slack, ...) at
        # consumer sites pass through. The schema-augmenting
        # ``BackgroundAwareTool`` wrapper is applied per request in
        # ``_AgentModel.stream`` when building the provider tool list.
        self._tools_map: dict[str, types.tools.Tool] = {}
        agent_tools: list[agent_runtime.Tool] = []
        for t in self._tools_list:
            self._tools_map[t.name] = t
            agent_tools.append(_AgentTool(t, self))
        self._agent_compactor = (
            _AgentCompactor(compactor, self) if compactor is not None else None
        )
        self.runtime = agent_runtime.AgentRuntime(
            model=self._agent_model,
            tools=agent_tools,
            compactor=self._agent_compactor,
            system=self._build_system(),
        )

        for fn in (
            self._track_activity,
            self._track_tool_registry,
            self._enforce_caps,
        ):
            self.runtime.observers.append(fn)

    # -- Properties / config surface ----------------------------------

    @property
    def budget(self) -> types.model.ContextBudget:
        """Context budget; auto-derived from the model when unset."""
        return self._budget

    @property
    def max_request_tokens(self) -> int:
        """Active per-request input token budget."""
        return self._budget.max_request_tokens

    @max_request_tokens.setter
    def max_request_tokens(self, value: int) -> None:
        """Set the per-request input token budget; bounded by the model.

        Args:
          value: New input token budget; must not exceed the model's cap.

        Raises:
          ValueError: If ``value`` exceeds the model's ``max_request_tokens``.

        """
        if value > self.model.max_request_tokens:
            raise ValueError(
                f"max_request_tokens={value:,} exceeds model's"
                f" {self.model.max_request_tokens:,}",
            )
        self._budget = dataclasses.replace(self._budget, max_request_tokens=value)

    @property
    def max_response_tokens(self) -> int:
        """Active per-request response token budget."""
        return self._budget.max_response_tokens

    @max_response_tokens.setter
    def max_response_tokens(self, value: int) -> None:
        """Set the per-request response token budget; bounded by the model.

        Args:
          value: New response token budget; must not exceed the model's cap.

        Raises:
          ValueError: If ``value`` exceeds the model's ``max_response_tokens``.

        """
        if value > self.model.max_response_tokens:
            raise ValueError(
                f"max_response_tokens={value:,} exceeds model's"
                f" {self.model.max_response_tokens:,}",
            )
        self._budget = dataclasses.replace(self._budget, max_response_tokens=value)

    def reset_budget(self) -> None:
        """Reset budget to model-derived defaults."""
        self._budget = types.model.ContextBudget.from_model(self.model)

    @property
    def thinking(self) -> str | None:
        """Extended-thinking mode (``"adaptive"`` etc.), or ``None`` to disable."""
        return self._thinking

    @thinking.setter
    def thinking(self, value: str | None) -> None:
        """Set the extended-thinking mode.

        Args:
          value: Mode string passed through to the provider, or ``None``.

        """
        self._thinking = value

    @property
    def effort(self) -> str | None:
        """Provider effort hint, or ``None`` when unset."""
        return self._effort

    @effort.setter
    def effort(self, value: str | None) -> None:
        """Set the provider effort hint; rejected if the model lacks support.

        Args:
          value: Effort hint string, or ``None`` to clear.

        Raises:
          ValueError: If the model does not support effort hints.

        """
        if value is not None and not self.model.supports_effort:
            raise ValueError(f"Model {self.model.model_id!r} does not support effort.")
        self._effort = value

    @property
    def cache_ttl(self) -> str:
        """Cache TTL marker (``"5m"`` or ``"1h"``)."""
        return self._cache_ttl

    @cache_ttl.setter
    def cache_ttl(self, value: str) -> None:
        """Set the cache TTL marker.

        Args:
          value: Either ``"5m"`` or ``"1h"``.

        Raises:
          ValueError: If ``value`` is neither ``"5m"`` nor ``"1h"``.

        """
        if value not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {value!r}")
        self._cache_ttl = value

    @property
    def service_tier(self) -> str | None:
        """OpenAI processing-tier hint, or ``None`` when unset."""
        return self._service_tier

    @service_tier.setter
    def service_tier(self, value: str | None) -> None:
        """Set the OpenAI service-tier hint; rejected when unsupported.

        Args:
          value: Tier name (``"auto"`` / ``"default"`` / ``"flex"`` /
              ``"priority"``), or ``None`` to clear.

        Raises:
          ValueError: If the model does not support a service-tier hint
              or ``value`` is not one of the accepted tiers.

        """
        if value is not None:
            valid = self.model.valid_service_tiers
            if not valid:
                raise ValueError(
                    f"Model {self.model.model_id!r} does not support service_tier.",
                )
            if value not in valid:
                quoted = ", ".join(repr(t) for t in valid)
                raise ValueError(
                    f"service_tier must be one of {quoted}, got {value!r}",
                )
        self._service_tier = value

    @property
    def session_id(self) -> str:
        """Short hex id assigned at agent construction."""
        return self._session_id

    @property
    def status(self) -> str:
        """Free-form status string surfaced by the status pane."""
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        """Set the status string (rendered in the status pane) and publish a ``types.runtime.StatusChanged`` event.

        Args:
          value: New status string.

        """
        if value == self._status:
            return
        self._status = value
        self.runtime.publish(types.runtime.StatusChanged(text=value))

    @property
    def history(self) -> list[types.history.HistoryEntry]:
        """Conversation history (read/write view of runtime.history)."""
        return self.runtime.history

    @property
    def inbox(self):
        """The runtime's inbox (``GatedDeque[types.runtime.RuntimeEvent]``)."""
        return self.runtime.inbox

    @property
    def work(self) -> asyncio.Task[None] | None:
        """The currently active foreground task (model call or compaction)."""
        return self.runtime.model_call or self.runtime.compact_task

    @property
    def tools_map(self) -> dict[str, types.tools.Tool]:
        """Map of tool name → rich tool (pre-wrap)."""
        return self._tools_map

    @property
    def tools(self) -> list[types.tools.Tool]:
        """Rich tools in registration order (pre-wrap copies)."""
        return list(self._tools_map.values())

    @property
    def system(self) -> str:
        """Assembled system prompt (base + per-tool contributions)."""
        return self._build_system()

    @property
    def total_cost_usd(self) -> float:
        """Cumulative USD cost across all recorded responses."""
        return self.cost_tracker.total_cost_usd

    @property
    def total_tokens(self) -> types.model.TokenCount:
        """Cumulative token counts across all recorded responses."""
        return self.cost_tracker.total

    @property
    def num_tool_call_rounds(self) -> int:
        """Cumulative count of responses that included tool calls."""
        return self.activity.num_tool_call_rounds

    @property
    def background(self) -> dict[str, BackgroundTaskEntry]:
        """Merged view: cohort-detached tools + explicit-bg + persistent + REPL pump."""
        merged: dict[str, BackgroundTaskEntry] = {}
        for qid, task in self.runtime.detached.items():
            name, started = self._tool_registry.get(qid, ("?", time.time()))
            merged[qid] = BackgroundTaskEntry(
                task=task,
                tool_name=name,
                queue_id=qid,
                started=started,
                kind="detached",
            )
        merged.update(self._bg)
        return merged

    def publish(self, event: types.runtime.RuntimeEvent) -> None:
        """Forward an event to the runtime's observer list.

        Args:
          event: Event to deliver to every observer.

        """
        self.runtime.publish(event)

    # -- Mutation methods ---------------------------------------------

    def swap_model(
        self, model: types.model.Model, *, spec: types.model.ModelSpec | None = None
    ) -> None:
        """Replace the active model.

        Clears ``thinking`` / ``effort`` when the new model lacks
        support so the agent's user-visible state matches what the
        provider will actually receive. Schedules ``close()`` on the
        swapped-out model so CLI providers' subprocess pools (the
        ``claude`` / ``gemini`` process plus its warming-spare task)
        don't leak past the swap.

        Args:
          model: New rich provider model.
          spec: Optional spec recording how the model was built.

        Raises:
          ValueError: Explicit budget exceeds the new model's limits.

        """
        if self._budget.max_request_tokens > model.max_request_tokens:
            raise ValueError(
                f"budget.max_request_tokens={self._budget.max_request_tokens:,}"
                f" exceeds new model's {model.max_request_tokens:,}",
            )
        if self._budget.max_response_tokens > model.max_response_tokens:
            raise ValueError(
                f"budget.max_response_tokens={self._budget.max_response_tokens:,}"
                f" exceeds new model's {model.max_response_tokens:,}",
            )
        old = self.model
        self.model = model
        self.model_spec = spec
        self._agent_model.set_inner(model)
        self.runtime.model = self._agent_model
        if not model.supports_thinking:
            self._thinking = None
        if not model.supports_effort:
            self._effort = None
        if spec is not None:
            last_models.record(spec.provider, spec.model_id)
        _schedule_close(old)

    def change_model(
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
    ) -> types.model.ModelSpec:
        """Resolve, build, and queue a model swap. The high-level API.

        Kwarg semantics: each defaults to ``None`` meaning "inherit from
        the current ``model_spec``." Note that ``account=None`` therefore
        inherits the current account override; setting ``account`` to
        the default backend account (literal ``None``) is not expressible
        via this API -- construct a ``types.model.ModelSpec`` and call
        :meth:`swap_model` directly for that corner.

        Cross-provider resolution when ``model_id`` is omitted:
        1. Prefer the current model id when the new provider's catalog
           knows it (same vendor, different auth subclass).
        2. Else use the last model recorded for the new provider in
           ``~/.sagent/last-models.json``.
        3. Else fall back to the new provider's ``DEFAULT_MODEL``.

        Queues a :class:`types.runtime.ModelSwitch` through the runtime inbox so any
        in-flight model call finishes against the OLD model (cost
        attribution, retry state) before the new model goes live.

        Args:
          provider: New provider class name, e.g. ``"AnthropicCLI"``.
          auth: New auth-method suffix.
          model_id: New provider-specific model id.
          account: New credential account override.

        Returns:
          target: Resolved target spec (the spec the swap will land on).

        Raises:
          ValueError: ``model_spec`` is unset, the resolved provider is
              unknown, or the resolved model id is rejected by the
              provider's catalog.

        """
        spec = self.model_spec
        if spec is None:
            raise ValueError("agent has no model_spec; cannot change_model")
        target = _resolve_target_spec(
            spec,
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
        )
        provider_obj = providers.build_provider(
            target.provider, target.auth, account=target.account
        )
        new_model = provider_obj.model(target.model_id)
        if target.provider != spec.provider:
            label = (
                f"{spec.provider}/{spec.model_id} -> "
                f"{target.provider}/{target.model_id}"
            )
        else:
            label = f"{spec.model_id} -> {target.model_id}"
        self.runtime.inbox.push_back(
            types.runtime.ModelSwitch(
                apply=lambda: self._apply_model_change(new_model, target),
                label=label,
            ),
        )
        return target

    async def relogin(self) -> None:
        """Re-authenticate the current provider; hot-reload live creds.

        Drives the provider class's ``login`` classmethod (which writes
        fresh OAuth credentials to disk), then re-reads them into the
        running provider's in-memory token state via
        :class:`types.providers.AuthReloadable.handle_auth_error`. Without the reload
        the in-memory ``_refresh_token`` is still the revoked one and
        the next refresh returns 400 -- so ``/login`` would appear to
        succeed but the auth error would keep firing.

        Raises:
          ValueError: ``model_spec`` is unset, the provider class is
              unknown, or the provider has no ``login`` classmethod.

        """
        spec = self.model_spec
        if spec is None:
            raise ValueError("agent has no model_spec; cannot relogin")
        prov_cls = getattr(providers, spec.provider, None)
        if prov_cls is None:
            raise ValueError(f"unknown provider {spec.provider!r}")
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            raise ValueError(f"provider {spec.provider!r} has no login method")
        login_fn()
        live_provider = getattr(self.model, "_provider", None)
        if isinstance(live_provider, types.providers.AuthReloadable):
            await live_provider.handle_auth_error()

    def system_prompt(self) -> str:
        """Assemble the full system prompt (system + tool contributions).

        Returns:
          prompt: System prompt rebuilt for the next request.

        """
        return self._build_system()

    def _apply_model_change(
        self, model: types.model.Model, spec: types.model.ModelSpec
    ) -> None:
        """Apply a high-level model change, resetting stale derived budgets.

        Publishes ``BudgetReset`` when the prior budget couldn't fit the
        new model -- the reset is destructive of any ``ContextBudget``
        customisation, so renderers surface a notification.
        """
        if (
            self._budget.max_request_tokens > model.max_request_tokens
            or self._budget.max_response_tokens > model.max_response_tokens
        ):
            prior = self._budget
            self._budget = types.model.ContextBudget.from_model(model)
            self.runtime.publish(
                types.runtime.BudgetReset(
                    model_id=model.model_id,
                    prior_max_request_tokens=prior.max_request_tokens,
                    prior_max_response_tokens=prior.max_response_tokens,
                    new_max_request_tokens=self._budget.max_request_tokens,
                    new_max_response_tokens=self._budget.max_response_tokens,
                )
            )
        self.swap_model(model, spec=spec)

    def resume(
        self,
        meta: SessionMeta,
        history: list[types.history.HistoryEntry],
        tool_state: ToolState,
    ) -> None:
        """Apply a persisted session snapshot to this agent.

        Restores cost / activity / compaction state, the original
        ``session_id`` and ``status``, and (when the persisted
        provider+model differ from the current one) swaps the model.
        Repopulates ``tool_state`` and reseeds its content cache from
        current disk for previously-touched files so post-resume
        ``check_stale`` doesn't fire on mtime drift.

        Args:
          meta: Persisted ``SessionMeta``.
          history: Persisted conversation history.
          tool_state: Persisted ``ToolState`` snapshot.

        """
        self.runtime.history.extend(history)
        self.tool_state = tool_state
        rebuild_content_cache(history, self.tool_state)
        if meta.session_id:
            self._session_id = meta.session_id
        if meta.status:
            self._status = meta.status
        self.cost_tracker.restore(total_cost_usd=meta.total_cost_usd, total=meta.tokens)
        self.activity.num_tool_call_rounds = meta.num_tool_call_rounds
        self.activity.elapsed_seconds = meta.total_active_elapsed_seconds
        self.compaction_state.compact_count = meta.compact_count
        self.compaction_state.summary_pointers = list(meta.summary_pointers)
        if meta.provider and meta.model_id and meta.model_id != self.model.model_id:
            restored = restore_model(meta)
            if restored is not None:
                new_model, new_spec = restored
                self.swap_model(new_model, spec=new_spec)

    # -- Foreground slot / cancel verbs --------------------------------

    def halt(self) -> None:
        """Cancel the current model call; wait for user input."""
        self.runtime.inbox.push_back(types.runtime.Halt())

    def kill_tool(self, qid: str) -> None:
        """Cancel one outstanding tool task.

        Args:
          qid: Cohort call id of the task to cancel.

        """
        self.runtime.inbox.push_back(types.runtime.Kill(call_id=qid))

    def kill_all_tools(self) -> None:
        """Cancel every outstanding tool task."""
        self.runtime.inbox.push_back(types.runtime.Kill())

    def shutdown(self, *, force: bool = False) -> None:
        """End ``serve_forever`` cleanly.

        Args:
          force: When True, also cancel foreground + visible bg jobs.

        """
        self._shutting_down = True
        if force:
            self.kill_all_tools()
            for job in list(self._bg.values()):
                if not job.hidden and not job.task.done():
                    _ = job.task.cancel()
        self.runtime.inbox.push_back(types.runtime.Quit())

    # -- Strategy methods ---------------------------------------------

    async def compact(self, args: str = "") -> None:
        """Preempt in-flight work and run compaction.

        Args:
          args: Free-form compaction instructions forwarded to the compactor.

        """
        await self._await_event(
            types.runtime.Compact(args=args), types.runtime.CompactComplete
        )

    async def recompact(self, args: str = "") -> None:
        """Reload the most recent pre-compact transcript and re-run compaction.

        Args:
          args: Free-form compaction instructions forwarded to the compactor.

        """
        await self._await_event(
            types.runtime.Recompact(args=args), types.runtime.CompactComplete
        )

    async def clear(self) -> None:
        """Preempt and wipe history + file-tracking caches."""
        self.tool_state.reset_file_tracking()
        self.runtime.inbox.push_back(types.runtime.Clear())

    async def serve_forever(self) -> None:
        """Drive the agent until ``shutdown`` is called."""
        with self._install_contextvars():
            await self.runtime.run_forever()

    async def run(
        self, msg: types.history.UserMessage
    ) -> AsyncGenerator[types.runtime.RuntimeEvent, None]:
        """Process one inbound message; drive rounds until idle.

        Convenience entrypoint used by tests and non-``serve_forever``
        callers.

        Args:
          msg: User message to push and run to idle.

        Yields:
          event: Each ``types.runtime.RuntimeEvent`` published until ``types.runtime.ModelIdle``.

        """
        events: asyncio.Queue[types.runtime.RuntimeEvent] = asyncio.Queue()
        idle = asyncio.Event()

        def _watch(event: types.runtime.RuntimeEvent) -> None:
            events.put_nowait(event)
            if isinstance(event, types.runtime.ModelIdle):
                idle.set()

        self.runtime.observers.append(_watch)
        try:
            self.runtime.inbox.push_back(msg)
            drive = asyncio.create_task(self.serve_forever())
            drive.add_done_callback(
                types.exceptions.log_task_exception(
                    logger, "Agent.run drive task crashed"
                ),
            )
            try:
                while True:
                    get_task = asyncio.create_task(events.get())
                    idle_task = asyncio.create_task(idle.wait())
                    done, _pending = await asyncio.wait(
                        {get_task, idle_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Drain any pending event before checking idle so the
                    # consumer sees the types.runtime.ModelIdle that triggered the flag.
                    if get_task in done:
                        yield get_task.result()
                    else:
                        _ = get_task.cancel()
                    if idle_task not in done:
                        _ = idle_task.cancel()
                    if idle.is_set() and events.empty():
                        break
            finally:
                self.shutdown(force=False)
                with contextlib.suppress(asyncio.CancelledError):
                    await drive
        finally:
            if _watch in self.runtime.observers:
                self.runtime.observers.remove(_watch)

    # -- Internal helpers ---------------------------------------------

    @contextlib.contextmanager
    def _install_contextvars(self):
        """Install per-agent ContextVars for the lifetime of the block.

        Non-persistent subagents inherit the parent's cost tracker so
        the root sees the full spawn-tree spend (and ``max_budget_usd``
        caps the tree, not just the root's own calls). Tool-state depth
        is incremented from the parent so ``AgentSpawn`` depth caps
        actually fire. Default-name collisions in ``agent_registry``
        are resolved with a numeric suffix so ``AgentSend`` can address
        a specific agent even when several share a base name.
        """
        agent_token = current_agent_var.set(self)
        parent_root = cost_root_var.get(None)
        cost_token: contextvars.Token[CostTracker | None] | None = (
            cost_root_var.set(self.cost_tracker)
            if self._persistent or parent_root is None
            else None
        )
        parent_state = tool_state_var.get(None)
        self.tool_state.depth = 0 if parent_state is None else parent_state.depth + 1
        # Persistent agents have a definite ``self.name`` set by
        # ``AgentSpawn._spawn_persistent`` and own their own task; their
        # identity must come from ``self.name`` directly. Inheriting
        # ``agent_label_var`` from the parent task (``asyncio.create_task``
        # copies the current context) would silently overwrite the parent's
        # registry entry -- e.g. spawning persistent ``"reviewer-opus"``
        # from a parent with label ``"Agent"`` would set
        # ``agent_registry["Agent"] = reviewer_opus``, and then every
        # ``AgentSend("Agent", ...)`` from any sub would route to the
        # reviewer instead of the running parent.
        if self._persistent:
            label = self.name
        else:
            base_label = agent_label_var.get("") or self.name
            label = unique_registry_label(base_label)
        label_token = agent_label_var.set(label)
        counter_token = agent_counter_var.set(itertools.count())
        state_token = tool_state_var.set(self.tool_state)
        agent_registry[label] = self
        try:
            yield
        finally:
            if agent_registry.get(label) is self:
                _ = agent_registry.pop(label, None)
            tool_state_var.reset(state_token)
            agent_counter_var.reset(counter_token)
            agent_label_var.reset(label_token)
            if cost_token is not None:
                cost_root_var.reset(cost_token)
            current_agent_var.reset(agent_token)

    async def _await_event(
        self,
        push: types.runtime.RuntimeEvent,
        complete: type[types.runtime.RuntimeEvent],
    ) -> None:
        """Push ``push`` and resolve when an event of ``complete`` type arrives."""
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def resolver(ev: types.runtime.RuntimeEvent) -> None:
            if isinstance(ev, complete) and not fut.done():
                fut.set_result(None)

        self.runtime.observers.append(resolver)
        try:
            self.runtime.inbox.push_back(push)
            await fut
        finally:
            if resolver in self.runtime.observers:
                self.runtime.observers.remove(resolver)

    def _build_system(self) -> str:
        """Assemble the system prompt from base spec + tool contributions."""
        spec = self._system_spec
        base = spec if isinstance(spec, str) else spec()
        parts: list[str] = [base] if base else []
        for tool in self._tools_map.values():
            contribution = tool.prompt()
            if contribution:
                parts.append(contribution)
        return "\n\n".join(parts)

    # -- Observers ----------------------------------------------------

    def record_response(self, response: types.model.ModelResponse) -> None:
        """Record a completed response into ``cost_tracker``.

        Writes through to ``cost_root_var.get(self.cost_tracker)`` so
        subagents accumulate into the root agent's tracker.

        Args:
          response: Completed model response with token counts and cost.

        Raises:
          RuntimeError: Cumulative cost reached ``max_budget_usd``.

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

    def _track_activity(self, event: types.runtime.RuntimeEvent) -> None:
        """Bracket round-chain elapsed time + count streamed chars.

        A round chain spans from the first ``types.runtime.ModelCallStarted`` of a
        user turn through ``types.runtime.ModelIdle`` (or terminal cancel). Mid-chain
        ``types.runtime.ModelResponseComplete`` events with ``tool_calls`` do not
        reset ``active`` so the status-pane spinner keeps ticking
        through tool execution windows -- the user sees continuous
        activity until the agent truly idles.
        """
        if isinstance(event, types.runtime.ModelCallStarted):
            if not self.activity.active:
                self.activity.active = True
                self.activity.current_call_start = asyncio.get_running_loop().time()
            self.activity.live_response_chars = 0
        elif isinstance(event, types.runtime.ModelResponsePartial):
            self.activity.live_response_chars += len(event.text)
        elif (
            isinstance(event, types.runtime.ModelResponseComplete)
            and event.message.tool_calls
        ):
            # Tool calls follow; spinner keeps ticking through the cohort.
            pass
        elif isinstance(
            event,
            (
                types.runtime.ModelResponseComplete,
                types.runtime.ModelIdle,
                types.runtime.ModelResponseCancelled,
                types.runtime.ModelResponseError,
            ),
        ):
            if self.activity.active:
                elapsed = (
                    asyncio.get_running_loop().time() - self.activity.current_call_start
                )
                self.activity.elapsed_seconds += max(0.0, elapsed)
                self.activity.active = False
                self.activity.current_call_start = 0.0
        elif isinstance(event, types.runtime.CompactStarted):
            self.activity.current_compact_start = asyncio.get_running_loop().time()
        elif isinstance(
            event, (types.runtime.CompactComplete, types.runtime.CompactFailed)
        ):
            self.activity.current_compact_start = 0.0

    def _track_tool_registry(self, event: types.runtime.RuntimeEvent) -> None:
        """Populate the cohort id → (tool_name, started) registry."""
        if isinstance(event, types.runtime.ModelResponseComplete):
            now = time.time()
            for tc in event.message.tool_calls:
                self._tool_registry[tc.id] = (tc.name, now)
            if event.message.tool_calls:
                self.activity.num_tool_call_rounds += 1

    def _enforce_caps(self, event: types.runtime.RuntimeEvent) -> None:
        """Push ``types.runtime.ModelResponseError`` when caps are hit."""
        if (
            isinstance(event, types.runtime.ModelResponseComplete)
            and self.max_tool_call_rounds is not None
            and self.activity.num_tool_call_rounds >= self.max_tool_call_rounds
            and event.message.tool_calls
        ):
            self.runtime.inbox.push_back(
                types.runtime.ModelResponseError(
                    RuntimeError(
                        "Tool-call-round limit reached"
                        f" ({self.max_tool_call_rounds} rounds)."
                        f" [{ERROR_MAX_TOOL_CALL_ROUNDS}]"
                    )
                )
            )

    def microcompact_history(self, history: list[types.history.HistoryEntry]) -> None:
        """Apply the compactor's in-place microcompaction to ``history``.

        No-op when no rich compactor is wired. Used by the model
        wrapper to trim stale tool results before each model call.

        Args:
          history: History list to mutate in place.

        """
        if self._agent_compactor is not None:
            self._agent_compactor.maintain(history)

    def cancel_background(self, job_id: str) -> None:
        """Remove ``job_id`` from the background-task registry, if present.

        Args:
          job_id: Queue id of the registered task to drop.

        """
        self._bg.pop(job_id, None)

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        """Add ``entry`` to the background-task registry under ``job_id``.

        Args:
          job_id: Queue id used as the registry key.
          entry: Background-task record to store.

        """
        self._bg[job_id] = entry

    async def compact_if_needed(
        self,
        history: list[types.history.HistoryEntry],
        model: types.model.Model,
    ) -> bool:
        """Proactively compact when the compactor says headroom is gone.

        Bridges the inner compactor's ``should_compact`` decision (which
        the runtime's lean ``Compactor`` protocol does not expose) to the
        synchronous ``compact_now`` path used for overflow recovery.

        Returns:
          progressed: ``True`` when no compaction was needed, or when
              compaction completed. ``False`` when ``compact_now``
              tried and failed. The bool mirrors :meth:`compact_now`'s
              contract so callers don't have to special-case the
              proactive vs reactive path.

        Args:
          history: Pre-compaction history snapshot (mutated in place if
              ``compact_now`` runs; same list as ``runtime.history``).
          model: Rich model whose tokenizer estimates seed the headroom
              check and whose ``max_request_tokens`` caps the budget.

        """
        if self._agent_compactor is None:
            return True
        used = estimate_total_tokens(self.system_prompt(), history, model)
        if not await self._agent_compactor.should_compact(
            input_tokens=used,
            max_request_tokens=model.max_request_tokens,
            max_response_tokens=self.max_response_tokens,
        ):
            return True
        return await self.compact_now()

    async def compact_now(self) -> bool:
        """Synchronous compact path used by ``_AgentModel`` for overflow recovery.

        Bypasses the inbox (the runtime would cancel our task if we
        pushed ``types.runtime.Compact``). Calls the inner compactor directly,
        replaces ``runtime.history`` in place.

        Returns:
          progressed: ``True`` when compaction completed (or no compactor is
              wired, which is the agent's chosen configuration); ``False``
              when the inner compactor raised. The overflow-recovery caller
              uses this signal to short-circuit instead of looping on
              unchanged history.

        """
        if self._agent_compactor is None:
            return True
        try:
            summary = await self._agent_compactor.compact(
                list(self.runtime.history),
                self._agent_model,
                "",
            )
        except Exception as exc:  # noqa: BLE001 -- compaction calls the model; catch-all routes UserFacingError to warning, others to exception
            types.exceptions.log_exception_or_warning(
                logger,
                "synchronous compaction failed during overflow recovery",
                exc,
            )
            self.runtime.history.append(
                types.history.UserMessage(
                    text=f"[Compaction error: {type(exc).__name__}: {exc}]"
                ),
            )
            return False
        self.runtime.history.clear()
        self.runtime.history.extend(summary)
        return True


def _context_overflow_error() -> types.exceptions.ContextOverflowError:
    """Build the user-facing exhaustion error.

    The renderer treats ``UserFacingError`` specially -- no ``ClassName:``
    prefix, no traceback -- so this message is what the user actually
    reads after recovery exhausts. Keep it actionable: name the verbs
    (``/clear``, ``/compact``, ``/model``) so the halt screen tells the
    user what to do, not just what went wrong.

    The underlying provider exception travels via ``__cause__`` at the
    raise site; operators inspecting logs see the technical detail via
    the exception chain, while end users see only the polished verbs.
    """
    return types.exceptions.ContextOverflowError(
        "Context window exhausted after auto-compaction. "
        "Use /clear to wipe history, /compact <hints> to retry with custom "
        "guidance, or /model to switch to a larger-window model.",
    )


def _resolve_target_spec(
    spec: types.model.ModelSpec,
    *,
    provider: str | None,
    auth: str | None,
    model_id: str | None,
    account: str | None,
) -> types.model.ModelSpec:
    """Resolve a ``change_model`` kwargs payload to a complete ``types.model.ModelSpec``.

    ``None`` kwargs inherit from ``spec``. The model-id branch implements
    the cross-provider preservation rule documented on
    :meth:`Agent.change_model`.
    """
    prov_name = provider or spec.provider
    final_auth = auth or spec.auth
    final_account = account if account is not None else spec.account
    if model_id is not None:
        final_model_id = model_id
    elif prov_name == spec.provider or _provider_knows_model(prov_name, spec.model_id):
        final_model_id = spec.model_id
    else:
        final_model_id = last_models.get(prov_name) or _default_model_for(prov_name)
    return types.model.ModelSpec(
        provider=prov_name,
        auth=final_auth,
        account=final_account,
        model_id=final_model_id,
    )


def _provider_knows_model(prov_name: str, model_id: str) -> bool:
    """Return True when the provider class's catalog includes ``model_id``.

    Reads ``cls.KNOWN_MODELS`` without instantiating so the probe is
    side-effect-free (no credential lookup).
    """
    cls = getattr(providers, prov_name, None)
    if cls is None:
        return False
    known = getattr(cls, "KNOWN_MODELS", None)
    if not isinstance(known, dict):
        return False
    return model_id in known


def _default_model_for(prov_name: str) -> str:
    """Return ``Provider.DEFAULT_MODEL`` for the named provider class."""
    cls = getattr(providers, prov_name, None)
    if cls is None:
        raise ValueError(f"unknown provider: {prov_name!r}")
    default = getattr(cls, "DEFAULT_MODEL", None)
    if not isinstance(default, str) or not default:
        raise ValueError(f"provider {prov_name!r} has no DEFAULT_MODEL")
    return default


def _schedule_close(model: types.model.Model) -> None:
    """Fire-and-forget async teardown for a swapped-out model.

    CLI-style providers (``AnthropicCLI``, ``GoogleCLI``) own subprocess
    pools via ``HotSpare`` and define ``async def close()``; API-key
    providers don't. Schedule the teardown on the running loop so the
    prior subprocess and its warming-spare task don't outlive the swap.
    No-op when no event loop is running (e.g. ``Agent.resume`` before
    ``serve_forever``): the model hasn't been used yet so there is
    nothing to close.
    """
    close = getattr(model, "close", None)
    if not callable(close):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    coro = close()
    if not asyncio.iscoroutine(coro):
        return
    task = loop.create_task(coro)
    task.add_done_callback(
        types.exceptions.log_task_exception(logger, "swapped-out model close failed"),
    )


class _AgentModel:
    """Bridges rich provider ``Model`` to runtime ``Model`` protocol.

    Args:
      inner: Rich provider model to wrap.
      agent: Owning ``Agent``; supplies request knobs and cost tracker.

    """

    def __init__(self, inner: types.model.Model, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    def set_inner(self, inner: types.model.Model) -> None:
        """Swap the wrapped model. Used by ``Agent.swap_model``.

        Args:
          inner: New rich provider model to wrap.

        """
        self._inner = inner

    async def stream(
        self,
        history: list[types.history.HistoryEntry],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> types.history.AssistantMessage:
        """Stream a response with retry + context-overflow recovery.

        Builds a ``types.model.ModelRequest`` from agent state, runs
        ``send_with_retry`` with persistent-mode + overflow recovery,
        records cost out-of-band, returns the final assistant message.

        Args:
          history: Conversation history (microcompacted in place first).
          system: System prompt to send.
          tools: Runtime-side tools forwarded by the engine.
          on_text: Callback for each streamed text chunk.
          on_thinking: Callback for each streamed thinking chunk.

        Returns:
          message: Final ``types.history.AssistantMessage`` from the provider.

        Raises:
          RuntimeError: Overflow recovery failed after the retry cap.

        """
        # Microcompact in place (no event; just mutate history).
        self._agent.microcompact_history(history)

        # Proactive compaction: ask the compactor whether headroom is
        # exhausted BEFORE handing the prompt to the provider. Without
        # this gate, ``compact_now`` only runs reactively (after a
        # provider 400 classified as overflow) -- which costs a full
        # round trip and depends on the provider rejecting the request
        # at all. Some providers accept oversized prompts up to a hard
        # ceiling, so the reactive path leaves a wide window where the
        # buffer's headroom (``ContextBudget.buffer_tokens``) is paid
        # for but never spent. ``compact_now`` mutates ``runtime.history``
        # in place; the local ``history`` reference is the same list.
        if not await self._agent.compact_if_needed(history, self._inner):
            raise _context_overflow_error()

        # Re-evaluate the system spec per request so callable sections
        # (e.g. cwd-aware ``environment``) stay live after ``cd``. The
        # runtime-passed ``system`` is a one-shot snapshot from
        # construction and is intentionally ignored here.
        del system

        # Build types.model.ModelRequest from runtime-passed args + agent state.
        # The runtime hands us ``runtime.Tool`` instances (the ``_AgentTool``
        # wrappers), which expose only ``name`` / ``run``. Providers need
        # the full rich Tool surface (``description``, ``directive_schema``,
        # ``prompt``, etc.) so look each one up in the agent's
        # ``tools_map`` by name and wrap with ``BackgroundAwareTool`` so
        # the LLM-visible schema advertises the ``background`` / ``delay``
        # properties without polluting the raw tool's identity.
        rich_tools: list[types.tools.Tool] = [
            self._agent.tools_map[t.name]
            if t.name == "BackgroundTask"
            else BackgroundAwareTool(self._agent.tools_map[t.name])
            for t in tools
        ]
        rich_thinking = self._agent.thinking if self._inner.supports_thinking else None
        rich_effort = self._agent.effort if self._inner.supports_effort else None
        rich_service_tier = (
            self._agent.service_tier if self._inner.valid_service_tiers else None
        )

        for attempt in range(MAX_OVERFLOW_RECOVERY + 1):
            fresh_system = self._agent.system_prompt()
            request = types.model.ModelRequest(
                messages=list(history),
                system=fresh_system or None,
                tools=rich_tools or None,
                max_response_tokens=self._agent.max_response_tokens,
                thinking=rich_thinking,
                effort=rich_effort,
                cache_ttl=self._agent.cache_ttl,
                service_tier=rich_service_tier,
            )
            try:
                response = await send_with_retry(
                    self._inner,
                    request,
                    on_text=on_text,
                    on_thinking=on_thinking,
                    max_attempts=self._agent.max_attempts,
                    persistent_retry=self._agent.persistent_retry,
                    publish_recoverable=lambda text: logger.info(
                        "recoverable: %s", text
                    ),
                    on_discarded_response=self._agent.record_response,
                )
            except Exception as exc:
                # Catch any exception the provider classifies as
                # context overflow, not just ``PromptTooLongError``.
                # Provider-side normalization can slip (e.g. unusual
                # HTTP status carrying overflow body text); the
                # canonical signal is ``is_context_overflow``.
                if not self._inner.is_context_overflow(exc):
                    raise
                if attempt >= MAX_OVERFLOW_RECOVERY:
                    raise _context_overflow_error() from exc
                logger.info("Context overflow recovery attempt %d", attempt)
                # Short-circuit when compaction itself failed -- looping
                # to retry the model on unchanged (or slightly longer)
                # history burns the retry budget on the same 400 (the
                # BUGS34 regression: three identical "Compaction failed"
                # lines followed by a cryptic RuntimeError).
                if not await self._agent.compact_now():
                    raise _context_overflow_error() from exc
                continue
            # Record cost out-of-band; the runtime's
            # types.runtime.ModelResponseComplete event has tokens=0 by
            # design (the runtime can't see tokens). Returning inside
            # the loop avoids the dead ``response is not None`` assert
            # the prior shape needed for type narrowing.
            self._agent.record_response(response)
            return response.message
        # Unreachable: the loop either ``return``s on success or raises
        # on the final attempt. Kept as a typed safety net under -O
        # where any ``assert`` would be elided.
        raise RuntimeError(
            "unreachable: overflow recovery loop must return or raise",
        )


class _AgentTool:
    """Bridges raw rich ``Tool`` to the runtime's lean ``Tool`` protocol.

    Per call, in order:

    - Pop the ``BackgroundAwareTool``-injected ``background`` / ``delay``
      keys from ``args`` so they don't reach the raw tool's schema
      validation or runtime.
    - Pre-validate ``args`` against the raw tool's
      ``directive_schema`` (required fields, ``additionalProperties``
      bound). Validation errors surface as
      ``types.history.ToolResult(is_error=True)`` with an ``InputValidationError:``
      header carrying the recovery hint so the model can self-correct
      without looping.
    - Publish ``types.runtime.ToolLabel`` for the REPL renderer.
    - When ``background`` / ``delay`` is set: spawn a detached task that
      will eventually post ``types.runtime.DetachedResult`` to the runtime inbox; the
      synchronous return is a ``[Running in background: <name>]``
      placeholder. The runtime splices the real result into that
      placeholder slot once the bg task completes.
    - Otherwise: await the raw tool's ``run``, then post-process
      (empty-result marker, oversized-content disk offload).

    Args:
      inner: Raw rich tool (no schema injection); validated against
          its own ``directive_schema``.
      agent: Owning ``Agent`` used for publishing events, registering
          background tasks, and looking up session-dir / budget for
          result persistence.

    """

    def __init__(self, inner: types.tools.Tool, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    @property
    def name(self) -> str:
        """Forward to the wrapped tool's name."""
        return self._inner.name

    async def run(self, args: Mapping[str, object]) -> types.history.ToolResult:
        """Validate, publish ``types.runtime.ToolLabel``, dispatch, post-process.

        Args:
          args: Directive arguments parsed by the runtime; includes the
              ``BackgroundAwareTool``-injected ``background`` / ``delay``
              keys when the model requested asynchronous execution.

        Returns:
          result: Tool result, an input-validation error, or a
              ``[Running in background: <name>]`` placeholder.

        """
        call_id = agent_runtime.current_call_id_var.get("")
        bg_requested, delay_sec, clean_args = split_bg_args(args)
        validation_error = _validate_input(
            self._inner.name, self._inner.directive_schema, clean_args
        )
        if validation_error is not None:
            return types.history.ToolResult(
                call_id=call_id,
                content=validation_error,
                is_error=True,
            )
        self._agent.runtime.publish(
            types.runtime.ToolLabel(
                call_id=call_id, text=self._inner.summary(clean_args)
            ),
        )
        if bg_requested or delay_sec > 0:
            task = asyncio.create_task(
                self._run_bg(call_id, clean_args, delay_sec),
            )
            task.add_done_callback(
                types.exceptions.log_task_exception(
                    logger, f"background tool {self._inner.name!r} crashed"
                ),
            )
            self._agent.register_background(
                call_id,
                BackgroundTaskEntry(
                    task=task,
                    tool_name=self._inner.name,
                    queue_id=call_id,
                    started=time.time(),
                    delay_sec=delay_sec,
                    kind="tool",
                ),
            )
            return types.history.ToolResult(
                call_id=call_id,
                content=f"[Running in background: {self._inner.name}]",
            )
        result = await self._inner.run(clean_args)
        return post_process_result(
            result,
            self._inner.name,
            session_dir=self._agent.session_dir,
            persist_threshold=self._agent.budget.persist_threshold,
        )

    async def _run_bg(
        self,
        call_id: str,
        args: Mapping[str, object],
        delay_sec: float,
    ) -> None:
        """Background-task body: optional sleep, run inner, post result."""
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)
        try:
            result = await self._inner.run(args)
            processed = post_process_result(
                result,
                self._inner.name,
                session_dir=self._agent.session_dir,
                persist_threshold=self._agent.budget.persist_threshold,
            )
            content, is_error = processed.content, processed.is_error
        except asyncio.CancelledError:
            self._agent.cancel_background(call_id)
            raise
        except Exception as exc:  # noqa: BLE001 -- surface arbitrary tool errors
            content = f"{type(exc).__name__}: {exc}"
            is_error = True
        self._agent.runtime.inbox.push_back(
            types.runtime.DetachedResult(
                call_id=call_id, content=content, is_error=is_error
            ),
        )
        self._agent.cancel_background(call_id)


class _AgentCompactor:
    """Bridges rich ``Compactor`` to the runtime's lean ``compact`` protocol.

    Threads ``transcript_path`` / ``direction`` / ``keep_recent`` /
    ``custom_instructions`` / ``summary_pointers`` from agent state.
    Runs the post-compact enrich pipeline after the inner compactor
    returns.

    Args:
      inner: Rich compactor whose ``compact`` is delegated to.
      agent: Owning ``Agent``; supplies transcript path, budget, tools.

    """

    def __init__(self, inner: types.compactor.Compactor, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Delegate to the inner compactor.

        The runtime's lean ``Compactor`` Protocol does not include
        ``should_compact`` (the runtime never asks; compaction is
        explicitly driven by ``Compact`` / ``Recompact`` events). The
        Agent layer's :class:`_AgentModel` invokes this on the wrapper
        directly to gate proactive compaction ahead of each provider
        call.
        """
        return await self._inner.should_compact(
            input_tokens=input_tokens,
            max_request_tokens=max_request_tokens,
            max_response_tokens=max_response_tokens,
        )

    async def compact(
        self,
        history: list[types.history.HistoryEntry],
        model: agent_runtime.Model,
        args: str = "",
    ) -> list[types.history.HistoryEntry]:
        """Run the rich compactor and apply post-compact enrichment.

        Args:
          history: Full history snapshot to compact.
          model: Runtime-side model (ignored; rich model used directly).
          args: Free-form compaction instructions forwarded to the compactor.

        Returns:
          summary: Compacted history; always ends with a ``types.history.UserMessage``
              or ``types.history.ToolResult`` so the runtime gate fires next round.

        """
        del model  # _AgentCompactor uses the agent's rich model directly
        n = self._agent.compaction_state.compact_count
        transcript_path = (
            self._agent.session_dir / f"pre_compact_{n}.jsonl"
            if self._agent.session_dir is not None
            else None
        )
        if transcript_path is not None:
            write_pre_compact_transcript(transcript_path, history)
        result = await self._inner.compact(
            history=history,
            model=self._agent.model,
            transcript_path=transcript_path,
            direction="from",
            keep_recent=self._agent.budget.keep_recent_on_compact,
            custom_instructions=args or None,
            summary_pointers=self._agent.compaction_state.summary_pointers or None,
        )

        # Post-compact enrich. ``compact_count`` is incremented AFTER so
        # ``pre_compact_{n}.jsonl`` and ``summary_{n}.md`` agree on ``n``.
        try:
            used = estimate_total_tokens(
                self._agent.system_prompt(), result, self._agent.model
            )
            headroom = (
                self._agent.max_response_tokens + self._agent.budget.buffer_tokens
            )
            await post_compact_enrich(
                result=result,
                history=result,
                state=self._agent.compaction_state,
                session_dir=self._agent.session_dir,
                tool_state=self._agent.tool_state,
                budget=self._agent.budget,
                tools=self._agent.tools_map,
                background_tasks=self._agent.background,
                estimate_tokens=self._agent.max_request_tokens - used,
                headroom=headroom,
            )
        except Exception as exc:  # noqa: BLE001 -- post_compact_enrich calls the model; catch-all routes UserFacingError to warning, others to exception
            types.exceptions.log_exception_or_warning(
                logger, "post_compact_enrich failed; continuing", exc
            )
        self._agent.compaction_state.compact_count += 1

        # The runtime's gate needs the last entry to be types.history.UserMessage or
        # types.history.ToolResult so the next iteration calls the model. If the
        # compactor returned an empty list or ended elsewhere, append
        # a continuation user message.
        if not result or not isinstance(
            result[-1], (types.history.UserMessage, types.history.ToolResult)
        ):
            result = [*result, types.history.UserMessage(text="[continuation]")]
        return result

    def maintain(self, history: list[types.history.HistoryEntry]) -> None:
        """Forward microcompaction to the inner compactor.

        Threads ``cost_tracker.last_response_time`` so the inner
        compactor can gate microcompaction on cache-cold likelihood.

        Args:
          history: History list to mutate in place.

        """
        self._inner.maintain(
            history,
            self._agent.tools_map,
            last_response_time=self._agent.cost_tracker.last_response_time,
        )


_INPUT_VALIDATION_PREFIX = "InputValidationError:"
_TOOL_INPUT_RECOVERY_HINT = (
    "This tool call was not executed because its JSON directive was missing or "
    "misstated required fields. Do not repeat the same empty or incomplete call. "
    "Either retry this tool with the required fields, choose a different tool "
    "that fits the task, or explain why the required value is unavailable."
)


def _validate_input(
    tool_name: str,
    schema: JSON,
    args: Mapping[str, object],
) -> str | None:
    """Pre-check ``args`` against ``schema``; return an error or ``None``.

    Catches the two failure modes that wedge the model on retry: missing
    ``required`` fields (typically a truncated ``stop_reason=max_tokens``
    tool_use) and unexpected fields (model hallucinated a key that
    doesn't exist).

    Args:
      tool_name: Tool name -- used in the error header.
      schema: The tool's frozen ``directive_schema``.
      args: Directive args parsed from the model output.

    Returns:
      error: Multi-line error text starting with
          ``InputValidationError:`` when validation fails, ``None`` on
          well-formed input.

    """
    required_raw = schema.get("required")
    required: list[str] = (
        [k for k in required_raw if isinstance(k, str)]
        if isinstance(required_raw, (list, tuple))
        else []
    )
    props_raw = schema.get("properties")
    accepted: list[str] = (
        [str(k) for k in props_raw] if isinstance(props_raw, Mapping) else []
    )
    missing = [k for k in required if k not in args]
    unexpected: list[str] = (
        [k for k in args if k not in set(accepted)]
        if schema.get("additionalProperties") is False
        else []
    )
    if not missing and not unexpected:
        return None
    issues: list[str] = [f"The required parameter `{k}` is missing." for k in missing]
    issues.extend(f"Unexpected parameter `{k}`." for k in unexpected)
    plural = "issues" if len(issues) > 1 else "issue"
    parts = [
        f"{_INPUT_VALIDATION_PREFIX} {tool_name} failed due to the following {plural}:",
        *issues,
    ]
    if required:
        keys = ", ".join(f"`{k}`" for k in required)
        parts.append(f"\n{tool_name} requires: {keys}.")
    if unexpected and accepted:
        keys = ", ".join(f"`{k}`" for k in accepted)
        parts.append(f"{tool_name} accepts: {keys}.")
    parts.append(f"\n{_TOOL_INPUT_RECOVERY_HINT}")
    return "\n".join(parts)
