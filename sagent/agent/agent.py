"""Agent: composition over :class:`AgentRuntime`.

Owns three wrappers and a small set of observers:

- :class:`_AgentModel` bridges the rich provider ``Model`` interface
  (``buffer`` / ``stream`` returning ``ModelResponse``) to the
  runtime's lean ``stream(history, system, tools, on_text,
  on_thinking) -> AssistantMessage`` protocol. Runs the retry loop
  with persistent-retry and overflow recovery. Records cost
  out-of-band on :attr:`Agent.cost_tracker`.

- :class:`_AgentTool` bridges a rich ``Tool`` (with metadata,
  ``summary`` / ``prompt``) to the runtime's
  minimal ``run(args) -> ToolResult`` protocol. Emits ``ToolLabel``
  before running, validates the directive against the tool's
  ``directive_schema``, postprocesses the result (empty-marker,
  oversized persist).

- :class:`_AgentCompactor` bridges the rich ``SummaryCompactor``
  interface to the runtime's lean ``compact(history, model, args)``
  protocol. Runs the post-compact enrich pipeline (file reattach,
  status injection, tool restore).

Observers track cost, activity (call timing, streamed chars), tool
registry (cohort id → tool name), and persistence (``SaveSession`` →
session.jsonl append). Budget caps are enforced at their own gates:
``_before_tool_spawn`` for tool-call rounds, ``_bill`` for USD.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn

import asyncio
import contextlib
import dataclasses
import inspect
import itertools
import json
import logging
import time
import uuid

from sagent import agents_md
from sagent.agent.background import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
    backgroundable,
    bg_augmented_schema,
    split_bg_args,
)
from sagent.agent.cache_waste import detect_cache_miss
from sagent.agent.compaction import (
    CompactionState,
    post_compact_enrich,
)
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.result_storage import post_process_result
from sagent.agent.retry import send_with_retry, service_error_snapshot
from sagent.agent.runtime import (
    AgentRuntime,
    GatedDeque,
    current_call_id_var,
    widen_barrier_mask,
)
from sagent.agent.session_io import (
    SessionMeta,
    install_session_persistence,
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
from sagent.catalog.table import UnknownModelError, UnsupportedTagError
from sagent.compaction.history import (
    MAX_CONSECUTIVE_COMPACT_FAILURES,
    estimate_entry_tokens,
)
from sagent.compaction.scrunch import (
    ScrunchTooLargeError,
    scrunch_to_fit,
)
from sagent.lib import last_models
from sagent.lib.tool_validation import validate_tool_input
from sagent.providers import providers
from sagent.providers.lib.errors import PolicyBlockedError
from sagent.request_materialization import materialize_request

# The one module imported whole rather than by name: this file touches 36
# of its symbols, and the flat list buried every other import here.
from sagent.types import runtime
from sagent.types.cost import TokenCost, TokenCount
from sagent.types.exceptions import (
    BudgetExhaustedError,
    ContextOverflowError,
    log_exception_or_warning,
    log_task_exception,
)
from sagent.types.model import (
    Model,
    ModelRecipe,
    ModelRequest,
    ModelResponse,
    ModelTerminationError,
    RequestTooLargeError,
    StreamInterruptedError,
    UsageSnapshot,
)
from sagent.types.providers import (
    AuthReloadable,
    ModelResolver,
    Provider,
)
from sagent.types.settings import (
    AgentSettings,
    default_buffer_tokens,
)
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    merge_mask_ranges,
    splice_safe_repair,
    unpaired_call_ids,
)
from sagent.types.tools import (
    DEFAULT_MAX_RESULT_CHARS,
    MAX_RESULT_TOKENS,
    MAX_ROUND_RESULT_CHARS,
    ResultBounded,
    Tool,
    ToolResultPolicy,
)


if TYPE_CHECKING:
    import contextvars

    from sagent.types.capability import (
        ModelCapability,
        ModelLimits,
        ModelSettings,
    )
    from sagent.types.compactor import (
        Compactor,
    )


logger = logging.getLogger(__name__)

SystemPromptArg = str | Callable[[], str]
"""System-prompt spec. ``str`` is literal; ``Callable[[], str]`` is
re-invoked per request so cwd-aware sections stay live after ``cd``."""

ERROR_MAX_TOOL_CALL_ROUNDS: Final = "error:max_tool_call_rounds"
REFUSED_RESULT: Final = "[withheld: the model refused the request carrying this result]"
MAX_OVERFLOW_RECOVERY = 3  # house-ignore[globals] -- Overflow-recovery retry count.

# Utilization fraction at which a rate-limit window earns a UI advisory.
# Matches Anthropic's own ``surpassed-threshold`` warning point. A separate,
# lower clear point gives hysteresis so a window oscillating around the warn
# line does not re-fire the advisory every response.
_USAGE_WARN_FRACTION = 0.75  # house-ignore[globals] -- UI advisory warn threshold.
_USAGE_CLEAR_FRACTION = 0.60  # house-ignore[globals] -- UI advisory clear threshold.


# Mirrors the ``max_request_tokens`` / ``max_response_tokens`` setters, which are the
# only other way to change these. Raises here rather than clamping: a caller who named a
# window the model cannot serve has a wrong belief, and silently shrinking it hides that
# until the numbers stop adding up somewhere else.
def _reject_budget_over_model(budget: AgentSettings, model: Model) -> None:
    """Reject an explicit budget that exceeds the model's own limits."""
    limits = model.limits
    for name, requested, ceiling in (
        ("max_request_tokens", budget.max_request_tokens, limits.max_request_tokens),
        ("max_response_tokens", budget.max_response_tokens, limits.max_response_tokens),
    ):
        if ceiling > 0 and requested is not None and requested > ceiling:
            raise ValueError(f"budget {name}={requested:,} exceeds model's {ceiling:,}")


# ``budget`` and the three knobs are two spellings of the same choices, and taking one
# silently dropped the other -- a cap the caller named went unenforced. ``AgentSettings``
# owns the defaults and their validation, so ``None`` here means "not named".
def _settings_from_knobs(
    budget: AgentSettings | None,
    *,
    max_attempts: int | None,
    max_tool_call_rounds: int | None,
    max_budget_usd: float | None,
) -> AgentSettings:
    """Return ``budget``, or settings built from the knobs; never both."""
    knobs = {
        "max_attempts": max_attempts,
        "max_tool_call_rounds": max_tool_call_rounds,
        "max_budget_usd": max_budget_usd,
    }
    named = sorted(name for name, value in knobs.items() if value is not None)
    if budget is None:
        settings = AgentSettings(
            max_tool_call_rounds=max_tool_call_rounds,
            max_budget_usd=max_budget_usd,
        )
        if max_attempts is None:
            return settings
        return dataclasses.replace(settings, max_attempts=max_attempts)
    if named:
        raise ValueError(f"pass {', '.join(named)} inside budget, not beside it")
    return budget


def _reject_bad_system_arg(system: object) -> None:
    """Reject non-system-prompt constructor values before runtime setup."""
    if not isinstance(system, str) and not callable(system):
        raise TypeError(
            f"system must be str or Callable[[], str], got {type(system).__name__}",
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class ActivityTracker:
    """Lifecycle counters for the status pane."""

    elapsed_seconds: float = 0.0
    """Cumulative wall-clock seconds spent in active model calls."""

    current_call_start: float = 0.0
    """Event-loop time when the current model call started (``0`` when idle)."""

    current_compact_start: float = 0.0
    """Event-loop time when the current compaction started (``0`` when idle)."""

    live_response_text: str = ""
    """Response text streamed so far in the current call. Tokenized as a
    whole (not per chunk) by readers so sub-token chunk boundaries don't
    floor to zero; the model's estimator is applied to the full string."""

    active: bool = False
    """True between ``types.runtime.ModelCallStarted`` and ``types.runtime.ModelResponseComplete`` /
    ``types.runtime.ModelIdle`` for the current call."""

    num_tool_call_rounds: int = 0
    """Cumulative count of responses that included tool calls."""


# Retained tool-call identities. Sized so a long session cannot grow the
# registry without bound while still outliving any plausible detached
# call: one entry is ~100 bytes, and 10k covers far more rounds than a
# session sustains between compactions.
_TOOL_REGISTRY_MAX: Final = 10_000


class Agent:
    """Conversation agent: composes :class:`AgentRuntime` with wrappers.

    Args:
      model: Rich provider model the agent calls.
      model_recipe: Optional spec recording how the model was built.
      system: Static system prompt string or a no-arg factory rebuilt
          each request.
      tools: Rich tools advertised to the model.
      compactor: Rich compactor used on ``types.runtime.Compact`` / ``types.runtime.Recompact`` and
          on overflow recovery.
      session_dir: Directory for session persistence and pre-compact
          transcripts; ``None`` disables both.
      budget: Explicit context and execution choices; omitted values derive live
          from the model. Excludes ``max_attempts``, ``max_tool_call_rounds``,
          and ``max_budget_usd``, which it already carries.
      max_attempts: Retry attempts inside ``send_with_retry``; ``None`` takes
          the ``AgentSettings`` default.
      name: Human-readable agent label.
      description: Agent description for parent agents and the UI.
      max_tool_call_rounds: Cap on tool-call rounds before the agent
          forces ``types.runtime.ModelResponseError``.
      max_budget_usd: Hard USD cap; ``record_response`` raises when hit.
      persistent_retry: Enable persistent-mode backoff for 429/529.
      frozen_system: Skip per-tool prompt contributions and the detached-
          activity note in :meth:`_build_system`, returning ``system``
          verbatim instead. Trades the liveness ``_build_system`` normally
          gives (fresh depth/budget reminders, newly-bundled tools, the
          detached note) for a byte-stable prompt a provider's prefix cache
          can reuse across otherwise-divergent agents. Set by ``AgentSpawn``
          hot spawns (see ``tools.agent_spawn``); not meant to be toggled by
          general callers mid-session.
      allow_background: Whether a tool call may carry ``background`` / ``delay``.
          A host whose turn must end with its reply sets it False: tools are
          then offered without those keys, and a call that sends them runs in
          the foreground, so no timer or background job can hold the turn open.
      tool_results: Per-result and aggregate tool-result limits; ``None`` derives
          them from the input window, which on a million-token window lets one
          result run to 250k tokens before it is moved to disk.
      owned_provider: The provider ``model`` came from, when this agent built it
          and so must ``close_sdk`` it on swap-away and teardown.

    Side effects:
      Constructing with a non-``None`` ``model_recipe`` (and
      :meth:`swap_model` with a non-``None`` ``spec``) writes
      the sagent ``last-models.json`` via ``last_models.record`` so the
      ``/model`` slash command can resume the same model_id when the
      user changes provider without naming a model. The write is
      ``fcntl.flock`` serialized so concurrent agent processes don't
      clobber each other.

    """

    def __init__(
        self,
        *,
        model: Model,
        model_recipe: ModelRecipe | None = None,
        system: SystemPromptArg = "",
        tools: list[Tool] | None = None,
        compactor: Compactor | None = None,
        session_dir: str | Path | None = None,
        budget: AgentSettings | None = None,
        max_attempts: int | None = None,
        name: str = "Agent",
        description: str = "An AI agent.",
        max_tool_call_rounds: int | None = None,
        max_budget_usd: float | None = None,
        persistent_retry: bool = False,
        frozen_system: bool = False,
        allow_background: bool = True,
        tool_results: ToolResultPolicy | None = None,
        owned_provider: Provider | None = None,
    ) -> None:
        _reject_bad_system_arg(system)
        budget = _settings_from_knobs(
            budget,
            max_attempts=max_attempts,
            max_tool_call_rounds=max_tool_call_rounds,
            max_budget_usd=max_budget_usd,
        )
        self.name = name
        self.description = description
        self.model = model
        self._owned_provider = owned_provider
        self.model_recipe = model_recipe
        if model_recipe is not None:
            last_models.record(model_recipe.provider, model_recipe.model_id)
        self._base_system_spec: SystemPromptArg = system
        self._system_spec: SystemPromptArg = system
        self._frozen_system = frozen_system
        self.allow_background = allow_background
        self._tools_list: list[Tool] = list(tools or [])
        self.compactor = compactor
        # An explicit budget goes through the same ceiling check the
        # setters apply. Without it, construction was the one way past
        # them: ``AgentSettings`` validates only non-negativity, so a
        # budget above the model's window was accepted here and shipped
        # on the first request, where the provider rejects it.
        _reject_budget_over_model(budget, model)
        self._budget = budget
        self._tool_results_override: ToolResultPolicy | None = tool_results
        self.persistent_retry = persistent_retry
        self.last_compact_error: Exception | None = None
        # Cache-inclusive input-token count from the most recent response
        # (input + cache_creation + cache_read), used as the anchor of the
        # proactive compaction trigger -- provider ground truth rather than a
        # client estimate. ``0`` until the first response, when the gate falls
        # back to a client-side estimate of the next request.
        self._last_input_tokens: int = 0
        self._last_measured_history: tuple[runtime.ModelContextEvent, ...] = ()

        self.cost_tracker = CostTracker()
        self.activity = ActivityTracker()
        self.tool_state = ToolState()
        self.compaction_state = CompactionState()
        self._bg: dict[str, BackgroundTaskEntry] = {}
        self._round_result_chars = 0
        self.observers: list[Callable[[runtime.RuntimeEvent], None]] = []
        # ``_tool_registry`` maps cohort call_id → (tool_name, started_at)
        # so ``background`` can synthesize ``BackgroundTaskEntry`` rows
        # for detached cohort members.
        self._tool_registry: dict[str, tuple[str, float]] = {}
        self._job_counter = itertools.count(1)
        self._job_ids_by_call_id: dict[str, str] = {}
        self._call_ids_by_job_id: dict[str, str] = {}
        self.session_dir: Path | None = (
            Path(session_dir) if session_dir is not None else None
        )
        self._session_id = (
            self.session_dir.name
            if self.session_dir is not None
            else uuid.uuid4().hex[:8]
        )
        self._status: str = ""
        self._checkpoint: runtime.Checkpoint | None = None
        self.checkpoint_updated_at: float = 0.0
        self.checkpoint_restored: bool = False
        self.child_checkpoints: dict[str, runtime.CheckpointChanged] = {}
        self.restored_child_checkpoints: set[str] = set()
        # Lifecycle policy set once at spawn. ``"oneshot"`` stops after the
        # first post-work idle (the drive-until-first-idle usage); ``"serviced"``
        # keeps its ``serve_forever`` loop alive, servicing its inbox until an
        # explicit shutdown. Both are addressable in ``agent_registry`` for
        # their whole life.
        self._lifecycle: Literal["oneshot", "serviced"] = "oneshot"
        # Per-agent cost accumulator for ``max_budget_usd`` enforcement. Cost
        # rolls up to the root ``cost_root_var`` tracker for the tree total;
        # this plain float tracks only THIS agent's own spend so a subagent's
        # budget cap is checked against its own calls, not the tree.
        self._own_spend = TokenCost()
        # True for agents spawned by ``AgentSpawn`` (both lifecycles). The root
        # REPL/CLI agent leaves it False so ``/send`` / ``/halt all`` can
        # target every subagent while never routing to the root or to self.
        self._is_subagent: bool = False
        self._shutting_down: bool = False
        self._close_task: asyncio.Task[None] | None = None
        self._driver_claim: object | None = None
        # Live ``serve_forever`` task from ``drive_until_first_idle``, kept so
        # a one-shot caller can await the loop to completion after shutdown.
        self._drive_task: asyncio.Task[None] | None = None
        # Rate-limit advisories already shown, keyed by ``(label, blocked)`` so
        # a window escalating warn -> blocked still re-fires; cleared when the
        # window drops below the hysteresis clear point.
        self._usage_warned: set[tuple[str, bool]] = set()

        # Wrappers. ``_agent_compactor`` is None when no rich compactor
        # was supplied (the runtime is satisfied with a None compactor).
        self._agent_model = _AgentModel(model, self)
        # ``_tools_map`` holds the RAW rich tool keyed by name so
        # isinstance / Protocol checks (CompactRestorable, Slack, ...) at
        # consumer sites pass through. The schema-augmenting
        # ``BackgroundAwareTool`` wrapper is applied per request in
        # ``_AgentModel.stream`` when building the provider tool list.
        self._tools_map: dict[str, Tool] = {}
        self._tools_version: int = 0
        self._live_tools_cache: tuple[int, list[Tool]] | None = None
        self.tool_gate: frozenset[str] | None = None
        """Names of the only tools the next model requests offer; ``None`` offers all.
        A host sets it for a call and clears it after; tool calls still run by name."""
        agent_tools: list[_AgentTool] = []
        for t in self._tools_list:
            self._tools_map[t.name] = t
            agent_tools.append(_AgentTool(t, self))
        self._agent_compactor = (
            _AgentCompactor(compactor, self) if compactor is not None else None
        )
        self.runtime = AgentRuntime(
            model=self._agent_model,
            tools=agent_tools,
            compactor=self._agent_compactor,
            session_id=self._session_id,
        )

        self.runtime.before_tool_spawn = self._before_tool_spawn
        # Let the runtime's idle gate see Agent-layer background jobs: a live
        # turn-scoped ``_bg`` tool is unfinished work, so ``AgentIdle`` must
        # not fire (and ``Agent.run`` must not reap) until it drains. Only
        # ``kind="tool"`` non-hidden, still-running jobs count -- mirroring
        # ``_should_cancel_background``'s taxonomy. Subagents and hidden
        # infra (REPL pump, watchdogs) live indefinitely by design;
        # counting them would wedge a one-shot ``Agent.run`` forever.
        self.runtime.has_pending_background = self._has_pending_background
        for fn in (
            self._track_activity,
            self._track_tool_registry,
            self._track_compaction,
            self._track_child_checkpoint,
        ):
            self.runtime.observers.append(fn)
        # Persistence is an Agent-level concern, not a caller concern.
        # Any Agent constructed with a ``session_dir`` -- root agents from
        # the CLI, child agents from ``AgentSpawn``, slack/v1 worker
        # agents -- self-installs its own ``session.jsonl`` observer.
        # ``resume()`` re-baselines via ``self._rebaseline_persistence``
        # so resumed tape records don't get rewritten to disk.
        self._rebaseline_persistence: Callable[[], None] | None = (
            install_session_persistence(self, self.session_dir)
            if self.session_dir is not None
            else None
        )

    # -- Properties / config surface ----------------------------------

    @property
    def settings(self) -> AgentSettings:
        """Return unresolved agent choices; ``None`` derives from the model."""
        return self._budget

    @property
    def budget(self) -> AgentSettings:
        """Resolve explicit choices against the active model limits."""
        limits = self.model.limits
        request = self._budget.max_request_tokens
        if request is None:
            request = limits.max_request_tokens
            if request <= 0:
                raise ValueError(
                    "model max_request_tokens is unknown; set an explicit AgentSettings cap",
                )
        elif limits.max_request_tokens > 0:
            request = min(request, limits.max_request_tokens)
        response = self._budget.max_response_tokens
        if response is None:
            response = limits.max_response_tokens
            if response <= 0:
                raise ValueError(
                    "model max_response_tokens is unknown; set an explicit AgentSettings cap",
                )
        elif limits.max_response_tokens > 0:
            response = min(response, limits.max_response_tokens)
        buffer = self._budget.buffer_tokens
        if buffer is None:
            buffer = default_buffer_tokens(request)
        return dataclasses.replace(
            self._budget,
            max_request_tokens=request,
            max_response_tokens=response,
            buffer_tokens=buffer,
        )

    @property
    def tool_results(self) -> ToolResultPolicy:
        """Derive the aggregate budget; a compactor owns aggregate context pressure.

        Returns:
          policy: Per-result limits and the applicable aggregate request budget.

        """
        override = self._tool_results_override
        if override is not None:
            return override
        policy = ToolResultPolicy.from_settings(self.budget)
        if self._agent_compactor is not None:
            # A newest-first request budget rewrites old results on every
            # turn once half the window fills, invalidating the prefix cache.
            # Keep stored results unchanged until the compactor replaces
            # context. Per-result disk off-loading still bounds fresh results.
            return dataclasses.replace(policy, message_budget_tokens=0)
        return policy

    @tool_results.setter
    def tool_results(self, policy: ToolResultPolicy) -> None:
        """Replace the per-result and aggregate limits derived from the budget."""
        self._tool_results_override = policy

    @property
    def max_request_bytes(self) -> int:
        """The active model's request-body byte ceiling (wire limit)."""
        return self.model.limits.max_request_bytes

    @property
    def max_result_tokens(self) -> int:
        """Return the tokens a self-bounding tool emits before it pages.

        Returns:
          tokens: The host's per-result cap, else :data:`MAX_RESULT_TOKENS`.

        """
        return self.tool_results.persist_tokens or MAX_RESULT_TOKENS

    def result_room_tokens(self) -> int:
        """Return the tokens a fresh tool result may add and still fit the context.

        A result over this is persisted when it arrives, so it never forces the
        request past :meth:`Compactor.largest_context` on its own.

        Returns:
          tokens: Room left before the largest request; at least ``1``.

        """
        budget = self.budget
        largest = (
            self._agent_compactor.largest_context(budget)
            if self._agent_compactor is not None
            else (budget.max_request_tokens or 0)
            - (budget.max_response_tokens or 0)
            - (budget.buffer_tokens or 0)
        )
        used = self._estimated_request_tokens(
            self.runtime.context().messages,
            self.model,
        )
        return max(1, largest - used)

    # Decided once, as the result arrives, and stored on the tape that way: a
    # later rewrite would change a prefix the provider already cached.
    def bound_result(
        self,
        result: runtime.ToolResult,
        *,
        tool: Tool,
    ) -> runtime.ToolResult:
        """Persist ``result`` past its tool's cap, the round's, or the context's.

        Args:
          result: Fresh result from ``tool``.
          tool: The tool that produced it; its ``max_result_chars`` applies.

        Returns:
          bounded: ``result``, a persisted preview of it, or, for a tool that
              is never persisted, an error asking for a smaller page.

        """
        declared = (
            tool.max_result_chars
            if isinstance(tool, ResultBounded)
            else DEFAULT_MAX_RESULT_CHARS
        )
        host = self.tool_results.persist_tokens
        room = self.result_room_tokens()
        if declared > 0:
            max_chars = min(
                declared,
                DEFAULT_MAX_RESULT_CHARS,
                max(1, MAX_ROUND_RESULT_CHARS - self._round_result_chars),
            )
            persist_tokens = min(room, host) if host > 0 else room
        else:
            # Read pages itself; persisting it hands back a path that only
            # Read can open, so the next read would spill again. A page that
            # cannot fit is refused instead, as Claude Code's Read does.
            tokens = self.approx_text_tokens(result.content)
            if tokens > room:
                return dataclasses.replace(
                    result,
                    content=(
                        f"Result ({tokens:,} tokens) exceeds the {room:,} tokens"
                        " left in context. Pass offset and limit to read a smaller"
                        " part."
                    ),
                    is_error=True,
                    attachments=(),
                )
            max_chars, persist_tokens = 0, 0
        processed = post_process_result(
            result,
            tool.name,
            session_dir=self.session_dir,
            persist_tokens=persist_tokens,
            max_chars=max_chars,
        )
        self._round_result_chars += len(processed.content)
        return processed

    def approx_text_tokens(self, text: str) -> int:
        """Delegate to the active model's tokenizer.

        Tools reach this through ``current_agent_var`` to bound their own
        output; routing through the model means a provider with a real
        tokenizer answers exactly rather than through a ratio.

        Args:
          text: Text to score.

        Returns:
          tokens: Approximate token count.

        """
        return self.model.approx_text_tokens(text)

    @property
    def max_request_tokens(self) -> int:
        """Active per-request input token budget."""
        value = self.budget.max_request_tokens
        if value is None:
            raise ValueError("resolved request budget is unset")
        return value

    @max_request_tokens.setter
    def max_request_tokens(self, value: int) -> None:
        """Set the per-request input token budget; bounded by a known model cap.

        Args:
          value: New input token budget; must not exceed a known model cap.

        Raises:
          ValueError: If ``value`` exceeds a known ``max_request_tokens``.

        """
        ceiling = self.model.limits.max_request_tokens
        if ceiling > 0 and value > ceiling:
            raise ValueError(
                f"max_request_tokens={value:,} exceeds model's {ceiling:,}",
            )
        self._budget = dataclasses.replace(self._budget, max_request_tokens=value)

    @property
    def max_response_tokens(self) -> int:
        """Active per-request response token budget."""
        value = self.budget.max_response_tokens
        if value is None:
            raise ValueError("resolved request budget is unset")
        return value

    @max_response_tokens.setter
    def max_response_tokens(self, value: int) -> None:
        """Set the response token budget; bounded by a known model cap.

        Args:
          value: New response token budget; must not exceed a known model cap.

        Raises:
          ValueError: If ``value`` exceeds a known ``max_response_tokens``.

        """
        ceiling = self.model.limits.max_response_tokens
        if ceiling > 0 and value > ceiling:
            raise ValueError(
                f"max_response_tokens={value:,} exceeds model's {ceiling:,}",
            )
        self._budget = dataclasses.replace(self._budget, max_response_tokens=value)

    @property
    def session_id(self) -> str:
        """Session directory name, or a generated id for ephemeral agents."""
        return self._session_id

    @property
    def is_serviced(self) -> bool:
        """True when this agent keeps serving its inbox after the first idle.

        ``"serviced"`` agents run ``serve_forever`` until an explicit
        shutdown; ``"oneshot"`` agents self-stop after the first post-work
        idle. Both are live, inbox-serviceable loops while running.
        """
        return self._lifecycle == "serviced"

    @property
    def is_subagent(self) -> bool:
        """True when spawned by ``AgentSpawn`` (either lifecycle)."""
        return self._is_subagent

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
        self.runtime.publish(runtime.StatusChanged(text=value))

    @property
    def checkpoint(self) -> runtime.Checkpoint | None:
        """Latest explicit report, independent of agent or external-job lifecycle."""
        return self._checkpoint

    def _track_child_checkpoint(self, event: runtime.RuntimeEvent) -> None:
        """Retain latest forwarded reports even after a child stops serving."""
        report = runtime.forwarded_checkpoint(event)
        if report is None:
            return
        label, change = report
        self.restored_child_checkpoints.discard(label)
        if change.checkpoint is None:
            self.child_checkpoints.pop(label, None)
        else:
            self.child_checkpoints[label] = change

    @checkpoint.setter
    def checkpoint(self, value: runtime.Checkpoint | None) -> None:
        """Publish changed state; a fresh report also confirms a restored snapshot."""
        if value == self._checkpoint and not self.checkpoint_restored:
            return
        self._checkpoint = value
        self.checkpoint_updated_at = time.time()
        self.checkpoint_restored = False
        self.runtime.publish(
            runtime.CheckpointChanged(
                checkpoint=value,
                observed_at=self.checkpoint_updated_at,
            ),
        )

    @property
    def history(self) -> list[runtime.ModelContextEvent]:
        """Resolved provider-facing context (read-only snapshot).

        Returns a fresh list each call. Mutations are silently lost --
        use ``runtime.append_history`` / ``append_splice`` /
        ``append_clear`` to evolve state.
        """
        return self.runtime.context().messages

    @property
    def inbox(
        self,
    ) -> GatedDeque[runtime.RuntimeEvent]:
        """The runtime's inbox."""
        return self.runtime.inbox

    @property
    def work(self) -> asyncio.Task[None] | None:
        """The currently active foreground task (model call or compaction)."""
        return self.runtime.model_call or self.runtime.compact_task

    @property
    def tools_map(self) -> Mapping[str, Tool]:
        """Read-only map of tool name → rich tool (pre-wrap)."""
        return MappingProxyType(self._tools_map)

    @property
    def tools(self) -> list[Tool]:
        """Rich tools in registration order (pre-wrap copies)."""
        return list(self._tools_map.values())

    def live_tools(self) -> list[Tool]:
        """Return the provider-visible tool surface for model requests.

        Cached against ``_tools_version`` -- wrapping every tool in
        ``BackgroundAwareTool`` per request is the hottest non-stream
        path; invalidation happens in :meth:`replace_tool` and anywhere
        else that mutates the registry. Returns a fresh list each call
        so downstream consumers may mutate without invalidating the
        cached wrappers.

        Returns:
          tools: Fresh provider-visible tool wrappers in registration order.

        """
        cached = self._live_tools_cache
        if cached is None or cached[0] != self._tools_version:
            cached = (
                self._tools_version,
                [
                    BackgroundAwareTool(tool)
                    if self.allow_background and backgroundable(tool)
                    else tool
                    for tool in self._tools_map.values()
                ],
            )
            self._live_tools_cache = cached
        gate = self.tool_gate
        return [tool for tool in cached[1] if gate is None or tool.name in gate]

    @property
    def system(self) -> str:
        """Assembled system prompt (base + per-tool contributions unless frozen)."""
        return self._build_system()

    @property
    def base_system_spec(self) -> str:
        """Base system prompt before tool or persistent IPC augmentation."""
        spec = self._base_system_spec
        return spec if isinstance(spec, str) else spec()

    @property
    def frozen_system(self) -> bool:
        """Whether :meth:`system_prompt` returns the spec verbatim (hot spawn)."""
        return self._frozen_system

    @property
    def max_attempts(self) -> int:
        """Maximum retry attempts from the owned settings."""
        return self._budget.max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int) -> None:
        self._budget = dataclasses.replace(self._budget, max_attempts=value)

    @property
    def max_tool_call_rounds(self) -> int | None:
        """Maximum tool-call rounds from the owned settings."""
        return self._budget.max_tool_call_rounds

    @max_tool_call_rounds.setter
    def max_tool_call_rounds(self, value: int | None) -> None:
        self._budget = dataclasses.replace(self._budget, max_tool_call_rounds=value)

    @property
    def max_budget_usd(self) -> float | None:
        """Maximum budget in USD, or ``None`` when uncapped."""
        return self._budget.max_budget_usd

    @max_budget_usd.setter
    def max_budget_usd(self, value: float | None) -> None:
        self._budget = dataclasses.replace(self._budget, max_budget_usd=value)

    @property
    def total_tokens(self) -> TokenCount:
        """Cumulative token counts across all recorded responses."""
        return self.cost_tracker.total

    @property
    def results_may_be_cleared(self) -> bool:
        """Whether the active model's server may drop old tool results."""
        return self.model.settings.manage_context_server_side

    @property
    def own_spend(self) -> TokenCost:
        """Cumulative spend of this agent alone; ``cost_tracker.spend`` is the tree's."""
        return self._own_spend

    @property
    def num_tool_call_rounds(self) -> int:
        """Cumulative count of responses that included tool calls."""
        return self.activity.num_tool_call_rounds

    @property
    def background(self) -> dict[str, BackgroundTaskEntry]:
        """Merged view: cohort-detached tools + explicit-bg + subagents + REPL pump."""
        self._forget_finished_job_ids()
        merged: dict[str, BackgroundTaskEntry] = {}
        for call_id, task in self.runtime.detached.items():
            name, started = self._tool_registry.get(call_id, ("?", time.time()))
            job_id = self.job_id_for_call(call_id)
            merged[job_id] = BackgroundTaskEntry(
                task=task,
                tool_name=name,
                queue_id=job_id,
                call_id=call_id,
                started=started,
                kind="detached",
            )
        merged.update(self._bg)
        return merged

    def publish(self, event: runtime.RuntimeEvent) -> None:
        """Forward an event to the runtime's observer list.

        Args:
          event: Event to deliver to every observer.

        """
        self.runtime.publish(event)

    # -- Mutation methods ---------------------------------------------

    def replace_tool(self, name: str, tool: Tool) -> None:
        """Swap the tool registered under ``name`` for ``tool``.

        Updates both the rich ``_tools_map`` (consumer-facing) and the
        runtime's ``_AgentTool`` wrapper inner so a swap propagates to
        the next ``run()`` call. Without the wrapper update the swap
        would silently no-op: the runtime invokes the ``_AgentTool`` it
        captured at construction time, not the rich tool returned by
        ``tools_map[name]``.

        Args:
          name: Registered tool name; must already exist on this agent.
          tool: New rich tool. Its ``name`` must match ``name``.

        Raises:
          KeyError: ``name`` is not registered on this agent.
          ValueError: ``tool.name`` does not match ``name``.

        """
        if name not in self._tools_map:
            raise KeyError(f"Unknown tool: {name!r}")
        if tool.name != name:
            raise ValueError(
                f"tool.name={tool.name!r} does not match registry key {name!r}",
            )
        self._tools_map[name] = tool
        self._tools_version += 1
        wrapper = self.runtime.tools_map.get(name)
        assert isinstance(wrapper, _AgentTool), (
            f"runtime tool {name!r} is not an _AgentTool wrapper"
        )
        wrapper.set_inner(tool)

    def swap_model(
        self,
        model: Model,
        *,
        spec: ModelRecipe | None = None,
        provider: Provider | None = None,
    ) -> None:
        """Replace the active model.

        Carries the outgoing selections onto the new model's settings and
        drops the ones it does not offer. Schedules ``close()`` on the
        swapped-out model so CLI providers' subprocess pools (the
        ``claude`` / ``gemini`` process plus its warming-spare task)
        don't leak past the swap, then ``close_sdk`` on the outgoing
        provider when this agent owned it.

        Args:
          model: New rich provider model.
          spec: Optional spec recording how the model was built.
          provider: The provider ``model`` came from, when this agent now owns
              it; ``None`` when the caller keeps ownership.

        """
        self.model_recipe = spec
        if spec is not None:
            last_models.record(spec.provider, spec.model_id)
        if model is self.model:
            return
        old = self.model
        old_provider = self._owned_provider
        self._owned_provider = provider
        carried = old.settings
        self.model = model
        # An account or auth change keeps the tokenizer, so the provider's
        # count still measures this history; dropping it forced the cruder
        # first-request estimate and could fire a spurious compaction.
        if model.capability.model_id != old.capability.model_id:
            self._last_input_tokens = 0
            self._last_measured_history = ()
        self._agent_model.set_inner(model)
        self.runtime.model = self._agent_model
        model.settings.adopt(carried)
        _ = _schedule_close(old, old_provider)

    def change_model(
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
    ) -> ModelRecipe:
        """Resolve, build, and queue a model swap. The high-level API.

        ``commit_model_change(prepare_model_change(...))``; a caller that must
        validate against the new model before committing uses the two halves.

        Kwarg semantics: each defaults to ``None`` meaning "inherit from
        the current ``model_recipe``." On a cross-provider change, a missing
        inherited named account retries the destination provider's default
        credentials. Explicit accounts and same-provider changes remain strict.
        Setting ``account`` to the default backend account (literal ``None``)
        is otherwise not expressible via this API -- construct a
        ``types.model.ModelRecipe`` and call :meth:`swap_model` directly.

        Cross-provider resolution when ``model_id`` is omitted:
        1. Prefer the current model id when the new provider's catalog
           knows it (same vendor, different auth subclass).
        2. Else use the last model recorded for the new provider in
           the sagent ``last-models.json``.
        3. Else fall back to the new provider's ``default`` catalog row.

        Queues a :class:`types.runtime.ModelSwitch` through the runtime inbox so any
        in-flight model call finishes against the OLD model (cost
        attribution, retry state) before the new model goes live.

        Args:
          provider: New provider class name, e.g. ``"AnthropicCLI"``.
          auth: New auth-method suffix.
          model_id: New provider-specific model id. Option tags ride
              along (e.g. ``opus-4.8+200k``); the provider's
              ``model()`` validates them.
          account: New credential account override.

        Returns:
          target: Resolved target spec (the spec the swap will land on).

        Raises:
          ValueError: ``model_recipe`` is unset, the resolved provider is
              unknown, or the resolved model id is rejected by the
              provider's catalog.

        """
        change = self.prepare_model_change(
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
        )
        self.commit_model_change(change)
        return change.recipe

    def prepare_model_change(
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
    ) -> ModelChange:
        """Resolve and build a model swap without queuing it.

        Same resolution as :meth:`change_model`. The caller owns the result:
        it either passes it to :meth:`commit_model_change` or, refusing it,
        calls :meth:`ModelChange.discard`.

        Args:
          provider: New provider class name; ``None`` inherits.
          auth: New auth-method suffix; ``None`` inherits.
          model_id: New provider-specific model id; ``None`` inherits.
          account: New credential account; ``None`` inherits.

        Returns:
          change: The built model, its recipe, its provider, and a label.

        Raises:
          ValueError: ``model_recipe`` is unset, the resolved provider is
              unknown, or the resolved model id is rejected by the
              provider's catalog.

        """
        spec = self.model_recipe
        if spec is None:
            raise ValueError("agent has no model_recipe; cannot change_model")
        target = _resolve_target_spec(
            spec,
            provider=provider,
            auth=auth,
            model_id=model_id,
            account=account,
        )
        provider_obj, resolved_account = providers.build_provider_with_account_fallback(
            target.provider,
            target.auth,
            account=target.account,
            fallback_to_default=(account is None and target.provider != spec.provider),
        )
        target = dataclasses.replace(target, account=resolved_account)
        try:
            new_model = provider_obj.model(target.model_id)
        except BaseException:
            _ = _schedule_provider_close(provider_obj)
            raise
        if target.provider != spec.provider:
            label = (
                f"{spec.provider}/{spec.model_id} -> "
                f"{target.provider}/{target.model_id}"
            )
        else:
            label = f"{spec.model_id} -> {target.model_id}"
        return ModelChange(
            model=new_model,
            recipe=target,
            provider=provider_obj,
            label=label,
        )

    def commit_model_change(
        self,
        change: ModelChange,
        *,
        then: Callable[[], None] | None = None,
    ) -> None:
        """Queue ``change`` so it lands once no model call is in flight.

        Queued through the runtime inbox so an in-flight call finishes on the
        OLD model (cost attribution, retry state) before the new one goes live.

        Args:
          change: A change from :meth:`prepare_model_change`.
          then: Runs right after the swap, in the same step: settings that
              must apply to the NEW model (thinking, limits) go here.

        """

        def _apply() -> None:
            self.swap_model(change.model, spec=change.recipe, provider=change.provider)
            if then is not None:
                then()

        self.runtime.inbox.push_back(
            runtime.ModelSwitch(apply=_apply, label=change.label),
        )

    async def relogin(self) -> None:
        """Re-authenticate the current provider; hot-reload live creds.

        Drives the provider class's ``login`` classmethod (which writes
        fresh OAuth credentials to disk), then re-reads them into the
        running provider's in-memory token state via
        :class:`types.providers.AuthReloadable.handle_auth_error`. Without the reload
        the in-memory ``_refresh_token`` is still the revoked one and
        the next refresh returns 400 -- so ``/login`` would appear to
        succeed but the auth error would keep firing.

        Finally, recovers an in-flight call that is wedged in a
        service-suspension backoff: the retry loop sleeps on a single
        uninterruptible ``asyncio.sleep`` until ``retry_at``, so fresh
        credentials would otherwise sit unused for the rest of that
        (possibly multi-hour) wait. ``Halt`` cancels the sleeping call and
        returns control to the user, and the stale suspension timestamp is
        cleared so the status pane stops showing "retrying in ...".

        Raises:
          ValueError: ``model_recipe`` is unset, the provider class is
              unknown, or the provider has no ``login`` classmethod.

        """
        spec = self.model_recipe
        if spec is None:
            raise ValueError("agent has no model_recipe; cannot relogin")
        prov_cls = providers.provider_class(
            spec.provider,
        )
        if prov_cls is None:
            raise ValueError(f"unknown provider {spec.provider!r}")
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            raise ValueError(f"provider {spec.provider!r} has no login method")
        # ``login`` blocks on a browser-callback wait (or ``input()`` in
        # manual mode) for up to several minutes. Running it inline would
        # freeze the single-threaded REPL event loop: the input pump
        # could not drain, so keystrokes pile up in the terminal and a
        # failed/never-returning auth wedges the whole session. Off-load
        # to a worker thread so the loop keeps servicing input and the
        # wait stays cancellable.
        login_kwargs: dict[str, object] = {}
        if spec.account is not None:
            with contextlib.suppress(TypeError, ValueError):
                if "account" in inspect.signature(login_fn).parameters:
                    login_kwargs["account"] = spec.account
        await asyncio.to_thread(login_fn, **login_kwargs)
        # Reach the owning provider to hot-reload credentials after the
        # re-login. This is a deliberate private-attribute access, not a
        # capability probe: every rich provider model carries
        # ``_provider`` (it is not part of the lean ``Model`` contract,
        # so there is no public accessor). The ``AuthReloadable`` check
        # then narrows to providers that actually refresh tokens.
        live_provider = getattr(self.model, "_provider", None)
        if isinstance(live_provider, AuthReloadable):
            await live_provider.handle_auth_error()
        # Break a wedged service-suspension sleep so the new credentials
        # take effect immediately instead of after the old ``retry_at``.
        self.runtime.service_suspended_until = None
        self.runtime.resume_retry_at = None
        if self.runtime.model_call is not None:
            self.runtime.inbox.push_back(runtime.Halt())

    def system_prompt(self) -> str:
        """Assemble the full system prompt (system + tool contributions).

        Returns:
          prompt: System prompt rebuilt for the next request, or the
              frozen spec verbatim when ``frozen_system`` is set.

        """
        return self._build_system()

    def resume(
        self,
        meta: SessionMeta,
        tape: list[TapeRecord],
        tool_state: ToolState,
    ) -> None:
        """Apply a persisted session snapshot to this agent.

        Restores cost / activity / compaction state, the original
        ``session_id`` and ``status``, and (when the persisted
        provider+model differ from the current one) swaps the model.
        Repopulates ``tool_state`` from the persisted snapshot; its
        ``_content_cache`` stays empty and reloads lazily on the first
        post-resume ``check_stale`` / ``consume_changed_files`` against
        disk (matching ``restore_tool_state``). Resume never eagerly
        re-reads touched paths: a path that has migrated onto a hung
        network mount would otherwise block the resume thread forever.

        Args:
          meta: Persisted ``SessionMeta``.
          tape: Persisted tape records loaded by ``load_session``.
          tool_state: Persisted ``ToolState`` snapshot.

        """
        if meta.session_id:
            self._session_id = meta.session_id
            self.runtime.session_id = meta.session_id
        self.runtime.replay_tape(tape)
        self.tool_state = tool_state
        if meta.status:
            self._status = meta.status
        self._checkpoint = meta.checkpoint
        self.checkpoint_updated_at = meta.checkpoint_updated_at
        self.checkpoint_restored = meta.checkpoint is not None
        self.child_checkpoints.clear()
        for event in meta.runtime_events:
            self._track_child_checkpoint(event)
        self.restored_child_checkpoints = set(self.child_checkpoints)
        self.cost_tracker.restore_totals(spend=meta.spend, total=meta.tokens)
        self._own_spend = meta.own_spend
        self.activity.num_tool_call_rounds = meta.num_tool_call_rounds
        self.activity.elapsed_seconds = meta.total_active_elapsed_seconds
        self.compaction_state.compact_count = meta.compact_count
        self.runtime.resume_retry_at = _latest_service_retry_at(meta.runtime_events)
        if meta.provider and meta.model_id and not self._runs_recipe_of(meta):
            restored = restore_model(meta)
            if restored is not None:
                new_model, new_spec, provider = restored
                self.swap_model(new_model, spec=new_spec, provider=provider)
        # The tape we just replayed came from session.jsonl. Without
        # rebaselining, the next ``SaveSession`` would write all those
        # same records back to the same file, duplicating them.
        if self._rebaseline_persistence is not None:
            self._rebaseline_persistence()

    # The model id alone cannot tell two recipes apart: one id served under another
    # provider, auth, or account is a different model, billed to a different account.
    def _runs_recipe_of(self, meta: SessionMeta) -> bool:
        """Whether the active model already is the one ``meta`` recorded."""
        spec = self.model_recipe
        return (
            spec is not None
            and meta.model_id == self.model.tagged_model_id
            and (meta.provider, meta.auth, meta.account)
            == (spec.provider, spec.auth, spec.account or "")
        )

    # -- Foreground slot / cancel verbs --------------------------------

    def halt(self) -> None:
        """Cancel the current model call; wait for user input."""
        self.runtime.inbox.push_back(runtime.Halt())

    def kill_tool(self, qid: str) -> None:
        """Cancel one outstanding tool task.

        Args:
          qid: Human job id or provider call id of the task to cancel.

        """
        call_id = self._call_id_for_job(qid)
        # ``_bg`` is keyed by job id, so a call id must be translated before the
        # cancel; ``Kill`` alone never reaches a job running outside the cohort.
        self.cancel_background(self._job_ids_by_call_id.get(call_id, qid))
        self.runtime.inbox.push_back(runtime.Kill(call_id=call_id))

    def kill_all_tools(self) -> None:
        """Cancel every outstanding tool task."""
        self._cancel_all_background()
        self.runtime.inbox.push_back(runtime.Kill())

    def publish_service_suspended(
        self,
        retry_at: float,
        delay_sec: float,
        server_supplied: bool,
        error: Exception,
    ) -> None:
        """Publish a durable event for a recoverable model-service block.

        Args:
          retry_at: Monotonic timestamp when retry may resume.
          delay_sec: Delay before retrying the suspended service.
          server_supplied: Whether the delay came from the provider.
          error: Exception that caused the service suspension.

        """
        spec = self.model_recipe
        self.runtime.service_suspended_until = retry_at
        self.runtime.publish(
            runtime.ModelServiceSuspended(
                provider=spec.provider if spec else type(self.model).__name__,
                auth=spec.auth if spec else "",
                account=spec.account if spec else None,
                model_id=self.model.tagged_model_id,
                retry_at=retry_at,
                delay_sec=delay_sec,
                server_supplied=server_supplied,
                error=service_error_snapshot(error),
            ),
        )

    def shutdown(self, *, force: bool = False) -> None:
        """End ``serve_forever`` cleanly.

        Cancels long-lived tasks that would otherwise outlive
        ``serve_forever`` and write to a dead inbox: every cohort-decayed
        ``runtime.detached`` task, and every explicit ``background: true``
        tool job in ``self._bg``. Serviced subagents are exempt -- they
        own their own ``serve_forever`` and are never touched by the
        parent's shutdown; oneshot subagents have already self-stopped.

        ``force=True`` additionally drains the foreground cohort by
        pushing a ``Kill()`` verb before ``Quit()``. The cooperative
        Kill+Quit sequence (rather than a single Kill-with-quit op) is
        deliberate: ``Kill()`` returns the cohort to a paired state via
        ``_stop_all_tools``; ``Quit()`` then lets ``run_forever`` observe
        the cleanly drained state and exit. Folding the two would skip
        the cohort cleanup gate.

        Args:
          force: When True, also push ``Kill()`` for the foreground cohort.

        """
        if not self._shutting_down:
            self._close_task = _schedule_close(self.model, self._owned_provider)
        self._shutting_down = True
        self._stop_driver(force=force)

    async def aclose(self) -> None:
        """Shut down the agent and await terminal model and provider cleanup."""
        self.shutdown()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                _close_model(self.model, self._owned_provider),
            )
        await asyncio.shield(self._close_task)

    def _stop_driver(self, *, force: bool = False) -> None:
        if force:
            self.kill_all_tools()
        self._cancel_all_detached()
        for job in list(self._bg.values()):
            if _should_cancel_background(job, mode="all"):
                _ = job.task.cancel()
        self.runtime.inbox.push_back(runtime.Quit())

    def _cancel_all_detached(self) -> None:
        """Cancel every runtime-detached task so none outlive ``serve_forever``."""
        for task in self.runtime.detached.values():
            if not task.done():
                _ = task.cancel()

    # -- Strategy methods ---------------------------------------------

    async def compact(self, args: str = "") -> None:
        """Preempt in-flight work and run compaction.

        Args:
          args: Free-form compaction instructions forwarded to the compactor.

        """
        await self._await_event(
            runtime.Compact(args=args),
            (runtime.CompactComplete, runtime.CompactFailed),
        )

    async def recompact(self, args: str = "") -> None:
        """Alias ``/compact`` using a distinct runtime event.

        Args:
          args: Free-form compaction instructions forwarded to the compactor.

        """
        await self._await_event(
            runtime.Recompact(args=args),
            (runtime.CompactComplete, runtime.CompactFailed),
        )

    async def clear(self) -> None:
        """Preempt and wipe history + per-tool recall caches.

        Resolves when the runtime publishes ``ClearComplete`` -- mirrors
        ``compact`` / ``recompact`` so callers can observe the wipe
        synchronously after ``await``.
        """
        self.tool_state.reset_tool_recall()
        self._last_input_tokens = 0
        self._last_measured_history = ()
        self._cancel_all_background()
        await self._await_event(
            runtime.Clear(),
            runtime.ClearComplete,
        )

    async def serve_forever(self) -> None:
        """Drive the agent until ``shutdown`` is called.

        Registration lifetime is decoupled from any single driver task:
        :meth:`register` writes the ``agent_registry`` entry before the
        loop starts and :meth:`deregister` removes it after the loop
        exits, so the stable label is addressable by ``/send`` and
        ``AgentSend`` for the agent's whole serving life -- one-shot and
        serviced alike.

        When a spawner already owns the registry entry (it registered
        ``self`` synchronously so the label is visible the instant the
        spawn tool returns), this reuses that label and does NOT own the
        deregister -- the spawner's teardown path does. Otherwise (the
        root agent, direct ``serve_forever`` callers) this self-registers
        and self-deregisters.

        Raises:
          RuntimeError: When another driver already holds this agent.

        """
        await self._serve(self._claim_driver("serve_forever"))

    # One driver per agent: a second loop on the same inbox steals its events, and
    # whichever exits first releases the guard the other still relies on. The claim
    # is a token rather than a bool so only the loop that holds it can release it.
    def _claim_driver(self, caller: str) -> object:
        """Claim the single-driver slot, or raise when another driver holds it."""
        if self._run_active:
            raise RuntimeError(
                f"Agent.{caller} is not reentrant; another caller already drives"
                f" this agent (session_id={self._session_id}).",
            )
        claim = object()
        self._driver_claim = claim
        return claim

    @property
    def _run_active(self) -> bool:
        return self._driver_claim is not None

    # The loop releases the claim it was started under, so a caller that claims and
    # returns while the loop keeps draining (``drive_until_first_idle``) never lapses
    # the guard early.
    async def _serve(self, claim: object) -> None:
        """Run the loop under ``claim``, releasing it when the loop exits."""
        preexisting = self._registered_label()
        owns_registry = preexisting is None
        label = preexisting or self._dedup_label()
        if owns_registry:
            self.register(label)
        try:
            with self._install_contextvars(label=label):
                await self.runtime.run_forever()
        finally:
            if self._driver_claim is claim:
                self._driver_claim = None
            if owns_registry:
                self.deregister(label)

    def _registered_label(self) -> str | None:
        """Return the ``agent_registry`` label already bound to ``self``."""
        for existing_label, agent in agent_registry.items():
            if agent is self:
                return existing_label
        return None

    def register(self, label: str) -> None:
        """Insert this agent into ``agent_registry`` under ``label``.

        Spawner-owned: the registry entry's lifetime is the agent's
        serving life, not one driver task's. Call :meth:`deregister` with
        the same label to remove it.

        Args:
          label: Deduplicated registry key (see :meth:`_dedup_label`).

        """
        agent_registry[label] = self

    def deregister(self, label: str) -> None:
        """Remove this agent's ``agent_registry`` entry under ``label``.

        Args:
          label: The label previously passed to :meth:`register`.

        """
        if agent_registry.get(label) is self:
            _ = agent_registry.pop(label, None)

    # Both lifecycles register under their stable label (``self.name`` for a spawned
    # child, or the inherited ``agent_label_var``). A numeric suffix is appended only on
    # collision so ``AgentSend`` can still address a specific agent when several share a
    # base name.
    def _dedup_label(self) -> str:
        """Return the stable spawn label, de-duplicated against collisions."""
        base_label = self.name or agent_label_var.get("") or "Agent"
        return unique_registry_label(base_label)

    async def run(
        self,
        msg: runtime.UserMessage,
    ) -> AsyncGenerator[runtime.RuntimeEvent, None]:
        """Process one inbound message; drive rounds until idle.

        Convenience entrypoint used by tests and non-``serve_forever``
        callers. **Single-driver contract**: this method stops its driver
        in its ``finally`` block, retaining the model for subsequent runs.
        Call ``aclose`` for terminal cleanup. Calling ``run`` while another task is
        already driving this agent (concurrent ``run`` or ``serve_forever``)
        would push a ``Quit()`` into the foreign driver's inbox on exit,
        killing it. The agent guards via a driver claim so the
        collision fails loudly rather than silently corrupting the
        in-flight driver.

        Args:
          msg: User message to push and run to idle.

        Yields:
          event: Each ``types.runtime.RuntimeEvent`` published until
              ``types.runtime.AgentIdle`` (the strict fully-drained edge,
              which also waits for detached / background tool work) or
              ``types.runtime.ModelResponseError``.

        Raises:
          RuntimeError: When another ``Agent.run`` is already in flight
              on the same agent.

        """
        # The claim is taken synchronously, before any ``await``: two concurrent
        # callers cannot both pass it under cooperative scheduling.
        claim = self._claim_driver("run")
        events: asyncio.Queue[runtime.RuntimeEvent] = asyncio.Queue()
        terminal = asyncio.Event()

        def _watch(event: runtime.RuntimeEvent) -> None:
            events.put_nowait(event)
            if isinstance(
                event,
                (runtime.AgentIdle, runtime.ModelResponseError),
            ):
                terminal.set()

        self.runtime.observers.append(_watch)
        try:
            self.runtime.inbox.push_back(msg)
            drive = asyncio.create_task(self._serve(claim))
            drive.add_done_callback(
                log_task_exception(logger, "Agent.run drive task crashed"),
            )
            try:
                while True:
                    get_task = asyncio.create_task(events.get())
                    terminal_task = asyncio.create_task(terminal.wait())
                    # Wait on the drive task too: a ``Quit`` makes
                    # ``run_forever`` return without publishing a terminal
                    # event, so ``terminal`` never sets. Without ``drive`` in
                    # the wait set, ``events.get()`` would block forever on an
                    # already-dead driver (external-Quit hang).
                    done, _pending = await asyncio.wait(
                        {get_task, terminal_task, drive},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Drain pending events before checking terminal state so
                    # consumers see the AgentIdle or ModelResponseError event.
                    if get_task in done:
                        yield get_task.result()
                    else:
                        _ = get_task.cancel()
                    if terminal_task not in done:
                        _ = terminal_task.cancel()
                    if terminal.is_set() and events.empty():
                        break
                    # Driver exited without a terminal event (``Quit`` /
                    # crash): drain any already-queued events, then stop --
                    # no further event will ever arrive.
                    if drive in done and events.empty():
                        break
            finally:
                self._stop_driver()
                with contextlib.suppress(asyncio.CancelledError):
                    await drive
                for task in (self.runtime.model_call, self.runtime.compact_task):
                    if task is not None:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                self.runtime.model_call = None
                self.runtime.compact_task = None
        finally:
            if _watch in self.runtime.observers:
                self.runtime.observers.remove(_watch)
            if self._driver_claim is claim:
                self._driver_claim = None

    async def drive_until_first_idle(
        self,
        msg: runtime.UserMessage,
        *,
        result_of: Callable[[list[runtime.ModelContextEvent]], runtime.ToolResult]
        | None = None,
    ) -> runtime.ToolResult:
        """Push ``msg``, serve until the first post-work idle, return its result.

        Unlike :meth:`run`, this does NOT shut down: it leaves the agent's
        ``serve_forever`` loop live so a serviced agent keeps servicing its
        inbox after the first reply. One-shot callers shut down explicitly
        after this returns.

        The runtime publishes a boot ``AgentIdle`` at the top of the first
        ``run_forever`` iteration -- before any work, with empty history.
        That boot edge is suppressed while history is empty; the method
        returns only on the first idle where the agent has actually
        produced work.

        Args:
          msg: User message to push and drive to the first work idle.
          result_of: Extractor turning the final history into a
              ``ToolResult``; defaults to the last assistant message's text.

        Returns:
          result: The first post-work reply as a ``ToolResult``.

        Raises:
          RuntimeError: When another driver is already in flight on this agent.

        """
        claim = self._claim_driver("drive_until_first_idle")
        first_idle: asyncio.Event = asyncio.Event()
        terminal_errors: list[BaseException] = []

        def _watch(event: runtime.RuntimeEvent) -> None:
            # Two terminal edges. (1) A post-work ``AgentIdle`` -- the normal
            # first-result return. (2) A ``ModelResponseError`` -- the child
            # produced no work idle, so returning lets the caller surface the
            # error instead of blocking until the (still-live) loop is shut down.
            if isinstance(event, runtime.ModelResponseError):
                terminal_errors.append(event.exception)
                first_idle.set()
            elif isinstance(event, runtime.AgentIdle) and bool(self.history):
                first_idle.set()

        self.runtime.observers.append(_watch)
        self.runtime.inbox.push_back(msg)
        drive = asyncio.create_task(self._serve(claim))
        # Expose the live loop task so a one-shot caller can await it to
        # completion after it pushes ``shutdown`` -- the loop keeps
        # running when this method returns.
        self._drive_task = drive
        try:
            drive.add_done_callback(
                log_task_exception(logger, "drive_until_first_idle drive task crashed"),
            )
            idle_task = asyncio.create_task(first_idle.wait())
            # Wait on the drive task too: a crash/Quit ends the loop without
            # ever firing a work idle, so ``first_idle`` would never set.
            _done, _pending = await asyncio.wait(
                {idle_task, drive},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not idle_task.done():
                _ = idle_task.cancel()
            # A drive that ended by raising produced no result at all;
            # extracting history anyway hands the caller an empty success
            # and buries the traceback in a done-callback log line.
            if drive.done() and not drive.cancelled():
                exc = drive.exception()
                if exc is not None:
                    raise exc
            if terminal_errors:
                raise terminal_errors[0]
            extractor = result_of or _default_last_assistant_result
            return extractor(self.history)
        finally:
            if _watch in self.runtime.observers:
                self.runtime.observers.remove(_watch)
            # A live loop keeps the claim; ``_serve`` releases it when the loop
            # exits, since the loop outlives this method by design.
            if drive.done() and self._driver_claim is claim:
                self._driver_claim = None

    # -- Internal helpers ---------------------------------------------

    # Every child inherits the root's cost tracker via ``cost_root_var`` so the root
    # sees the full spawn-tree spend (and ``max_budget_usd`` caps each agent against its
    # own ``_own_spend``, not the shared sink). Tool-state depth is incremented from the
    # parent so ``AgentSpawn`` depth caps actually fire.
    #
    # Registry ownership is NOT held here -- :meth:`register` / :meth:`deregister` own
    # the ``agent_registry`` entry with a lifetime decoupled from any one driver task.
    # When ``label`` is None (direct callers, tests) this CM registers and deregisters
    # itself so the registry entry still exists for the block; when ``label`` is passed
    # (``serve_forever``) the caller owns the registry and this CM only binds the
    # ContextVars.
    @contextlib.contextmanager
    def _install_contextvars(
        self,
        *,
        label: str | None = None,
    ) -> Generator[None, None, None]:
        """Install per-agent ContextVars for the lifetime of the block."""
        agent_token = current_agent_var.set(self)
        parent_root = cost_root_var.get(None)
        # Every child inherits the root cost sink so cost rolls up to one
        # place. Only the true root (no ambient sink) installs itself.
        cost_token: contextvars.Token[CostTracker | None] | None = (
            cost_root_var.set(self.cost_tracker) if parent_root is None else None
        )
        parent_state = tool_state_var.get(None)
        prior_depth = self.tool_state.depth
        self.tool_state.depth = 0 if parent_state is None else parent_state.depth + 1
        owns_registry = label is None
        if label is None:
            label = self._dedup_label()
            self.register(label)
        label_token = agent_label_var.set(label)
        counter_token = agent_counter_var.set(itertools.count())
        state_token = tool_state_var.set(self.tool_state)
        try:
            yield
        finally:
            if owns_registry:
                self.deregister(label)
            tool_state_var.reset(state_token)
            agent_counter_var.reset(counter_token)
            agent_label_var.reset(label_token)
            if cost_token is not None:
                cost_root_var.reset(cost_token)
            current_agent_var.reset(agent_token)
            # ``depth`` is a plain instance field, not a ContextVar, so it needs
            # explicit restoration to match the token resets above -- otherwise a
            # reused agent re-entered under a different parent starts from a stale
            # depth.
            self.tool_state.depth = prior_depth

    async def _await_event(
        self,
        push: runtime.RuntimeEvent,
        complete: type[runtime.RuntimeEvent] | tuple[type[runtime.RuntimeEvent], ...],
    ) -> None:
        """Push ``push`` and resolve when an event of ``complete`` type arrives."""
        fut = asyncio.get_running_loop().create_future()

        def resolver(ev: runtime.RuntimeEvent) -> None:
            if isinstance(ev, complete) and not fut.done():
                fut.set_result(None)

        self.runtime.observers.append(resolver)
        try:
            self.runtime.inbox.push_back(push)
            await fut
        finally:
            if resolver in self.runtime.observers:
                self.runtime.observers.remove(resolver)

    # Intentionally rebuilt per request: ``_system_spec`` and each tool's ``prompt()``
    # may be callable and capture mutable state (cwd, registered subagents, persistent
    # IPC peers) that must stay live across ``cd`` and registry mutations. Caching here
    # would freeze that state on the next provider call.
    #
    # Exception: ``self._frozen_system`` (set by a hot ``AgentSpawn``) opts OUT of that
    # liveness on purpose -- ``_system_spec`` already holds a complete, pre-rendered
    # snapshot (typically the parent's own ``system`` at spawn time), and re-appending
    # this agent's own tools' ``prompt()`` contributions on top would immediately
    # reintroduce the prefix drift hot spawning exists to avoid (see #361: a bundled
    # ``BackgroundTask`` or a different spawn-depth line the parent never had). Returned
    # verbatim, with no tool contributions or detached-note injection.
    def _build_system(self) -> str:
        """Assemble the system prompt from base spec + tool contributions."""
        spec = self._system_spec
        base = spec if isinstance(spec, str) else spec()
        if self._frozen_system:
            return base
        parts: list[str] = [base] if base else []
        gate = self.tool_gate
        for tool in self._tools_map.values():
            if gate is not None and tool.name not in gate:
                continue
            contribution = tool.prompt() or ""
            if contribution:
                parts.append(contribution)
        # Teach the model that ``DetachedArrived`` history turns are runtime
        # result deliveries, not a callable tool. Injected the moment any tool
        # detaches -- before the first synthesized ``DetachedArrived`` pair
        # lands -- so the prohibition precedes the imitable pattern rather than
        # co-occurring with it (the reactive injection taught the model the
        # exact shape it then copied). Prompts without detached activity stay
        # lean. Paired with the runtime guard in ``_run_tool_and_post``.
        if self._has_detached_activity():
            parts.append(runtime.DETACHED_ARRIVED_SYSTEM_NOTE)
        return "\n\n".join(parts)

    # Proactive: a live ``runtime.detached`` task means a ``[detached]`` stub is already
    # in context and a forward delivery is pending, so the note must appear now. The
    # history check covers the window after the task completed but its synthesized pair
    # still resolves into context.
    def _has_detached_activity(self) -> bool:
        """Return True when a tool is detached or a ``DetachedArrived`` turn is present."""
        if self.runtime.detached:
            return True
        return any(
            isinstance(entry, runtime.AssistantMessage)
            and any(tc.name == runtime.DETACHED_ARRIVED_TOOL for tc in entry.tool_calls)
            for entry in self.runtime.context().messages
        )

    # -- Observers ----------------------------------------------------

    def record_response(self, response: ModelResponse) -> None:
        """Record a completed response: tokens self-only, cost to the root sink.

        Tokens (and per-call provenance) go to ``self.cost_tracker`` so the
        status pane's per-agent token count stays self-only. Cost goes to
        the root cost sink (``cost_root_var``) so ``root.spend`` is
        the whole spawn tree's spend, counted exactly once per response. The
        per-agent budget cap is checked against ``_own_spend`` -- this
        agent's own spend -- so a subagent's cap governs the subagent, not
        the tree.

        Args:
          response: Completed model response with token counts and cost.

        Raises:
          BudgetExhaustedError: This agent's own cost reached ``max_budget_usd``.

        """
        previous_tokens = self.cost_tracker.last_request
        previous_model_id = self.cost_tracker.last_model_id
        previous_response_time = self.cost_tracker.last_response_time
        model_id = self.model.tagged_model_id
        self.cost_tracker.record_tokens(response, model_id=model_id)
        self._record_cache_waste(
            previous=previous_tokens,
            current=response.tokens,
            idle_sec=time.time() - previous_response_time,
            model_changed=bool(previous_model_id) and previous_model_id != model_id,
        )
        # Anchor the proactive compaction trigger on the provider's exact
        # prompt size, every cache pool included.
        self._last_input_tokens = response.tokens.prompt
        self._last_measured_history = tuple(self.runtime.context().messages)
        self._bill(response)

    # Re-sending a refused context only repeats the refusal (and, for a safeguard
    # block, flags the account again), while halting until /clear threw away the
    # whole session. The refused material is what arrived since the model last
    # answered, so that tail alone leaves the context; the conversation before it
    # stays, and the next input continues from there.
    def withhold_refused_tail(self) -> None:
        """Mask what arrived since the last assistant turn out of the context.

        Tool results become error stubs that keep their call ids, so the
        assistant turn that requested them stays paired; user-side messages
        are dropped.
        """
        resolved = self.runtime.context()
        messages, origins = resolved.messages, resolved.origins
        tail = next(
            (
                i + 1
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], runtime.AssistantMessage)
            ),
            0,
        )
        if tail == len(messages):
            return
        masked = list(dict.fromkeys(origins[tail:]))
        # A splice's payload renders as one contiguous segment, so the first
        # masked origin may also carry entries before the tail; those ride in
        # this payload rather than vanish with the splice.
        start = origins.index(masked[0])
        withheld = tuple(
            runtime.ToolResult(
                call_id=entry.call_id,
                content=REFUSED_RESULT,
                is_error=True,
            )
            for entry in messages[tail:]
            if isinstance(entry, runtime.ToolResult)
        )
        payload = (*messages[start:tail], *withheld)
        # Masking a splice's ref kills it, which lapses its own mask and would
        # resurrect what it replaced (``_append_or_coalesce_user`` absorbs the
        # same way), so its ranges are carried into this mask.
        absorbed = (
            r
            for ref in masked
            if isinstance(splice := self.runtime.record_at(ref), ContextSplice)
            for r in splice.mask
        )
        self.runtime.append_splice(
            mask=merge_mask_ranges(
                (
                    *absorbed,
                    *(
                        MaskRange(
                            session_id=ref.session_id,
                            lo=ref.ordinal,
                            hi=ref.ordinal,
                        )
                        for ref in masked
                    ),
                ),
            ),
            insert_after=origins[start - 1] if start > 0 else None,
            payload=payload,
            strategy="refused_tail",
            paired_externally=unpaired_call_ids(payload),
            discards_content=True,
        )

    def record_side_response(self, response: ModelResponse) -> None:
        """Bill a call made outside the conversation, such as a summary or consult.

        Cost, token totals, and the budget cap apply exactly as for a turn.
        The compaction anchor and the cache-miss baseline do not move: the
        side call's prompt is not this conversation, so letting it stand in
        for one made a full context read as nearly empty.

        Args:
          response: Completed side-call response with token counts and cost.

        Raises:
          BudgetExhaustedError: This agent's own cost reached ``max_budget_usd``.

        """
        self.cost_tracker.record_side_tokens(
            response,
            model_id=self.model.tagged_model_id,
        )
        self._bill(response)

    def _bill(self, response: ModelResponse) -> None:
        """Charge one response's cost and enforce this agent's budget cap."""
        cost_sink = cost_root_var.get(None) or self.cost_tracker
        cost_sink.record_cost(response)
        self._own_spend = self._own_spend + response.spend
        if (
            self.max_budget_usd is not None
            and self._own_spend.total >= self.max_budget_usd
        ):
            raise BudgetExhaustedError(
                total_cost_usd=self._own_spend.total,
                max_budget_usd=self.max_budget_usd,
            )
        self._surface_usage_warning()

    # Best-effort: a diagnostic must never break the turn it observes, so any failure
    # here is swallowed rather than propagated.
    def _record_cache_waste(
        self,
        *,
        previous: TokenCount,
        current: TokenCount,
        idle_sec: float,
        model_changed: bool,
    ) -> None:
        """Detect and record an avoidable prompt-cache miss for this turn."""
        try:
            miss = detect_cache_miss(
                previous,
                current,
                idle_sec=idle_sec,
                cache_ttl_sec=self.model.settings.cache_ttl_sec,
                model_changed=model_changed,
                spend=self.model.spend,
            )
        except Exception:
            logger.debug("cache-waste detection failed", exc_info=True)
            return
        if miss is not None:
            self.cost_tracker.record_cache_miss(miss)

    # Reads the provider's normalized :class:`UsageSnapshot` after a response and warns
    # once per window each time it crosses the warning threshold, so a near-limit window
    # is surfaced before it blocks. Providers without telemetry return ``None`` and this
    # is a no-op.
    def _surface_usage_warning(self) -> None:
        """Publish an advisory ``NoticeMessage`` when a usage window is high."""
        snapshot = self.model.usage_snapshot()
        if snapshot is None:
            return
        for window in snapshot.windows:
            util = window.utilization
            over = window.blocked or (util is not None and util >= _USAGE_WARN_FRACTION)
            below_clear = util is not None and util < _USAGE_CLEAR_FRACTION
            if not over:
                # Re-arm only once the window has clearly recovered (hysteresis),
                # or when utilization is unknown and it is no longer blocked.
                if below_clear or util is None:
                    self._usage_warned.discard((window.label, True))
                    self._usage_warned.discard((window.label, False))
                continue
            key = (window.label, window.blocked)
            if key in self._usage_warned:
                continue
            # A fresh blocked advisory supersedes a prior warn for the window.
            self._usage_warned.add(key)
            pct = f"{util:.0%}" if util is not None else "limit"
            state = "blocked" if window.blocked else f"{pct} used"
            self.runtime.publish(
                runtime.NoticeMessage(
                    text=f"[usage: {window.label} window {state}]",
                    tier="advisory",
                ),
            )

    # A round chain spans from the first ``types.runtime.ModelCallStarted`` of a user
    # turn through ``types.runtime.ModelIdle`` (or terminal cancel). Mid-chain
    # ``types.runtime.ModelResponseComplete`` events with ``tool_calls`` do not reset
    # ``active`` so the status-pane spinner keeps ticking through tool execution windows
    # -- the user sees continuous activity until the agent truly idles.
    def _track_activity(self, event: runtime.RuntimeEvent) -> None:
        """Bracket round-chain elapsed time + count streamed chars."""
        if isinstance(event, runtime.ModelCallStarted):
            now = asyncio.get_running_loop().time()
            # Restamp ``current_call_start`` on every call so the status
            # pane measures the current call's age, not the whole round
            # chain. Bank the prior call's elapsed into the cumulative
            # session total so the bracket display is unaffected.
            if self.activity.active and self.activity.current_call_start > 0:
                prior = now - self.activity.current_call_start
                self.activity.elapsed_seconds += max(0.0, prior)
            self.activity.active = True
            self.activity.current_call_start = now
            self.activity.live_response_text = ""
        elif isinstance(event, runtime.ModelResponsePartial):
            # Resume timing if the prior chunk arrived after a suspension.
            if self.activity.active and self.activity.current_call_start == 0.0:
                self.activity.current_call_start = asyncio.get_running_loop().time()
            # Accumulate raw text; readers tokenize the whole string so a
            # chunk shorter than one token does not floor to zero tokens.
            self.activity.live_response_text += event.text
        elif isinstance(event, runtime.ModelResponseThinking):
            if self.activity.active and self.activity.current_call_start == 0.0:
                self.activity.current_call_start = asyncio.get_running_loop().time()
        elif isinstance(event, runtime.ModelServiceSuspended):
            # Bank active time so the suspension sleep doesn't count.
            if self.activity.active and self.activity.current_call_start > 0:
                elapsed = (
                    asyncio.get_running_loop().time() - self.activity.current_call_start
                )
                self.activity.elapsed_seconds += max(0.0, elapsed)
                self.activity.current_call_start = 0.0
        elif (
            isinstance(event, runtime.ModelResponseComplete)
            and event.message.tool_calls
        ):
            # Tool calls follow; spinner keeps ticking through the cohort.
            pass
        elif isinstance(
            event,
            (
                runtime.ModelResponseComplete,
                runtime.ModelIdle,
                runtime.ModelResponseCancelled,
                runtime.ModelResponseError,
            ),
        ):
            if self.activity.active:
                if self.activity.current_call_start > 0:
                    elapsed = (
                        asyncio.get_running_loop().time()
                        - self.activity.current_call_start
                    )
                    self.activity.elapsed_seconds += max(0.0, elapsed)
                self.activity.active = False
                self.activity.current_call_start = 0.0
        elif isinstance(event, runtime.CompactStarted):
            self.activity.current_compact_start = asyncio.get_running_loop().time()
        elif isinstance(
            event,
            (
                runtime.CompactComplete,
                runtime.CompactFailed,
            ),
        ):
            self.activity.current_compact_start = 0.0

    # Entries must outlive their turn -- the renderer resolves a result back to its
    # tool, and a detached one can land many rounds later -- so the registry is bounded
    # by age rather than pruned per call. Without a bound it grows one entry per tool
    # call for the entire session.
    def _track_tool_registry(self, event: runtime.RuntimeEvent) -> None:
        """Populate the cohort id → (tool_name, started) registry."""
        if isinstance(event, runtime.ModelResponseComplete):
            now = time.time()
            for tc in event.message.tool_calls:
                self._tool_registry[tc.id] = (tc.name, now)
            if event.message.tool_calls:
                self.activity.num_tool_call_rounds += 1
                self._round_result_chars = 0
            self._prune_tool_registry()

    # Detached and still-running calls are kept whatever their age: their results have
    # not arrived yet, so forgetting them would leave the renderer unable to attribute
    # the late arrival.
    def _prune_tool_registry(self) -> None:
        """Drop the oldest completed entries once the registry is large."""
        if len(self._tool_registry) <= _TOOL_REGISTRY_MAX:
            return
        live = set(self.runtime.detached) | {
            job.call_id for job in self._bg.values() if job.call_id
        }
        stale = sorted(
            (started, call_id)
            for call_id, (_name, started) in self._tool_registry.items()
            if call_id not in live
        )
        for _started, call_id in stale[: len(self._tool_registry) - _TOOL_REGISTRY_MAX]:
            _ = self._tool_registry.pop(call_id, None)

    def _track_compaction(self, event: runtime.RuntimeEvent) -> None:
        """Update compaction state after a barrier lands."""
        if isinstance(event, runtime.CompactComplete) and event.records:
            self._last_input_tokens = 0
            self._last_measured_history = ()
            self.tool_state.reset_tool_recall()
            self.compaction_state.compact_count += 1
            self.compaction_state.compact_failures = 0

    # Runs pre-increment: ``num_tool_call_rounds`` reflects completed rounds, so this
    # response would be round ``num + 1``. Block when that next round would exceed
    # ``max_tool_call_rounds``.
    def _before_tool_spawn(
        self,
        message: runtime.AssistantMessage,
    ) -> runtime.RuntimeEvent | None:
        """Reject capped tool rounds before runtime spawns tool tasks."""
        if (
            message.tool_calls
            and self.max_tool_call_rounds is not None
            and self.activity.num_tool_call_rounds + 1 > self.max_tool_call_rounds
        ):
            return runtime.ModelResponseError(self._tool_round_limit_error())
        return None

    def _tool_round_limit_error(self) -> RuntimeError:
        """Build the max-tool-call-rounds error."""
        return RuntimeError(
            "Tool-call-round limit reached"
            f" ({self.max_tool_call_rounds} rounds)."
            f" [{ERROR_MAX_TOOL_CALL_ROUNDS}]",
        )

    def cancel_background(self, job_id: str) -> None:
        """Cancel and forget a background task, if present.

        Symmetric to :meth:`forget_background`: both clear the
        ``call_id`` <-> ``job_id`` mapping after popping the registry
        entry. Without that symmetry, a tool-kind cancel left the map
        entry in place and poisoned later job-id minting for fresh
        calls that happened to reuse the same provider call_id.

        Args:
          job_id: Queue id of the registered task to cancel.

        """
        job = self._bg.pop(job_id, None)
        if job is None:
            job = self.background.get(job_id)
        if job is None:
            return
        call_id = job.call_id or job.queue_id
        if job.kind == "detached":
            _ = self.runtime.discard_detached(call_id)
        self._forget_job_id(call_id)
        if not job.task.done():
            job.task.cancel()

    def forget_background(self, job_id: str) -> None:
        """Remove one background job without cancelling its task."""
        job = self._bg.pop(job_id, None)
        if job is not None:
            self._forget_job_id(job.call_id or job.queue_id)

    def tool_name_for_call(self, call_id: str) -> tuple[str, float]:
        """Return ``(tool_name, started_at)`` for a dispatched call id.

        Renderers need the originating tool to honour its display
        settings; the registry is populated when the assistant turn
        lands, so a result always resolves.

        Args:
          call_id: Provider call id from the ``ToolResult``.

        Returns:
          entry: ``(tool_name, started_at)``; ``("", 0.0)`` when unknown.

        """
        return self._tool_registry.get(call_id, ("", 0.0))

    def job_id_for_call(self, call_id: str) -> str:
        """Return the stable human job id for a provider call id.

        Args:
          call_id: Provider call id to map to a stable job id.

        Returns:
          job_id: Stable human-readable job id for the call.

        """
        job_id = self._job_ids_by_call_id.get(call_id)
        if job_id is not None:
            return job_id
        job_id = f"job-{next(self._job_counter)}"
        self._job_ids_by_call_id[call_id] = job_id
        self._call_ids_by_job_id[job_id] = call_id
        return job_id

    def _call_id_for_job(self, job_or_call_id: str) -> str:
        """Resolve a human job id to its provider call id when known."""
        return self._call_ids_by_job_id.get(job_or_call_id, job_or_call_id)

    # A detached call leaves ``runtime.detached`` on completion without telling the
    # agent, so the ids minted for the job listing are freed here, by liveness, rather
    # than at a completion hook that does not exist.
    def _forget_finished_job_ids(self) -> None:
        """Drop job ids whose call is neither detached nor a registered job."""
        live = set(self.runtime.detached) | {
            job.call_id or job.queue_id for job in self._bg.values()
        }
        for call_id in [c for c in self._job_ids_by_call_id if c not in live]:
            self._forget_job_id(call_id)

    def _forget_job_id(self, call_id: str) -> None:
        """Forget a completed or cancelled provider call's human job id."""
        job_id = self._job_ids_by_call_id.pop(call_id, None)
        if job_id is not None:
            self._call_ids_by_job_id.pop(job_id, None)

    def _cancel_all_background(self) -> None:
        """Cancel and forget every explicit background tool job."""
        for job_id, job in tuple(self._bg.items()):
            if _should_cancel_background(job, mode="tools_only"):
                self.cancel_background(job_id)

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        """Add ``entry`` to the background-task registry under ``job_id``.

        Args:
          job_id: Queue id used as the registry key.
          entry: Background-task record to store.

        """
        self._bg[job_id] = entry

    # Feeds the runtime's ``_fully_drained`` gate (and thus ``AgentIdle`` / one-shot
    # ``Agent.run`` termination). Only ``kind="tool"``, non-hidden, not-yet-done jobs
    # count -- the same taxonomy ``_should_cancel_background`` uses to decide what a
    # turn owns. Subagents (``kind="subagent"``) and hidden infra (REPL pump, watchdogs)
    # live past the turn by design, so counting them would wedge ``Agent.run`` on work
    # that never ends.
    #
    # Do NOT bound a tool's duration here. Boundedness is the tool's responsibility, not
    # the agent's: ``Bash`` self-caps its timeout, and truly unbounded work uses the
    # tool's own fire-and-forget path. Adding a grace-timeout at this gate would reap
    # slow-but-finite tools mid-flight and silently drop their results -- the exact bug
    # this background tracking was added to fix. A tool that never returns is a tool
    # defect; fix it at the tool.
    def _has_pending_background(self) -> bool:
        """Return True iff a turn-scoped background tool is still running."""
        return any(
            job.kind == "tool" and not job.hidden and not job.task.done()
            for job in self._bg.values()
        )

    async def compact_if_needed(
        self,
        history: list[runtime.ModelContextEvent],
        model: Model,
        byte_compact_trigger: float = 0.8,
    ) -> bool:
        """Proactively compact when the compactor says headroom is gone.

        Bridges the inner compactor's ``should_compact`` decision (which
        the runtime's lean ``Compactor`` protocol does not expose) to the
        synchronous ``compact_now`` path used for overflow recovery.

        The trigger anchors on ``_last_input_tokens`` -- the provider's
        exact cache-inclusive count for the most recent response -- as
        ground truth, plus a client-side estimate of entries appended
        since that response. Only the first request of a process (or a
        resume) has no such count; it falls back entirely to a
        ``approx_request_tokens`` estimate of the pending request.

        Args:
          history: Pre-compaction history snapshot. ``compact_now``
              appends a barrier override to the tape; callers should
              re-resolve via ``runtime.context().messages`` after.
          model: Rich model whose tokenizer seeds the first-request
              fallback estimate and whose ``max_request_tokens`` caps the
              budget.
          byte_compact_trigger: Fraction of the model's
              ``max_request_bytes`` at which the proactive gate fires on
              byte pressure. The headroom below 1.0 reserves margin for the
              request bytes ``_compactable_wire_bytes`` does NOT count
              (system prompt, tool schemas, message text, JSON framing), so
              compaction lands before the full request body crosses the wire
              limit. A per-call default-valued kwarg (self-documenting and
              overridable) parallel to the compactor's token-side
              ``utilization_trigger``; must be in ``(0, 1]``.

        Raises:
          ValueError: If ``byte_compact_trigger`` is not in ``(0, 1]``.

        Returns:
          progressed: ``True`` when no compaction was needed, or when
              compaction completed. ``False`` when ``compact_now``
              tried and failed. The bool mirrors :meth:`compact_now`'s
              contract so callers don't have to special-case the
              proactive vs reactive path.

        """
        if byte_compact_trigger <= 0.0 or byte_compact_trigger > 1.0:
            raise ValueError(
                f"byte_compact_trigger must be in (0, 1], got {byte_compact_trigger!r}",
            )
        if self._agent_compactor is None:
            return True
        used = self._estimated_request_tokens(history, model)
        token_gate = self._agent_compactor.should_compact(
            current_tokens=used,
            largest_context=self._agent_compactor.largest_context(self.budget),
            system_tokens=model.approx_text_tokens(self.system_prompt()),
        )
        # Byte gate: request *bytes* are a budget the token gate cannot see.
        # Attachment bytes can approach the wire ceiling
        # (``Model.max_request_bytes``) while the token estimate stays well
        # under the window. Fire compaction (which sheds history attachment
        # bytes) when the estimated request byte payload crosses the trigger
        # fraction -- independent of, and OR'd with, the token gate. A
        # non-positive ``max_request_bytes`` means "no wire limit" (offline /
        # self-hosted), disabling the byte gate.
        byte_gate = model.limits.max_request_bytes > 0 and self._compactable_wire_bytes(
            history,
            model,
        ) >= int(model.limits.max_request_bytes * byte_compact_trigger)
        if not token_gate and not byte_gate:
            self.compaction_state.compact_failures = 0
            return True
        # Circuit breaker: after N consecutive auto-compact failures, stop
        # retrying and let the caller surface the prior error. Manual
        # ``/compact`` goes through ``_compact_and_post`` and is unaffected.
        if self.compaction_state.compact_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES:
            logger.warning(
                "auto-compaction circuit breaker open: %d consecutive failures",
                self.compaction_state.compact_failures,
            )
            # ``_AgentModel.stream`` asserts ``last_compact_error is not
            # None`` after a False return so it can raise a useful error.
            # The prior failure that tripped the breaker is the right
            # error to surface; if it was cleared (e.g. by a successful
            # interleaved manual ``/compact``) the breaker would already
            # have reset, so a stale ``None`` here means a coupling
            # invariant slipped. Synthesize a ContextOverflow rather than
            # letting the assert fire.
            if self.last_compact_error is None:
                self.last_compact_error = _context_overflow_error()
            return False
        return await self.compact_now()

    # The provider's count for the last request is ground truth only while the
    # history it measured still prefixes the one about to be sent; then just the
    # appended entries (this turn's tool results, new user input) are estimated, with
    # no system/tools since the anchor already counts them. Once the measured history
    # is rewritten -- a coalesced retype after an error, a model swap -- the count
    # describes a different request, so it is replaced by a full estimate. Adding the
    # two counted the whole conversation twice and fired spurious compactions.
    def _estimated_request_tokens(
        self,
        history: Sequence[runtime.ModelContextEvent],
        model: Model,
    ) -> int:
        """Estimate the token size of the request about to be sent."""
        anchor = self._last_measured_history
        budget = self.tool_results.message_budget_tokens
        if self._last_input_tokens > 0 and tuple(history[: len(anchor)]) == anchor:
            since = history[len(anchor) :]
            if not since:
                return self._last_input_tokens
            return self._last_input_tokens + model.approx_request_tokens(
                materialize_request(
                    ModelRequest(messages=list(since)),
                    tool_result_budget_tokens=budget,
                ),
            )
        # Estimated against ``live_tools()`` (``BackgroundAwareTool`` wrappers
        # applied) so the injected ``background`` / ``delay`` schema counts toward
        # the budget, matching what the next request will carry.
        return model.approx_request_tokens(
            materialize_request(
                ModelRequest(
                    messages=list(history),
                    system=self.system_prompt(),
                    tools=self.live_tools(),
                ),
                tool_result_budget_tokens=budget,
            ),
        )

    # Counts only the prefix up to and including the last ``AssistantMessage`` -- the
    # already-sent history compaction can summarize. Bytes appended *since* that
    # response are this turn's own input (the user's fresh message, this turn's tool
    # results); ``_strip_attachments`` would destroy them before the model sees them, so
    # gating on them is both useless (nothing prior to shed) and harmful (strips the
    # user's request). A single fresh read is instead bounded at the source by the read
    # tool's rendered-byte cap.
    #
    # Measured over the materialized prefix so attachments on tool results that
    # ``materialize_request`` drops are not counted, and via
    # :func:`_wire_attachment_bytes` so the estimate matches the base64, post-resize
    # payload that actually ships.
    def _compactable_wire_bytes(
        self,
        history: Sequence[runtime.ModelContextEvent],
        model: Model,
    ) -> int:
        """Wire bytes of attachments compaction can shed, as they will ship."""
        last_assistant = _last_assistant_index(history)
        if last_assistant is None:
            return 0
        prefix = history[: last_assistant + 1]
        materialized = materialize_request(
            ModelRequest(messages=list(prefix)),
            tool_result_budget_tokens=self.tool_results.message_budget_tokens,
        )
        return _wire_attachment_bytes(
            materialized.messages,
            max_image_bytes=model.limits.max_image_bytes,
        )

    async def compact_now(self) -> bool:
        """Execute the synchronous compact path used by ``_AgentModel`` for overflow recovery.

        Bypasses the inbox (the runtime would cancel our task if we
        pushed ``types.runtime.Compact``). Calls the inner compactor
        directly and appends the resulting barrier override to the
        runtime's tape. On failure, stashes the
        underlying exception on :attr:`last_compact_error` so the
        proactive / reactive raise sites in :class:`_AgentModel` can
        distinguish true context overflow (polished remediation
        message) from a transient transport / auth / generic error
        (surface verbatim).

        Returns:
          progressed: ``True`` when compaction completed (or no compactor is
              wired, which is the agent's chosen configuration); ``False``
              when the inner compactor raised. The overflow-recovery caller
              uses this signal to short-circuit instead of looping on
              unchanged history.

        """
        if self._agent_compactor is None:
            self.last_compact_error = None
            return True
        active_compact = self.runtime.compact_task
        if active_compact is not None and not active_compact.done():
            await active_compact
            self.last_compact_error = None
            self.compaction_state.compact_failures = 0
            return True
        tape_len = len(self.runtime.tape)
        tape_snapshot = tuple(self.runtime.tape)
        context_snapshot = self.runtime.context().messages
        # Mirror the inbox-arm handler at ``runtime.py: case Compact():``:
        # ``append_history`` for each lifecycle marker so resume / session
        # replay see the same CompactStarted / CompactComplete /
        # CompactFailed entries the asynchronous ``/compact`` path lands.
        # Skipping ``append_history`` here loses tape observability of
        # every synchronous overflow-recovery compaction.
        started = runtime.CompactStarted()
        self.runtime.append_history(started)
        self.publish(started)
        try:
            override = await self._agent_compactor.compact(
                tape_snapshot,
                context_snapshot,
                self._agent_model,
                self.runtime.mint_ref,
                custom_instructions=None,
            )
            # Adoption validates splice carry-forward and must report failures
            # through the normal CompactFailed path. Widen only over the
            # captured tape; later arrivals were absent from the summary.
            override = widen_barrier_mask(
                override,
                tape_snapshot,
            )
            # A summary replaces the region it masks, so it is expected to be
            # shorter; the payload-carry check governs merging producers.
            self.runtime.adopt_record(override, discards_content=True)
        except Exception as exc:  # noqa: BLE001 -- Compaction calls the model, so failures route to warning or exception handling.
            log_exception_or_warning(
                logger,
                "synchronous compaction failed during overflow recovery",
                exc,
            )
            self.compaction_state.compact_failures += 1
            # Issue#316 #6: the failure is not conversation. The published
            # ``CompactFailed`` below is taped and rendered; a synthesized
            # ``[Compaction error: ...]`` ``UserMessage`` only polluted model
            # context.
            failed = runtime.CompactFailed(exception=exc, tape_len=tape_len)
            self.runtime.append_history(failed)
            self.publish(failed)
            self.last_compact_error = exc
            return False
        complete = runtime.CompactComplete.from_override(override)
        self.runtime.append_history(complete)
        self.publish(complete)
        self.last_compact_error = None
        self._last_input_tokens = 0
        self._last_measured_history = ()
        self.compaction_state.compact_failures = 0
        return True


def _latest_service_retry_at(
    events: Sequence[runtime.RuntimeEvent],
) -> float | None:
    """Return the latest ``ModelServiceSuspended.retry_at`` in ``events``."""
    for event in reversed(events):
        if isinstance(event, runtime.ModelServiceSuspended):
            return event.retry_at
    return None


# The renderer treats ``UserFacingError`` specially -- no ``ClassName:`` prefix, no
# traceback -- so this message is what the user actually reads after recovery exhausts.
# Keep it actionable: name the verbs (``/clear``, ``/compact``, ``/model``) so the halt
# screen tells the user what to do, not just what went wrong.
#
# Structured ``attempts`` / ``final_tokens`` ride along on the exception object for
# operators inspecting logs; the user-facing message stays the polished remediation
# text. The underlying provider exception travels via ``__cause__`` at the raise site.
def _context_overflow_error(
    *,
    attempts: int = 0,
    final_tokens: int | None = None,
) -> ContextOverflowError:
    """Build the user-facing exhaustion error."""
    return ContextOverflowError(
        "Context window exhausted after auto-compaction. "
        "Use /clear to wipe history, /compact <hints> to retry with custom "
        "guidance, or /model to switch to a larger-window model.",
        attempts=attempts,
        final_tokens=final_tokens,
    )


# The runtime publishes its first ``AgentIdle`` at the top of the first ``run_forever``
# iteration -- before the agent has done any work. An empty history is the unambiguous
# marker of that boot edge. Shared by :meth:`Agent.drive_until_first_idle` (which must
# not return on boot) and the persistent-child forwarder (which must not spam a boot
# ping).
# The default result extractor for :meth:`Agent.drive_until_first_idle` when the caller
# supplies none. ``AgentSpawn`` passes its own ``AgentSend``-aware extractor instead.
def _default_last_assistant_result(
    history: list[runtime.ModelContextEvent],
) -> runtime.ToolResult:
    """Return the last ``AssistantMessage``'s text as a ``ToolResult``."""
    for entry in reversed(history):
        if isinstance(entry, runtime.AssistantMessage):
            return runtime.ToolResult(call_id="", content=entry.text)
    return runtime.ToolResult(call_id="", content="")


# The boundary between already-sent history (compactable) and this turn's fresh, un-
# sheddable entries: both the token-delta estimate and the byte gate split history here.
def _last_assistant_index(
    history: Sequence[runtime.ModelContextEvent],
) -> int | None:
    """Index of the last ``AssistantMessage`` in ``history``, or ``None``."""
    return next(
        (
            idx
            for idx in range(len(history) - 1, -1, -1)
            if isinstance(history[idx], runtime.AssistantMessage)
        ),
        None,
    )


# The wire carries base64 (``ceil(4/3)`` expansion), and the provider serializer resizes
# images to ``max_image_bytes`` before encoding while leaving PDFs/documents un-resized
# (see ``providers/*._attachment_block`` and ``sagent.lib.image.resize``). So per
# attachment the wire cost is:
#
# - image: ``4/3 * min(len(data), max_image_bytes)`` - PDF/other: ``4/3 * len(data)``
#
# Counting raw bytes (as the first cut did) both under-counts (ignores base64) and over-
# counts (ignores resize); this mirrors what actually ships so the byte gate compares
# wire-bytes to a wire-byte ceiling.
def _wire_attachment_bytes(
    messages: Sequence[runtime.ModelContextEvent],
    *,
    max_image_bytes: int,
) -> int:
    """Estimate the on-the-wire byte size of all attachments in ``messages``."""
    total = 0
    for entry in messages:
        if not isinstance(
            entry,
            (
                runtime.UserMessage,
                runtime.AgentSendMessage,
                runtime.ToolResult,
            ),
        ):
            continue
        for att in entry.attachments:
            raw = len(att.data)
            # Images are resized to ``max_image_bytes`` before encoding, so
            # clamp -- but ``max_image_bytes <= 0`` is the "no cap" sentinel
            # (0 = unlimited everywhere: ModelProfile, image_lib.resize, the
            # byte gate). ``min(raw, 0)`` would zero every image, so guard it.
            effective = (
                min(raw, max_image_bytes)
                if att.descriptor.startswith("image/") and max_image_bytes > 0
                else raw
            )
            # Exact base64 wire size is ``4 * ceil(n / 3)``, not the floored
            # ``n * 4 // 3`` -- the floor under-counts non-multiple-of-3
            # payloads, biasing the gate to fire one turn late.
            total += 4 * ((effective + 2) // 3)
    return total


# Sums attachment wire bytes (:func:`_wire_attachment_bytes`) and the UTF-8 byte length
# of every text surface that ships on the wire: the system prompt, each message's text /
# tool-result content, and -- for an ``AssistantMessage`` -- its ``thinking_blocks`` and
# ``tool_calls`` arguments (both serialized inline by the provider transports). Only
# tool-schema and JSON-framing bytes are omitted: they are bounded and roughly constant,
# so the estimate stays a conservative LOWER bound that never false-rejects a request
# the provider would accept.
#
# Callers must pass the MATERIALIZED messages (post :func:`materialize_request`), so
# elided tool results are sized at their wire placeholder rather than their pre-elision
# length -- otherwise the guard over-counts bytes the request never carries.
def _wire_request_bytes(
    messages: Sequence[runtime.ModelContextEvent],
    *,
    system: str,
    max_image_bytes: int,
) -> int:
    """Estimate the whole request body's wire byte size."""
    total = _wire_attachment_bytes(messages, max_image_bytes=max_image_bytes)
    total += len(system.encode("utf-8"))
    for entry in messages:
        if isinstance(
            entry,
            (runtime.UserMessage, runtime.AgentSendMessage),
        ):
            total += len(entry.text.encode("utf-8"))
        elif isinstance(entry, runtime.AssistantMessage):
            total += len(entry.text.encode("utf-8"))
            for block in entry.thinking_blocks:
                total += len(json.dumps(block, default=str).encode("utf-8"))
            for call in entry.tool_calls:
                total += len(json.dumps(call.args, default=str).encode("utf-8"))
        else:
            total += len(entry.content.encode("utf-8"))
    return total


# When the underlying compactor failure is itself classified as context overflow (true
# exhaustion: ``PromptTooLongError`` or a provider-specific overflow exception), return
# the polished :func:`_context_overflow_error` so the halt screen surfaces the
# ``/clear`` / ``/compact`` / ``/model`` remediation. Otherwise return the underlying
# exception verbatim so transport drops, auth failures, and other unrelated errors don't
# masquerade as context exhaustion -- the failure mode that misled session ``bc528d70``.
def _compact_failure_error(last_err: Exception, model: Model) -> Exception:
    """Pick the user-facing error after a failed ``compact_now``."""
    if model.is_context_overflow(last_err):
        return _context_overflow_error()
    return last_err


# ``None`` kwargs inherit from ``spec``. The model-id branch implements the cross-
# provider preservation rule documented on :meth:`Agent.change_model`.
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModelChange:
    """A built, not yet committed model swap; whoever holds it owns its resources."""

    model: Model
    """The new model."""

    recipe: ModelRecipe
    """How ``model`` was built."""

    provider: Provider
    """The provider ``model`` came from; the agent owns it once committed."""

    label: str
    """Human-readable ``old -> new`` label."""

    def discard(self) -> None:
        """Close the model and its provider; for a change that is not committed."""
        _ = _schedule_close(self.model, self.provider)


def _resolve_target_spec(
    spec: ModelRecipe,
    *,
    provider: str | None,
    auth: str | None,
    model_id: str | None,
    account: str | None,
) -> ModelRecipe:
    """Resolve a ``change_model`` kwargs payload to a complete ``types.model.ModelRecipe``."""
    prov_name = provider or spec.provider
    if auth is not None:
        final_auth = auth
    elif prov_name == spec.provider:
        final_auth = spec.auth
    else:
        final_auth = providers.default_auth_for_provider(
            prov_name,
        )
    final_account = account if account is not None else spec.account
    if model_id is not None:
        final_model_id = model_id
    elif prov_name == spec.provider or _provider_knows_model(prov_name, spec.model_id):
        final_model_id = spec.model_id
    else:
        final_model_id = last_models.get(prov_name) or "default"
    return ModelRecipe(
        provider=prov_name,
        auth=final_auth,
        account=final_account,
        model_id=final_model_id,
    )


def _provider_knows_model(prov_name: str, model_id: str) -> bool:
    """Return True when the provider class's catalog includes ``model_id``."""
    cls = providers.provider_class(prov_name)
    if not isinstance(cls, ModelResolver):
        return False
    try:
        cls.catalog.resolve(model_id)
    except (UnknownModelError, UnsupportedTagError):
        return False
    return True


# ``Model.close()`` is a required contract method: CLI-style providers
# (``AnthropicCLI``, ``GoogleCLI``) tear down their subprocess pools; API providers
# close their SDK/HTTP client; resource-free models no-op. Schedule the teardown on the
# running loop so the prior subprocess and its warming-spare task don't outlive the
# swap. No-op when no event loop is running (e.g. ``Agent.resume`` before
# ``serve_forever``): the model hasn't been used yet so there is nothing to close.
def _schedule_close(
    model: Model,
    provider: Provider | None,
) -> asyncio.Task[None] | None:
    """Schedule async teardown and return its task when a loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(_close_model(model, provider))
    task.add_done_callback(
        log_task_exception(logger, "model close failed"),
    )
    return task


# The model first, then the provider: a model may still hold a request on the client
# the provider owns, and ``Provider.close_sdk`` puts the client's teardown with whoever
# built the provider.
async def _close_model(model: Model, provider: Provider | None) -> None:
    """Close ``model``, then the provider it came from when this agent owned it."""
    try:
        await model.close()
    finally:
        if provider is not None:
            await provider.close_sdk()


def _schedule_provider_close(provider: Provider) -> asyncio.Task[None] | None:
    """Schedule ``close_sdk`` on a provider whose model was never adopted."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(provider.close_sdk())
    task.add_done_callback(log_task_exception(logger, "provider close failed"))
    return task


class ObservedModel:
    """Record every completed call through a rich model, including discarded ones.

    Foreground turns, compaction, and scrunch all receive this wrapper, so no
    path can drop billed usage. Validation is ``send_with_retry``'s job, the
    one boundary every model call crosses. Delegated metadata and token
    helpers retain the wrapped model's exact behavior.
    """

    def __init__(
        self,
        inner: Model,
        observe: Callable[[ModelResponse], None],
    ) -> None:
        self._inner = inner
        self._observe = observe

    @property
    def capability(self) -> ModelCapability:
        """Return the wrapped model's capability."""
        return self._inner.capability

    @property
    def settings(self) -> ModelSettings:
        """Return the wrapped model's selected settings."""
        return self._inner.settings

    @property
    def limits(self) -> ModelLimits:
        """Return the wrapped model's active limits."""
        return self._inner.limits

    @property
    def tagged_model_id(self) -> str:
        """Return the wrapped model's display id."""
        return self._inner.tagged_model_id

    def approx_text_tokens(self, text: str) -> int:
        """Delegate local text-token estimation."""
        return self._inner.approx_text_tokens(text)

    def approx_image_tokens(self, data: bytes) -> int:
        """Delegate local image-token estimation."""
        return self._inner.approx_image_tokens(data)

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Delegate local request-token estimation."""
        return self._inner.approx_request_tokens(request)

    async def actual_text_tokens(self, text: str) -> int:
        """Delegate exact text-token measurement."""
        return await self._inner.actual_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Delegate exact image-token measurement."""
        return await self._inner.actual_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Delegate exact request-token measurement."""
        return await self._inner.actual_request_tokens(request)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Buffer one response, then account for it.

        Args:
          request: Provider request to buffer.

        Returns:
          response: Accounted provider response.

        """
        try:
            response = await self._inner.buffer(request)
        except StreamInterruptedError as exc:
            self._observe(exc.response)
            raise
        self._observe(response)
        return response

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[runtime.RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Stream one response, then account for it.

        Args:
          request: Provider request to stream.
          publish: Optional sink for live runtime events.

        Returns:
          response: Accounted provider response.

        """
        try:
            response = await self._inner.stream(request, publish)
        except StreamInterruptedError as exc:
            self._observe(exc.response)
            raise
        self._observe(response)
        return response

    def spend(self, tokens: TokenCount) -> TokenCost:
        """Delegate token pricing."""
        return self._inner.spend(tokens)

    def is_context_overflow(self, error: Exception) -> bool:
        """Delegate context-overflow classification."""
        return self._inner.is_context_overflow(error)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Delegate provider retry classification."""
        return self._inner.is_retryable_provider_error(error)

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Delegate normalized rate-limit telemetry."""
        return self._inner.usage_snapshot()

    async def close(self) -> None:
        """Delegate resource teardown."""
        await self._inner.close()


class _AgentModel:
    """Bridges rich provider ``Model`` to runtime ``Model`` protocol.

    Args:
      inner: Rich provider model to wrap.
      agent: Owning ``Agent``; supplies request knobs and cost tracker.

    """

    def __init__(self, inner: Model, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    def set_inner(self, inner: Model) -> None:
        """Swap the wrapped model. Used by ``Agent.swap_model``.

        Args:
          inner: New rich provider model to wrap.

        """
        self._inner = inner

    async def stream(
        self,
        history: list[runtime.ModelContextEvent],
        publish: Callable[[runtime.RuntimeEvent], None],
    ) -> runtime.AssistantMessage:
        """Stream a response with retry + context-overflow recovery.

        Builds a ``types.model.ModelRequest`` from agent state, runs
        ``send_with_retry`` with persistent-mode + overflow recovery,
        records cost out-of-band, returns the final assistant message.

        ``system`` and ``tools`` come from agent state (not the runtime)
        so callable system specs (e.g. cwd-aware ``environment``) and
        registry mutations stay live across the call.

        Args:
          history: Conversation history (resolved view passed by runtime).
          publish: Runtime event sink for every streamed event (text,
              thinking, and CLI tool labels).

        Returns:
          message: Final ``types.runtime.AssistantMessage`` from the provider.

        Raises:
          ContextOverflowError: Overflow recovery failed after the retry
              cap (``MAX_OVERFLOW_RECOVERY`` attempts), or the
              proactive / reactive ``compact_now`` itself failed with a
              classified-overflow underlying error.
          Exception: Underlying provider exception surfaced verbatim on
              non-overflow compaction failures, when retry classification
              gave up, or when the stream raised a non-retryable error.

        """
        # Proactive compaction: ask the compactor whether headroom is
        # exhausted BEFORE handing the prompt to the provider. Without
        # this gate, ``compact_now`` only runs reactively (after a
        # provider 400 classified as overflow) -- which costs a full
        # round trip and depends on the provider rejecting the request
        # at all. Some providers accept oversized prompts up to a hard
        # ceiling, so the reactive path leaves a wide window where the
        # buffer's headroom (``AgentSettings.buffer_tokens``) is paid
        # for but never spent. ``compact_now`` appends a barrier override
        # to the runtime tape; we refetch the resolved view below.
        if not await self._agent.compact_if_needed(history, self._inner):
            self._raise_compact_failure()
        # Compaction may have appended a barrier override; refetch the
        # resolved view so subsequent attempts in this call see it.
        history = self._agent.runtime.context().messages

        # Compute the system prompt once per ``stream`` call. Each tool's
        # ``prompt()`` may capture cwd / registry / persistent-IPC state,
        # but none of that state changes between overflow-recovery
        # attempts within a single stream call -- so rebuilding up to
        # ``MAX_OVERFLOW_RECOVERY + 1`` times per call is wasted work.
        # ``Agent.system_prompt()`` callers outside this loop continue to
        # observe live state via ``_build_system`` on demand. Computed before
        # the pre-send guard so the guard can count the system-prompt bytes.
        cached_system = self._agent.system_prompt()

        # Pre-send byte guard: compaction has now shed every attachment it
        # can (the summarized region). If the request STILL exceeds the
        # provider's byte ceiling, the overage is in un-sheddable bytes --
        # this turn's own fresh input (a user-pasted oversize PDF that bypasses
        # the read tool's rendered-byte bound, or a text-heavy fresh request).
        # Sending it would burn ``MAX_OVERFLOW_RECOVERY + 1`` full round-trips
        # before the identical byte-overflow exhaustion. Reject up front with
        # the typed error so the user gets immediate, actionable feedback.
        #
        # Measure the MATERIALIZED request (same artifact ``send_with_retry``
        # ships), so tool results elided by ``tool_result_budget_tokens`` are
        # sized at their wire placeholder -- not their pre-elision length.
        # Byte accounting always runs over that materialized view.
        # ``_wire_request_bytes`` counts
        # attachments and every text surface but omits tool-schema / JSON
        # framing, so it stays a conservative lower bound that never
        # false-rejects a request the provider would accept; a genuine miss
        # still hits the reactive 413.
        max_request_bytes = self._inner.limits.max_request_bytes
        if max_request_bytes > 0:
            guard_messages = materialize_request(
                ModelRequest(messages=list(history)),
                tool_result_budget_tokens=self._agent.tool_results.message_budget_tokens,
            ).messages
            wire_bytes = _wire_request_bytes(
                guard_messages,
                system=cached_system,
                max_image_bytes=self._inner.limits.max_image_bytes,
            )
            if wire_bytes > max_request_bytes:
                raise RequestTooLargeError(
                    f"Request payload ({wire_bytes:,} bytes) exceeds the provider "
                    f"limit ({max_request_bytes:,} bytes) and cannot be compacted "
                    "away (it is this turn's own input). Attach a smaller file, send "
                    "less text, or read large PDFs/images in smaller page ranges.",
                )

        # Wrap every rich tool with ``BackgroundAwareTool`` so the
        # provider-visible schema advertises the ``background`` /
        # ``delay`` properties without polluting the raw tool's
        # identity.
        rich_tools = self._agent.live_tools()

        # Consume the persisted resume deadline ONCE, before the loop. It is a
        # one-shot wall-clock wait carried across a process restart (a prior
        # ``ModelServiceSuspended`` retry_at); the first ``send_with_retry``
        # honors it, and any overflow-recovery re-attempt must NOT replay a
        # wait the first send already satisfied. Clearing it up front makes the
        # single-use semantic explicit rather than relying on it being nulled
        # mid-iteration.
        resume_retry_at = self._agent.runtime.resume_retry_at
        self._agent.runtime.resume_retry_at = None
        observed_model = ObservedModel(self._inner, self._agent.record_response)

        for attempt in range(MAX_OVERFLOW_RECOVERY + 1):
            request = materialize_request(
                ModelRequest(
                    messages=list(history),
                    system=cached_system,
                    tools=rich_tools,
                    max_response_tokens=self._agent.max_response_tokens,
                ),
                tool_result_budget_tokens=self._agent.tool_results.message_budget_tokens,
            )
            # Hand the one-shot resume wait to THIS attempt only, then clear it
            # unconditionally -- before the send can raise. ``send_with_retry``
            # honors the wait at its top (before the first stream), so by the
            # time it returns OR raises (overflow, auth, anything) the wait is
            # already spent. Clearing on the success path alone would replay the
            # wait on every overflow-recovery iteration.
            attempt_resume_retry_at = resume_retry_at
            resume_retry_at = None
            try:
                response = await send_with_retry(
                    observed_model,
                    request,
                    publish=publish,
                    max_attempts=self._agent.max_attempts,
                    persistent_retry=self._agent.persistent_retry,
                    publish_recoverable=lambda text: logger.info(
                        "recoverable: %s",
                        text,
                    ),
                    on_service_suspended=self._agent.publish_service_suspended,
                    resume_retry_at=attempt_resume_retry_at,
                )
            except PolicyBlockedError:
                self._agent.withhold_refused_tail()
                raise
            except ModelTerminationError as exc:
                if exc.stop_reason == "model_refusal":
                    self._agent.withhold_refused_tail()
                raise
            except Exception as exc:
                # Two distinct overflow conditions route to the same
                # recovery action (compaction sheds history tokens AND
                # attachment bytes), but carry different exhaustion
                # remediation:
                #   - token-context overflow (``is_context_overflow``):
                #     a larger-window model helps.
                #   - byte wire-limit (``RequestTooLargeError``, the ~32MB
                #     ceiling driven by attachment bytes): a larger window
                #     does NOT help -- the byte ceiling is the same. Keyed
                #     on the typed error the provider raises rather than a
                #     Model-protocol method, so it stays provider-agnostic.
                # Provider-side normalization can slip (e.g. unusual HTTP
                # status carrying overflow body text), so the canonical
                # signal for the token case is ``is_context_overflow``.
                byte_overflow = isinstance(exc, RequestTooLargeError)
                if not byte_overflow and not self._inner.is_context_overflow(exc):
                    raise
                if attempt >= MAX_OVERFLOW_RECOVERY:
                    if byte_overflow:
                        # No ``/model`` advice: the byte ceiling is fixed
                        # across models, so a wider window cannot relieve it.
                        raise RequestTooLargeError(
                            "Request exceeds the provider byte limit even after"
                            " auto-compaction. Use /clear to wipe history, or"
                            " re-read large PDFs/images in smaller page ranges"
                            " so fewer attachment bytes ship at once.",
                        ) from exc
                    raise _context_overflow_error(attempts=attempt) from exc
                logger.info(
                    "%s recovery attempt %d",
                    "byte-overflow" if byte_overflow else "context-overflow",
                    attempt,
                )
                # Short-circuit when compaction itself failed -- looping
                # to retry the model on unchanged (or slightly longer)
                # history burns the retry budget on the same 400 (the
                # BUGS34 regression: three identical "Compaction failed"
                # lines followed by a cryptic RuntimeError).
                if not await self._agent.compact_now():
                    self._raise_compact_failure()
                # Refetch resolved view: ``compact_now`` appended a
                # barrier override.
                history = self._agent.runtime.context().messages
                continue
            return response.message
        raise AssertionError("overflow recovery returns or raises on its last attempt")

    def _raise_compact_failure(self) -> NoReturn:
        """Raise the user-facing error for a compaction that returned False."""
        last_err = self._agent.last_compact_error
        if last_err is None:
            raise ValueError("compact_now sets the error on every False return")
        raise _compact_failure_error(last_err, self._inner) from last_err


class _AgentTool:
    """Bridges raw rich ``Tool`` to the runtime's lean ``Tool`` protocol.

    Per call, in order:

    - Pop the ``BackgroundAwareTool``-injected ``background`` / ``delay``
      keys from ``args`` so they don't reach the raw tool's schema
      validation or runtime.
    - Pre-validate ``args`` against the raw tool's
      ``directive_schema`` (required fields, ``additionalProperties``
      bound). Validation errors surface as
      ``types.runtime.ToolResult(is_error=True)`` with an ``InputValidationError:``
      header carrying the recovery hint so the model can self-correct
      without looping.
    - Publish ``types.runtime.ToolLabel`` for the REPL renderer.
    - When ``background`` / ``delay`` is set: spawn a detached task that
      will eventually post ``types.runtime.DetachedResult`` to the runtime inbox; the
      synchronous return is a ``[Running in background: <name>]``
      placeholder. That placeholder stays the answer to this call; the real
      result arrives later as NEW forward context (a synthetic
      ``DetachedArrived`` pair), so nothing the model already read is
      rewritten underneath it.
    - Otherwise: await the raw tool's ``run``, then post-process
      (empty-result marker, oversized-content disk offload).

    Args:
      inner: Raw rich tool (no schema injection); validated against
          its own ``directive_schema``.
      agent: Owning ``Agent`` used for publishing events, registering
          background tasks, and looking up session-dir / budget for
          result persistence.

    """

    def __init__(self, inner: Tool, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    def set_inner(self, inner: Tool) -> None:
        """Swap the wrapped tool. Used by :meth:`Agent.replace_tool`.

        Args:
          inner: New raw rich tool to wrap.

        """
        self._inner = inner

    @property
    def name(self) -> str:
        """Forward to the wrapped tool's name."""
        return self._inner.name

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Forward to the wrapped tool's serialization key.

        The runtime dispatches this wrapper, not the raw tool, so
        same-file grouping depends on the key reaching it here.

        Args:
          args: Directive arguments parsed by the runtime.

        Returns:
          key: The inner tool's serialization key, or ``None``.

        """
        return self._inner.serialize_key(args)

    async def run(self, args: Mapping[str, object]) -> runtime.ToolResult:
        """Validate, publish ``types.runtime.ToolLabel``, dispatch, post-process.

        Args:
          args: Directive arguments parsed by the runtime; includes the
              ``BackgroundAwareTool``-injected ``background`` / ``delay``
              keys when the model requested asynchronous execution.

        Returns:
          result: Tool result, an input-validation error, or a
              ``[Running in background: <name>]`` placeholder.

        """
        # ``current_call_id_var`` is set by the runtime's
        # ``_run_tool_and_post`` before dispatch; an empty fallback only
        # appears when this method is invoked directly (unit tests).
        # Production callers always have a non-empty id, so emitted
        # ``ToolLabel`` / ``ToolResult`` records correlate to the
        # originating assistant tool_use.
        call_id = current_call_id_var.get("")
        # The raw args are validated against the schema they were offered, control
        # keys included, so a mistyped ``delay`` is an input error rather than a
        # decode failure inside ``split_bg_args``. A tool that owns a control key
        # gets every argument untouched.
        controlled = backgroundable(self._inner)
        validation_error = validate_tool_input(
            self._inner.name,
            bg_augmented_schema(self._inner.directive_schema)
            if controlled
            else self._inner.directive_schema,
            args,
        )
        bg_requested, delay_sec, clean_args = (
            split_bg_args(args) if controlled else (False, 0.0, dict(args))
        )
        # A model can send the keys without being offered them; the host that forbade
        # them is counting on the call to end with the turn.
        if not self._agent.allow_background:
            bg_requested, delay_sec = False, 0.0
        if validation_error is not None:
            # Publish a label even on validation failure so scrollback stays
            # consistent with every other tool outcome (each is preceded by a
            # dim tool-call line). The label renders as a plain dim line, not a
            # "running" indicator. Use the tool name rather than ``summary``:
            # the args failed validation, so ``summary`` may choke on them.
            self._agent.runtime.publish(
                runtime.ToolLabel(call_id=call_id, text=self._inner.name),
            )
            return runtime.ToolResult(
                call_id=call_id,
                content=validation_error,
                is_error=True,
            )
        try:
            label = self._inner.summary(clean_args)
        except Exception:
            # ``summary`` is best-effort UX (renders ahead of execution).
            # Any failure -- ``KeyError`` when the model omits an optional
            # arg, ``TypeError`` from a stale signature, an author bug --
            # must not lose the ``ToolLabel`` record or surface as a
            # ``ToolResult(is_error=True)`` with no human label. Fall back
            # to the tool's name and log so author bugs are still
            # findable in operator logs.
            logger.exception("tool %r summary() failed", self._inner.name)
            label = self._inner.name
        self._agent.runtime.publish(
            runtime.ToolLabel(call_id=call_id, text=label),
        )
        if bg_requested or delay_sec > 0:
            job_id = self._agent.job_id_for_call(call_id)
            task = asyncio.create_task(
                self._run_bg(call_id, job_id, clean_args, delay_sec),
            )
            task.add_done_callback(
                log_task_exception(
                    logger,
                    f"background tool {self._inner.name!r} crashed",
                ),
            )
            self._agent.register_background(
                job_id,
                BackgroundTaskEntry(
                    task=task,
                    tool_name=self._inner.name,
                    queue_id=job_id,
                    call_id=call_id,
                    started=time.time(),
                    delay_sec=delay_sec,
                    kind="tool",
                ),
            )
            return runtime.ToolResult(
                call_id=call_id,
                content=f"{runtime.RUNNING_PREFIX}{self._inner.name}]",
                kind=runtime.ToolResultKind.PENDING,
            )
        result = await self._inner.run(clean_args)
        if not result.call_id:
            result = dataclasses.replace(result, call_id=call_id)
        result = self._inject_conditional_rules(result, clean_args)
        return self._agent.bound_result(result, tool=self._inner)

    def _inject_conditional_rules(
        self,
        result: runtime.ToolResult,
        args: Mapping[str, object],
    ) -> runtime.ToolResult:
        """Prepend matching ``paths:`` AGENTS.md rules to file-tool results."""
        if result.is_error or self._inner.name not in {"Read", "Edit", "Write"}:
            return result
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            return result
        state = self._agent.tool_state
        cwd = Path(state.bash_cwd)
        path = Path(raw_path)
        if not path.is_absolute():
            path = cwd / path
        reminder, matched = agents_md.conditional_rules_for_paths(
            cwd,
            [path],
            config=agents_md.AgentsMdConfig(
                additional_dirs=[Path(d) for d in state.additional_dirs],
            ),
            exclude=state.invoked_rules,
        )
        if not reminder:
            return result
        state.invoked_rules.update(matched)
        return dataclasses.replace(result, content=f"{reminder}\n{result.content}")

    async def _run_bg(
        self,
        call_id: str,
        job_id: str,
        args: Mapping[str, object],
        delay_sec: float,
    ) -> None:
        """Background-task body: optional sleep, run inner, post result."""
        try:
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)
            result = await self._inner.run(args)
            if not result.call_id:
                result = dataclasses.replace(result, call_id=call_id)
            result = self._inject_conditional_rules(result, args)
            processed = self._agent.bound_result(result, tool=self._inner)
        except asyncio.CancelledError:
            # Two cancellation paths, distinguished by registry membership
            # at cancel time (checked below):
            #
            # 1. Registry-popped first (``cancel_background`` / ``/kill``,
            #    and ``clear`` via ``_cancel_all_background``): the job is
            #    already gone from ``background``. The absence is the cleanup
            #    signal -- consume the cancellation and exit normally so the
            #    kill verb's cleanup path stays synchronous.
            #
            # 2. Registry still populated (``Agent.shutdown``, which cancels
            #    ``job.task`` WITHOUT popping first; or a raw external
            #    ``task.cancel``): post a ``[cancelled]`` ``DetachedResult``
            #    so the splice site sees a paired result for the assistant's
            #    tool_use, then re-raise so the cancel chain reaches the
            #    scheduler -- asyncio requires every ``CancelledError`` catch
            #    to re-raise or be the explicit endpoint of the cancel chain
            #    (only the registry-popped path is that endpoint).
            if job_id not in self._agent.background:
                logger.debug(
                    "background tool %r cancelled via registry pop (job_id=%s)",
                    self._inner.name,
                    job_id,
                )
                return
            logger.warning(
                "background tool %r cancelled externally with job_id=%s still"
                " in registry; pairing tool_use with [cancelled] and re-raising",
                self._inner.name,
                job_id,
            )
            self._agent.runtime.inbox.push_back(
                runtime.DetachedResult(
                    result=runtime.ToolResult(
                        call_id=call_id,
                        content=runtime.CANCELLED_PLACEHOLDER,
                        is_error=True,
                        kind=runtime.ToolResultKind.CANCELLED,
                    ),
                ),
            )
            self._agent.forget_background(job_id)
            raise
        except Exception as exc:
            logger.exception("background tool %r failed", self._inner.name)
            processed = runtime.ToolResult(
                call_id=call_id,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        # Order matters and must stay await-free: push the result, THEN
        # drop the registry entry. The two statements run atomically under
        # cooperative scheduling (no ``await`` between), so ``_fully_drained``
        # can never observe an empty inbox AND an empty ``_bg`` mid-handoff.
        # Even were they to interleave, the ordering is belt-and-suspenders:
        # after ``push_back`` the inbox is non-empty (gate stays False); the
        # registry only clears once the result is already queued. Inserting an
        # ``await`` between these lines would open a spurious-``AgentIdle``
        # window -- do not.
        self._agent.runtime.inbox.push_back(
            runtime.DetachedResult(result=processed),
        )
        self._agent.forget_background(job_id)


class _AgentCompactor:
    """Bridges rich ``Compactor`` to the runtime's lean ``compact`` protocol.

    Threads ``custom_instructions`` from the runtime's compact args.
    Runs the post-compact enrich pipeline after the inner compactor
    returns.

    Args:
      inner: Rich compactor whose ``compact`` is delegated to.
      agent: Owning ``Agent``; supplies budget, tools, tool state.

    """

    def __init__(self, inner: Compactor, agent: Agent) -> None:
        self._inner = inner
        self._agent = agent

    def largest_context(self, budget: AgentSettings) -> int:
        """Delegate to the inner compactor.

        Forwarded for the same reason as :meth:`should_compact`: the two
        are one heuristic, so a wrapper exposing only the predicate would
        let a caller sizing against the scalar reach a different strategy
        than the caller asking the question.

        Args:
          budget: Agent settings used to calculate the largest context.

        Returns:
          tokens: Largest context size supported by the budget.

        """
        return self._inner.largest_context(budget)

    def should_compact(
        self,
        current_tokens: int,
        largest_context: int,
        system_tokens: int = 0,
    ) -> bool:
        """Delegate to the inner compactor.

        The runtime's lean ``Compactor`` Protocol does not include
        ``should_compact`` (the runtime never asks; compaction is
        explicitly driven by ``Compact`` / ``Recompact`` events). The
        Agent layer's :class:`_AgentModel` invokes this on the wrapper
        directly to gate proactive compaction ahead of each provider
        call.

        Args:
          current_tokens: Estimated tokens currently in the request.
          largest_context: Maximum context accepted by the budget.
          system_tokens: Estimated system-prompt token count.

        Returns:
          should_compact: Whether the current context needs compaction.

        """
        return self._inner.should_compact(
            current_tokens=current_tokens,
            largest_context=largest_context,
            system_tokens=system_tokens,
        )

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[runtime.ModelContextEvent],
        model: object,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Run the rich compactor and apply post-compact enrichment.

        Args:
          tape: Append-only session tape.
          context: Resolved provider-facing context to compact.
          model: Runtime-side model (ignored; rich model used directly).
          mint_ref: Factory for fresh ``TapeRef`` values.
          custom_instructions: Free-form compaction instructions
              forwarded to the rich inner compactor.

        Returns:
          override: Barrier override carrying the compacted payload,
              ready for the runtime to append to ``runtime.tape``.

        """
        del model  # _AgentCompactor uses the agent's rich model directly.
        override = await self._inner.compact(
            tape=tape,
            context=context,
            model=ObservedModel(self._agent.model, self._agent.record_side_response),
            mint_ref=mint_ref,
            custom_instructions=custom_instructions,
        )
        payload: list[runtime.ModelContextEvent] = list(override.payload)
        # Memoize system prompt + live tools for the two token-estimate
        # request builds below: agent state does not change inside one
        # ``compact`` call, so rebuilding both per estimate is wasted work.
        cached_system = self._agent.system_prompt()
        cached_tools = self._agent.live_tools()

        # Post-compact enrich operates on the override's mutable payload
        # before the runtime freezes and appends. Split into two
        # try/except blocks so the failure message names the subsystem
        # that actually raised: an estimate failure must not be blamed on
        # ``post_compact_enrich``.
        try:
            used = self._agent.model.approx_request_tokens(
                materialize_request(
                    ModelRequest(
                        messages=payload,
                        system=cached_system,
                        tools=cached_tools,
                    ),
                    tool_result_budget_tokens=self._agent.tool_results.message_budget_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- The estimator invokes provider classification during recovery, so failures route to warning or exception handling.
            log_exception_or_warning(
                logger,
                "post-compact token estimate failed; skipping enrich",
                exc,
            )
        else:
            # The complement of ``largest_context``: what the strategy holds
            # back is exactly what enrich may not spend.
            headroom = self._agent.max_request_tokens - self._inner.largest_context(
                self._agent.budget,
            )
            try:
                await post_compact_enrich(
                    history=payload,
                    tool_state=self._agent.tool_state,
                    reattach=self._inner.reattach,
                    estimate_text_tokens=self._agent.model.approx_text_tokens,
                    tools=self._agent.tools_map,
                    background_tasks=self._agent.background,
                    estimate_tokens=self._agent.max_request_tokens - used,
                    headroom=headroom,
                )
            except Exception as exc:  # noqa: BLE001 -- Post-compaction enrichment calls the model, so failures route to warning or exception handling.
                log_exception_or_warning(
                    logger,
                    "post_compact_enrich failed; continuing",
                    exc,
                )
        # Carry the producer's external-pair declaration into the repair: a
        # preserved parent whose tool is still detached is answered by the
        # ``[detached]`` stub on the tape, not by a synthetic interrupt.
        payload = _repair_compact_payload(payload, override.paired_externally)
        if not payload or isinstance(payload[-1], runtime.AssistantMessage):
            payload.append(runtime.UserMessage(text="[continuation]"))

        # Scrunch rescue: when the producer's normal output still
        # exceeds the agent's input budget, partition the payload
        # oldest-first and re-run the producer per partition until the
        # resolved view fits. The same ``largest_context`` the proactive
        # gate and the willRetriggerNextTurn check below consume, so a
        # payload cannot skip scrunch yet be flagged as already over
        # threshold. Without this, the next provider call sees the same
        # overflow that triggered compaction and the recovery loop wedges.
        target = self._inner.largest_context(self._agent.budget)
        # A degenerate budget (response + buffer >= window) leaves no room
        # to scrunch into; ``scrunch_to_fit`` rejects a non-positive
        # partition cap. Skip rather than raise -- the willRetriggerNextTurn
        # annotation below still flags the payload as oversized.
        if target > 0:
            try:
                payload_tokens_for_scrunch = self._agent.model.approx_request_tokens(
                    materialize_request(
                        ModelRequest(
                            messages=payload,
                            system=cached_system,
                            tools=cached_tools,
                        ),
                        tool_result_budget_tokens=(
                            self._agent.tool_results.message_budget_tokens
                        ),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- The estimator invokes provider classification during recovery.
                log_exception_or_warning(
                    logger,
                    "pre-scrunch token estimate failed; skipping scrunch",
                    exc,
                )
            else:
                if payload_tokens_for_scrunch > target:
                    # ``scrunch_to_fit`` measures messages only, but the gate
                    # above measured the whole request. Reserve the fixed
                    # system+tools overhead so the messages-only budget
                    # scrunch fits leaves room for it; otherwise a payload
                    # whose messages already fit produces no reduction and the
                    # next provider call re-overflows on the same overhead.
                    overhead = payload_tokens_for_scrunch - estimate_entry_tokens(
                        self._agent.model,
                        payload,
                    )
                    scrunch_target = max(1, target - max(0, overhead))
                    payload = await self._scrunch_payload(
                        payload=payload,
                        mint_ref=mint_ref,
                        target_input_tokens=scrunch_target,
                    )
                    # Scrunch re-runs the producer per partition and may
                    # emit a fresh ``AssistantMessage`` whose tool_calls
                    # have no local ``ToolResult``. Re-repair so the final
                    # payload keeps the wire-pairing invariant the pre-scrunch
                    # repair established; otherwise ``unpaired_call_ids``
                    # below would declare those ids ``paired_externally``
                    # with no real external partner and the next provider
                    # call would 400.
                    payload = _repair_compact_payload(
                        payload,
                        override.paired_externally,
                    )
                    if not payload or isinstance(payload[-1], runtime.AssistantMessage):
                        payload.append(runtime.UserMessage(text="[continuation]"))

        # ``willRetriggerNextTurn`` prediction: ask the inner compactor
        # whether the new payload would already cross its own
        # auto-compact threshold; if so, the next turn would re-trigger
        # compaction immediately -- likely an infinite loop. Mark
        # ``fallback_reason`` so observers can surface the condition
        # without blocking the current call. Routing through the
        # compactor's ``should_compact`` (rather than rebuilding the
        # threshold inline) keeps proactive-gate and retrigger-prediction
        # in lockstep: any compactor that overrides headroom math stays
        # the single source of truth.
        fallback_reason = override.fallback_reason
        try:
            payload_tokens = self._agent.model.approx_request_tokens(
                materialize_request(
                    ModelRequest(
                        messages=payload,
                        system=cached_system,
                        tools=cached_tools,
                    ),
                    tool_result_budget_tokens=self._agent.tool_results.message_budget_tokens,
                ),
            )
            would_retrigger = self._inner.should_compact(
                current_tokens=payload_tokens,
                largest_context=self._inner.largest_context(self._agent.budget),
                system_tokens=self._agent.model.approx_text_tokens(cached_system),
            )
            if would_retrigger:
                msg = (
                    f"compacted payload ({payload_tokens} tok) already exceeds"
                    f" auto-compact threshold; next turn would re-trigger"
                    f" compaction"
                )
                logger.warning(msg)
                fallback_reason = (
                    f"{fallback_reason}; {msg}" if fallback_reason else msg
                )
        except Exception as exc:  # noqa: BLE001 -- The estimator invokes provider classification during recovery..
            log_exception_or_warning(
                logger,
                "willRetriggerNextTurn estimate skipped; continuing",
                exc,
            )

        if (
            tuple(payload) == override.payload
            and fallback_reason == override.fallback_reason
        ):
            return override
        # ``dataclasses.replace`` re-runs ``ContextSplice.__post_init__``,
        # which validates payload pairing. The validator treats
        # ``paired_externally`` strictly: an id appearing there is the
        # promise that its pair lives *outside* the payload, never a
        # generic "skip checks for this id". Recompute the field from
        # the post-rewrite payload (``_repair_compact_payload`` may
        # have synthesized local TRs for what the producer declared
        # external, in which case those ids are no longer external) so
        # the declaration stays honest.
        return dataclasses.replace(
            override,
            payload=tuple(payload),
            fallback_reason=fallback_reason,
            paired_externally=unpaired_call_ids(payload),
        )

    # The bridge runs scrunch as an internal rescue: scrunch's per- partition splices
    # land in a throwaway tape, the final resolved view is extracted, and the bridge
    # folds it into a single return payload. The runtime sees one combined splice, not
    # the per-pass audit trail -- scrunch's structured-summary calls already log per-
    # pass diagnostics.
    async def _scrunch_payload(
        self,
        *,
        payload: list[runtime.ModelContextEvent],
        mint_ref: Callable[[], TapeRef],
        target_input_tokens: int,
    ) -> list[runtime.ModelContextEvent]:
        """Apply the scrunch maneuver to ``payload`` until it fits."""
        # Build a throwaway tape mirroring ``payload`` so scrunch can
        # reference its entries by ordinal. The runtime never sees these
        # refs: ``scratch_tape`` is local and never escapes, so the
        # process-local ``scratch_session`` may collide with a prior
        # scrunch's id (``id(payload)`` reuses a freed address) with no
        # externally visible effect.
        scratch_session = f"scrunch-{id(self)}-{id(payload)}"
        scratch_tape: list[TapeRecord] = [
            ReferrableTapeEvent(
                ref=TapeRef(session_id=scratch_session, ordinal=i),
                event=entry,
            )
            for i, entry in enumerate(payload)
        ]
        try:
            result = await scrunch_to_fit(
                context=payload,
                tape=scratch_tape,
                model=ObservedModel(
                    self._agent.model,
                    self._agent.record_side_response,
                ),
                compactor=self._inner,
                mint_ref=mint_ref,
                target_input_tokens=target_input_tokens,
                max_partition_tokens=target_input_tokens,
            )
        except ScrunchTooLargeError as exc:
            # Scrunch hit its forward-progress floor: typically a
            # single message larger than the partition cap. The
            # bridge surfaces the failure to the caller via the
            # ``fallback_reason`` channel rather than raising; the
            # next provider call may still overflow, but at least the
            # bridge returns a well-formed splice.
            logger.warning("scrunch could not make progress: %s", exc)
            return payload
        # ``scrunch_to_fit`` returns the final resolved view it tracked
        # internally. Consume it directly: re-deriving the view from
        # ``result.splices`` would compose the per-pass masks under the
        # resolver's cover-the-cover undelete semantics and resurrect
        # content an earlier pass folded away (a later pass masks an
        # earlier pass's splice ref). The executor's flat replacement is
        # the ground truth.
        return list(result.view)


def _repair_compact_payload(
    payload: Sequence[runtime.ModelContextEvent],
    paired_externally: frozenset[str] = frozenset(),
) -> list[runtime.ModelContextEvent]:
    """List-wrap :func:`tape.splice_safe_repair` so the caller can append."""
    return list(splice_safe_repair(payload, paired_externally=paired_externally))


# Two cancel callers differ in scope:
#
# - ``tools_only`` -- ``Agent._cancel_all_background`` (called from ``clear()`` and
# ``kill_all_tools()``). Drops only user-scheduled ``background: true`` tool jobs;
# ``detached`` (cohort-decayed) and ``subagent`` survive because those have their own
# lifecycle owners. - ``all`` -- ``Agent.shutdown``. Drops every visible job whose owner
# is the exiting process; a serviced subagent owns its own ``serve_forever`` and shuts
# itself down, so it is exempt. A oneshot subagent has self-stopped after its first
# result, so it is effectively gone already.
#
# Both modes skip ``hidden=True`` infra (REPL pump, watchdogs).
def _should_cancel_background(
    job: BackgroundTaskEntry,
    *,
    mode: Literal["tools_only", "all"],
) -> bool:
    """Decide whether a background ``job`` should be cancelled."""
    if job.hidden:
        return False
    if mode == "tools_only":
        return job.kind == "tool"
    if job.task.done():
        return False
    return not (job.kind == "subagent" and job.lifecycle == "serviced")
