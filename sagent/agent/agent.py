"""Agent: conversation loop with tool dispatch, logging, sessions.

See ``agent/__init__.py`` for architecture (inbox zero, queue
contracts) and ``tools/__init__.py`` for Tool and Message theory.

Usage::

    from sagent.agent import Agent
    from sagent.providers import Anthropic
    from sagent.compactor import SummaryCompactor
    from sagent import tools

    provider = Anthropic.from_env()
    opus1m = provider.model("claude-opus-4-7+1m")

    # Simple - string system prompt. Bash takes its sibling tools as
    # ``peers`` so its ``[bash-lint]`` feature can suggest dedicated
    # replacements (Grep/Read/...) for invocations like ``grep foo .``.
    read = tools.Read()
    grep = tools.Grep()
    agent = Agent(
        model=opus1m,
        system="You are a scientist.",
        tools=[tools.Bash(peers=(read, grep)), read, grep],
    )

    # Sectioned - dict with static strings and dynamic callables
    read = tools.Read()
    grep = tools.Grep()
    agent = Agent(
        model=opus1m,
        system={
            "identity": "You are a scientist.",
            "methodology": Path("SCIENTIST.md").read_text(),
            "environment": lambda: f"cwd: {os.getcwd()}",
        },
        tools=[tools.Bash(peers=(read, grep)), read, grep],
        compactor=SummaryCompactor(),
        session_dir="~/.sessions/scientist",
    )

    response = await agent.run(json_freeze({"prompt": "design an experiment"}))

In a CLI the ``cli.resolve_tools`` helper wires peers automatically.
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

from sagent.agent.compaction import (
    CompactionState,
    post_compact_enrich,
)
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.dispatch import (
    add_tool_input_batch_hint,
    conditional_rules_for_request,
    invoke_tool,
    is_request_read_only,
    tc_directive,
    tc_tool_id,
    tool_call_label,
)
from sagent.agent.retry import (
    RateLimitError,
    RetriesExhaustedError,
    send_with_retry,
)
from sagent.agent.session_io import (
    SessionMeta,
    load_message,
    load_session,
    rebuild_tool_state_from_messages,
    repair_dangling_tool_calls,
    restore_model,
    save_session,
    text_from_msg,
)
from sagent.custom_exceptions import (
    ModelTerminationError,
)
from sagent.custom_types import (
    Compactor,
    ContextBudget,
    JsonMessage,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    MultipartDescriptor,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
    reset_id_counter,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.compaction import write_pre_compact_transcript
from sagent.lib.descriptors import (
    has_error,
    is_binary,
    is_multipart,
    is_thinking,
)
from sagent.lib.json import JSON, bool_val, int_val, json_freeze
from sagent.lib.message import (
    get_queue_id,
    get_tool_name,
    response_text,
    response_tool_calls,
    thinking_text,
)
from sagent.providers.lib.stop_reason import BENIGN_STOP_REASONS
from sagent.sessions import parse_jsonl
from sagent.tools.background_task import (
    BackgroundAwareTool,
    BackgroundTaskEntry,
)
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
from sagent.tools.result_storage import (
    ReplacementState,
    enforce_message_budget,
    inject_empty_marker,
    persist_result,
)


logger = logging.getLogger(__name__)


# System prompt: string or dict of named sections.
# Dict values: strings (static) or callables (evaluated each model request).
SystemPrompt = str | dict[str, str | Callable[[], str]]


_MAX_COMPACT_FAILURES = 3
_MAX_UNSAVED_EVENTS = 1000

# Sentinel placed in ``Agent.inbox`` by the REPL pump to signal session end.
QUIT_SENTINEL = "text/x-quit"

# Error codes returned in ToolResponse.content for programmatic matching.
ERROR_NO_PROMPT = "error:no_prompt"
ERROR_MAX_TOOL_CALL_ROUNDS = "error:max_tool_call_rounds"

# Max-tokens recovery: when the API reports the response was cut off
# mid-stream, preserve any partial assistant output and inject this
# meta user message so the model resumes mid-thought instead of
# starting over with an apology.
_MAX_TOKENS_RECOVERY_NUDGE = (
    "Output token limit hit. Resume directly - no apology, no recap of "
    "what you were doing. Pick up mid-thought if that is where the cut "
    "happened. Break remaining work into smaller pieces."
)
_MAX_TOKENS_RECOVERY_LIMIT = 3
_MAX_CONTEXT_OVERFLOW_RECOVERY_LIMIT = 3

# Stop reasons that mean "response was cut off mid-stream and the
# model should resume".
_RECOVERABLE_TRUNCATION = frozenset(
    {"max_tokens", "model_context_window_exceeded"},
)


class Agent:
    """Conversation agent with tool dispatch, logging, sessions."""

    # Tool attribute. Agent results are the final summary of a
    # subagent's work; dropping one on cache-cold would hide the whole
    # subtask's output. Always keep the last message.
    supports_microcompaction: bool = False

    def __init__(
        self,
        *,
        model: Model,
        model_spec: ModelSpec | None = None,
        system: SystemPrompt = "",
        tools: list[Tool] | None = None,
        compactor: Compactor | None = None,
        name: str = "Agent",
        description: str = "An AI agent.",
        max_tool_call_rounds: int | None = None,
        max_attempts: int = 5,
        thinking: str | None = "adaptive",
        effort: str | None = None,
        session_dir: str | Path | None = None,
        budget: ContextBudget | None = None,
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
            }
        )
        self._model = model
        self._model_spec = model_spec
        self._budget = budget
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
        self._system = system
        # Reject duplicate tool IDs up front - dict-from-list would
        # silently drop earlier entries, making the eventual "tool X
        # has unexpected behavior" bug hard to trace.
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            if t.tool_id in self._tools:
                raise ValueError(f"Duplicate tool: {t.tool_id!r}")
            self._tools[t.tool_id] = t
        self._compactor = compactor
        self._track_changed_files = track_changed_files
        self._max_tool_call_rounds = max_tool_call_rounds
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._max_attempts = max_attempts
        self._thinking = thinking
        self._cache_ttl: str = "5m"
        if effort is not None and not model.supports_effort:
            raise ValueError(
                f"Model {model.model_id!r} does not support effort"
                f" (got effort={effort!r}).",
            )
        self._effort = effort
        self._persistent_retry = persistent_retry
        self._cost_tracker = CostTracker(max_budget_usd=max_budget_usd)
        self._messages: list[Message] = []
        self._max_tokens_recoveries = 0
        self._session_dir: Path | None = None
        self._session_id = str(uuid.uuid4())[:8]
        self._status: str = ""
        self._event_log: list[dict[str, object]] = []
        self._compact_state = CompactionState()
        self._num_tool_call_rounds = 0
        self._tool_state = ToolState()
        self._replacement_state = ReplacementState(
            persist_threshold=self.budget.persist_threshold,
            message_budget=self.budget.message_budget_chars,
        )
        # Inbox: async deque shared between the REPL, tools, and
        # background tasks. User messages go to front (put_left);
        # everything else appends at back (put). The agent loop
        # drains it at ONE point -- the top of each iteration --
        # and injects items as user messages before the next LLM
        # call. No other code should drain the inbox.
        self.inbox: Deque[Message] = Deque()
        self._background_tasks: dict[str, BackgroundTaskEntry] = {}
        self._has_background_support = (
            "application/x-tool-backgroundtask" in self._tools
        )
        self._events: asyncio.Queue[Message | None] | None = None
        self._persistent: bool = False
        self._active: bool = False
        self._request_start_time: float = 0.0
        self._live_model_response_chars: int = 0
        self._last_elapsed: float = 0.0
        self._last_model_tokens = TokenCount()
        self._last_run_cost_usd: float = 0.0
        self._total_active_elapsed_seconds: float = 0.0
        # Snapshots taken at root-run start; at run end, the subtree
        # ledger is folded into ``_cost_tracker`` so it accumulates the
        # full subtree across runs rather than parent-only spend.
        self._tracker_snap_cost_usd: float = 0.0
        self._tracker_snap_tokens: TokenCount = TokenCount()
        self._active_cost_ledger: CostLedger | None = None
        self.active_children: dict[str, object] = {}
        self.inflight: asyncio.Task[Message] | None = None
        if session_dir is not None:
            self._session_dir = Path(session_dir)
            # Scope persisted tool-results to this session dir. Without
            # a session, the replacement state falls back to /tmp/sagent_results.
            self._replacement_state.storage_dir = self._session_dir
            self._load_session()

    @property
    def budget(self) -> ContextBudget:
        """Context budget (auto-derived from model if not explicit)."""
        if self._budget is not None:
            return self._budget
        return ContextBudget.from_model(self._model)

    @property
    def max_request_tokens(self) -> int:
        """Maximum request tokens."""
        return self.budget.max_request_tokens

    @max_request_tokens.setter
    def max_request_tokens(self, value: int) -> None:
        """Set the maximum request tokens."""
        if value > self._model.max_request_tokens:
            raise ValueError(
                f"max_request_tokens={value:,} exceeds model's"
                f" {self._model.max_request_tokens:,}",
            )
        self._budget = dataclasses.replace(self.budget, max_request_tokens=value)

    @property
    def max_response_tokens(self) -> int:
        """Maximum response tokens."""
        return self.budget.max_response_tokens

    @max_response_tokens.setter
    def max_response_tokens(self, value: int) -> None:
        """Set the maximum response tokens."""
        if value > self._model.max_response_tokens:
            raise ValueError(
                f"max_response_tokens={value:,} exceeds model's"
                f" {self._model.max_response_tokens:,}",
            )
        self._budget = dataclasses.replace(self.budget, max_response_tokens=value)

    @property
    def tool_state(self) -> ToolState:
        """The agent's ToolState (for REPL abort coordination)."""
        return self._tool_state

    @property
    def messages(self) -> list[Message]:
        """The conversation history (mutable)."""
        return self._messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        """Replace the conversation history."""
        self._messages = value

    @property
    def total_cost_usd(self) -> float:
        """Cumulative subtree USD cost for this session.

        Includes parent and all descendant agents across every completed
        root run. While a root run is active, returns the running total
        so the toolbar updates as descendant calls complete.
        """
        if self._active and self._active_cost_ledger is not None:
            return self._tracker_snap_cost_usd + self._active_cost_ledger.total_cost_usd
        return self._cost_tracker.total_cost_usd

    @property
    def cache_tokens(self) -> tuple[int, int]:
        """(cache_creation, cache_read) totals across this session."""
        tokens = self.total_tokens
        return (tokens.cache_creation_tokens, tokens.cache_read_tokens)

    @property
    def tools(self) -> list[Tool]:
        """The tool set available to this agent (order preserved)."""
        return list(self._tools.values())

    @property
    def system(self) -> SystemPrompt:
        """The system prompt this agent was constructed with."""
        return self._system

    @property
    def model(self) -> Model:
        """The model this agent uses."""
        return self._model

    def swap_model(self, model: Model, *, spec: ModelSpec) -> None:
        """Replace the active model and its spec.

        Args:
          model: New model instance.
          spec: Recipe that produced ``model``.

        """
        self._model = model
        self._model_spec = spec

    @property
    def model_spec(self) -> ModelSpec | None:
        """The recipe used to build ``self.model``, when known.

        ``tools.Agent`` reads this on the parent to let the LLM swap
        provider/auth/model/account per spawn. ``None`` means the
        parent's model was built outside the ``ModelSpec`` path (e.g.
        direct ``Model`` injection in tests) - the tool then falls
        back to inheriting ``self.model`` as-is.
        """
        return self._model_spec

    @property
    def background_tasks(self) -> dict[str, BackgroundTaskEntry]:
        """Active background tasks keyed by queue id."""
        return self._background_tasks

    @property
    def compactor(self) -> Compactor | None:
        """The compactor this agent uses, if any."""
        return self._compactor

    @property
    def max_attempts(self) -> int:
        """Retry cap for transient model failures."""
        return self._max_attempts

    @property
    def thinking(self) -> str | None:
        """Thinking mode for this agent (e.g. ``"adaptive"``). ``None`` disables."""
        return self._thinking

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
            raise ValueError(
                f"cache_ttl must be '5m' or '1h', got {value!r}",
            )
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
        """Set the session status and persist immediately.

        Args:
          status: New status string for the terminal titlebar.

        """
        self._status = status
        self._save_session()
        self._emit(TextMessage(status, "text/x-signal-status-changed"))

    @property
    def cost_ledger(self) -> CostLedger | None:
        """The active subtree-wide ledger, or None outside a call.

        Returns the ``CostLedger`` currently installed via
        ``cost_ledger_var`` - either the one this agent created as
        root, or one inherited from an ancestor. Outside ``run``
        the var is unset and this returns None.
        """
        return cost_ledger_var.get()

    @property
    def active(self) -> bool:
        """True while ``run()`` is executing."""
        return self._active

    @property
    def request_start_time(self) -> float:
        """Event-loop timestamp when the current (or last) run started."""
        return self._request_start_time

    @property
    def live_model_response_tokens(self) -> int:
        """Live output-token estimate (chars // 4) for the current run."""
        return self._live_model_response_chars // self.budget.chars_per_token

    @property
    def last_elapsed(self) -> float:
        """Wall-clock seconds the most recent completed run took."""
        return self._last_elapsed

    @property
    def last_model_request_tokens(self) -> int:
        """Request tokens from the most recently completed model call."""
        return self._last_model_tokens.input_tokens

    @property
    def last_model_response_tokens(self) -> int:
        """Response tokens from the most recently completed run."""
        return self._last_model_tokens.output_tokens

    @property
    def last_run_tokens(self) -> TokenCount:
        """Token counts for the active root run, including descendant agents."""
        if self._active_cost_ledger is not None:
            return self._active_cost_ledger.tokens
        return self._last_model_tokens

    @property
    def last_run_cost_usd(self) -> float:
        """Cost for the active root run, including descendant agents."""
        if self._active_cost_ledger is not None:
            return self._active_cost_ledger.total_cost_usd
        return self._last_run_cost_usd

    @property
    def total_active_elapsed_seconds(self) -> float:
        """Cumulative wall-clock seconds spent in ``run`` across the session.

        Live during an active run: sums the stored historical total with
        ``(now - request_start_time)`` so the toolbar ticks in real time.
        """
        if self._active:
            loop = asyncio.get_running_loop()
            return self._total_active_elapsed_seconds + (
                loop.time() - self._request_start_time
            )
        return self._total_active_elapsed_seconds

    @property
    def total_tokens(self) -> TokenCount:
        """Cumulative subtree token counts across the session.

        Mirrors ``total_cost_usd`` — includes descendants and updates
        live during an active root run.
        """
        if self._active and self._active_cost_ledger is not None:
            return self._tracker_snap_tokens + self._active_cost_ledger.tokens
        return self._cost_tracker.total

    async def run_forever(
        self,
    ) -> AsyncGenerator[Message | None, None]:
        """Run continuously: read prompts from inbox, process prompts, repeat.

        Loops until ``QUIT_SENTINEL`` is received from inbox. Each prompt
        triggers one ``run()`` call. Cancellation of the inflight task (via
        ``inflight.cancel()``) is handled gracefully -- the loop
        continues and waits for the next prompt.

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
                    self._clear_context(str(m.content))
                else:
                    prompts.append(str(m.content))
            if not prompts:
                continue
            prompt = "\n\n".join(prompts)
            handle = self.run(json_freeze({"prompt": prompt}))
            try:
                async for event in handle:
                    yield event
                await handle
            except asyncio.CancelledError:
                pass
            except (RateLimitError, RetriesExhaustedError) as e:
                logger.warning("run_forever: %s", e)
                yield TextMessage(str(e), "text/x-error")
            except Exception as e:  # noqa: BLE001 -- REPL safety net for unexpected errors
                logger.debug("run_forever: run raised", exc_info=True)
                yield TextMessage(f"⚠ {type(e).__name__}: {e}", "text/x-error")
            yield None

    # -- Tool interface ----------------------------------------

    def summary(self, msg: Message) -> str:
        """Return a help description for this agent.

        Args:
          msg: The requesting message.

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

    def run(self, directive: JSON) -> RunHandle:
        """Run the agent on a prompt.

        Returns a handle that is both awaitable (for the final result)
        and async-iterable (for streaming events)::

            # Batch:
            result = await agent.run(directive)

            # Streaming:
            async for event in agent.run(directive):
                render(event)

            # Both:
            handle = agent.run(directive)
            async for event in handle:
                render(event)
            result = await handle

        Args:
          directive: JSON mapping with at least a ``"prompt"`` key.

        Returns:
          handle: Awaitable and async-iterable run handle.

        """
        self._cost_tracker.call_output_tokens = 0
        self._active = True
        self._request_start_time = asyncio.get_running_loop().time()
        self._live_model_response_chars = 0
        parent_state = tool_state_var.get(None)
        self.tool_state.depth = 0 if parent_state is None else parent_state.depth + 1
        self._events = asyncio.Queue()
        self.tool_state.abort_event.clear()
        task = asyncio.create_task(self._run_impl(directive))
        self.inflight = task
        return RunHandle(self, task)

    async def _run_impl(self, directive: JSON) -> Message:
        """Execute the send loop, emitting x-done and None sentinel on exit."""
        agent_token = current_agent_var.set(self)
        # The root run owns the UI snapshot; children inherit this ContextVar ledger.
        ledger = cost_ledger_var.get()
        if ledger is None:
            ledger = CostLedger()
            self._active_cost_ledger = ledger
            ledger_token = cost_ledger_var.set(ledger)
            self._tracker_snap_cost_usd = self._cost_tracker.total_cost_usd
            self._tracker_snap_tokens = self._cost_tracker.total
        else:
            ledger_token = None
        counter_token = agent_counter_var.set(itertools.count())
        label = agent_label_var.get("") or self.name
        label_token = agent_label_var.set(label)
        agent_registry[label] = self
        try:
            with tool_state_context(self.tool_state):
                return await self._send_impl(directive)
        except asyncio.CancelledError:
            self._emit(TextMessage("[interrupted]", "text/x-interrupted"))
            raise
        finally:
            self._save_session()
            assert self._events is not None
            self._events.put_nowait(
                JsonMessage(
                    json_freeze(
                        {
                            "input_tokens": self._cost_tracker.last_request.input_tokens,
                            "output_tokens": self._cost_tracker.call_output_tokens,
                        }
                    ),
                    "application/x-done",
                )
            )
            self._events.put_nowait(None)
            if not self._persistent:
                agent_registry.pop(label, None)
            agent_label_var.reset(label_token)
            current_agent_var.reset(agent_token)
            agent_counter_var.reset(counter_token)
            if ledger_token is not None:
                cost_ledger_var.reset(ledger_token)

    async def _send_impl(
        self,
        directive: JSON,
    ) -> Message:
        """Core send loop.  Pursues inbox zero: drains the inbox at
        the top of each iteration, calls the LLM, dispatches tools,
        and loops until the inbox is empty and the LLM is done.
        """
        prompt = str(directive.get("prompt", ""))
        if not prompt:
            raise ValueError(f"No prompt provided. [{ERROR_NO_PROMPT}]")

        self._max_tokens_recoveries = 0
        self._messages = repair_dangling_tool_calls(self.messages)
        await self._maybe_drain_clear_request()
        self.inbox.put(TextMessage(prompt, "text/x-user-message"))

        request_num = 0
        while True:
            if (
                self._max_tool_call_rounds is not None
                and request_num >= self._max_tool_call_rounds
            ):
                return TextMessage(
                    f"Tool-call-round limit reached"
                    f" ({self._max_tool_call_rounds} rounds)."
                    f" [{ERROR_MAX_TOOL_CALL_ROUNDS}]",
                    "text/x-error",
                )
            self._drain_inbox(request_num)
            if not self.messages:
                return TextMessage("", "text/plain")
            system = await self._prepare_request()
            response = await self._model_call(system)
            tool_calls = self._log_response(response, request_num)
            request_num += 1

            # -- Stop-reason guards --
            if response.stop_reason == "model_refusal":
                raise RuntimeError(
                    "Model refused to respond (content filter or usage policy)."
                )
            if (
                response.stop_reason not in BENIGN_STOP_REASONS
                and response.stop_reason not in _RECOVERABLE_TRUNCATION
            ):
                raise ModelTerminationError(response)

            # -- Truncation recovery --
            if response.stop_reason in _RECOVERABLE_TRUNCATION and not tool_calls:
                self._max_tokens_recoveries += 1
                if self._max_tokens_recoveries > _MAX_TOKENS_RECOVERY_LIMIT:
                    raise RuntimeError(
                        response_text(response.content)
                        + "\n\n[truncated: recovery attempts exhausted]"
                    )
                self.messages.append(
                    TextMessage(
                        _MAX_TOKENS_RECOVERY_NUDGE,
                        "text/x-user-message",
                    ),
                )
                continue
            if response.stop_reason not in _RECOVERABLE_TRUNCATION:
                self._max_tokens_recoveries = 0

            # -- Tool dispatch --
            if tool_calls:
                self._num_tool_call_rounds += 1
                self._publish_stats()
                await self._run_tools(tool_calls)
                if not self.messages:
                    return TextMessage("", "text/plain")
                continue

            # -- Done unless inbox has pending work --
            if not self.inbox:
                return TextMessage(
                    response_text(response.content),
                    "text/plain",
                )

    def _emit(self, event: Message) -> None:
        """Put an event on the streaming queue (no-op outside ``run``)."""
        if self._events is not None:
            self._events.put_nowait(event)

    def _finish_run(self) -> None:
        """Post-run cleanup called by ``RunHandle`` after the task completes."""
        self.inflight = None
        self._active = False
        loop = asyncio.get_running_loop()
        self._last_elapsed = loop.time() - self._request_start_time
        self._total_active_elapsed_seconds += self._last_elapsed
        # Children record their local call; only the root snapshots the shared subtree ledger.
        if self._active_cost_ledger is not None:
            self._last_model_tokens = self._active_cost_ledger.tokens
            self._last_run_cost_usd = self._active_cost_ledger.total_cost_usd
            self._cost_tracker.fold(
                snapshot_cost_usd=self._tracker_snap_cost_usd,
                snapshot_tokens=self._tracker_snap_tokens,
                run_ledger=self._active_cost_ledger,
            )
            self._active_cost_ledger = None
        else:
            self._last_model_tokens = TokenCount(
                input_tokens=self._cost_tracker.last_request.input_tokens,
                output_tokens=self._cost_tracker.call_output_tokens,
            )
            self._last_run_cost_usd = self._cost_tracker.total_cost_usd
        self._events = None

    # -- Request helpers ----------------------------------------------

    async def _prepare_request(self) -> str:
        """Pre-model-call: maintain, compact, build system prompt."""
        if self.compactor is not None:
            self.compactor.maintain(
                self.messages,
                self._tools,
                read_cache=self.tool_state.read_cache,
                last_response_time=self._cost_tracker.last_response_time,
            )
        system = _build_system_prompt(
            self.system,
            self._tools,
            track_changed_files=self._track_changed_files,
        )
        input_tokens = max(
            self._cost_tracker.last_request.input_tokens,
            _estimate_total_tokens(system, self.messages, self.model),
        )
        compacted = await self._maybe_compact(input_tokens)
        if not compacted and input_tokens > (
            self.max_request_tokens
            - self.max_response_tokens
            - self.budget.buffer_tokens
        ):
            await self._force_compact()
        return system

    async def _model_call(self, system: str) -> ModelResponse:
        """Build request, send with retry, account tokens."""
        request = _build_model_request(
            messages=self.messages,
            system=system,
            tools=list(self._tools.values()),
            has_background_support=self._has_background_support,
            max_response_tokens=self.max_response_tokens,
            thinking=self.thinking,
            effort=self.effort,
            cache_ttl=self._cache_ttl,
        )

        def _emit_text(chunk: str) -> None:
            self._live_model_response_chars += len(chunk)
            self._emit(TextMessage(chunk, "text/plain"))

        on_text = _emit_text if self._events is not None else None
        chars_before = self._live_model_response_chars
        try:
            response = await self._send_with_context_recovery(request, on_text)
        except asyncio.CancelledError:
            chars_this_call = self._live_model_response_chars - chars_before
            partial = _estimate_cancelled_response(
                system,
                request.messages,
                self.model,
                chars_this_call,
            )
            with contextlib.suppress(RuntimeError):
                self._account_response(partial)
            raise
        self._account_response(response)
        self.messages.append(response.content)
        return response

    async def _send_with_context_recovery(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None,
    ) -> ModelResponse:
        last_error: Exception | None = None
        for recovery_attempt in range(_MAX_CONTEXT_OVERFLOW_RECOVERY_LIMIT + 1):
            try:
                return await send_with_retry(
                    self.model,
                    request,
                    on_text=on_text,
                    max_attempts=self.max_attempts,
                    persistent_retry=self._persistent_retry,
                    log_event=self._log_event,
                    on_discarded_response=self._account_response,
                )
            except Exception as e:
                if not self.model.is_context_overflow(e):
                    raise
                last_error = e
                if recovery_attempt >= _MAX_CONTEXT_OVERFLOW_RECOVERY_LIMIT:
                    break
                self._log_event(
                    "context_overflow_compact",
                    recovery_attempt=recovery_attempt,
                )
                await self._force_compact()
                request = dataclasses.replace(
                    request,
                    messages=list(self.messages),
                    system=_build_system_prompt(
                        self.system,
                        self._tools,
                        track_changed_files=self._track_changed_files,
                    ),
                )
        msg = (
            "context overflow recovery failed after "
            f"{_MAX_CONTEXT_OVERFLOW_RECOVERY_LIMIT} compactions"
        )
        raise RuntimeError(msg) from last_error

    def _log_response(
        self,
        response: ModelResponse,
        request_num: int,
    ) -> list[Message]:
        """Emit thinking event, log response, return tool calls."""
        thinking = _response_thinking(response.content)
        if thinking:
            self._emit(TextMessage(thinking, "text/x-thinking"))
        tool_calls = response_tool_calls(response.content)
        self._log_event(
            "model_response",
            request=request_num,
            stop_reason=response.stop_reason,
            text_len=len(response_text(response.content)),
            tool_count=len(tool_calls),
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            thinking=bool(thinking),
            streaming=self._events is not None,
        )
        return tool_calls

    async def _run_tools(self, tool_calls: list[Message]) -> None:
        """Dispatch tools, enforce budget, append results.

        Tool calls with ``background: true`` (or ``delay: N``) in
        their directive are dispatched as fire-and-forget async tasks.
        A placeholder result is appended immediately; the real result
        lands in the inbox when the task completes.
        """
        fg_calls: list[Message] = []
        for req in tool_calls:
            if self._events is not None:
                tid = tc_tool_id(req)
                tool = self._tools.get(tid)
                desc = tool_call_label(tool, req)
                self._live_model_response_chars += len(desc)
                self._emit(TextMessage(desc, "text/x-tool-label"))
            directive = tc_directive(req)
            delay = int_val(directive.get("delay"), 0)
            bg = bool_val(directive.get("background"), False) or delay > 0
            if bg:
                self._dispatch_background(req, delay)
            else:
                fg_calls.append(req)
        if fg_calls:
            await self._dispatch_foreground(fg_calls)
        await self._maybe_drain_compact_request()
        await self._maybe_drain_recompact_request()
        await self._maybe_drain_clear_request()

    async def _dispatch_foreground(self, tool_calls: list[Message]) -> None:
        """Synchronous tool dispatch (original path)."""
        tool_results = await self._dispatch_tools(tool_calls)
        tool_names = {get_queue_id(tc): tc_tool_id(tc) for tc in tool_calls}
        tool_results = enforce_message_budget(
            tool_results,
            tool_names,
            self._replacement_state,
        )
        tool_results = add_tool_input_batch_hint(tool_results)
        for raw_tr in tool_results:
            resolved = _wrap_errors_for_llm(raw_tr)
            self.messages.append(resolved)
            self._emit(
                MultipartMessage(
                    cast(tuple[Message, ...], raw_tr.content),
                    cast("MultipartDescriptor", raw_tr.descriptor),
                )
            )

    def _dispatch_background(self, req: Message, delay_sec: float) -> None:
        """Fire-and-forget a tool call as a background task."""
        qid = get_queue_id(req)
        tool_name = get_tool_name(req)
        task = asyncio.create_task(self._bg_worker(req, delay_sec))
        self.background_tasks[qid] = BackgroundTaskEntry(
            task=task,
            tool_name=tool_name,
            queue_id=qid,
            started=time.time(),
            delay_sec=delay_sec,
        )
        placeholder = MultipartMessage(
            (
                TextMessage(qid, "text/x-queue-id"),
                TextMessage(f"[Running in background: {tool_name}]", "text/plain"),
            ),
            "multipart/x-tool-result",
            parent_id=req.id,
        )
        self.messages.append(_wrap_errors_for_llm(placeholder))
        self._emit(TextMessage(f"[Background: {tool_name}]", "text/plain"))

    async def _bg_worker(self, req: Message, delay_sec: float) -> Message:
        """Background task: optional sleep, then dispatch, then inbox."""
        qid = get_queue_id(req)
        tool_name = get_tool_name(req)
        try:
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)
            result = await self._invoke_tool_safe(req)
            result_text = text_from_msg(result)
            self.inbox.put(
                TextMessage(
                    f"[Background tool completed: {tool_name} ({qid})]\n\n{result_text}",
                    "text/x-user-message",
                )
            )
            return result
        except asyncio.CancelledError:
            self.inbox.put(
                TextMessage(
                    f"[Background tool cancelled: {tool_name} ({qid})]",
                    "text/x-user-message",
                )
            )
            raise
        except Exception as e:
            self.inbox.put(
                TextMessage(
                    f"[Background tool failed: {tool_name} ({qid})]\n\n"
                    f"{type(e).__name__}: {e}",
                    "text/x-user-message",
                )
            )
            raise
        finally:
            self.background_tasks.pop(qid, None)

    def _drain_inbox(self, request_num: int) -> None:
        """Drain inbox into conversation history.

        This is the only place that converts inbox text into conversation
        history or context-changing slash-command effects. Producers
        (REPL, keybindings, background tasks, agent sends) may prioritize
        delivery with ``put_left`` but must not mutate history directly.

        On request 0 the drain is the initial prompt (user bar already
        rendered by the REPL), so skip the ``text/x-user-injected`` event.
        On later iterations the event triggers a mid-request user bar.
        """
        drained = self.inbox.drain()
        if not drained:
            return
        kept: list[str] = []
        for item in drained:
            if item.descriptor == "text/x-clear-request":
                self._clear_context(str(item.content))
            else:
                kept.append(str(item.content))
        if not kept:
            return
        text = "\n\n".join(kept)
        self.messages.append(TextMessage(text, "text/x-user-message"))
        self._log_event("user_message", request=request_num, content_len=len(text))
        if request_num > 0:
            self._emit(TextMessage(text, "text/x-user-injected"))

    def _account_response(self, response: ModelResponse) -> None:
        """Update token/cost counters after a model response."""
        self._cost_tracker.record(
            response,
            model_id=self.model.model_id,
            ledger=cost_ledger_var.get(),
        )
        # Reset the in-flight chars/4 estimator: those tokens are now
        # in ``_cost_tracker.total``. Leaving the counter live would
        # double-count completed calls in the toolbar between rounds.
        self._live_model_response_chars = 0
        self._publish_stats()

    # -- Tool dispatch -------------------------------------------------

    async def _invoke_tool_safe(self, req: Message) -> Message:
        """Invoke a tool, converting unexpected exceptions to error results."""
        try:
            return await invoke_tool(self._tools, req, self._events)
        except Exception as e:  # noqa: BLE001 -- dispatch safety net
            logger.debug("Tool %s raised", get_tool_name(req), exc_info=True)
            error_text = f"{type(e).__name__}: {e}"
            self._emit(TextMessage(error_text, "text/x-error"))
            return MultipartMessage(
                (
                    TextMessage(get_queue_id(req), "text/x-queue-id"),
                    TextMessage(error_text, "text/x-error"),
                ),
                "multipart/x-tool-result",
                parent_id=req.id,
            )

    async def _dispatch_tools(self, requests: list[Message]) -> list[Message]:
        """Execute tool calls; batch read-only, serialize the rest."""
        self.tool_state.bash_parse_cache.clear()
        for req in requests:
            self._log_event(
                "tool_call",
                tool=tc_tool_id(req),
                tool_id=get_queue_id(req),
                input=tc_directive(req),
            )
        safe_flags = [is_request_read_only(r, self._tool_state) for r in requests]
        batches = _partition_batches(safe_flags)
        out: dict[int, Message] = {}
        for batch in batches:
            if len(batch) == 1:
                out[batch[0]] = await self._invoke_tool_safe(requests[batch[0]])
            else:
                vals = await asyncio.gather(
                    *(self._invoke_tool_safe(requests[i]) for i in batch)
                )
                out.update(dict(zip(batch, vals, strict=True)))
        results = [out[i] for i in range(len(requests))]
        results, log_entries = _postprocess_results(
            requests, results, self._replacement_state, self.tool_state
        )
        for entry in log_entries:
            self._log_event("tool_result", **entry)
        return results

    # -- Compaction ----------------------------------------------------

    async def _maybe_compact(self, input_tokens: int) -> bool:
        """Run compactor if configured and threshold reached.

        Returns True if compaction ran this request.
        """
        if self.compactor is None:
            return False
        if self._compact_state.compacting:
            return False
        if self._compact_state.compact_failures >= _MAX_COMPACT_FAILURES:
            return False
        should = await self.compactor.should_compact(
            input_tokens=input_tokens,
            max_request_tokens=self.max_request_tokens,
            max_response_tokens=self.max_response_tokens,
        )
        if not should:
            return False
        await self._do_compact()
        return True

    async def _force_compact(self) -> None:
        """Force compaction regardless of threshold.

        Raises ``RuntimeError`` when compaction has already failed
        ``_MAX_COMPACT_FAILURES`` times - the conversation is now
        guaranteed to blow the context window, so silently returning
        would only defer the failure to the next API call (where it
        surfaces as an opaque 400). Surface it immediately so the
        retry layer / operator sees the root cause.
        """
        if self.compactor is None:
            return
        if self._compact_state.compacting:
            return
        if self._compact_state.compact_failures >= _MAX_COMPACT_FAILURES:
            msg = (
                f"Compaction disabled after {self._compact_state.compact_failures} "
                f"failures; cannot reduce conversation below context"
                f" window. Clear the session or raise the cap."
            )
            logger.warning(msg)
            raise RuntimeError(msg)
        await self._do_compact()

    def _publish_stats(self) -> None:
        """Refresh ``tool_state.stats`` so the Diagnostics tool sees current data."""
        c = self._cost_tracker
        self.tool_state.stats = {
            "num_tool_call_rounds": self._num_tool_call_rounds,
            "total_input_tokens": c.total.input_tokens,
            "total_output_tokens": c.total.output_tokens,
            "input_tokens": c.last_request.input_tokens,
            "cache_creation_tokens": c.total.cache_creation_tokens,
            "cache_read_tokens": c.total.cache_read_tokens,
            "total_cost_usd": c.total_cost_usd,
            "max_request_tokens": self.max_request_tokens,
            "max_response_tokens": self.max_response_tokens,
        }

    async def _maybe_drain_compact_request(self) -> None:
        """Honor a pending Compact-tool request, if any.

        The Compact tool signals via ``tool_state.compact_requested``
        rather than running compaction synchronously - the actual
        rewrite has to happen between model requests so the tool's own result
        message is part of the history getting summarized. Clears
        the signal whether or not compaction succeeds, so a single
        failed attempt doesn't loop on every subsequent model request.
        """
        pending = self.tool_state.compact_requested
        if pending is None:
            return
        self.tool_state.compact_requested = None
        if self.compactor is None:
            self._log_event("compact_request_ignored", reason="no_compactor")
            return
        self._log_event(
            "compact_request_drain",
            has_instructions=bool(pending.strip()),
        )
        await self._do_compact(
            custom_instructions=pending or None,
            count_toward_budget=False,
        )

    async def _maybe_drain_recompact_request(self) -> None:
        """Honor a pending Recompact-tool request, if any.

        Reload ``pre_compact.jsonl``, re-run the compactor with the
        caller's ``custom_instructions``, and install the new summary.
        Preserves the original transcript on disk so subsequent
        Recompacts still work. Rolls the current history back to the
        pre-Recompact state if the compactor fails - we must not
        leave the user sitting in a restored-but-not-compacted state
        (which would typically overflow the context window).
        """
        pending = self.tool_state.recompact_requested
        if pending is None:
            return
        self.tool_state.recompact_requested = None
        if self.compactor is None:
            self._log_event("recompact_request_ignored", reason="no_compactor")
            return
        if self._session_dir is None:
            self._log_event("recompact_request_ignored", reason="no_session_dir")
            return
        transcript = self._session_dir / "pre_compact.jsonl"
        if not transcript.exists():
            self._log_event("recompact_request_ignored", reason="no_pre_compact")
            return
        try:
            raw_text = transcript.read_text(encoding="utf-8")
        except OSError as e:
            self._log_event("recompact_load_failed", error=str(e))
            return
        try:
            loaded = [load_message(rec) for rec in parse_jsonl(raw_text)]
        except (KeyError, AssertionError, TypeError) as e:
            self._log_event("recompact_deserialize_failed", error=str(e))
            return
        if not loaded:
            self._log_event("recompact_request_ignored", reason="empty_pre_compact")
            return

        self._log_event(
            "recompact_request_drain",
            has_instructions=bool(pending.strip()),
            restored_messages=len(loaded),
        )
        saved = list(self.messages)
        self._messages = loaded
        success = await self._do_compact(
            custom_instructions=pending or None,
            write_transcript=False,
            count_toward_budget=False,
        )
        if not success:
            # _do_compact already logged; restore so the agent isn't
            # left in a pre-compact (likely over-budget) state. Note
            # that ``_do_compact`` unconditionally saves the session
            # - on failure, it just persisted ``loaded`` (pre-compact
            # transcript). Save again after the rollback so disk
            # matches the restored in-memory state; otherwise a crash
            # before the next model request leaves a corrupted session file.
            self._messages = saved
            self._save_session()
            self._log_event("recompact_rolled_back")

    async def _maybe_drain_clear_request(self) -> None:
        """Honor a pending Clear-tool request, if any.

        Wipes messages and the in-memory file-tracking caches on
        ``tool_state`` so the next model request starts with no recollection
        of previously-read/edited files. Token counters reset; cost
        totals do NOT reset (cost is a cumulative session metric,
        not a context-window metric).
        """
        pending = self.tool_state.clear_requested
        if pending is None:
            return
        self.tool_state.clear_requested = None
        self._clear_context(pending)

    def _clear_context(self, reason: str) -> None:
        """Wipe conversation context and file-tracking state."""
        self._log_event("clear_request_drain", reason=reason)
        self._messages = []
        self._cost_tracker.last_request = TokenCount()
        self._num_tool_call_rounds = 0
        # Reset file-tracking caches so Edit/Write's "must be read
        # first" invariant starts fresh on the cleared session.
        self.tool_state.reset_file_tracking()
        self._save_session()

    async def _do_compact(
        self,
        *,
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        write_transcript: bool = True,
        count_toward_budget: bool = True,
    ) -> bool:
        """Run compaction, re-attach files, reset token counts.

        ``keep_recent`` and ``custom_instructions`` override the
        compactor's defaults for this call - used by the Compact-tool
        drain path so the user's guidance wins over whatever the
        configured compactor was built with.

        ``write_transcript`` is True for first-time compactions (they
        produce the authoritative pre-compact snapshot). Recompact
        passes False so the original ``pre_compact_N.jsonl`` is
        preserved across retries.

        ``count_toward_budget`` controls whether a failure increments
        ``_compact_failures`` (which can ultimately disable autonomous
        compaction). False for manually-triggered paths (the Compact
        and Recompact tool drains) - the user deliberately asking for
        a compaction shouldn't burn the runaway-compactor safety
        budget; that budget exists to detect an actually-broken
        compactor during autonomous use.

        Returns True on success, False on failure (compactor raised
        and messages were left unchanged).
        """
        assert self.compactor is not None
        self._compact_state.compacting = True
        self._log_event(
            "compact",
            messages_before=len(self.messages),
            tokens_before=self._cost_tracker.last_request.input_tokens,
            keep_recent=keep_recent,
            has_instructions=bool(custom_instructions),
            write_transcript=write_transcript,
        )
        try:
            # Persist before _save_session overwrites session.jsonl.
            # Numbered transcripts so nothing is ever lost.
            transcript_path: Path | None = None
            if self._session_dir is not None:
                transcript_path = (
                    self._session_dir
                    / f"pre_compact_{self._compact_state.compact_count}.jsonl"
                )
                if write_transcript:
                    write_pre_compact_transcript(
                        transcript_path,
                        self.messages,
                    )
            # Extract prior summary pointers before compaction
            # overwrites messages. These survive across compactions
            # as in-context "note to self" references.
            prior_pointers = list(self._compact_state.summary_pointers)
            result = await self.compactor.compact(
                messages=self.messages,
                model=self.model,
                transcript_path=transcript_path,
                keep_recent=keep_recent,
                custom_instructions=custom_instructions,
                summary_pointers=prior_pointers or None,
            )
            self._messages = result
        except Exception as e:  # noqa: BLE001 -- compact/model can raise varied errors
            error_type = type(e).__name__
            error_msg = str(e)
            if count_toward_budget:
                self._compact_state.compact_failures += 1
                logger.warning(
                    "Compaction failed (%d/%d): %s: %s",
                    self._compact_state.compact_failures,
                    _MAX_COMPACT_FAILURES,
                    error_type,
                    error_msg,
                )
            else:
                logger.warning(
                    "Compaction failed (manual; budget unchanged): %s: %s",
                    error_type,
                    error_msg,
                )
            logger.debug("Compaction traceback", exc_info=True)
            self._log_event(
                "compact_done",
                messages_after=len(self.messages),
                success=False,
                error_type=error_type,
                error=error_msg,
            )
            self._save_session()
            return False
        finally:
            self._compact_state.compacting = False

        system = _build_system_prompt(
            self.system,
            self._tools,
            track_changed_files=self._track_changed_files,
        )
        used = _estimate_total_tokens(system, self.messages, self.model)
        headroom = self.max_response_tokens + self.budget.buffer_tokens
        await post_compact_enrich(
            result=result,
            messages=self.messages,
            state=self._compact_state,
            session_dir=self._session_dir,
            tool_state=self.tool_state,
            budget=self.budget,
            tools=self._tools,
            background_tasks=self.background_tasks,
            estimate_tokens=self.max_request_tokens - used,
            headroom=headroom,
        )
        self._compact_state.compact_count += 1
        self._cost_tracker.last_request = TokenCount()
        self._compact_state.compact_failures = 0
        self._log_event(
            "compact_done",
            messages_after=len(self.messages),
            success=True,
        )
        self._save_session()
        return True

    # -- Structured event logging --------------------------------------

    def _log_event(self, event: str, **data: object) -> None:
        """Record a structured event.

        Between saves the buffer can grow without bound - a session
        that loops in tools without reaching model_finished never triggers
        a save. Cap the buffer in both the session and no-session
        paths: when a ``session_dir`` is configured, try to flush to
        disk via ``_save_session`` (persisting every event); fall back
        to in-memory truncation if the save raises (disk full, read-only
        FS) so logging never takes down the send loop.
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
            if self._session_dir is not None:
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

    # -- Session persistence -------------------------------------------

    def _save_session(self) -> None:
        """Save conversation to session.jsonl."""
        if self._session_dir is None:
            return
        spec = self.model_spec
        meta = SessionMeta(
            session_id=self.session_id,
            model_id=self.model.model_id,
            provider=spec.provider if spec else "",
            auth=spec.auth if spec else "",
            account=spec.account or "" if spec else "",
            name=self.name,
            status=self.status,
            tokens=TokenCount(
                input_tokens=self._cost_tracker.total.input_tokens,
                output_tokens=self._cost_tracker.total.output_tokens,
                cache_creation_tokens=self._cost_tracker.total.cache_creation_tokens,
                cache_read_tokens=self._cost_tracker.total.cache_read_tokens,
            ),
            total_cost_usd=self._cost_tracker.total_cost_usd,
            num_tool_call_rounds=self._num_tool_call_rounds,
            compact_count=self._compact_state.compact_count,
            summary_pointers=self._compact_state.summary_pointers,
            bash_cwd=self.tool_state.bash_cwd,
            total_active_elapsed_seconds=self._total_active_elapsed_seconds,
        )
        save_session(
            self._session_dir / "session.jsonl",
            meta=meta.serialize(),
            messages=self.messages,
            event_log=self._event_log,
        )
        self._event_log.clear()

    def _load_session(self) -> None:
        """Load conversation from disk, auto-migrating legacy layouts."""
        if self._session_dir is None:
            return
        result = load_session(
            self._session_dir,
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
        self._messages = repair_dangling_tool_calls(messages)
        rebuild_tool_state_from_messages(
            self.messages,
            self.tool_state,
        )
        if meta:
            m = SessionMeta.deserialize(meta)
            self._session_id = m.session_id or self.session_id
            self._status = m.status
            self._cost_tracker.restore(
                total_cost_usd=m.total_cost_usd,
                total=TokenCount(
                    input_tokens=m.tokens.input_tokens,
                    output_tokens=m.tokens.output_tokens,
                    cache_creation_tokens=m.tokens.cache_creation_tokens,
                    cache_read_tokens=m.tokens.cache_read_tokens,
                ),
            )
            self._num_tool_call_rounds = m.num_tool_call_rounds
            self._compact_state.compact_count = m.compact_count
            self._compact_state.summary_pointers = list(m.summary_pointers)
            if m.bash_cwd:
                self.tool_state.bash_cwd = m.bash_cwd
            self._total_active_elapsed_seconds = m.total_active_elapsed_seconds
            if m.provider and m.model_id:
                restored = restore_model(m)
                if restored is not None:
                    self._model, self._model_spec = restored
        self._cost_tracker.last_response_time = time.time()
        logger.info(
            "Resumed session %s (%d messages)",
            self.session_id,
            len(self.messages),
        )


class RunHandle:
    """Dual-interface handle: awaitable for the result, async-iterable for events.

    Returned by ``Agent.run()``. Streaming callers iterate events;
    batch callers await the result. Both may be used on the same handle
    (iterate first, then await).
    """

    __slots__ = ("_agent", "_drained", "_finished", "_task")

    def __init__(self, agent: Agent, task: asyncio.Task[Message]) -> None:
        self._agent = agent
        self._task = task
        self._drained = False
        self._finished = False

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncGenerator[Message, None]:
        """Yield streaming events until the run completes."""
        events = self._agent._events  # noqa: SLF001 -- RunHandle is Agent's intimate helper
        assert events is not None
        while True:
            event = await events.get()
            if event is None:
                self._drained = True
                break
            yield event
        with contextlib.suppress(asyncio.CancelledError):
            await self._finish()

    def __await__(self) -> Generator[object, None, Message]:
        return self._await_result().__await__()

    async def _await_result(self) -> Message:
        """Drain any unconsumed events, then return the task result."""
        if not self._drained:
            events = self._agent._events  # noqa: SLF001 -- RunHandle is Agent's intimate helper
            if events is not None:
                while True:
                    event = await events.get()
                    if event is None:
                        self._drained = True
                        break
        return await self._finish()

    async def _finish(self) -> Message:
        """Return the task result and finalize agent state exactly once."""
        try:
            return await self._task
        finally:
            if not self._finished:
                self._finished = True
                self._agent._finish_run()  # noqa: SLF001 -- RunHandle is Agent's intimate helper


def _build_system_prompt(
    system: SystemPrompt,
    tools: dict[str, Tool],
    *,
    track_changed_files: bool,
) -> str:
    """Assemble system prompt from sections + per-request contributions."""
    parts: list[str]
    if isinstance(system, str):
        parts = [system]
    else:
        parts = []
        for value in system.values():
            content = value if isinstance(value, str) else value()
            if content:
                parts.append(content)
    if track_changed_files:
        diff = changed_files_context()
        if diff:
            parts.append(diff)
    for tool in tools.values():
        section = tool.prompt()
        if section:
            parts.append(section)
    return "\n\n".join(parts)


def _estimate_total_tokens(
    system: str,
    messages: list[Message],
    model: Model,
) -> int:
    """Estimate total tokens across messages plus the system prompt."""
    total = model.estimate_text_token_count(system)
    for msg in messages:
        total += _estimate_message_tokens(msg, model)
    return total


def _estimate_message_tokens(msg: Message, model: Model) -> int:
    if is_multipart(msg.descriptor):
        return sum(
            _estimate_message_tokens(p, model)
            for p in cast(tuple[Message, ...], msg.content)
        )
    if is_binary(msg.descriptor):
        return model.estimate_image_token_count(cast(bytes, msg.content))
    return model.estimate_text_token_count(str(msg.content))


def _response_thinking(msg: Message) -> str:
    """Extract thinking text from a model response."""
    if not is_multipart(msg.descriptor):
        return ""
    parts = cast(tuple[Message, ...], msg.content)
    return "\n".join(
        t for p in parts if is_thinking(p.descriptor) for t in (thinking_text(p),) if t
    )


def _wrap_errors_for_llm(tr: Message) -> Message:
    """Wrap text/x-error parts in <tool_use_error> for the LLM conversation.

    Preserves id and timestamp -- same message, reformatted for LLM.
    """
    changed = False
    parts: list[Message] = []
    for p in cast(tuple[Message, ...], tr.content):
        if p.descriptor == "text/x-error":
            parts.append(
                dataclasses.replace(
                    p,
                    content=f"<tool_use_error>{p.content}</tool_use_error>",
                )
            )
            changed = True
        else:
            parts.append(p)
    if not changed:
        return tr
    return dataclasses.replace(tr, content=tuple(parts))


def _partition_batches(safe_flags: list[bool]) -> list[list[int]]:
    """Group indices into consecutive safe runs or single unsafe items."""
    batches: list[list[int]] = []
    for i, safe in enumerate(safe_flags):
        if batches and safe and safe_flags[batches[-1][0]]:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def _postprocess_results(
    requests: list[Message],
    results: list[Message],
    replacement_state: ReplacementState,
    tool_state: ToolState,
) -> tuple[list[Message], list[dict[str, object]]]:
    """Inject empty markers, persist oversized results, append conditional rules."""
    seen_rules: set[Path] = set()
    log_entries: list[dict[str, object]] = []
    for i, (req, r) in enumerate(zip(requests, results, strict=True)):
        text = text_from_msg(r)
        content = inject_empty_marker(get_tool_name(req), text)
        r_parts = cast(tuple[Message, ...], r.content)
        if not has_error(r):
            preview = persist_result(
                get_queue_id(r), tc_tool_id(req), content, replacement_state
            )
            if preview is not None:
                content = preview
            reminder = conditional_rules_for_request(req, tool_state, seen_rules)
            if reminder:
                content = content.rstrip() + "\n\n" + reminder
        if content != text:
            non_text = tuple(
                p for p in r_parts if p.descriptor not in ("text/plain", "text/x-error")
            )
            results[i] = dataclasses.replace(
                r,
                content=(
                    TextMessage(content, "text/plain"),
                    *non_text,
                ),
            )
        log_entries.append(
            {
                "tool_id": get_queue_id(r),
                "is_error": has_error(r),
                "content_len": len(content),
            }
        )
    return results, log_entries


def _build_model_request(
    *,
    messages: list[Message],
    system: str,
    tools: list[Tool],
    has_background_support: bool,
    max_response_tokens: int,
    thinking: str | None,
    effort: str | None,
    cache_ttl: str,
) -> ModelRequest:
    """Assemble a ``ModelRequest`` from agent state."""
    known_tools: list[Tool] | None = None
    if tools:
        known_tools = (
            cast(
                list[Tool],
                [
                    t
                    if t.tool_id == "application/x-tool-backgroundtask"
                    else BackgroundAwareTool(t)
                    for t in tools
                ],
            )
            if has_background_support
            else tools
        )
    return ModelRequest(
        messages=list(messages),
        system=system,
        tools=known_tools,
        max_response_tokens=max_response_tokens,
        thinking=thinking,
        effort=effort,
        cache_ttl=cache_ttl,
    )


def _estimate_cancelled_response(
    system: str,
    messages: list[Message],
    model: Model,
    chars_streamed: int,
) -> ModelResponse:
    """Build an estimated ``ModelResponse`` for a cancelled request."""
    estimated_input = _estimate_total_tokens(system, messages, model)
    estimated_output = max(1, chars_streamed // 4) if chars_streamed > 0 else 0
    p = model.pricing
    estimated_cost = (
        estimated_input * p.request + estimated_output * p.response
    ) / 1_000_000
    return ModelResponse(
        content=TextMessage("", "text/plain"),
        tokens=TokenCount(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        ),
        total_cost=estimated_cost,
    )
