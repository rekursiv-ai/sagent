"""Adapter-layer types: ModelRequest, ModelResponse, Tool, Model, ...

Splits cleanly from ``agent/runtime.py``:

- ``agent/runtime.py`` (locked): the runtime engine, the dispatch loop,
  the history dataclasses (``UserMessage``, ``AssistantMessage``,
  ``ToolResult``, ``ToolCall``, ``BytesMessage``), the ``RuntimeEvent``
  union, and the minimal ``Tool`` / ``Model`` / ``Compactor`` protocols
  the runtime consumes.

- This module: the richer interfaces the *wrappers* in ``agent/agent.py``
  bridge to those minimal protocols. Providers expose ``ProviderModel``
  with ``buffer(request: ModelRequest) -> ModelResponse``; tools expose
  ``Tool`` with metadata (``tool_id``, ``description``,
  ``directive_schema``, ``summary`` / ``summary_result`` / ``prompt``)
  plus the ``run(args) -> ToolResult`` signature. The Agent layer
  composes them.

See ``docs/private/agent_v4_contract.md`` for the binding spec.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from sagent.agent.runtime import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
)
from sagent.lib.json import JSON


if TYPE_CHECKING:
    # ToolState lives in tools/core.py; we forward-reference it on
    # ``CompactRestorable.post_compact_restore`` to avoid pulling the
    # tools layer into the runtime types module.
    from sagent.tools.core import ToolState


@dataclass(frozen=True, slots=True, kw_only=True)
class Pricing:
    """Per-million-token prices in USD."""

    request: float = 0.0
    """Price per million input tokens."""

    response: float = 0.0
    """Price per million output tokens."""

    cache_write: float = 0.0
    """Price per million tokens written to prompt cache."""

    cache_read: float = 0.0
    """Price per million tokens served from prompt cache."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCount:
    """Immutable 4-tuple of token counts returned by a model request."""

    input_tokens: int = 0
    """Input (prompt) tokens."""

    output_tokens: int = 0
    """Output (response) tokens."""

    cache_creation_tokens: int = 0
    """Tokens spent creating cache breakpoints."""

    cache_read_tokens: int = 0
    """Tokens served from prompt cache."""

    def __add__(self, other: TokenCount) -> TokenCount:
        return TokenCount(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens
            + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    def __sub__(self, other: TokenCount) -> TokenCount:
        return TokenCount(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens
            - other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens - other.cache_read_tokens,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequest:
    """Full conversation sent to an LLM backend."""

    messages: list[HistoryEntry]
    """Conversation history sent to the model."""

    system: str | None = None
    """System prompt; ``None`` omits it from the request."""

    tools: list[Tool] | None = None
    """Tools advertised to the model; ``None`` sends no tool schema."""

    max_response_tokens: int | None = None
    """Max output tokens; ``None`` uses the model default."""

    temperature: float = 1.0
    """Sampling temperature."""

    thinking: str | None = None
    """Extended-thinking mode; ``None`` disables thinking."""

    effort: str | None = None
    """Effort hint (Anthropic: ``low``/``medium``/``high``/``xhigh``/
    ``max``; Qwen3: any non-``none`` value enables hybrid thinking).
    ``None`` omits the field so the API applies its own default."""

    cache_ttl: str = "5m"
    """Prompt-cache TTL (``5m`` or ``1h``); providers without prompt
    caching ignore this field."""

    stop_sequences: tuple[str, ...] = ()
    """Optional stop sequences; provider-specific support."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponse:
    """What comes back from an LLM backend."""

    message: AssistantMessage
    """Parsed assistant message (text, thinking blocks, tool calls)."""

    tokens: TokenCount = field(default_factory=TokenCount)
    """Token counts for this request."""

    stop_reason: str = "model_finished"
    """Canonical stop reason; see ``providers/lib/stop_reason.py``."""

    stop_sequence: str | None = None
    """Matched stop sequence, if any."""

    message_id: str = ""
    """Provider-assigned message id."""

    request_id: str = ""
    """HTTP-level request id from the response headers."""

    input_cost: float = 0.0
    """USD cost of the prompt tokens."""

    output_cost: float = 0.0
    """USD cost of the generated tokens."""

    total_cost: float = 0.0
    """Total USD cost of the request."""


@runtime_checkable
class Tool(Protocol):
    """Tool interface for the wrapper layer.

    The runtime sees only ``name`` + ``run``; the wrapper layer
    consumes the rest (``tool_id``, ``description``,
    ``directive_schema``, ``summary``, ``summary_result``, ``prompt``,
    ``supports_microcompaction``).
    """

    name: str
    """Human-readable tool name, e.g. ``"Bash"``."""

    tool_id: str
    """MIME-style identifier, e.g. ``"application/x-tool-bash"``."""

    description: str
    """Human/model-facing description rendered into the tool schema."""

    directive_schema: JSON
    """Frozen JSON Schema for the tool's directive."""

    supports_microcompaction: bool
    """Whether old results of this tool are eligible for microcompaction
    (clearing on cache-cold)."""

    def summary(self, args: Mapping[str, object]) -> str:
        """Build a short label for a pending invocation.

        Args:
          args: Directive arguments to be passed to ``run``.

        Returns:
          label: Pre-execution label for renderers.

        """
        ...

    def summary_result(self, result: ToolResult) -> str | None:
        """Build a short receipt for a completed invocation.

        Args:
          result: The completed ``ToolResult``.

        Returns:
          receipt: Short receipt line, or ``None`` to suppress it.

        """
        ...

    def prompt(self) -> str:
        """Per-request system-prompt contribution for this tool.

        Returns:
          contribution: Prompt fragment, or ``""`` for no contribution.

        """
        ...

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the tool with parsed args.

        Must not raise; populate ``ToolResult(is_error=True)`` on
        failure.

        Args:
          args: Directive arguments parsed from the model output.

        Returns:
          result: Completed tool result.

        """
        ...


@runtime_checkable
class Model(Protocol):
    """Provider-side model interface.

    The Agent layer's ``_AgentModel`` wrapper bridges this richer
    interface to ``runtime.Model``'s lean ``stream(history, system,
    tools, on_text, on_thinking) -> AssistantMessage`` form. Cost is
    recorded out-of-band via ``agent.cost_tracker.record(response)``.
    """

    @property
    def max_request_tokens(self) -> int:
        """Maximum input tokens the model accepts."""
        ...

    @property
    def model_id(self) -> str:
        """Provider-specific model identifier."""
        ...

    @property
    def max_response_tokens(self) -> int:
        """Maximum output tokens the model can generate."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether the model supports token-by-token streaming."""
        ...

    @property
    def supports_thinking(self) -> bool:
        """Whether the model supports extended thinking."""
        ...

    @property
    def supports_effort(self) -> bool:
        """Whether the model accepts an effort hint."""
        ...

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt caching."""
        ...

    @property
    def supports_context_management(self) -> bool:
        """Whether the provider manages context overflow internally."""
        ...

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether the provider retries internally on transient
        failures.
        """
        ...

    @property
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account-based authentication."""
        ...

    @property
    def max_image_dim(self) -> int:
        """Maximum image dimension (pixels) accepted by the API."""
        ...

    @property
    def max_image_bytes(self) -> int:
        """Maximum image size (bytes) accepted by the API."""
        ...

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate input token count for a text string.

        Args:
          text: Text to score.

        Returns:
          tokens: Approximate input token count.

        """
        ...

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate input token count for an image.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Approximate input token count.

        """
        ...

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        ...

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request and return the complete response (no streaming).

        Args:
          request: Fully-built model request.

        Returns:
          response: Completed ``ModelResponse``.

        """
        ...

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Send a request and stream tokens through the callbacks.

        Args:
          request: Fully-built model request.
          on_text: Called per text chunk; ``None`` disables text streaming.
          on_thinking: Called per thinking chunk; ``None`` disables it.

        Returns:
          response: Completed ``ModelResponse``.

        """
        ...

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify an error as a context-window overflow.

        Args:
          error: Exception raised by the provider call.

        Returns:
          overflow: True when ``error`` indicates context overflow.

        """
        ...

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Classify an error using provider-specific retry heuristics.

        Args:
          error: Exception raised by the provider call.

        Returns:
          retryable: True when the provider treats ``error`` as transient.

        """
        ...


@runtime_checkable
class Provider(Protocol):
    """Factory for model backends. ``None`` → provider's ``DEFAULT_MODEL``."""

    def model(
        self, model_id: str | None = None, /, max_request_tokens: int | None = None
    ) -> Model:
        """Build a model backend.

        Args:
          model_id: Provider-specific id; ``None`` selects the
              provider's ``DEFAULT_MODEL``.
          max_request_tokens: Override for the model's input cap.

        Returns:
          model: A ``Model`` ready to handle requests.

        """
        ...

    def utility_model(self) -> Model:
        """Build the cheapest/fastest model for utility tasks.

        Returns:
          model: A low-cost ``Model`` for internal use (summarizers, etc.).

        """
        ...


@runtime_checkable
class Compactor(Protocol):
    """Conversation compaction strategy.

    The Agent layer's ``_AgentCompactor`` wrapper bridges this rich
    interface to ``runtime.Compactor``'s lean ``compact(history,
    model, args) -> list[HistoryEntry]`` form, threading
    ``transcript_path``, ``direction``, ``keep_recent``,
    ``custom_instructions``, and ``summary_pointers`` from agent
    state.
    """

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Decide whether the conversation should be compacted now.

        Args:
          input_tokens: Estimated current input token count.
          max_request_tokens: Budget cap for input tokens.
          max_response_tokens: Reserved output tokens (subtracted from cap).

        Returns:
          should: True when compaction should run before the next call.

        """
        ...

    async def compact(
        self,
        history: list[HistoryEntry],
        model: Model,
        transcript_path: Path | None = None,
        direction: Literal["from", "up_to"] = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[HistoryEntry]:
        """Summarize the conversation into a compact entry list.

        Args:
          history: Full conversation history.
          model: Model used to write the summary.
          transcript_path: Optional path of the pre-compact transcript on
              disk (for ``Recompact``).
          direction: ``"from"`` keeps tail, ``"up_to"`` keeps head.
          keep_recent: Number of recent entries preserved verbatim.
          custom_instructions: Extra instructions for the summarizer.
          summary_pointers: ``(path, topic)`` pairs to surface in the summary.

        Returns:
          summary: Compacted history.

        """
        ...

    def maintain(
        self,
        history: list[HistoryEntry],
        tools: dict[str, Tool],
        **kwargs: object,
    ) -> None:
        """Apply between-request context maintenance (microcompaction).

        Args:
          history: Conversation history; mutated in place.
          tools: Tool registry; consulted for tool-specific trimming rules.
          **kwargs: Reserved for future maintenance hooks.

        """


@runtime_checkable
class CompactRestorable(Protocol):
    """Optional protocol for tools that restore state after compaction.

    Tools opt in by implementing ``post_compact_restore``; the
    compactor wrapper invokes it on every tool that satisfies the
    protocol.
    """

    async def post_compact_restore(
        self,
        history: list[HistoryEntry],
        tool_state: ToolState,
        *,
        budget_chars: int = 100_000,
    ) -> None:
        """Re-inject tool-specific context into history after compaction.

        Best-effort; failure is logged and swallowed by the caller.

        Args:
          history: Post-compaction history; mutated in place.
          tool_state: Active tool state for tool-specific lookups.
          budget_chars: Character budget the hook should respect.

        """


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBudget:
    """How an Agent allocates its context window across competing uses.

    Use ``from_model`` for proportional defaults; override individual
    fields with ``dataclasses.replace`` or pass an explicit instance
    to ``Agent(budget=...)``.
    """

    max_request_tokens: int
    """Maximum input tokens the agent will send."""

    max_response_tokens: int
    """Maximum output tokens reserved for the response."""

    chars_per_token: int = 4
    """Approximate characters per token for budget math."""

    buffer_tokens: int = 0
    """Headroom (tokens) before force-compaction triggers."""

    reattach_count: int = 5
    """Number of recently-read files to re-attach post-compaction."""

    reattach_max_chars: int = 0
    """Per-file character cap for re-attached files."""

    reattach_budget: int = 0
    """Total character budget for all re-attached files."""

    persist_threshold: int = 0
    """Per-result character threshold for disk offloading."""

    message_budget_chars: int = 0
    """Per-request aggregate character budget for tool results."""

    keep_recent_on_compact: int | None = None
    """Number of recent history entries the compactor preserves
    verbatim; ``None`` lets the compactor choose."""

    def __post_init__(self) -> None:
        if self.max_request_tokens <= 0:
            raise ValueError(
                f"max_request_tokens must be > 0, got {self.max_request_tokens}"
            )
        if self.max_response_tokens <= 0:
            raise ValueError(
                f"max_response_tokens must be > 0, got {self.max_response_tokens}"
            )
        if self.buffer_tokens >= self.max_request_tokens:
            raise ValueError(
                f"buffer_tokens ({self.buffer_tokens}) must be <"
                f" max_request_tokens ({self.max_request_tokens})"
            )
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {self.chars_per_token}")

    @classmethod
    def from_model(cls, model: Model) -> ContextBudget:
        """Derive proportional ``ContextBudget`` defaults from a model's limits.

        Args:
          model: Model whose ``max_request_tokens`` and
              ``max_response_tokens`` seed the budget.

        Returns:
          budget: New ``ContextBudget`` with proportional defaults.

        """
        inp = model.max_request_tokens
        out = model.max_response_tokens
        cpt = 4
        return cls(
            max_request_tokens=inp,
            max_response_tokens=out,
            chars_per_token=cpt,
            buffer_tokens=max(inp // 15, 8_000),
            reattach_count=5,
            reattach_max_chars=cpt * max(inp // 40, 2_000),
            reattach_budget=cpt * max(inp // 4, 10_000),
            persist_threshold=cpt * max(inp // 4, 20_000),
            message_budget_chars=cpt * max(inp // 2, 20_000),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSpec:
    """Recipe for building a ``Model`` from CLI-style strings."""

    provider: str
    """Provider class name, e.g. ``"Anthropic"``, ``"Google"``."""

    auth: str
    """Auth method suffix, e.g. ``"api"``, ``"sub"``."""

    model_id: str
    """Provider-specific model identifier."""

    account: str | None = None
    """Optional account override (used by account auth)."""
