"""Model contract and its data classes.

The "how do I call a model" types: the ``Model`` Protocol, the request
and response shapes, the budget and pricing primitives, and the
``ModelSpec`` recipe used to build a Model from CLI-style strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
)


if TYPE_CHECKING:
    # ``ModelRequest.tools`` references ``Tool`` from ``tools.py``;
    # ``tools.py`` references ``ToolResult`` from ``history.py``. No
    # runtime cycle because ``from __future__ import annotations`` makes
    # ``Tool`` a forward string at definition time.
    from sagent.types.tools import Tool


__all__ = [
    "CONTEXT_TAGS",
    "ContextBudget",
    "Model",
    "ModelRequest",
    "ModelResponse",
    "ModelSpec",
    "ModelTerminationError",
    "Pricing",
    "PromptTooLongError",
    "StreamInterruptedError",
    "TokenCount",
    "base_model_id",
    "default_buffer_tokens",
]


CONTEXT_TAGS = ("+1m", "+200k")
"""Window-size suffixes a sagent model id may carry (e.g. ``...+1m``)."""


def base_model_id(model_id: str) -> str:
    """Strip a trailing context-window tag, yielding the canonical model id.

    A sagent model id may carry a ``+1m`` / ``+200k`` window suffix, but the
    wire id, capability lookups, and metadata tables all key off the base id.
    Matching is case-insensitive; an id with no known tag is returned as-is.

    Args:
      model_id: Model id, possibly with a trailing window tag.

    Returns:
      base_id: ``model_id`` without its window tag.

    """
    lower = model_id.lower()
    for tag in CONTEXT_TAGS:
        if lower.endswith(tag):
            return model_id[: -len(tag)]
    return model_id


@dataclass(frozen=True, slots=True, kw_only=True)
class Pricing:
    """Per-million-token prices in USD.

    ``fast_request`` / ``fast_response`` apply when the provider reports
    that a request actually ran in fast mode (Anthropic ``usage.speed``
    == ``"fast"``). The server is authoritative: a request opted in
    that fell back to standard speed is billed at standard rates.
    """

    request: float = 0.0
    """Price per million input tokens."""

    response: float = 0.0
    """Price per million output tokens."""

    cache_write: float = 0.0
    """Price per million tokens written to prompt cache."""

    cache_read: float = 0.0
    """Price per million tokens served from prompt cache."""

    fast_request: float = 0.0
    """Price per million input tokens when the server billed fast mode.
    ``0.0`` (default) means fast mode isn't priced; callers should not
    select fast pricing for this model."""

    fast_response: float = 0.0
    """Price per million output tokens when the server billed fast mode."""


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
        # Runtime guard against non-TokenCount operands: returning
        # ``NotImplemented`` lets Python try the reflected dunder or
        # raise a clear ``TypeError``. The annotation promises
        # ``TokenCount``; pyright sees the isinstance as redundant, but
        # at runtime callers reaching through ``object``-typed plumbing
        # (status pane, persisted-metadata round-trip) can still pass a
        # non-``TokenCount``; the unchecked code raised
        # ``AttributeError`` mid-expression there.
        if not isinstance(other, TokenCount):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented  # pyright: ignore[reportUnreachable]
        return TokenCount(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens
            + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    def __sub__(self, other: TokenCount) -> TokenCount:
        # See ``__add__``: same runtime guard, same reasoning.
        if not isinstance(other, TokenCount):  # pyright: ignore[reportUnnecessaryIsInstance]
            return NotImplemented  # pyright: ignore[reportUnreachable]
        # Clamp at zero per field: a snapshot taken before a user-initiated
        # ``CostTracker.restore_totals`` may exceed the post-restore total,
        # producing a negative delta the status pane would render as
        # "-12 tokens". Subtraction is unchecked everywhere else; this is
        # the one place an external mutation breaks monotonicity.
        return TokenCount(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            cache_creation_tokens=max(
                0, self.cache_creation_tokens - other.cache_creation_tokens
            ),
            cache_read_tokens=max(0, self.cache_read_tokens - other.cache_read_tokens),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequest:
    """Full conversation sent to an LLM backend."""

    messages: list[ModelContextEvent]
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
    """Effort hint; provider-specific accepted values (see provider
    docs -- e.g. Anthropic accepts ``low``/``medium``/``high``/``xhigh``/
    ``max``, Qwen3 enables hybrid thinking on any non-``none`` value).
    ``None`` omits the field so the API applies its own default."""

    cache_ttl: Literal["5m", "1h"] = "5m"
    """Prompt-cache TTL; providers without prompt caching ignore this."""

    service_tier: str | None = None
    """Processing-tier hint; accepted values are provider-specific (see
    ``Model.valid_service_tiers``). Providers without service-tier
    support ignore this field. ``None`` omits the hint so the API
    applies its own default."""

    latency: str | None = None
    """Cross-provider latency hint; currently only ``"fast"`` is defined.
    Each provider maps it to its own wire field: Anthropic Opus 4.6/4.7/4.8
    to ``speed="fast"`` (fast mode), OpenAI to ``service_tier="priority"``.
    Providers without a fast path reject it at the ``Agent``/``AgentSelf``
    boundary (the setter validates against ``Model.valid_latency_modes``);
    a request that still reaches such a provider has it dropped. ``None``
    requests the default."""

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


class PromptTooLongError(Exception):
    """Raised by providers when the prompt exceeds model limits."""

    def __init__(
        self,
        message: str = "prompt too long",
        *,
        actual_tokens: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_tokens = actual_tokens
        self.limit_tokens = limit_tokens

    @property
    def token_gap(self) -> int | None:
        """Tokens over the limit; ``None`` if unknown, ``0`` if exactly at cap.

        Contract:
          - ``None``: ``actual_tokens`` or ``limit_tokens`` is unknown.
          - ``0``: prompt sits exactly at the limit (provider rejected
            it but the gap was zero -- treat as at-cap, not "unknown").
          - ``>0``: prompt overshoot, in tokens.

        Callers branching on "is this overflow recoverable?" should use
        ``gap is not None and gap >= 0`` for the known-shape case; the
        old ``gap > 0`` branch silently merged "at cap" into "unknown".
        """
        if self.actual_tokens is not None and self.limit_tokens is not None:
            return max(0, self.actual_tokens - self.limit_tokens)
        return None


class StreamInterruptedError(Exception):
    """Stream indicated tool use but delivered no tool blocks."""

    def __init__(self, response: ModelResponse) -> None:
        tokens = response.tokens
        super().__init__(
            "Stream indicated tool_use but delivered no tool blocks "
            f"(input_tokens={tokens.input_tokens}, "
            f"output_tokens={tokens.output_tokens}, "
            f"stop_reason={response.stop_reason!r}).",
        )
        self.response = response


class ModelTerminationError(Exception):
    """Model stopped with an unrecognized non-benign ``stop_reason``."""

    def __init__(self, response: ModelResponse) -> None:
        tool_count = len(response.message.tool_calls)
        text_len = len(response.message.text)
        super().__init__(
            f"Model stopped with unrecognized stop_reason="
            f"{response.stop_reason!r} (tool_calls={tool_count}, "
            f"text_len={text_len}).",
        )
        self.response = response
        self.stop_reason = response.stop_reason


@runtime_checkable
class Model(Protocol):
    """Provider-side model interface.

    The Agent layer's ``_AgentModel`` wrapper bridges this richer
    interface to the runtime's lean ``stream(history, system,
    tools, on_text, on_thinking) -> AssistantMessage`` form. Cost is
    recorded out-of-band via ``Agent.record_response``, which writes
    through to the root ``CostTracker``.
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
    def valid_thinking_states(self) -> tuple[str, ...]:
        """Accepted ``/thinking`` states for this provider/model.

        Provider-specific, derived from the model's thinking capability
        (see :func:`sagent.thinking.valid_thinking_states`).
        Always contains ``off-hide``. Excludes ``-show`` states when the
        provider forces server-side redaction (no readable text), and
        excludes ``redact-hide`` when the provider exposes no redaction
        mode. A state outside this set is a rejected request, not a no-op.
        """
        ...

    @property
    def supports_effort(self) -> bool:
        """Whether the model accepts an effort hint."""
        ...

    @property
    def valid_efforts(self) -> tuple[str, ...]:
        """Accepted ``effort`` values; empty when unsupported.

        Provider-specific. Anthropic exposes ``low`` / ``medium`` /
        ``high`` / ``xhigh`` / ``max``; OpenAI reasoning models and Google
        expose their own sets. An empty tuple means the model takes no
        effort hint (``supports_effort`` is ``False``). A value outside
        this set is a rejected request, not a silent no-op.
        """
        ...

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt caching."""
        ...

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """Accepted ``service_tier`` values; empty when unsupported.

        Provider-specific. OpenAI chat-completions exposes ``"auto"`` /
        ``"default"`` / ``"flex"`` / ``"priority"``; OpenAI Codex
        subscription only exposes ``"priority"``; Anthropic Messages
        exposes ``"auto"`` / ``"standard_only"``. An empty tuple means
        the request field is dropped.
        """
        ...

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """Accepted ``latency`` values; empty when unsupported.

        Anthropic Opus 4.6/4.7/4.8 (API + subscription transports) and
        OpenAI (API + subscription) accept ``("fast",)``. The Anthropic
        CLI transport and providers without a fast path return ``()``.
        ``None`` is always implicitly valid and means default latency.
        """
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

    def approx_text_tokens(self, text: str) -> int:
        """Cheap local estimate of input tokens for a text string.

        Synchronous; no I/O. Uses ``chars_per_token`` or a local heuristic.

        Args:
          text: Text to score.

        Returns:
          tokens: Approximate input token count.

        """
        ...

    def approx_image_tokens(self, data: bytes) -> int:
        """Cheap local estimate of input tokens for an image.

        Synchronous; no I/O. Uses a fixed per-image constant or a
        dimension-based formula.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Approximate input token count.

        """
        ...

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Cheap local estimate of input tokens for a full request.

        Synchronous; no I/O. Sums all wire-bearing surfaces of
        ``request``: system prompt, every text-bearing field on every
        history entry (including ``ToolCall.args`` and
        ``thinking_blocks``), image attachments, and the tools schema.
        Use this for hot-path proactive compaction decisions.

        Args:
          request: Fully-built model request.

        Returns:
          tokens: Approximate input token count.

        """
        ...

    async def actual_text_tokens(self, text: str) -> int:
        """Provider's best-truth input-token count for a text string.

        Asynchronous because some providers must roundtrip
        (Anthropic ``messages.count_tokens``, Google ``models.countTokens``).
        Providers with a local tokenizer (OpenAI ``tiktoken``, self-hosted
        HF) answer without I/O.

        Args:
          text: Text to score.

        Returns:
          tokens: Best-truth input token count.

        """
        ...

    async def actual_image_tokens(self, data: bytes) -> int:
        """Provider's best-truth input-token count for an image.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Best-truth input token count.

        """
        ...

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Provider's best-truth input-token count for a full request.

        Falls back to ``approx_request_tokens`` on providers without a
        truth source (CLI variants without tokenizer access).

        Args:
          request: Fully-built model request.

        Returns:
          tokens: Best-truth input token count.

        """
        ...

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        ...

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request and return the complete response.

        Semantically equivalent to ``stream(request, None, None)``:
        both return the same parsed ``ModelResponse``. Providers
        implement whichever transport is native to their SDK and
        delegate the other; callers pick ``buffer`` when they have no
        use for streaming callbacks.

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


def default_buffer_tokens(max_request_tokens: int) -> int:
    """Proportional compaction headroom for a given input window.

    The single source of truth for force-compaction headroom: both
    ``ContextBudget.from_model`` (the reactive/scrunch reservation) and
    ``SummaryCompactor.should_compact`` (the proactive trigger) derive
    their buffer here, so one rule governs when compaction fires across a
    model swap. Scales as ``max_request_tokens // 15`` with an 8_000-token
    floor, capped at half the window so it never collides with the
    ``buffer_tokens < max_request_tokens`` budget invariant.

    Args:
      max_request_tokens: The model's input-token window.

    Returns:
      buffer: Tokens of headroom reserved below the effective cap.

    """
    return min(max(max_request_tokens // 15, 8_000), max(max_request_tokens // 2, 0))


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
        if self.buffer_tokens < 0 or self.buffer_tokens >= self.max_request_tokens:
            raise ValueError(
                f"buffer_tokens ({self.buffer_tokens}) must be in"
                f" [0, max_request_tokens={self.max_request_tokens})"
            )
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {self.chars_per_token}")
        if self.reattach_count < 0:
            raise ValueError(f"reattach_count must be >= 0, got {self.reattach_count}")
        if self.reattach_max_chars < 0:
            raise ValueError(
                f"reattach_max_chars must be >= 0, got {self.reattach_max_chars}"
            )
        if self.reattach_budget < 0:
            raise ValueError(
                f"reattach_budget must be >= 0, got {self.reattach_budget}"
            )
        if self.persist_threshold < 0:
            raise ValueError(
                f"persist_threshold must be >= 0, got {self.persist_threshold}"
            )
        if self.message_budget_chars < 0:
            raise ValueError(
                f"message_budget_chars must be >= 0, got {self.message_budget_chars}"
            )
        if self.keep_recent_on_compact is not None and self.keep_recent_on_compact < 0:
            raise ValueError(
                "keep_recent_on_compact must be >= 0 or None, got"
                f" {self.keep_recent_on_compact}"
            )

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
        buffer = default_buffer_tokens(inp)
        return cls(
            max_request_tokens=inp,
            max_response_tokens=out,
            chars_per_token=cpt,
            buffer_tokens=buffer,
            reattach_count=5,
            reattach_max_chars=cpt * max(inp // 40, 2_000),
            reattach_budget=cpt * max(inp // 4, 10_000),
            persist_threshold=cpt * max(inp // 4, 20_000),
            message_budget_chars=cpt * max(inp // 2, 20_000),
            # ``None`` defers to the compactor's own ``keep_recent``
            # default, which adapts per-strategy; baking a number here
            # would override that without the caller knowing.
            keep_recent_on_compact=None,
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

    def __post_init__(self) -> None:
        # Empty ``provider``/``auth``/``model_id`` produce a degenerate
        # spec that the provider factory rejects with a confusing
        # "no such provider" error far from the construction site.
        # ``account`` may legitimately be empty / ``None`` (default
        # backend).
        if not self.provider:
            raise ValueError("ModelSpec.provider must be non-empty")
        if not self.auth:
            raise ValueError("ModelSpec.auth must be non-empty")
        if not self.model_id:
            raise ValueError("ModelSpec.model_id must be non-empty")
