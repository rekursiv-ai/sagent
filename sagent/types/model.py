"""Model contract and its data classes.

The "how do I call a model" types: the ``Model`` Protocol, the request
and response shapes, the budget and pricing primitives, and the
``ModelRecipe`` recipe used to build a Model from CLI-style strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

from sagent.types.capability import ModelCapability, ModelLimits
from sagent.types.cost import (
    PriceCatalogProduct,
    TokenCost,
    TokenCount,
)
from sagent.types.exceptions import UserFacingError
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    RuntimeEvent,
)
from sagent.types.thinking import (
    ALL_THINKING_EFFORTS,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
)


if TYPE_CHECKING:
    # ``ModelRequest.tools`` references ``Tool`` from ``tools.py``;
    # ``tools.py`` references ``ToolResult`` from ``history.py``. No
    # runtime cycle because ``from __future__ import annotations`` makes
    # ``Tool`` a forward string at definition time.
    from sagent.types.tools import Tool


__all__ = [
    "ALL_THINKING_EFFORTS",
    "CONTEXT_TAGS",
    "LATENCY_TAGS",
    "ContextBudget",
    "Model",
    "ModelCapability",
    "ModelLimits",
    "ModelRecipe",
    "ModelRequest",
    "ModelResponse",
    "ModelSpec",
    "ModelTerminationError",
    "PromptTooLongError",
    "RequestTooLargeError",
    "StreamInterruptedError",
    "ThinkingBudget",
    "ThinkingEffort",
    "ThinkingOutput",
    "TokenCount",
    "UsageSnapshot",
    "UsageWindow",
    "base_model_id",
    "default_buffer_tokens",
    "latency_from_model_id",
    "split_model_id",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSpec(ModelCapability):
    """A capability resolved to one context tag and one latency tier."""

    context_limits: ModelLimits = field(default_factory=ModelLimits)
    """The one context's limits; ``narrow`` collapsed the mapping."""

    context: str = ""
    """The selected context tag (``""`` or e.g. ``"+1m"``)."""

    serve_fast: bool = False
    """Whether this instance requests the fast tier."""

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """Latency hints reachable here.

        ``fast`` needs both transport support and a way to bill it: a
        priced fast row (Anthropic) or a fast service tier (OpenAI).
        """
        if "fast" not in self.latency_modes:
            return ()
        if not self.serves_fast and "priority" not in self.service_tiers:
            return ()
        return tuple(sorted(self.latency_modes))

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """Service tiers reachable here."""
        return tuple(sorted(self.service_tiers))

    def spend(
        self, tokens: TokenCount, *, served_fast: bool | None = None
    ) -> TokenCost:
        """Price ``tokens`` at the tier its request size falls into.

        Tier selection uses the whole prompt: ordinary, cache-write, and
        cache-read pools stay disjoint for billing, but vendors size the
        tier from their sum.

        Args:
          tokens: What the server reported.
          served_fast: Overrides ``serve_fast`` when the vendor reports
              the speed it actually served (Anthropic's ``usage.speed``):
              a request that asked for fast can fall back to standard and
              is billed at standard rates.

        Returns:
          spend: USD cost, per bucket.

        """
        fast = self.serve_fast if served_fast is None else served_fast
        prompt = tokens.request + tokens.cache_write + tokens.cache_read
        return self.prices[PriceCatalogProduct(fast, prompt)] * tokens

    @property
    def tagged_model_id(self) -> str:
        """Display id carrying its option tags; ``model_id`` is the wire id."""
        return f"{self.model_id}{self.context}{'+fast' if self.serve_fast else ''}"

    @classmethod
    def narrow(
        cls,
        cap: ModelCapability,
        /,
        *,
        context: str = "",
        fast: bool = False,
    ) -> Self:
        """Resolve ``cap`` to one context tag.

        A ``ModelSpec`` IS a ``ModelCapability``, so an already-narrowed
        spec type-checks as input here. Narrowing one again keeps its
        tags rather than resetting them, which would leave a spec
        claiming the default context while carrying another's limits.

        Args:
          cap: Catalog row, already met with its transport.
          context: Context tag to select (``""`` for the default).
          fast: Whether this instance serves the fast tier.

        Returns:
          spec: The narrowed spec.

        """
        limits = cap.context_limits
        if isinstance(cap, ModelSpec):
            context = context or cap.context
            fast = fast or cap.serve_fast
        return cls(
            **{
                f.name: getattr(cap, f.name)
                for f in fields(ModelCapability)
                if f.name != "context_limits"
            },
            context=context,
            serve_fast=fast,
            context_limits=limits
            if isinstance(limits, ModelLimits)
            else limits[context],
        )


CONTEXT_TAGS: Final = ("+1m", "+200k")
"""Window-size suffixes a sagent model id may carry (e.g. ``...+1m``)."""

LATENCY_TAGS: Final = ("+fast",)
"""Latency suffixes a sagent model id may carry (e.g. ``...+fast``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageWindow:
    """Utilization of one provider rate-limit window.

    A normalized, provider-agnostic view of a single rolling limit window
    (e.g. Anthropic's 5h / 7d, OpenAI's request / token windows).
    """

    label: str
    """Human-facing window name (e.g. ``"5h"``, ``"7d"``, ``"requests"``)."""

    utilization: float | None = None
    """Fraction in ``[0, 1]`` consumed, or ``None`` when unknown."""

    resets_at: float | None = None
    """Unix wall-clock seconds when the window rolls over, or ``None``."""

    blocked: bool = False
    """True when this window currently rejects (vs. merely warns)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageSnapshot:
    """Normalized rate-limit usage across a provider's windows.

    The provider-agnostic superset surface: every provider maps its own
    rate-limit telemetry (Anthropic ``unified-*`` headers, OpenAI
    ``x-ratelimit-*`` headers) into this shape so the REPL can surface
    usage uniformly. ``None`` from :meth:`Model.usage_snapshot` means "no
    telemetry"; an empty ``windows`` tuple means "telemetry present, nothing
    to report" -- producers here never emit the latter (they return ``None``).
    """

    windows: tuple[UsageWindow, ...] = ()
    """Per-window utilization, in provider declaration order (not sorted)."""


def split_model_id(model_id: str) -> tuple[str, frozenset[str]]:
    """Split a model id into its base id and trailing option tags.

    A sagent model id may carry ``+``-suffixed option tags -- a
    ``+1m`` / ``+200k`` context window and/or ``+fast`` latency -- in
    any order (e.g. ``claude-opus-4-8+1m+fast``). Matching is
    case-insensitive; unknown suffixes stay part of the base id.

    Args:
      model_id: Model id, possibly with trailing option tags.

    Returns:
      base_id: ``model_id`` without its option tags.
      tags: The stripped tags, lowercased (e.g. ``{"+1m", "+fast"}``).

    """
    known = CONTEXT_TAGS + LATENCY_TAGS
    tags: set[str] = set()
    base = model_id
    while True:
        lower = base.lower()
        tag = next((t for t in known if lower.endswith(t)), None)
        if tag is None:
            return base, frozenset(tags)
        tags.add(tag)
        base = base[: -len(tag)]


def base_model_id(model_id: str) -> str:
    """Strip trailing option tags, yielding the canonical model id.

    The wire id, capability lookups, and metadata tables all key off
    the base id.

    Args:
      model_id: Model id, possibly with trailing option tags.

    Returns:
      base_id: ``model_id`` without its option tags.

    """
    return split_model_id(model_id)[0]


def latency_from_model_id(model_id: str) -> str | None:
    """Return the latency hint encoded in a model id's option tags.

    Args:
      model_id: Model id, possibly with trailing option tags.

    Returns:
      latency: ``"fast"`` when the id carries a ``+fast`` tag, else ``None``.

    """
    return "fast" if "+fast" in split_model_id(model_id)[1] else None


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
    """Effort hint; ``None`` omits the field. See ``spec.supported_thinking_efforts``."""

    cache_ttl: Literal["5m", "1h"] = "5m"
    """Prompt-cache TTL; providers without prompt caching ignore this."""

    service_tier: str | None = None
    """Processing-tier hint; ``None`` omits it. See ``spec.valid_service_tiers``."""

    latency: str | None = None
    """Cross-provider latency hint; only ``"fast"`` is defined."""

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

    spend: TokenCost = field(default_factory=TokenCost)
    """USD cost of this request, per token bucket."""

    @property
    def total_cost(self) -> float:
        """Total USD cost of the request."""
        return self.spend.total


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


class RequestTooLargeError(UserFacingError):
    """Raised when a request exceeds the provider's byte wire-limit.

    Distinct from :class:`PromptTooLongError`: the prompt may fit the
    token context window yet exceed the fixed byte ceiling on the HTTP
    request (Anthropic's ~32 MB ``request_too_large``), driven by image /
    PDF attachment bytes that contribute few tokens. Switching to a
    larger-context model does not help -- the byte ceiling is the same --
    so recovery must shed attachment bytes (compaction) rather than widen
    the window.
    """


class StreamInterruptedError(Exception):
    """A stream ended before delivering what it announced.

    Two shapes arrive here: a turn that declared ``tool_use`` and sent no
    tool blocks, and an SSE stream that ended without its terminator. The
    message names whichever happened -- it used to claim tool use in both
    cases, so a text-only connection drop was reported as a tool-call
    failure and sent the reader hunting for tools never in play.
    """

    def __init__(self, response: ModelResponse) -> None:
        tokens = response.tokens
        what = (
            "indicated tool_use but delivered no tool blocks"
            if response.stop_reason == "tool_use"
            else "ended before completing"
        )
        super().__init__(
            f"Stream {what} "
            f"(input_tokens={tokens.request}, "
            f"output_tokens={tokens.response}, "
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
    interface to the runtime's lean ``stream(history, publish) ->
    AssistantMessage`` form. Cost is recorded out-of-band via
    ``Agent.record_response``, which writes through to the root
    ``CostTracker``.
    """

    @property
    def spec(self) -> ModelSpec:
        """What this model can do, narrowed to the selected context.

        Every capability question -- window size, byte ceilings, which
        thinking efforts the wire accepts, whether a fast tier exists --
        is answered from here. Reading it needs no credentials and no
        request.

        Declared read-only so an implementation may store it as a plain
        attribute or derive it; a bare attribute annotation would reject
        the latter.
        """
        ...

    def approx_text_tokens(self, text: str) -> int:
        """Cheap local estimate of input tokens for a text string.

        Synchronous; no I/O. The chars-per-token ratio (or a real
        tokenizer) is a provider-internal detail. Consumers must call
        this for any token estimate and must NOT divide a character count
        by a ratio of their own -- that mislabels chars as tokens and
        breaks for non-linear tokenizers.

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

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request and return the complete response.

        Semantically equivalent to ``stream(request, None)``: both
        return the same parsed ``ModelResponse``. Providers implement
        whichever transport is native to their SDK and delegate the
        other; callers pick ``buffer`` when they have no use for the
        streaming sink.

        Args:
          request: Fully-built model request.

        Returns:
          response: Completed ``ModelResponse``.

        """
        ...

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Send a request and stream events through ``publish``.

        Args:
          request: Fully-built model request.
          publish: Sink for every streamed ``RuntimeEvent`` -- text
              chunks (``ModelResponsePartial``), thinking chunks
              (``ModelResponseThinking``), and, for CLI transports,
              ``ToolLabel`` items emitted from inside the subprocess.
              ``None`` disables streaming (the response is still parsed
              and returned).

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

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Return normalized rate-limit usage from the latest response.

        The provider-agnostic superset surface: providers that receive
        rate-limit telemetry (Anthropic ``unified-*`` headers, OpenAI
        ``x-ratelimit-*`` headers) map it into a :class:`UsageSnapshot`.
        Providers without telemetry return ``None``.

        Returns:
          snapshot: Latest usage across windows, or ``None`` when the
              provider exposes no rate-limit telemetry.

        """
        ...

    async def close(self) -> None:
        """Release any resources the model holds.

        Required, total, and idempotent. CLI-style providers tear down
        their subprocess pool; API providers close their SDK/HTTP
        client; a model that holds nothing returns immediately. Always
        safe to call more than once and safe to call on a never-used
        model. Callers (e.g. ``Agent.swap_model`` / shutdown) invoke it
        unconditionally -- there is no "does this model define close?"
        probe, because every model answers.
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
    """Approximate characters per token, for the two re-attach caps only.

    Every OTHER field below is a token count. Re-attach reads whole files
    off disk before any tokenizer sees them, so its two caps stay in
    characters; nothing else may reintroduce the conversion.
    """

    buffer_tokens: int = 0
    """Headroom (tokens) before force-compaction triggers."""

    reattach_count: int = 5
    """Number of recently-read files to re-attach post-compaction."""

    reattach_max_chars: int = 0
    """Per-file character cap for re-attached files."""

    reattach_budget: int = 0
    """Total character budget for all re-attached files."""

    persist_tokens: int = 0
    """Per-result token threshold for disk offloading."""

    message_budget_tokens: int = 0
    """Per-request aggregate token budget for tool results."""

    keep_recent_on_compact: int | None = None
    """Recent entries the compactor keeps verbatim; ``None`` lets it choose."""

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
        if self.persist_tokens < 0:
            raise ValueError(f"persist_tokens must be >= 0, got {self.persist_tokens}")
        if self.message_budget_tokens < 0:
            raise ValueError(
                f"message_budget_tokens must be >= 0, got {self.message_budget_tokens}"
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
          model: Model whose limits and measured tokenizer density seed
              the budget.

        Returns:
          budget: New ``ContextBudget`` with proportional defaults.

        """
        inp = model.spec.context_limits.max_request_tokens
        out = model.spec.context_limits.max_response_tokens
        # Only the two re-attach caps still need the ratio: they bound
        # file bytes read off disk, before any tokenizer sees them.
        cpt = max(1, round(model.spec.chars_per_token))
        buffer = default_buffer_tokens(inp)
        return cls(
            max_request_tokens=inp,
            max_response_tokens=out,
            chars_per_token=cpt,
            buffer_tokens=buffer,
            reattach_count=5,
            reattach_max_chars=cpt * max(inp // 40, 2_000),
            reattach_budget=cpt * max(inp // 4, 10_000),
            # No floor. A floor is not a bound: at ``gpt-4``'s 8_192-token
            # window a 20_000-token floor let ONE tool result exceed the
            # whole context 2.4x. ``offset`` reaches whatever a smaller
            # cap left behind, so erring small is recoverable and erring
            # large is not.
            persist_tokens=inp // 4,
            message_budget_tokens=inp // 2,
            # ``None`` defers to the compactor's own ``keep_recent``
            # default, which adapts per-strategy; baking a number here
            # would override that without the caller knowing.
            keep_recent_on_compact=None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRecipe:
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
            raise ValueError("ModelRecipe.provider must be non-empty")
        if not self.auth:
            raise ValueError("ModelRecipe.auth must be non-empty")
        if not self.model_id:
            raise ValueError("ModelRecipe.model_id must be non-empty")
