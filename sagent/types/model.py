"""Model contract and its data classes.

The ``Model`` Protocol, the request and response shapes, ``AgentSettings``,
and the ``ModelRecipe`` used to build a Model from CLI-style strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    get_args,
    runtime_checkable,
)

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.types.cost import TokenCost, TokenCount
from sagent.types.exceptions import UserFacingError
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    RuntimeEvent,
)


if TYPE_CHECKING:
    # ``ModelRequest.tools`` references ``Tool`` from ``tools.py``;
    # ``tools.py`` references ``ToolResult`` from ``history.py``. No
    # runtime cycle because ``from __future__ import annotations`` makes
    # ``Tool`` a forward string at definition time.
    from sagent.types.tools import Tool


__all__ = [
    "CONTEXT_TAGS",
    "AgentSettings",
    "Model",
    "ModelRecipe",
    "ModelRequest",
    "ModelResponse",
    "ModelTerminationError",
    "PromptTooLongError",
    "RequestTooLargeError",
    "StreamInterruptedError",
    "UsageSnapshot",
    "UsageWindow",
    "base_model_id",
    "default_buffer_tokens",
    "split_model_id",
]


# Derived from ``ContextTag``, not restated: a tag the type admits but a
# hand-written tuple omitted would be unparseable while type-checking clean.
# The default window's ``""`` is filtered out -- no id carries it.
CONTEXT_TAGS: Final = tuple(t for t in get_args(ContextTag.__value__) if t)
"""Window-size suffixes a model id may carry (e.g. ``...+1m``)."""


def split_model_id(model_id: str) -> tuple[str, frozenset[ContextTag]]:
    """Split a model id into its base id and trailing context tags.

    Tags may appear in any order; matching is case-insensitive, and an
    unknown suffix stays part of the base id.

    Args:
      model_id: Model id, possibly with trailing context tags.

    Returns:
      base_id: ``model_id`` without its tags.
      tags: The stripped tags, lowercased (e.g. ``{"+1m"}``).

    """
    tags: set[ContextTag] = set()
    base = model_id
    while True:
        lower = base.lower()
        tag = next((t for t in CONTEXT_TAGS if lower.endswith(t)), None)
        if tag is None:
            return base, frozenset(tags)
        tags.add(tag)
        base = base[: -len(tag)]


def base_model_id(model_id: str) -> str:
    """Strip trailing context tags, yielding the canonical model id.

    Args:
      model_id: Model id, possibly with trailing context tags.

    Returns:
      base_id: ``model_id`` without its tags.

    """
    return split_model_id(model_id)[0]


def default_buffer_tokens(max_request_tokens: int) -> int:
    """Return proportional compaction headroom for a given input window.

    Seeds ``AgentSettings.buffer_tokens``; ``Compactor.largest_context``,
    not this function, defines the compaction threshold.

    Args:
      max_request_tokens: The model's input-token window.

    Returns:
      buffer: Tokens of headroom reserved below the effective cap.

    """
    return min(max(max_request_tokens // 15, 8_000), max(max_request_tokens // 2, 0))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSettings:
    """What one Agent chose, within what its model's capability offers.

    There is no ``AgentCapability``: a capability needs an external
    declarer, and an Agent has none -- its ceiling is ``model.capability``.
    Use :meth:`from_limits` for proportional defaults.
    """

    max_request_tokens: int
    """Maximum input tokens the agent will send."""

    max_response_tokens: int
    """Maximum output tokens reserved for the response."""

    buffer_tokens: int = 0
    """Headroom a ``Compactor`` deducts in ``largest_context``, in tokens."""

    max_attempts: int = 5
    """Retry attempts inside one send before the error surfaces."""

    max_tool_call_rounds: int | None = None
    """Cap on tool-call rounds per turn; ``None`` for no cap."""

    max_budget_usd: float | None = None
    """Hard spend cap for this agent's own tree; ``None`` for no cap."""

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
        if self.max_attempts < 1:
            # A zero would send nothing at all: the loop checks the attempt
            # count before the first send, so "no retries" is 1, not 0.
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.max_tool_call_rounds is not None and self.max_tool_call_rounds < 0:
            raise ValueError(
                "max_tool_call_rounds must be >= 0 or None, got"
                f" {self.max_tool_call_rounds}"
            )
        if self.max_budget_usd is not None and self.max_budget_usd < 0:
            raise ValueError(
                f"max_budget_usd must be >= 0 or None, got {self.max_budget_usd}"
            )

    @classmethod
    def from_limits(cls, limits: ModelLimits) -> AgentSettings:
        """Derive proportional defaults from the selected context's limits.

        Takes ``ModelLimits`` rather than a ``Model`` so a caller sizing a
        window it has not built yet -- ``swap_model`` rescaling to a
        candidate -- reaches the same definition as one that has.

        Args:
          limits: Ceilings of the context tag the model selected, i.e.
              ``model.settings.limits(model.capability)``.

        Returns:
          settings: New ``AgentSettings`` with proportional defaults.

        """
        return cls(
            max_request_tokens=limits.max_request_tokens,
            max_response_tokens=limits.max_response_tokens,
            buffer_tokens=default_buffer_tokens(limits.max_request_tokens),
        )


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequest:
    """Full conversation sent to an LLM backend.

    Payload only: a knob a caller CHOOSES lives on ``model.settings``,
    where its capability can validate it.
    """

    messages: list[ModelContextEvent]
    """Conversation history sent to the model."""

    system: str | None = None
    """System prompt; ``None`` omits it from the request."""

    tools: list[Tool] | None = None
    """Tools advertised to the model; ``None`` sends no tool schema."""

    max_response_tokens: int | None = None
    """Max output tokens; ``None`` uses the model default."""

    # Not on ``ModelSettings``: a continuous range cannot join the
    # membership check that validates every other knob.
    temperature: float = 1.0
    """Sampling temperature; transports without the knob ignore it."""

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

    ``model_id``, ``retries_internally``, and ``account_auth`` are read
    from :attr:`capability`; a transport declares them and ``&`` computes
    them, so a second home here would make every provider restate the meet.
    """

    @property
    def capability(self) -> ModelCapability:
        """The catalog row met with this transport's restrictions."""
        ...

    @property
    def settings(self) -> ModelSettings:
        """What this instance chose, validated against :attr:`capability`."""
        ...

    @property
    def limits(self) -> ModelLimits:
        """Ceilings of the selected context tag.

        ``settings.limits(capability)``, named here because ~40 call sites
        read a window and would otherwise each spell the lookup.
        """
        ...

    @property
    def tagged_model_id(self) -> str:
        """Display id carrying its context tag; ``capability.model_id`` is bare."""
        ...

    def approx_text_tokens(self, text: str) -> int:
        """Estimate input tokens for a text string, locally and synchronously.

        Callers must use this rather than dividing a character count by a
        ratio of their own; that mislabels chars as tokens and breaks for
        non-linear tokenizers.

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

    def spend(self, tokens: TokenCount) -> TokenCost:
        """Price ``tokens`` at the tier this instance's settings select.

        Args:
          tokens: What the server reported.

        Returns:
          spend: USD cost, per bucket.

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
        # model that the provider factory rejects with a confusing
        # "no such provider" error far from the construction site.
        # ``account`` may legitimately be empty / ``None`` (default
        # backend).
        if not self.provider:
            raise ValueError("ModelRecipe.provider must be non-empty")
        if not self.auth:
            raise ValueError("ModelRecipe.auth must be non-empty")
        if not self.model_id:
            raise ValueError("ModelRecipe.model_id must be non-empty")
