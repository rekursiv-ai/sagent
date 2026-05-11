"""Core types: Message, BatchingTool, StreamingTool, Tool, Model, Provider, ModelRequest/Response.

See ``tools/__init__.py`` for the tool thesis, descriptor conventions,
error handling contract, and peer pattern documentation.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable
from typing_extensions import TypeIs

import base64
import itertools
import time


if TYPE_CHECKING:
    from sagent.tools.core import ToolState

from sagent.lib.descriptors import (
    BytesDescriptor,
    MultipartDescriptor,
    TextDescriptor,
)


__all__ = ["BytesDescriptor", "MultipartDescriptor", "TextDescriptor"]
from sagent.lib.json import (
    JSON,
    MutableJSON,
    MutableJSONValue,
    json_freeze,
    json_unfreeze,
)


type MessageContent = str | bytes | JSON | tuple[Message, ...]

# -- Message variants -------------------------------------------------------

_id_counter: itertools.count[int] = itertools.count()


def reset_id_counter(start: int) -> None:
    global _id_counter  # noqa: PLW0603 -- module-level counter requires global statement
    _id_counter = itertools.count(start)


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageBase:
    parent_id: int = -1
    id: int = field(default_factory=lambda: next(_id_counter))
    timestamp: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        if type(self) is MessageBase:
            raise TypeError(
                "Use TextMessage, MultipartMessage, BytesMessage, or JsonMessage"
            )

    def serialize(self) -> MutableJSON:
        return _serialize(cast("Message", self))

    @staticmethod
    def deserialize(
        data: MutableJSON,
    ) -> TextMessage | MultipartMessage | BytesMessage | JsonMessage:
        return _deserialize(data)


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class TextMessage(MessageBase):
    content: str = ""
    descriptor: TextDescriptor = "text/plain"


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class MultipartMessage(MessageBase):
    content: tuple[Message, ...] = ()
    descriptor: MultipartDescriptor = "multipart/x-tool-result"


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class BytesMessage(MessageBase):
    content: bytes = b""
    descriptor: BytesDescriptor = "image/png"


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class JsonMessage(MessageBase):
    content: JSON = field(default_factory=lambda: cast(JSON, {}))
    descriptor: str = "application/json"


type Message = TextMessage | MultipartMessage | BytesMessage | JsonMessage


def is_message(obj: object) -> TypeIs[Message]:
    """Type-narrowing check: True when *obj* is any Message variant."""
    return isinstance(obj, MessageBase)


def _serialize(msg: Message) -> MutableJSON:
    timing: MutableJSON = {
        "_id": msg.id,
        "_parent_id": msg.parent_id,
        "_timestamp": msg.timestamp,
    }
    content: MutableJSONValue
    if isinstance(msg, MultipartMessage):
        content = cast(MutableJSONValue, [_serialize(p) for p in msg.content])
    elif isinstance(msg, BytesMessage):
        content = cast(
            MutableJSONValue, {"_bytes": base64.b64encode(msg.content).decode()}
        )
    elif isinstance(msg, JsonMessage):
        content = json_unfreeze(msg.content)
    else:
        content = msg.content
    return {"descriptor": msg.descriptor, "content": content, **timing}


def _deserialize(data: MutableJSON) -> Message:
    descriptor = str(data["descriptor"])
    raw = data.get("content")
    if isinstance(raw, list):
        parts = tuple(_deserialize(cast(MutableJSON, p)) for p in raw)
        return _restore(
            MultipartMessage(
                content=parts, descriptor=cast(MultipartDescriptor, descriptor)
            ),
            data,
        )
    if isinstance(raw, dict) and "_bytes" in raw:
        payload = base64.b64decode(cast(str, cast(MutableJSON, raw)["_bytes"]))
        return _restore(
            BytesMessage(content=payload, descriptor=cast(BytesDescriptor, descriptor)),
            data,
        )
    if isinstance(raw, dict):
        return _restore(
            JsonMessage(
                content=json_freeze(cast(MutableJSON, raw)), descriptor=descriptor
            ),
            data,
        )
    return _restore(
        TextMessage(
            content=str(raw) if raw is not None else "",
            descriptor=cast(TextDescriptor, descriptor),
        ),
        data,
    )


def _restore(msg: Message, data: MutableJSON) -> Message:
    for attr, key in (
        ("id", "_id"),
        ("parent_id", "_parent_id"),
        ("timestamp", "_timestamp"),
    ):
        v = data.get(key)
        if isinstance(v, int):
            object.__setattr__(msg, attr, v)
    return msg


class _BaseTool(Protocol):
    """Shared shape for all tools -- only ``run`` differs."""

    name: str
    """Human-readable tool name, e.g. ``"Bash"``."""

    tool_id: str
    """MIME identifier, e.g. ``"application/x-tool-bash"``."""

    description: str
    """Human/model-facing description shown in the tool schema."""

    directive_schema: JSON
    """JSON Schema describing the shape of directives for this tool."""

    supports_microcompaction: bool
    """Whether this tool opts into per-result microcompaction."""

    def summary(self, msg: Message) -> str:
        """Return a human-readable summary of a pending invocation."""
        ...

    def summary_result(self, result: Message) -> str | None:
        """Return a human-readable summary of a completed invocation."""
        ...

    def prompt(self) -> str | None:
        """Build the system prompt section for this tool.

        Return ``None`` to signal "no change from last request" -- avoids
        cache invalidation when content is stable.
        Return ``""`` for an explicitly empty section.
        """


@runtime_checkable
class BatchingTool(_BaseTool, Protocol):
    """Batch tool -- single await, single result."""

    async def run(self, msg: Message) -> Message:
        """Execute the tool; return ``descriptor="text/x-error"`` on failure."""
        ...


@runtime_checkable
class StreamingTool(_BaseTool, Protocol):
    """Streaming tool -- yields intermediate events as an async generator.

    The final yielded ``Message`` is the tool result; all preceding
    yields are intermediate events forwarded to the REPL event queue.

    Concrete tools set ``streaming: Literal[True] = True`` and implement
    ``run`` as ``async def run(...) -> AsyncGenerator[Message, None]``.

    Note: the protocol declares ``run`` as ``def``, not ``async def``.
    In Python, calling an async generator function returns the
    ``AsyncGenerator`` directly -- no ``await`` needed. An ``async def``
    protocol signature would mean ``await tool.run(msg)`` → generator,
    but concrete async generators don't work that way: ``tool.run(msg)``
    already *is* the generator. The non-async signature matches reality
    and gives the natural callsite: ``async for event in tool.run(msg)``.
    """

    streaming: Literal[True]
    """Discriminator: always ``True`` for streaming tools."""

    def run(self, msg: Message) -> AsyncGenerator[Message, None]:
        """Execute the tool, yielding events. Last yield is the result."""
        ...


Tool = BatchingTool | StreamingTool


# Lives here (not in a helper module) solely for TypeIs narrowing:
# callers get automatic type coercion in both branches without casts.
def is_streaming(tool: Tool) -> TypeIs[StreamingTool]:
    """Type-narrowing check for streaming tools."""
    return getattr(tool, "streaming", False)


@dataclass(frozen=True, slots=True, kw_only=True)
class Pricing:
    """Per-million-token prices in USD."""

    request: float = 0.0
    response: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCount:
    """Immutable 4-tuple of token counts returned by a model request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

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
    """Full conversation sent to an LLM backend (internal)."""

    messages: list[Message]
    """Conversation history as a flat list of Messages."""

    system: str | None = None
    """System prompt; ``None`` omits it from the request."""

    tools: list[Tool] | None = None
    """Tools advertised to the model; ``None`` sends no tool schema."""

    max_response_tokens: int | None = None
    """Max output tokens. ``None`` uses the model's default."""

    temperature: float = 1.0
    """Sampling temperature."""

    thinking: str | None = None
    """Extended-thinking mode; ``None`` disables thinking."""

    effort: str | None = None
    """Effort hint (Anthropic: "low"/"medium"/"high"/"xhigh"/"max";
    Qwen3: any non-"none" value enables hybrid thinking). ``None`` omits
    the field so the API applies its own default. The API is the source
    of truth for accepted values and per-model support."""

    cache_ttl: str = "5m"
    """Prompt-cache TTL for the request's cache breakpoint. ``"5m"`` is
    Anthropic's default (1.25x input rate to write); ``"1h"`` is the
    extended-cache option (2x input rate to write). Reads price the same
    either way. Providers without prompt caching ignore this field."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponse:
    """What comes back from an LLM backend (internal)."""

    content: Message
    """Model response as a ``multipart/x-model-message`` Message."""

    tokens: TokenCount = field(default_factory=TokenCount)
    """Token counts for this request."""

    stop_reason: str = "model_finished"
    """Why generation stopped (normalized to canonical vocabulary)."""

    stop_sequence: str | None = None
    """Matched stop sequence, if any."""

    message_id: str = ""
    """Provider-assigned message ID (Anthropic ``message.id``)."""

    request_id: str = ""
    """HTTP-level request ID from the response headers."""

    input_cost: float = 0.0
    """USD cost of the prompt tokens."""

    output_cost: float = 0.0
    """USD cost of the generated tokens."""

    total_cost: float = 0.0
    """Total USD cost of the request."""


@runtime_checkable
class Model(Protocol):
    """Internal interface for LLM backends."""

    @property
    def max_request_tokens(self) -> int:
        """Maximum input tokens the model accepts."""
        ...

    @property
    def model_id(self) -> str:
        """Provider-specific model identifier, e.g. ``"claude-opus-4-7+1m"``."""
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
        """Whether the provider manages context window overflow internally."""
        ...

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether the provider retries internally on transient failures."""
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
          text: Input text to estimate.

        Returns:
          count: Estimated token count.

        """
        ...

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate input token count for an image.

        Args:
          data: Raw image bytes.

        Returns:
          count: Estimated token count.

        """
        ...

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        ...

    async def buffer(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Send a request and return the complete response.

        Args:
          request: Full model request.

        Returns:
          response: Complete model response.

        """
        ...

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Send a request with optional streaming callbacks.

        Args:
          request: Full model request.
          on_text: Callback invoked with each text chunk during streaming.
          on_thinking: Callback invoked with each thinking chunk during
              streaming. Providers that don't expose thinking deltas
              (or models without thinking enabled) ignore this argument
              and the caller falls back to extracting thinking content
              from the final response.

        Returns:
          response: Complete model response.

        """
        ...

    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error is a context-window overflow.

        Args:
          error: Exception raised during a model request.

        Returns:
          is_overflow: True if the error is a context-window overflow.

        """
        ...

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Check whether an error is a provider-specific transient retryable.

        Use for SDK quirks the cross-provider classifier in
        ``retry.py`` can't see (e.g. an Anthropic error with no HTTP
        status that still carries ``body["type"] == "server_error"``).
        Status-code / transport-error classification stays in
        ``retry.py``; this hook only adds the per-provider edge cases.

        Args:
          error: Exception raised during a model request.

        Returns:
          retryable: True if the provider considers this transient.

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
          model_id: Provider-specific model ID. ``None`` uses the default.
          max_request_tokens: Override the model's default max input tokens.

        Returns:
          model: Configured model backend.

        """
        ...

    def utility_model(self) -> Model:
        """Return the cheapest/fastest model for utility tasks.

        Returns:
          model: Utility model backend.

        """
        ...


@runtime_checkable
class Compactor(Protocol):
    """Interface for conversation compaction strategies."""

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Determine whether the conversation should be compacted now.

        Args:
          input_tokens: Current input token count.
          max_request_tokens: Maximum input tokens the model accepts.
          max_response_tokens: Maximum output tokens reserved for response.

        Returns:
          needed: True if compaction is needed.

        """
        ...

    async def compact(
        self,
        messages: list[Message],
        model: Model,
        transcript_path: Path | None = None,
        direction: Literal["from", "up_to"] = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[Message]:
        """Summarize the conversation into a compact message list.

        Args:
          messages: Full conversation history.
          model: Model backend for summarization.
          transcript_path: Path to persist the transcript.
          direction: Compaction direction -- ``"from"`` or ``"up_to"``.
          keep_recent: Number of recent messages to preserve verbatim.
          custom_instructions: Extra instructions for the summarizer.
          summary_pointers: Key-value pairs to include in the summary.

        Returns:
          compacted: Compacted message list.

        """
        ...

    def maintain(
        self,
        messages: list[Message],
        tools: dict[str, Tool],
        **kwargs: object,
    ) -> None:
        """Perform between-request context maintenance.

        Args:
          messages: Current conversation messages (mutated in place).
          tools: Active tool registry.
          **kwargs: Additional provider-specific options.

        """


@runtime_checkable
class CompactRestorable(Protocol):
    """Optional protocol for tools that restore state after compaction."""

    async def post_compact_restore(
        self,
        messages: list[Message],
        tool_state: ToolState,
        *,
        budget_chars: int = 100_000,
    ) -> None:
        """Re-inject tool-specific context into messages after compaction.

        Args:
          messages: Post-compaction messages (mutated in place).
          tool_state: Tool state to restore from.
          budget_chars: Maximum characters to re-inject.

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
    """Maximum output tokens reserved for the model's response."""

    chars_per_token: int = 4
    """Approximate characters per token for budget calculations."""

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
                f"buffer_tokens ({self.buffer_tokens}) must be < max_request_tokens"
                f" ({self.max_request_tokens})"
            )
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {self.chars_per_token}")

    @classmethod
    def from_model(cls, model: Model) -> ContextBudget:
        """Derive proportional defaults from a model's limits.

        Args:
          model: Model backend whose limits define the budget.

        Returns:
          budget: Context budget with proportional defaults.

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


# -- Agent observer events (v3) ---------------------------------------------
#
# The agent publishes these to its observer list (synchronous fan-out).
# 13 closed-set event types -- adding requires justification per
# docs/private/agent_refactor.md §18.3.


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class UserBarEvent:
    """The agent received a user message; render as a full-width bar."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class TextChunkEvent:
    """A streaming text fragment from the model."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ThinkingEvent:
    """A thinking block from the model."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ToolLabelEvent:
    """Multi-line label for a tool invocation about to execute."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ToolResultEvent:
    """A completed tool result; ``msg`` is the ``multipart/x-tool-result``."""

    msg: Message


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class StreamEndEvent:
    """Streaming text for one model call has ended; flush buffered markdown."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ErrorEvent:
    """A user-visible error; render as a red dim line."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class InterruptedEvent:
    """The current step was cancelled; render the dim ``[interrupted]`` line."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class StatusUpdateEvent:
    """The agent's status was set; render via terminal title."""

    text: str


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class TurnCompleteEvent:
    """The current turn finished with no tool calls (model is idle)."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ChildEvent:
    """An event from a child agent, wrapped with its label."""

    label: str
    inner: Event


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ChildDoneEvent:
    """A child agent completed its turn; carries totals for the gutter."""

    label: str
    elapsed: float
    tokens: int
    cost: float


type Event = (
    UserBarEvent
    | TextChunkEvent
    | ThinkingEvent
    | ToolLabelEvent
    | ToolResultEvent
    | StreamEndEvent
    | ErrorEvent
    | InterruptedEvent
    | StatusUpdateEvent
    | TurnCompleteEvent
    | ChildEvent
    | ChildDoneEvent
)
