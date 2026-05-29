"""Runtime event vocabulary.

This module is the root type module for events that happen during an
agent session. It defines provider-visible model context events and
dispatch-only runtime events. It must stay leaf-only: do not import
tape/model/tool modules here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import dataclasses
import itertools
import time


_id_counter: itertools.count[int] = itertools.count()


def _empty_headers() -> dict[str, str]:
    return {}


__all__ = [
    "AgentIdle",
    "AgentSendDeferredMessage",
    "AgentSendMessage",
    "AgentSendQueuedMessage",
    "AssistantMessage",
    "BudgetReset",
    "BytesMessage",
    "ChildDoneEvent",
    "ChildEvent",
    "Clear",
    "CohortComplete",
    "CohortStarted",
    "Compact",
    "CompactComplete",
    "CompactFailed",
    "CompactStarted",
    "Detach",
    "DetachedResult",
    "Halt",
    "Kill",
    "ModelCallStarted",
    "ModelContextEvent",
    "ModelIdle",
    "ModelResponseCancelled",
    "ModelResponseComplete",
    "ModelResponseError",
    "ModelResponsePartial",
    "ModelResponseThinking",
    "ModelServiceSuspended",
    "ModelSwitch",
    "ModelSwitchRejected",
    "Quit",
    "Recompact",
    "RuntimeEvent",
    "SaveSession",
    "ServiceErrorSnapshot",
    "SessionMessage",
    "StatusChanged",
    "ToolCall",
    "ToolLabel",
    "ToolResult",
    "ToolResultPartial",
    "Undetach",
    "UserDeferredMessage",
    "UserMessage",
    "UserQueuedMessage",
    "reset_id_counter",
]


def reset_id_counter(start: int) -> None:
    """Reset the ``SessionMessage`` id counter."""
    global _id_counter  # noqa: PLW0603 -- module-level counter requires global statement
    _id_counter = itertools.count(start)


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class BytesMessage:
    """Binary payload (image, PDF)."""

    data: bytes
    """Raw bytes of the payload."""

    descriptor: str
    """MIME-style content type (e.g. ``image/png``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionMessage:
    """Common fields for provider-visible session messages."""

    id: int = dataclasses.field(default_factory=lambda: next(_id_counter))
    """Monotonically increasing per-session message id."""

    parent_id: int = -1
    """Id of the message this one responds to, or ``-1``."""

    timestamp: float = dataclasses.field(default_factory=time.time)
    """Unix wall-clock seconds when the message was created."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    """Provider-assigned call id (e.g. ``toolu_01...``)."""

    name: str
    """Tool name the model wants to invoke."""

    args: Mapping[str, object]
    """Parsed directive arguments."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage(SessionMessage):
    """Human-authored user-role text the model should see."""

    text: str
    """Plain-text content."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads sent alongside the text."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendMessage(SessionMessage):
    """Agent-authored user-role text the model should see."""

    source: str
    """Agent label that produced the message."""

    text: str
    """Plain-text content."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads sent alongside the text."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage(SessionMessage):
    """Model response: text and/or tool calls."""

    text: str = ""
    """User-visible response text."""

    thinking_blocks: tuple[Mapping[str, object], ...] = ()
    """Provider thinking blocks (opaque dicts)."""

    tool_calls: tuple[ToolCall, ...] = ()
    """Tool invocations requested by the model."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(SessionMessage):
    """Result of one tool invocation."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    content: str
    """Result text shown to the model."""

    is_error: bool = False
    """True when the tool raised or signalled failure."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads produced by the tool."""

    diff: str = ""
    """Unified-diff fragment for renderers (e.g. Edit/Write)."""

    diff_file_path: str = ""
    """Absolute path the ``diff`` applies to."""

    hint: str = ""
    """Optional follow-up nudge surfaced to the model."""

    summary: str = ""
    """Optional short post-execution receipt line."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CompactStarted:
    """Compaction task has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactComplete:
    """Compaction task finished."""

    records: tuple[object, ...] = ()
    """Runtime-only overrides appended by the compactor."""

    token_before: int = 0
    """Token count before compaction."""

    token_after: int = 0
    """Token count after compaction."""

    payload_entries: int = 0
    """Number of provider entries in the compacted payload."""

    fallback_reason: str = ""
    """Why fallback history was used instead of a summary."""

    preserved_tail_count: int = 0
    """Number of tail entries preserved verbatim in fallback mode."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactFailed:
    """Compaction task raised; runtime keeps the tape as-is."""

    exception: BaseException
    """The raised exception surfaced to observers."""

    tape_len: int
    """Tape length captured before compaction."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserQueuedMessage:
    """Human-authored queued input that preempts at the next safe boundary."""

    text: str
    """Plain-text content to commit as a ``UserMessage``."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads to merge alongside ``text``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UserDeferredMessage:
    """Human-authored input that waits for full model idle."""

    text: str
    """Plain-text content to commit as a ``UserMessage``."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads to merge alongside ``text``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendQueuedMessage:
    """Agent-authored queued input that preempts at the next safe boundary."""

    source: str
    """Agent label that produced the message."""

    text: str
    """Plain-text content to commit as an ``AgentSendMessage``."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads to merge alongside ``text``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendDeferredMessage:
    """Agent-authored input that waits for full model idle."""

    source: str
    """Agent label that produced the message."""

    text: str
    """Plain-text content to commit as an ``AgentSendMessage``."""

    attachments: tuple[BytesMessage, ...] = ()
    """Image/PDF payloads to merge alongside ``text``."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Quit:
    """Shut down the agent."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Halt:
    """Cancel model call, wait for user. Tools keep running."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Clear:
    """Detach tools, wipe history, wait for user."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Kill:
    """Cancel one or all tool tasks."""

    call_id: str | None = None
    """Specific call to cancel, or ``None`` to cancel all."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Detach:
    """Stub one or all tools, let them finish in background."""

    call_id: str | None = None
    """Specific call to detach, or ``None`` for all."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Undetach:
    """Re-gate model on a detached tool's completion."""

    call_id: str | None = None
    """Specific call to re-gate on, or ``None`` for all."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitch:
    """Queue a model swap; the runtime applies it once safe."""

    apply: Callable[[], None]
    """Closure that performs the swap."""

    label: str = ""
    """Optional human-readable label shown to renderers."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitchRejected:
    """Model swap failed; runtime keeps the existing model."""

    exception: BaseException
    """The raised exception surfaced to observers."""

    label: str = ""
    """Human-readable switch label, when available."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetReset:
    """Agent ``ContextBudget`` was reset to fit the new model."""

    model_id: str
    """Provider-specific model id the budget was sized for."""

    prior_max_request_tokens: int
    """Pre-reset input cap."""

    prior_max_response_tokens: int
    """Pre-reset output cap."""

    new_max_request_tokens: int
    """Post-reset input cap."""

    new_max_response_tokens: int
    """Post-reset output cap."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelCallStarted:
    """Model streaming call has been spawned."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponsePartial:
    """Streaming text chunk from the model."""

    text: str
    """Newly arrived text chunk."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseThinking:
    """Streaming thinking chunk from the model."""

    text: str
    """Newly arrived thinking chunk."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseComplete:
    """Model finished streaming."""

    message: AssistantMessage
    """Final assembled ``AssistantMessage``."""

    generation: int = -1
    """Runtime model-call generation that produced this response."""

    input_tokens: int = 0
    """Input token count reported by the provider."""

    output_tokens: int = 0
    """Output token count reported by the provider."""

    cache_creation_tokens: int = 0
    """Tokens spent creating cache breakpoints."""

    cache_read_tokens: int = 0
    """Tokens served from cache."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseCancelled:
    """Model call was cancelled mid-stream."""

    output_chars_estimate: int = 0
    """Approximate chars streamed before cancel."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelResponseError:
    """Unrecoverable failure (creds expired, retries exhausted)."""

    exception: BaseException
    """The raised exception surfaced to observers."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceErrorSnapshot:
    """Serializable provider error details safe for session logs."""

    type_name: str
    """Provider exception class name."""

    message: str
    """Human-readable exception message."""

    status: int | None = None
    """HTTP status code, when known."""

    headers: Mapping[str, str] = dataclasses.field(default_factory=_empty_headers)
    """Allowlisted diagnostic response headers."""

    body: str = ""
    """Truncated response body excerpt."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelServiceSuspended:
    """External model service temporarily rejected work; retry is scheduled."""

    provider: str
    """Provider class/key backing the model call."""

    auth: str
    """Provider auth flavor."""

    account: str | None
    """Provider account slot, or ``None`` for the provider default."""

    model_id: str
    """Concrete provider model id."""

    retry_at: float
    """Unix wall-clock seconds when the retry may resume."""

    delay_sec: float
    """Sleep duration selected for this retry."""

    server_supplied: bool
    """True when provider response supplied the retry delay."""

    error: ServiceErrorSnapshot
    """Sanitized provider error snapshot."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelIdle:
    """Model finished with no tool calls."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class AgentIdle:
    """Agent has fully drained: about to block on inbox with no work."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CohortStarted:
    """Tool cohort has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultPartial:
    """Streaming chunk from a tool."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Newly arrived output chunk."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachedResult:
    """A previously-detached tool completed."""

    result: ToolResult
    """Full result from the completed tool."""

    @property
    def call_id(self) -> str:
        return self.result.call_id

    @property
    def content(self) -> str:
        return self.result.content

    @property
    def is_error(self) -> bool:
        return self.result.is_error


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CohortComplete:
    """All tool results for the current cohort have arrived."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Compact:
    """Trigger context compaction."""

    args: str = ""
    """Free-form compaction instructions for the compactor."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Recompact:
    """Alias for ``/compact``; trigger context compaction."""

    args: str = ""
    """Free-form compaction instructions for the compactor."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class SaveSession:
    """Signals observers to persist session state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusChanged:
    """Agent ``status`` field was updated."""

    text: str
    """New status string."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolLabel:
    """Pre-execution label for a tool call."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Short label rendered above the call."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvent:
    """Wrapped event from a child agent."""

    label: str
    """Child agent's display label."""

    inner: RuntimeEvent
    """The forwarded child event."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildDoneEvent:
    """Child agent completed; carries totals for the gutter."""

    label: str
    """Child agent's display label."""

    elapsed: float
    """Wall-clock seconds the child ran."""

    tokens: int
    """Total tokens (input + output) the child consumed."""

    cost: float
    """Total USD cost attributable to the child."""


type ModelContextEvent = UserMessage | AgentSendMessage | AssistantMessage | ToolResult


type RuntimeEvent = (
    Quit
    | Halt
    | Clear
    | Kill
    | Detach
    | Undetach
    | UserMessage
    | AgentSendMessage
    | UserQueuedMessage
    | UserDeferredMessage
    | AgentSendQueuedMessage
    | AgentSendDeferredMessage
    | ModelSwitch
    | ModelSwitchRejected
    | BudgetReset
    | ModelCallStarted
    | ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelServiceSuspended
    | ModelIdle
    | AgentIdle
    | CohortStarted
    | ToolResultPartial
    | ToolResult
    | DetachedResult
    | CohortComplete
    | Compact
    | Recompact
    | CompactStarted
    | CompactComplete
    | CompactFailed
    | SaveSession
    | StatusChanged
    | ToolLabel
    | ChildEvent
    | ChildDoneEvent
)
