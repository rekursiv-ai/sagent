"""Runtime event vocabulary.

The history is session records; one model turn is the ``Response`` it came in.
Everything else here is an event the runtime takes from its inbox or publishes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import asyncio

from sagent.model.exceptions import RetryableError
from sagent.model.types import Response
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    Attachment,
    SystemMessage,
    ToolCall,
    TurnContext,
    UserMessage,
)


__all__ = [
    "AgentIdle",
    "AgentSendDeferredMessage",
    "AgentSendQueuedMessage",
    "Charged",
    "ChildDoneEvent",
    "ChildEvent",
    "Clear",
    "ClearComplete",
    "CohortComplete",
    "CohortStarted",
    "Compact",
    "CompactComplete",
    "CompactFailed",
    "CompactStarted",
    "CompactionSplice",
    "Detach",
    "DetachedResult",
    "Halt",
    "Job",
    "Kill",
    "LazyEvent",
    "Markers",
    "ModelCallStarted",
    "ModelCallSuspended",
    "ModelContextEvent",
    "ModelIdle",
    "ModelResponseCancelled",
    "ModelResponseComplete",
    "ModelResponseError",
    "ModelResponsePartial",
    "ModelResponseThinking",
    "Nudge",
    "Quit",
    "Retry",
    "RuntimeEvent",
    "SaveSession",
    "Scheduled",
    "StatusChanged",
    "UserDeferredMessage",
    "UserQueuedMessage",
    "labeled_agent_send_text",
    "wire_role",
]


def labeled_agent_send_text(entry: AgentToAgentMessage) -> str:
    """Return ``entry``'s text prefixed with its ``[from <sender>]: `` label.

    Args:
      entry: A message from another agent.

    Returns:
      text: The text, labeled once; a labeled text is returned as it is.

    """
    prefix = f"[from {entry.sender}]: "
    text = entry.content or ""
    return text if text.startswith(prefix) else f"{prefix}{text}"


# The state of a stand-in result is ``ToolResult.stub``, never its text, so these
# words may change and a tool's own output that reads like one is never taken for it.
@dataclass(frozen=True, slots=True, kw_only=True)
class Markers:
    """What the runtime writes into the context for the calls it answers itself."""

    detached: str = (
        "[detached: tool still running; real result arrives in a later message]"
    )
    """A detached call's result; the real one arrives later, as a ``delivery`` turn."""

    cancelled: str = "[cancelled: tool killed before completion; no result will follow]"
    """A killed call's result."""

    running: str = "[Running in background: "
    """How a backgrounded call's result starts: ``<running><tool> as job <id>]``."""

    delivery: str = "BackgroundTask"
    """The tool a detached call's real result is delivered under, as a ``foreground``
    of the call: what the model would call to wait for it, so a copy is a real call."""

    arrived: str = "DetachedArrived"
    """What deliveries were once made under; none runs it, as journals may hold it."""

    arrival: str = ":detached"
    """What the delivering call's id adds to the detached call's."""

    mimic: str = "DetachedArrived:mimic:"
    """How a call the model made under a delivery's id is renamed: ``<mimic><n>``."""

    note: str = (
        "Note on detached calls: a tool call detached to the background (it ran"
        " long, or a message arrived while it ran) is answered at once with a"
        " stub. When it finishes, its result is delivered to you as a"
        " `BackgroundTask` foreground call of that job, made for you. Call"
        " `BackgroundTask` foreground yourself only to wait for a job."
    )
    """What the system prompt tells the model about deliveries."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactStarted:
    """Compaction task has been spawned."""


class CompactionSplice(Protocol):
    """What a compaction returns, as ``CompactComplete.from_override`` reads it."""

    @property
    def payload(self) -> tuple[ModelContextEvent, ...]:
        """The entries the splice puts in the context."""
        ...

    @property
    def token_before(self) -> int:
        """Tokens of what the splice replaces."""
        ...

    @property
    def token_after(self) -> int:
        """Tokens of what it puts in their place."""
        ...

    @property
    def fallback_reason(self) -> str:
        """Why the summary fell back to keeping history, if it did."""
        ...

    @property
    def preserved_tail_count(self) -> int:
        """How many recent entries the fallback kept verbatim."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactComplete:
    """Compaction task finished."""

    token_before: int = 0
    token_after: int = 0
    payload_entries: int = 0
    fallback_reason: str = ""
    """Why the summary fell back to keeping history; ``""`` when it did not."""

    preserved_tail_count: int = 0
    """Recent entries kept verbatim beside the summary."""

    @classmethod
    def from_override(cls, override: CompactionSplice) -> CompactComplete:
        """Return the completion event for a compactor's splice.

        Args:
          override: The splice the compactor returned.

        Returns:
          completion: Every field read off ``override``.

        """
        return cls(
            token_before=override.token_before,
            token_after=override.token_after,
            payload_entries=len(override.payload),
            fallback_reason=override.fallback_reason,
            preserved_tail_count=override.preserved_tail_count,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactFailed:
    """Compaction raised; the tape is as it was."""

    exception: BaseException
    tape_len: int


@dataclass(frozen=True, slots=True, kw_only=True)
class UserQueuedMessage:
    """The user's input, held until the model is done with the current turn."""

    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class UserDeferredMessage:
    """The user's input, held until queued input has been answered too."""

    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendQueuedMessage:
    """An agent's input, held until the model is done with the current turn."""

    sender: str
    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendDeferredMessage:
    """An agent's input, held until queued input has been answered too."""

    sender: str
    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Quit:
    """Shut down the agent."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Halt:
    """Cancel model call, wait for user. Tools keep running."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Clear:
    """Cancel the model and every tool, wipe history, wait for the user."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearComplete:
    """Published after the runtime finishes processing a ``Clear``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Kill:
    """Cancel one or all tool tasks."""

    call_id: str | None = None
    """The call to cancel; ``None`` for all."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Detach:
    """A call the model waits on ran past the runtime's limit: answer it, let it run on."""

    call_id: str
    """A call of the group to detach."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCallStarted:
    """Model streaming call has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponsePartial:
    """Streaming text chunk from the model."""

    text: str
    record: int = 0
    """The index in the reply's records of the record the text joins."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseThinking:
    """Streaming thinking chunk from the model."""

    text: str
    record: int = 0
    """The index in the reply's records of the record the reasoning joins."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseComplete:
    """Model finished streaming."""

    response: Response
    generation: int = -1
    """The model call that answered; a stale one's reply is dropped."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseCancelled:
    """Model call was cancelled mid-stream."""

    output_chars_estimate: int = 0
    """Characters streamed before the cancel."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseError:
    """A model call failed past retrying, such as on expired credentials.

    The turn waits for the user, a new model, or a ``Retry``.
    """

    exception: BaseException


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCallSuspended:
    """A model call failed in a way that may pass; the turn waits, then calls again.

    It calls again at ``until``, or sooner on the user's message, a new model, or
    a ``Retry``. Unlike ``ModelResponseError``, the turn has not ended.
    """

    exception: RetryableError
    until: float
    """When the next call goes, in seconds since the epoch."""

    attempt: int
    """Failures in a row, this one counted; a ``Retry`` starts them again."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Retry:
    """Call the model now for the turn a failure stopped; nothing when none did.

    What may get past the failure sends it: a new model, account, or sign-in.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelIdle:
    """Model finished with no tool calls."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentIdle:
    """Agent has fully drained: about to block on inbox with no work."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CohortStarted:
    """Tool cohort has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachedResult:
    """A previously-detached tool completed."""

    result: AnyToolResult

    @property
    def call_id(self) -> str:
        """Return the result's call ID."""
        return self.result.call_id


@dataclass(frozen=True, slots=True, kw_only=True)
class LazyEvent:
    """A result held until the next event that drives a model round: none of its own."""

    payload: AnyToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class CohortComplete:
    """All tool results for the current cohort have arrived."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Compact:
    """Trigger context compaction."""

    args: str = ""
    """What the summary must keep."""

    trigger: Literal["auto", "manual"] = "manual"
    """Whether the runtime started it, or someone asked for it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SaveSession:
    """Signals observers to persist session state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusChanged:
    """Agent ``status`` field was updated."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvent:
    """An event of a child agent, under its label."""

    label: str
    inner: RuntimeEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildDoneEvent:
    """A child agent finished: how long it ran, and what it used."""

    label: str
    elapsed: float
    tokens: int
    """Every token the child's model calls read and wrote."""

    cost: float
    """What the child's model calls cost, in dollars."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Charged:
    """Spend counted against the agent that its own replies did not bring."""

    usd: float
    by: str
    """Who spent it: a child's label, or this agent's own for a compaction."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Scheduled:
    """Work to run at the runtime's next safe point, between turns."""

    work: Callable[[], Awaitable[None]]
    """What to run once nothing is streaming, compacting, or gating the model."""

    done: asyncio.Future[None]
    """Resolved when ``work`` returns, or set to what it raised."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Nudge:
    """A harness note that wakes an idle agent: kept as a ``nudge`` SystemMessage."""

    content: str
    """What the agent is told, such as the replies it owes."""


type ModelContextEvent = (
    UserMessage
    | AgentToAgentMessage
    | Response
    | AnyToolResult
    | SystemMessage
    | TurnContext
)
"""One entry of the conversation history; a ``SystemMessage`` or ``TurnContext`` is no one's turn."""


def wire_role(entry: ModelContextEvent) -> str | None:
    """Return the provider-wire role for ``entry``.

    Args:
      entry: One entry of the history.

    Returns:
      role: ``"user"`` for a user or agent message, ``"assistant"`` for a
          model turn, ``None`` for a tool result or a harness note.

    """
    if isinstance(entry, (UserMessage, AgentToAgentMessage)):
        return "user"
    if isinstance(entry, Response):
        return "assistant"
    return None


type RuntimeEvent = (
    Quit
    | Halt
    | Clear
    | ClearComplete
    | Kill
    | Detach
    | UserMessage
    | AgentToAgentMessage
    | Nudge
    | UserQueuedMessage
    | UserDeferredMessage
    | AgentSendQueuedMessage
    | AgentSendDeferredMessage
    | ModelCallStarted
    | ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelCallSuspended
    | Retry
    | ModelIdle
    | AgentIdle
    | CohortStarted
    | AnyToolResult
    | DetachedResult
    | LazyEvent
    | CohortComplete
    | Compact
    | CompactStarted
    | CompactComplete
    | CompactFailed
    | SaveSession
    | StatusChanged
    | ChildEvent
    | ChildDoneEvent
    | Charged
    | Scheduled
)


class Job(Protocol):
    """A tool call the runtime started, in the foreground or the background."""

    @property
    def call(self) -> ToolCall:
        """The call it runs: the tool, its id, its arguments."""
        ...

    @property
    def started(self) -> float:
        """When it started, in seconds since the epoch."""
        ...

    @property
    def state(
        self,
    ) -> Literal[
        "running",
        "sleeping",
        "detached",
    ]:
        """Where it is: the model waits on ``running``, not on ``sleeping`` or ``detached``."""
        ...

    def cancel(self) -> None:
        """Stop it; the runtime records it as cancelled."""
        ...
