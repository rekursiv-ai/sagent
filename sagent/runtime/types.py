"""Runtime event vocabulary.

The history is session records; one model turn is the ``Response`` it came in.
Everything else here is an event the runtime takes from its inbox or publishes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import asyncio
import math

from sagent.model.exceptions import RateLimitedError, RetryableError
from sagent.model.types import Response
from sagent.session.types import (
    AgentToAgentMessage,
    Attachment,
    SystemMessage,
    ToolCall,
    ToolResult,
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
    "DetachSchedule",
    "DetachedResult",
    "Halt",
    "Job",
    "Kill",
    "LazyEvent",
    "LoginSwitched",
    "LoginsSpent",
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
    "Quit",
    "Retry",
    "RuntimeEvent",
    "SaveSession",
    "Scheduled",
    "Skipped",
    "StatusChanged",
    "UserDeferredMessage",
    "UserQueuedMessage",
    "Wake",
    "WakeSchedule",
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


# Code tells a pending or cancelled call's record by ``ToolResult.state``, never by
# this text, so the words may change, and a tool's output that happens to match
# them is never mistaken for a placeholder.
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
    """Answer a call the model waits on, and its group, with a stub; let them run on."""

    call_id: str
    """A call of the group to detach."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeSchedule:
    """When a long call wakes the model: early, then less often, as a person glances.

    The gap to the next wake is ``max(first_sec, growth * elapsed)``: a wake so
    timed fires only when the call wrote something since the model last looked.
    ``max_gap_sec`` and the prompt cache's deadline cap the gap; a capped wake
    fires regardless, so a quiet call is still seen and the cache stays warm.

    Raises:
      ValueError: A knob that would never wake, or wake without end.

    """

    first_sec: float = 15.0
    """The first wake, and the shortest gap."""

    growth: float = 1.0
    """The next gap as a multiple of the time the call has run: ``1`` doubles the
    time each wake, ``0`` keeps the gap at ``first_sec``."""

    max_gap_sec: float = 300.0
    """The longest the model goes without a look."""

    def __post_init__(self) -> None:
        for name, value in (
            ("first_sec", self.first_sec),
            ("max_gap_sec", self.max_gap_sec),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a number of seconds above 0, got {value}.",
                )
        if not (math.isfinite(self.growth) and self.growth >= 0):
            raise ValueError(f"growth must be a number at least 0, got {self.growth}.")

    def gap(self, elapsed_sec: float, *, cache_left_sec: float) -> tuple[float, bool]:
        """Return the seconds to the next wake, and whether it fires without new output.

        Args:
          elapsed_sec: How long the call has run.
          cache_left_sec: How long the prompt cache stays warm, less a margin;
            ``math.inf`` when it does not bound the wait.

        Returns:
          gap: Seconds until the wake.
          forced: Whether a cap set it, so it fires whatever the call wrote.

        """
        paced = max(self.first_sec, self.growth * elapsed_sec)
        cap = min(cache_left_sec, self.max_gap_sec)
        return min(paced, cap), cap <= paced

    def text(self) -> str:
        """Return the knobs as a setting takes them: ``15s 1 300s``."""
        return f"{self.first_sec:g}s {self.growth:g} {self.max_gap_sec:g}s"


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachSchedule:
    """How each tool's long calls wake the model; ``None`` never wakes, nor detaches."""

    default: WakeSchedule | None = WakeSchedule()
    """Each tool's, but those ``tools`` names."""

    tools: Mapping[str, WakeSchedule | None] = field(
        default_factory=dict[str, "WakeSchedule | None"],
    )
    """A tool's own, by name, over ``default``."""

    def of(self, tool: str) -> WakeSchedule | None:
        """Return how the tool named ``tool`` wakes the model."""
        return self.tools.get(tool, self.default)


@dataclass(frozen=True, slots=True, kw_only=True)
class Wake:
    """A long call's wake time came: show the model what it wrote since it last looked."""

    call_id: str
    forced: bool
    """Whether a cap set the time, so it fires whatever the call wrote."""


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

    # The rest are a backoff's waits, minutes at most: telling each would wake
    # whoever is told every few seconds of an outage.
    @property
    def opens_a_quota_wait(self) -> bool:
        """Whether this is the first wait of a spent quota, which may last days."""
        return isinstance(self.exception, RateLimitedError) and self.attempt == 1


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

    result: ToolResult

    @property
    def call_id(self) -> str:
        """Return the result's call ID."""
        return self.result.call_id


@dataclass(frozen=True, slots=True, kw_only=True)
class LazyEvent:
    """A result held until the next event that drives a model round: none of its own."""

    payload: ToolResult


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
class LoginSwitched:
    """The model moved to another login of its provider: off a spent one, or back."""

    provider: str
    was: str
    """The login left: ``default``, or a named account."""

    now: str
    """The login the next call signs in as."""

    window: str = ""
    """The window ``was`` spent, as ``5-hour``, or ``signed out``; ``""`` when ``was``
    was not left for a failure: back after a reset, or as the order says."""

    until: float = 0.0
    """When that window resets, in seconds since the epoch; ``0`` when unknown."""

    first: str = ""
    """The first login in the order: ``now`` itself, unless ``now`` stands in for it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LoginsSpent:
    """Every login of the provider has spent the window the model needs."""

    provider: str
    resets: tuple[tuple[str, float], ...]
    """Each login and when it resets, soonest first."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Skipped:
    """A directive or skill file left unread, such as for frontmatter that will not parse."""

    text: str
    """Which file, and why; what the user is shown."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Scheduled:
    """Work to run at the runtime's next safe point, between turns."""

    work: Callable[[], Awaitable[None]]
    """What to run once nothing is streaming, compacting, or gating the model."""

    done: asyncio.Future[None]
    """Resolved when ``work`` returns, or set to what it raised."""


type ModelContextEvent = (
    UserMessage
    | AgentToAgentMessage
    | Response
    | ToolResult
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
    | Wake
    | UserMessage
    | AgentToAgentMessage
    | SystemMessage
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
    | LoginSwitched
    | LoginsSpent
    | Retry
    | ModelIdle
    | AgentIdle
    | CohortStarted
    | ToolResult
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
    | Skipped
)


class Job(Protocol):
    """A tool call the runtime started, in the foreground or the background."""

    @property
    def call(self) -> ToolCall:
        """The call it runs: the tool, its id, its arguments."""
        ...

    @property
    def started(self) -> float:
        """When it was asked for, in seconds since the epoch."""
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
