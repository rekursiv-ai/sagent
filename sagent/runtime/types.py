"""What the runtime is asked, and what it is doing.

A ``RuntimeRequest`` goes into the inbox and asks the runtime to act; each ends
once, told as its ``...Completed(outcome)``. A ``RuntimeState`` is what the runtime
publishes. A
request's ``sender`` is who asks; ``None`` is the session's own user.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Protocol, override

import math

from sagent.model.cost import TokenCost
from sagent.model.exceptions import RateLimitedError, RetryableError
from sagent.model.types import Response
from sagent.session.types import (
    AgentIdle,
    AgentMessage,
    ContextCompaction,
    ModelIdle,
    Outcome,
    SessionContext,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


__all__ = [
    "AgentMessageCompleted",
    "ChildEvent",
    "CompactionSplice",
    "Job",
    "LoginSwitched",
    "ModelContextEvent",
    "ModelHalt",
    "ModelHaltCompleted",
    "ModelMessage",
    "ModelMessageCompleted",
    "ModelResponseCompleted",
    "ModelResponseStarted",
    "ModelResponseSuspended",
    "ModelResponseTextStreamed",
    "ModelResponseThinkingStreamed",
    "ModelRetry",
    "ModelRetryCompleted",
    "NextWake",
    "RuntimeEvent",
    "RuntimeRequest",
    "RuntimeState",
    "SessionClear",
    "SessionClearCompleted",
    "SessionCompact",
    "SessionCompactCompleted",
    "SessionCompactStarted",
    "SessionCompacted",
    "SessionContextUpdate",
    "SessionContextUpdateCompleted",
    "SessionPrune",
    "SessionPruneCompleted",
    "SessionQuit",
    "SessionQuitCompleted",
    "SessionRecordAppended",
    "SessionSaveDue",
    "SessionSpendCharged",
    "SessionSummary",
    "SessionSummaryCompleted",
    "SessionTitleChanged",
    "Stubs",
    "SystemMessageCompleted",
    "ToolCallCompleted",
    "ToolCohortCompleted",
    "ToolCohortStarted",
    "ToolDetach",
    "ToolDetachCompleted",
    "ToolHalt",
    "ToolHaltCompleted",
    "ToolMessageCompleted",
    "ToolWakeSchedules",
    "UserMessageCompleted",
    "WakeSchedule",
    "wire_role",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelHalt:
    """Stop the model's reply; the calls it waited on run on, detached."""

    timestamp: datetime | None = None
    sender: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMessage:
    """What the model replied."""

    timestamp: datetime | None = None
    response: Response


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRetry:
    """Call the model again for the turn a failure stopped."""

    timestamp: datetime | None = None
    sender: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionClear:
    """Stop the model and every tool, and wipe the history."""

    timestamp: datetime | None = None
    sender: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCompact:
    """Replace the history with a summary of it."""

    timestamp: datetime | None = None
    sender: str | None = None
    summary_must_keep: str = ""
    trigger: Literal["auto", "manual"] = "manual"


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionContextUpdate:
    """Put ``context`` in force."""

    timestamp: datetime | None = None
    sender: str | None = None
    context: SessionContext


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionPrune:
    """Replace the output of each named call with a stub."""

    timestamp: datetime | None = None
    sender: str | None = None
    call_ids: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionQuit:
    """End the session."""

    timestamp: datetime | None = None
    sender: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDetach:
    """Stop waiting on a call and its group; they run on."""

    timestamp: datetime | None = None
    sender: str | None = None
    call_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolHalt:
    """Stop a call; ``None`` stops every one."""

    timestamp: datetime | None = None
    sender: str | None = None
    call_id: str | None = None


type RuntimeRequest = (
    AgentMessage
    | ModelHalt
    | ModelMessage
    | ModelRetry
    | SessionClear
    | SessionCompact
    | SessionContextUpdate
    | SessionPrune
    | SessionQuit
    | SystemMessage
    | ToolCall
    | ToolDetach
    | ToolHalt
    | ToolMessage
    | UserMessage
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentMessageCompleted:
    """An agent's message was answered, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvent:
    """What a child agent did, under its label."""

    label: str
    event: RuntimeEvent


@dataclass(frozen=True, slots=True, kw_only=True)
class LoginSwitched:
    """The model moved to another login of its provider: off a spent one, or back."""

    provider: str
    from_login: str
    to_login: str
    spent_window: str = ""
    spent_window_resets_at: datetime | None = None
    first_login_in_order: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelHaltCompleted:
    """The model's reply was stopped, or there was none to stop."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMessageCompleted:
    """The model's reply was taken in, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseCompleted:
    """A model call ended; it says nothing of the turn."""

    outcome: Outcome[Response]
    chars_streamed_before_cancel: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseStarted:
    """A model call began."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseSuspended:
    """A model call failed in a way that may pass; the turn waits, then calls again."""

    exception: RetryableError
    retry_at: datetime
    failures_in_a_row: int
    provider: str = ""
    reset_at_by_login: Mapping[str, datetime] = MappingProxyType({})

    # The rest are a backoff's waits, minutes at most: telling each would wake
    # whoever is told every few seconds of an outage.
    @property
    def opens_a_quota_wait(self) -> bool:
        """Whether this is the first wait of a spent quota, which may last days."""
        return (
            isinstance(self.exception, RateLimitedError) and self.failures_in_a_row == 1
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseTextStreamed:
    """Text the model wrote, as it streams."""

    text: str
    response_record_index: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelResponseThinkingStreamed:
    """Reasoning the model wrote, as it streams."""

    text: str
    response_record_index: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRetryCompleted:
    """The model was called again, or no failure had stopped a turn."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionClearCompleted:
    """The history was wiped, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCompacted:
    """What a compaction did to the history."""

    masked_tokens: int = 0
    replacement_tokens: int = 0
    replacement_entries: int = 0
    summary_fallback_reason: str = ""
    entries_kept_verbatim: int = 0

    @classmethod
    def from_splice(cls, splice: CompactionSplice) -> SessionCompacted:
        """Return what a compactor's splice did.

        Args:
          splice: The splice the compactor returned.

        Returns:
          compacted: Every field read off ``splice``.

        """
        return cls(
            masked_tokens=splice.masked_tokens,
            replacement_tokens=splice.replacement_tokens,
            replacement_entries=len(splice.replacement),
            summary_fallback_reason=splice.summary_fallback_reason,
            entries_kept_verbatim=splice.entries_kept_verbatim,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCompactCompleted:
    """A compaction ended; unless it succeeded, the tape is as it was."""

    outcome: Outcome[SessionCompacted]
    tape_len_before: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCompactStarted:
    """A compaction began."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionContextUpdateCompleted:
    """A context was put in force, or was not."""

    outcome: Outcome[SessionContext]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionPruneCompleted:
    """The named outputs were replaced, or were not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionQuitCompleted:
    """The session ended."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSaveDue:
    """The session is consistent: observers may persist it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSpendCharged:
    """Spend counted against the agent that its own replies did not bring."""

    spend: TokenCost
    label: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionRecordAppended:
    """A request was written to the log."""

    record: RuntimeRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSummary:
    """What a span of the log was about."""

    record_span: range
    text: str
    text_embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSummaryCompleted:
    """A span of the log was summarized, or was not."""

    outcome: Outcome[SessionSummary]


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionTitleChanged:
    """The agent set the session's title, as the terminal shows it."""

    title: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemMessageCompleted:
    """The harness's message was answered, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallCompleted:
    """A tool call ended."""

    outcome: Outcome[ToolMessage]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCohortCompleted:
    """The wait on a reply's calls ended: every result came, or it was called off."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCohortStarted:
    """The model began to wait on a reply's calls."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDetachCompleted:
    """A call was answered with a stub and runs on, or was not."""

    outcome: Outcome[ToolMessage]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolHaltCompleted:
    """A call was stopped, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolMessageCompleted:
    """A tool call's message was answered, or was not."""

    outcome: Outcome[None]


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessageCompleted:
    """A person's message was answered, or was not."""

    outcome: Outcome[None]


type RuntimeState = (
    AgentIdle
    | AgentMessageCompleted
    | ChildEvent
    | LoginSwitched
    | ModelHaltCompleted
    | ModelIdle
    | ModelMessageCompleted
    | ModelResponseCompleted
    | ModelResponseStarted
    | ModelResponseSuspended
    | ModelResponseTextStreamed
    | ModelResponseThinkingStreamed
    | ModelRetryCompleted
    | SessionClearCompleted
    | SessionCompactCompleted
    | SessionCompactStarted
    | SessionContextUpdateCompleted
    | SessionPruneCompleted
    | SessionQuitCompleted
    | SessionRecordAppended
    | SessionSaveDue
    | SessionSpendCharged
    | SessionSummaryCompleted
    | SessionTitleChanged
    | SystemMessageCompleted
    | ToolCallCompleted
    | ToolCohortCompleted
    | ToolCohortStarted
    | ToolDetachCompleted
    | ToolHaltCompleted
    | ToolMessageCompleted
    | UserMessageCompleted
)


type RuntimeEvent = RuntimeRequest | RuntimeState


# Code tells a pending or cancelled call's record by ``ToolMessage.state``, never by
# this text, so the words may change, and a tool's output that happens to match
# them is never mistaken for a placeholder.
@dataclass(frozen=True, slots=True, kw_only=True)
class Stubs:
    """What the runtime writes into the context for the calls it answers itself."""

    detached: str = (
        "[detached: tool still running; real result arrives in a later message]"
    )
    cancelled: str = "[cancelled: tool killed before completion; no result will follow]"
    pruned: str = "[pruned: old tool output removed to keep the context small]"
    background_prefix: str = "[Running in background: "


@dataclass(frozen=True, slots=True, kw_only=True)
class NextWake:
    """When a long call next wakes the model."""

    delay_sec: float
    fires_even_without_new_output: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeSchedule:
    """When a long call wakes the model: early, then less often, as a person glances.

    The gap to the next wake is ``max(first_gap_sec, gap_growth * elapsed)``: a wake so
    timed fires only when the call wrote something since the model last looked.
    ``max_gap_sec`` and the prompt cache's deadline cap the gap; a capped wake
    fires regardless, so a quiet call is still seen and the cache stays warm.

    Raises:
      ValueError: A knob that would never wake, or wake without end.

    """

    first_gap_sec: float = 15.0
    gap_growth: float = 1.0
    max_gap_sec: float = 300.0

    def __post_init__(self) -> None:
        for name, value in (
            ("first_gap_sec", self.first_gap_sec),
            ("max_gap_sec", self.max_gap_sec),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a number of seconds above 0, got {value}.",
                )
        if not (math.isfinite(self.gap_growth) and self.gap_growth >= 0):
            raise ValueError(
                f"gap_growth must be a number at least 0, got {self.gap_growth}.",
            )

    def delay_sec_until_next_wake(
        self,
        elapsed_sec: float,
        *,
        cache_left_sec: float,
    ) -> NextWake:
        """Return when the next wake comes, and whether it fires without new output.

        Args:
          elapsed_sec: How long the call has run.
          cache_left_sec: How long the prompt cache stays warm, less a margin;
            ``math.inf`` when it does not bound the wait.

        Returns:
          next_wake: Its delay; a cap set it when it fires without new output.

        """
        paced = max(self.first_gap_sec, self.gap_growth * elapsed_sec)
        cap = min(cache_left_sec, self.max_gap_sec)
        return NextWake(
            delay_sec=min(paced, cap),
            fires_even_without_new_output=cap <= paced,
        )

    @override
    def __str__(self) -> str:
        """Return the schedule as the setting takes it: ``15s 1 300s``."""
        return f"{self.first_gap_sec:g}s {self.gap_growth:g} {self.max_gap_sec:g}s"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolWakeSchedules:
    """How each tool's long calls wake the model; ``None`` never wakes, nor detaches."""

    default: WakeSchedule | None = WakeSchedule()
    per_tool: Mapping[str, WakeSchedule | None] = field(
        default_factory=dict[str, "WakeSchedule | None"],
    )


class CompactionSplice(Protocol):
    """What a compaction returns, as ``SessionCompacted.from_splice`` reads it."""

    @property
    def replacement(self) -> tuple[ModelContextEvent, ...]:
        """The entries the splice puts in the context."""
        ...

    @property
    def masked_tokens(self) -> int:
        """Tokens of what the splice replaces."""
        ...

    @property
    def replacement_tokens(self) -> int:
        """Tokens of what it puts in their place."""
        ...

    @property
    def summary_fallback_reason(self) -> str:
        """Why the summary fell back to keeping history, if it did."""
        ...

    @property
    def entries_kept_verbatim(self) -> int:
        """How many recent entries the fallback kept verbatim."""
        ...


type ModelContextEvent = (
    UserMessage
    | AgentMessage
    | Response
    | ToolMessage
    | SystemMessage
    | SessionContext
    | ContextCompaction
)


def wire_role(entry: ModelContextEvent) -> str | None:
    """Return the provider-wire role for ``entry``.

    Args:
      entry: One entry of the history.

    Returns:
      role: ``"user"`` for a user or agent message, ``"assistant"`` for a
          model turn, ``None`` for a tool result or a harness note.

    """
    if isinstance(entry, (UserMessage, AgentMessage)):
        return "user"
    if isinstance(entry, Response):
        return "assistant"
    return None


class Job(Protocol):
    """A tool call the runtime started, in the foreground or the background."""

    @property
    def call(self) -> ToolCall:
        """The call it runs: the tool, its id, its arguments."""
        ...

    @property
    def elapsed_sec(self) -> float:
        """How long since the model asked for it, by ``call.timestamp``."""
        ...

    @property
    def status(self) -> Literal["running", "sleeping", "detached"]:
        """Where it is: the model waits on ``running``, not on ``sleeping`` or ``detached``."""
        ...

    @property
    def runs_until_stopped(self) -> bool:
        """Whether it ends only when it is stopped, as its tool said once it started."""
        ...

    def cancel(self) -> None:
        """Stop it; the runtime records it as cancelled."""
        ...
