"""Dispatch protocol event vocabulary.

The data classes consumed by ``agent.runtime.AgentRuntime``'s inbox
match block and published to observers. Pure data; no behaviour. Used
by the engine, renderers, observers, persistence, and tests.

Naming clash note: ``sagent.types.runtime`` (this
module) holds the *event* types; ``sagent.agent.runtime``
holds the *engine* that consumes them. Different packages, no Python
conflict; the parallel naming signals the relationship.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sagent.types.history import (
    AssistantMessage,
    BytesMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)


__all__ = [
    "BudgetReset",
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
    "HistoryEntryUpdated",
    "Kill",
    "ModelCallStarted",
    "ModelIdle",
    "ModelResponseCancelled",
    "ModelResponseComplete",
    "ModelResponseError",
    "ModelResponsePartial",
    "ModelResponseThinking",
    "ModelSwitch",
    "ModelSwitchRejected",
    "Quit",
    "Recompact",
    "RuntimeEvent",
    "SaveSession",
    "StatusChanged",
    "ToolLabel",
    "ToolResultPartial",
    "Undetach",
    "UserQueuedMessage",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class UserQueuedMessage:
    """User context that doesn't preempt. Waits for cohort to finish."""

    text: str
    """Plain-text content to merge into the next ``UserMessage``."""

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
    """Specific call to detach, or ``None`` to detach all."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class Undetach:
    """Re-gate model on a detached tool's completion."""

    call_id: str | None = None
    """Specific call to re-gate on, or ``None`` for all."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitch:
    """Queue a model swap; the runtime applies it once safe.

    The slash handler builds the new model synchronously and packages
    the swap as ``apply``. The runtime defers the call until
    ``model_call`` and ``compact_task`` are both ``None`` so the
    in-flight response finishes against the OLD model (cost
    attribution, retry state, etc. stay self-consistent) and only the
    NEXT call uses the new model.
    """

    apply: Callable[[], None]
    """Closure that performs the swap (typically ``agent.swap_model``)."""

    label: str = ""
    """Optional human-readable label shown to renderers (e.g. ``old -> new``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSwitchRejected:
    """Model swap failed; runtime keeps the existing model."""

    exception: BaseException
    """The raised exception surfaced to observers."""

    label: str = ""
    """Human-readable switch label, when available."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetReset:
    """Agent ``ContextBudget`` was reset to fit the new model.

    Published by ``Agent._apply_model_change`` when a queued model
    swap would have overflowed the prior budget (typical case:
    swapping from a 1M-context model to a 200K-context model). The
    budget is replaced with ``ContextBudget.from_model(new)`` so the
    derived fields (``buffer_tokens``, ``reattach_*``, etc.) cohere
    with the new context window. Renderers surface this so a user
    with customised budgets knows to re-apply them post-swap.
    """

    model_id: str
    """Provider-specific model id the budget was sized for."""

    prior_max_request_tokens: int
    """Pre-reset input cap."""

    prior_max_response_tokens: int
    """Pre-reset output cap."""

    new_max_request_tokens: int
    """Post-reset input cap (matches the new model)."""

    new_max_response_tokens: int
    """Post-reset output cap (matches the new model)."""


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


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class ModelIdle:
    """Model finished with no tool calls."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class AgentIdle:
    """Agent has fully drained: about to block on inbox with no work.

    Edge-triggered: fires once per transition into the blocked-on-inbox
    state, suppressed until the next ``drain()`` returns work. Distinct
    from :class:`ModelIdle`, which fires per-round when the model
    produces no tool calls without considering inbox emptiness, cohort,
    detached tools, compaction, mid-stream buffer, or inbox gate state.

    Not published on cold start (the runtime initializes the
    ``_was_idle`` flag so the first iteration suppresses publish until
    the agent has consumed at least one batch of work).
    """


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CohortStarted:
    """Tool cohort has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultPartial:
    """Streaming chunk from a tool (e.g. Bash long output)."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Newly arrived output chunk."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachedResult:
    """A previously-detached tool completed."""

    call_id: str
    """Id of the originating ``ToolCall``."""

    content: str
    """Result text from the completed tool."""

    is_error: bool = False
    """True when the tool raised or signalled failure."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEntryUpdated:
    """An existing history entry was mutated in-place (splice).

    Publish-only. Emitted when ``DetachedResult`` (or its in-batch
    sibling) splices real tool output into a ``[detached]`` placeholder
    via ``dataclasses.replace``. Same ``id``, new ``content``.

    Without this event the persistence observer never learns of the
    update -- its delta-based append (``history[persisted_len:]``)
    can't see in-place mutations, so resumed sessions would load the
    stale placeholder. The persistence observer listens for this event
    and re-emits the entry; the loader dedupes by ``id`` so last-write-
    wins.
    """

    entry: HistoryEntry
    """The updated history entry (carries id + new content)."""


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
    """Reload last pre-compact transcript and re-run compaction."""

    args: str = ""
    """Free-form compaction instructions for the compactor."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class CompactStarted:
    """Compaction task has been spawned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactComplete:
    """Compaction finished; splice summary into history."""

    summary: list[HistoryEntry]
    """Compactor output to install at the head of history."""

    snapshot_len: int
    """History length captured before compaction; entries appended after
    that index are preserved post-splice."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactFailed:
    """Compaction task raised; runtime keeps prior history.

    Mirror of ``CompactComplete`` for the failure path. The dispatch
    loop's handler clears ``compact_task`` (so subsequent ``ModelSwitch``
    / model-call gates unblock) and splices a visible
    ``[Compaction error: ...]`` ``UserMessage`` into history so the
    model can react.
    """

    exception: BaseException
    """The compactor's raised exception."""

    snapshot_len: int
    """History length captured before compaction (for symmetry with
    ``CompactComplete``)."""


@dataclass(frozen=True, slots=True)  # check-dataclass: ignore[kw_only]
class SaveSession:
    """Signals observers to persist session state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusChanged:
    """Agent ``status`` field was updated.

    Publish-only: emitted by ``Agent.status`` setter. Renderers update
    the terminal title, persistence observers re-flush ``meta`` so the
    new status survives a crash even without a history delta.
    """

    text: str
    """New status string."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolLabel:
    """Pre-execution label for a tool call (REPL rendering).

    Publish-only: tool wrappers fan this out via ``runtime.publish``
    before invoking the inner tool. Never enters the inbox, never
    hits the match block.
    """

    call_id: str
    """Id of the originating ``ToolCall``."""

    text: str
    """Short label rendered above the call (e.g. ``Read(path)``)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildEvent:
    """Wrapped event from a child agent (AgentSpawn forwarding).

    Publish-only: AgentSpawn's observer wraps each child event and
    fans it out on the parent's runtime. Never enters the inbox.
    """

    label: str
    """Child agent's display label."""

    inner: RuntimeEvent
    """The forwarded child event."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildDoneEvent:
    """Child agent completed; carries totals for the gutter.

    Publish-only: AgentSpawn emits one at completion. Never enters
    the inbox.
    """

    label: str
    """Child agent's display label."""

    elapsed: float
    """Wall-clock seconds the child ran."""

    tokens: int
    """Total tokens (input + output) the child consumed."""

    cost: float
    """Total USD cost attributable to the child."""


type RuntimeEvent = (
    Quit
    | Halt
    | Clear
    | Kill
    | Detach
    | Undetach
    | UserMessage
    | UserQueuedMessage
    | ModelSwitch
    | ModelSwitchRejected
    | BudgetReset
    | ModelCallStarted
    | ModelResponsePartial
    | ModelResponseThinking
    | ModelResponseComplete
    | ModelResponseCancelled
    | ModelResponseError
    | ModelIdle
    | AgentIdle
    | CohortStarted
    | ToolResultPartial
    | ToolResult
    | DetachedResult
    | HistoryEntryUpdated
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
