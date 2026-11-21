"""Records of an agent session, one per act, and how a request ends.

What the model saw is the last ``ContextClear`` and every record after it.
Every request ends once, as ``Succeeded``, ``Cancelled`` or ``Failed``. A
message's ``due_after`` names a state the runtime publishes, ``ModelIdle`` or
``AgentIdle``, so those are defined here beside the messages.
``None`` means the source did not say; ``extra`` keeps what no field names, so
a claude or codex transcript converts here and back without loss.

Typical usage example:

  records: list[SessionRecord] = [
      UserMessage(content="fix the bug"),
      ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
      ToolMessage(
          call_id="1",
          state="done",
          stdout=("...",),
          result=ReadResult(path="a.py"),
      ),
      AssistantMessage(content="fixed"),
  ]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Literal,
    Protocol,
    Self,
    cast,
    overload,
    runtime_checkable,
)
from typing_extensions import TypeVar

from treekle import PlainTree, ReadError, from_plain, immutable, to_plain

from sagent.model.capability import SummaryKind, ThinkingEffort
from sagent.model.cost import TokenCost


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle import Classes, MutablePlainTree


__all__ = [
    "CONCRETE_TYPES",
    "FORMER_RESULT_KINDS",
    "FORMER_TAPE_KINDS",
    "RENAMED",
    "RENAMED_KINDS",
    "AgentIdle",
    "AgentMessage",
    "AgentSelfResult",
    "AgentSendResult",
    "AgentSpawnResult",
    "AssistantMessage",
    "Attachment",
    "BackgroundTaskResult",
    "BashResult",
    "Cancelled",
    "ContextClear",
    "ContextCompaction",
    "ContextState",
    "EditResult",
    "Failed",
    "GlobResult",
    "GrepResult",
    "IncompleteRecord",
    "KnownToolResult",
    "LineSpan",
    "ListResult",
    "McpResult",
    "MemoryResult",
    "ModelIdle",
    "ModelTurn",
    "Outcome",
    "Pairing",
    "PairingRole",
    "PaperAuthorResult",
    "PaperDetailsResult",
    "PaperFetchResult",
    "PaperSearchResult",
    "PlanResult",
    "PlayAudioResult",
    "ReadResult",
    "RecallResult",
    "RemoteToolResult",
    "RequestMeasurement",
    "Result_co",
    "SessionContext",
    "SessionRecord",
    "SessionResult",
    "SkillResult",
    "Splice",
    "StreamErr_co",
    "StreamOut_co",
    "Succeeded",
    "SystemMessage",
    "Thinking",
    "TokenUsage",
    "ToolCall",
    "ToolMessage",
    "TranscriptItem",
    "UncategorizedRecord",
    "UncategorizedToolResult",
    "UserMessage",
    "WebBrowserResult",
    "WebFetchResult",
    "WebSearchHit",
    "WebSearchResult",
    "WriteResult",
    "as_sent",
    "current_plain",
    "first_name",
    "former_kind",
    "labeled_text",
    "late_messages",
    "now_timestamp",
    "pairings",
    "parsed_timestamp",
    "record_classes",
    "renamed_records",
    "stated_timestamp",
    "stored_names",
    "stored_plain",
    "stored_record",
    "timestamp_as_written",
]

CONCRETE_TYPES: Final = MappingProxyType[type, type](
    {Mapping: MappingProxyType, Sequence: tuple},
)
"""How a record's JSON fields are stored: written as the plain objects the
provider sent, and read back frozen. :func:`stored_plain` and
:func:`stored_record` apply it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Succeeded[T]:
    """It did what was asked."""

    value: T


@dataclass(frozen=True, slots=True, kw_only=True)
class Cancelled:
    """It was stopped first."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Failed:
    """It could not."""

    exception: BaseException


type Outcome[T] = Succeeded[T] | Cancelled | Failed


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentIdle:
    """The agent has nothing left to do: no work, nothing owed, nothing queued."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelIdle:
    """The model's work on the turn stopped: finished, halted, or failed."""

    outcome: Outcome[None]


def _no_json() -> Mapping[str, PlainTree]:
    """Return an empty JSON field."""
    return MappingProxyType[str, PlainTree]({})


# A frozen record holding a dict could still change, and would store as
# ``py/dict``; frozen, it stores as the plain object the provider sent.
def _freeze(record: object, *names: str) -> None:
    """Hold each JSON field ``names`` of ``record`` read-only, however it was given."""
    for name in names:
        object.__setattr__(
            record,
            name,
            immutable(cast("object", getattr(record, name))),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionContext:
    """Settings in force for every record after it, until the next one."""

    context_id: int | None = None
    timestamp: datetime | None = None
    user: str | None = None
    """Whose session it is: who a message that names no sender, or names them, is from."""

    cwd: Path | None = None
    """Where the tools run."""

    environment: str | None = None
    """The git state, platform, shell, and knowledge cutoff, one per line."""

    agents_md: str | None = None
    """The AGENTS.md text in force; ``None`` when unchanged since the last one."""

    permission: str | None = None
    model: str | None = None
    effort: ThinkingEffort | None = None
    tools: tuple[str, ...] | None = None
    """The tools the model may call, by name; ``None`` for every one offered."""

    skipped_instruction_files: tuple[Path, ...] = ()
    """AGENTS.md, rule, and skill files the model was not given."""

    summary_kind: SummaryKind | None = None
    encoding: Mapping[str, PlainTree] = field(default_factory=_no_json)
    """How the source file spelled its bytes, so a rewrite reproduces them."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "encoding", "extra")

    def text(self) -> str:
        """Return what the model reads: where it works, as what, and when.

        Returns:
          text: One line per setting the record states; empty when it states none.

        """
        lines = (
            f"User: {self.user}" if self.user is not None else None,
            f"Working directory: {self.cwd}" if self.cwd is not None else None,
            self.environment,
            f"Model: {self.model}" if self.model is not None else None,
            f"Tools you may call: {', '.join(self.tools) or 'none'}"
            if self.tools is not None
            else None,
            f"Current time: {_local(self.timestamp)}"
            if self.timestamp is not None
            else None,
            "Instruction files not given to you: "
            + ", ".join(str(p) for p in self.skipped_instruction_files)
            if self.skipped_instruction_files
            else None,
            self.agents_md,
        )
        return "\n".join(filter(None, lines))


# The model greets by the time of day where the user is, so the offset alone is not
# enough: the zone's name says which day and hour it is there.
def _local(timestamp: datetime) -> str:
    """Return a time as ``Mon 2026-10-26 17:44 -07:00 (PDT)``, here."""
    here = timestamp.astimezone()
    offset = here.strftime("%z")
    return f"{here:%a %Y-%m-%d %H:%M} {offset[:3]}:{offset[3:]} ({here.tzname()})"


def now_timestamp() -> datetime:
    """Return the time now, local, with its UTC offset, to the second: a ``timestamp``."""
    return datetime.now().astimezone().replace(microsecond=0)


# With no offset a time compares with none that has one, so two records of one
# session could not be put in order.
def parsed_timestamp(spelled: str) -> datetime | None:
    """Return an ISO 8601 time, one with no offset read as UTC; ``None`` if it is none.

    Args:
      spelled: A time as a source wrote it.

    Returns:
      timestamp: The time, with an offset. A reader that must write ``spelled``
        back keeps it beside the record, as ``extra``'s ``$timestamp_as_written``.

    """
    try:
        when = datetime.fromisoformat(spelled)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


# A record given another time since it was read is written at that time: the
# spelling kept beside it names the time it was read with.
def timestamp_as_written(
    timestamp: datetime | None,
    beside: Mapping[str, object],
) -> str | None:
    """Return the text a record's time is written as.

    Args:
      timestamp: The record's time.
      beside: The record's ``extra``, or an ``UncategorizedRecord``'s payload,
        which may keep the time's own spelling as ``$timestamp_as_written``.

    Returns:
      written: The spelling kept, while it names ``timestamp``; else the time
        as ``isoformat`` spells it, and ``None`` for none.

    """
    kept = beside.get("$timestamp_as_written")
    if isinstance(kept, str) and parsed_timestamp(kept) == timestamp:
        return kept
    return None if timestamp is None else timestamp.isoformat()


@dataclass(frozen=True, slots=True, kw_only=True)
class Attachment:
    """Binary content and its media type."""

    mime_descriptor: str
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage:
    """What a person said."""

    context_id: int | None = None
    timestamp: datetime | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    sender: str | None = None
    """Who said it; ``None`` for the session's own user (``SessionContext.user``).

    A label the model reads (``labeled_text``), as whoever sent the record wrote
    it. It grants nothing: who may do what to a session is decided from the
    author a server stamps on a line it took in, never from this."""

    due_after: type[AgentIdle | ModelIdle] | None = None
    """What a runtime sent it waits for before it takes it in: the model done
    (``ModelIdle``), or no call running either (``AgentIdle``). ``None`` waits for
    neither, and is what a record taken in holds."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage:
    """What the model said."""

    context_id: int | None = None
    timestamp: datetime | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class Thinking:
    """The model's reasoning: readable, sealed, or summarized."""

    context_id: int | None = None
    timestamp: datetime | None = None
    content: str | None = None
    encrypted: str | None = None
    summary: str | None = None
    reasoning_id: str | None = None
    """OpenAI's id for its reasoning (``""`` if unstated); ``None`` if not OpenAI's."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """A tool invocation the model asked for."""

    context_id: int | None = None
    timestamp: datetime | None = None
    call_id: str
    name: str
    arguments: Mapping[str, PlainTree] = field(default_factory=_no_json)
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "arguments", "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class UncategorizedToolResult:
    """Result of a tool that defines no result class of its own."""

    exit_code: int = 0


Result_co = TypeVar("Result_co", covariant=True, default=object)
"""Type of a tool's result, such as ``ReadResult``; it has an ``exit_code``."""

StreamOut_co = TypeVar("StreamOut_co", covariant=True, default=object)
"""Type of each item a tool writes to standard output."""

StreamErr_co = TypeVar("StreamErr_co", covariant=True, default=object)
"""Type of each item a tool writes to standard error."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Splice:
    """One replacement in an edit: ``before`` became ``after``.

    An insertion has an empty ``before``; a deletion an empty ``after``.
    """

    before: str | None = None
    after: str | None = None
    lead: str | None = None
    """Unchanged lines a diff showed before the change, ``@@`` header included."""

    trail: str | None = None
    """Unchanged lines a diff showed after the change."""

    start: int | None = None
    """First replaced line, one-based; ``None`` when located by content."""

    count: int | None = None
    """Lines replaced."""

    bare: frozenset[str] = frozenset()
    """Which of ``before``/``after``/``trail`` the diff marked as lacking a final newline."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LineSpan:
    """A range of a file's lines; ``Read`` writes one when it returns part of a file."""

    start: int | None = None
    """One-based number of the first line; ``None`` for line 1."""

    count: int | None = None
    """Number of lines; ``None`` for every line from ``start`` on."""


@runtime_checkable
class _Exited(Protocol):
    """A tool's result: it has the call's exit code."""

    exit_code: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolMessage(Generic[Result_co, StreamOut_co, StreamErr_co]):
    """The record of one tool call: its state, its output, and the tool's result.

    A tool does not build this record. The caller that runs the tool does, with
    ``sagent.tools.types.settled``.
    """

    context_id: int | None = None
    timestamp: datetime | None = None
    call_id: str
    wire_id: str = ""
    """The id a call run on another machine went by on the wire, which a relay
    between logs it by; ``""`` for a call run here. Not ``call_id``: the model's ids
    are unique only within one agent's window, and a machine serves every agent of
    a session."""

    due_after: type[AgentIdle | ModelIdle] | None = None
    """As ``UserMessage.due_after``."""

    state: Literal["pending", "done", "failed", "cancelled"]
    """``pending``: the tool still runs; this record is its placeholder, or news of
    what it wrote since. ``done``: the tool returned ``result``. ``failed``: the tool raised.
    ``cancelled``: the call was stopped before the tool returned."""

    stdout: tuple[StreamOut_co, ...] = ()
    """The items the tool wrote to standard output, in order: text, and typed items
    such as an ``Attachment``, a ``Splice``, or a ``LineSpan``."""

    stderr: tuple[StreamErr_co, ...] = ()
    """The items the tool wrote to standard error, in order."""

    result: Result_co | None = None
    """The tool's result; ``None`` when the tool returned none."""

    shown: str | None = None
    """Text the model reads instead of ``stdout`` and ``stderr``: a placeholder's
    note, or what a prune or a compaction left."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")

    @property
    def is_error(self) -> bool:
        """Whether the call failed.

        A call failed when its tool raised, when it was cancelled, or when its
        result has a nonzero exit code. A ``BashResult`` is the exception: its
        exit code is the command's, and the tool still ran the command.
        """
        result = self.result
        return self.state in ("failed", "cancelled") or (
            isinstance(result, _Exited)
            and not isinstance(result, BashResult)
            and result.exit_code not in (None, 0)
        )

    @property
    def attachments(self) -> tuple[Attachment, ...]:
        """The attachments in ``stdout``."""
        return tuple(item for item in self.stdout if isinstance(item, Attachment))

    def text(self) -> str:
        """Return the text the model reads for this call.

        Returns:
          text: ``shown`` if it is set. Otherwise the text items of ``stdout``,
            then of ``stderr``, then ``[exit N]`` for a ``BashResult`` whose
            exit code ``N`` is nonzero.

        """
        if self.shown is not None:
            return self.shown
        result = self.result
        output = "".join(
            item for item in (*self.stdout, *self.stderr) if isinstance(item, str)
        )
        if not isinstance(result, BashResult) or result.exit_code in (None, 0):
            return output
        # The note is a line of its own, after the output's last.
        line = "" if not output or output.endswith("\n") else "\n"
        return f"{output}{line}[exit {result.exit_code}]"

    def restated(self, text: str) -> Self:
        """Return this result reading ``text``, as a prune or compaction leaves it.

        Args:
          text: What the model reads instead; its attachments go with the rest.

        Returns:
          result: This result, showing ``text``, with no attachment.

        """
        return replace(self, shown=text).attached(())

    def attached(self, attachments: Sequence[Attachment]) -> Self:
        """Return this result with ``attachments`` in place of its own.

        Args:
          attachments: What it holds instead, after whatever else it wrote.

        Returns:
          result: This result, its other items in their order.

        """
        kept = tuple(item for item in self.stdout if not isinstance(item, Attachment))
        return replace(self, stdout=(*kept, *attachments))


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadResult:
    """Result of a ``Read`` call: the file read, and why no lines were returned."""

    path: str | None = None
    """The file read, resolved."""

    status: (
        Literal["unchanged", "binary", "undecodable", "empty", "past_end"] | None
    ) = None
    """Why the file's lines were not returned; ``None`` when they were."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteResult:
    """Result of a ``Write`` call: the file written and its new content."""

    path: str | None = None
    """The file written, resolved."""

    content: str | None = None
    """The text written."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class EditResult:
    """Result of an ``Edit`` call: the file edited. Each replacement is a ``Splice`` on stdout."""

    path: str | None = None
    """The file edited, resolved."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchHit:
    """One hit of a web search; ``WebSearch`` writes each to stdout after its text."""

    url: str | None = None
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchResult:
    """Result of a ``WebSearch`` call: the query and how long the search took."""

    query: str | None = None
    duration_sec: float | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class WebFetchResult:
    """Result of a ``WebFetch`` call: the URL, the HTTP status, the time, and the size."""

    url: str | None = None
    code: int | None = None
    """The HTTP status."""

    duration_sec: float | None = None
    size: int | None = None
    """Characters of the page, before any cut."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSpawnResult:
    """Result of an ``AgentSpawn`` call: the child agent and what its run used."""

    agent_id: str | None = None
    """The child's label."""

    agent_kind: str | None = None
    """``persistent``, or ``watcher``."""

    prompt: str | None = None
    model: str | None = None
    state: str | None = None
    """``completed`` or ``failed``; ``running`` for a watcher just started."""

    tokens: int | None = None
    spend: TokenCost | None = None
    duration_sec: float | None = None
    tool_calls: int | None = None
    output_file: str | None = None
    """The child's journal, when it keeps one."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class BashResult:
    """Result of a ``Bash`` call: the command run and its exit code."""

    command: tuple[str, ...] | None = None
    exit_code: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WebBrowserResult:
    """Result of a ``WebBrowser`` call: the action taken."""

    action: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GrepResult:
    """Result of a ``Grep`` call: the pattern and the path searched."""

    pattern: str = ""
    path: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GlobResult:
    """Result of a ``Glob`` call: the pattern and the directory searched."""

    pattern: str = ""
    path: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ListResult:
    """Result of a ``List`` call: the directory listed."""

    path: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResult:
    """Result of a ``Skill`` call: the skill loaded."""

    skill: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendResult:
    """Result of an ``AgentSend`` call: the recipient and the delivery delay."""

    to: str = ""
    delay_sec: float | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class BackgroundTaskResult:
    """Result of a ``BackgroundTask`` call: the operation and the job it named."""

    operation: str = ""
    job_id: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResult:
    """Result of an ``AgentSession`` call: the action and the agent whose session it was."""

    action: str = ""
    agent: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryResult:
    """Result of a ``Memory`` call: the action and the memory's name."""

    action: str = ""
    name: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayAudioResult:
    """Result of a ``PlayAudio`` call: the file played."""

    path: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSearchResult:
    """Result of a ``PaperSearch`` call: the query and the backend searched."""

    query: str = ""
    source: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperDetailsResult:
    """Result of a ``PaperDetails`` call: the operation and the backend asked."""

    operation: str | None = None
    source: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperAuthorResult:
    """Result of a ``PaperAuthor`` call: the operation."""

    operation: str | None = None
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteToolResult:
    """Result of a ``RemoteTool`` call: the endpoint asked."""

    endpoint: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class McpResult:
    """Result of a call to an MCP server's tool: the server and the tool."""

    server: str = ""
    tool: str = ""
    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSelfResult:
    """Result of an ``AgentSelf`` call."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperFetchResult:
    """Result of a ``PaperFetch`` call."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanResult:
    """Result of a ``Plan`` call."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallResult:
    """Result of a ``Recall`` call."""

    exit_code: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemMessage:
    """Instructions or status from the harness, such as a reminder."""

    context_id: int | None = None
    timestamp: datetime | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    subtype: str | None = None
    call_id: str | None = None
    """The call a former ``wake`` note was news of; a resume reads it as that call's
    news (``agent._as_today``). ``None`` for a note about no call."""

    due_after: type[AgentIdle | ModelIdle] | None = None
    """As ``UserMessage.due_after``."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestMeasurement:
    """A measured prompt and optional appended group, identified without storing text."""

    tokenizer: str
    model: str
    frame: str
    prefix: str
    records: int
    prompt: int
    block: str = ""
    block_records: int = 0
    block_tokens: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenUsage:
    """Token accounting and rate limits a provider reported."""

    context_id: int | None = None
    timestamp: datetime | None = None
    info: Mapping[str, PlainTree] = field(default_factory=_no_json)
    rate_limits: Mapping[str, PlainTree] = field(default_factory=_no_json)
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)
    measurement: RequestMeasurement | None = None

    def __post_init__(self) -> None:
        _freeze(self, "info", "rate_limits", "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextState:
    """Context injected into what the model sees."""

    context_id: int | None = None
    timestamp: datetime | None = None
    kind: str = ""
    content: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextCompaction:
    """A compaction happened; the ``ContextClear`` after it holds the result."""

    context_id: int | None = None
    timestamp: datetime | None = None
    summary: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)
    """Provider bookkeeping, and the instructions the compaction was given."""

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextClear:
    """A context window starts here, with exactly this content."""

    context_id: int | None = None
    timestamp: datetime | None = None
    cleared_session_id: str | None = None
    """The session this one replaced; ``None`` for a fresh conversation."""

    system_prompt: str | None = None
    summary: str | None = None
    """What replaces the cleared history; ``None`` when nothing was kept."""

    history: tuple[SessionRecord, ...] = ()
    """Records the new window starts with."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentMessage:
    """A message between agents."""

    context_id: int | None = None
    timestamp: datetime | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    sender: str | None = None
    recipient: str | None = None
    due_after: type[AgentIdle | ModelIdle] | None = None
    """As ``UserMessage.due_after``."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class UncategorizedRecord:
    """A source record no other type describes."""

    context_id: int | None = None
    timestamp: datetime | None = None
    kind: str
    payload: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "payload")


@dataclass(frozen=True, slots=True, kw_only=True)
class IncompleteRecord:
    """A source line that could not be parsed."""

    text: str


type KnownToolResult = ToolMessage[
    AgentSelfResult
    | AgentSendResult
    | AgentSpawnResult
    | BackgroundTaskResult
    | BashResult
    | EditResult
    | GlobResult
    | GrepResult
    | ListResult
    | McpResult
    | MemoryResult
    | PaperAuthorResult
    | PaperDetailsResult
    | PaperFetchResult
    | PaperSearchResult
    | PlanResult
    | PlayAudioResult
    | ReadResult
    | RecallResult
    | RemoteToolResult
    | SessionResult
    | SkillResult
    | WebBrowserResult
    | WebFetchResult
    | WebSearchResult
    | WriteResult
    | UncategorizedToolResult
    | None,
    Attachment | LineSpan | Splice | WebSearchHit | str,
    str,
]
"""A tool call's record limited to the result classes and stream item types
this module defines; ``host.wire`` checks a peer's record against it."""

type TranscriptItem = UserMessage | AssistantMessage | Thinking | ToolCall | ToolMessage
type SessionRecord = (
    TranscriptItem
    | SystemMessage
    | TokenUsage
    | ContextState
    | ContextCompaction
    | ContextClear
    | AgentMessage
    | SessionContext
    | UncategorizedRecord
    | IncompleteRecord
)


@runtime_checkable
class ModelTurn(Protocol):
    """A model turn held as one entry, as a context holds a reply: its records."""

    @property
    def records(self) -> Sequence[AssistantMessage | Thinking | ToolCall]:
        """What the model said and called, in order."""
        ...


type PairingRole = Literal[
    "reply",
    "answer",
    "elsewhere",
    "news",
    "orphan",
    "repeat",
    "message",
    "interrupt",
    "aside",
    "note",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Pairing:
    """Where one entry stands among the calls and their results.

    ``role`` is one of:

    - ``reply``: a model turn, or one of its records in a log.
    - ``answer``: the first result of a call the model waits on.
    - ``elsewhere``: a result of a call another entry answers (``paired_externally``).
    - ``news``: a later message of a call, its stub having answered it.
    - ``orphan``: a result among a turn's results that answers none of them.
    - ``repeat``: a result after its call's result.
    - ``message``: a user-side turn, with no call waiting.
    - ``interrupt``: a user's message while calls wait: they go unanswered.
    - ``aside``: a peer's message while calls wait: they may still be answered.
    - ``note``: no one's turn.
    """

    role: PairingRole
    waiting: tuple[str, ...] = ()
    """The calls that waited for a result as the entry came."""

    left_open: tuple[str, ...] = ()
    """The calls a reply or an interrupt found unanswered: none will be now."""

    reused: tuple[str, ...] = ()
    """The ids a reply calls under that an earlier entry named."""


def pairings(
    entries: Sequence[SessionRecord | ModelTurn],
    *,
    paired_externally: frozenset[str] = frozenset(),
    orphans_are_news: bool = True,
) -> tuple[list[Pairing], tuple[str, ...]]:
    """Return where each entry stands among the calls and results, and the calls left open.

    The one reading of how calls and results pair: a provider's checks, a splice's
    checks, and the repairs all read it. A model turn is one entry holding its
    records, as in a context, or a run of its records, as in a log. A call run apart
    talks back more than once, each time as a ``ToolMessage`` of its id: first its
    stub, then its news at each wake, then its result; only the first answers it.

    Args:
      entries: A context, a splice's replacement, or a log.
      paired_externally: Calls answered outside ``entries``: none waits here.
      orphans_are_news: Whether a result of a call no earlier entry named is news,
        as in a context a compaction took its call from; a log another CLI wrote
        holds such a result where it writes its call later.

    Returns:
      pairings: One per entry, in order.
      open: The calls still waiting at the end.

    """
    waiting: dict[str, None] = {}
    named: set[str] = set()
    ended: set[str] = set()
    out: list[Pairing] = []
    in_reply = False
    for entry in entries:
        now = tuple(waiting)
        if isinstance(entry, AssistantMessage | Thinking | ToolCall | ModelTurn):
            opens = isinstance(entry, ModelTurn) or not in_reply
            left = now if opens else ()
            if opens:
                ended.update(waiting)
                waiting = {}
            ids = (
                [r.call_id for r in entry.records if isinstance(r, ToolCall)]
                if isinstance(entry, ModelTurn)
                else [entry.call_id]
                if isinstance(entry, ToolCall)
                else []
            )
            reused = tuple(i for i in ids if i in named)
            named.update(ids)
            ended.difference_update(ids)
            waiting.update(dict.fromkeys(i for i in ids if i not in paired_externally))
            in_reply = not isinstance(entry, ModelTurn)
            out.append(
                Pairing(role="reply", waiting=now, left_open=left, reused=reused),
            )
            continue
        role: PairingRole
        left_open: tuple[str, ...] = ()
        if isinstance(entry, ToolMessage):
            cid = entry.call_id
            if cid in waiting:
                role = "answer"
                del waiting[cid]
            elif waiting:
                role = "orphan"
            elif cid in ended:
                role = "repeat"
            elif cid in paired_externally:
                role = "elsewhere"
            elif orphans_are_news or cid in named:
                role = "news"
            else:
                role = "orphan"
            if role != "orphan" and entry.state != "pending":
                ended.add(cid)
            named.add(cid)
        elif isinstance(entry, UserMessage | AgentMessage):
            if not waiting:
                role = "message"
            elif isinstance(entry, UserMessage):
                role, left_open = "interrupt", now
                ended.update(waiting)
                waiting = {}
            else:
                role = "aside"
        else:
            # A note or a measure written between a reply's records is no one's turn.
            out.append(Pairing(role="note", waiting=now))
            continue
        in_reply = False
        out.append(Pairing(role=role, waiting=now, left_open=left_open))
    return out, tuple(waiting)


def late_messages(
    entries: Sequence[SessionRecord | ModelTurn],
    *,
    orphans_are_news: bool = True,
) -> set[int]:
    """Return where a call's later message is: its news or result, after its stub.

    Args:
      entries: A context or a log.
      orphans_are_news: As ``pairings`` takes it.

    Returns:
      late: The positions in ``entries`` of the messages that answer no call.

    """
    found, _ = pairings(entries, orphans_are_news=orphans_are_news)
    return {at for at, pairing in enumerate(found) if pairing.role == "news"}


def as_sent[E: SessionRecord | ModelTurn](
    entries: Sequence[E],
    *,
    orphans_are_news: bool = True,
) -> list[E | UserMessage]:
    """Return ``entries`` as a provider takes them: each late message as user text.

    A provider takes one tool result per call, right after the turn that made
    it, so a call's news (``late_messages``) is the user's side, headed with the
    call it is of; a wake's news names its call already.

    Args:
      entries: A context or a log.
      orphans_are_news: As ``late_messages`` takes it.

    Returns:
      sent: ``entries``, each late ``ToolMessage`` a ``UserMessage``.

    """
    late = late_messages(entries, orphans_are_news=orphans_are_news)
    return [
        _told(entry) if at in late and isinstance(entry, ToolMessage) else entry
        for at, entry in enumerate(entries)
    ]


def labeled_text(entry: UserMessage | AgentMessage) -> str:
    """Return ``entry``'s text, labeled with who sent it.

    An agent's reads ``[from <agent>]: ``, and a named person's ``[from user
    <name>]: ``, so one of each by the same name are told apart. A message that
    names no sender, as the session's own user's does, is returned as it is.

    Args:
      entry: A message.

    Returns:
      text: The text, labeled once; a labeled text is returned as it is.

    """
    text = entry.content or ""
    if entry.sender is None:
        return text
    who = f"user {entry.sender}" if isinstance(entry, UserMessage) else entry.sender
    prefix = f"[from {who}]: "
    return text if text.startswith(prefix) else f"{prefix}{text}"


def stated_timestamp(record: SessionRecord) -> str | None:
    """Return the text ``record``'s time is written as, as its source spelled it.

    Args:
      record: Any session record.

    Returns:
      written: :func:`timestamp_as_written` of its time and what it keeps beside
        it; ``None`` for a record with no time.

    """
    if isinstance(record, IncompleteRecord):
        return None
    return timestamp_as_written(
        record.timestamp,
        record.payload if isinstance(record, UncategorizedRecord) else record.extra,
    )


def record_classes() -> dict[str, type]:
    """Return each dataclass a session's stored form names.

    That is each class a record holds, and each state a message's ``due_after``
    names. No record holds an outcome, so ``Succeeded``, ``Cancelled`` and
    ``Failed`` are not among them.

    Returns:
      classes: Each class, by the name its stored form carries.

    """
    return {
        kind.__name__: kind
        for kind in (
            SessionContext,
            Attachment,
            UserMessage,
            AssistantMessage,
            Thinking,
            ToolCall,
            Splice,
            LineSpan,
            ToolMessage,
            UncategorizedToolResult,
            ReadResult,
            WriteResult,
            EditResult,
            WebSearchHit,
            WebSearchResult,
            WebFetchResult,
            AgentSpawnResult,
            BashResult,
            WebBrowserResult,
            GrepResult,
            GlobResult,
            ListResult,
            SkillResult,
            AgentSendResult,
            BackgroundTaskResult,
            SessionResult,
            MemoryResult,
            PlayAudioResult,
            PaperSearchResult,
            PaperDetailsResult,
            PaperAuthorResult,
            RemoteToolResult,
            McpResult,
            AgentSelfResult,
            PaperFetchResult,
            PlanResult,
            RecallResult,
            SystemMessage,
            RequestMeasurement,
            TokenUsage,
            ContextState,
            ContextCompaction,
            ContextClear,
            AgentMessage,
            UncategorizedRecord,
            IncompleteRecord,
            AgentIdle,
            ModelIdle,
        )
    }


# Read only: treekle names each class once, so an old tag is renamed in the plain
# tree before it decodes. Kept until no stored record carries an old name.
RENAMED: Final = MappingProxyType[str, str](
    {
        "ShellCommandResult": "BashResult",
        "FileReadResult": "ReadResult",
        "FileWriteResult": "WriteResult",
        "FileEditResult": "EditResult",
        "AgentStatusResult": "AgentSpawnResult",
        "FunctionResult": "UncategorizedToolResult",
    },
)
"""Maps each class name that was renamed to the name that replaced it."""

_FORMER_RESULT_RECORDS: Final = frozenset(
    {
        "AgentSelfResult",
        "AgentSendResult",
        "AgentSpawnResult",
        "BackgroundTaskResult",
        "BashResult",
        "EditResult",
        "GlobResult",
        "GrepResult",
        "ListResult",
        "McpResult",
        "MemoryResult",
        "PaperAuthorResult",
        "PaperDetailsResult",
        "PaperFetchResult",
        "PaperSearchResult",
        "PlanResult",
        "PlayAudioResult",
        "ReadResult",
        "RecallResult",
        "RemoteToolResult",
        "SessionResult",
        "SkillResult",
        "ToolErrorResult",
        "UncategorizedToolResult",
        "WebBrowserResult",
        "WebFetchResult",
        "WebSearchResult",
        "WriteResult",
    },
)
"""Names of the classes once stored as a whole tool-call record, ``call_id``
included. A stored record tagged with one is read as a ``ToolMessage``."""


_RESULT_FIELDS: Final = MappingProxyType[str, tuple[str, ...]](
    {
        "BashResult": ("command",),
        "WebBrowserResult": ("action",),
        "GrepResult": ("pattern", "path"),
        "GlobResult": ("pattern", "path"),
        "ListResult": ("path",),
        "SkillResult": ("skill",),
        "AgentSendResult": ("to", "delay_sec"),
        "BackgroundTaskResult": ("operation", "job_id"),
        "SessionResult": ("action", "agent"),
        "MemoryResult": ("action", "name"),
        "PlayAudioResult": ("path",),
        "PaperSearchResult": ("query", "source"),
        "PaperDetailsResult": ("operation", "source"),
        "PaperAuthorResult": ("operation",),
        "RemoteToolResult": ("endpoint",),
        "McpResult": ("server", "tool"),
        "AgentSelfResult": (),
        "PaperFetchResult": (),
        "PlanResult": (),
        "RecallResult": (),
        "UncategorizedToolResult": (),
        "ReadResult": ("path", "status"),
        "WriteResult": ("path", "content"),
        "EditResult": ("path",),
        "WebSearchResult": ("query", "duration_sec"),
        "WebFetchResult": ("url", "code", "duration_sec", "size"),
        "AgentSpawnResult": (
            "agent_id",
            "agent_kind",
            "prompt",
            "model",
            "state",
            "tokens",
            "duration_sec",
            "tool_calls",
            "output_file",
        ),
    },
)
"""Maps each result class to the fields it copies from a former record of the same name."""

FORMER_RESULT_KINDS: Final = frozenset(
    {*RENAMED, "WebSearchResults", *_FORMER_RESULT_RECORDS},
)
"""Every class name a tool-call record was once stored under. Some, such as
``ReadResult``, are result classes now."""

RENAMED_KINDS: Final = MappingProxyType[str, str](
    dict.fromkeys(sorted(FORMER_RESULT_KINDS), "ToolMessage"),
)
"""Maps each former record kind to its kind now, which is always ``ToolMessage``."""


def renamed_records() -> Mapping[str, str]:
    """Return each record class's former name, mapped to the name that replaces it.

    A store holds records under either, so whoever reads stored names takes both
    (:func:`stored_names`). Until a class bears its new name nothing is stored
    under it, and a tag naming it is read as any unknown tag is.

    Returns:
      renamed: Former name to new name, each written out, so the keys stay as
        written when the classes are renamed.

    """
    return MappingProxyType(
        {
            "AgentToAgentMessage": "AgentMessage",
            "ToolResult": "ToolMessage",
            "TurnContext": "SessionContext",
        },
    )


def stored_names(name: str) -> tuple[str, ...]:
    """Return every name the record class called ``name`` is stored under.

    Args:
      name: A record class's name, the former or the new one.

    Returns:
      names: ``name`` first, then the names :func:`renamed_records` pairs it with.

    """
    return (
        name,
        *(old for old, new in renamed_records().items() if new == name),
        *(new for old, new in renamed_records().items() if old == name),
    )


def first_name(name: str) -> str:
    """Return the name the record class called ``name`` was first stored under.

    Args:
      name: A record class's name, the former or the new one.

    Returns:
      first: Its former name when it has one, else ``name``: what a key derived
        from a class's name before it was renamed still holds.

    """
    return next(
        (old for old, new in renamed_records().items() if new == name),
        name,
    )


FORMER_TAPE_KINDS: Final = frozenset(
    {
        "history",
        "context_splice",
        "context_override",
        "context_clear",
        "clear",
        "update",
        "persistent_agent",
    },
)
"""Each ``kind`` of an old sagent log's records that built its tape."""


def former_kind(result: ToolMessage) -> str:
    """Return the legacy record class name for ``result``.

    Each tool's call was once stored as a record class of its own. A store or a
    report that groups calls by kind still uses those names, so a legacy row and
    a new one of the same tool are grouped together.

    Args:
      result: The record of a tool call.

    Returns:
      kind: The class name of ``result.result``, such as ``BashResult``. When
        that is ``None`` or an ``UncategorizedToolResult``: ``ToolErrorResult``
        for a failed call, ``UncategorizedToolResult`` for a done call, and
        ``ToolMessage`` for a pending or cancelled one.

    """
    tool_result = result.result
    if tool_result is not None and not isinstance(
        tool_result,
        UncategorizedToolResult,
    ):
        return type(tool_result).__name__
    if result.state == "failed":
        return "ToolErrorResult"
    return "UncategorizedToolResult" if result.state == "done" else "ToolMessage"


# Every journal, row, and document stores a record through this pair, so each
# names its classes the same way: a writer left to call ``to_plain`` itself wrote
# import paths, and its rows outlived the module the classes moved out of.
def stored_plain(
    record: object,
    *,
    classes: Classes = MappingProxyType({}),
) -> dict[str, MutablePlainTree]:
    """Return ``record`` as stored plain data, which :func:`stored_record` reads.

    Args:
      record: A record, or any class ``classes`` names.
      classes: The classes a tag may name, by the name it writes; empty for
        :func:`record_classes`.

    Returns:
      plain: The record, each class tagged by its name alone and each ``JSON``
        field the plain object the provider sent.

    """
    plain = to_plain(
        record,
        classes=classes or record_classes(),
        sort=False,  # pragma: no mutate[truthiness]
        default_concrete_types=CONCRETE_TYPES,
    )
    assert isinstance(plain, dict)
    _each_stamped(plain, _spell_as_written, _stamped_tags())
    return plain


@overload
def stored_record(
    plain: object,
    *,
    classes: Classes = ...,
) -> SessionRecord: ...


@overload
def stored_record[R](
    plain: object,
    target: type[R],
    *,
    classes: Classes = ...,
) -> R: ...


# A union alias is no ``type``: its reader narrows the result to its members.
@overload
def stored_record(
    plain: object,
    target: object,
    *,
    classes: Classes = ...,
) -> object: ...


def stored_record(
    plain: object,
    target: object = SessionRecord,
    *,
    classes: Classes = MappingProxyType({}),
) -> object:
    """Return the record :func:`stored_plain` wrote as ``plain``.

    Args:
      plain: Stored plain data, frozen or not.
      target: The record class, or a union of them; any session record by
        default.
      classes: The classes a tag may name, as :func:`stored_plain` took them.

    Returns:
      record: The record, its ``JSON`` fields frozen.

    Raises:
      ReadError: ``plain`` does not read as ``target``.

    """
    current = current_plain(plain)
    _each_stamped(current, _keep_spelling, _stamped_tags())
    try:
        return from_plain(
            current,
            target,
            classes=classes or record_classes(),
            default_concrete_types=CONCRETE_TYPES,
        )
    # Treekle sets each field a tagged record holds, so one its class lacks fails
    # on the class's slots; it does not read, as a field of no tagged record.
    except AttributeError as error:
        raise ReadError(str(error)) from error


# ``WebSearchResults`` was renamed ``WebSearchResult``, which until then named one
# of its hits. So ``WebSearchResult`` means a hit only inside a ``WebSearchResults``.
def current_plain(plain: object, names: Mapping[str, str] | None = None) -> object:
    """Return stored plain data in the shape :func:`stored_plain` writes now.

    :func:`stored_record` applies this itself. A reader that edits the plain
    data before decoding it, as ``host.wire`` does, applies it first.

    Args:
      plain: Stored plain data, frozen or not, written by any version.
      names: Maps each former class tag to its tag now; ``None`` for the tags
        this module renamed.

    Returns:
      plain: ``plain`` with each former class tag renamed and each former
        tool-call record converted to a ``ToolMessage``.

    """
    if names is None:
        names = _tags_now()
    if isinstance(plain, Mapping):
        fields = cast("Mapping[str, object]", plain)
        tag = fields.get("py/object")
        inner = names
        if tag == "WebSearchResults":
            tag, inner = "WebSearchResult", {**names, "WebSearchResult": "WebSearchHit"}
        elif isinstance(tag, str):
            tag = names.get(tag, tag)
        # ``py/dict`` holds plain data, such as a call's arguments; nothing in it
        # is a record, whatever its keys.
        current = {
            key: tag
            if key == "py/object"
            else value
            if key == "py/dict"
            else current_plain(value, inner)
            for key, value in fields.items()
        }
        # A ``FunctionResult`` held its tool's name, which ``ToolCall.name`` has.
        if fields.get("py/object") == "FunctionResult":
            _ = current.pop("name", None)
        # A former record has a ``call_id``; a result class of the same name does
        # not. A ``ToolMessage`` with no ``state`` is the former placeholder record.
        if isinstance(tag, str) and (
            (tag in _FORMER_RESULT_RECORDS and "call_id" in current)
            or (tag == "ToolMessage" and "state" not in current)
        ):
            return _converted_record(tag, current)
        # ``result`` was once the bare exit code of a tool with no result class.
        exit_code = current.get("result")
        if tag == "ToolMessage" and isinstance(exit_code, int):
            current["result"] = {
                "py/object": "UncategorizedToolResult",
                "exit_code": exit_code,
            }
        held = current.get("extra")
        if tag == "Thinking" and isinstance(held, Mapping):
            current |= _reasoning_id_moved_out_of_extra(
                cast("Mapping[str, object]", held),
                stated="reasoning_id" in current,
            )
        return current
    # A store returns its rows frozen, so a list may arrive as a tuple.
    if isinstance(plain, (list, tuple)):
        items = cast("Sequence[object]", plain)
        return [current_plain(value, names) for value in items]
    return plain


# A time was once the source's own text. One spelled as ``datetime.isoformat`` writes
# it reads as it is. Any other keeps its spelling in ``extra``, which a converter
# writes back in the time's place; one that is no time is kept there alone. An
# ``UncategorizedRecord`` has no ``extra``, so its payload keeps it.
def _keep_spelling(record: dict[str, object]) -> None:
    """Hold one stored record's time as a ``datetime`` reads it, in place."""
    spelled = record.get("timestamp")
    if not isinstance(spelled, str):
        return
    when = parsed_timestamp(spelled)
    stored = None if when is None else when.isoformat()
    if stored == spelled:
        return
    held = _beside(record)
    kept = record.get(held)
    beside = (
        dict(cast("Mapping[str, object]", kept)) if isinstance(kept, Mapping) else {}
    )
    record.update(
        {"timestamp": stored, held: {**beside, "$timestamp_as_written": spelled}},
    )


# Two writers once kept OpenAI's reasoning id in ``extra``: sagent's OpenAI provider as
# ``reasoning_id``, and the codex reader as the reasoning item's own ``id``.
def _reasoning_id_moved_out_of_extra(
    held: Mapping[str, object],
    *,
    stated: bool,
) -> dict[str, object]:
    """Return the ``Thinking`` fields that move a stored ``extra``'s reasoning id out."""
    residual = dict(held)
    if "reasoning_id" in residual:
        moved = residual.pop("reasoning_id")
        return (
            {"extra": residual}
            if stated
            else {"reasoning_id": moved, "extra": residual}
        )
    if stated or residual.get("type") != "reasoning":
        return {}
    if "id" not in residual:
        return {"reasoning_id": "", "extra": residual | {"$id_absent": True}}
    if not isinstance(residual["id"], str):
        return {}
    return {"reasoning_id": residual.pop("id"), "extra": residual}


# ``ToolMessage.text()`` of the converted record must equal the text the model read
# from the former one, to the character.
def _converted_record(tag: str, former: Mapping[str, object]) -> dict[str, object]:
    """Return the plain ``ToolMessage`` for ``former``, a record stored as class ``tag``."""
    stub = former.get("stub")
    state = (
        stub
        if stub in ("pending", "cancelled")
        else "failed"
        if tag == "ToolErrorResult"
        else "done"
    )
    exit_code = former.get("exit_code") if tag == "BashResult" else None
    stdout: list[object] = []
    stderr: list[object] = []
    output_fields = {"content"}
    # The fields that held typed items, each with its items' class, in the order
    # the items follow the text on ``stdout``.
    item_fields = [("attachments", "Attachment")]
    if tag == "BashResult":
        stdout.append(former.get("stdout"))
        stderr.append(former.get("stderr"))
        output_fields = {"stdout", "stderr", "exit_code"}
    elif tag in ("WriteResult", "EditResult"):
        verb = "Wrote" if tag == "WriteResult" else "Edited"
        stdout.append(f"{verb} {former.get('path') or 'the file'}.")
        item_fields.insert(0, ("edits", "Splice"))
        output_fields = {"edits"}
    elif tag == "WebSearchResult":
        hits = [
            cast("Mapping[str, object]", hit)
            for hit in _tuple_items(former.get("content"))
        ]
        stdout.append(
            "\n\n".join(
                "\n".join(
                    str(part)
                    for part in (hit.get("title"), hit.get("url"), hit.get("snippet"))
                    if part is not None
                )
                for hit in hits
            ),
        )
        item_fields.insert(0, ("content", "WebSearchHit"))
    elif tag == "ToolErrorResult":
        stderr.append(former.get("content"))
    else:
        stdout.append(former.get("content"))
        # ``ranges`` held the lines a read returned, each as ``(start, count)``.
        stdout.extend(
            {"py/object": "LineSpan", "start": start, "count": count}
            for start, count in map(_tuple_items, _tuple_items(former.get("ranges")))
        )
        output_fields = {"content", "ranges"}
    # A field declared ``tuple[X, ...]`` stored each ``X`` without a class tag.
    stdout.extend(
        {"py/object": item_class, **cast("Mapping[str, object]", item)}
        for item_field, item_class in item_fields
        for item in _tuple_items(former.get(item_field))
    )
    record_fields = ("context_id", "timestamp", "call_id", "wire_id", "shown")
    result_fields = _RESULT_FIELDS.get(tag, ()) if state == "done" else ()
    converted_fields = {
        "py/object",
        "stub",
        "is_error",
        "attachments",
        "extra",
        *record_fields,
        *output_fields,
        *result_fields,
    }
    # A former field with no place in the record is kept in ``extra``, as JSON.
    extra = {
        **{
            key: _json_value(value)
            for key, value in former.items()
            if key not in converted_fields and value is not None
        },
        **cast("Mapping[str, object]", former.get("extra") or {}),
    }
    marked_error = former.get("is_error") is True
    result: object = None
    if state == "done":
        result = {
            "py/object": tag if tag in _RESULT_FIELDS else "UncategorizedToolResult",
            **{
                key: former[key] for key in result_fields if former.get(key) is not None
            },
            **({"exit_code": int(marked_error)} if tag != "BashResult" else {}),
            **({"exit_code": exit_code} if isinstance(exit_code, int) else {}),
        }
    # ``failed`` and ``cancelled`` are errors already; a pending placeholder marked
    # an error needs a result to stay one.
    elif marked_error and state == "pending":
        result = {"py/object": "UncategorizedToolResult", "exit_code": 1}
    # A ``BashResult``'s exit code is the command's and never makes the call an
    # error, so a shell record marked an error becomes a failed call. It keeps its
    # result, which holds the command.
    if tag == "BashResult" and marked_error and state == "done":
        state = "failed"
    return {
        "py/object": "ToolMessage",
        **{key: former[key] for key in record_fields if key in former},
        "state": state,
        "stdout": [item for item in stdout if item not in (None, "")],
        "stderr": [item for item in stderr if item not in (None, "")],
        **({} if result is None else {"result": result}),
        **({"extra": extra} if extra else {}),
    }


def _tuple_items(stored: object) -> list[object]:
    """Return the items of a stored tuple; ``[]`` for a field that was not stored."""
    if isinstance(stored, Mapping):
        stored = cast("Mapping[str, object]", stored).get("py/tuple")
    if isinstance(stored, (list, tuple)):
        return list(cast("Sequence[object]", stored))
    return []


def _json_value(stored: object) -> object:
    """Return ``stored`` as JSON: each stored tuple becomes a list."""
    if isinstance(stored, Mapping):
        fields = cast("Mapping[str, object]", stored)
        if set(fields) == {"py/tuple"}:
            return [_json_value(item) for item in _tuple_items(fields)]
        return {key: _json_value(value) for key, value in fields.items()}
    if isinstance(stored, (list, tuple)):
        return [_json_value(item) for item in cast("Sequence[object]", stored)]
    return stored


# A time is stored as its source spelled it, so a stored record is the bytes it
# always was; read, it is the time, with the spelling kept beside it.
def _each_stamped(
    plain: object,
    change: Callable[[dict[str, object]], None],
    stamped: frozenset[str],
) -> None:
    """Apply ``change`` to each record in ``plain`` tagged one of ``stamped``, in place."""
    if isinstance(plain, (list, tuple)):
        for item in cast("Sequence[object]", plain):
            _each_stamped(item, change, stamped)
        return
    if not isinstance(plain, dict):
        return
    node = cast("dict[str, object]", plain)
    tag = node.get("py/object")
    for key, value in node.items():
        # ``py/dict`` holds plain data, such as a call's arguments; nothing in it
        # is a record, whatever its keys.
        if key != "py/dict":
            _each_stamped(value, change, stamped)
    # A line no writer of ours wrote may tag a record with anything.
    if isinstance(tag, str) and tag in stamped:
        change(node)
    if tag == "ContextClear":
        for held in _untagged(node.get("history")):
            change(held)


# A record with no time is stored with no ``timestamp``, so the spelling goes where
# a record's time is stored: after its tag and ``context_id``.
def _spell_as_written(record: dict[str, object]) -> None:
    """Store one record's time as its source spelled it, in place."""
    held = cast("dict[str, object]", record[_beside(record)])
    if "$timestamp_as_written" not in held:
        return
    stored = record.get("timestamp")
    when = parsed_timestamp(stored) if isinstance(stored, str) else None
    written = timestamp_as_written(when, held)
    del held["$timestamp_as_written"]
    if written == stored:
        return
    fields = {key: value for key, value in record.items() if key != "timestamp"}
    lead = {
        key: fields.pop(key) for key in ("py/object", "context_id") if key in fields
    }
    record.clear()
    record.update(lead | {"timestamp": written} | fields)


# An ``UncategorizedRecord`` has no ``extra``: its payload keeps the spelling.
def _beside(record: Mapping[str, object]) -> str:
    """Return the field of a stored record that keeps its time's spelling."""
    return "payload" if record.get("py/object") == "UncategorizedRecord" else "extra"


# A window's first records are stored as a ``SessionRecord`` each, which leaves a
# ``UserMessage``, the first of that union, with no tag.
def _untagged(history: object) -> list[dict[str, object]]:
    """Return the records of a stored ``ContextClear.history`` that carry no tag."""
    if not isinstance(history, (list, tuple)):
        return []
    return [
        cast("dict[str, object]", item)
        for item in cast("Sequence[object]", history)
        if isinstance(item, dict) and "py/object" not in item
    ]


def _told(late: ToolMessage) -> UserMessage:
    """Return a call's late message as the user-side text a provider is sent."""
    text = late.text()
    if late.state != "pending":
        text = f"[call {late.call_id} {late.state}]\n{text}"
    return UserMessage(
        timestamp=late.timestamp,
        content=text,
        attachments=tuple(i for i in late.stdout if isinstance(i, Attachment)),
    )


# Read per stored line, so it is worked out once per process.
@cache
def _tags_now() -> Mapping[str, str]:
    """Return each former class tag's tag now."""
    return MappingProxyType[str, str](
        {
            **RENAMED,
            **{
                old: new
                for old, new in renamed_records().items()
                if new in record_classes()
            },
        },
    )


# Read per record stored and read, so it is worked out once per process.
@cache
def _stamped_tags() -> frozenset[str]:
    """Return the tag of each stored record that says when it happened, in ``timestamp``."""
    return frozenset(
        name
        for name, kind in record_classes().items()
        if is_dataclass(kind) and any(f.name == "timestamp" for f in fields(kind))
    )
