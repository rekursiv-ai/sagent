"""Records of an agent session, one per act.

What the model saw is the last ``ContextClear`` and every record after it.
``None`` means the source did not say; ``extra`` keeps what no field names, so
a claude or codex transcript converts here and back without loss.

Typical usage example:

  records: list[SessionRecord] = [
      UserMessage(content="fix the bug"),
      ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
      ReadResult(call_id="1", path="a.py", content="..."),
      AssistantMessage(content="fixed"),
  ]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Self,
    cast,
    get_args,
    overload,
    override,
)

from treekle import PlainTree, from_plain, immutable, to_plain

from sagent.model.capability import SummaryKind, ThinkingEffort


if TYPE_CHECKING:
    from treekle import Classes, MutablePlainTree


__all__ = [
    "CONCRETE_TYPES",
    "RENAMED",
    "RENAMED_KINDS",
    "AgentSelfResult",
    "AgentSendResult",
    "AgentSpawnResult",
    "AgentToAgentMessage",
    "AnyToolResult",
    "AssistantMessage",
    "Attachment",
    "BackgroundTaskResult",
    "BashResult",
    "ContextClear",
    "ContextCompaction",
    "ContextState",
    "EditResult",
    "FunctionResult",
    "GlobResult",
    "GrepResult",
    "IncompleteRecord",
    "ListResult",
    "McpResult",
    "MemoryResult",
    "PaperAuthorResult",
    "PaperDetailsResult",
    "PaperFetchResult",
    "PaperSearchResult",
    "PlayAudioResult",
    "ReadResult",
    "RemoteToolResult",
    "RequestMeasurement",
    "SessionRecord",
    "SessionResult",
    "SkillResult",
    "Splice",
    "SystemMessage",
    "Thinking",
    "TokenUsage",
    "ToolCall",
    "ToolErrorResult",
    "ToolResult",
    "TranscriptItem",
    "TurnContext",
    "UncategorizedRecord",
    "UncategorizedToolResult",
    "UserMessage",
    "WebBrowserResult",
    "WebFetchResult",
    "WebSearchHit",
    "WebSearchResult",
    "WriteResult",
    "record_classes",
    "stored_plain",
    "stored_record",
    "tool_result_classes",
]


CONCRETE_TYPES: Final = MappingProxyType[type, type](
    {Mapping: MappingProxyType, Sequence: tuple},
)
"""How a record's JSON fields are stored: written as the plain objects the
provider sent, and read back frozen. :func:`stored_plain` and
:func:`stored_record` apply it."""


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
class TurnContext:
    """Settings in force for every record after it, until the next one."""

    context_id: int | None = None
    timestamp: str | None = None
    cwd: str | None = None
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
            f"Working directory: {self.cwd}" if self.cwd is not None else None,
            self.environment,
            f"Model: {self.model}" if self.model is not None else None,
            f"Tools you may call: {', '.join(self.tools) or 'none'}"
            if self.tools is not None
            else None,
            f"Current time: {_local(self.timestamp)}"
            if self.timestamp is not None
            else None,
            self.agents_md,
        )
        return "\n".join(filter(None, lines))


# The model greets by the time of day where the user is, so the offset alone is not
# enough: the zone's name says which day and hour it is there.
def _local(timestamp: str) -> str:
    """Return an ISO 8601 time as ``Mon 2026-10-26 17:44 -07:00 (PDT)``, here; else as is."""
    try:
        when = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if when.tzinfo is None:
        return timestamp
    here = when.astimezone()
    offset = here.strftime("%z")
    return f"{here:%a %Y-%m-%d %H:%M} {offset[:3]}:{offset[3:]} ({here.tzname()})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Attachment:
    """Binary content and its media type."""

    mime_descriptor: str
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class UserMessage:
    """What the user said."""

    context_id: int | None = None
    timestamp: str | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantMessage:
    """What the model said."""

    context_id: int | None = None
    timestamp: str | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class Thinking:
    """The model's reasoning: readable, sealed, or summarized."""

    context_id: int | None = None
    timestamp: str | None = None
    content: str | None = None
    encrypted: str | None = None
    summary: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """A tool invocation the model asked for."""

    context_id: int | None = None
    timestamp: str | None = None
    call_id: str
    name: str
    arguments: Mapping[str, PlainTree] = field(default_factory=_no_json)
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "arguments", "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult:
    """What a tool call produced; subclasses say what the tool did."""

    context_id: int | None = None
    timestamp: str | None = None
    call_id: str
    is_error: bool | None = None
    stub: (
        Literal[
            "pending",
            "cancelled",
        ]
        | None
    ) = None
    """What a stand-in result stands for: a call still running, or killed; ``None`` for a real one."""

    shown: str | None = None
    """What the model reads in place of :meth:`body`: a failure's message, a
    stand-in's note, or what a prune, compaction, or restatement left; the type
    and its fields stay."""

    attachments: tuple[Attachment, ...] = ()
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")

    def text(self) -> str:
        """Return what the model reads for this result: :attr:`shown`, else :meth:`body`."""
        return self.shown if self.shown is not None else self.body()

    def body(self) -> str:
        """Return what the model reads of this result's own fields."""
        return ""

    def restated(self, text: str) -> Self:
        """Return this result reading ``text``, as a prune or compaction leaves it.

        Args:
          text: What the model reads instead; its attachments go with the rest.

        Returns:
          result: Of this type, with these fields, showing ``text``.

        """
        return replace(self, shown=text, attachments=())


@dataclass(frozen=True, slots=True, kw_only=True)
class UncategorizedToolResult(ToolResult):
    """A vendor tool's result that no sagent tool's type describes."""

    content: str | None = None

    @override
    def body(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolErrorResult(ToolResult):
    """A call no tool answered: it raised, named no tool, or was refused."""

    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class BashResult(ToolResult):
    """A finished shell command."""

    command: tuple[str, ...] | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None

    @override
    def body(self) -> str:
        status = f"\n[exit {self.exit_code}]" if self.exit_code not in (None, 0) else ""
        return f"{self.stdout}{self.stderr}{status}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadResult(ToolResult):
    """Content read from a file."""

    path: str | None = None
    content: str | None = None
    ranges: tuple[tuple[int | None, int | None], ...] = ()
    """Lines returned as one-based ``(start, count)``; ``None`` is open-ended; empty is all."""

    status: (
        Literal["unchanged", "binary", "undecodable", "empty", "past_end"] | None
    ) = None
    """Why no lines came back, which :attr:`shown` says; ``None`` when they did."""

    @override
    def body(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteResult(ToolResult):
    """Content written to a file."""

    path: str | None = None
    content: str | None = None

    @override
    def body(self) -> str:
        return f"Wrote {self.path or 'the file'}."


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
class EditResult(ToolResult):
    """Replacements made in a file; empty when only the path is known."""

    path: str | None = None
    edits: tuple[Splice, ...] = ()

    @override
    def body(self) -> str:
        return f"Edited {self.path or 'the file'}."


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchHit:
    """One web search hit."""

    url: str | None = None
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchResult(ToolResult):
    """The hits a web search returned."""

    query: str | None = None
    duration_sec: float | None = None
    content: tuple[WebSearchHit, ...] = ()

    @override
    def body(self) -> str:
        return "\n\n".join(
            "\n".join(
                part for part in (hit.title, hit.url, hit.snippet) if part is not None
            )
            for hit in self.content
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class WebFetchResult(ToolResult):
    """Content fetched from a URL."""

    url: str | None = None
    content: str | None = None
    code: int | None = None
    duration_sec: float | None = None
    size: int | None = None

    @override
    def body(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class WebBrowserResult(ToolResult):
    """A browser action and its inspection or interaction output."""

    action: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSpawnResult(ToolResult):
    """State or final output of a subagent."""

    agent_id: str | None = None
    agent_kind: str | None = None
    prompt: str | None = None
    content: str | None = None
    model: str | None = None
    state: str | None = None
    tokens: int | None = None
    duration_sec: float | None = None
    tool_calls: int | None = None
    output_file: str | None = None

    @override
    def body(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class GrepResult(ToolResult):
    """A Grep: the pattern searched for, and what it found."""

    pattern: str = ""
    path: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class GlobResult(ToolResult):
    """A Glob: the pattern matched, and the paths it found."""

    pattern: str = ""
    path: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class ListResult(ToolResult):
    """A List: the directory listed, and its entries."""

    path: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillResult(ToolResult):
    """A Skill: the skill loaded, and its body."""

    skill: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSendResult(ToolResult):
    """An AgentSend: whom it was sent to, and how it went."""

    to: str = ""
    delay_sec: float | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class BackgroundTaskResult(ToolResult):
    """A BackgroundTask: the operation on a job, and what it said."""

    operation: str = ""
    job_id: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSelfResult(ToolResult):
    """An AgentSelf: what it changed or reported of the agent."""

    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResult(ToolResult):
    """A Session: the action on a session, and what it read or did."""

    action: str = ""
    agent: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryResult(ToolResult):
    """A Memory: the memory saved or forgotten."""

    action: str = ""
    name: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayAudioResult(ToolResult):
    """A PlayAudio: the sound played."""

    path: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSearchResult(ToolResult):
    """A PaperSearch: the query, and the papers it found."""

    query: str = ""
    source: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperDetailsResult(ToolResult):
    """A PaperDetails: papers' metadata, or one paper's references or citations."""

    operation: str | None = None
    source: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperAuthorResult(ToolResult):
    """A PaperAuthor: authors found, their metadata, or their papers."""

    operation: str | None = None
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperFetchResult(ToolResult):
    """A PaperFetch: where each paper was saved."""

    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteToolResult(ToolResult):
    """A RemoteTool: the endpoint asked, and what it said."""

    endpoint: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class McpResult(ToolResult):
    """An MCP server's tool: the server and tool, and what it returned."""

    server: str = ""
    tool: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionResult(ToolResult):
    """A function tool: its name, and what it returned."""

    name: str = ""
    content: str = ""

    @override
    def body(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemMessage:
    """Instructions or status from the harness, such as a reminder."""

    context_id: int | None = None
    timestamp: str | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    subtype: str | None = None
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
    timestamp: str | None = None
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
    timestamp: str | None = None
    kind: str = ""
    content: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextCompaction:
    """A compaction happened; the ``ContextClear`` after it holds the result."""

    context_id: int | None = None
    timestamp: str | None = None
    summary: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")

    """Provider bookkeeping, and the instructions the compaction was given."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextClear:
    """A context window starts here, with exactly this content."""

    context_id: int | None = None
    timestamp: str | None = None
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
class AgentToAgentMessage:
    """A message between agents."""

    context_id: int | None = None
    timestamp: str | None = None
    content: str | None = None
    attachments: tuple[Attachment, ...] = ()
    sender: str | None = None
    recipient: str | None = None
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")


@dataclass(frozen=True, slots=True, kw_only=True)
class UncategorizedRecord:
    """A source record no other type describes."""

    context_id: int | None = None
    timestamp: str | None = None
    kind: str
    payload: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "payload")


@dataclass(frozen=True, slots=True, kw_only=True)
class IncompleteRecord:
    """A source line that could not be parsed."""

    text: str


type AnyToolResult = (
    BashResult
    | ReadResult
    | WriteResult
    | EditResult
    | WebSearchResult
    | WebFetchResult
    | WebBrowserResult
    | AgentSpawnResult
    | GrepResult
    | GlobResult
    | ListResult
    | SkillResult
    | AgentSendResult
    | BackgroundTaskResult
    | AgentSelfResult
    | SessionResult
    | MemoryResult
    | PlayAudioResult
    | PaperSearchResult
    | PaperDetailsResult
    | PaperAuthorResult
    | PaperFetchResult
    | RemoteToolResult
    | McpResult
    | FunctionResult
    | UncategorizedToolResult
    | ToolErrorResult
    # A runtime stand-in: a call still running, or killed, says only ``shown``.
    | ToolResult
)
type TranscriptItem = (
    UserMessage | AssistantMessage | Thinking | ToolCall | AnyToolResult
)
type SessionRecord = (
    TranscriptItem
    | SystemMessage
    | TokenUsage
    | ContextState
    | ContextCompaction
    | ContextClear
    | AgentToAgentMessage
    | TurnContext
    | UncategorizedRecord
    | IncompleteRecord
)


def tool_result_classes() -> tuple[type[AnyToolResult], ...]:
    """Return each class :data:`AnyToolResult` names, in its order.

    Returns:
      classes: Every tool result type, the bare runtime stand-in included.

    """
    union = AnyToolResult.__value__  # pyright: ignore[reportAny] -- Typeshed types an alias's value as Any.
    return cast("tuple[type[AnyToolResult], ...]", get_args(union))


def record_classes() -> dict[str, type]:
    """Return each dataclass a session holds.

    Returns:
      classes: Each class, by the name its stored form carries.

    """
    return {
        kind.__name__: kind
        for kind in (
            TurnContext,
            Attachment,
            UserMessage,
            AssistantMessage,
            Thinking,
            ToolCall,
            ToolResult,
            GrepResult,
            GlobResult,
            ListResult,
            SkillResult,
            AgentSendResult,
            BackgroundTaskResult,
            AgentSelfResult,
            SessionResult,
            MemoryResult,
            PlayAudioResult,
            PaperSearchResult,
            PaperDetailsResult,
            PaperAuthorResult,
            PaperFetchResult,
            RemoteToolResult,
            McpResult,
            FunctionResult,
            UncategorizedToolResult,
            ToolErrorResult,
            BashResult,
            ReadResult,
            WriteResult,
            Splice,
            EditResult,
            WebSearchHit,
            WebSearchResult,
            WebFetchResult,
            WebBrowserResult,
            AgentSpawnResult,
            SystemMessage,
            RequestMeasurement,
            TokenUsage,
            ContextState,
            ContextCompaction,
            ContextClear,
            AgentToAgentMessage,
            UncategorizedRecord,
            IncompleteRecord,
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
    },
)
"""Each record type's former stored name, to its name now."""

RENAMED_KINDS: Final = MappingProxyType[str, str](
    {**RENAMED, "WebSearchResults": "WebSearchResult"},
)
"""Each top-level record's former name, to its name now: as :data:`RENAMED`, and the
web search, whose old name nested only inside it means a hit now."""


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
        sort=False,  # pragma: no mutate -- treekle reads ``sort`` by truth: None is False.
        default_concrete_types=CONCRETE_TYPES,
    )
    assert isinstance(plain, dict)
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
    return from_plain(
        _renamed(plain),
        target,
        classes=classes or record_classes(),
        default_concrete_types=CONCRETE_TYPES,
    )


# ``WebSearchResults`` is now ``WebSearchResult``, which named one of its hits; a hit is
# renamed only under an old ``WebSearchResults``, the one place it nested.
def _renamed(plain: object, names: Mapping[str, str] = RENAMED) -> object:
    """Return ``plain`` with each former ``py/object`` name read as its name now."""
    if isinstance(plain, Mapping):
        fields = cast("Mapping[str, object]", plain)
        tag = fields.get("py/object")
        inner = names
        if tag == "WebSearchResults":
            tag, inner = "WebSearchResult", {**names, "WebSearchResult": "WebSearchHit"}
        elif isinstance(tag, str):
            tag = names.get(tag, tag)
        return {
            key: tag if key == "py/object" else _renamed(value, inner)
            for key, value in fields.items()
        }
    if isinstance(plain, list):
        return [_renamed(value, names) for value in cast("list[object]", plain)]
    return plain
