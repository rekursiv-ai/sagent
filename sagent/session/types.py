"""Records of an agent session, one per act.

What the model saw is the last ``ContextClear`` and every record after it.
``None`` means the source did not say; ``extra`` keeps what no field names, so
a claude or codex transcript converts here and back without loss.

Typical usage example:

  records: list[SessionRecord] = [
      UserMessage(content="fix the bug"),
      ToolCall(call_id="1", name="Read", arguments={"path": "a.py"}),
      FileReadResult(call_id="1", path="a.py", content="..."),
      AssistantMessage(content="fixed"),
  ]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, cast, override

from treekle import PlainTree, immutable

from sagent.model.capability import SummaryKind, ThinkingEffort


__all__ = [
    "CONCRETE_TYPES",
    "AgentStatusResult",
    "AgentToAgentMessage",
    "AnyToolResult",
    "AssistantMessage",
    "Attachment",
    "ContextClear",
    "ContextCompaction",
    "ContextState",
    "FileEditResult",
    "FileReadResult",
    "FileWriteResult",
    "IncompleteRecord",
    "SessionRecord",
    "ShellCommandResult",
    "Splice",
    "SystemMessage",
    "Thinking",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "TranscriptItem",
    "TurnContext",
    "UncategorizedRecord",
    "UncategorizedToolResult",
    "UserMessage",
    "WebFetchResult",
    "WebSearchResult",
    "WebSearchResults",
    "record_classes",
]


CONCRETE_TYPES: Final = MappingProxyType[type, type](
    {Mapping: MappingProxyType, Sequence: tuple},
)
"""How records are stored: ``to_plain`` given this and the record's class as its
target, and ``from_plain`` given this, write each JSON field as the plain object
the provider sent, and read it back frozen."""


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
            f"Working directory: {self.cwd}" if self.cwd is not None else "",
            self.environment if self.environment is not None else "",
            f"Model: {self.model}" if self.model is not None else "",
            f"Tools you may call: {', '.join(self.tools) or 'none'}"
            if self.tools is not None
            else "",
            f"Current time: {_local(self.timestamp)}"
            if self.timestamp is not None
            else "",
            self.agents_md if self.agents_md is not None else "",
        )
        return "\n".join(line for line in lines if line)


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

    attachments: tuple[Attachment, ...] = ()
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

    def __post_init__(self) -> None:
        _freeze(self, "extra")

    def text(self) -> str:
        """Return what the model reads for this result."""
        return ""


@dataclass(frozen=True, slots=True, kw_only=True)
class UncategorizedToolResult(ToolResult):
    """A tool result no other type describes."""

    content: str | None = None

    @override
    def text(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ShellCommandResult(ToolResult):
    """A finished shell command."""

    command: tuple[str, ...] | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None

    @override
    def text(self) -> str:
        status = f"\n[exit {self.exit_code}]" if self.exit_code not in (None, 0) else ""
        return f"{self.stdout}{self.stderr}{status}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FileReadResult(ToolResult):
    """Content read from a file."""

    path: str | None = None
    content: str | None = None
    ranges: tuple[tuple[int | None, int | None], ...] = ()
    """Lines returned as one-based ``(start, count)``; ``None`` is open-ended; empty is all."""

    @override
    def text(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FileWriteResult(ToolResult):
    """Content written to a file."""

    path: str | None = None
    content: str | None = None

    @override
    def text(self) -> str:
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
class FileEditResult(ToolResult):
    """Replacements made in a file; empty when only the path is known."""

    path: str | None = None
    edits: tuple[Splice, ...] = ()

    @override
    def text(self) -> str:
        return f"Edited {self.path or 'the file'}."


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchResult:
    """One web search hit."""

    url: str | None = None
    title: str | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearchResults(ToolResult):
    """The hits a web search returned."""

    query: str | None = None
    duration_sec: float | None = None
    content: tuple[WebSearchResult, ...] = ()

    @override
    def text(self) -> str:
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
    def text(self) -> str:
        return self.content or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStatusResult(ToolResult):
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
    def text(self) -> str:
        return self.content or ""


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
class TokenUsage:
    """Token accounting and rate limits a provider reported."""

    context_id: int | None = None
    timestamp: str | None = None
    info: Mapping[str, PlainTree] = field(default_factory=_no_json)
    rate_limits: Mapping[str, PlainTree] = field(default_factory=_no_json)
    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)

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
    ShellCommandResult
    | FileReadResult
    | FileWriteResult
    | FileEditResult
    | WebSearchResults
    | WebFetchResult
    | AgentStatusResult
    | UncategorizedToolResult
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
            UncategorizedToolResult,
            ShellCommandResult,
            FileReadResult,
            FileWriteResult,
            Splice,
            FileEditResult,
            WebSearchResult,
            WebSearchResults,
            WebFetchResult,
            AgentStatusResult,
            SystemMessage,
            TokenUsage,
            ContextState,
            ContextCompaction,
            ContextClear,
            AgentToAgentMessage,
            UncategorizedRecord,
            IncompleteRecord,
        )
    }
