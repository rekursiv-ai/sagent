"""Compaction: summarize conversation when context fills up.

- Structured 9-section summary with <analysis> scratchpad
- NO_TOOLS preamble + trailer (tool-use prevention)
- Prompt-too-long retry with message truncation
- Continuation suppression
- Microcompaction (clear old clearable tool results)

Usage::

    from sagent.compactor import SummaryCompactor
    compactor = SummaryCompactor()
    agent = Agent(model=sonnet, compactor=compactor, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import dataclasses
import logging
import re
import time

from sagent.agent.retry import send_with_retry
from sagent.lib.compaction import CLEARED
from sagent.tools.core import read_asset, recipe_dict
from sagent.types.exceptions import PromptTooLongError
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelRequest
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)

MICROCOMPACT_KEEP_RECENT = 5


def _compactor_path(key: str) -> str:
    """Look up an asset path under the recipe's ``compactor`` section."""
    comp = recipe_dict("compactor")
    if key not in comp:
        raise FileNotFoundError(f"compactor.{key} not in recipe")
    return comp[key]


DEFAULT_COMPACT_PROMPT = read_asset(_compactor_path("full"))
DEFAULT_PARTIAL_COMPACT_PROMPT = read_asset(_compactor_path("partial"))
_NO_TOOLS_PREAMBLE = read_asset(_compactor_path("no_tools_preamble"))
_NO_TOOLS_TRAILER = read_asset(_compactor_path("no_tools_trailer"))
_SYSTEM = read_asset(_compactor_path("system")).strip()
_CONTINUATION_TEMPLATE = read_asset(_compactor_path("continuation"))

_RE_ANALYSIS = re.compile(r"<analysis>[\s\S]*?</analysis>")
_RE_SUMMARY = re.compile(r"<summary>([\s\S]*?)</summary>")

_MAX_FALLBACK_SUMMARY_CHARS = 10_000


def _entry_chars(entry: HistoryEntry) -> int:
    """Approximate character count of an entry's payload."""
    if isinstance(entry, UserMessage):
        return len(entry.text)
    if isinstance(entry, AssistantMessage):
        n = len(entry.text)
        for tc in entry.tool_calls:
            n += len(tc.name) + sum(len(str(v)) for v in tc.args.values())
        return n
    return len(entry.content)


def _groups_to_drop(
    groups: list[list[HistoryEntry]],
    error: PromptTooLongError,
    chars_per_token: int = 4,
) -> int:
    """How many leading groups to drop to cover the token gap."""
    gap = error.token_gap
    if gap is None:
        return max(1, len(groups) // 5)
    target_chars = gap * chars_per_token
    chars = 0
    for i, g in enumerate(groups):
        for entry in g:
            chars += _entry_chars(entry)
        if chars >= target_chars:
            return i + 1
    return max(1, len(groups) // 5)


class SummaryCompactor:
    """Compacts by summarizing the full conversation.

    - Structured 9-section summary with <analysis> scratchpad
    - Tool-use prevention (preamble + trailer)
    - Prompt-too-long retry (drop oldest rounds, up to N attempts)
    - Post-compaction continuation suppression

    Args:
      prompt: Full compaction prompt template.
      partial_prompt: Prompt used when ``keep_recent`` preserves a tail.
      buffer_tokens: Headroom before ``should_compact`` returns True.
      chars_per_token: Conversion factor for token-gap to char-gap math.
      max_attempts: Retries on ``PromptTooLongError`` before giving up.
      keep_recent: Default ``keep_recent`` when ``compact`` doesn't override.
      proactive: When True, the continuation resumes work autonomously.
      model: Optional model override; otherwise uses the caller's model.

    Raises:
      ValueError: ``max_attempts`` is less than 1.

    """

    def __init__(
        self,
        *,
        prompt: str = DEFAULT_COMPACT_PROMPT,
        partial_prompt: str = DEFAULT_PARTIAL_COMPACT_PROMPT,
        buffer_tokens: int = 13_000,
        chars_per_token: int = 4,
        max_attempts: int = 3,
        keep_recent: int = 0,
        proactive: bool = False,
        model: Model | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._prompt = prompt
        self._partial_prompt = partial_prompt
        self._buffer_tokens = buffer_tokens
        self._chars_per_token = chars_per_token
        self._max_attempts = max_attempts
        self._keep_recent = keep_recent
        self._proactive = proactive
        self._model = model

    @property
    def proactive(self) -> bool:
        """Whether the compactor resumes autonomously after compaction."""
        return self._proactive

    def maintain(
        self,
        history: list[HistoryEntry],
        tools: dict[str, Tool],
        *,
        last_response_time: float = 0.0,
        gap_sec: float = 3600.0,
        keep_recent: int = MICROCOMPACT_KEEP_RECENT,
    ) -> None:
        """Replace old clearable ``ToolResult`` entries with a placeholder.

        Skips when the prompt cache is likely still warm (``time.time() -
        last_response_time <= gap_sec``); microcompacting a warm-cache
        request would force re-tokenization for no real saving.

        Args:
          history: History list to mutate in place.
          tools: Tool registry; only ``supports_microcompaction`` results clear.
          last_response_time: Wall-clock seconds of the last response.
              ``0.0`` falls through the gate so first-call sessions
              still get cleared.
          gap_sec: Cache-warm threshold; default 1 hour.
          keep_recent: Number of recent clearable results preserved.

        """
        if last_response_time and (time.time() - last_response_time) <= gap_sec:
            return
        microcompact(history, tools, keep_recent=keep_recent)

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Return True when ``input_tokens`` crosses the compact threshold.

        Args:
          input_tokens: Estimated current input token count.
          max_request_tokens: Budget cap for input tokens.
          max_response_tokens: Reserved output tokens; subtracted from cap.

        Returns:
          should: True when compaction should run before the next call.

        """
        effective = max_request_tokens - max_response_tokens
        return input_tokens >= max(0, effective - self._buffer_tokens)

    async def compact(
        self,
        history: list[HistoryEntry],
        model: Model,
        transcript_path: Path | None = None,
        direction: Literal["from", "up_to"] = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[HistoryEntry]:
        """Summarize ``history`` with a structured format.

        Args:
          history: Conversation history to summarize.
          model: Fallback model when the compactor has none of its own.
          transcript_path: Optional path of the pre-compact transcript on disk.
          direction: ``"from"`` keeps the tail, ``"up_to"`` keeps the head.
          keep_recent: Override for the default ``keep_recent`` count;
              ``None`` falls back to the value supplied at construction.
          custom_instructions: Extra guidance appended to the prompt.
          summary_pointers: ``(path, topic)`` pairs surfaced in the continuation.

        Returns:
          summary: Compacted history with the continuation user message
              placed per ``direction``.

        """
        compact_model = self._model or model
        history = _strip_attachments(history)
        effective_keep = self._keep_recent if keep_recent is None else keep_recent

        if effective_keep > 0 and len(history) > effective_keep:
            if direction == "from":
                to_summarize, to_keep = _safe_split(
                    history, effective_keep, direction="from"
                )
            else:
                to_keep, to_summarize = _safe_split(
                    history, effective_keep, direction="up_to"
                )
        else:
            to_summarize = history
            to_keep = []

        if to_keep:
            body = self._partial_prompt
            if direction == "up_to":
                body = (
                    body.rstrip() + "\n\nNote: in this compaction the EARLIER messages"
                    " are preserved verbatim before this summary. Focus"
                    " your summary on the RECENT portion of the"
                    " conversation, after the retained prefix."
                )
        else:
            body = self._prompt
        if custom_instructions and custom_instructions.strip():
            body = (
                body.rstrip()
                + "\n\nAdditional guidance from the user:\n"
                + custom_instructions.strip()
            )
        prompt = _NO_TOOLS_PREAMBLE + body + _NO_TOOLS_TRAILER

        groups = _group_history_by_round(to_summarize)
        summary_text: str | None = None
        for attempt in range(self._max_attempts):
            entries: list[HistoryEntry] = [e for g in groups for e in g]
            # Anthropic requires the first message role to be ``user``;
            # dropping groups can leave ``entries[0]`` as an
            # ``AssistantMessage``. Prepend a synthetic user bridge.
            if entries and not isinstance(entries[0], UserMessage):
                entries = [
                    UserMessage(text="[earlier messages elided]"),
                    *entries,
                ]
            request = ModelRequest(
                messages=[*entries, UserMessage(text=prompt)],
                system=_SYSTEM,
                tools=None,
            )
            try:
                response = await send_with_retry(
                    compact_model,
                    request,
                    max_attempts=self._max_attempts,
                    persistent_retry=False,
                    publish_recoverable=lambda text: logger.info(
                        "compactor recoverable: %s", text
                    ),
                )
                summary_text = response.message.text
                break
            except PromptTooLongError as exc:
                drop = _groups_to_drop(groups, exc, self._chars_per_token)
                logger.warning(
                    "Prompt too long (attempt %d/%d), dropping %d groups.",
                    attempt + 1,
                    self._max_attempts,
                    drop,
                )
                groups = groups[drop:]

        if summary_text is None:
            logger.warning("Compaction failed after %d attempts.", self._max_attempts)
            return [
                UserMessage(
                    text="Compaction failed. Previous context lost. Start fresh.",
                ),
            ]

        raw = summary_text or "(compaction produced no output)"
        summary = _format_summary(raw)
        logger.info(
            "Compacted %d entries → summary (%d chars), kept %d recent.",
            len(to_summarize),
            len(summary),
            len(to_keep),
        )
        continuation = UserMessage(
            text=build_continuation(
                summary,
                transcript_path=transcript_path,
                recent_preserved=bool(to_keep),
                proactive=self._proactive,
                summary_pointers=summary_pointers,
            ),
        )
        if direction == "from":
            return [continuation, *to_keep]
        return [*to_keep, continuation]


def microcompact(
    history: list[HistoryEntry],
    tools: dict[str, Tool],
    *,
    keep_recent: int = MICROCOMPACT_KEEP_RECENT,
) -> None:
    """Clear old clearable tool results in place.

    Walks the ``AssistantMessage`` entries to learn which tool produced
    each ``ToolResult`` (via ``ToolCall.name``), then replaces the
    ``content`` of clearable older ``ToolResult`` entries with
    ``CLEARED``. The most recent ``keep_recent`` clearable results are
    preserved.

    Args:
      history: History list to mutate in place.
      tools: Tool registry; only ``supports_microcompaction`` results clear.
      keep_recent: Number of most-recent clearable results to preserve.

    """
    tool_for_call: dict[str, str] = {}
    for entry in history:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                tool_for_call.setdefault(tc.id, tc.name)

    clearable: list[int] = []
    for i, entry in enumerate(history):
        if not isinstance(entry, ToolResult) or entry.content == CLEARED:
            continue
        tool = tools.get(tool_for_call.get(entry.call_id, ""))
        if tool is not None and getattr(tool, "supports_microcompaction", False):
            clearable.append(i)

    to_clear = clearable[:-keep_recent] if keep_recent > 0 else clearable
    for i in to_clear:
        entry = history[i]
        if isinstance(entry, ToolResult):
            history[i] = dataclasses.replace(
                entry, content=CLEARED, attachments=(), diff="", hint="", summary=""
            )


def _format_summary(raw: str) -> str:
    """Strip <analysis>, extract <summary> content."""
    text = _RE_ANALYSIS.sub("", raw)
    m = _RE_SUMMARY.search(text)
    if m:
        text = f"Summary:\n{m.group(1).strip()}"
    elif len(text) > _MAX_FALLBACK_SUMMARY_CHARS:
        text = text[:_MAX_FALLBACK_SUMMARY_CHARS] + "\n...(truncated)"
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


def build_continuation(
    summary: str,
    transcript_path: str | Path | None = None,
    recent_preserved: bool = False,
    proactive: bool = False,
    summary_pointers: list[tuple[str, str]] | None = None,
) -> str:
    """Build the post-compaction continuation message from asset templates.

    Args:
      summary: Compacted summary text to embed.
      transcript_path: Optional disk path of the full pre-compact transcript.
      recent_preserved: True when a recent tail was kept verbatim.
      proactive: When True, the resume directive runs autonomously.
      summary_pointers: ``(path, topic)`` pairs surfaced ahead of the summary.

    Returns:
      message: Continuation user-message text ready to splice into history.

    """
    pointers = ""
    if summary_pointers:
        lines = [
            "Prior context summaries (read files for full detail)."
            " Any pending tasks from these summaries that have not"
            " been completed are still active.",
            *(f"- {path}: {topic}" for path, topic in summary_pointers),
        ]
        pointers = "\n".join(lines) + "\n\n"
    transcript = ""
    if transcript_path is not None:
        transcript = (
            "\n\nThe full conversation transcript is at:"
            f" {transcript_path}. Consult it for exact code,"
            " error output, or other pre-compaction details."
        )
    recent = (
        "\n\nThe most recent messages appear below in their original form."
        if recent_preserved
        else ""
    )
    if proactive:
        resume = (
            "\n\nContinue working autonomously. Do not ask the user"
            " any questions - resume directly from the last task."
            " If uncertain, pick the most reasonable option and proceed."
        )
    else:
        resume = (
            "\nResume work immediately without preamble. Do not"
            " reference or acknowledge this summary, do not recap"
            " prior work, and do not use transition phrases"
            ' ("I\'ll continue", "picking up where we left off",'
            " etc.). Proceed as though no interruption occurred."
        )
    return (
        _CONTINUATION_TEMPLATE.replace("{{pointers}}", pointers)
        .replace("{{summary}}", summary)
        .replace("{{transcript}}", transcript)
        .replace("{{recent}}", recent)
        .replace("{{resume}}", resume)
        .strip()
    )


def _group_history_by_round(
    history: list[HistoryEntry],
) -> list[list[HistoryEntry]]:
    """Group history entries by API round-trip for safe truncation.

    A new round starts at each ``UserMessage`` (and at the first entry
    if it isn't a ``UserMessage``). ``AssistantMessage`` + its
    matching ``ToolResult`` entries cluster together so truncation
    never orphans a tool result.
    """
    groups: list[list[HistoryEntry]] = []
    current: list[HistoryEntry] = []
    for entry in history:
        if isinstance(entry, UserMessage) and current:
            groups.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        groups.append(current)
    return groups


def _safe_split(
    history: list[HistoryEntry],
    keep_recent: int,
    *,
    direction: Literal["from", "up_to"],
) -> tuple[list[HistoryEntry], list[HistoryEntry]]:
    """Split ``history`` so tool_use/tool_result pairs stay together.

    Starts from the requested split index (``len - keep_recent`` for
    ``"from"``; ``keep_recent`` for ``"up_to"``) and snaps the boundary
    left until no ``AssistantMessage.tool_calls`` in the prefix lacks a
    matching ``ToolResult`` in the prefix.

    Returns ``(to_summarize, to_keep)`` for ``"from"`` direction, or
    ``(to_keep, to_summarize)`` for ``"up_to"``.
    """
    n = len(history)
    idx = n - keep_recent if direction == "from" else keep_recent
    idx = max(0, min(idx, n))
    while idx > 0 and _prefix_has_unresolved_tool_use(history, idx):
        idx -= 1
    if idx == 0:
        return history, []
    return history[:idx], history[idx:]


def _prefix_has_unresolved_tool_use(
    history: list[HistoryEntry], split_idx: int
) -> bool:
    """True if ``history[:split_idx]`` contains any unresolved tool_use."""
    if split_idx <= 0:
        return False
    prefix = history[:split_idx]
    resolved: set[str] = {e.call_id for e in prefix if isinstance(e, ToolResult)}
    for entry in prefix:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                if tc.id not in resolved:
                    return True
    return False


def _strip_attachments(
    history: list[HistoryEntry],
) -> list[HistoryEntry]:
    """Drop binary attachments before summarization.

    Replaces each attachment with a ``[image]`` / ``[document]`` marker
    appended to the text so the model retains awareness that media was
    present without paying for the bytes again.
    """
    out: list[HistoryEntry] = []
    for entry in history:
        if isinstance(entry, UserMessage) and entry.attachments:
            marked = _attach_markers(entry.text, entry.attachments)
            out.append(dataclasses.replace(entry, text=marked, attachments=()))
        elif isinstance(entry, ToolResult) and entry.attachments:
            marked = _attach_markers(entry.content, entry.attachments)
            out.append(dataclasses.replace(entry, content=marked, attachments=()))
        else:
            out.append(entry)
    return out


def _attach_markers(content: str, attachments: tuple[object, ...]) -> str:
    """Suffix ``content`` with ``[image]`` / ``[document]`` markers."""
    markers = [
        "[document]"
        if getattr(a, "descriptor", "").startswith("application/pdf")
        else "[image]"
        for a in attachments
    ]
    if content:
        markers.insert(0, content)
    return " ".join(markers)
