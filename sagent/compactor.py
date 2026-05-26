"""Compaction: summarize conversation when context fills up.

- Structured 9-section summary with <analysis> scratchpad
- NO_TOOLS preamble + trailer (tool-use prevention)
- Prompt-too-long retry with message truncation
- Continuation suppression

Usage::

    from sagent.compactor import SummaryCompactor
    compactor = SummaryCompactor()
    agent = Agent(model=sonnet, compactor=compactor, ...)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import dataclasses
import logging
import re

from sagent.agent.retry import send_with_retry
from sagent.tools.core import read_asset, recipe_dict
from sagent.types.exceptions import PromptTooLongError
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelRequest, ModelResponse
from sagent.types.tape import ContextSplice, TapeRecord, TapeRef


logger = logging.getLogger(__name__)


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
    - Optional two-call self-verification (``verify_summary``)

    Args:
      prompt: Full compaction prompt template.
      partial_prompt: Prompt used when ``keep_recent`` preserves a tail.
      buffer_tokens: Headroom before ``should_compact`` returns True.
      chars_per_token: Conversion factor for token-gap to char-gap math.
      max_attempts: Retries on ``PromptTooLongError`` before giving up.
      keep_recent: Default ``keep_recent`` when ``compact`` doesn't override.
      proactive: When True, the continuation resumes work autonomously.
      verify_summary: When True, run a second LLM call after summarization
          asking the same model to critique the summary and fill gaps;
          use the improved output. Doubles compaction token cost and
          wall-clock; opt in when summary fidelity matters more.
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
        direction: Literal["from", "up_to"] = "from",
        proactive: bool = False,
        verify_summary: bool = False,
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
        self._direction: Literal["from", "up_to"] = direction
        self._proactive = proactive
        self._verify_summary = verify_summary
        self._model = model

    @property
    def proactive(self) -> bool:
        """Whether the compactor resumes autonomously after compaction."""
        return self._proactive

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
        tape: Sequence[TapeRecord],
        context: Sequence[HistoryEntry],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Summarize ``context`` into a barrier override.

        Direction (``"from"`` keeps tail, ``"up_to"`` keeps head) and
        ``keep_recent`` are constructor parameters. The barrier override
        replaces the visible context with ``payload = continuation``
        plus the preserved tail (or prefix); the runtime appends it.

        Args:
          tape: Append-only session tape (unused; reserved for future
              fine-grained suppression strategies).
          context: Resolved provider-facing context to summarize.
          model: Fallback model when the compactor has none of its own.
          mint_ref: Factory minting fresh ``TapeRef`` values.
          custom_instructions: Extra guidance appended to the prompt.

        Returns:
          override: Barrier override with the compacted payload, ready
              for the runtime to append.

        """
        compact_model = self._model or model
        # The barrier mask covers everything currently on the tape so
        # every alive splice gets absorbed; the new summary becomes the
        # sole live editor.
        mask: tuple[tuple[TapeRef, TapeRef], ...] = (
            ((tape[0].ref, tape[-1].ref),) if tape else ()
        )
        history = _strip_attachments(list(context))
        direction = self._direction
        effective_keep = self._keep_recent
        if direction == "from":
            effective_keep = max(effective_keep, _trailing_user_tail_len(history))

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

        token_before = sum(_entry_chars(e) for e in history) // self._chars_per_token

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
        entries: list[HistoryEntry] = []
        for attempt in range(self._max_attempts):
            entries = _request_entries(groups)
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
            fallback = UserMessage(
                text="Compaction failed. Previous context summarized on disk only.",
            )
            if direction == "from":
                payload = (fallback, *to_keep)
            else:
                payload = (*to_keep, fallback)
            token_after = sum(_entry_chars(e) for e in payload) // self._chars_per_token
            return ContextSplice(
                ref=mint_ref(),
                mask=mask,
                insert_after=None,
                payload=payload,
                strategy="summary_fallback",
                token_before=token_before,
                token_after=token_after,
                fallback_reason=f"summary failed after {self._max_attempts} attempts",
                preserved_tail_count=len(to_keep),
            )

        raw = summary_text or "(compaction produced no output)"
        if self._verify_summary:
            try:
                raw = await self._verify(
                    compact_model,
                    entries,
                    raw,
                )
            except Exception as exc:  # noqa: BLE001 -- verification is best-effort
                logger.warning("summary verification failed; using original: %s", exc)
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
                recent_preserved=bool(to_keep),
                proactive=self._proactive,
            ),
        )
        if direction == "from":
            payload = (continuation, *to_keep)
        else:
            payload = (*to_keep, continuation)
        token_after = sum(_entry_chars(e) for e in payload) // self._chars_per_token
        return ContextSplice(
            ref=mint_ref(),
            mask=mask,
            insert_after=None,
            payload=payload,
            strategy="summary",
            token_before=token_before,
            token_after=token_after,
        )

    async def _verify(
        self,
        compact_model: Model,
        original_entries: list[HistoryEntry],
        raw_summary: str,
    ) -> str:
        """Self-verification probe: ask the model to critique its own summary.

        Re-runs the model with the original entries plus the produced
        summary, asking it to identify any missing technical details,
        file paths, tool results, or user constraints and emit an
        improved summary. Returns the improved text when non-empty,
        otherwise the original.

        Args:
          compact_model: Model used for the verification call (same one
              that produced the summary).
          original_entries: Entries that fed the original summarization
              call, used to ground the critique.
          raw_summary: Raw summary text (pre-format) the verification
              should improve.

        Returns:
          improved: Verified summary text, or ``raw_summary`` unchanged
              when the verification call returned nothing substantive.

        """
        probe = (
            "You produced this summary of the prior conversation:\n\n"
            f"<previous_summary>\n{raw_summary}\n</previous_summary>\n\n"
            "Review your summary against the conversation above. Did you"
            " omit any specific technical details, file paths, tool"
            " results, user constraints, pending tasks, or recent"
            " decisions? If yes, emit a corrected complete summary in"
            " the same format. If the summary is already complete and"
            " accurate, emit exactly the token IDENTICAL on a single"
            " line and nothing else."
        )
        groups = _group_history_by_round(original_entries)
        response: ModelResponse | None = None
        for attempt in range(self._max_attempts):
            request = ModelRequest(
                messages=[*_request_entries(groups), UserMessage(text=probe)],
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
                        "verifier recoverable: %s", text
                    ),
                )
                break
            except PromptTooLongError as exc:
                drop = _groups_to_drop(groups, exc, self._chars_per_token)
                logger.warning(
                    "Verifier prompt too long (attempt %d/%d), dropping %d groups.",
                    attempt + 1,
                    self._max_attempts,
                    drop,
                )
                groups = groups[drop:]
        if response is None:
            raise PromptTooLongError("summary verification prompt too long")
        improved = response.message.text.strip()
        if not improved or improved.upper() == "IDENTICAL":
            return raw_summary
        return improved


def _request_entries(groups: list[list[HistoryEntry]]) -> list[HistoryEntry]:
    """Flatten request groups and normalize provider-facing shape."""
    entries = [entry for group in groups for entry in group]
    entries = _drop_orphan_tool_results(entries)
    if entries and not isinstance(entries[0], UserMessage):
        return [UserMessage(text="[earlier messages elided]"), *entries]
    return entries


def _drop_orphan_tool_results(
    entries: list[HistoryEntry],
) -> list[HistoryEntry]:
    """Filter out ``ToolResult`` entries with no preceding matching ``ToolCall``.

    Walks ``entries`` in order, tracking ``call_id``s seen on
    ``AssistantMessage.tool_calls``. Any ``ToolResult`` whose
    ``call_id`` was not previously declared is dropped.

    Args:
      entries: Provider-facing entries about to be sent in a request.

    Returns:
      filtered: ``entries`` with orphan ``ToolResult`` records removed.

    """
    declared: set[str] = set()
    out: list[HistoryEntry] = []
    for entry in entries:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                declared.add(tc.id)
            out.append(entry)
        elif isinstance(entry, ToolResult):
            if entry.call_id in declared:
                out.append(entry)
        else:
            out.append(entry)
    return out


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
    recent_preserved: bool = False,
    proactive: bool = False,
) -> str:
    """Build the post-compaction continuation message from asset templates.

    Args:
      summary: Compacted summary text to embed.
      recent_preserved: True when a recent tail was kept verbatim.
      proactive: When True, the resume directive runs autonomously.

    Returns:
      message: Continuation user-message text ready to splice into history.

    """
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
        _CONTINUATION_TEMPLATE.replace("{{summary}}", summary)
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


def _trailing_user_tail_len(history: list[HistoryEntry]) -> int:
    """Count consecutive user messages at the end of ``history``."""
    count = 0
    for entry in reversed(history):
        if not isinstance(entry, UserMessage):
            break
        count += 1
    return count


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
    safe = _safe_split_boundaries(history)
    while idx > 0 and not safe[idx]:
        idx -= 1
    if idx == 0:
        return history, []
    return history[:idx], history[idx:]


def _safe_split_boundaries(history: list[HistoryEntry]) -> list[bool]:
    """Return whether each prefix boundary has no unresolved tool calls."""
    unresolved: set[str] = set()
    safe = [True]
    for entry in history:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                unresolved.add(tc.id)
        elif isinstance(entry, ToolResult):
            unresolved.discard(entry.call_id)
        safe.append(not unresolved)
    return safe


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
