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

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import dataclasses
import hashlib
import logging
import re
import time

from sagent.agent.context import resolve_context
from sagent.agent.retry import send_with_retry
from sagent.tools.core import read_asset, recipe_dict
from sagent.types.exceptions import PromptTooLongError
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelRequest
from sagent.types.tape import (
    ContextOverride,
    HistoryRecord,
    TapeRecord,
    TapeRef,
)
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)

MICROCOMPACT_KEEP_RECENT = 3
"""Number of most-recent fully-clearable AM-blocks to keep verbatim.

An "AM-block" is an ``AssistantMessage`` plus its matching ``ToolResult``
records. Keeping a block preserves both the AM (with its tool_calls) and
all of its TRs unchanged. Blocks older than the kept window collapse into
a single text-only ``AssistantMessage`` per block whose ``text`` carries
a human-readable summary.
"""

MICROCOMPACT_SUMMARY_MARKER = "[microcompacted:"
"""Sentinel prefix appended to a microcompacted ``AssistantMessage.text``.

Used for idempotency: when the marker is present, the AM has already
been microcompacted and won't be touched again. The full marker shape
is ``[microcompacted: <tool>(<short args>); <tool>(<short args>); ...]``.
"""

MICROCOMPACT_MIN_CLEARABLE_BYTES = 5_000
"""Threshold below which ``SummaryCompactor.maintain`` skips microcompact.

Cache rebuild on Anthropic costs ~$0.05 per 250K tokens; rewriting the
tape to save <5KB of context is net negative. Above this threshold, the
savings on subsequent cached requests outweigh the rebuild cost.
"""

MICROCOMPACT_MIN_INTERVAL_SEC = 300.0
"""Minimum seconds between microcompact firings (5 min ~ Anthropic prompt-cache TTL).

Prevents thrashing the prompt cache when individual results barely cross
the byte threshold. The cache will likely have expired naturally by the
time this interval has passed.
"""


def _clearable_bytes(
    context: Sequence[HistoryEntry],
    tools: Mapping[str, Tool],
) -> int:
    """Sum of ``ToolResult.content`` bytes for visible clearable tool exchanges.

    Used by :meth:`SummaryCompactor.maintain` to decide whether the
    accumulated clearable content justifies a microcompact firing
    (and the prompt-cache rebuild it causes).

    "Clearable" matches :func:`microcompact`'s filter: the matching AM
    is still visible, the AM's text doesn't already carry the
    microcompact marker, and the tool opts into
    ``supports_microcompaction``.
    """
    # Build call_id -> tool_name + AM lookup from visible context.
    tool_for_call: dict[str, str] = {}
    am_text_for_call: dict[str, str] = {}
    for entry in context:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                tool_for_call[tc.id] = tc.name
                am_text_for_call[tc.id] = entry.text
    total = 0
    for entry in context:
        if not isinstance(entry, ToolResult):
            continue
        tool_name = tool_for_call.get(entry.call_id)
        if tool_name is None:
            continue
        if MICROCOMPACT_SUMMARY_MARKER in am_text_for_call.get(entry.call_id, ""):
            continue
        tool = tools.get(tool_name)
        if tool is None or not getattr(tool, "supports_microcompaction", False):
            continue
        total += len(entry.content)
    return total


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
      microcompact_enabled: Kill switch for the per-AM-block microcompact
          pass. When False, ``maintain()`` returns ``()`` unconditionally.
      microcompact_keep_recent: Number of most-recent fully-clearable
          AM-blocks preserved verbatim. Older blocks collapse into a
          text-only summary AM.

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
        microcompact_enabled: bool = True,
        microcompact_keep_recent: int = MICROCOMPACT_KEEP_RECENT,
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
        self._microcompact_enabled = microcompact_enabled
        self._microcompact_keep_recent = microcompact_keep_recent
        self._last_microcompact_time: float = 0.0

    @property
    def proactive(self) -> bool:
        """Whether the compactor resumes autonomously after compaction."""
        return self._proactive

    def maintain(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[HistoryEntry],
        tools: dict[str, Tool],
        mint_ref: Callable[[], TapeRef],
    ) -> tuple[ContextOverride, ...]:
        """Produce microcompaction overrides for clearable tool exchanges.

        Gated by two thresholds:

        1. ``MICROCOMPACT_MIN_INTERVAL_SEC`` since the last microcompact
           firing. Prevents thrashing the prompt cache for incremental
           savings.
        2. ``MICROCOMPACT_MIN_CLEARABLE_BYTES`` of clearable content in
           the current visible context. Below this threshold the cache
           rebuild cost outweighs the savings.

        Args:
          tape: Append-only session tape.
          context: Resolved provider-facing context.
          tools: Tool registry; only ``supports_microcompaction`` results clear.
          mint_ref: Factory minting fresh ``TapeRef`` values.

        Returns:
          overrides: Microcompaction overrides; empty when the interval
              or byte threshold isn't met or no clearable exchange exists.

        """
        if not self._microcompact_enabled:
            return ()
        now = time.time()
        if (
            self._last_microcompact_time
            and (now - self._last_microcompact_time) < MICROCOMPACT_MIN_INTERVAL_SEC
        ):
            return ()
        clearable_bytes = _clearable_bytes(context, tools)
        if clearable_bytes < MICROCOMPACT_MIN_CLEARABLE_BYTES:
            return ()
        # Dedup first so microcompact's reverse walk sees the dedup
        # markers (short) and doesn't bother clearing them. The combined
        # tuple is appended to the tape in returned order.
        dedup_overrides = dedup_tool_results(tape, context, mint_ref)
        dedup_tape = (*tape, *dedup_overrides)
        dedup_context = (
            resolve_context(dedup_tape).messages if dedup_overrides else context
        )
        compact_overrides = microcompact(
            dedup_tape,
            dedup_context,
            tools,
            mint_ref,
            keep_recent=self._microcompact_keep_recent,
        )
        if dedup_overrides or compact_overrides:
            self._last_microcompact_time = now
        return (*dedup_overrides, *compact_overrides)

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
    ) -> ContextOverride:
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
        del tape  # reserved
        compact_model = self._model or model
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
            entries = [e for g in groups for e in g]
            # Drop orphan ``ToolResult`` entries before sending. Resolved
            # contexts can carry them when an override suppresses an
            # ``AssistantMessage`` whose ``ToolResult`` survives, or when
            # legacy load passed through malformed history. Providers
            # reject ``function_call_output`` blocks whose ``call_id``
            # has no matching ``function_call`` earlier in the request
            # (OpenAI 400 ``No tool call found for function call output``).
            entries = _drop_orphan_tool_results(entries)
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
            fallback = UserMessage(
                text="Compaction failed. Previous context summarized on disk only.",
            )
            if direction == "from":
                payload = (fallback, *to_keep)
            else:
                payload = (*to_keep, fallback)
            token_after = sum(_entry_chars(e) for e in payload) // self._chars_per_token
            return ContextOverride(
                ref=mint_ref(),
                suppresses=(),
                inject_after=None,
                payload=payload,
                strategy="summary_fallback",
                barrier=True,
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
        return ContextOverride(
            ref=mint_ref(),
            suppresses=(),
            inject_after=None,
            payload=payload,
            strategy="summary",
            barrier=True,
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
        request = ModelRequest(
            messages=[*original_entries, UserMessage(text=probe)],
            system=_SYSTEM,
            tools=None,
        )
        response = await send_with_retry(
            compact_model,
            request,
            max_attempts=self._max_attempts,
            persistent_retry=False,
            publish_recoverable=lambda text: logger.info(
                "verifier recoverable: %s", text
            ),
        )
        improved = response.message.text.strip()
        if not improved or improved.upper() == "IDENTICAL":
            return raw_summary
        return improved


def microcompact(
    tape: Sequence[TapeRecord],
    context: Sequence[HistoryEntry],
    tools: dict[str, Tool],
    mint_ref: Callable[[], TapeRef],
    *,
    keep_recent: int = MICROCOMPACT_KEEP_RECENT,
) -> tuple[ContextOverride, ...]:
    """Build microcompaction overrides for clearable AM-blocks.

    An AM-block is an ``AssistantMessage`` plus its matching
    ``ToolResult`` records. A block is "fully clearable" when every
    one of its tool_calls is from a tool with
    ``supports_microcompaction=True`` and every call has a matching
    ``ToolResult`` somewhere on the tape.

    For each fully-clearable block older than the ``keep_recent``
    window, this emits ONE override that:

    - Suppresses the AM HR and ALL of the block's TR HRs (plus any
      visible OVs that contribute TRs for those call_ids -- typically
      detached splice OVs that replaced placeholder TRs).
    - Injects one text-only replacement ``AssistantMessage`` whose
      ``tool_calls`` is empty and whose ``text`` is the AM's original
      text suffixed with a human-readable summary marker like
      ``[microcompacted: Bash(ls -la); Read(foo.py)]``.

    Design notes:

    - Partial blocks (some clearable, some not) are NOT touched. A
      partial replacement would orphan the kept TRs from their AM
      because the replacement lands in a different resolved slot.
    - The marker eliminates a class of model-mimicry bugs that arose
      when calls were stubbed with marker args (``{"_microcompacted": ...}``)
      that the model copied verbatim.
    - The marker also serves as idempotency: subsequent calls skip
      already-microcompacted blocks.

    Args:
      tape: Append-only session tape (used to locate suppression refs
          and slot anchors).
      context: Resolved provider-facing context to inspect.
      tools: Tool registry; only ``supports_microcompaction`` exchanges
          clear.
      mint_ref: Factory minting fresh ``TapeRef`` values.
      keep_recent: Number of most-recent fully-clearable AM-blocks to
          keep verbatim.

    Returns:
      overrides: One ``ContextOverride`` per cleared AM-block; empty
          when no fully-clearable block survives the ``keep_recent`` gate.

    """
    # Walk tape, tracking the slot anchor before each AssistantMessage,
    # the AssistantMessage record itself, and (per call_id) the
    # ``ToolResult`` record it produced. ``call_to_block`` maps call_id
    # -> index into the assistant block list. ``tr_extra_sources_by_call``
    # records visible OVs whose payload provides a ``ToolResult`` for a
    # given call_id -- e.g. a detached splice OV that replaced the
    # original placeholder TR. Microcompact must suppress these too,
    # otherwise the splice's TR remains visible alongside the cleared
    # TR (duplicate).
    assistant_blocks: list[tuple[TapeRef, TapeRef, AssistantMessage]] = []
    call_to_block: dict[str, int] = {}
    tool_for_call: dict[str, str] = {}
    tr_records: dict[str, tuple[TapeRef, ToolResult]] = {}
    tr_extra_sources_by_call: dict[str, list[TapeRef]] = {}
    hidden: set[TapeRef] = set()
    for record in reversed(tape):
        if record.ref in hidden:
            continue
        if isinstance(record, ContextOverride):
            hidden.update(record.suppresses)
            for payload_entry in record.payload:
                if isinstance(payload_entry, ToolResult):
                    tr_extra_sources_by_call.setdefault(
                        payload_entry.call_id, []
                    ).append(record.ref)
            if record.barrier:
                break
    prior_history_ref: TapeRef | None = None
    for record in tape:
        if not isinstance(record, HistoryRecord):
            continue
        entry = record.entry
        if isinstance(entry, AssistantMessage):
            # ``prior_history_ref`` is the slot before this AM; that's
            # the anchor where the AM and its TR replacements render.
            block_anchor = prior_history_ref or _SENTINEL_HEAD_ANCHOR
            block_idx = len(assistant_blocks)
            assistant_blocks.append((block_anchor, record.ref, entry))
            for tc in entry.tool_calls:
                tool_for_call.setdefault(tc.id, tc.name)
                call_to_block[tc.id] = block_idx
        elif isinstance(entry, ToolResult):
            tr_records.setdefault(entry.call_id, (record.ref, entry))
        prior_history_ref = record.ref

    # An AM-block is "fully clearable" iff every tool_call in it is
    # supports_microcompaction=True AND has a matching TR AND the AM's
    # text doesn't already carry the marker (idempotency). Partial
    # blocks (some clearable, some not) are left untouched: clearing
    # only some calls would orphan the kept TRs from their AM (the
    # microcompact replacement lands in a different slot than the kept
    # TRs, breaking AM/TR adjacency).
    visible_am_by_call: dict[str, AssistantMessage] = {}
    for entry in context:
        if isinstance(entry, AssistantMessage):
            for tc in entry.tool_calls:
                visible_am_by_call.setdefault(tc.id, entry)

    fully_clearable_blocks: list[int] = []
    for block_idx, (_anchor, _am_ref, am_entry) in enumerate(assistant_blocks):
        if not am_entry.tool_calls:
            continue
        first_visible = visible_am_by_call.get(am_entry.tool_calls[0].id)
        if first_visible is None:
            continue
        if MICROCOMPACT_SUMMARY_MARKER in first_visible.text:
            continue
        all_clearable = True
        for tc in am_entry.tool_calls:
            if tc.id not in tr_records:
                all_clearable = False
                break
            tool = tools.get(tool_for_call.get(tc.id, ""))
            if tool is None or not getattr(tool, "supports_microcompaction", False):
                all_clearable = False
                break
        if not all_clearable:
            continue
        fully_clearable_blocks.append(block_idx)

    # Preserve tape order so "most recent N AM-blocks kept" matches the
    # chronological semantic.
    fully_clearable_blocks.sort(
        key=lambda i: assistant_blocks[i][1].ordinal,
    )
    to_clear_block_idxs = (
        fully_clearable_blocks[:-keep_recent]
        if keep_recent > 0
        else fully_clearable_blocks
    )
    if not to_clear_block_idxs:
        return ()

    overrides: list[ContextOverride] = []

    # One override per fully-clearable AM-block. The override suppresses
    # the AM AND all of its TR records (plus any sibling overrides that
    # contribute those TRs -- detached splice OVs that replaced
    # placeholder TRs). The payload is one text-only AM (tool_calls=())
    # with a summary marker. Because all calls are cleared, there's no
    # AM/TR adjacency to preserve and the replacement can render in
    # whatever slot the resolver picks for its anchor.
    for block_idx in to_clear_block_idxs:
        anchor, am_ref, am_entry = assistant_blocks[block_idx]
        summary = "; ".join(_stub_args_summary(tc, tools) for tc in am_entry.tool_calls)
        marker = f"{MICROCOMPACT_SUMMARY_MARKER} {summary}]"
        new_text = (am_entry.text + "\n" + marker) if am_entry.text else marker
        replacement = dataclasses.replace(
            am_entry,
            text=new_text,
            tool_calls=(),
        )
        tr_refs = tuple(tr_records[tc.id][0] for tc in am_entry.tool_calls)
        extra_tr_sources: tuple[TapeRef, ...] = tuple(
            ref
            for tc in am_entry.tool_calls
            for ref in tr_extra_sources_by_call.get(tc.id, ())
        )
        suppresses = (am_ref, *tr_refs, *extra_tr_sources)
        overrides.append(
            ContextOverride(
                ref=mint_ref(),
                suppresses=suppresses,
                inject_after=anchor if anchor is not _SENTINEL_HEAD_ANCHOR else None,
                payload=(replacement,),
                strategy="microcompact_block",
            ),
        )

    return tuple(overrides)


# Sentinel: a non-None marker distinct from any real TapeRef, used to
# represent "no prior slot exists" without confusing ``inject_after=None``
# (which is itself the head-anchor signal at override-construction time).
_SENTINEL_HEAD_ANCHOR = TapeRef(session_id="\0", ordinal=-1)


DEDUP_MIN_CONTENT_CHARS = 500
"""Skip dedup for results below this size; the marker would be bigger than the content."""


def dedup_tool_results(
    tape: Sequence[TapeRecord],
    context: Sequence[HistoryEntry],
    mint_ref: Callable[[], TapeRef],
) -> tuple[ContextOverride, ...]:
    """Replace duplicate-content ``ToolResult``s with brief references.

    Walks the resolved context and hashes each ``ToolResult.content``.
    For every duplicate occurrence (second and beyond) whose content
    exceeds :data:`DEDUP_MIN_CONTENT_CHARS`, emits a ``ContextOverride``
    that suppresses the duplicate's current source and injects a marker
    referencing the first call_id with identical content.

    Common case: the agent re-reads the same file across many turns.
    The 89-reads-of-trackinizer.py pattern in real sessions stores the
    full content N times. After dedup, only the first occurrence is
    full; the rest collapse to ``[duplicate of call_id X: same content]``.

    Args:
      tape: Append-only session tape.
      context: Resolved provider-facing context (the message list).
      mint_ref: Factory minting fresh ``TapeRef`` values.

    Returns:
      overrides: One override per deduplicated occurrence. Empty when
          no duplicates qualify.

    """
    resolved = resolve_context(tape)
    if resolved.messages is not context and list(resolved.messages) != list(context):
        # Caller passed a stale context; re-resolve to match origins.
        # This should not happen in normal flow but guards against
        # accidentally combining mismatched (tape, context) pairs.
        resolved = resolve_context(tape)
    messages = resolved.messages
    origins = resolved.origins
    slot_anchors = resolved.slot_anchors
    tape_by_ref = {r.ref: r for r in tape}

    first_seen: dict[str, str] = {}  # hash -> first call_id
    overrides: list[ContextOverride] = []
    for idx, entry in enumerate(messages):
        if not isinstance(entry, ToolResult):
            continue
        if entry.is_error:
            continue
        if len(entry.content) < DEDUP_MIN_CONTENT_CHARS:
            continue
        digest = hashlib.sha256(entry.content.encode("utf-8")).hexdigest()[:16]
        first = first_seen.get(digest)
        if first is None:
            first_seen[digest] = entry.call_id
            continue
        if first == entry.call_id:
            # Same call_id repeated in context (microcompact splits AM/TR
            # across overrides; harmless but skip).
            continue
        origin = origins[idx]
        origin_record = tape_by_ref.get(origin)
        if origin_record is None:
            continue
        if isinstance(origin_record, ContextOverride):
            # Suppress the OV that contributes the duplicate, AND any
            # records that OV was already suppressing (so they stay
            # hidden after we hide their suppressor).
            suppresses = (origin, *origin_record.suppresses)
        else:
            suppresses = (origin,)
        marker = ToolResult(
            call_id=entry.call_id,
            content=f"[duplicate of call_id {first}: same content]",
        )
        anchor = slot_anchors[idx]
        overrides.append(
            ContextOverride(
                ref=mint_ref(),
                suppresses=suppresses,
                inject_after=anchor,
                payload=(marker,),
                strategy="dedup_tool_result",
                paired_externally=frozenset({entry.call_id}),
            ),
        )
    return tuple(overrides)


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


def _stub_args_summary(tc: ToolCall, tools: Mapping[str, Tool]) -> str:
    """Return the tool's ``summary(args)`` output, or fall back to the name."""
    tool = tools.get(tc.name)
    if tool is None:
        return tc.name
    try:
        summary = tool.summary(tc.args)
    except Exception:  # noqa: BLE001 -- tool.summary is user-defined; survive failures
        return tc.name
    return summary or tc.name


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
