"""Compaction: summarize conversation when context fills up.

- Structured 9-section summary with <analysis> scratchpad
- NO_TOOLS preamble + trailer (tool-use prevention)
- Prompt-too-long retry with message truncation
- Continuation suppression
- Post-compaction file re-attachment
- Microcompaction (time-gated result clearing)

Usage::

    from sagent.compactor import SummaryCompactor
    compactor = SummaryCompactor()
    agent = Agent(model=sonnet, compactor=compactor, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import dataclasses
import logging
import re
import time

from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import (
    Message,
    Model,
    ModelRequest,
    MultipartMessage,
    TextMessage,
    Tool,
)
from sagent.lib.compaction import CLEARED, tool_result_text
from sagent.lib.descriptors import (
    collect_binary,
    collect_text,
    is_user_message,
    strip_binary,
)
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    get_tool_name,
    response_tool_calls,
)
from sagent.tools.core import (
    ReadCacheEntry,
    read_asset,
    recipe_dict,
)


logger = logging.getLogger(__name__)

MICROCOMPACT_KEEP_RECENT = 5


# -- Prompts ---------------------------------------------------------------


def _compactor_path(key: str) -> str:
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


def _groups_to_drop(
    groups: list[list[Message]],
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
        for m in g:
            chars += len(str(m.content))
        if chars >= target_chars:
            return i + 1
    return max(1, len(groups) // 5)


# -- Implementation ----------------------------------------------------


class SummaryCompactor:
    """Compacts by summarizing the full conversation.

    - Structured 9-section summary with <analysis> scratchpad
    - Tool-use prevention (preamble + trailer)
    - Prompt-too-long retry (truncate oldest messages, up to 3×)
    - Post-compaction continuation suppression
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
        microcompact_gap_sec: float = 3600.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {max_attempts}",
            )
        self._prompt = prompt
        self._partial_prompt = partial_prompt
        self._buffer_tokens = buffer_tokens
        self._chars_per_token = chars_per_token

        self._max_attempts = max_attempts
        self._keep_recent = keep_recent
        self._proactive = proactive
        self._model = model
        self._microcompact_gap_sec = microcompact_gap_sec

    @property
    def proactive(self) -> bool:
        """Whether the compactor resumes autonomously after compaction."""
        return self._proactive

    def maintain(
        self,
        messages: list[Message],
        tools: dict[str, Tool],
        **kwargs: object,
    ) -> None:
        """Perform between-request context maintenance via microcompaction.

        Args:
          messages: Current conversation messages (mutated in place).
          tools: Active tool registry.
          **kwargs: Accepts ``read_cache`` and ``last_response_time``.

        """
        raw_cache = kwargs.get("read_cache")
        read_cache: dict[str, ReadCacheEntry] = {}  # default empty
        if isinstance(raw_cache, dict):
            read_cache = cast(dict[str, ReadCacheEntry], raw_cache)
        raw_time = kwargs.get("last_response_time", 0.0)
        last_response_time = (
            float(raw_time) if isinstance(raw_time, (int, float)) else 0.0
        )
        microcompact(
            messages,
            tools,
            read_cache,
            last_response_time=last_response_time,
            gap_sec=self._microcompact_gap_sec,
        )

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        """Determine whether the conversation should be compacted now.

        Args:
          input_tokens: Current input token count.
          max_request_tokens: Maximum input tokens the model accepts.
          max_response_tokens: Maximum output tokens reserved for response.

        Returns:
          needed: True if compaction is needed.

        """
        effective = max_request_tokens - max_response_tokens
        return input_tokens >= max(0, effective - self._buffer_tokens)

    async def compact(
        self,
        messages: list[Message],
        model: Model,
        transcript_path: Path | None = None,
        direction: Literal["from", "up_to"] = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[Message]:
        """Summarize conversation with structured format.

        Args:
          messages: Full conversation history.
          model: Model backend for summarization.
          transcript_path: Path to persist the transcript.
          direction: ``"from"`` preserves the suffix (summarize older
            tail), ``"up_to"`` preserves the prefix (summarize newer
            portion).
          keep_recent: Number of recent messages to preserve verbatim.
            Overrides the compactor's configured default for this call.
          custom_instructions: Extra instructions appended to the
            compaction prompt (e.g. from ``/compact <instructions>``).
          summary_pointers: Key-value pairs to include in the summary.

        Returns:
          compacted: Compacted message list.

        """
        compact_model = self._model or model
        messages = _strip_attachments(messages)
        effective_keep = self._keep_recent if keep_recent is None else keep_recent

        # Partial compaction: split into summarize + keep. Use
        # ``_safe_split`` so the summarize slice never ends with an
        # assistant tool_use without a matching tool_result - that
        # shape 400s the API.
        if effective_keep > 0 and len(messages) > effective_keep:
            if direction == "from":
                to_summarize, to_keep = _safe_split(
                    messages, effective_keep, direction="from"
                )
            else:  # "up_to" - keep the prefix, summarize the rest.
                to_keep, to_summarize = _safe_split(
                    messages, effective_keep, direction="up_to"
                )
        else:
            to_summarize = messages
            to_keep = []

        # Build prompt: preamble + body + trailer.
        # Two prompts: one for full compaction (default), one for
        # partial compaction (any time `to_keep` is non-empty,
        # regardless of direction). Choose between them up-front.
        if to_keep:
            body = self._partial_prompt
            if direction == "up_to":
                # The partial-compaction prompt is written assuming the
                # kept slice is the recent suffix. For prefix-preserving
                # ("up_to"), append a clarifying note.
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

        # Retry loop: drop oldest API rounds on prompt-too-long.
        # Grouping keeps tool_use/tool_result pairs together so
        # truncation doesn't orphan tool results.
        groups = _group_messages_by_round(to_summarize)
        for attempt in range(self._max_attempts):
            msgs = [m for g in groups for m in g]
            # Anthropic requires the first message role to be ``user``;
            # dropping groups can leave ``msgs[0]`` as an ``AssistantMessage``
            # (every non-first group starts with one). Prepend a
            # synthetic user bridge so the API-level invariant holds.
            if msgs and not is_user_message(msgs[0].descriptor):
                msgs = [
                    TextMessage(
                        "[earlier messages elided]",
                        "text/x-user-message",
                    ),
                    *msgs,
                ]
            request = ModelRequest(
                messages=[
                    *msgs,
                    TextMessage(prompt, "text/x-user-message"),
                ],
                system=_SYSTEM,
                tools=None,
            )
            try:
                # stream() not buffer(): the Anthropic SDK rejects
                # non-streaming requests it estimates will exceed 10min.
                # Large-context compaction hits that threshold.
                response = await compact_model.stream(request=request)
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
        else:
            logger.warning(
                "Compaction failed after %d attempts.",
                self._max_attempts,
            )
            return [
                TextMessage(
                    "Compaction failed. Previous context lost. Start fresh.",
                    "text/x-user-message",
                ),
            ]

        raw = _response_text(response.content) or "(compaction produced no output)"
        summary = _format_summary(raw)
        logger.info(
            "Compacted %d messages → summary (%d chars), kept %d recent messages.",
            len(to_summarize),
            len(summary),
            len(to_keep),
        )
        continuation = TextMessage(
            build_continuation(
                summary,
                transcript_path=transcript_path,
                recent_preserved=bool(to_keep),
                proactive=self._proactive,
                summary_pointers=summary_pointers,
            ),
            "text/x-user-message",
        )
        # "from" → [continuation, ...kept_tail]
        # "up_to" → [...kept_prefix, continuation]
        if direction == "from":
            msgs = [continuation, *to_keep]
        else:
            msgs = [*to_keep, continuation]
        return msgs


def microcompact(
    messages: list[Message],
    tools: dict[str, Tool],
    read_cache: dict[str, ReadCacheEntry],
    *,
    last_response_time: float,
    gap_sec: float,
    keep_recent: int = MICROCOMPACT_KEEP_RECENT,
) -> None:
    """Clear old tool results when cache is cold.

    Only fires when the time gap since the last response exceeds
    ``gap_sec`` (cache TTL expired). Replaces old tool results
    whose tool has ``supports_microcompaction=True`` with a
    cleared marker. Mutates ``messages`` in place and pops
    invalidated entries from ``read_cache``.

    Args:
      messages: Conversation messages (mutated in place).
      tools: Active tool registry.
      read_cache: Read cache entries (mutated in place).
      last_response_time: Epoch time of last model response.
      gap_sec: Minimum seconds since last response before clearing.
      keep_recent: Number of recent clearable results to preserve.

    """
    if time.time() - last_response_time <= gap_sec:
        return

    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].descriptor == "multipart/x-model-message":
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        return

    tool_names: dict[str, str] = {}
    tool_inputs: dict[str, dict[str, object]] = {}
    for msg in messages:
        if msg.descriptor == "multipart/x-model-message":
            for tc in response_tool_calls(msg):
                qid = get_queue_id(tc)
                tool_names.setdefault(qid, f"application/x-tool-{get_tool_name(tc)}")
                tool_inputs.setdefault(qid, dict(get_directive(tc)))

    clearable: list[int] = []
    for i, msg in enumerate(messages):
        if i >= last_assistant_idx:
            break
        if msg.descriptor != "multipart/x-tool-result":
            continue
        text = tool_result_text(msg)
        if text == CLEARED:
            continue
        t = tools.get(tool_names.get(get_queue_id(msg), ""))
        if t is not None and t.supports_microcompaction:
            clearable.append(i)

    to_clear = clearable[:-keep_recent] if keep_recent > 0 else clearable
    for i in to_clear:
        msg = messages[i]
        qid = get_queue_id(msg)
        if tool_names.get(qid) == "application/x-tool-read":
            fp = tool_inputs.get(qid, {}).get("file_path", "")
            if isinstance(fp, str) and fp:
                read_cache.pop(str(Path(fp).resolve()), None)
        messages[i] = dataclasses.replace(
            msg,
            content=(
                TextMessage(qid, "text/x-queue-id"),
                TextMessage(
                    CLEARED,
                    "text/plain",
                ),
            ),
        )


def _response_text(msg: Message) -> str:
    """Extract joined text/plain parts from a model response."""
    parts = cast(tuple[Message, ...], msg.content)
    return "\n".join(str(p.content) for p in parts if p.descriptor == "text/plain")


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
      summary: Compacted summary text.
      transcript_path: Path to the full transcript file.
      recent_preserved: Whether recent messages are preserved verbatim.
      proactive: Whether to resume autonomously without prompting.
      summary_pointers: Prior context summaries as (path, topic) pairs.

    Returns:
      continuation: Formatted continuation message string.

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


def _group_messages_by_round(
    messages: list[Message],
) -> list[list[Message]]:
    """Group messages by API round-trip for safe truncation.

    Split only when the assistant ``message.id`` changes. Two undefined
    ids compare equal and do NOT force a split (so providers without
    stable ids - Google, older sessions - cluster adjacent assistant
    messages into one round rather than fragmenting).

    Returns:
      groups: List of message groups keeping tool_use/tool_result
          pairs adjacent so truncation never orphans a result.

    """
    groups: list[list[Message]] = []
    current: list[Message] = []
    last_assistant_id = ""
    for msg in messages:
        if msg.descriptor == "multipart/x-model-message":
            msg_id = get_queue_id(msg)
            if msg_id != last_assistant_id and current:
                groups.append(current)
                current = [msg]
            else:
                current.append(msg)
            last_assistant_id = msg_id
        else:
            current.append(msg)
    if current:
        groups.append(current)
    return groups


def _safe_split(
    messages: list[Message],
    keep_recent: int,
    *,
    direction: Literal["from", "up_to"],
) -> tuple[list[Message], list[Message]]:
    """Split ``messages`` so tool_use/tool_result pairs stay together.

    Starts from the requested split index (``len - keep_recent`` for
    ``"from"``; ``keep_recent`` for ``"up_to"``) and snaps the boundary
    left until no tool_use/result pair straddles it.

    Two shapes to avoid, both of which 400 the Anthropic API when
    present at the slice edges:

    - Prefix ends with an assistant ``tool_use`` whose matching
      ``tool_result`` lives in the suffix.
    - Suffix starts with a ``tool_result`` whose matching ``tool_use``
      lives in the prefix.

    Both describe the same straddle; we check the first, which also
    implies the second when messages are in valid Anthropic order.

    Returns ``(to_summarize, to_keep)`` for ``"from"`` direction, or
    ``(to_keep, to_summarize)`` for ``"up_to"``. Shared slicing in
    both branches - the caller binds positionally.
    """
    n = len(messages)
    idx = n - keep_recent if direction == "from" else keep_recent
    # Clamp to valid index range. Callers already bound ``keep_recent``
    # against ``len(messages)``, but the helper is tested in isolation
    # and should not silently produce negative-index slices (which
    # become "from the end" in Python - wrong answer for ``"from"``).
    idx = max(0, min(idx, n))
    while idx > 0 and _prefix_has_unresolved_tool_use(messages, idx):
        idx -= 1
    if idx == 0:
        return messages, []
    return messages[:idx], messages[idx:]


def _prefix_has_unresolved_tool_use(messages: list[Message], split_idx: int) -> bool:
    """True if ``messages[:split_idx]`` contains any unresolved tool_use.

    Scans every model response in the prefix - not just the last
    one - so a pathological session with multiple unresolved tool_uses
    (e.g. one recovered by ``_repair_dangling_tool_calls`` plus one at
    the tail) is still caught. The prefix is unresolved iff ANY
    assistant's tool_call ids are missing a matching tool_result
    later in the prefix.
    """
    if split_idx <= 0:
        return False
    prefix = messages[:split_idx]
    resolved: set[str] = {
        get_queue_id(m) for m in prefix if m.descriptor == "multipart/x-tool-result"
    }
    for m in prefix:
        if m.descriptor == "multipart/x-model-message":
            for part in cast(tuple[Message, ...], m.content):
                if (
                    part.descriptor == "multipart/x-tool-call"
                    and get_queue_id(part) not in resolved
                ):
                    return True
    return False


def _strip_attachments(messages: list[Message]) -> list[Message]:
    """Remove binary attachments before compaction (recursive).

    Replaces each binary attachment with a ``[image]`` or ``[document]``
    marker so the model retains awareness that media was present.
    """
    out: list[Message] = []
    for msg in messages:
        binary = collect_binary(msg)
        if not binary:
            out.append(msg)
            continue
        text = collect_text(msg)
        marked = _attach_markers(text, binary)
        stripped = strip_binary(msg)
        if stripped is None or stripped.descriptor == msg.descriptor:
            # Fully binary or same structure -- replace with text.
            if msg.descriptor == "multipart/x-tool-result":
                out.append(
                    MultipartMessage(
                        (
                            TextMessage(get_queue_id(msg), "text/x-queue-id"),
                            TextMessage(marked, "text/plain"),
                        ),
                        "multipart/x-tool-result",
                    )
                )
            else:
                out.append(TextMessage(marked, "text/x-user-message"))
        else:
            out.append(stripped)
    return out


def _attach_markers(content: str, attachments: list[Message]) -> str:
    """Suffix ``content`` with ``[image]`` / ``[document]`` markers."""
    markers = [
        "[document]" if p.descriptor == "application/pdf" else "[image]"
        for p in attachments
    ]
    if content:
        markers.insert(0, content)
    return " ".join(markers)
