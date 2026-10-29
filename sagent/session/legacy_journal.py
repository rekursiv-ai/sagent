"""Old sagent session logs, read as one window of today's records.

The experimental sagent wrote ``session.jsonl`` as a tape: ``kind: history``
records, and ``context_splice`` records that hide some of them and show a
payload in their place; before splices, ``context_override``, ``context_clear``
and ``update`` records did the same. Before the tape, v3 wrote ``kind:
message`` records of nested ``descriptor`` parts.

:func:`convert` plays a log back to what its model last saw, as that sagent
did, and returns it as one window: a ``ContextClear``, then each record. Its
sessions shared today's directories and file name, ``<id>/session.jsonl``:
:func:`upgrade` rewrites one in place as today's journal, and :func:`find_old`
finds those not yet rewritten. Only this module knows the old formats:
deleting it ends their support.

Typical usage example:

  if is_legacy(path):
      upgrade(path)
  async with JsonlJournal(path) as journal:
      list(journal)                              # The old session, converted.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import base64
import binascii
import glob
import logging
import os
import tempfile

from treekle import PlainTree, ReadError, from_plain, json_loads

from sagent.lib.userdirs import data_dir
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import Response
from sagent.runtime.context import (
    alive_splices,
    masked_refs_by_alive,
    resolve_context,
)
from sagent.runtime.tape import (
    ContextSplice,
    InvalidPayloadError,
    MaskRange,
    ReferrableTapeEvent,
    TapeRecord,
    TapeRef,
    full_tape_mask,
    mask_contains_ref,
    merge_mask_ranges,
    splice_safe_repair,
    unpaired_call_ids,
)
from sagent.session.jsonl import record_line
from sagent.session.projects import is_session, legacy_log
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SessionRecord,
    SystemMessage,
    Thinking,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from sagent.runtime.types import ModelContextEvent


__all__ = [
    "convert",
    "find_old",
    "is_legacy",
    "upgrade",
]


logger = logging.getLogger(__name__)


type _Record = Mapping[str, PlainTree]


def is_legacy(path: Path) -> bool:
    """Return whether ``path`` is an old sagent log: its first record is untagged.

    Today's journal tags every record with its class, as ``py/object``.

    Args:
      path: The log.

    Returns:
      legacy: ``False`` for a log of today's records, an empty one, one
        that is no JSON, or one not yet written.

    """
    if not path.exists():
        return False
    for record in _records(path):
        return "py/object" not in record
    return False


def convert(path: Path) -> tuple[SessionRecord, ...]:
    """Return the old log at ``path`` as one window: what its model last saw.

    Args:
      path: The log.

    Returns:
      records: A ``ContextClear``, then each record, as a live journal writes
        a window.

    """
    meta: _Record = dict[str, PlainTree]()
    tape: list[_Record] = []
    for record in _records(path):
        kind = record.get("kind")
        # Some writers left the kind off their meta records.
        if kind == "meta" or (kind is None and "session_id" in record):
            meta = record
        elif kind == "message":
            v4 = _from_v3(record)
            if v4 is not None:
                tape.append(v4)
        elif kind in (
            "history",
            "context_splice",
            "context_override",
            "context_clear",
            "clear",
            "update",
        ):
            tape.append(record)
    tape_records = _tape(tape, _str(meta.get("session_id")))
    seen = splice_safe_repair(resolve_context(tape_records).messages)
    clear = ContextClear(
        timestamp=datetime.now(UTC).isoformat(),
        extra={"spend_usd": _spent(meta), "converted_from": str(path)},
    )
    return (
        clear,
        *(
            record
            for entry in seen
            for record in (entry.records if isinstance(entry, Response) else (entry,))
        ),
    )


def upgrade(path: Path) -> bool:
    """Rewrite the old log at ``path`` as today's journal of :func:`convert`.

    The log itself moves to ``projects.legacy_log``, whole: the journal keeps
    only its last window. A log with nothing said is left as it is: there is
    nothing to resume.

    Args:
      path: The log.

    Returns:
      upgraded: Whether it held something said, and so was rewritten.

    """
    records = convert(path)
    if len(records) <= 1:
        return False
    # Written beside it, then the log moved aside: a crash leaves the log whole.
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=".session-", suffix=".jsonl")
    with os.fdopen(fd, "wb") as file:
        file.writelines(record_line(record) for record in records)
    os.link(path, legacy_log(path.parent))
    Path(staged).replace(path)
    return True


def find_old(prefix: str, *, root: Path | None = None) -> list[Path]:
    """Return the log of each old session not yet rewritten whose id starts with ``prefix``.

    Args:
      prefix: The start of an id; matched literally.
      root: Where projects live; the per-user root when omitted.

    Returns:
      logs: Each session's ``<id>/session.jsonl`` still in an old format, in
        path order.

    """
    base = (
        root if root is not None else data_dir() / "rekursiv-ai" / "sagent" / "projects"
    )
    return sorted(
        log
        for log in base.glob(f"**/{glob.escape(prefix)}*/session.jsonl")
        if is_session(log) and is_legacy(log)
    )


def _records(path: Path) -> Iterator[_Record]:
    """Yield each JSON object ``path`` holds, one per line; skip the rest."""
    with path.open("rb") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                yield from_plain(json_loads(line), dict[str, PlainTree])
            except ValueError:
                logger.warning("skipping a line of %s that holds no record", path)


# Resolved as the old loader did: synthetic refs for records that carry none, a
# duplicated ref moved past every other, and ordinal order.
def _tape(records: Sequence[_Record], session_id: str) -> list[TapeRecord]:
    """Return the old records as today's tape."""
    tape: list[TapeRecord] = []
    ids: dict[int, int] = {}
    cursor = 0
    for record in records:
        kind = record.get("kind")
        ref = _ref(record.get("ref")) or TapeRef(session_id=session_id, ordinal=cursor)
        built: TapeRecord | None = None
        if kind == "history":
            event = _event(record)
            if event is not None:
                built = ReferrableTapeEvent(ref=ref, event=event)
                if isinstance(event, UncategorizedToolResult):
                    ids[_int(record.get("id"), -1)] = len(tape)
        elif kind in ("clear", "context_clear"):
            built = _splice(ref=ref, mask=full_tape_mask(tape), strategy="clear")
        elif kind == "update":
            _patch(tape, ids.get(_int(record.get("id"), -1)), record)
        else:
            built = _splice(
                ref=ref,
                mask=_mask(record, ref)
                if kind == "context_override"
                else _ranges(record),
                insert_after=_ref(
                    record.get(
                        "inject_after"
                        if kind == "context_override"
                        else "insert_after",
                    ),
                ),
                payload=_payload(record.get("payload")),
                paired=frozenset(_strs(record.get("paired_externally"))),
                strategy=_str(record.get("strategy")),
            )
        if built is not None:
            tape.append(built)
            cursor = max(cursor, built.ref.ordinal + 1)
    return sorted(_unique(tape), key=lambda r: r.ref.ordinal)


def _splice(
    *,
    ref: TapeRef,
    mask: tuple[MaskRange, ...],
    strategy: str,
    insert_after: TapeRef | None = None,
    payload: tuple[ModelContextEvent, ...] = (),
    paired: frozenset[str] = frozenset(),
) -> ContextSplice:
    """Return the splice; a payload not valid on its own is repaired first."""
    mask = merge_mask_ranges(mask)
    try:
        return ContextSplice(
            ref=ref,
            mask=mask,
            insert_after=insert_after,
            payload=payload,
            strategy=strategy,
            paired_externally=paired,
        )
    except InvalidPayloadError:
        repaired = splice_safe_repair(payload, paired_externally=paired)
        return ContextSplice(
            ref=ref,
            mask=mask,
            insert_after=insert_after,
            payload=repaired,
            strategy=strategy,
            paired_externally=unpaired_call_ids(repaired),
        )


# Two writers could take one ordinal. A splice re-appended at a masked position was an
# edit applied twice, and is dropped; anything else is someone's work, and moves on,
# keeping its position's fate: a splice written after it that hid the position hides
# where it moved, and one written before it never meant to.
def _unique(tape: list[TapeRecord]) -> list[TapeRecord]:
    """Return ``tape`` with each ref held once."""
    masked = masked_refs_by_alive(tape, alive_splices(tape))
    after = max(
        (
            *(r.ref.ordinal for r in tape),
            *(m.hi for r in tape if isinstance(r, ContextSplice) for m in r.mask),
        ),
        default=-1,
    )
    seen: set[TapeRef] = set()
    moved: list[tuple[int, TapeRef, TapeRef]] = []
    kept: list[tuple[int, TapeRecord]] = []
    for index, record in enumerate(tape):
        if record.ref not in seen:
            seen.add(record.ref)
            kept.append((index, record))
        elif not (isinstance(record, ContextSplice) and record.ref in masked):
            after += 1
            fresh = replace(record.ref, ordinal=after)
            moved.append((index, record.ref, fresh))
            kept.append((index, replace(record, ref=fresh)))
    return [_carried(record, at=index, moved=moved) for index, record in kept]


def _carried(
    record: TapeRecord,
    *,
    at: int,
    moved: Sequence[tuple[int, TapeRef, TapeRef]],
) -> TapeRecord:
    """Return ``record``, a splice at ``at`` also hiding what moved from its mask."""
    if not isinstance(record, ContextSplice):
        return record
    extra = tuple(
        MaskRange(session_id=fresh.session_id, lo=fresh.ordinal, hi=fresh.ordinal)
        for index, ref, fresh in moved
        if index < at and mask_contains_ref(record.mask, ref)
    )
    if not extra:
        return record
    return replace(record, mask=merge_mask_ranges(record.mask + extra))


def _patch(tape: list[TapeRecord], at: int | None, record: _Record) -> None:
    """Rewrite the result at ``at`` as an ``update`` record says."""
    if at is None:
        return
    held = tape[at]
    assert isinstance(held, ReferrableTapeEvent)
    result = held.event
    assert isinstance(result, UncategorizedToolResult)
    content = _str(record.get("content"))
    tape[at] = replace(
        held,
        event=replace(
            result,
            content=content,
            is_error=_bool(record.get("is_error")),
            stub=_stub(None, content),
        ),
    )


def _ref(raw: PlainTree) -> TapeRef | None:
    """Return the ``{session_id, ordinal}`` ref, or ``None`` when malformed."""
    record = _object(raw)
    session_id, ordinal = record.get("session_id"), record.get("ordinal")
    if not isinstance(session_id, str) or isinstance(ordinal, bool):
        return None
    if not isinstance(ordinal, int) or ordinal < 0:
        return None
    return TapeRef(session_id=session_id, ordinal=ordinal)


def _ranges(record: _Record) -> tuple[MaskRange, ...]:
    """Return a splice's ``[[from, to], ...]`` mask; a malformed range is dropped."""
    ranges: list[MaskRange] = []
    for pair in _list(record.get("mask")):
        ends = _list(pair)
        if len(ends) != 2:
            continue
        lo, hi = _ref(ends[0]), _ref(ends[1])
        if lo is None or hi is None or lo.session_id != hi.session_id:
            continue
        if lo.ordinal <= hi.ordinal:
            ranges.append(
                MaskRange(session_id=lo.session_id, lo=lo.ordinal, hi=hi.ordinal),
            )
    return tuple(ranges)


# An override named the records it hid; a barrier with none hid everything up to its
# anchor, or before itself.
def _mask(record: _Record, ref: TapeRef) -> tuple[MaskRange, ...]:
    """Return an override's mask."""
    hidden = [
        r for raw in _list(record.get("suppresses")) if (r := _ref(raw)) is not None
    ]
    if hidden:
        return tuple(
            MaskRange(session_id=r.session_id, lo=r.ordinal, hi=r.ordinal)
            for r in hidden
        )
    if not _bool(record.get("barrier")):
        return ()
    anchor = _ref(record.get("inject_after"))
    if anchor is not None:
        return (MaskRange(session_id=anchor.session_id, lo=0, hi=anchor.ordinal),)
    return (MaskRange(session_id=ref.session_id, lo=0, hi=max(0, ref.ordinal - 1)),)


def _payload(raw: PlainTree) -> tuple[ModelContextEvent, ...]:
    """Return a splice payload's entries."""
    return tuple(
        event for item in _list(raw) if (event := _event(_object(item))) is not None
    )


def _event(record: _Record) -> ModelContextEvent | None:
    """Return the old history entry as today's; ``None`` for one the model never saw."""
    kind = record.get("type")
    text = _str(record.get("text"))
    attachments = _attachments(record.get("attachments"))
    timestamp = _time(record.get("timestamp"))
    if kind == "user":
        return UserMessage(timestamp=timestamp, content=text, attachments=attachments)
    if kind == "agent_send":
        return AgentToAgentMessage(
            timestamp=timestamp,
            content=text,
            attachments=attachments,
            sender=_str(record.get("source")) or None,
        )
    if kind == "system_message":
        return SystemMessage(timestamp=timestamp, content=text)
    if kind == "assistant":
        return _turn(record, text=text, timestamp=timestamp)
    if kind == "tool_result":
        content = _str(record.get("content"))
        return UncategorizedToolResult(
            timestamp=timestamp,
            call_id=_str(record.get("call_id")),
            content=content,
            is_error=_bool(record.get("is_error")),
            stub=_stub(record.get("result_kind"), content),
            attachments=attachments,
        )
    return None


# Gemini's signature rides on its part; a call with no id or name was never sent.
def _turn(record: _Record, *, text: str, timestamp: str | None) -> Response:
    """Return an old assistant entry as one model turn."""
    signed = _signature(record)
    calls = [
        ToolCall(
            timestamp=timestamp,
            call_id=call_id,
            name=name,
            arguments=_object(call.get("args")),
            extra=_signature(call),
        )
        for call in map(_object, _list(record.get("tool_calls")))
        if (call_id := _str(call.get("id"))) and (name := _str(call.get("name")))
    ]
    said = (
        [AssistantMessage(timestamp=timestamp, content=text, extra=signed)]
        if text
        else []
    )
    thinking = [
        t for b in _list(record.get("thinking_blocks")) if (t := _thinking(_object(b)))
    ]
    return Response(
        records=(*thinking, *said, *calls),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="",
    )


def _signature(record: _Record) -> dict[str, PlainTree]:
    """Return a part's Gemini signature as today's ``extra``."""
    signature = _str(record.get("thought_signature"))
    return {"thoughtSignature": signature} if signature else {}


# Anthropic seals its thinking with a signature, or redacts it whole; OpenAI seals its
# reasoning beside its id.
def _thinking(block: _Record) -> Thinking | None:
    """Return a thinking block as today's ``Thinking``; ``None`` for one unknown."""
    kind = block.get("type")
    if kind == "thinking":
        return Thinking(
            content=_str(block.get("thinking")),
            encrypted=_str(block.get("signature")) or None,
        )
    if kind == "redacted_thinking":
        return Thinking(encrypted=_str(block.get("data")) or None)
    if kind != "reasoning":
        return None
    content = "".join(_str(_object(p).get("text")) for p in _list(block.get("content")))
    summary = "".join(_str(_object(p).get("text")) for p in _list(block.get("summary")))
    return Thinking(
        content=content or _str(block.get("text")) or None,
        summary=summary or None,
        encrypted=_str(block.get("encrypted_content")) or None,
        extra={"reasoning_id": reasoning_id}
        if (reasoning_id := _str(block.get("id")))
        else {},
    )


def _stub(kind: PlainTree, content: str) -> Literal["pending", "cancelled"] | None:
    """Return what a stand-in result stood for, from its kind or, before kinds, its text."""
    if kind in ("pending", "cancelled"):
        return "pending" if kind == "pending" else "cancelled"
    if kind == "final":
        return None
    if content.startswith(
        (
            "[detached: tool still running; real result arrives in a later message]",
            "[Running in background: ",
        ),
    ):
        return "pending"
    if content == "[cancelled: tool killed before completion; no result will follow]":
        return "cancelled"
    return None


def _attachments(raw: PlainTree) -> tuple[Attachment, ...]:
    """Return the ``{mime, data}`` attachments; one that does not decode is dropped."""
    out: list[Attachment] = []
    for item in map(_object, _list(raw)):
        mime, data = _str(item.get("mime")), _str(item.get("data"))
        try:
            out.append(
                Attachment(
                    mime_descriptor=mime,
                    data=base64.b64decode(data, validate=True),
                ),
            )
        except binascii.Error:
            logger.warning("dropping an attachment that does not decode")
    return tuple(a for a in out if a.mime_descriptor and a.data)


# Some writers took the monotonic clock, which no date reads from.
def _time(raw: PlainTree) -> str | None:
    """Return seconds since the epoch as ISO 8601; ``None`` for no plausible date."""
    seconds = raw if isinstance(raw, int | float) and not isinstance(raw, bool) else 0.0
    if seconds > 1e15:
        seconds /= 1e9
    return datetime.fromtimestamp(seconds, UTC).isoformat() if seconds > 1e9 else None


# ``spend`` is the agent's and its children's, as an agent's spend counts today;
# ``own_spend``, its own alone, stands in for a log that kept nothing else.
def _spent(meta: _Record) -> float:
    """Return what the log says its agent and its children spent, in USD."""
    for key in ("spend", "own_spend"):
        buckets = _object(meta.get(key))
        if buckets:
            return sum(
                v
                for v in buckets.values()
                if isinstance(v, int | float) and not isinstance(v, bool)
            )
    cost = meta.get("total_cost_usd")
    return (
        float(cost)
        if isinstance(cost, int | float) and not isinstance(cost, bool)
        else 0.0
    )


# v3 parts nest: a turn holds text, thinking, and calls, each a call's id and arguments.
def _from_v3(record: _Record) -> _Record | None:
    """Return a v3 ``message`` record as the tape's ``history`` record."""
    descriptor = record.get("descriptor")
    parts = [_object(p) for p in _list(record.get("content"))]
    base: dict[str, PlainTree] = {
        "kind": "history",
        "timestamp": record.get("_timestamp"),
        "id": record.get("_id"),
    }
    if descriptor == "text/x-user-message":
        return {**base, "type": "user", "text": _str(record.get("content"))}
    if descriptor == "multipart/x-user-message":
        return {
            **base,
            "type": "user",
            "text": _joined(parts, "text/plain"),
            "attachments": _v3_attachments(parts),
        }
    if descriptor == "multipart/x-model-message":
        return {
            **base,
            "type": "assistant",
            "text": _joined(parts, "text/plain"),
            "thinking_blocks": [
                p.get("content")
                for p in parts
                if _str(p.get("descriptor")).startswith("application/x-thinking-")
            ],
            "tool_calls": [
                _v3_call(p)
                for p in parts
                if p.get("descriptor") == "multipart/x-tool-call"
            ],
        }
    if descriptor == "multipart/x-tool-result":
        return {
            **base,
            "type": "tool_result",
            "call_id": _joined(parts, "text/x-queue-id"),
            "content": "\n".join(
                t
                for p in parts
                if p.get("descriptor") in ("text/plain", "text/x-error")
                if (t := _str(p.get("content")))
            ),
            "is_error": any(p.get("descriptor") == "text/x-error" for p in parts),
            "attachments": _v3_attachments(parts),
        }
    return None


def _v3_call(part: _Record) -> dict[str, PlainTree]:
    """Return a v3 call part as the tape's ``{id, name, args}``."""
    pieces = [_object(p) for p in _list(part.get("content"))]
    tool = next(
        (
            p
            for p in pieces
            if _str(p.get("descriptor")).startswith("application/x-tool-")
        ),
        dict[str, PlainTree](),
    )
    return {
        "id": _joined(pieces, "text/x-queue-id"),
        "name": _str(tool.get("descriptor")).removeprefix("application/x-tool-"),
        "args": _object(tool.get("content")),
    }


def _v3_attachments(parts: Sequence[_Record]) -> list[PlainTree]:
    """Return a v3 message's image parts as the tape's ``{mime, data}``."""
    return [
        {
            "mime": _str(p.get("descriptor")),
            "data": _str(_object(p.get("content")).get("_bytes")),
        }
        for p in parts
        if _str(p.get("descriptor")).startswith("image/")
    ]


def _joined(parts: Sequence[_Record], descriptor: str) -> str:
    """Return the text of the parts of one ``descriptor``, a line each."""
    return "\n".join(
        t
        for p in parts
        if p.get("descriptor") == descriptor
        if (t := _str(p.get("content")))
    )


def _object(raw: PlainTree) -> dict[str, PlainTree]:
    """Return ``raw`` as a JSON object; ``{}`` for anything else."""
    try:
        return from_plain(raw, dict[str, PlainTree], default={})
    except ReadError:
        return {}


def _list(raw: PlainTree) -> list[PlainTree]:
    """Return ``raw`` as a JSON array; ``[]`` for anything else."""
    try:
        return from_plain(raw, list[PlainTree], default=[])
    except ReadError:
        return []


def _strs(raw: PlainTree) -> list[str]:
    """Return the strings of a JSON array."""
    return [s for s in _list(raw) if isinstance(s, str)]


def _str(raw: PlainTree) -> str:
    """Return ``raw`` if a string, else ``""``."""
    return raw if isinstance(raw, str) else ""


def _int(raw: PlainTree, default: int) -> int:
    """Return ``raw`` if an int, else ``default``."""
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else default


def _bool(raw: PlainTree) -> bool:
    """Return ``raw`` if a bool, else ``False``."""
    return raw if isinstance(raw, bool) else False
