#!/usr/bin/env python3
"""One-shot migrator from v3 ``session.jsonl`` to v4.

Reads a v3 session file (or every ``session.jsonl`` under a directory
tree) and writes a sibling ``session.v4.jsonl`` alongside. The v3 files
are left untouched.

v3 records carry ``Message`` shapes (``descriptor`` + ``content``).
The v4 schema is flat ``TapeEvent`` dataclasses encoded with
``kind: history``; see ``docs/private/agent_v4_contract.md`` §6.

Translation rules:

- ``text/x-user-message`` → ``UserMessage(text=content)``
- ``multipart/x-user-message`` → ``UserMessage(text=joined text parts,
  attachments=image parts)``
- ``multipart/x-model-message`` → ``AssistantMessage(text=joined text
  parts, thinking_blocks=structured thinking parts,
  tool_calls=parsed tool_call parts)``
- ``multipart/x-tool-result`` → ``ToolResult(call_id=queue id,
  content=joined plain/error text, is_error=any error part present,
  attachments=image parts, hint=joined hint parts)``

``application/x-file-stat`` and ``application/x-bash-state`` parts are
dropped; the v4 ``tool_state`` snapshot supersedes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import argparse
import base64
import json
import logging
import sys

from sagent.lib.custom_json import IntCodec


logger = logging.getLogger(__name__)


def _id(rec: Mapping[str, object]) -> int:
    """Return the v3 record ``_id`` as an int, defaulting to ``0``."""
    return IntCodec.coerce(rec.get("_id"), 0)


def _parent_id(rec: Mapping[str, object]) -> int:
    """Return the v3 record ``_parent_id`` as an int, defaulting to ``-1``."""
    return IntCodec.coerce(rec.get("_parent_id"), -1)


def _decode_bytes_content(content: object) -> bytes | None:
    """Decode a v3 ``{"_bytes": "<base64>"}`` wrapper to raw bytes."""
    if isinstance(content, dict):
        d = cast(Mapping[str, object], content)
        b64 = d.get("_bytes")
        if isinstance(b64, str):
            try:
                return base64.b64decode(b64)
            except (ValueError, TypeError):
                return None
    return None


def _part_text(part: Mapping[str, object]) -> str:
    """Return the stringified ``content`` field of a v3 part."""
    raw = part.get("content")
    return str(raw) if raw is not None else ""


def _is_image(descriptor: str) -> bool:
    """Return True iff ``descriptor`` is an image MIME type."""
    return descriptor.startswith("image/")


def _att_v4(descriptor: str, data: bytes) -> dict[str, str]:
    """Build a v4 attachment dict from a MIME type and raw bytes."""
    return {
        "mime": descriptor,
        "data": base64.b64encode(data).decode("ascii"),
    }


def _convert_user_text(rec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 ``text/x-user-message`` to a v4 user record."""
    return {
        "kind": "history",
        "type": "user",
        "text": _part_text(rec),
        "attachments": [],
        "id": _id(rec),
        "parent_id": _parent_id(rec),
        "timestamp": _legacy_ts_to_seconds(rec.get("_timestamp")),
    }


def _legacy_ts_to_seconds(raw: object) -> float:
    """Normalize a v3 timestamp (seconds or nanoseconds) to float seconds."""
    if isinstance(raw, (int, float)):
        v = float(raw)
        # v3 wrote ``time.time_ns()``; if it looks ns-scale, downshift.
        if v > 1e15:
            return v / 1e9
        return v
    return 0.0


def _convert_user_multipart(rec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 ``multipart/x-user-message`` to a v4 user record."""
    texts: list[str] = []
    atts: list[dict[str, str]] = []
    parts = rec.get("content")
    if isinstance(parts, list):
        for p in cast(list[object], parts):
            if not isinstance(p, dict):
                continue
            pp = cast(Mapping[str, object], p)
            desc = str(pp.get("descriptor") or "")
            if desc == "text/plain":
                texts.append(_part_text(pp))
            elif _is_image(desc):
                data = _decode_bytes_content(pp.get("content"))
                if data is not None:
                    atts.append(_att_v4(desc, data))
    return {
        "kind": "history",
        "type": "user",
        "text": "\n".join(t for t in texts if t),
        "attachments": atts,
        "id": _id(rec),
        "parent_id": _parent_id(rec),
        "timestamp": _legacy_ts_to_seconds(rec.get("_timestamp")),
    }


def _parse_tool_call(part: Mapping[str, object]) -> dict[str, object] | None:
    """Extract ``{id, name, args}`` from a v3 ``multipart/x-tool-call`` part."""
    children = part.get("content")
    if not isinstance(children, list):
        return None
    call_id = ""
    name = ""
    args: dict[str, object] = {}
    for c in cast(list[object], children):
        if not isinstance(c, dict):
            continue
        cp = cast(Mapping[str, object], c)
        desc = str(cp.get("descriptor") or "")
        if desc == "text/x-queue-id":
            call_id = _part_text(cp)
        elif desc.startswith("application/x-tool-"):
            name = desc[len("application/x-tool-") :]
            raw_args = cp.get("content")
            if isinstance(raw_args, dict):
                args = dict(cast(Mapping[str, object], raw_args))
    if not call_id and not name:
        return None
    return {"id": call_id, "name": name, "args": args}


def _convert_assistant(rec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 ``multipart/x-model-message`` to a v4 assistant record."""
    texts: list[str] = []
    thinking: list[dict[str, object]] = []
    tool_calls: list[dict[str, object]] = []
    parts = rec.get("content")
    if isinstance(parts, list):
        for p in cast(list[object], parts):
            if not isinstance(p, dict):
                continue
            pp = cast(Mapping[str, object], p)
            desc = str(pp.get("descriptor") or "")
            if desc == "text/plain":
                texts.append(_part_text(pp))
            elif desc == "application/x-thinking-structured":
                payload = pp.get("content")
                if isinstance(payload, dict):
                    thinking.append(dict(cast(Mapping[str, object], payload)))
            elif desc == "multipart/x-tool-call":
                tc = _parse_tool_call(pp)
                if tc is not None:
                    tool_calls.append(tc)
    return {
        "kind": "history",
        "type": "assistant",
        "text": "\n".join(t for t in texts if t),
        "thinking_blocks": thinking,
        "tool_calls": tool_calls,
        "id": _id(rec),
        "parent_id": _parent_id(rec),
        "timestamp": _legacy_ts_to_seconds(rec.get("_timestamp")),
    }


def _convert_tool_result(rec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 ``multipart/x-tool-result`` to a v4 tool_result record."""
    call_id = ""
    texts: list[str] = []
    hints: list[str] = []
    atts: list[dict[str, str]] = []
    is_error = False
    parts = rec.get("content")
    if isinstance(parts, list):
        for p in cast(list[object], parts):
            if not isinstance(p, dict):
                continue
            pp = cast(Mapping[str, object], p)
            desc = str(pp.get("descriptor") or "")
            if desc == "text/x-queue-id":
                call_id = _part_text(pp)
            elif desc == "text/plain":
                texts.append(_part_text(pp))
            elif desc == "text/x-error":
                is_error = True
                texts.append(_part_text(pp))
            elif desc == "text/x-hint-tool-use-nudge":
                hints.append(_part_text(pp))
            elif _is_image(desc):
                data = _decode_bytes_content(pp.get("content"))
                if data is not None:
                    atts.append(_att_v4(desc, data))
            # application/x-file-stat and application/x-bash-state: drop.
    return {
        "kind": "history",
        "type": "tool_result",
        "call_id": call_id,
        "content": "\n".join(t for t in texts if t),
        "is_error": is_error,
        "diff": "",
        "diff_file_path": "",
        "hint": "\n".join(h for h in hints if h),
        "summary": "",
        "attachments": atts,
        "id": _id(rec),
        "parent_id": _parent_id(rec),
        "timestamp": _legacy_ts_to_seconds(rec.get("_timestamp")),
    }


def _convert_message(rec: Mapping[str, object]) -> dict[str, object] | None:
    """Dispatch a v3 message record to the per-descriptor converter."""
    descriptor = str(rec.get("descriptor") or "")
    if descriptor == "text/x-user-message":
        return _convert_user_text(rec)
    if descriptor == "multipart/x-user-message":
        return _convert_user_multipart(rec)
    if descriptor == "multipart/x-model-message":
        return _convert_assistant(rec)
    if descriptor == "multipart/x-tool-result":
        return _convert_tool_result(rec)
    logger.warning("Dropping unknown v3 descriptor: %s", descriptor)
    return None


def _convert_meta(rec: Mapping[str, object]) -> dict[str, object]:
    """Translate a v3 meta record to its v4 equivalent."""
    # Drop the v3 ``version`` field; v4 has no such marker.
    out = {k: v for k, v in rec.items() if k != "version"}
    out["kind"] = "meta"
    return out


def iter_v4_records(lines: Iterable[str]) -> Iterable[dict[str, object]]:
    """Translate an iterable of v3 JSONL lines into v4 records.

    Args:
      lines: Lines from a v3 ``session.jsonl``.

    Yields:
      record: One v4 ``TapeEvent`` / meta / clear dict per translatable input.

    """
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed v3 line.")
            continue
        if not isinstance(record, dict):
            continue
        rec = cast(Mapping[str, object], record)
        kind = rec.get("kind")
        if kind == "meta":
            yield _convert_meta(rec)
        elif kind == "message":
            v4 = _convert_message(rec)
            if v4 is not None:
                yield v4
        elif kind == "clear":
            yield {"kind": "clear", "_timestamp": rec.get("_timestamp", 0)}
        else:
            logger.warning("Skipping unknown v3 kind: %r", kind)


def migrate_file(src: Path, dst: Path) -> int:
    """Translate one v3 session file to v4 alongside it.

    Args:
      src: Path to the v3 ``session.jsonl`` file to read.
      dst: Path to the v4 output file to write (parents created).

    Returns:
      count: Number of v4 records written.

    """
    count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for v4 in iter_v4_records(fin):
            _ = fout.write(json.dumps(v4) + "\n")
            count += 1
    return count


def _iter_targets(root: Path) -> Iterable[Path]:
    """Yield session.jsonl paths under ``root`` (or just ``root`` if a file)."""
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("session.jsonl"))


def main(argv: list[str] | None = None) -> int:
    """Entry point for the one-shot migrator.

    Args:
      argv: Optional CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
      exit_code: ``0`` on success, ``1`` if ``path`` does not exist.

    """
    parser = argparse.ArgumentParser(
        description="Migrate v3 sagent session.jsonl files to v4.",
    )
    _ = parser.add_argument(
        "path",
        type=Path,
        help="A v3 session.jsonl file, or a directory tree to scan.",
    )
    _ = parser.add_argument(
        "--suffix",
        default=".v4.jsonl",
        help="Output suffix appended to ``session`` (default: .v4.jsonl).",
    )
    _ = parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing v4 output files.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root: Path = cast(Path, args.path)
    if not root.exists():
        logger.error("path does not exist: %s", root)
        return 1

    targets = list(_iter_targets(root))
    if not targets:
        logger.info("No session.jsonl files under %s", root)
        return 0

    total = 0
    for src in targets:
        dst = src.with_name(f"session{args.suffix}")
        if dst.exists() and not args.overwrite:
            logger.info("skip (exists): %s", dst)
            continue
        n = migrate_file(src, dst)
        total += n
        logger.info("migrated %d records: %s -> %s", n, src, dst)
    logger.info("done: %d records across %d files", total, len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
