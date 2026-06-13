#!/usr/bin/env python3
"""Chronologically merge the chat/ and sagent/ main.jsonl logs.

Usage:
    python merge_jsonl.py                          # both default paths → stdout
    python merge_jsonl.py --output merged.jsonl    # write to file
    python merge_jsonl.py --since 2026-06-01       # only records on/after a date
    python merge_jsonl.py --runtime sagent         # filter to one runtime
    python merge_jsonl.py path/a.jsonl path/b.jsonl  # explicit inputs

Each emitted record is the original record with one added field:

    "runtime": "channel" | "sagent"

so consumers can tell which side produced what. The original record
shape is otherwise unchanged: ``{ts, from, to, body, [runtime]}``.

Designed for end-of-day routine and ad-hoc forensics — fast, no
external deps, no in-memory cap other than the OS file open limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import argparse
import heapq
import json
import sys


_DEFAULT_CHANNEL = (
    Path(__file__).resolve().parent.parent.parent / "channel" / "main.jsonl"
)
_DEFAULT_SAGENT = Path(__file__).resolve().parent.parent / "main.jsonl"


def _iter_records(path: Path, runtime_tag: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(ts, record)`` pairs from a jsonl file with the runtime stamped."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines silently; chat/ has historically
                # had a few from credit-balance errors mixed in.
                continue
            if not isinstance(rec, dict) or "ts" not in rec:
                continue
            rec.setdefault("runtime", runtime_tag)
            yield rec["ts"], rec


def merge(
    sources: list[tuple[Path, str]],
    *,
    since: str | None = None,
    only_runtime: str | None = None,
) -> Iterator[dict[str, Any]]:
    """K-way merge across sources by ``ts``.

    ``ts`` is ISO-8601 with a uniform precision (``YYYY-MM-DDTHH:MM:SS.mmmZ``),
    so lexicographic string ordering matches temporal ordering — no
    parsing needed in the hot path.
    """
    streams = [_iter_records(p, tag) for p, tag in sources]
    for ts, rec in heapq.merge(*streams, key=lambda pair: pair[0]):
        if since is not None and ts < since:
            continue
        if only_runtime is not None and rec.get("runtime") != only_runtime:
            continue
        yield rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit jsonl paths (default: both channel/ and sagent/ logs).",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write merged stream to this path (default: stdout).",
    )
    ap.add_argument(
        "--since",
        help="Drop records with ts < this prefix (e.g. '2026-06-01').",
    )
    ap.add_argument(
        "--runtime",
        choices=("channel", "sagent"),
        help="Filter to one runtime.",
    )
    args = ap.parse_args()

    if args.paths:
        # User-supplied: tag each by its parent directory name when
        # heuristically recognisable, else by filename stem.
        sources = []
        for p in args.paths:
            tag = (
                "channel"
                if "channel" in p.parts
                else "sagent"
                if "sagent" in p.parts
                else p.stem
            )
            sources.append((p, tag))
    else:
        sources = [
            (_DEFAULT_CHANNEL, "channel"),
            (_DEFAULT_SAGENT, "sagent"),
        ]

    stream = merge(sources, since=args.since, only_runtime=args.runtime)
    out = (
        sys.stdout if args.output is None else open(args.output, "w", encoding="utf-8")
    )
    try:
        for rec in stream:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
