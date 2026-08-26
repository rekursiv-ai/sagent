#!/bin/sh
# ruff: noqa: EXE003, D300, T201 -- Polyglot shell/Python script; CLI reports.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")/../../../.." run --frozen --no-sync python3 "$0" "$@"
Repair sessions truncated by the overbroad user-coalesce splice.

A coalesce splice that landed on a barrier's payload absorbed that barrier's
whole-tape mask while injecting only the merged user message, so the resolved
view collapsed to a single entry. The tape is append-only and the damage is a
persisted record, so a code fix cannot undo it: this appends a kill-splice
masking the poison splice's own ref, which under the resolver's undelete
semantics lapses its masking and resurfaces the conversation.

The original file is copied to a timestamped ``.bak-<ns>`` sibling first.

Examples:
  sh repair_truncated_sessions.py --dry-run
  sh repair_truncated_sessions.py <session-dir>

'''
# fmt: on

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import argparse
import logging
import shutil
import time

from sagent.agent.context import resolve_context
from sagent.agent.session_io import (
    append_session,
    load_session,
    session_file_lock,
)
from sagent.lib.userdirs import data_dir
from sagent.sessions import restrict_path
from sagent.types.tape import (
    ContextSplice,
    MaskRange,
    TapeRecord,
    TapeRef,
)


logger = logging.getLogger(__name__)


def poison_splices(tape: Sequence[TapeRecord]) -> list[ContextSplice]:
    """Return coalesce splices that deleted content they absorbed.

    Aliveness is NOT a filter -- see the comment on the measurement below. A
    dead poison splice still truncates, because the barrier that killed it
    re-injected the already-truncated view as its own payload.

    Detected by measurement, not by inspecting the mask: killing the splice
    and re-resolving is the only test that distinguishes damage from a
    legitimately wide mask. A load-time repair barrier masks the whole tape and
    re-injects every message, so it is lossless despite its span; conversely
    the barrier a poison coalesce absorbed is usually absent from disk (it was
    synthesized inside ``load_session``), so there is nothing to compare its
    payload against. Both mask-span heuristics mis-classified real sessions.

    A coalesce legitimately hides the one tail entry it merged, so only a
    recovery of more than one entry counts as damage.

    Args:
      tape: Loaded tape records.

    Returns:
      poison: Coalesce splices whose removal restores hidden conversation.

    """
    visible = len(resolve_context(tape).messages)
    out: list[ContextSplice] = []
    for record in tape:
        if not isinstance(record, ContextSplice):
            continue
        if record.strategy != "user_coalesce":
            continue
        # Already repaired: a prior run's kill-splice masks this ref, and its
        # recovery would otherwise re-flag the splice on every subsequent run.
        if any(
            isinstance(other, ContextSplice)
            and other.strategy == "undo_overbroad_coalesce"
            and any(r.contains(record.ref) for r in other.mask)
            for other in tape
        ):
            continue
        # Measure, don't estimate: append a kill-splice over this record and
        # ask the resolver what the view becomes. Every heuristic over mask
        # spans mis-classified real sessions in both directions -- a lossless
        # barrier looks wide, and the barrier this splice absorbed is often
        # not even on disk. Recovering more than the one merged tail entry is
        # the definition of the damage, so it is what gets tested.
        #
        # Aliveness is deliberately NOT a filter: after one more resume the
        # load-time barrier absorbs the poison and re-injects the truncated
        # view, leaving the poison dead while its truncation lives inside that
        # barrier's payload. Filtering on alive missed exactly that session.
        # Compare against killing the derived barriers ALONE. Those barriers
        # sit above every coalesce, so removing them resurfaces content no
        # matter which coalesce is probed -- that shared recovery is not this
        # splice's doing. The damage attributable to the splice is what its own
        # death adds on top, and a healthy coalesce adds exactly the one tail
        # entry it merged.
        derived = kill_refs(tape, record)[1:]
        baseline = visible
        if derived:
            probe_derived = [*tape, _kill_splice(tape, derived, "baseline_probe")]
            baseline = len(resolve_context(probe_derived).messages)
        probe = [*tape, _kill_splice(tape, kill_refs(tape, record), "poison_probe")]
        if len(resolve_context(probe).messages) > baseline + 1:
            out.append(record)
    return out


def kill_refs(tape: Sequence[TapeRecord], poison: ContextSplice) -> list[TapeRef]:
    """Return ``poison``'s ref plus every later splice derived from it.

    Killing the coalesce alone is not enough. A later barrier -- the one
    ``load_session`` synthesizes on the next resume -- masks the whole tape and
    re-injects the view it found, which by then is already the truncated one.
    That barrier is what renders, so the recovery only appears once it dies
    too. Any later splice whose mask covers the poison inherited the damage and
    must lapse with it.

    Args:
      tape: Loaded tape records.
      poison: The truncating coalesce splice.

    Returns:
      refs: Refs to mask so the original conversation resurfaces.

    """
    refs = [poison.ref]
    for other in tape:
        if not isinstance(other, ContextSplice):
            continue
        if other.ref.ordinal <= poison.ref.ordinal:
            continue
        if any(r.contains(poison.ref) for r in other.mask):
            refs.append(other.ref)
    return refs


def _kill_splice(
    tape: Sequence[TapeRecord], refs: Sequence[TapeRef], strategy: str
) -> ContextSplice:
    """Build an empty-payload splice masking exactly ``refs``."""
    return ContextSplice(
        ref=TapeRef(
            session_id=refs[0].session_id,
            ordinal=max(r.ref.ordinal for r in tape) + 1,
        ),
        mask=tuple(
            MaskRange(session_id=r.session_id, lo=r.ordinal, hi=r.ordinal)
            for r in sorted(refs, key=lambda r: r.ordinal)
        ),
        insert_after=None,
        payload=(),
        strategy=strategy,
    )


def repair_session(session_dir: Path) -> bool:
    """Append a kill-splice undoing any poison splice in ``session_dir``.

    Args:
      session_dir: Directory holding ``session.jsonl``.

    Returns:
      repaired: True when a repair was written, False when nothing was wrong.

    """
    loaded = load_session(session_dir)
    if loaded is None:
        return False
    _meta, tape, _state = loaded
    poison = poison_splices(tape)
    if not poison:
        return False

    session_file = session_dir / "session.jsonl"
    refs = [ref for s in poison for ref in kill_refs(tape, s)]
    # Backup, re-read, mint, and append all under ONE lock. The backup is the
    # only rollback for this repair, so it has to describe the bytes the repair
    # actually modifies: copying first let a live append land in between, and
    # restoring that backup would have discarded the record. Re-reading inside
    # the lock likewise closes the mint window that put duplicate refs into
    # three real sessions. The lock is reentrant, so the inner append
    # re-enters rather than deadlocking.
    with session_file_lock(session_file):
        backup = session_file.with_name(f"{session_file.name}.bak-{time.time_ns()}")
        _ = shutil.copy(session_file, backup)
        # ``copy`` carries the source mode across, so a legacy 0644 transcript
        # yields a 0644 backup -- the copy re-publishing what the original
        # leaked. Every transcript is owner-only regardless of what it was.
        restrict_path(backup, 0o600)
        logger.info("backed up %s to %s", session_file, backup.name)
        current = load_session(session_dir)
        append_session(
            session_file,
            tape_delta=[
                _kill_splice(
                    current[1] if current is not None else tape,
                    sorted(set(refs), key=lambda r: r.ordinal),
                    "undo_overbroad_coalesce",
                )
            ],
        )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """The main function. Return the process exit code.

    Args:
      argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    roots = (
        [Path(p) for p in args.session_dirs]
        if args.session_dirs
        else [data_dir() / "rekursiv-ai" / "sagent" / "projects"]
    )
    candidates: list[Path] = []
    for root in roots:
        if (root / "session.jsonl").exists():
            candidates.append(root)
            continue
        candidates.extend(sorted(p.parent for p in root.glob("**/session.jsonl")))

    damaged = 0
    for session_dir in candidates:
        # Inspection must not write, and loading a damaged session copies it
        # aside by default -- so ``--dry-run`` left a backup for exactly the
        # sessions an operator runs a dry run against.
        loaded = load_session(session_dir, preserve_corrupt=not args.dry_run)
        if loaded is None:
            continue
        _meta, tape, _state = loaded
        poison = poison_splices(tape)
        if not poison:
            continue
        damaged += 1
        visible = len(resolve_context(tape).messages)
        print(
            f"{session_dir.name}: {visible} visible message(s),"
            f" {len(poison)} poison splice(s) at"
            f" {[s.ref.ordinal for s in poison]}"
        )
        if not args.dry_run and repair_session(session_dir):
            after = load_session(session_dir)
            assert after is not None
            print(f"  repaired -> {len(resolve_context(after[1]).messages)} messages")
    if damaged == 0:
        print(f"no truncated sessions among {len(candidates)} scanned")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "session_dirs",
        nargs="*",
        help="Session directories to repair. Defaults to every known session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report damaged sessions without modifying them.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
