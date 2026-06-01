"""Standalone compaction logic.

This package collects compaction logic that does **not** depend on
``Agent`` runtime state, so each piece is unit-testable in isolation:

- :mod:`files` -- post-compaction file re-attachment and the
  ``MICROCOMPACTED_ARGS_KEY`` sentinel used by ``repl/replay``.
- :mod:`history` -- pure history mutators (``append_to_first_user``)
  and the auto-compaction failure cap constant.
- :mod:`scrunch` -- the partition-from-oldest scrunch maneuver
  (planner + executor) that rescues an oversized resolved view by
  running the producer compactor in passes.
- :mod:`summary` -- the LLM-driven :class:`SummaryCompactor` strategy.

The Agent-coupled glue (``CompactionState``, ``post_compact_enrich``,
background-status re-surface) lives in
:mod:`sagent.agent.compaction` and imports the
pieces above.
"""

from __future__ import annotations
