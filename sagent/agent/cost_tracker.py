"""Per-session token and cost accounting.

The single store: every model response across the entire spawn tree
writes through ``Agent._record_response`` to the ``CostTracker`` that
the root agent installed in ``cost_root_var`` at lifecycle open.
Sub-agent responses land in the root's tracker by ContextVar
inheritance; persistent sub-agents shadow the var with their own
tracker so they accumulate independently.

Two methods, both name what they do:

- :meth:`record` -- write one model response. Always live; status pane reads
  ``tracker.total*`` directly.
- :meth:`restore_totals` -- session-resume hook; overwrites cumulative
  totals (``total_cost_usd`` + ``total``) from persisted metadata.
  Per-call provenance (``calls_by_model``, ``last_request``,
  ``last_response_time``) is intentionally *not* restored: those describe
  the live process's call history, and resume restarts that history.

There is no fold step, no snapshot, no second store. ``Agent`` keeps a
small ``_run_start`` snapshot for ``last_run_*`` deltas, but that is
internal bookkeeping, not a separate cost store.
"""

from __future__ import annotations

import dataclasses
import time

from sagent.types.model import ModelResponse, TokenCount


@dataclasses.dataclass(kw_only=True, slots=True)
class CostTracker:
    """Per-agent cumulative cost; the only cost store."""

    last_request: TokenCount = dataclasses.field(default_factory=TokenCount)
    """Token counts from the most recent response."""

    total: TokenCount = dataclasses.field(default_factory=TokenCount)
    """Cumulative token counts across all recorded responses."""

    total_cost_usd: float = 0.0
    """Cumulative USD cost."""

    calls_by_model: dict[str, int] = dataclasses.field(default_factory=dict)
    """Map from model id to number of recorded calls."""

    last_response_time: float = dataclasses.field(default_factory=time.time)
    """Wall-clock seconds of the last ``record``.

    Seeded to construction time so renderers that show "time since last
    response" produce a sane non-zero delta before the first ``record``;
    treat any value within the first second of process start as a
    placeholder rather than a real model response."""

    def record(self, response: ModelResponse, *, model_id: str) -> None:
        """Update totals from one completed model response.

        Args:
          response: Completed model response with token counts and cost.
          model_id: Model identifier for per-model call tracking.

        """
        self.last_request = response.tokens
        self.last_response_time = time.time()
        self.total = self.total + response.tokens
        self.total_cost_usd += response.total_cost
        self.calls_by_model[model_id] = self.calls_by_model.get(model_id, 0) + 1

    def restore_totals(self, *, total_cost_usd: float, total: TokenCount) -> None:
        """Overwrite cumulative totals from persisted session metadata.

        Counterpart to :meth:`record` for the session-resume path: seeds
        the cumulative-total fields (``total_cost_usd`` and ``total``)
        with the values written by an earlier session before any new
        responses are recorded on top.

        Per-call provenance fields (``calls_by_model``, ``last_request``,
        ``last_response_time``) are intentionally left untouched: they
        describe the *live* process's recorded calls, and the resumed
        process starts that history fresh.

        Args:
          total_cost_usd: Persisted cumulative USD cost.
          total: Persisted cumulative token counts.

        """
        self.total_cost_usd = total_cost_usd
        self.total = total
