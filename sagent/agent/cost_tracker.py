"""Per-session token and cost accounting.

Recording is split across two sinks. Cost rolls up to one place; tokens
stay local:

- Every agent calls :meth:`record_tokens` on its OWN tracker, so
  ``total`` is that agent's self-only token count (what the status pane
  renders per agent).
- Cost is recorded once on the root cost sink -- the ``CostTracker`` the
  root agent installed in ``cost_root_var`` at lifecycle open -- via
  :meth:`record_cost`, so ``root.spend`` is the whole spawn tree's spend
  counted exactly once per response. Each agent separately tracks its own
  cost on ``Agent._own_spend`` for its ``max_budget_usd`` cap; the cap is
  per-agent, the rollup is tree-wide.

Three methods, each named for what it does:

- :meth:`record_tokens` -- token totals + per-call provenance, self-only.
- :meth:`record_cost` -- cumulative USD cost; root sink only.
- :meth:`restore_totals` -- session-resume hook; overwrites cumulative
  totals (``spend`` + ``total``) from persisted metadata.
  Per-call provenance (``calls_by_model``, ``last_request``,
  ``last_response_time``) is intentionally *not* restored: those describe
  the live process's call history, and resume restarts that history.
"""

from __future__ import annotations

import dataclasses
import time

from sagent.types.cost import TokenCost, TokenCount
from sagent.types.model import ModelResponse


@dataclasses.dataclass(kw_only=True, slots=True)
class CostTracker:
    """Per-agent cumulative cost; the only cost store."""

    last_request: TokenCount = dataclasses.field(default_factory=TokenCount)
    """Token counts from the most recent response."""

    total: TokenCount = dataclasses.field(default_factory=TokenCount)
    """Cumulative token counts across all recorded responses."""

    spend: TokenCost = dataclasses.field(default_factory=TokenCost)
    """Cumulative USD cost, per token bucket."""

    calls_by_model: dict[str, int] = dataclasses.field(default_factory=dict)
    """Map from model id to number of recorded calls."""

    last_response_time: float = dataclasses.field(default_factory=time.time)
    """Wall-clock seconds of the last recorded response.

    Seeded to construction time so renderers that show "time since last
    response" produce a sane non-zero delta before the first record;
    treat any value within the first second of process start as a
    placeholder rather than a real model response."""

    def record_tokens(self, response: ModelResponse, *, model_id: str) -> None:
        """Update token totals + per-call provenance from one response.

        The token half of the recording split: every agent calls this on
        its OWN tracker so ``total`` stays self-only (the status pane's
        per-agent token count). The cost half is :meth:`record_cost`,
        which the root sink alone accumulates for the tree rollup.

        Args:
          response: Completed model response with token counts.
          model_id: Model identifier for per-model call tracking.

        """
        self.last_request = response.tokens
        self.last_response_time = time.time()
        self.total = self.total + response.tokens
        self.calls_by_model[model_id] = self.calls_by_model.get(model_id, 0) + 1

    def record_cost(self, response: ModelResponse) -> None:
        """Add one response's cost to the cumulative USD total.

        The cost half of the recording split: recorded once on the root
        cost sink (via ``cost_root_var``) so ``spend`` is the whole spawn
        tree's spend, counted exactly once per response.

        Args:
          response: Completed model response carrying ``spend``.

        """
        self.spend = self.spend + response.spend

    def restore_totals(self, *, spend: TokenCost, total: TokenCount) -> None:
        """Overwrite cumulative totals from persisted session metadata.

        Counterpart to :meth:`record_tokens` / :meth:`record_cost` for the
        session-resume path: seeds
        the cumulative-total fields (``spend`` and ``total``) with the
        values written by an earlier session before any new responses are
        recorded on top.

        Per-call provenance fields (``calls_by_model``, ``last_request``,
        ``last_response_time``) are intentionally left untouched: they
        describe the *live* process's recorded calls, and the resumed
        process starts that history fresh.

        Args:
          spend: Persisted cumulative USD cost.
          total: Persisted cumulative token counts.

        """
        self.spend = spend
        self.total = total
