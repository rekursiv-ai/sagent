"""Per-session token and cost accounting."""

from __future__ import annotations

import dataclasses
import time

from sagent.custom_types import ModelResponse, TokenCount
from sagent.tools.core import CostLedger


@dataclasses.dataclass(kw_only=True, slots=True)
class CostTracker:
    """Accumulates token counts and USD cost across model requests."""

    max_budget_usd: float | None = None
    last_request: TokenCount = dataclasses.field(default_factory=TokenCount)
    total: TokenCount = dataclasses.field(default_factory=TokenCount)
    call_output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_response_time: float = dataclasses.field(default_factory=time.time)

    def record(
        self,
        response: ModelResponse,
        *,
        model_id: str,
        ledger: CostLedger | None,
    ) -> None:
        """Update all counters from a model response.

        Args:
          response: Completed model response with token counts.
          model_id: Model identifier for ledger attribution.
          ledger: Subtree-wide cost ledger, if active.

        Raises:
          RuntimeError: If cumulative cost exceeds the budget.

        """
        self.last_request = response.tokens
        self.total = self.total + response.tokens
        self.call_output_tokens += response.tokens.output_tokens
        self.total_cost_usd += response.total_cost
        if ledger is not None:
            ledger.accumulate(response, model_id)
        self.last_response_time = time.time()
        if (
            self.max_budget_usd is not None
            and self.total_cost_usd >= self.max_budget_usd
        ):
            raise RuntimeError(
                f"Budget exhausted: ${self.total_cost_usd:.2f}"
                f" >= ${self.max_budget_usd:.2f}"
            )

    def fold(
        self,
        *,
        snapshot_cost_usd: float,
        snapshot_tokens: TokenCount,
        run_ledger: CostLedger,
    ) -> None:
        """Replace cumulative totals with ``snapshot + run_ledger``.

        Called by the root ``Agent`` at run-end to fold a completed
        run's full subtree (parent + descendants, captured in
        ``run_ledger``) into the cumulative tracker. The snapshot is
        the tracker's value at run start; replacing rather than adding
        rolls back the parent-only delta accrued via ``record()``
        during the run, so ``total*`` becomes session-cumulative
        subtree across runs.
        """
        self.total_cost_usd = snapshot_cost_usd + run_ledger.total_cost_usd
        self.total = snapshot_tokens + run_ledger.tokens

    def restore(self, *, total_cost_usd: float, total: TokenCount) -> None:
        """Overwrite cumulative totals from persisted session metadata.

        Counterpart to ``fold``: ``fold`` composes the run-end ledger
        with a snapshot, ``restore`` seeds the tracker from disk on
        session resume. Both go through the tracker rather than poking
        its fields from outside.
        """
        self.total_cost_usd = total_cost_usd
        self.total = total
