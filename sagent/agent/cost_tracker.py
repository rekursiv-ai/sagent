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
