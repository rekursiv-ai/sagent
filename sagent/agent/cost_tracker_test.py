"""Tests for CostTracker."""

from __future__ import annotations

import pytest

from sagent.agent.cost_tracker import CostTracker
from sagent.custom_types import (
    Message,
    ModelResponse,
    MultipartMessage,
    TokenCount,
)
from sagent.tools.core import CostLedger


def _empty_message() -> Message:
    return MultipartMessage((), "multipart/x-model-message")


def _response(
    *,
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
) -> ModelResponse:
    return ModelResponse(
        content=_empty_message(),
        tokens=TokenCount(input_tokens=input_tokens, output_tokens=output_tokens),
        total_cost=total_cost,
    )


class TestRecord:
    def test_accumulates_tokens_and_cost(self) -> None:
        tracker = CostTracker()
        tracker.record(
            _response(input_tokens=10, output_tokens=5, total_cost=0.10),
            model_id="m",
            ledger=None,
        )
        tracker.record(
            _response(input_tokens=20, output_tokens=8, total_cost=0.20),
            model_id="m",
            ledger=None,
        )
        assert tracker.total.input_tokens == 30
        assert tracker.total.output_tokens == 13
        assert tracker.total_cost_usd == pytest.approx(0.30)

    def test_budget_exhaust_raises(self) -> None:
        tracker = CostTracker(max_budget_usd=0.05)
        with pytest.raises(RuntimeError, match="Budget exhausted"):
            tracker.record(
                _response(input_tokens=1, output_tokens=1, total_cost=0.10),
                model_id="m",
                ledger=None,
            )


class TestFold:
    def test_fold_replaces_with_snapshot_plus_ledger(self) -> None:
        """``fold`` rolls back the parent-only delta accrued via record() and
        substitutes the run's full subtree (parent + descendants) captured
        in the ledger.
        """
        tracker = CostTracker()
        # Pre-existing cumulative subtree from earlier completed runs.
        tracker.total_cost_usd = 1.00
        tracker.total = TokenCount(input_tokens=100, output_tokens=50)
        snap_cost = tracker.total_cost_usd
        snap_tokens = tracker.total
        # Mid-run: parent records its own call into tracker (and ledger).
        ledger = CostLedger()
        parent_call = _response(input_tokens=10, output_tokens=5, total_cost=0.20)
        tracker.record(parent_call, model_id="m", ledger=ledger)
        # Children record into the same ledger, NOT into parent's tracker.
        child_call = _response(input_tokens=7, output_tokens=3, total_cost=0.30)
        ledger.accumulate(child_call, "child-m")
        # Run end: fold replaces tracker totals with snapshot + ledger.
        tracker.fold(
            snapshot_cost_usd=snap_cost,
            snapshot_tokens=snap_tokens,
            run_ledger=ledger,
        )
        assert tracker.total_cost_usd == pytest.approx(1.50)  # 1.00 + (0.20 + 0.30)
        assert tracker.total.input_tokens == 100 + 10 + 7
        assert tracker.total.output_tokens == 50 + 5 + 3

    def test_fold_idempotent_across_two_runs(self) -> None:
        """Two completed runs must accumulate cleanly: each fold composes
        with the prior fold's result.
        """
        tracker = CostTracker()
        # Run 1.
        snap1_cost = tracker.total_cost_usd
        snap1_tokens = tracker.total
        ledger1 = CostLedger()
        tracker.record(
            _response(input_tokens=10, output_tokens=5, total_cost=0.20),
            model_id="m",
            ledger=ledger1,
        )
        ledger1.accumulate(
            _response(input_tokens=7, output_tokens=3, total_cost=0.30), "child"
        )
        tracker.fold(
            snapshot_cost_usd=snap1_cost,
            snapshot_tokens=snap1_tokens,
            run_ledger=ledger1,
        )
        run1_cost = tracker.total_cost_usd
        run1_tokens = tracker.total
        # Run 2 starts where run 1 left off.
        snap2_cost = tracker.total_cost_usd
        snap2_tokens = tracker.total
        ledger2 = CostLedger()
        tracker.record(
            _response(input_tokens=4, output_tokens=2, total_cost=0.05),
            model_id="m",
            ledger=ledger2,
        )
        ledger2.accumulate(
            _response(input_tokens=11, output_tokens=6, total_cost=0.15), "child"
        )
        tracker.fold(
            snapshot_cost_usd=snap2_cost,
            snapshot_tokens=snap2_tokens,
            run_ledger=ledger2,
        )
        # Cumulative across both runs.
        assert tracker.total_cost_usd == pytest.approx(run1_cost + 0.05 + 0.15)
        assert tracker.total.input_tokens == run1_tokens.input_tokens + 4 + 11
        assert tracker.total.output_tokens == run1_tokens.output_tokens + 2 + 6


class TestRestore:
    def test_overwrites_cumulative_totals(self) -> None:
        """``restore`` is the disk-resume counterpart to ``fold``: both
        funnel total mutation through the tracker rather than poking
        its fields from outside.
        """
        tracker = CostTracker()
        tracker.record(
            _response(input_tokens=999, output_tokens=999, total_cost=99.0),
            model_id="m",
            ledger=None,
        )
        tracker.restore(
            total_cost_usd=1.50,
            total=TokenCount(input_tokens=100, output_tokens=50),
        )
        assert tracker.total_cost_usd == pytest.approx(1.50)
        assert tracker.total.input_tokens == 100
        assert tracker.total.output_tokens == 50


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
