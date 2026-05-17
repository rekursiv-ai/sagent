"""Tests for ``agent.cost_tracker``: per-session cost accounting."""

from __future__ import annotations

import pytest

from sagent.agent.cost_tracker import CostTracker
from sagent.types.history import AssistantMessage
from sagent.types.model import ModelResponse, TokenCount


def _make_response(*, total_cost: float = 0.0, **tokens: int) -> ModelResponse:
    """Build a ``ModelResponse`` with explicit token counts + cost."""
    return ModelResponse(
        message=AssistantMessage(text="x"),
        tokens=TokenCount(**tokens),
        total_cost=total_cost,
    )


def test_cost_tracker_defaults() -> None:
    t = CostTracker()
    assert t.total_cost_usd == 0.0
    assert isinstance(t.total, TokenCount)
    assert isinstance(t.last_request, TokenCount)
    assert t.calls_by_model == {}


def test_cost_tracker_record_accumulates_cost() -> None:
    t = CostTracker()
    t.record(_make_response(total_cost=0.10), model_id="m1")
    t.record(_make_response(total_cost=0.25), model_id="m1")
    assert t.total_cost_usd == pytest.approx(0.35)


def test_cost_tracker_record_counts_calls_per_model() -> None:
    t = CostTracker()
    t.record(_make_response(), model_id="a")
    t.record(_make_response(), model_id="a")
    t.record(_make_response(), model_id="b")
    assert t.calls_by_model == {"a": 2, "b": 1}


def test_cost_tracker_record_updates_last_request() -> None:
    t = CostTracker()
    r = _make_response(total_cost=0.5)
    t.record(r, model_id="m")
    assert t.last_request is r.tokens


def test_cost_tracker_record_updates_last_response_time() -> None:
    t = CostTracker()
    before = t.last_response_time
    t.record(_make_response(), model_id="m")
    assert t.last_response_time >= before


def test_cost_tracker_restore_overwrites_totals() -> None:
    t = CostTracker()
    t.record(_make_response(total_cost=0.10), model_id="m")
    persisted_total = TokenCount()
    t.restore(total_cost_usd=99.0, total=persisted_total)
    assert t.total_cost_usd == 99.0
    assert t.total is persisted_total


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
