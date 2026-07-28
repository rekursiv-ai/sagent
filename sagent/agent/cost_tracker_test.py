"""Tests for ``agent.cost_tracker``: per-session cost accounting."""

from __future__ import annotations

import pytest

from sagent.agent.cost_tracker import CostTracker
from sagent.types.model import ModelResponse, TokenCount
from sagent.types.runtime import AssistantMessage


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


def test_cost_tracker_record_cost_accumulates_cost() -> None:
    t = CostTracker()
    t.record_cost(_make_response(total_cost=0.10))
    t.record_cost(_make_response(total_cost=0.25))
    assert t.total_cost_usd == pytest.approx(0.35)


def test_cost_tracker_record_cost_leaves_tokens_untouched() -> None:
    """``record_cost`` is cost-only: it must not move token totals."""
    t = CostTracker()
    t.record_cost(_make_response(total_cost=0.10, input_tokens=99))
    assert t.total == TokenCount()
    assert t.calls_by_model == {}


def test_cost_tracker_record_tokens_counts_calls_per_model() -> None:
    t = CostTracker()
    t.record_tokens(_make_response(), model_id="a")
    t.record_tokens(_make_response(), model_id="a")
    t.record_tokens(_make_response(), model_id="b")
    assert t.calls_by_model == {"a": 2, "b": 1}


def test_cost_tracker_record_tokens_leaves_cost_untouched() -> None:
    """``record_tokens`` is token-only: it must not move ``total_cost_usd``."""
    t = CostTracker()
    t.record_tokens(_make_response(total_cost=5.0, input_tokens=10), model_id="m")
    assert t.total_cost_usd == 0.0
    assert t.total == TokenCount(input_tokens=10)


def test_cost_tracker_record_tokens_updates_last_request() -> None:
    t = CostTracker()
    r = _make_response(total_cost=0.5)
    t.record_tokens(r, model_id="m")
    assert t.last_request is r.tokens


def test_cost_tracker_record_tokens_updates_last_response_time() -> None:
    t = CostTracker()
    before = t.last_response_time
    t.record_tokens(_make_response(), model_id="m")
    assert t.last_response_time >= before


def test_cost_tracker_restore_totals_overwrites_totals() -> None:
    t = CostTracker()
    t.record_cost(_make_response(total_cost=0.10))
    persisted_total = TokenCount()
    t.restore_totals(total_cost_usd=99.0, total=persisted_total)
    assert t.total_cost_usd == 99.0
    assert t.total is persisted_total


def test_cost_tracker_restore_totals_preserves_per_call_provenance() -> None:
    """The contract: only cumulative totals are restored.

    ``calls_by_model``, ``last_request``, and ``last_response_time``
    describe the *live* process's recorded calls; resume restarts that
    history. A future "restore everything" hook can extend the signature
    if a caller needs it.
    """
    t = CostTracker()
    t.record_tokens(_make_response(total_cost=0.10), model_id="old")
    before_calls = dict(t.calls_by_model)
    before_last_request = t.last_request
    before_last_response_time = t.last_response_time
    t.restore_totals(total_cost_usd=99.0, total=TokenCount())
    assert t.calls_by_model == before_calls
    assert t.last_request is before_last_request
    assert t.last_response_time == before_last_response_time


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
