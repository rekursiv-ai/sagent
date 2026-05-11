"""Tests for ``CostTracker`` -- the single cost store."""

from __future__ import annotations

import pytest

from sagent.agent.cost_tracker import CostTracker
from sagent.custom_types import (
    Message,
    ModelResponse,
    MultipartMessage,
    TokenCount,
)


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
        )
        tracker.record(
            _response(input_tokens=20, output_tokens=8, total_cost=0.20),
            model_id="m",
        )
        assert tracker.total.input_tokens == 30
        assert tracker.total.output_tokens == 13
        assert tracker.total_cost_usd == pytest.approx(0.30)

    def test_per_model_call_counts(self) -> None:
        tracker = CostTracker()
        tracker.record(
            _response(input_tokens=1, output_tokens=1, total_cost=0.0),
            model_id="m1",
        )
        tracker.record(
            _response(input_tokens=1, output_tokens=1, total_cost=0.0),
            model_id="m1",
        )
        tracker.record(
            _response(input_tokens=1, output_tokens=1, total_cost=0.0),
            model_id="m2",
        )
        assert tracker.calls_by_model == {"m1": 2, "m2": 1}

    def test_last_request_overwrites_per_call(self) -> None:
        tracker = CostTracker()
        tracker.record(
            _response(input_tokens=10, output_tokens=5, total_cost=0.10),
            model_id="m",
        )
        tracker.record(
            _response(input_tokens=20, output_tokens=8, total_cost=0.20),
            model_id="m",
        )
        # last_request reflects only the most recent call, not the sum.
        assert tracker.last_request.input_tokens == 20
        assert tracker.last_request.output_tokens == 8


class TestRestore:
    def test_overwrites_cumulative_totals(self) -> None:
        """``restore`` is the disk-resume hook: overwrites totals from a
        persisted session before any new ``record`` calls are layered.
        """
        tracker = CostTracker()
        tracker.record(
            _response(input_tokens=999, output_tokens=999, total_cost=99.0),
            model_id="m",
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
