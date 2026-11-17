"""Contracts for Plan: the plan it accepts, the ones it refuses, and its summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from treekle import immutable

import pytest

from sagent.session.types import PlanResult
from sagent.tools.plan import MAX_STEPS, Plan
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


def _plan(*statuses: str) -> list[PlainTree]:
    return [{"step": f"Step {k}", "status": s} for k, s in enumerate(statuses, 1)]


@pytest.mark.asyncio
async def test_a_plan_says_how_far_it_is(tmp_path: Path) -> None:
    result = await settled(
        Plan().call(
            ToolContext(cwd=tmp_path),
            # Frozen, as the runtime hands a tool the model's arguments.
            immutable({"plan": _plan("completed", "in_progress", "pending")}),
        ),
        call_id="c1",
    )

    assert result.result == PlanResult()
    # The step in progress is named, so the model ticks it off once it is done
    # rather than all steps at the end: live runs did (2026-11-16), their work in
    # parallel batches, where a Plan call of its own costs a round.
    assert result.stdout == (
        (
            "Plan updated: 1 of 3 done. Once 'Step 2' is done, mark it completed in"
            " a Plan call sent with the next step's first tool calls."
        ),
    )
    assert result.stderr == ()


@pytest.mark.parametrize(
    ("plan", "said"),
    [
        (_plan(*["pending"] * 12), "Plan updated: 0 of 12 done."),
        (_plan("completed", "pending"), "Plan updated: 1 of 2 done."),
        (_plan("completed", "completed"), "Plan updated: 2 of 2 done."),
    ],
    ids=["most-steps", "none-in-progress", "all-done"],
)
@pytest.mark.asyncio
async def test_a_plan_may_hold_twelve_steps_and_none_in_progress(
    tmp_path: Path,
    plan: list[PlainTree],
    said: str,
) -> None:
    result = await settled(
        Plan().call(
            ToolContext(cwd=tmp_path),
            {"plan": plan, "explanation": "Reading done; the parser is next."},
        ),
    )

    assert (result.result, result.stdout) == (PlanResult(), (said,))


@pytest.mark.parametrize(
    ("plan", "why"),
    [
        ([], "A plan needs at least one step."),
        (_plan(*["pending"] * 13), "A plan holds at most 12 steps, got 13."),
        (
            _plan("in_progress", "in_progress"),
            "Only one step may be in_progress, got 2.",
        ),
        ([{"step": "  ", "status": "pending"}], "Step 1 has no text."),
        ([{"step": "Read", "status": "done"}], "'status' must be one of"),
        ([{"step": "Read"}], "Missing argument 'status'"),
    ],
)
@pytest.mark.asyncio
async def test_a_plan_that_does_not_fit_is_refused_saying_why(
    tmp_path: Path,
    plan: list[PlainTree],
    why: str,
) -> None:
    result = await settled(Plan().call(ToolContext(cwd=tmp_path), {"plan": plan}))

    assert result.result == PlanResult(exit_code=1)
    assert result.stdout == ()
    assert why in "".join(result.stderr)


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        (
            {"plan": _plan("completed", "in_progress", "pending")},
            "Plan 1 of 3 done · Step 2",
        ),
        ({"plan": _plan("completed", "completed")}, "Plan 2 of 2 done"),
        ({"plan": _plan("pending", "pending")}, "Plan 0 of 2 done"),
        ({"plan": "three steps"}, "Plan"),
        ({}, "Plan"),
        # A plan the tool refuses is no plan: the screen gives it no count.
        ({"plan": _plan(*["pending"] * 13)}, "Plan"),
        ({"plan": _plan("in_progress", "in_progress")}, "Plan"),
    ],
)
def test_summary_counts_the_steps_done_and_names_the_one_in_progress(
    arguments: dict[str, PlainTree],
    summary: str,
) -> None:
    assert Plan().summary(arguments) == summary


@pytest.mark.parametrize(
    ("plan", "said"),
    [
        (
            _plan("completed", "in_progress"),
            (
                "Plan updated: 1 of 2 done. Once 'Step 2' is done, mark it completed in"
                " a Plan call before you answer."
            ),
        ),
        # A step reopened, every step after it done: none is next.
        (
            _plan("in_progress", "completed"),
            (
                "Plan updated: 1 of 2 done. Once 'Step 1' is done, mark it completed in"
                " a Plan call before you answer."
            ),
        ),
    ],
    ids=["last", "reopened"],
)
@pytest.mark.asyncio
async def test_the_last_step_left_is_to_be_ticked_before_the_answer(
    tmp_path: Path,
    plan: list[PlainTree],
    said: str,
) -> None:
    result = await settled(Plan().call(ToolContext(cwd=tmp_path), {"plan": plan}))

    assert result.stdout == (said,)


def test_the_model_is_told_the_limit_the_tool_holds_it_to() -> None:
    assert f"At most {MAX_STEPS} steps." in Plan.description


def test_a_plan_runs_beside_anything(tmp_path: Path) -> None:
    assert Plan().serialize_key(ToolContext(cwd=tmp_path), {"plan": []}) is None


def test_the_schema_asks_for_codexs_plan_shape() -> None:
    properties = Plan.schema["properties"]
    assert isinstance(properties, dict)
    plan = properties["plan"]
    assert isinstance(plan, dict)
    assert plan["items"] == {
        "type": "object",
        "properties": {
            "step": {
                "type": "string",
                "description": "What this step does, in 5-7 words.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "Where it stands.",
            },
        },
        "required": ["step", "status"],
        "additionalProperties": False,
    }
    assert properties["explanation"] == {
        "type": "string",
        "description": "One line on where things stand, or why the plan changed.",
    }
    assert Plan.schema["required"] == ["plan"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
