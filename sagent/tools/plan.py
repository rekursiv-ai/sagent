"""Plan: the agent's steps for the work in hand, and where each stands.

The model sends the whole list each time, as codex's ``update_plan`` takes it: each
step's text and its status, one ``in_progress`` at most. Nothing is kept here: the
latest call is the plan, and the session's records hold every call, so a viewer of
the session draws the steps from them.

Typical usage example:

  step = {"step": "Read the failing test", "status": "in_progress"}
  result = await settled(Plan().call(context, {"plan": [step]}))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Final, Literal

from sagent.journal.types import PlanResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "MAX_STEPS",
    "Plan",
    "Step",
]


MAX_STEPS: Final = 12
"""The most steps a plan may hold."""


type Status = Literal[
    "pending",
    "in_progress",
    "completed",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """One step of a plan."""

    step: Annotated[str, "What this step does, in 5-7 words."]
    status: Annotated[Status, "Where it stands."]


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    """Say the steps of the work in hand, and where each stands."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        plan: Annotated[tuple[Step, ...], "Every step, in order."]
        explanation: Annotated[
            str,
            "One line on where things stand, or why the plan changed.",
        ] = ""

    name = "Plan"
    description = (
        (files(__package__) / "assets" / "plan.md")
        .read_text(encoding="utf-8")
        .format(max_steps=MAX_STEPS)
    )
    schema = json_schema(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return how far the plan is, as sagent's screen and a viewer's live frame show it.

        Args:
          arguments: What the model passed.

        Returns:
          summary: ``Plan 2 of 5 done · <the step in progress>``, the step left out when
            none is in progress; ``Plan`` alone for arguments ``call`` refuses.

        """
        try:
            plan = _checked(arguments)
        except ValueError:
            return "Plan"
        current = _current(plan)
        return f"Plan {_done(plan)}{f' · {current}' if current else ''}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: a plan shares nothing."""
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[PlanResult, str, str]:
        """Check the plan and say how far it is.

        Args:
          context: The calling agent's state; unused.
          arguments: What the model passed, as ``Plan.Arguments``.

        Returns:
          streaming_result: How many steps are done and, while one is in progress, a
            reminder to mark it completed once done; or why the plan was refused.

        """
        del context
        try:
            plan = _checked(arguments)
        except ValueError as e:
            return whole_output(
                ToolOutput(
                    stderr=(str(e),),
                    result=PlanResult(exit_code=1),
                ),
            )
        said = f"Plan updated: {_done(plan)}."
        current = _current(plan)
        if current:
            # Without this live runs set their plan, then ticked every step at the end:
            # they work in parallel batches, where a Plan call of its own costs a round.
            after = plan[[step.status for step in plan].index("in_progress") + 1 :]
            then = (
                "sent with the next step's first tool calls"
                if any(step.status != "completed" for step in after)
                else "before you answer"
            )
            said += (
                f" Once {current!r} is done, mark it completed in a Plan call {then}."
            )
        return whole_output(
            ToolOutput(stdout=(said,), result=PlanResult()),
        )


def _checked(arguments: Mapping[str, PlainTree]) -> tuple[Step, ...]:
    """Return the plan ``arguments`` set; raise ``ValueError`` saying why it is none."""
    plan = parse(arguments, Plan.Arguments).plan
    if not plan:
        raise ValueError("A plan needs at least one step.")
    if len(plan) > MAX_STEPS:
        raise ValueError(
            f"A plan holds at most {MAX_STEPS} steps, got {len(plan)}.",
        )
    active = sum(step.status == "in_progress" for step in plan)
    if active > 1:
        raise ValueError(f"Only one step may be in_progress, got {active}.")
    blank = next((k for k, step in enumerate(plan, 1) if not step.step.strip()), 0)
    if blank:
        raise ValueError(f"Step {blank} has no text.")
    return plan


def _done(plan: tuple[Step, ...]) -> str:
    """Return how far ``plan`` is, one count everywhere: ``2 of 5 done``."""
    done = sum(step.status == "completed" for step in plan)
    return f"{done} of {len(plan)} done"


def _current(plan: tuple[Step, ...]) -> str:
    """Return the text of the step in progress in ``plan``; empty when none is."""
    return next((step.step for step in plan if step.status == "in_progress"), "")
