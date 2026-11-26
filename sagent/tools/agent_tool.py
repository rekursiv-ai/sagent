"""Resolve agent-tool targets using the caller's authority."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
    from sagent.tools.types import ToolContext


def target_agent(
    team: Team,
    *,
    context: ToolContext,
    name: str,
    agent: str,
    reading: bool = False,
) -> Agent:
    """Resolve one local target and enforce the caller's mutation ceiling.

    Args:
      team: The caller's team.
      context: The calling agent's context.
      name: The tool being called.
      agent: A label or relation; empty targets the caller.
      reading: Whether this operation may read unrelated agents.

    Returns:
      target: The authorized local agent.

    Raises:
      ValueError: The caller, tool, or target is unavailable or unauthorized.

    """
    if context.label not in team:
        raise ValueError(f"{name} needs a calling agent in the team.")
    labels = team.resolve(agent, by=context.label) if agent else [context.label]
    if len(labels) != 1:
        raise ValueError(f"{agent!r} names {len(labels)} agents; name one.")
    (label,) = labels
    if (
        label != context.label
        and not reading
        and label
        not in team.resolve(
            "descendants",
            by=context.label,
        )
    ):
        raise ValueError("Mutations require self or a descendant.")
    if label not in team:
        raise ValueError(f"Agent {label!r} is not local to this team.")
    return team[label]
