"""Junior-SWE role factory.

Same edit scope and tool set as senior SWE, but capped at
``max_tool_call_rounds=20`` to enforce the escalation discipline.
"""

from __future__ import annotations

from pathlib import Path

from .common import MODEL_HAIKU, build_agent


_ROLE_MD = Path(__file__).with_suffix("").with_name("junior-swe.md")
_MAX_ROUNDS_BEFORE_ESCALATION = 20


def build():
    """Construct the Junior-SWE Agent."""
    from sagent import tools

    return build_agent(
        role_name="junior-swe",
        role_md_path=_ROLE_MD,
        tools=[
            tools.Read(),
            tools.Edit(),
            tools.Write(),
            tools.Bash(),
            tools.Grep(),
            tools.Glob(),
        ],
        model_id=MODEL_HAIKU,
        max_tool_call_rounds=_MAX_ROUNDS_BEFORE_ESCALATION,
    )
