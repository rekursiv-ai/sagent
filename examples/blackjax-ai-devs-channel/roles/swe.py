"""SWE role factory."""

from __future__ import annotations

from pathlib import Path

from .common import MODEL_SONNET, build_agent


_ROLE_MD = Path(__file__).with_suffix("").with_name("swe.md")


def build():
    """Construct the SWE Agent.

    SWE has full code-edit scope across blackjax/, sampling-book/, and
    tuningfork/ (excluding tuningfork/experiments/, which is reserved
    for the statistician). Tool set covers reading, editing, running
    tests, and coordinating with peers via the plugin's MCP server
    (``mcp__sagent_chat__sagent_send``).

    Model: sonnet (see ``project/.claude/agents/swe.md`` model: sonnet).
    """
    from sagent import tools

    return build_agent(
        role_name="swe",
        role_md_path=_ROLE_MD,
        tools=[
            tools.Read(),
            tools.Edit(),
            tools.Write(),
            tools.Bash(),
            tools.Grep(),
            tools.Glob(),
        ],
        model_id=MODEL_SONNET,
    )
