"""TL role factory."""

from __future__ import annotations

from pathlib import Path

from .common import MODEL_FABLE, build_agent


_ROLE_MD = Path(__file__).with_suffix("").with_name("tl.md")


def build():
    """Construct the TL Agent.

    TL is read-only by scope (no Edit/Write); coordinates by reading
    code, inspecting git history, and routing work to peers via the
    plugin's MCP server (``mcp__sagent_chat__sagent_send``). Bash is
    included for read-only verbs (git log/diff/show, ls, cat, grep,
    journalctl), but TL must not run mutating commands.

    Peer messaging + self-defer + status-report live in the MCP
    server's tool catalog and are NOT in ``tools=[...]``. See
    ``mcp_sagent/server.py`` and the per-agent ``--mcp-config``
    threaded by :func:`common.build_agent`.

    Model: opus (see ``project/.claude/agents/tl.md`` model: opus).
    """
    from sagent import tools

    return build_agent(
        role_name="tl",
        role_md_path=_ROLE_MD,
        tools=[
            tools.Read(),
            tools.Grep(),
            tools.Glob(),
            tools.Bash(),
            tools.WebSearch(),
            tools.WebFetch(),
        ],
        model_id=MODEL_FABLE,
    )
