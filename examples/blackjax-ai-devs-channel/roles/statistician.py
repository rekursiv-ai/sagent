"""Statistician role factory.

Sandbox: ``tuningfork/experiments/``. Resolved from the monorepo root
detected at build time so the same role module works on any host that
has the standard layout (``blackjax-devs/{blackjax,sampling-book,
tuningfork,claude-config}``).
"""

from __future__ import annotations

from pathlib import Path

from .common import MODEL_SONNET, build_agent


_ROLE_MD = Path(__file__).with_suffix("").with_name("statistician.md")


def _monorepo_root() -> Path:
    """Find the BlackJAX monorepo root by walking up from this file.

    Layout: ``<root>/<sagent-fork>/examples/blackjax-ai-devs-channel/roles/statistician.py``.
    Walking up 5 levels lands at the monorepo root (``blackjax-devs/``).
    """
    return Path(__file__).resolve().parents[5]


def _sandbox_root() -> Path:
    return _monorepo_root() / "tuningfork" / "experiments"


def build():
    """Construct the Statistician Agent.

    Read-anywhere. Edit/Write restricted to ``tuningfork/experiments/``
    via sandboxed tool wrappers (see ``sandboxed_tools.py``).
    """
    import sys

    from sagent import tools

    plugin_dir = str(Path(__file__).resolve().parent.parent)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    import sandboxed_tools

    sandbox = _sandbox_root()
    return build_agent(
        role_name="statistician",
        role_md_path=_ROLE_MD,
        tools=[
            tools.Read(),
            tools.Grep(),
            tools.Glob(),
            tools.Bash(),
            tools.WebSearch(),
            tools.WebFetch(),
            sandboxed_tools.SandboxedWrite(sandbox_root=sandbox),
            sandboxed_tools.SandboxedEdit(sandbox_root=sandbox),
        ],
        model_id=MODEL_SONNET,
    )
