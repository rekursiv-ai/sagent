"""Tech-writer role factory.

The role label is ``tech-writer`` (hyphenated, matches chat/'s
convention). The Python module filename uses an underscore because
Python module names can't contain hyphens; the markdown source file
stays hyphenated to match the role label.
"""

from __future__ import annotations

from pathlib import Path

from .common import MODEL_HAIKU, build_agent


_ROLE_MD = Path(__file__).with_suffix("").with_name("tech-writer.md")


def build():
    """Construct the Tech-Writer Agent."""
    from sagent import tools

    return build_agent(
        role_name="tech-writer",
        role_md_path=_ROLE_MD,
        tools=[
            tools.Read(),
            tools.Grep(),
            tools.Glob(),
            tools.Edit(),
            tools.Write(),
            tools.Bash(),
            tools.WebSearch(),
            tools.WebFetch(),
        ],
        model_id=MODEL_HAIKU,
    )
