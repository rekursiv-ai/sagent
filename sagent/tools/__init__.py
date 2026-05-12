"""Built-in tool implementations.

See ``sagent/__init__.py`` for the Tool contract, Message theory,
descriptor conventions, and error policy.

Peers
-----
Tools in the same agent can interact through the *peer* pattern.
A tool that accepts ``peers`` at construction inspects its siblings
for specific capabilities via duck-typing (Protocol check).

``Bash`` is the canonical example: it takes ``peers`` and checks
each for a ``bash_match(trees: Sequence[Node]) -> str | None``
method. When the model uses Bash for something a peer handles
better (e.g. ``grep foo .`` when ``Grep`` is available), the peer's
matcher fires and Bash prepends a ``<system-reminder>`` hint to its
result, nudging the model toward the dedicated tool.

This keeps tools loosely coupled -- Bash doesn't import Grep, it
just checks for the ``bash_match`` protocol.
"""

from __future__ import annotations

from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.background_task import BackgroundTask
from sagent.tools.bash import Bash
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    ToolState,
    changed_files_context,
    get_tool_state,
    has_been_read,
    mark_read,
    opt_int,
    opt_str,
    tool,
    tool_state_context,
)
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.linear import Linear
from sagent.tools.list import List
from sagent.tools.paper_author import PaperAuthor
from sagent.tools.paper_details import PaperDetails
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.paper_search import PaperSearch
from sagent.tools.play_audio import PlayAudio
from sagent.tools.read import Read
from sagent.tools.skill import Skill
from sagent.tools.slack import Slack
from sagent.tools.web_fetch import WebFetch
from sagent.tools.web_search import WebSearch
from sagent.tools.wiki import Wiki
from sagent.tools.write import Write


__all__ = (
    "TOOL_RESULT_MAX_CHARS",
    "AgentSelf",
    "AgentSend",
    "AgentSpawn",
    "BackgroundTask",
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "Linear",
    "List",
    "PaperAuthor",
    "PaperDetails",
    "PaperFetch",
    "PaperSearch",
    "PlayAudio",
    "Read",
    "Skill",
    "Slack",
    "ToolState",
    "WebFetch",
    "WebSearch",
    "Wiki",
    "Write",
    "changed_files_context",
    "get_tool_state",
    "has_been_read",
    "mark_read",
    "opt_int",
    "opt_str",
    "tool",
    "tool_state_context",
)
