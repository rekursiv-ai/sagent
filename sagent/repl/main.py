"""``run_repl``: orchestrate an interactive REPL on top of ``agent``.

Builds a prompt-toolkit session with ``FileHistory`` + ``AutoSuggest``
+ bottom toolbar + multi-line input + custom keybindings, wires it to
:class:`PromptInputHandler` plus :func:`repl_handler_set`, posts the
bootstrap message, and calls ``agent.run_loop()``. Returns when the user
types ``quit`` or sends EOF.

Important: the ``rich.Console`` is constructed INSIDE the
``patch_stdout`` context. ``patch_stdout`` swaps ``sys.stdout`` /
``sys.stderr`` for a proxy that routes writes above the prompt.
``rich.Console`` snapshots its file handle at construction; if built
beforehand its writes bypass the proxy, collide with prompt-toolkit's
terminal control codes, and disappear under ``erase_when_done``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import functools

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

from sagent.custom_types import TextMessage
from sagent.repl.console import ConsolePrinter
from sagent.repl.input import PromptInputHandler
from sagent.repl.keybindings import build_key_bindings
from sagent.repl.prompt import (
    PromptToolkitInputSource,
    dynamic_prompt,
)
from sagent.repl.render import repl_handler_set
from sagent.repl.replay import replay_messages
from sagent.repl.toolbar import render_toolbar


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


_DEFAULT_HISTORY = Path.home() / ".sagent_history"


async def run_repl(
    agent: Agent,
    *,
    history: Path | None = None,
) -> None:
    """Drive ``agent`` interactively until the user types ``/quit``.

    Registers REPL handlers (input + render) on the supplied agent,
    then runs the dispatch loop. The rich console is constructed
    inside ``patch_stdout`` so its stderr reference picks up the
    proxy that routes writes above the prompt. Resumed sessions are
    replayed into scrollback before the prompt opens.

    Args:
      agent: The agent to drive.
      history: Path to the input-history file. ``None`` -> ``~/.sagent_history``.

    """
    history_path = history or _DEFAULT_HISTORY
    style = PTStyle.from_dict(
        {
            "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
            "queued": "fg:ansibrightblack",
            "prompt": "bold",
        },
    )
    with patch_stdout(raw=True):
        console = Console(stderr=True)
        session: PromptSession[str] = PromptSession(
            functools.partial(dynamic_prompt, agent),
            multiline=True,
            erase_when_done=True,
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=functools.partial(render_toolbar, agent),
            refresh_interval=0.2,
            key_bindings=build_key_bindings(agent),
            enable_open_in_editor=False,
            style=style,
        )
        printer = ConsolePrinter(console)
        for handler in repl_handler_set(printer):
            agent.register(handler)
        agent.register(
            PromptInputHandler(
                PromptToolkitInputSource(session, agent=agent, console=console),
                printer=printer,
            ),
        )
        replay_messages(agent, console, console)
        if agent.status:
            printer.set_terminal_title(agent.status)
        elif agent.name:
            printer.set_terminal_title(agent.name)
        agent.inbox.put(TextMessage("", "text/x-bootstrap"))
        await agent.run_loop()
