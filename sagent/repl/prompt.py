"""``PromptToolkitInputSource``: prompt-toolkit-backed :class:`InputSource`.

Wraps a :class:`prompt_toolkit.PromptSession` and yields lines via
``next_line()``. Returns ``None`` when the user types ``exit`` or
``quit`` (lowercase) so the REPL input pump shuts the loop
down cleanly. Other exit signals (Ctrl-D / EOFError, Ctrl-C while
idle / KeyboardInterrupt) also map to ``None``.

Drain on quit: when the user types ``quit``/``exit`` we drain any
queued (uncommitted) inbox text first and surface a one-line preview
so the user sees what got discarded before the session closes.

Also exposes :func:`dynamic_prompt` -- the callable handed to
``PromptSession`` so the prompt shows a dim preview of the queued
tail entry (mirrors v1's ``input_pane.dynamic_prompt``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from prompt_toolkit.formatted_text import FormattedText
from rich.text import Text

from sagent.repl.input import InputSource
from sagent.repl.slash import QUIT_WORDS


if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from rich.console import Console

    from sagent.agent.agent import Agent


class PromptToolkitInputSource(InputSource):
    """Async input source backed by a :class:`prompt_toolkit.PromptSession`.

    Attributes:
      session: The owning prompt-toolkit session.
      agent: Agent whose inbox to drain on quit (optional).
      console: Console used for the discard-preview line on quit.

    """

    def __init__(
        self,
        session: PromptSession[str],
        *,
        agent: Agent | None = None,
        console: Console | None = None,
    ) -> None:
        self._session = session
        self._agent = agent
        self._console = console

    @override
    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        try:
            text = await self._session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            self._drain_for_quit()
            return None
        stripped = text.strip()
        if stripped.lower() in QUIT_WORDS:
            self._drain_for_quit()
            return None
        return text

    def _drain_for_quit(self) -> None:
        """Surface and drain any queued inbox text before the loop ends."""
        if self._agent is None:
            return
        tail = self._agent.inbox.peek_tail()
        if tail is None or self._console is None:
            self._agent.inbox.drain()
            return
        preview = str(tail.content).replace("\n", " ")[:80]
        self._console.print(
            Text(f"[discarding queued message: {preview}]", style="dim yellow"),
        )
        self._agent.inbox.drain()


def dynamic_prompt(agent: Agent) -> FormattedText:
    """Build the dynamic prompt with a dim preview of the queued tail entry.

    When the inbox has a queued user message at its tail, render a one-line
    preview above the ``> `` prompt so the user sees what's about to be sent.

    Args:
      agent: Agent whose inbox is inspected for queued text.

    Returns:
      prompt: Formatted text fragments for prompt-toolkit.

    """
    parts: list[tuple[str, str]] = []
    tail = agent.inbox.peek_tail()
    if tail is not None and tail.descriptor == "text/x-user-message":
        parts.append(("class:queued", _collapse_preview(str(tail.content))))
        parts.append(("", "\n"))
    parts.append(("class:prompt", "> "))
    return FormattedText(parts)


def _collapse_preview(text: str, width: int = 60) -> str:
    """Collapse multi-line text into a one-line preview with a count suffix."""
    text = text.rstrip("\n")
    paras = text.split("\n\n")
    first = paras[0].split("\n")[0]
    if len(first) > width:
        first = first[: width - 1] + "…"
    extra_paras = len(paras) - 1
    if extra_paras > 0:
        return (
            f"{first} (+{extra_paras} more paragraph{'s' if extra_paras != 1 else ''})"
        )
    extra_lines = len(paras[0].split("\n")) - 1
    if extra_lines > 0:
        return f"{first} (+{extra_lines} more line{'s' if extra_lines != 1 else ''})"
    return first
