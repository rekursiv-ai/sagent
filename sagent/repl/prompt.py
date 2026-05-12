"""``PromptToolkitInputSource``: prompt-toolkit-backed :class:`InputSource`.

Wraps a :class:`prompt_toolkit.PromptSession` and yields lines via
``next_line()``. Returns ``None`` when the user types ``exit`` or
``quit`` (lowercase) so the REPL input pump shuts the loop down
cleanly. Other exit signals (Ctrl-D / EOFError, Ctrl-C while idle /
KeyboardInterrupt) also map to ``None``.

Pending-user buffer
~~~~~~~~~~~~~~~~~~~

The runtime's ``GatedDeque`` doesn't support tag-based peek / pop,
so the REPL keeps its own ``list[str]`` of texts the user typed while
the runtime was busy. New input either commits-now (push
``UserMessage`` to ``agent.runtime.inbox``) or queues into this list
depending on whether the agent has foreground work in flight. Up
arrow pops the latest entry; the dynamic prompt shows it as a dim
preview while the agent is busy.

Drain on quit
~~~~~~~~~~~~~

When the user types ``quit`` / ``exit`` we surface the most recent
pending text (if any) so they see what got discarded before the
session closes.
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
    """Async input source backed by a :class:`prompt_toolkit.PromptSession`."""

    pending: list[str]
    """List of texts the user typed while the agent was busy. New input
    appends; Up pops the latest; the dynamic prompt previews the tail."""

    def __init__(
        self,
        session: PromptSession[str],
        *,
        pending: list[str],
        console: Console | None = None,
    ) -> None:
        self._session = session
        self.pending = pending
        self._console = console

    @override
    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        try:
            text = await self._session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            self._surface_pending_on_quit()
            return None
        stripped = text.strip()
        if stripped.lower() in QUIT_WORDS:
            self._surface_pending_on_quit()
            return None
        return text

    def _surface_pending_on_quit(self) -> None:
        """Surface the tail of the pending buffer before the loop ends."""
        if not self.pending or self._console is None:
            return
        tail = self.pending[-1]
        preview = tail.replace("\n", " ")[:80]
        self._console.print(
            Text(f"[discarding queued message: {preview}]", style="dim yellow"),
        )
        self.pending.clear()


def dynamic_prompt(agent: Agent, pending: list[str]) -> FormattedText:
    """Build the dynamic prompt with a dim preview of the tail pending text.

    Only renders when the agent is *busy* (a model call or compaction
    is in flight, or a tool cohort is outstanding): then the user's
    typed message is genuinely waiting and the preview is honest UX.
    When idle, the text is about to be committed and surfacing it as
    "queued" during that brief race window is misleading.

    ``Up`` lifts the preview message back into the buffer for editing.

    Args:
      agent: Agent whose busy state gates the preview.
      pending: REPL-local pending-text buffer (tail is previewed).

    Returns:
      formatted: The prompt's formatted text.

    """
    parts: list[tuple[str, str]] = []
    is_busy = agent.work is not None or bool(agent.runtime.cohort)
    if is_busy and pending:
        parts.append(("class:queued", _collapse_preview(pending[-1])))
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
