"""Dynamic prompt for the REPL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.formatted_text import FormattedText


if TYPE_CHECKING:
    from sagent.agent import Agent


def dynamic_prompt(agent: Agent) -> FormattedText:
    """Build the dynamic prompt with a dim preview of the queued entry.

    Args:
      agent: Agent whose inbox is inspected for queued text.

    Returns:
      prompt: Formatted text fragments for prompt_toolkit.

    """
    parts: list[tuple[str, str]] = []
    tail = agent.inbox.peek_tail()
    if tail is not None:
        preview_text = str(tail.content)
        if tail.descriptor == "text/x-clear-request":
            preview_text = f"/clear {preview_text}" if preview_text else "/clear"
        parts.append(("class:queued", _collapse_prompt_preview(preview_text)))
        parts.append(("", "\n"))
    parts.append(("class:prompt", "> "))
    return FormattedText(parts)


def _collapse_prompt_preview(text: str, width: int = 60) -> str:
    r"""Collapse ``text`` to a one-line preview for the prompt.

    Paragraphs (``\n\n`` separated) count as units; bare newlines within
    a paragraph count as lines. Only the first line of the first
    paragraph is shown, with a ``(+N more …)`` suffix.
    """
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
