"""Tests for ``repl.prompt``: dynamic prompt + pending-buffer surfacing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

import asyncio

from prompt_toolkit.formatted_text import FormattedText

from sagent.agent.agent import Agent
from sagent.repl.prompt import (
    PromptToolkitInputSource,
    _collapse_preview,
    dynamic_prompt,
)


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    cohort: set[str] = field(default_factory=set)


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    work: object = None
    runtime: _FakeRuntime = field(default_factory=_FakeRuntime)


def _as_agent(a: _FakeAgent) -> Agent:
    return cast(Agent, a)


def test_dynamic_prompt_idle_no_preview() -> None:
    """Idle agent renders only the ``> `` prompt token."""
    fp = dynamic_prompt(_as_agent(_FakeAgent()), ["queued"])
    assert isinstance(fp, FormattedText)
    assert list(fp) == [("class:prompt", "> ")]


def test_dynamic_prompt_busy_with_pending_renders_preview() -> None:
    a = _FakeAgent(work=object())
    fp = dynamic_prompt(_as_agent(a), ["hello world"])
    parts = list(fp)
    assert parts[0] == ("class:queued", "hello world")
    assert parts[1] == ("", "\n")
    assert parts[2] == ("class:prompt", "> ")


def test_dynamic_prompt_busy_no_pending_omits_preview() -> None:
    a = _FakeAgent(work=object())
    fp = dynamic_prompt(_as_agent(a), [])
    assert list(fp) == [("class:prompt", "> ")]


def test_dynamic_prompt_cohort_busy_renders_preview() -> None:
    a = _FakeAgent()
    a.runtime.cohort.add("c1")
    fp = dynamic_prompt(_as_agent(a), ["mid-cohort"])
    parts = list(fp)
    assert parts[0] == ("class:queued", "mid-cohort")


def test_dynamic_prompt_only_previews_tail() -> None:
    a = _FakeAgent(work=object())
    fp = dynamic_prompt(_as_agent(a), ["old1", "old2", "tail"])
    parts = list(fp)
    assert parts[0] == ("class:queued", "tail")


def test_dynamic_prompt_compacting_busy() -> None:
    """``compact_task`` shows up as ``agent.work`` (truthy) -> busy."""
    a = _FakeAgent(work=object())
    fp = dynamic_prompt(_as_agent(a), ["queued during compact"])
    parts = list(fp)
    assert parts[0] == ("class:queued", "queued during compact")


def test_collapse_preview_short_passthrough() -> None:
    assert _collapse_preview("hello") == "hello"


def test_collapse_preview_truncates_long_first_line() -> None:
    out = _collapse_preview("x" * 100, width=20)
    assert out.endswith("…")
    assert len(out) == 20


def test_collapse_preview_extra_lines_suffix() -> None:
    out = _collapse_preview("line1\nline2\nline3")
    assert out == "line1 (+2 more lines)"


def test_collapse_preview_one_extra_line_singular() -> None:
    out = _collapse_preview("line1\nline2")
    assert out == "line1 (+1 more line)"


def test_collapse_preview_extra_paragraph_suffix() -> None:
    out = _collapse_preview("para1\n\npara2\n\npara3")
    assert out == "para1 (+2 more paragraphs)"


def test_collapse_preview_one_extra_paragraph_singular() -> None:
    out = _collapse_preview("para1\n\npara2")
    assert out == "para1 (+1 more paragraph)"


def test_collapse_preview_strips_trailing_newlines() -> None:
    assert _collapse_preview("hello\n\n") == "hello"


def test_next_line_returns_typed_text() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        return "hello"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, pending=[])
    line = asyncio.run(src.next_line())
    assert line == "hello"


def test_next_line_quit_returns_none() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        return "/quit"

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, pending=[])
    line = asyncio.run(src.next_line())
    assert line is None


def test_next_line_eof_returns_none() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        raise EOFError

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, pending=[])
    line = asyncio.run(src.next_line())
    assert line is None


def test_next_line_keyboard_interrupt_returns_none() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        raise KeyboardInterrupt

    session.prompt_async = _prompt_async
    src = PromptToolkitInputSource(session, pending=[])
    line = asyncio.run(src.next_line())
    assert line is None


def test_quit_surfaces_pending_preview() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        return "/quit"

    session.prompt_async = _prompt_async
    console = MagicMock()
    pending = ["queued line"]
    src = PromptToolkitInputSource(session, pending=pending, console=console)
    line = asyncio.run(src.next_line())
    assert line is None
    console.print.assert_called_once()
    assert pending == []


def test_quit_without_console_swallows_preview() -> None:

    session = MagicMock()

    async def _prompt_async() -> str:
        return "/quit"

    session.prompt_async = _prompt_async
    pending = ["queued"]
    src = PromptToolkitInputSource(session, pending=pending, console=None)
    line = asyncio.run(src.next_line())
    assert line is None
    # pending list left alone when there's no console to surface to.
    assert pending == ["queued"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
