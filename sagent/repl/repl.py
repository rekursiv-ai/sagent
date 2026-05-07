"""Agent request execution and REPL main loop."""

from __future__ import annotations

from pathlib import Path

import asyncio
import contextlib
import functools

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.text import Text

from sagent.agent import QUIT_SENTINEL, Agent
from sagent.custom_types import Message, TextMessage
from sagent.repl.custom_types import RenderState
from sagent.repl.handlers import (
    emit_markdown,
    flush_child_text_buffers,
    render_batch,
    replay_messages,
)
from sagent.repl.input_pane import dynamic_prompt
from sagent.repl.keybindings import build_key_bindings
from sagent.repl.render import (
    render_toolbar,
    set_terminal_title,
)
from sagent.repl.slash_commands import handle_slash_command


_DEFAULT_HISTORY = Path.home() / ".sagent_history"

_MAX_BATCH = 64
_FRAME_INTERVAL = 0.05  # 20fps max render rate


async def run_repl(
    agent: Agent,
    *,
    name: str = "agent",
    history: str | None = None,
) -> None:
    r"""Run the interactive REPL.

    Architecture:
    - ``_pump`` feeds ``PromptSession`` input into ``agent.inbox``.
    - ``agent.run_forever`` owns the request loop, reading from
      ``agent.inbox`` between model requests.
    - The render loop in this coroutine streams events from both tasks
      above the live prompt via ``patch_stdout``.
    - Ctrl+C (keybindings) sets ``abort_event`` and cancels
      ``agent.inflight``; the agent loop continues waiting for
      the next prompt.

    Args:
      agent: Agent instance to drive.
      name: Display name shown on startup.
      history: Path to the input history file.

    """
    with patch_stdout(raw=True):
        console = Console(stderr=True)
        out = Console(stderr=True)
        events: asyncio.Queue[Message | None] = asyncio.Queue()

        repl_style = PTStyle.from_dict(
            {
                "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
                "queued": "fg:ansibrightblack",
                "prompt": "bold",
            },
        )

        session: PromptSession[str] = PromptSession(
            functools.partial(dynamic_prompt, agent),
            history=FileHistory(str(history or _DEFAULT_HISTORY)),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=build_key_bindings(agent),
            multiline=True,
            enable_open_in_editor=False,
            bottom_toolbar=functools.partial(render_toolbar, agent),
            refresh_interval=0.2,
            style=repl_style,
            erase_when_done=True,
        )

        console.print(Text(f"[{name}] Ready. Type 'quit' to exit.", style="dim"))
        replay_messages(agent, console, out)
        if agent.status:
            set_terminal_title(agent.status)

        async def _drive_agent(
            a: Agent,
            q: asyncio.Queue[Message | None],
        ) -> None:
            async for event in a.run_forever():
                q.put_nowait(event)

        pump = asyncio.create_task(_pump(agent, session, console, events))
        agent_task = asyncio.create_task(_drive_agent(agent, events))
        render_frame = RenderState(console=console, out=out)
        try:
            while True:
                get_task = asyncio.create_task(events.get())
                done, _ = await asyncio.wait(
                    {get_task, agent_task, pump},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if (
                    pump in done
                    and not pump.cancelled()
                    and pump.exception() is not None
                ):
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                    pump.result()  # re-raise
                if get_task not in done:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
                    _drain(events, render_frame)
                    break
                event = get_task.result()
                if event is None:
                    _flush_render_frame(render_frame)
                    render_frame = RenderState(console=console, out=out)
                    continue
                batch: list[Message] = [event]
                stop = _collect_batch(events, batch)
                if not stop and agent.active:
                    await asyncio.sleep(_FRAME_INTERVAL)
                    stop = _collect_batch(events, batch)
                render_batch(batch, render_frame)
                if stop:
                    _flush_render_frame(render_frame)
                    render_frame = RenderState(console=console, out=out)
        finally:
            agent_task.cancel()
            pump.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(agent_task, pump)


async def _pump(
    agent: Agent,
    session: PromptSession[str],
    console: Console,
    events: asyncio.Queue[Message | None],
) -> None:
    """Read user input from the prompt and feed it into the agent inbox."""
    while True:
        try:
            text = await session.prompt_async()
        except (EOFError, KeyboardInterrupt):
            agent.inbox.put(QUIT_SENTINEL)
            return
        if text.strip().lower() in ("quit", "exit"):
            if agent.inbox:
                preview = (agent.inbox.peek_tail() or "").replace("\n", " ")[:80]
                console.print(
                    Text(
                        f"[discarding queued message: {preview}]",
                        style="dim yellow",
                    ),
                )
                agent.inbox.drain()
            agent.inbox.put(QUIT_SENTINEL)
            return
        stripped = text.strip()
        if handle_slash_command(agent, console, stripped):
            continue
        parts: list[str] = agent.inbox.drain()
        if stripped:
            parts.append(stripped)
        if not parts:
            continue
        full = "\n\n".join(parts)
        events.put_nowait(TextMessage(full, "text/x-signal-user-input"))
        agent.inbox.put(full)


def _collect_batch(
    events: asyncio.Queue[Message | None],
    batch: list[Message],
    cap: int = _MAX_BATCH,
) -> bool:
    """Drain more events into batch synchronously. Returns True if request boundary hit."""
    while len(batch) < cap:
        try:
            nxt = events.get_nowait()
        except asyncio.QueueEmpty:
            return False
        if nxt is None:
            return True
        batch.append(nxt)
    return False


def _drain(events: asyncio.Queue[Message | None], render_frame: RenderState) -> None:
    """Flush all buffered events after agent_task finishes."""
    while True:
        try:
            event = events.get_nowait()
        except asyncio.QueueEmpty:
            break
        if event is None:
            _flush_render_frame(render_frame)
            render_frame = RenderState(
                console=render_frame.console, out=render_frame.out
            )
        else:
            batch: list[Message] = [event]
            stop = _collect_batch(events, batch)
            render_batch(batch, render_frame)
            if stop:
                _flush_render_frame(render_frame)
                render_frame = RenderState(
                    console=render_frame.console, out=render_frame.out
                )
    _flush_render_frame(render_frame)


def _flush_render_frame(render_frame: RenderState) -> None:
    """Flush remaining text and child buffers for a completed frame."""
    flush_child_text_buffers(render_frame)
    remaining = render_frame.buf.strip()
    if remaining:
        render_frame.out.print()
        emit_markdown(render_frame.out, remaining)
        render_frame.out.print()
    elif render_frame.printed_header:
        render_frame.out.print()
    elif render_frame.done_event is not None:
        render_frame.console.print(Text("[no response]", style="dim"))
