"""Render persisted history into scrollback on resume.

Resumed sessions display the same scrollback the live REPL produces:
user messages as bars, model responses as Markdown, thinking blocks
as the dim "Thinking" preface, tool labels via each tool's own
``summary(args)``, tool results via the shared :func:`render_tool_result`.
Closes with a single footer containing count, spend, model, and modes.

Live and replay both go through ``Printer`` + ``render_tool_result``;
adding a new render concern lights up in both paths automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import functools

from sagent.agent.context import alive_splices, masked_refs_by_alive
from sagent.compaction.files import MICROCOMPACTED_ARGS_KEY
from sagent.repl.render import (
    make_render_observer,
    render_tool_result,
)
from sagent.tools.display import ToolDisplay, row_spec
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    CompactStarted,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    TapeEvent,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from sagent.agent.agent import Agent
    from sagent.repl.render import Printer, RenderObserver
    from sagent.types.tape import TapeRecord
    from sagent.types.tools import Tool


def replay_messages(agent: Agent, printer: Printer) -> None:
    """Render persisted history into scrollback.

    Walks the mask-resolved view of the tape: entries covered by an
    alive ``ContextSplice``'s mask vanish; each alive splice's payload
    renders via the same ladder. The resumed scrollback matches what
    the model sees going forward.

    Args:
      agent: Agent whose ``history`` to replay.
      printer: Printer that receives all replayed output.

    """
    tape = agent.runtime.tape
    if not tape:
        return
    tools = agent.tools_map
    policy = functools.partial(_replay_output_policy, tools, _tool_names_by_call(tape))
    # Replay is a one-shot pass over a frozen tape; snapshot
    # ``show_thinking`` once and thread the same bool through both the
    # observer (via a constant-returning closure) and ``_render_entry``.
    # A throwaway observer just for replay: replay rendering doesn't
    # attach to ``agent.observers`` (no live dispatch path), so we build
    # a one-shot instance and hand-feed it each tape event.
    show_thinking = agent.show_thinking
    render_event = make_render_observer(
        printer,
        show_thinking=lambda: show_thinking,
        output_policy=policy,
    )
    alive = alive_splices(tape)
    masked = masked_refs_by_alive(tape, alive)
    rendered_messages = 0
    for record in tape:
        if isinstance(record, ContextSplice):
            if record.ref not in alive:
                continue
            for payload_entry in record.payload:
                rendered_messages += _render_entry(
                    payload_entry,
                    printer=printer,
                    render_event=render_event,
                    tools=tools,
                    show_thinking=show_thinking,
                    output_policy=policy,
                )
            continue
        assert isinstance(record, ReferrableTapeEvent)
        # ``CompactStarted`` is a live, in-progress marker -- it renders as
        # "[compacting history…]". On resume the compaction it announced has
        # already completed (its ``CompactComplete`` follows in the tape), so
        # replaying it prints a misleading "compacting" line into static
        # scrollback. The completion summary is the durable record; drop the
        # transient start marker in both masked and non-masked ranges.
        if isinstance(record.event, CompactStarted):
            continue
        if record.ref in masked:
            # Non-payload runtime markers inside a masked range still surface
            # -- they are dispatch-only events, not the masked conversation
            # content.
            if not isinstance(
                record.event,
                (UserMessage, AgentSendMessage, AssistantMessage, ToolResult),
            ):
                render_event(record.event)
            continue
        rendered_messages += _render_entry(
            record.event,
            printer=printer,
            render_event=render_event,
            tools=tools,
            show_thinking=show_thinking,
            output_policy=policy,
        )
    parts = ["resumed", f"{rendered_messages} messages"]
    cost = agent.cost_tracker.spend.total
    if cost > 0:
        parts.append(f"${cost:.2f}")
    parts.extend(_mode_parts(agent))
    printer.write_line(f"── {' · '.join(parts)} ──")


def _mode_parts(agent: Agent) -> list[str]:
    """Return model and non-default mode fragments for the resume footer."""
    parts: list[str] = []
    spec = agent.model_recipe
    if spec is not None:
        parts.append(f"{spec.provider}/{spec.model_id}")
        parts.append(f"auth={spec.auth}")
        if spec.account:
            parts.append(f"account={spec.account}")
    if agent.thinking is not None:
        parts.append(f"thinking={agent.thinking}")
    if agent.effort is not None:
        parts.append(f"effort={agent.effort}")
    if agent.cache_ttl != "5m":
        parts.append(f"cache_ttl={agent.cache_ttl}")
    if agent.service_tier is not None:
        parts.append(f"service_tier={agent.service_tier}")
    if agent.latency is not None:
        parts.append(f"latency={agent.latency}")
    return parts


def _render_entry(
    entry: TapeEvent,
    *,
    printer: Printer,
    render_event: RenderObserver,
    tools: Mapping[str, Tool],
    show_thinking: bool,
    output_policy: Callable[[str], ToolDisplay],
) -> int:
    """Render one tape event; return 1 if it counts as a message, else 0."""
    match entry:
        case UserMessage(text=text):
            printer.write_user_bar(text)
            return 1
        case AgentSendMessage(source=source, text=text):
            printer.write_agent_bar(source, text)
            return 1
        case AssistantMessage(
            text=text,
            thinking_blocks=blocks,
            tool_calls=calls,
        ):
            if show_thinking:
                for block in blocks:
                    body = str(block.get("thinking") or block.get("text") or "")
                    if body:
                        printer.write_thinking(body)
            if text.strip():
                printer.write_markdown(text)
            for tc in calls:
                display = output_policy(tc.id)
                printer.write_tool_label(
                    _label_for_call(tc, tools),
                    command=display.command,
                    lang=display.command_lang,
                )
            return 1
        case ToolResult():
            render_tool_result(
                printer, entry, output=output_policy(entry.call_id).output
            )
            return 1
        case _:
            render_event(entry)
            return 0


def _tool_names_by_call(tape: Sequence[TapeRecord]) -> dict[str, str]:
    """Index ``call_id -> tool name`` for the whole tape, once.

    Live rendering resolves this through the agent's call registry,
    which a resumed session does not have. Recovering it by SCANNING the
    tape per call is quadratic: measured at 200/800/1600 calls, resume
    took 0.004/0.052/0.592s -- doubling the calls multiplied the work by
    up to eleven, so a long session stalls the pane on ``--resume``.

    Args:
      tape: Full session tape.

    Returns:
      names: Tool name for every call id the tape opened.

    """
    out: dict[str, str] = {}
    for record in tape:
        event = record.event if isinstance(record, ReferrableTapeEvent) else None
        if isinstance(event, AssistantMessage):
            for tc in event.tool_calls:
                out[tc.id] = tc.name
    return out


def _replay_output_policy(
    tools: Mapping[str, Tool],
    names_by_call: Mapping[str, str],
    call_id: str,
) -> ToolDisplay:
    """Return the output policy for the tool that produced ``call_id``."""
    tool = tools.get(names_by_call.get(call_id, ""))
    return ToolDisplay() if tool is None else row_spec(tool)


def _label_for_call(tc: ToolCall, tools: Mapping[str, Tool]) -> str:
    """Return the rendered label for ``tc``, honoring microcompacted stubs.

    Microcompaction preserves the original ``tool.summary(args)`` output
    inside ``args[MICROCOMPACTED_ARGS_KEY]`` so resume can still render
    the historical label even though the original args (``file_path``,
    ``cmd``, etc.) have been replaced. Without this check every
    microcompacted ``Read`` falls back to ``Read.summary``'s ``"?"``
    placeholder and the resumed scrollback loses every filename.

    The ``and stored`` guard rejects empty strings (not just non-str
    types): an empty stored label is no more useful than no label, so
    fall back to the live ``tool.summary`` rather than emit a blank
    scrollback entry on a session whose microcompactor stamped an empty
    placeholder.
    """
    stored = tc.args.get(MICROCOMPACTED_ARGS_KEY) if tc.args else None
    if isinstance(stored, str) and stored:
        return stored
    tool = tools.get(tc.name)
    return tool.summary(tc.args) if tool is not None else tc.name
