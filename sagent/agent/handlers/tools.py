"""ToolBatchHandler and ToolBatchResultHandler.

Handles ``multipart/x-tool-batch`` (list of tool calls from the model).
Distinguishes foreground from background tool calls; runs foreground
calls with read-only batching and serializing rules; appends a
placeholder for each background call and spawns the worker; posts
``multipart/x-tool-batch-result`` when foreground dispatch finishes.

``ToolBatchResultHandler`` then routes each tool result to history
and triggers the next ``text/x-model-call``.

Compact / recompact / clear flags raised by tools during dispatch are
converted to inbox messages here -- this is the bridge between the
existing ``tool_state`` flag mechanism and the new descriptor spine.
A future slice will migrate the tools themselves to post messages
directly and remove the flag conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import asyncio
import dataclasses
import logging
import time

from sagent.agent.dispatch import (
    add_tool_input_batch_hint,
    conditional_rules_for_request,
    invoke_tool,
    is_request_read_only,
    tc_directive,
    tc_tool_id,
    tool_call_label,
)
from sagent.agent.handlers.base import (
    InlineHandler,
    SpawnedHandler,
)
from sagent.agent.session_io import text_from_msg
from sagent.custom_types import (
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.descriptors import has_error
from sagent.lib.json import bool_val, int_val
from sagent.lib.message import get_queue_id, get_tool_name
from sagent.tools.background_task import BackgroundTaskEntry
from sagent.tools.result_storage import (
    enforce_message_budget,
    inject_empty_marker,
    persist_result,
)


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.agent.agent import Agent
    from sagent.tools.core import ToolState
    from sagent.tools.result_storage import ReplacementState


logger = logging.getLogger(__name__)


class ToolBatchHandler(SpawnedHandler):
    """Handle ``multipart/x-tool-batch``: dispatch tools, post the result batch.

    Foreground tools dispatch via the existing ``invoke_tool`` path
    with read-only batching and unsafe serialization. Background tools
    spawn workers that post their final result back to the inbox as
    ``text/x-user-message`` (existing behavior preserved).
    """

    descriptors: tuple[str, ...] = ("multipart/x-tool-batch",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Dispatch the batch, post the result envelope.

        On cancellation (``/abort`` mid-batch), synthesizes
        ``[interrupted]`` placeholders for any unfilled foreground tool
        calls before re-raising. Without that step the assistant
        message's ``tool_use`` blocks would have no matching
        ``tool_result`` and the next provider request would 400 with
        ``tool_use ids were found without tool_result blocks``.
        """
        tool_calls = list(cast("tuple[Message, ...]", msg.content))
        if tool_calls:
            agent.activity.num_tool_call_rounds += 1
        fg_calls: list[Message] = []
        for req in tool_calls:
            # Surface a UI label for each call BEFORE dispatch so the
            # user sees what's running while the tool executes.
            tool = agent._tools.get(tc_tool_id(req))  # noqa: SLF001 -- handler intimate
            label = tool_call_label(tool, req)
            _ = agent.inbox.put(TextMessage(label, "text/x-tool-label"))
            directive = tc_directive(req)
            delay = int_val(directive.get("delay"), 0)
            background = bool_val(directive.get("background"), False) or delay > 0
            if background:
                _dispatch_background(agent, req, delay)
            else:
                fg_calls.append(req)
        try:
            results = await _dispatch_foreground(agent, fg_calls) if fg_calls else []
        except asyncio.CancelledError:
            # Post each synthetic tool_result directly so HistoryHandler
            # appends them, but skip the batch-result envelope so
            # ToolBatchResultHandler does NOT auto-post a follow-up
            # text/x-model-call. After abort, control returns to the
            # user instead of immediately re-engaging the model.
            for req in fg_calls:
                _ = agent.inbox.put(_interrupted_result(req))
            raise
        _ = agent.inbox.put(
            MultipartMessage(
                tuple(results),
                "multipart/x-tool-batch-result",
                parent_id=msg.id,
            ),
        )
        _drain_tool_state_signals(agent)


def _interrupted_result(req: Message) -> Message:
    """Synthetic ``[interrupted]`` tool-result for an unfilled foreground call."""
    return MultipartMessage(
        (
            TextMessage(get_queue_id(req), "text/x-queue-id"),
            TextMessage("[interrupted]", "text/x-error"),
        ),
        "multipart/x-tool-result",
        parent_id=req.id,
    )


class ToolBatchResultHandler(InlineHandler):
    """Handle ``multipart/x-tool-batch-result``: fan out + trigger next call.

    Posts each individual ``multipart/x-tool-result`` to the inbox so
    ``HistoryHandler`` and any render handlers see them. Then posts a
    follow-up ``text/x-model-call`` to advance the conversation.
    """

    descriptors: tuple[str, ...] = ("multipart/x-tool-batch-result",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Fan out each tool result; post a model-call to continue."""
        for result in cast("tuple[Message, ...]", msg.content):
            agent.inbox.put(result)
        agent.inbox.put(TextMessage("", "text/x-model-call", parent_id=msg.id))


async def _dispatch_foreground(
    agent: Agent,
    tool_calls: list[Message],
) -> list[Message]:
    """Execute foreground calls with the existing batching rules."""
    raw_results = await _dispatch_tools(agent, tool_calls)
    tool_names = {get_queue_id(tc): tc_tool_id(tc) for tc in tool_calls}
    results = enforce_message_budget(
        raw_results,
        tool_names,
        agent.replacement_state,
    )
    results = add_tool_input_batch_hint(results)
    return [_wrap_errors_for_llm(r) for r in results]


async def _dispatch_tools(
    agent: Agent,
    requests: list[Message],
) -> list[Message]:
    """Read-only-batched, serialized otherwise; mirrors ``Agent._dispatch_tools``."""
    agent.tool_state.bash_parse_cache.clear()
    for req in requests:
        agent.log_event(
            "tool_call",
            tool=tc_tool_id(req),
            tool_id=get_queue_id(req),
            input=tc_directive(req),
        )
    safe = [is_request_read_only(r, agent.tool_state) for r in requests]
    batches = _partition_batches(safe)
    results: dict[int, Message] = {}
    for batch in batches:
        if len(batch) == 1:
            results[batch[0]] = await _invoke_tool_safe(agent, requests[batch[0]])
        else:
            done = await asyncio.gather(
                *(_invoke_tool_safe(agent, requests[i]) for i in batch),
            )
            results.update(dict(zip(batch, done, strict=True)))
    ordered = [results[i] for i in range(len(requests))]
    finalized = _postprocess_results(
        requests,
        ordered,
        agent.replacement_state,
        agent.tool_state,
    )
    for r in finalized:
        agent.log_event(
            "tool_result",
            tool_id=get_queue_id(r),
            is_error=has_error(r),
            content_len=len(text_from_msg(r)),
        )
    return finalized


async def _invoke_tool_safe(agent: Agent, req: Message) -> Message:
    """Invoke ``req``; convert any exception to a structured tool-result error.

    For streaming tools, intermediate yields land on a local queue that
    we drain into the agent's inbox after the call completes so they
    surface to RenderStream / the run-handle iterator.
    """
    events: asyncio.Queue[Message | None] = asyncio.Queue()
    try:
        result = await invoke_tool(agent._tools, req, events)  # noqa: SLF001 -- handler intimate
    except Exception as e:  # noqa: BLE001 -- dispatch safety net
        logger.debug("Tool %s raised", get_tool_name(req), exc_info=True)
        result = MultipartMessage(
            (
                TextMessage(get_queue_id(req), "text/x-queue-id"),
                TextMessage(f"{type(e).__name__}: {e}", "text/x-error"),
            ),
            "multipart/x-tool-result",
            parent_id=req.id,
        )
    while True:
        try:
            event = events.get_nowait()
        except asyncio.QueueEmpty:
            break
        if event is None:
            continue
        agent.inbox.put(event)
    return result


def _partition_batches(safe: list[bool]) -> list[list[int]]:
    """Group consecutive safe indices; emit each unsafe index alone."""
    batches: list[list[int]] = []
    for i, is_safe in enumerate(safe):
        if batches and is_safe and safe[batches[-1][0]]:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


def _postprocess_results(
    requests: list[Message],
    results: list[Message],
    replacement_state: ReplacementState,
    tool_state: ToolState,
) -> list[Message]:
    """Inject empty markers, persist oversized results, append conditional rules."""
    seen_rules: set[Path] = set()
    out = list(results)
    for i, (req, r) in enumerate(zip(requests, results, strict=True)):
        text = text_from_msg(r)
        content = inject_empty_marker(get_tool_name(req), text)
        r_parts = cast("tuple[Message, ...]", r.content)
        if not has_error(r):
            preview = persist_result(
                get_queue_id(r),
                tc_tool_id(req),
                content,
                replacement_state,
            )
            if preview is not None:
                content = preview
            reminder = conditional_rules_for_request(req, tool_state, seen_rules)
            if reminder:
                content = content.rstrip() + "\n\n" + reminder
        if content != text:
            non_text = tuple(
                p for p in r_parts if p.descriptor not in ("text/plain", "text/x-error")
            )
            out[i] = dataclasses.replace(
                r,
                content=(
                    TextMessage(content, "text/plain"),
                    *non_text,
                ),
            )
    return out


def _wrap_errors_for_llm(tr: Message) -> Message:
    """Wrap ``text/x-error`` parts in ``<tool_use_error>`` for the LLM transcript."""
    changed = False
    parts: list[Message] = []
    for p in cast("tuple[Message, ...]", tr.content):
        if p.descriptor == "text/x-error":
            parts.append(
                dataclasses.replace(
                    p,
                    content=f"<tool_use_error>{p.content}</tool_use_error>",
                ),
            )
            changed = True
        else:
            parts.append(p)
    if not changed:
        return tr
    return dataclasses.replace(tr, content=tuple(parts))


def _dispatch_background(agent: Agent, req: Message, delay_sec: float) -> None:
    """Append a placeholder and spawn the background worker."""
    qid = get_queue_id(req)
    tool_name = get_tool_name(req)
    task = asyncio.create_task(_bg_worker(agent, req, delay_sec))
    agent.background_tasks[qid] = BackgroundTaskEntry(
        task=task,
        tool_name=tool_name,
        queue_id=qid,
        started=time.time(),
        delay_sec=delay_sec,
    )
    placeholder = MultipartMessage(
        (
            TextMessage(qid, "text/x-queue-id"),
            TextMessage(
                f"[Running in background: {tool_name}]",
                "text/plain",
            ),
        ),
        "multipart/x-tool-result",
        parent_id=req.id,
    )
    agent.inbox.put(_wrap_errors_for_llm(placeholder))


async def _bg_worker(
    agent: Agent,
    req: Message,
    delay_sec: float,
) -> Message:
    """Sleep, dispatch, post the result back as ``text/x-user-message``."""
    qid = get_queue_id(req)
    tool_name = get_tool_name(req)
    try:
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)
        result = await _invoke_tool_safe(agent, req)
        agent.inbox.put(
            TextMessage(
                f"[Background tool completed: {tool_name} ({qid})]\n\n"
                f"{text_from_msg(result)}",
                "text/x-user-message",
            ),
        )
        return result
    except asyncio.CancelledError:
        agent.inbox.put(
            TextMessage(
                f"[Background tool cancelled: {tool_name} ({qid})]",
                "text/x-user-message",
            ),
        )
        raise
    except Exception as e:
        agent.inbox.put(
            TextMessage(
                f"[Background tool failed: {tool_name} ({qid})]\n\n"
                f"{type(e).__name__}: {e}",
                "text/x-user-message",
            ),
        )
        raise
    finally:
        agent.background_tasks.pop(qid, None)


def _drain_tool_state_signals(agent: Agent) -> None:
    """Convert ``tool_state`` flags raised by tools into inbox messages.

    Bridge between the existing flag mechanism and the new descriptor
    spine. Slice 3+ migrates the Compact/Recompact/Clear tools to post
    messages directly; this helper is removed at that point.
    """
    if agent.tool_state.compact_requested is not None:
        instructions = agent.tool_state.compact_requested
        agent.tool_state.compact_requested = None
        agent.inbox.put(
            TextMessage(instructions or "", "text/x-compact-request"),
        )
    if agent.tool_state.recompact_requested is not None:
        instructions = agent.tool_state.recompact_requested
        agent.tool_state.recompact_requested = None
        agent.inbox.put(
            TextMessage(instructions or "", "text/x-uncompact-request"),
        )
    if agent.tool_state.clear_requested is not None:
        reason = agent.tool_state.clear_requested
        agent.tool_state.clear_requested = None
        agent.inbox.put_left(
            TextMessage(reason or "", "text/x-clear-request"),
        )
