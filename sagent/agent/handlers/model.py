"""ModelCallHandler and ModelResponseHandler.

``ModelCallHandler`` builds a ``ModelRequest`` from the current
history, sends it via ``send_with_retry``, posts streaming text as
``text/plain`` events, emits ``text/x-thinking`` for any
thinking content in the response, and posts the final response message
back to the inbox. On a context-overflow error from the provider,
runs in-band compaction and retries up to ``MAX_OVERFLOW_RECOVERY``
times before raising. On a max-tokens / mid-stream truncation stop
reason, nudges the model to resume up to ``MAX_TRUNCATION_RECOVERY``
times.

``ModelResponseHandler`` enforces stop-reason invariants and routes
the response to tool dispatch when it carries tool calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast, override

import asyncio
import contextlib
import dataclasses
import logging

from sagent.agent.handlers.base import (
    InlineHandler,
    SpawnedHandler,
)
from sagent.agent.handlers.compact import (
    estimate_total_tokens,
    run_compaction,
)
from sagent.agent.retry import send_with_retry
from sagent.custom_exceptions import ModelTerminationError
from sagent.custom_types import (
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.descriptors import is_thinking
from sagent.lib.message import (
    response_tool_calls,
    thinking_text,
)
from sagent.providers.lib.stop_reason import BENIGN_STOP_REASONS
from sagent.tools.background_task import BackgroundAwareTool
from sagent.tools.core import cost_ledger_var


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


logger = logging.getLogger(__name__)


MAX_OVERFLOW_RECOVERY = 3
MAX_TRUNCATION_RECOVERY = 3

# Stop reasons that mean "response was cut off mid-stream and the
# model should resume". Mirrors v1's ``_RECOVERABLE_TRUNCATION``.
_RECOVERABLE_TRUNCATION = frozenset(
    {"max_tokens", "model_context_window_exceeded"},
)

# Re-prompt nudge for truncation recovery (parity with v1).
_TRUNCATION_NUDGE = (
    "Output token limit hit. Resume directly - no apology, no recap of "
    "what you were doing. Pick up mid-thought if that is where the cut "
    "happened. Break remaining work into smaller pieces."
)


class ModelCallHandler(SpawnedHandler):
    """Handle ``text/x-model-call``: send the request, post the response.

    Reads ``cache_ttl``, ``thinking``, and ``effort`` from the agent at
    each call so ``AgentSelf`` (cache_ttl swap) and ``/model`` (effort
    swap) take effect mid-session without re-registration.
    """

    descriptors: tuple[str, ...] = ("text/x-model-call",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Build a ``ModelRequest``, send with retry + overflow recovery.

        Acquires ``agent.model_lock`` for the full provider call, so
        multiple ``text/x-model-call`` messages arriving back-to-back
        serialize instead of issuing concurrent provider requests.

        Enforces ``agent.max_tool_call_rounds``: when the cumulative
        tool-call rounds reach the cap, posts a synthetic ``text/x-error``
        response in place of issuing another model call. The handle
        await unwraps that response as the final result, matching v1's
        ``ERROR_MAX_TOOL_CALL_ROUNDS`` contract.

        Stream chunks are posted via ``text/plain`` as the
        provider yields them. For buffer-only providers, the response's
        ``text/plain`` parts are synthesized into chunks so the
        renderer sees one rendering path. A trailing
        ``text/x-stream-end`` marker terminates the rendered line. Any
        thinking content in the response is emitted as
        ``text/x-thinking`` so RenderThinking can display it.
        """
        del msg
        if _round_limit_exceeded(agent):
            _post_round_limit_response(agent)
            return
        streamed_chars: int = 0

        def _emit_chunk(chunk: str) -> None:
            nonlocal streamed_chars
            streamed_chars += len(chunk)
            agent.inbox.put(TextMessage(chunk, "text/plain"))

        async with agent.model_lock:
            request = self._build_request(agent)
            try:
                response = await self._send_with_overflow_recovery(
                    agent,
                    request,
                    _emit_chunk,
                )
            except asyncio.CancelledError:
                # Account the partial response so the toolbar / cost
                # tracker reflect the cancelled call (parity with v1's
                # ``_estimate_cancelled_response``).
                _account_cancelled(agent, request, streamed_chars)
                agent.inbox.put(TextMessage("", "text/x-stream-end"))
                agent.inbox.put(TextMessage("[interrupted]", "text/x-interrupted"))
                raise
            agent.cost_tracker.record(
                response,
                model_id=agent.model.model_id,
                ledger=cost_ledger_var.get(),
            )
            if streamed_chars == 0:
                # Buffer-only provider; synthesize chunks so the
                # renderer never has to know whether streaming happened.
                for part in _text_parts(response.content):
                    agent.inbox.put(
                        TextMessage(part, "text/plain"),
                    )
            agent.inbox.put(TextMessage("", "text/x-stream-end"))
            # Surface thinking content (if any) so RenderThinking
            # displays it. Subscribers see one ``text/x-thinking`` per
            # response, never duplicated by Render*.
            for thinking in _thinking_blocks(response.content):
                agent.inbox.put(TextMessage(thinking, "text/x-thinking"))
            # Stash the stop reason on the activity tracker so
            # ``ModelResponseHandler`` can guard on it. Doing this
            # out-of-band keeps history identical to v1 (which
            # appends only ``response.content``); subsequent
            # provider requests rebuild from history and never see a
            # synthetic stop-reason part.
            agent.activity.last_stop_reason = response.stop_reason
            agent.inbox.put(response.content)
            # Truncation recovery: if the response was cut off
            # mid-stream and carried no tool calls, nudge the model to
            # resume. Bounded to ``MAX_TRUNCATION_RECOVERY`` attempts;
            # the response itself is still posted to history so the
            # partial output isn't lost.
            if _is_truncated_no_tools(response):
                _maybe_post_truncation_nudge(agent)

    def _build_request(self, agent: Agent) -> ModelRequest:
        """Snapshot history + system prompt into a ``ModelRequest``.

        ``thinking``, ``effort``, ``cache_ttl`` come from the agent so
        ``AgentSelf`` mutations (cache_ttl swap) take effect on the
        next call. ``BackgroundAwareTool`` wrapping for tools is
        applied when the agent has the BackgroundTask tool registered
        (mirrors v1's ``has_background_support`` path).
        """
        tools_list: list[Tool] = list(agent._tools.values())  # noqa: SLF001 -- handler intimate
        if tools_list and "application/x-tool-backgroundtask" in agent._tools:  # noqa: SLF001 -- handler intimate
            tools_list = [
                t
                if t.tool_id == "application/x-tool-backgroundtask"
                else cast("Tool", BackgroundAwareTool(t))
                for t in tools_list
            ]
        return ModelRequest(
            messages=list(agent.history),
            system=agent.system_prompt(),
            tools=tools_list or None,
            max_response_tokens=agent.max_response_tokens,
            thinking=agent.thinking,
            effort=agent.effort,
            cache_ttl=agent.cache_ttl,
        )

    async def _send_with_overflow_recovery(
        self,
        agent: Agent,
        request: ModelRequest,
        on_text: Callable[[str], None],
    ) -> ModelResponse:
        """Send; on context-overflow run compaction in-band and retry.

        Adaptive-thinking fallback: some models (e.g. Haiku 4.5) advertise
        ``supports_thinking=True`` but reject ``thinking="adaptive"`` with
        a 400. When the API explicitly says thinking isn't supported, we
        clear ``agent.thinking`` for the rest of the session and retry
        once -- so ``/model`` to a non-adaptive model just works.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_OVERFLOW_RECOVERY + 1):
            try:
                return await send_with_retry(
                    agent.model,
                    request,
                    on_text=on_text,
                    max_attempts=agent.max_attempts,
                    persistent_retry=agent.persistent_retry,
                    log_event=agent.log_event,
                    on_discarded_response=lambda r: agent.cost_tracker.record(
                        r,
                        model_id=agent.model.model_id,
                        ledger=cost_ledger_var.get(),
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if _is_thinking_unsupported(e) and request.thinking is not None:
                    logger.info(
                        "Model %r rejected thinking=%r; falling back to None",
                        agent.model.model_id,
                        request.thinking,
                    )
                    agent.set_thinking(None)
                    request = dataclasses.replace(request, thinking=None)
                    continue
                if not agent.model.is_context_overflow(e):
                    raise
                last_error = e
                if attempt >= MAX_OVERFLOW_RECOVERY:
                    break
                logger.info("Context overflow recovery attempt %d", attempt)
                _ = await run_compaction(agent, count_toward_budget=True)
                request = dataclasses.replace(
                    request,
                    messages=list(agent.history),
                    system=agent.system_prompt(),
                )
        msg = (
            "context overflow recovery failed after "
            f"{MAX_OVERFLOW_RECOVERY} compactions"
        )
        raise RuntimeError(msg) from last_error


def _text_parts(response: Message) -> list[str]:
    """Extract ``text/plain`` content from a ``multipart/x-model-message``."""
    if not isinstance(response, MultipartMessage):
        return []
    return [
        str(part.content)
        for part in response.content
        if part.descriptor == "text/plain"
    ]


def _thinking_blocks(response: Message) -> list[str]:
    """Extract thinking text from each thinking part in the response."""
    if not isinstance(response, MultipartMessage):
        return []
    out: list[str] = []
    for part in response.content:
        if is_thinking(part.descriptor):
            text = thinking_text(part)
            if text:
                out.append(text)
    return out


def _is_truncated_no_tools(response: ModelResponse) -> bool:
    """True if the response was cut off mid-stream and has no tool calls."""
    if response.stop_reason not in _RECOVERABLE_TRUNCATION:
        return False
    return not response_tool_calls(response.content)


def _maybe_post_truncation_nudge(agent: Agent) -> None:
    """Post a recovery nudge unless we've exhausted the retry budget.

    Attempt counter lives on ``agent.activity`` so it survives across
    handler invocations and resets cleanly on the next non-truncated
    response (handled by ``ModelResponseHandler``).
    """
    attempts = agent.activity.truncation_recoveries
    if attempts >= MAX_TRUNCATION_RECOVERY:
        agent.inbox.put(
            TextMessage(
                "[truncation recovery exhausted]",
                "text/x-error",
            ),
        )
        return
    agent.activity.truncation_recoveries = attempts + 1
    agent.inbox.put(TextMessage(_TRUNCATION_NUDGE, "text/x-user-message"))


def _account_cancelled(
    agent: Agent,
    request: ModelRequest,
    chars_streamed: int,
) -> None:
    """Estimate cost for a cancelled request and record it (parity w/ v1)."""
    estimated_input = max(
        agent.cost_tracker.last_request.input_tokens,
        estimate_total_tokens(request.system or "", request.messages, agent.model),
    )
    estimated_output = (
        max(1, chars_streamed // agent.budget.chars_per_token)
        if chars_streamed > 0
        else 0
    )
    pricing = agent.model.pricing
    estimated_cost = (
        estimated_input * pricing.request + estimated_output * pricing.response
    ) / 1_000_000
    partial = ModelResponse(
        content=TextMessage("", "text/plain"),
        tokens=TokenCount(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
        ),
        total_cost=estimated_cost,
    )
    with contextlib.suppress(RuntimeError):
        agent.cost_tracker.record(
            partial,
            model_id=agent.model.model_id,
            ledger=cost_ledger_var.get(),
        )


def _is_thinking_unsupported(exc: BaseException) -> bool:
    """Heuristic match for "thinking is not supported on this model" 400s.

    Anthropic returns this for ``thinking="adaptive"`` on Haiku 4.5 and
    similar; we recognize the message text rather than the SDK class so
    behavior survives library upgrades.
    """
    msg = str(exc).lower()
    return "thinking is not supported" in msg or (
        "thinking" in msg and "not supported" in msg
    )


def _round_limit_exceeded(agent: Agent) -> bool:
    """Return True if the agent has reached its tool-call-round cap."""
    cap = agent.max_tool_call_rounds
    if cap is None:
        return False
    return agent.activity.num_tool_call_rounds >= cap


def _post_round_limit_response(agent: Agent) -> None:
    """Synthesize a ``multipart/x-model-message`` carrying the round-cap error.

    Mirrors v1's ``ERROR_MAX_TOOL_CALL_ROUNDS`` contract: the run
    handle awaiter unwraps the last model message; we make it the
    error so callers see the failure as the run's final result.
    """
    # Lazy import to break the agent <-> handlers circular import.
    from sagent.agent.agent import (  # noqa: PLC0415 -- circular
        ERROR_MAX_TOOL_CALL_ROUNDS,
    )

    cap = agent.max_tool_call_rounds
    text = (
        f"Tool-call-round limit reached ({cap} rounds). [{ERROR_MAX_TOOL_CALL_ROUNDS}]"
    )
    agent.inbox.put(TextMessage("", "text/x-stream-end"))
    agent.inbox.put(
        MultipartMessage(
            (TextMessage(text, "text/x-error"),),
            "multipart/x-model-message",
        ),
    )


class ModelResponseHandler(InlineHandler):
    """Enforce stop-reason invariants and route tool batches.

    Stop-reason invariants (parity with v1 ``_send_impl``):
      - ``model_refusal`` -> raise (the model declined for policy).
      - Anything outside ``BENIGN_STOP_REASONS`` AND
        ``_RECOVERABLE_TRUNCATION`` -> raise ``ModelTerminationError``.

    History append is owned by ``HistoryHandler`` (registered first
    on the same descriptor); this handler only routes follow-ups.
    """

    descriptors: tuple[str, ...] = ("multipart/x-model-message",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Guard stop reasons; route tool-batch envelope when calls present."""
        stop_reason = agent.activity.last_stop_reason
        agent.activity.last_stop_reason = None
        if stop_reason == "model_refusal":
            raise RuntimeError(
                "Model refused to respond (content filter or usage policy)."
            )
        if (
            stop_reason
            and stop_reason not in BENIGN_STOP_REASONS
            and stop_reason not in _RECOVERABLE_TRUNCATION
        ):
            raise ModelTerminationError(
                ModelResponse(
                    content=msg,
                    tokens=TokenCount(),
                    stop_reason=stop_reason,
                ),
            )
        if stop_reason not in _RECOVERABLE_TRUNCATION:
            agent.activity.truncation_recoveries = 0
        tool_calls = response_tool_calls(msg)
        if tool_calls:
            agent.inbox.put(
                MultipartMessage(
                    tuple(tool_calls),
                    "multipart/x-tool-batch",
                    parent_id=msg.id,
                )
            )
