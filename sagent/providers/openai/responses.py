"""Shared OpenAI Responses request construction and stream consumption."""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Mapping
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import asyncio
import base64
import inspect
import json
import logging

from sagent.catalog.openai import reasoning_effort
from sagent.lib import debug_log
from sagent.lib.custom_json import DictCodec, IntCodec, json_unfreeze
from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    error_status_code,
    find_response_not_read,
    is_context_overflow_text,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.usage import openai_usage
from sagent.providers.openai import token_count
from sagent.types.capability import ModelCapability, ModelSettings, ServiceTier
from sagent.types.cost import TokenCount
from sagent.types.exceptions import UserFacingError
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
    UsageSnapshot,
    base_model_id,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    BytesMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


if TYPE_CHECKING:
    from openai.types import responses
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsStreaming,
    )
    from openai.types.responses.response_input_param import FunctionCallOutput
    from openai.types.shared.reasoning_effort import ReasoningEffort

    import openai

    from sagent.lib.image import resize
else:
    from wrapt import lazy_import

    openai = lazy_import("openai")
    responses = lazy_import("openai.types.responses")
    resize = lazy_import("sagent.lib.image", "resize")
logger = logging.getLogger(__name__)
_STREAM_IDLE_TIMEOUT = 600.0  # config-globals: ignore -- stream idle timeout


class ResponsesProvider(Protocol):
    """SDK ownership shared by API-key and OAuth providers."""

    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Return the provider-owned SDK for the current event loop.

        Returns:
          sdk: Authenticated client shared by this provider's models.

        """
        ...


class _OpenAIResponsesModel(ModelDefaults):
    """Responses transport with local history and provider-owned clients."""

    def __init__(
        self,
        *,
        provider: ResponsesProvider,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._last_usage: UsageSnapshot | None = None

    @property
    @override
    def capability(self) -> ModelCapability:
        """Return the model's capabilities on this transport.

        Returns:
          capability: Supported model settings, limits, and prices.

        """
        return self._capability

    @property
    @override
    def settings(self) -> ModelSettings:
        """Return the settings selected for this model instance.

        Returns:
          settings: Mutable selections validated against the model's capabilities.

        """
        return self._settings

    @override
    def approx_text_tokens(self, text: str) -> int:
        """Estimate text tokens using the model's local tokenizer.

        Args:
          text: Text to count, treating special-token spellings as ordinary text.

        Returns:
          tokens: Tokenizer count, or chars/4 when the tokenizer is unknown.

        """
        return token_count.approx_text_tokens(text, model_id=self._wire_model_id)

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        """Estimate image tokens with the model's patch or tile formula.

        Args:
          data: Encoded image bytes.

        Returns:
          tokens: Model-specific estimate, or zero when dimensions are unavailable.

        """
        return token_count.approx_image_tokens(
            data, model_id=self._wire_model_id, max_edge=self.limits.max_image_edge_px
        )

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify token-context overflow separately from request-byte limits.

        Args:
          error: Provider exception to classify.

        Returns:
          overflow: Whether shortening the token context can address the error.

        """
        if is_request_too_large(error_status_code(error), str(error)):
            return False
        return is_context_overflow_text(str(error))

    @property
    def _wire_model_id(self) -> str:
        return base_model_id(self.capability.model_id)

    def _effective_service_tier(self) -> ServiceTier | None:
        tier = self.settings.service_tier
        return tier if tier != "auto" else None

    # Model swaps must not close the provider's SDK shared by sibling models.
    @override
    def usage_snapshot(self) -> UsageSnapshot | None:
        """Return quota telemetry from the latest response headers.

        Returns:
          usage: Normalized quota windows, or None when unavailable.

        """
        return self._last_usage

    @override
    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Recognize the provider's retry hint in statusless errors.

        Args:
          error: Provider exception to inspect.

        Returns:
          retryable: Whether the error text explicitly invites retrying.

        """
        return "you can retry" in str(error).lower()

    def _reasoning_effort(self) -> ReasoningEffort | None:
        if self.settings.thinking_effort == "none":
            return None
        return reasoning_effort(
            self.settings.thinking_effort, model_id=self.capability.model_id
        )

    def _build_kwargs(self, request: ModelRequest) -> ResponseCreateParamsStreaming:
        body: ResponseCreateParamsStreaming = {
            "model": self._wire_model_id,
            "input": _build_input(
                request,
                max_image_dim=self.limits.max_image_edge_px,
                max_image_bytes=self.limits.max_image_bytes,
            ),
            "instructions": request.system or "",
            "stream": True,
            "store": False,
        }
        if request.tools:
            body["tools"] = _build_tools(request.tools)
        effort = self._reasoning_effort()
        if effort is not None:
            body["reasoning"] = {"effort": effort, "summary": "auto"}
            body["include"] = ["reasoning.encrypted_content"]
            if self._wire_model_id.startswith(("gpt-5.6", "gpt-6")):
                body["reasoning"]["context"] = "all_turns"
        if request.max_response_tokens is not None:
            body["max_output_tokens"] = request.max_response_tokens
        if self.capability.thinking_effort == frozenset({"none"}):
            body["temperature"] = request.temperature
        tier = self._effective_service_tier()
        if tier is not None:
            body["service_tier"] = tier
        return body

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Stream a Responses request and assemble its message and usage.

        Args:
          request: Local history, tools, and generation settings to send.
          publish: Callback for text and reasoning events; None disables publication.

        Returns:
          response: Completed message, stop reason, token counts, and cost.

        Raises:
          PromptTooLongError: The server reports token-context overflow.
          RequestTooLargeError: The request exceeds the server's byte limit.
          StreamInterruptedError: The stream ends without a terminal response.
          StreamingResponseNotReadError: The SDK hides a streaming HTTP error body.
          TimeoutError: The stream exceeds its idle timeout.

        """
        sdk = await self._provider.get_sdk()
        debug_log.trace(
            "api_call",
            kind="openai_responses",
            model=self._wire_model_id,
            service_tier=self._effective_service_tier(),
        )
        try:
            raw = await sdk.responses.with_raw_response.create(
                **self._build_kwargs(request)
            )
            self._last_usage = openai_usage(raw.headers)
            events = raw.parse(to=openai.AsyncStream[responses.ResponseStreamEvent])
            return await _consume_stream(events, model=self, publish=publish)
        except Exception as exc:
            if find_response_not_read(exc) is not None:
                raise StreamingResponseNotReadError(provider_name="OpenAI") from exc
            raise_if_request_too_large(error_status_code(exc), str(exc), cause=exc)
            if self.is_context_overflow(exc):
                raise PromptTooLongError(str(exc)) from exc
            raise


_FINISH_MAP: Final[dict[str, str]] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
}

_STREAM_ERROR_STATUS: Final[dict[str, int]] = {
    "rate_limit": 429,
    "rate_limit_error": 429,
    "rate_limit_exceeded": 429,
    "api_error": 500,
    "overloaded_error": 529,
    "server_error": 500,
}


def _build_tools(
    tools: list[Tool],
) -> list[responses.FunctionToolParam]:
    """Translate each ``Tool`` into a Responses API function-tool param."""
    return [_build_tool(t) for t in tools]


def _build_tool(tool: Tool) -> responses.FunctionToolParam:
    """Wire-shape a single ``Tool`` for the Responses API."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": cast(
            dict[str, object] | None, json_unfreeze(tool.directive_schema)
        ),
        "strict": None,
    }


def _build_input(
    request: ModelRequest,
    *,
    max_image_dim: int = 0,
    max_image_bytes: int = 0,
) -> responses.ResponseInputParam:
    """Convert history entries to Responses API input items.

    Args:
      request: Fully-built model request.
      max_image_dim: Maximum image dimension (pixels); larger inputs are
          resized before encoding. ``0`` (the default) means no cap; the
          live caller passes the model profile's ``max_image_dim``.
      max_image_bytes: Maximum image size in bytes after resize. ``0`` (the
          default) means no cap.

    Returns:
      items: Responses API input items in send order.

    """
    ids = IdRemapper("fc_")
    items: responses.ResponseInputParam = []
    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            items.append(
                _build_user_item(entry, max_image_dim, max_image_bytes),
            )
        elif isinstance(entry, AssistantMessage):
            _build_assistant_items(entry, items, ids)
        else:
            items.append(
                _build_tool_result_item(
                    entry,
                    ids,
                    max_image_dim=max_image_dim,
                    max_image_bytes=max_image_bytes,
                )
            )
    return items


def _build_user_item(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
) -> responses.EasyInputMessageParam:
    """Wire-shape a user-side entry, inlining any image attachments."""
    image_atts = [att for att in entry.attachments if _is_image_attachment(att)]
    non_image_atts = [att for att in entry.attachments if not _is_image_attachment(att)]
    for att in non_image_atts:
        logger.warning(
            "OpenAI: skipping non-image attachment (mime=%s)",
            att.descriptor,
        )
    if not image_atts:
        return {"role": "user", "content": entry.text}
    blocks: list[responses.ResponseInputContentParam] = []
    if entry.text:
        blocks.append({"type": "input_text", "text": entry.text})
    for att in image_atts:
        raw, mime = resize(att.data, max_dim=max_image_dim, max_bytes=max_image_bytes)
        b64 = base64.b64encode(raw).decode()
        blocks.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime};base64,{b64}",
            }
        )
    return {"role": "user", "content": blocks}


def _is_image_attachment(att: BytesMessage) -> bool:
    """True for ``BytesMessage`` attachments whose descriptor is an image mime."""
    return att.descriptor.startswith("image/")


def _build_assistant_items(
    entry: AssistantMessage,
    items: responses.ResponseInputParam,
    ids: IdRemapper,
) -> None:
    """Expand an AssistantMessage into assistant + function_call items."""
    for block in entry.thinking_blocks:
        if block.get("type") == "reasoning" and isinstance(
            block.get("encrypted_content"), str
        ):
            # OpenAI documents replaying the reasoning output item verbatim in
            # stateless mode. Copy the opaque mapping so persisted session data
            # never mutates while the SDK serializes the request.
            items.append(cast(responses.ResponseReasoningItemParam, dict(block)))
    if entry.text:
        items.append({"role": "assistant", "content": entry.text})
    for tc in entry.tool_calls:
        native_id = ids.map(tc.id)
        items.append(
            {
                "type": "function_call",
                "id": native_id,
                "call_id": native_id,
                "name": tc.name,
                "arguments": json.dumps(dict(tc.args)),
                "status": "completed",
            }
        )


def _build_tool_result_item(
    entry: ToolResult,
    ids: IdRemapper,
    *,
    max_image_dim: int = 0,
    max_image_bytes: int = 0,
) -> FunctionCallOutput:
    """Encode tool output with its image attachments and matching call id."""
    text = (
        f"[Error] {entry.content}"
        if entry.is_error and entry.content
        else entry.content
    )
    output: str | responses.ResponseFunctionCallOutputItemListParam = text
    if entry.attachments:
        parts: responses.ResponseFunctionCallOutputItemListParam = []
        if text:
            parts.append({"type": "input_text", "text": text})
        for attachment in entry.attachments:
            if _is_image_attachment(attachment):
                raw, mime = resize(
                    attachment.data, max_dim=max_image_dim, max_bytes=max_image_bytes
                )
                parts.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:{mime};base64,{base64.b64encode(raw).decode()}",
                    }
                )
            else:
                logger.warning(
                    "OpenAI: skipping non-image attachment (mime=%s)",
                    attachment.descriptor,
                )
        output = parts or text
    return {
        "type": "function_call_output",
        "call_id": ids.map(entry.call_id),
        "output": output,
        "status": "completed",
    }


class _UsageDetails(Protocol):
    """Minimum SDK usage-details surface needed for forward-compatible fields."""

    def to_dict(self) -> dict[str, object]: ...


def _terminal_metadata(response: object) -> tuple[str, str, int, int, int, int]:
    """Extract terminal response metadata, including SDK-extra usage fields."""
    message_id = str(getattr(response, "id", "") or "")
    status = str(getattr(response, "status", "") or "")
    usage = getattr(response, "usage", None)
    if usage is None:
        return message_id, status, 0, 0, 0, 0
    input_tokens = IntCodec.coerce(getattr(usage, "input_tokens", None), 0)
    output_tokens = IntCodec.coerce(getattr(usage, "output_tokens", None), 0)
    cache_read = 0
    cache_write = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        # OpenAI SDK 2.44 predates the typed ``cache_write_tokens`` field, but
        # Stainless models preserve unknown response fields and expose them
        # through ``to_dict``. Reading the mapping keeps new usage data without
        # an Any cast or waiting for an SDK schema release.
        raw_details = cast(_UsageDetails, details).to_dict()
        cache_read = IntCodec.coerce(raw_details.get("cached_tokens"), 0)
        cache_write = IntCodec.coerce(raw_details.get("cache_write_tokens"), 0)
    return (
        message_id,
        status,
        input_tokens,
        output_tokens,
        cache_read,
        cache_write,
    )


class _OpenAIStreamError(UserFacingError):
    """In-band Responses error retaining retry/rate-limit classification data."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        normalized_code = code or "unknown_error"
        normalized_type = (
            "rate_limit_error"
            if normalized_code in {"rate_limit", "rate_limit_exceeded"}
            else normalized_code
        )
        self.status_code = _STREAM_ERROR_STATUS.get(normalized_code)
        self.body: Mapping[str, object] = {
            "type": "error",
            "error": {
                "type": normalized_type,
                "code": normalized_code,
                "message": message,
                "param": param,
            },
        }


async def _consume_stream(
    stream: AsyncIterable[object],
    *,
    model: _OpenAIResponsesModel,
    publish: Callable[[RuntimeEvent], None] | None,
) -> ModelResponse:
    """Parse a Responses API event stream into an assembled ModelResponse."""
    text_parts: list[str] = []
    refusal_parts: list[str] = []
    thinking_parts: list[str] = []
    encrypted_reasoning: list[Mapping[str, object]] = []
    tool_calls: list[ToolCall] = []
    tool_args: dict[str, list[str]] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    message_id = ""
    finish_reason: str | None = None
    completed = False

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STREAM_IDLE_TIMEOUT
    # Single cleanup rule: every path that does not return a terminal response
    # closes the stream. A mid-stream error event (raises below), an idle
    # timeout / cancellation, and the no-completion fallthrough all leave the
    # SSE connection open otherwise -- a per-error connection leak. ``completed``
    # gates the one exit that legitimately keeps reading until done.
    try:
        async with asyncio.timeout_at(deadline) as watchdog:
            async for event in stream:
                watchdog.reschedule(loop.time() + _STREAM_IDLE_TIMEOUT)
                if isinstance(event, responses.ResponseRefusalDeltaEvent):
                    refusal_parts.append(event.delta)
                    text_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponsePartial(event.delta))
                elif isinstance(event, responses.ResponseRefusalDoneEvent):
                    # Normally deltas already contain the full refusal. Keep a
                    # done-only stream useful and append only an unseen suffix.
                    seen = "".join(refusal_parts)
                    suffix = (
                        event.refusal[len(seen) :]
                        if event.refusal.startswith(seen)
                        else ""
                    )
                    if not seen:
                        suffix = event.refusal
                    if suffix:
                        refusal_parts.append(suffix)
                        text_parts.append(suffix)
                        if publish is not None:
                            publish(ModelResponsePartial(suffix))
                elif isinstance(event, responses.ResponseTextDeltaEvent):
                    text_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponsePartial(event.delta))
                elif isinstance(
                    event,
                    (
                        responses.ResponseReasoningTextDeltaEvent,
                        responses.ResponseReasoningSummaryTextDeltaEvent,
                    ),
                ):
                    thinking_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponseThinking(event.delta))
                elif isinstance(
                    event,
                    responses.ResponseFunctionCallArgumentsDeltaEvent,
                ):
                    tool_args.setdefault(event.item_id, []).append(event.delta)
                elif isinstance(event, responses.ResponseOutputItemDoneEvent):
                    item = event.item
                    if item.type == "function_call":
                        tc_id = str(item.call_id or "")
                        tc_name = str(item.name or "")
                        item_id = item.id or ""
                        delta_args = "".join(tool_args.get(item_id, []))
                        done_args = str(item.arguments or "")
                        args = _parse_tool_arguments(
                            delta_args,
                            done_args,
                            tool_name=tc_name,
                            call_id=tc_id,
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tc_id,
                                name=tc_name,
                                args=args,
                            )
                        )
                    elif item.type == "reasoning":
                        reasoning_item = item.to_dict(exclude_none=True)
                        if isinstance(reasoning_item.get("encrypted_content"), str):
                            encrypted_reasoning.append(reasoning_item)
                elif isinstance(event, responses.ResponseErrorEvent):
                    raise _openai_stream_event_error(event)
                elif isinstance(event, responses.ResponseFailedEvent):
                    raise _openai_stream_response_error(event.response)
                elif isinstance(
                    event,
                    (
                        responses.ResponseCompletedEvent,
                        responses.ResponseIncompleteEvent,
                    ),
                ):
                    resp = event.response
                    (
                        message_id,
                        finish_reason,
                        input_tokens,
                        output_tokens,
                        cache_read,
                        cache_write,
                    ) = _terminal_metadata(resp)
                    if finish_reason == "incomplete":
                        incomplete = getattr(resp, "incomplete_details", None)
                        if getattr(incomplete, "reason", None) == "content_filter":
                            finish_reason = "content_filter"
                    debug_log.trace(
                        "api_response",
                        kind="openai_responses",
                        service_tier=getattr(resp, "service_tier", None),
                    )
                    # Set last: if a usage/attr read above raises on an SDK shape
                    # change, ``completed`` stays False so the ``finally`` still
                    # closes the stream instead of leaking the connection.
                    completed = True
    finally:
        if not completed:
            await _close_stream(stream)

    if refusal_parts and finish_reason in (None, "completed", "stop"):
        finish_reason = "content_filter"
    response = _build_stream_response(
        text_parts=text_parts,
        thinking_parts=thinking_parts,
        encrypted_reasoning=encrypted_reasoning,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write=cache_write,
        cache_read=cache_read,
        finish_reason=finish_reason,
        message_id=message_id,
        model=model,
    )
    if not completed:
        raise StreamInterruptedError(response)
    return response


def _openai_stream_event_error(event: object) -> UserFacingError:
    code = getattr(event, "code", None)
    param = getattr(event, "param", None)
    message = getattr(event, "message", "OpenAI stream error")
    details = ["OpenAI stream error"]
    if isinstance(code, str) and code:
        details.append(f"code={code}")
    if isinstance(param, str) and param:
        details.append(f"param={param}")
    details.append(str(message))
    return _OpenAIStreamError(
        ": ".join(details),
        code=code if isinstance(code, str) else None,
        param=param if isinstance(param, str) else None,
    )


def _openai_stream_response_error(response: object) -> UserFacingError:
    response_id = getattr(response, "id", "")
    status = getattr(response, "status", "")
    error = getattr(response, "error", None)
    message = getattr(error, "message", None) if error is not None else None
    code = getattr(error, "code", None) if error is not None else None
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete is not None else None
    details = ["OpenAI stream ended without completion"]
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    if isinstance(response_id, str) and response_id:
        details.append(f"response_id={response_id}")
    if isinstance(code, str) and code:
        details.append(f"code={code}")
    if isinstance(reason, str) and reason:
        details.append(f"reason={reason}")
    if isinstance(message, str) and message:
        details.append(message)
    return _OpenAIStreamError(
        ": ".join(details),
        code=code if isinstance(code, str) else None,
    )


def _build_stream_response(
    *,
    text_parts: list[str],
    thinking_parts: list[str],
    encrypted_reasoning: list[Mapping[str, object]],
    tool_calls: list[ToolCall],
    input_tokens: int,
    output_tokens: int,
    cache_write: int,
    cache_read: int,
    finish_reason: str | None,
    message_id: str,
    model: _OpenAIResponsesModel,
) -> ModelResponse:
    raw_reason = _FINISH_MAP.get(finish_reason or "", finish_reason)

    # The Responses API reports a cache-inclusive prompt total; keep ordinary,
    # cache-write, and cache-read pools disjoint in Sagent's normalized usage.
    non_cached_input = max(0, input_tokens - cache_read - cache_write)
    tokens = TokenCount(
        request=non_cached_input,
        response=output_tokens,
        cache_write=cache_write,
        cache_read=cache_read,
    )
    thinking_blocks: list[Mapping[str, object]] = list(encrypted_reasoning)
    if thinking_parts:
        thinking_blocks.append({"type": "reasoning", "text": "".join(thinking_parts)})
    return ModelResponse(
        message=AssistantMessage(
            text="".join(text_parts),
            thinking_blocks=tuple(thinking_blocks),
            tool_calls=tuple(tool_calls),
        ),
        tokens=tokens,
        stop_reason=normalize_stop_reason(
            raw_reason,
            kind="openai",
            has_tool_use=bool(tool_calls),
        ),
        message_id=message_id,
        request_id=message_id,
        spend=model.spend(tokens),
    )


async def _close_stream(stream: object) -> None:
    """Close a Responses stream after timeout, cancellation, or stream error.

    Runs in a ``finally`` / cleanup context, so a failure HERE must never
    replace the error that triggered the close (e.g. a ``UserFacingError`` from a
    ``ResponseFailedEvent``). A broken pipe on ``aclose`` would otherwise surface
    to the user instead of the real cause -- swallow and log it.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.debug("stream close raised during cleanup", exc_info=True)


def _parse_tool_arguments(
    delta_args: str,
    done_args: str,
    *,
    tool_name: str,
    call_id: str,
) -> dict[str, object]:
    """Parse streamed function-call arguments with completed-item fallback."""
    saw_args = False
    parsed_by_source: dict[str, dict[str, object]] = {}
    for source, args_str in (("delta", delta_args), ("done", done_args)):
        if not args_str:
            continue
        saw_args = True
        try:
            parsed = DictCodec.coerce(json.loads(args_str), default=None)
        except json.JSONDecodeError:
            logger.warning(
                "OpenAI Responses tool arguments were invalid JSON: "
                "source=%s tool=%s call_id=%s chars=%d",
                source,
                tool_name,
                call_id,
                len(args_str),
            )
            continue
        except TypeError:
            logger.warning(
                "OpenAI Responses tool arguments were not a JSON object: "
                "source=%s tool=%s call_id=%s",
                source,
                tool_name,
                call_id,
            )
            continue
        parsed_by_source[source] = parsed
        if not parsed:
            logger.warning(
                "OpenAI Responses tool arguments were an empty JSON object: "
                "source=%s tool=%s call_id=%s chars=%d",
                source,
                tool_name,
                call_id,
                len(args_str),
            )
    if not saw_args:
        logger.warning(
            "OpenAI Responses tool arguments were empty: tool=%s call_id=%s",
            tool_name,
            call_id,
        )
    delta = parsed_by_source.get("delta")
    done = parsed_by_source.get("done")
    if delta is not None and done is not None and delta != done:
        logger.warning(
            "OpenAI Responses tool arguments differed between delta and done: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d "
            "delta_keys=%d done_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta),
            len(done),
        )
    if done:
        logger.debug(
            "OpenAI Responses selected completed tool arguments: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d "
            "delta_keys=%d done_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta) if delta is not None else -1,
            len(done),
        )
        return done
    if delta:
        logger.debug(
            "OpenAI Responses selected streamed delta tool arguments: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d delta_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta),
        )
        return delta
    if done is not None:
        return done
    if delta is not None:
        return delta
    return {}
