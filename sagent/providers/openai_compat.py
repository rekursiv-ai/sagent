"""OpenAI chat-completions compatible base.

Concrete providers (OpenAI, Kimi, Qwen, MiniMax, any local vLLM/SGLang
box) subclass ``OpenAICompat`` and override a handful of class attrs::

    class Kimi(OpenAICompat):
        DEFAULT_MODEL = "kimi-k2.6"
        ENV_VAR = "MOONSHOT_API_KEY"
        BASE_URL = "https://api.moonshot.ai/v1"
        KNOWN_MODELS = {...}

The model backend (``OpenAICompatModel``) exposes hooks for
provider-specific tweaks (``_reasoning_field``, ``_effort_model_ids``,
``_transform_body``). Overriding one method on a subclass is enough
for most provider divergence.

Usage::

    provider = Kimi.from_env()
    model = provider.model()        # DEFAULT_MODEL
    response = await model.buffer(request)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Self, cast

import asyncio
import base64
import json
import logging
import math
import os


if TYPE_CHECKING:
    import httpx

    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    httpx = lazy_import("httpx")  # 100ms cold
    image_lib = lazy_import("sagent.lib.image")

from sagent.custom_exceptions import PromptTooLongError
from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib.descriptors import is_image
from sagent.lib.json import (
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_freeze,
    json_unfreeze,
)
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    get_tool_name,
    tool_call_message,
)
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.stop_reason import normalize_stop_reason


logger = logging.getLogger(__name__)

_STREAM_IDLE_TIMEOUT = 600.0


class OpenAICompat:
    """Base provider for OpenAI chat-completions compatible endpoints.

    Subclasses override the class attrs below. ``model()`` returns an
    ``OpenAICompatModel`` wired to this provider.
    """

    DEFAULT_MODEL: ClassVar[str] = ""
    DEFAULT_UTILITY_MODEL: ClassVar[str] = ""
    ENV_VAR: ClassVar[str] = ""
    BASE_URL: ClassVar[str] = ""
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {}
    # Model class to instantiate. Subclasses can override with a
    # subclass of ``OpenAICompatModel`` that tweaks hooks.
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]]

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        # Allow ``base_url`` override for self-hosted endpoints without
        # subclassing (e.g. ``Qwen.from_env(base_url="http://box:8000/v1")``).
        self.base_url = base_url or self.BASE_URL

    @classmethod
    def from_key(cls, api_key: str, *, base_url: str | None = None) -> Self:
        """Build provider from an API key.

        Args:
          api_key: Provider API key.
          base_url: Override the default endpoint URL.

        Returns:
          provider: Provider instance.

        """
        return cls(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls, *, base_url: str | None = None) -> Self:
        """Build provider from the class's ``ENV_VAR``.

        Args:
          base_url: Override the default endpoint URL.

        Returns:
          provider: Provider instance.

        Raises:
          RuntimeError: If the API key is not configured.

        """
        if not cls.ENV_VAR:
            raise RuntimeError(f"{cls.__name__} has no ENV_VAR set.")
        key = os.environ.get(cls.ENV_VAR, "")
        if not key:
            raise RuntimeError(f"{cls.__name__} API key not configured.")
        return cls(api_key=key, base_url=base_url)

    def model(
        self,
        model_id: str | None = None,
        max_request_tokens: int | None = None,
    ) -> OpenAICompatModel:
        """Create a model backend.

        Args:
          model_id: Model ID. ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override max input tokens. ``None`` uses profile default.

        Returns:
          model: Chat-completions model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        # Fail fast on unknown model IDs -- every supported model must
        # be in KNOWN_MODELS with explicit limits and pricing.
        profile = self.KNOWN_MODELS.get(mid)
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for {type(self).__name__}."
                f" Known models: {known}",
            )
        return self.MODEL_CLASS(
            provider=self,
            model_id=mid,
            profile=profile,
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens
            ),
        )

    def utility_model(self) -> OpenAICompatModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Utility model backend.

        """
        mid = self.DEFAULT_UTILITY_MODEL or self.DEFAULT_MODEL
        return self.model(mid)


class OpenAICompatModel:
    """Chat-completions model backend.

    Hook methods (``_reasoning_field``, ``_is_effort_model``,
    ``_transform_body``) are cheap overrides for provider quirks.
    """

    # Message field carrying reasoning/thinking text on responses.
    # Kimi/Qwen/DeepSeek use ``reasoning_content``; OpenAI surfaces
    # reasoning separately via the Responses API only (leave as None).
    _reasoning_field: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        provider: OpenAICompat,
        model_id: str,
        profile: ModelProfile,
        max_request_tokens: int,
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._profile = profile
        self._max_request_tokens = max_request_tokens
        # Persistent client for connection reuse. Lazily initialized
        # because tests patch the backing ``httpx.AsyncClient``.
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reused ``AsyncClient`` (per-model). Lazy-created."""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient()
            return self._client

    @property
    def max_request_tokens(self) -> int:
        """Maximum input token count."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Model identifier string."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Maximum output token count."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """Whether the model supports streaming responses."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether the model surfaces reasoning/thinking text."""
        return self._reasoning_field is not None

    @property
    def supports_effort(self) -> bool:
        """Whether the model accepts a reasoning effort parameter."""
        return self._is_effort_model(self._model_id)

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt cache control."""
        return False

    @property
    def supports_context_management(self) -> bool:
        """Whether the provider supports context management."""
        return False

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether the provider supports persistent retry."""
        return False

    @property
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account authentication."""
        return False

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate token count for text using 4 chars/token heuristic.

        Args:
          text: Input text to estimate.

        Returns:
          count: Estimated token count.

        """
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        return self._profile.pricing

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate token count for an image using OpenAI's tile formula.

        Args:
          data: Raw image bytes.

        Returns:
          count: Estimated token count.

        """
        # OpenAI: 85 base + 170 per 512x512 tile (high detail).
        # https://platform.openai.com/docs/guides/vision/calculating-costs
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        tiles = math.ceil(dims[0] / 512) * math.ceil(dims[1] / 512)
        return 85 + tiles * 170

    @property
    def max_image_dim(self) -> int:
        """Maximum image dimension in pixels."""
        return 2048

    @property
    def max_image_bytes(self) -> int:
        """Maximum image size in bytes."""
        return 20 * 1024 * 1024

    def _is_effort_model(self, model_id: str) -> bool:
        """Override: does ``model_id`` accept ``reasoning_effort``?"""
        del model_id
        return False

    def _transform_body(
        self,
        body: MutableJSON,
        request: ModelRequest,
    ) -> MutableJSON:
        """Override: apply provider-specific request-body tweaks."""
        del request
        return body

    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error indicates context-window overflow.

        Args:
          error: Exception raised by the API call.

        Returns:
          overflow: ``True`` if the error is a context-length overflow.

        """
        msg = str(error).lower()
        return "context_length_exceeded" in msg or "maximum context length" in msg

    @property
    def _endpoint(self) -> str:
        return f"{self._provider.base_url.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._provider.api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, request: ModelRequest, *, stream: bool) -> MutableJSON:
        """Build the chat-completions request body."""
        body: MutableJSON = cast(
            MutableJSON,
            {
                "model": self._model_id,
                "messages": build_messages(
                    request, self.max_image_dim, self.max_image_bytes
                ),
                "temperature": request.temperature,
            },
        )
        if request.max_response_tokens is not None:
            body["max_tokens"] = request.max_response_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        if request.effort is not None and self.supports_effort:
            body["reasoning_effort"] = request.effort
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": json_unfreeze(t.directive_schema),
                    },
                }
                for t in request.tools
            ]
        return self._transform_body(body, request)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a non-streaming chat-completions request.

        Args:
          request: Model request to send.

        Returns:
          response: Translated model response.

        """
        body = self._build_body(request, stream=False)
        client = await self._get_client()
        r = await client.post(
            self._endpoint,
            json=body,
            headers=self._headers,
            timeout=120.0,
        )
        if r.status_code == 400:
            msg = r.text.lower()
            if "context_length_exceeded" in msg or "too long" in msg:
                raise PromptTooLongError(r.text)
        r.raise_for_status()
        resp = parse_response(
            r.json(),
            pricing=self._profile.pricing,
            reasoning_field=self._reasoning_field,
        )
        logger.debug(
            "API response: tokens=%d/%d, stop=%s",
            resp.tokens.input_tokens,
            resp.tokens.output_tokens,
            resp.stop_reason,
        )
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream a chat-completions SSE response.

        An asyncio watchdog resets on every delivered chunk so a
        server that goes silent mid-stream trips
        ``asyncio.TimeoutError`` after ``_STREAM_IDLE_TIMEOUT``
        rather than hanging for the full HTTP read deadline.

        Args:
          request: Model request to send.
          on_text: Callback invoked with each text chunk as it arrives.

        Returns:
          response: Assembled model response after the stream closes.

        """
        body = self._build_body(request, stream=True)
        client = await self._get_client()
        async with client.stream(
            "POST",
            self._endpoint,
            json=body,
            headers={**self._headers, "Accept": "text/event-stream"},
            # Give httpx a generous read deadline; idle detection
            # is handled by the asyncio watchdog inside
            # ``consume_stream``.
            timeout=httpx.Timeout(_STREAM_IDLE_TIMEOUT, connect=30.0),
        ) as r:
            if r.status_code == 400:
                err_body = (await r.aread()).decode(errors="replace")
                msg = err_body.lower()
                if "context_length_exceeded" in msg or "too long" in msg:
                    raise PromptTooLongError(err_body)
            r.raise_for_status()
            return await consume_stream(
                r,
                on_text=on_text,
                pricing=self._profile.pricing,
                reasoning_field=self._reasoning_field,
            )


def build_messages(
    request: ModelRequest,
    max_image_dim: int = 2048,
    max_image_bytes: int = 20 * 1024 * 1024,
) -> list[MutableJSON]:
    """Convert internal Message list to OpenAI chat-completions format.

    Tool-returned images are accumulated across consecutive
    tool results and emitted in a single synthetic user message after
    the run. This preserves OpenAI's "tool messages must be contiguous"
    rule: a sequence of tool results bound to the same model response
    stays together, with one user-image payload tacked on at the end.

    Args:
      request: Model request containing messages and system prompt.
      max_image_dim: Maximum image dimension in pixels.
      max_image_bytes: Maximum image size in bytes.

    Returns:
      messages: Chat-completions message list.

    """
    ids = IdRemapper("call_")
    messages: list[MutableJSON] = []
    pending_images: list[MutableJSON] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for msg in request.messages:
        if msg.descriptor == "text/x-user-message":
            _flush_images(messages, pending_images)
            messages.append({"role": "user", "content": cast(str, msg.content)})
        elif msg.descriptor == "multipart/x-user-message":
            _flush_images(messages, pending_images)
            messages.append(_build_user_message(msg, max_image_dim, max_image_bytes))
        elif msg.descriptor == "multipart/x-model-message":
            _flush_images(messages, pending_images)
            parts_mm = cast(tuple[Message, ...], msg.content)
            text_content: str | None = None
            tool_calls_wire: list[MutableJSON] = []
            for part in parts_mm:
                if part.descriptor == "text/plain":
                    text_content = cast(str, part.content)
                elif part.descriptor == "multipart/x-tool-call":
                    directive = get_directive(part)
                    tool_calls_wire.append(
                        {
                            "id": ids.map(get_queue_id(part)),
                            "type": "function",
                            "function": {
                                "name": get_tool_name(part),
                                "arguments": json.dumps(json_unfreeze(directive)),
                            },
                        }
                    )
            m: MutableJSON = {"role": "assistant", "content": text_content}
            if tool_calls_wire:
                m["tool_calls"] = cast(MutableJSONValue, tool_calls_wire)
            messages.append(m)
        elif msg.descriptor == "multipart/x-tool-result":
            parts_tr = cast(tuple[Message, ...], msg.content)
            text = "\n".join(
                str(p.content)
                for p in parts_tr
                if p.descriptor in ("text/plain", "text/x-error")
            )
            image_parts_tr: list[Message] = [
                p for p in parts_tr if is_image(p.descriptor)
            ]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ids.map(get_queue_id(msg)),
                    "content": text,
                }
            )
            for part in image_parts_tr:
                raw, mime_type = image_lib.resize(
                    cast(bytes, part.content),
                    max_dim=max_image_dim,
                    max_bytes=max_image_bytes,
                )
                b64 = base64.b64encode(raw).decode()
                pending_images.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    }
                )
    _flush_images(messages, pending_images)
    return messages


def _flush_images(messages: list[MutableJSON], pending: list[MutableJSON]) -> None:
    """Emit buffered image blocks as a synthetic user message, then clear.

    OpenAI's tool-result messages can't carry image blocks; we attach
    them as a follow-up user message containing just images.
    """
    if pending:
        messages.append({"role": "user", "content": list(pending)})
        pending.clear()


def _build_user_message(
    msg: Message,
    max_image_dim: int = 2048,
    max_image_bytes: int = 20 * 1024 * 1024,
) -> MutableJSON:
    """Build a multipart user message, inlining image attachments as data URLs."""
    content_parts = cast(tuple[Message, ...], msg.content)
    blocks: list[MutableJSON] = []
    text_content: str | None = None
    for part in content_parts:
        if part.descriptor == "text/plain":
            text_content = cast(str, part.content)
        elif is_image(part.descriptor):
            raw, mime = image_lib.resize(
                cast(bytes, part.content),
                max_dim=max_image_dim,
                max_bytes=max_image_bytes,
            )
            b64 = base64.b64encode(raw).decode()
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        else:
            logger.warning(
                "OpenAI-compat: skipping non-image attachment (mime=%s)",
                part.descriptor,
            )
    if text_content is not None:
        blocks.append({"type": "text", "text": text_content})
    if not blocks:
        return {"role": "user", "content": text_content or ""}
    return cast(MutableJSON, {"role": "user", "content": blocks})


def _extract_usage(usage: MutableJSON) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, cache_read) from a usage dict."""
    input_tokens = int_val(usage.get("prompt_tokens"), 0)
    output_tokens = int_val(usage.get("completion_tokens"), 0)
    raw_details = usage.get("prompt_tokens_details")
    details: MutableJSON = (
        cast(MutableJSON, raw_details) if isinstance(raw_details, dict) else {}
    )
    cache_read = int_val(details.get("cached_tokens"), 0)
    return input_tokens, output_tokens, cache_read


def parse_response(
    data: MutableJSON,
    *,
    pricing: Pricing,
    reasoning_field: str | None,
) -> ModelResponse:
    """Convert a non-streaming chat-completions body to ModelResponse.

    Args:
      data: Raw JSON response body.
      pricing: Token pricing for cost computation.
      reasoning_field: Message field carrying reasoning text, or ``None``.

    Returns:
      response: Translated model response.

    """
    choices = cast(list[MutableJSON], data["choices"])
    choice = choices[0]
    message = cast(MutableJSON, choice["message"])
    msg_parts: list[Message] = []

    if reasoning_field:
        raw_thinking = message.get(reasoning_field)
        if isinstance(raw_thinking, str) and raw_thinking:
            msg_parts.append(TextMessage(raw_thinking, "text/x-thinking"))

    text_content = cast(str | None, message.get("content"))
    if text_content:
        msg_parts.append(TextMessage(text_content, "text/plain"))

    raw_tcs = cast(list[MutableJSON], message.get("tool_calls") or [])
    for tc in raw_tcs:
        func = cast(MutableJSON, tc["function"])
        tc_name = cast(str, func["name"])
        tc_id = cast(str, tc["id"])
        parsed = _parse_tool_arguments(
            cast(str, func.get("arguments") or ""),
            source="message",
            tool_name=tc_name,
            call_id=tc_id,
        )
        msg_parts.append(tool_call_message(tc_id, tc_name, json_freeze(parsed)))

    usage = cast(MutableJSON, data.get("usage") or {})
    input_tokens, output_tokens, cache_read = _extract_usage(usage)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        max(0, input_tokens - cache_read),
        output_tokens,
        cache_read=cache_read,
    )
    message_id = cast(str, data.get("id") or "")
    has_tool_use = any(p.descriptor == "multipart/x-tool-call" for p in msg_parts)
    all_parts: list[Message] = []
    if message_id:
        all_parts.append(TextMessage(message_id, "text/x-queue-id"))
    all_parts.extend(msg_parts)
    return ModelResponse(
        content=MultipartMessage(
            tuple(all_parts),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            cast(str | None, choice.get("finish_reason")),
            kind="openai",
            has_tool_use=has_tool_use,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )


async def consume_stream(
    r: httpx.Response,
    *,
    on_text: Callable[[str], None] | None,
    pricing: Pricing,
    reasoning_field: str | None,
) -> ModelResponse:
    """Parse an SSE stream into an assembled ModelResponse.

    Tool-call arguments are streamed as sparse per-index deltas; we
    accumulate then json.loads at the end. ``reasoning_field`` (when
    set) captures the provider's reasoning/thinking delta as a string
    - not surfaced live via ``on_text`` (extended thinking is emitted
    post-request, not streamed).

    The outer ``asyncio.timeout_at`` watchdog resets on every line so
    a stalled server trips ``asyncio.TimeoutError`` after
    ``_STREAM_IDLE_TIMEOUT`` of inactivity (matches the Anthropic
    provider's idle-reset pattern).

    Args:
      r: HTTP response with an open SSE stream.
      on_text: Callback invoked with each text chunk as it arrives.
      pricing: Token pricing for cost computation.
      reasoning_field: Message field carrying reasoning text, or ``None``.

    Returns:
      response: Assembled model response after the stream closes.

    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_id: dict[int, str] = {}
    tool_name: dict[int, str] = {}
    tool_args: dict[int, list[str]] = {}
    finish_reason: str | None = None
    message_id = ""
    usage: MutableJSON = {}

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STREAM_IDLE_TIMEOUT
    async with asyncio.timeout_at(deadline) as watchdog:
        async for raw_line in r.aiter_lines():
            watchdog.reschedule(loop.time() + _STREAM_IDLE_TIMEOUT)
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                event = cast(MutableJSON, json.loads(data_str))
            except json.JSONDecodeError:
                continue
            if not message_id:
                message_id = str(event.get("id") or "")
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = cast(MutableJSON, event_usage)
            raw_choices = cast(list[MutableJSON], event.get("choices") or [])
            if not raw_choices:
                continue
            choice = raw_choices[0]
            fr = choice.get("finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason = fr
            delta = cast(MutableJSON, choice.get("delta") or {})
            content_chunk = delta.get("content")
            if isinstance(content_chunk, str) and content_chunk:
                text_parts.append(content_chunk)
                if on_text is not None:
                    on_text(content_chunk)
            if reasoning_field:
                think_chunk = delta.get(reasoning_field)
                if isinstance(think_chunk, str) and think_chunk:
                    thinking_parts.append(think_chunk)
            for tc in cast(list[MutableJSON], delta.get("tool_calls") or []):
                idx_raw = tc.get("index")
                idx = idx_raw if isinstance(idx_raw, int) else 0
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id:
                    tool_id[idx] = tc_id
                func = cast(MutableJSON, tc.get("function") or {})
                name = func.get("name")
                if isinstance(name, str) and name:
                    tool_name[idx] = name
                args_chunk = func.get("arguments")
                if isinstance(args_chunk, str) and args_chunk:
                    tool_args.setdefault(idx, []).append(args_chunk)

    msg_parts: list[Message] = []
    if thinking_parts:
        msg_parts.append(TextMessage("".join(thinking_parts), "text/x-thinking"))
    if text_parts:
        msg_parts.append(TextMessage("".join(text_parts), "text/plain"))
    for idx in sorted(tool_id):
        args_str = "".join(tool_args.get(idx, []))
        tc_name = tool_name.get(idx, "")
        tc_id = tool_id[idx]
        args = _parse_tool_arguments(
            args_str,
            source="delta",
            tool_name=tc_name,
            call_id=tc_id,
        )
        msg_parts.append(tool_call_message(tc_id, tc_name, json_freeze(args)))

    input_tokens, output_tokens, cache_read = _extract_usage(usage)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        max(0, input_tokens - cache_read),
        output_tokens,
        cache_read=cache_read,
    )
    has_tool_use = any(p.descriptor == "multipart/x-tool-call" for p in msg_parts)
    all_parts_s: list[Message] = []
    if message_id:
        all_parts_s.append(TextMessage(message_id, "text/x-queue-id"))
    all_parts_s.extend(msg_parts)
    return ModelResponse(
        content=MultipartMessage(
            tuple(all_parts_s),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            finish_reason,
            kind="openai",
            has_tool_use=has_tool_use,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )


def _parse_tool_arguments(
    args_str: str,
    *,
    source: str,
    tool_name: str,
    call_id: str,
) -> MutableJSON:
    """Parse OpenAI-compatible tool arguments, preserving existing fallback."""
    if not args_str:
        logger.warning(
            "OpenAI-compatible tool arguments were empty: source=%s tool=%s call_id=%s",
            source,
            tool_name,
            call_id,
        )
        return {}
    try:
        parsed = json.loads(args_str)
    except json.JSONDecodeError:
        logger.warning(
            "OpenAI-compatible tool arguments were invalid JSON: "
            "source=%s tool=%s call_id=%s chars=%d",
            source,
            tool_name,
            call_id,
            len(args_str),
        )
        return {}
    if isinstance(parsed, dict):
        return cast(MutableJSON, parsed)
    logger.warning(
        "OpenAI-compatible tool arguments were not a JSON object: "
        "source=%s tool=%s call_id=%s",
        source,
        tool_name,
        call_id,
    )
    return {}


# Wire default model class last so subclasses can also use the default.
OpenAICompat.MODEL_CLASS = OpenAICompatModel
