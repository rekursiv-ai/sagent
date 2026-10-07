"""Google provider (Gemini models).

Usage::

    from sagent.providers import Google

    provider = Google.from_key("AIza...")
    # or: export GOOGLE_API_KEY=AIza... and use Google.from_env()
    flash = provider.model("gemini-3-flash-preview")
    response = await flash.buffer(request)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import asyncio
import base64
import json
import logging
import math
import os
import uuid


if TYPE_CHECKING:
    import httpx2

    from sagent.lib import image
    from sagent.types.capability import (
        ModelCapability,
        ModelLimits,
        ModelSettings,
    )
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold.
    image = lazy_import("sagent.lib.image")

from sagent.catalog.google import api, models, thinking_config
from sagent.catalog.table import ModelCatalog
from sagent.lib.codec import MutablePlainTree, from_plain, mutable
from sagent.providers.lib.errors import (
    error_status_code,
    is_context_overflow_text,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.lib.perloop import PerLoop
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.types.cost import TokenCount
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    UserMessage,
)


logger = logging.getLogger(__name__)

_STREAM_IDLE_TIMEOUT = 600.0  # house-ignore[globals] -- Stream idle timeout dial.
_API_BASE: Final = "https://generativelanguage.googleapis.com/v1beta"


# Image / wire byte limits shared by every Gemini model (verified Jun 2026):
#   - No per-image pixel cap: "There isn't a specific limit to the number of
#     pixels in an image" -- larger images are tiled into 768x768 tiles
#     server-side (https://ai.google.dev/gemini-api/docs/image-understanding;
#     Firebase AI Logic input-file-requirements). So ``max_image_dim=0`` (no
#     client resize) and ``max_image_bytes=0`` (no per-image cap).
#   - The only documented limit is the TOTAL inline request size (text +
#     system + inline bytes), carried as the catalog's ``max_request_bytes``;
#     the byte-aware compaction gate enforces it across the whole request.


class Google:
    """Google provider - creates Gemini model backends."""

    catalog = ModelCatalog(models=models(), transport=api())

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_key(cls, api_key: str) -> Google:
        """Create provider from an API key.

        Args:
          api_key: Google AI Studio API key.

        Returns:
          provider: Configured Google provider instance.

        """
        return cls(api_key=api_key)

    @classmethod
    def from_env(cls) -> Google:
        """Create provider from ``GOOGLE_API_KEY`` env var.

        Returns:
          provider: Configured Google provider instance.

        Raises:
          RuntimeError: If ``GOOGLE_API_KEY`` is not set.

        """
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("Google API key not configured.")
        return cls(api_key=key)

    def model(
        self,
        model_id: str | None = None,
    ) -> _GeminiModel:
        """Create a model backend.

        Args:
          model_id: Model ID; ``None`` uses catalog key ``default``.

        Returns:
          model: Gemini model backend.

        Raises:
          ValueError: If ``model_id`` is not in the catalog.

        """
        mid = model_id if model_id is not None else "default"
        capability, settings = self.catalog.resolve(mid)
        return _GeminiModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    async def close_sdk(self) -> None:
        """Do nothing: each model owns and closes its own client."""


class GeminiProvider(Protocol):
    """What ``_GeminiModel`` reads off the provider that built it."""

    @property
    def api_key(self) -> str:
        """Key sent as ``x-goog-api-key``; empty when the transport uses OAuth."""
        ...


class _GeminiModel(ModelDefaults):
    """Gemini model backend."""

    def __init__(
        self,
        provider: GeminiProvider,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        # Per loop: an httpx2.AsyncClient holds a connection pool owned by
        # the loop that opened it, and the guarding lock binds to the loop
        # that first contends on it. Sharing either across loops raises
        # "bound to a different event loop" or hangs a waiter.
        self._clients: PerLoop[httpx2.AsyncClient | None] = PerLoop(lambda: None)
        self._client_lock: PerLoop[asyncio.Lock] = PerLoop(asyncio.Lock)

    @property
    def _client(self) -> httpx2.AsyncClient | None:
        """The running loop's HTTP client, if one has been opened."""
        return self._clients.peek()

    @_client.setter
    def _client(self, value: httpx2.AsyncClient) -> None:
        """Install a client for this loop, replacing any existing one."""
        self._clients.set(value)

    async def _get_client(self) -> httpx2.AsyncClient:
        """Return a reused ``AsyncClient``. Lazy-created."""
        client = self._clients.get()
        if client is not None:
            return client
        async with self._client_lock.get():
            client = self._clients.get()
            if client is None:
                client = httpx2.AsyncClient()
                self._clients.set(client)
            return client

    @override
    async def close(self) -> None:
        """Close this loop's HTTP client.

        This loop's only: a pool belongs to the loop that opened it, so
        closing another loop's client from here breaks a pool that loop
        is still using rather than releasing it.
        """
        client = self._clients.peek()
        self._clients.clear()
        if client is not None:
            await client.aclose()

    @property
    @override
    def capability(self) -> ModelCapability:
        """What this model offers on this transport."""
        return self._capability

    @property
    @override
    def settings(self) -> ModelSettings:
        """What this instance chose."""
        return self._settings

    @override
    def approx_text_tokens(self, text: str) -> int:
        """Estimate locally from the catalog's measured character ratio."""
        return int(len(text) / self.capability.approx_chars_per_token)

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate via Gemini's tile formula (``tiles * 258``)."""
        return gemini_image_tokens(data)

    @override
    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Call ``:countTokens`` for the exact server-side count."""
        model_id = self.capability.wire_model_id or self.capability.model_id
        url = f"{_API_BASE}/models/{model_id}:countTokens"
        body = _build_request(request, self.capability, self.settings, self.limits)
        client = await self._get_client()
        r = await client.post(
            url,
            json={"generateContentRequest": {"model": f"models/{model_id}", **body}},
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._provider.api_key,
            },
            timeout=60.0,
        )
        raise_for_gemini_status(r.status_code, r.text)
        r.raise_for_status()
        return from_plain(
            cast(dict[str, MutablePlainTree], r.json()).get("totalTokens"),
            int,
            default=0,
        )

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify an error as a token context-window overflow.

        Excludes the byte wire-limit (HTTP 413): that is a different
        condition handled by ``RequestTooLargeError`` -- a larger-window
        model does not relieve the byte ceiling.

        Args:
          error: Exception raised by the provider call.

        Returns:
          overflow: True when ``error`` indicates token-context overflow.

        """
        if is_request_too_large(error_status_code(error), str(error)):
            return False
        return is_context_overflow_text(str(error))

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Stream via ``:streamGenerateContent`` (``alt=sse``).

        Args:
          request: Fully-built model request.
          publish: Called per streamed event; ``None`` disables streaming.

        Returns:
          response: Parsed ``ModelResponse`` with usage and cost filled in.

        Raises:
          PromptTooLongError: Server reports context overflow.
          RequestTooLargeError: Server reports the request-byte ceiling.
          httpx2.HTTPStatusError: Any other error status.

        """
        model_id = self.capability.wire_model_id or self.capability.model_id
        url = f"{_API_BASE}/models/{model_id}:streamGenerateContent?alt=sse"
        body = _build_request(request, self.capability, self.settings, self.limits)
        client = await self._get_client()
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._provider.api_key,
            },
            timeout=httpx2.Timeout(_STREAM_IDLE_TIMEOUT, connect=30.0),
        ) as r:
            if 400 <= r.status_code < 500:
                raise_for_gemini_status(
                    r.status_code,
                    (await r.aread()).decode(errors="replace"),
                )
            r.raise_for_status()
            return await _consume_gemini_stream(r, publish=publish, model=self)


def gemini_image_tokens(data: bytes) -> int:
    """Estimate Gemini image tokens: 258 per 768x768 tile.

    Args:
      data: Encoded image bytes.

    Returns:
      tokens: Estimated input tokens; 0 when the dimensions are unreadable.

    References:
      https://ai.google.dev/gemini-api/docs/tokens
        Images <=384px count as one tile; larger ones tile at 768x768.

    """
    dims = image.get_dimensions(data)
    if dims is None:
        return 0
    return math.ceil(dims[0] / 768) * math.ceil(dims[1] / 768) * 258


def raise_for_gemini_status(status: int, body: str) -> None:
    """Classify a Gemini 4xx body: byte ceiling first, then token overflow.

    Shared by every Gemini HTTP transport so the classification lives once,
    on the cross-vendor phrase set rather than a bare "too long" match that
    also fires on tool-schema validation errors. Non-overflow statuses fall
    through to the caller's ``raise_for_status``.

    Args:
      status: HTTP status of the failing response.
      body: Decoded response body.

    Raises:
      RequestTooLargeError: The body or status names the request-byte ceiling.
      PromptTooLongError: The body names a token context overflow.

    """
    if status < 400 or status >= 500:
        return
    raise_if_request_too_large(status, body)
    if is_context_overflow_text(body):
        raise PromptTooLongError(body)


def _strip_additional_properties(schema: MutablePlainTree) -> MutablePlainTree:
    """Remove ``additionalProperties`` recursively for Gemini tool schemas."""
    if isinstance(schema, dict):
        return {
            k: _strip_additional_properties(v)
            for k, v in schema.items()
            if k != "additionalProperties"
        }
    if isinstance(schema, list):
        return cast(
            MutablePlainTree,
            [_strip_additional_properties(item) for item in schema],
        )
    return schema


# Gemini groups consecutive ``functionResponse`` parts into one ``role=user`` content.
# Tool results with image attachments emit ``functionResponse`` + ``inlineData``
# siblings in that same user content per Gemini's rules.
def _build_request(
    request: ModelRequest,
    capability: ModelCapability,
    settings: ModelSettings,
    limits: ModelLimits,
) -> dict[str, MutablePlainTree]:
    """Convert history entries to the Gemini API request body."""
    # Build tool_use_id → function name mapping from prior model responses
    # so we can echo the right name when emitting functionResponse parts.
    call_names: dict[str, str] = {
        tc.id: tc.name
        for entry in request.messages
        if isinstance(entry, AssistantMessage)
        for tc in entry.tool_calls
    }

    contents: list[dict[str, MutablePlainTree]] = []
    pending_tool_parts: list[dict[str, MutablePlainTree]] = []
    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            parts: list[dict[str, MutablePlainTree]] = []
            if entry.text:
                parts.append({"text": entry.text})
            for att in entry.attachments:
                block = _attachment_part(
                    att,
                    limits.max_image_edge_px,
                    limits.max_image_bytes,
                )
                if block is not None:
                    parts.append(block)
            if not parts:
                parts.append({"text": ""})
            # Coalesce: when a user message lands mid-cohort (preempt
            # with detached stubs), append its parts into the same
            # role=user content that holds the pending functionResponse
            # parts. A standalone role=user message would violate
            # Gemini's user/model alternation requirement.
            if pending_tool_parts:
                pending_tool_parts.extend(parts)
                _flush_tool_parts(contents, pending_tool_parts)
            else:
                contents.append({"role": "user", "parts": [*parts]})
        elif isinstance(entry, AssistantMessage):
            _flush_tool_parts(contents, pending_tool_parts)
            model_parts: list[dict[str, MutablePlainTree]] = []
            if entry.text:
                text_part: dict[str, MutablePlainTree] = {"text": entry.text}
                # Gemini 3.x requires the model's thought signature echoed back
                # on its parts in subsequent requests, else the API rejects the
                # continuation. Omitted when empty (older models / no thinking).
                if entry.thought_signature:
                    text_part["thoughtSignature"] = entry.thought_signature
                model_parts.append(text_part)
            for tc in entry.tool_calls:
                fc_args: dict[str, MutablePlainTree] = mutable(tc.args)
                fc_part: dict[str, MutablePlainTree] = {
                    "functionCall": {
                        "name": tc.name,
                        "args": fc_args,
                    },
                }
                if tc.thought_signature:
                    fc_part["thoughtSignature"] = tc.thought_signature
                model_parts.append(fc_part)
            if model_parts:
                contents.append({"role": "model", "parts": [*model_parts]})
        else:
            # ToolResult: role=user with functionResponse part(s); image
            # attachments emit as inlineData siblings in the same user
            # content.
            func_name = call_names.get(entry.call_id, entry.call_id)
            text = entry.content
            if entry.is_error and text:
                text = f"[Error] {text}"
            pending_tool_parts.append(
                {
                    "functionResponse": {
                        "name": func_name,
                        "response": {"content": text},
                    },
                },
            )
            for att in entry.attachments:
                block = _attachment_part(
                    att,
                    limits.max_image_edge_px,
                    limits.max_image_bytes,
                )
                if block is not None:
                    pending_tool_parts.append(block)
    _flush_tool_parts(contents, pending_tool_parts)

    thinking = thinking_config(capability, settings)
    gen_config: dict[str, MutablePlainTree] = {}
    if thinking is None:
        gen_config["temperature"] = request.temperature
    if request.max_response_tokens is not None:
        gen_config["maxOutputTokens"] = request.max_response_tokens
    if thinking is not None:
        gen_config["thinkingConfig"] = cast(MutablePlainTree, thinking)
    body: dict[str, MutablePlainTree] = {
        "contents": [*contents],
        "generationConfig": gen_config,
    }
    if request.system:
        body["systemInstruction"] = {"parts": [{"text": request.system}]}
    if request.tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": _strip_additional_properties(
                            cast(
                                MutablePlainTree,
                                mutable(t.directive_schema),
                            ),
                        ),
                    }
                    for t in request.tools
                ],
            },
        ]
    return body


def _attachment_part(
    att: object,
    max_image_dim: int,
    max_image_bytes: int,
) -> dict[str, MutablePlainTree] | None:
    """Translate a ``BytesMessage`` attachment to a Gemini ``inlineData`` part."""
    data = getattr(att, "data", None)
    descriptor = getattr(att, "descriptor", "")
    if not isinstance(data, bytes) or not isinstance(descriptor, str):
        return None
    is_image = descriptor.startswith("image/")
    if not (is_image or descriptor == "application/pdf"):
        logger.warning(
            "Google: skipping attachment with unsupported mime=%s",
            descriptor,
        )
        return None
    raw = data
    mime = descriptor
    if is_image:
        raw, mime = image.resize(raw, max_dim=max_image_dim, max_bytes=max_image_bytes)
    b64 = base64.b64encode(raw).decode()
    return {"inlineData": {"mimeType": mime, "data": b64}}


def _flush_tool_parts(
    contents: list[dict[str, MutablePlainTree]],
    pending: list[dict[str, MutablePlainTree]],
) -> None:
    """Emit buffered tool-response parts as a user message, then clear."""
    if pending:
        contents.append({"role": "user", "parts": list(pending)})
        pending.clear()


# Each ``data:`` line is a full GenerateContentResponse JSON object with partial
# content; we accumulate text and tool calls across events. ``publish`` receives a
# ``RuntimeEvent`` per chunk; ``None`` disables streaming.
async def _consume_gemini_stream(
    r: httpx2.Response,
    *,
    publish: Callable[[RuntimeEvent], None] | None = None,
    model: _GeminiModel,
    chunk_unwrap: Callable[[dict[str, MutablePlainTree]], dict[str, MutablePlainTree]]
    | None = None,
) -> ModelResponse:
    """Parse SSE stream from :streamGenerateContent?alt=sse."""
    text_chunks: list[str] = []
    text_signature: str = ""
    thinking_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: dict[str, MutablePlainTree] = {}
    finish_reason: str | None = None
    malformed_chunks = 0
    parsed_chunks = 0

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STREAM_IDLE_TIMEOUT
    async with asyncio.timeout_at(deadline) as watchdog:
        async for raw_line in r.aiter_lines():
            watchdog.reschedule(loop.time() + _STREAM_IDLE_TIMEOUT)
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if not data_str:
                continue
            try:
                event = cast(dict[str, MutablePlainTree], json.loads(data_str))
            except json.JSONDecodeError as exc:
                malformed_chunks += 1
                logger.warning("Google stream malformed JSON chunk: %s", exc)
                continue
            parsed_chunks += 1
            if chunk_unwrap is not None:
                event = chunk_unwrap(event)
            event_usage = event.get("usageMetadata")
            if isinstance(event_usage, dict):
                usage = event_usage
            candidates = cast(
                list[dict[str, MutablePlainTree]],
                event.get("candidates") or [],
            )
            if not candidates:
                continue
            first = candidates[0]
            fr = first.get("finishReason")
            if isinstance(fr, str) and fr:
                finish_reason = fr
            content = cast(dict[str, MutablePlainTree], first.get("content") or {})
            parts = cast(list[dict[str, MutablePlainTree]], content.get("parts") or [])
            for part in parts:
                if "text" in part:
                    # Gemini 3.x attaches the model's thought signature to its
                    # answer part; capture it so it can be echoed back next turn.
                    if "thoughtSignature" in part:
                        text_signature = str(part["thoughtSignature"])
                    chunk = part.get("text")
                    if isinstance(chunk, str) and part.get("thought") is True:
                        thinking_chunks.append(chunk)
                        if publish is not None:
                            publish(ModelResponseThinking(chunk))
                    elif isinstance(chunk, str):
                        text_chunks.append(chunk)
                        if publish is not None:
                            publish(ModelResponsePartial(chunk))
                elif "functionCall" in part:
                    fc = cast(dict[str, MutablePlainTree], part["functionCall"])
                    fc_name = fc.get("name") or ""
                    fc_args = cast(dict[str, MutablePlainTree], fc.get("args") or {})
                    if isinstance(fc_name, str):
                        tc_id = f"call_{uuid.uuid4().hex[:24]}"
                        tool_calls.append(
                            ToolCall(
                                id=tc_id,
                                name=fc_name,
                                args=cast(Mapping[str, object], fc_args),
                                thought_signature=cast(
                                    str,
                                    part.get("thoughtSignature", ""),
                                ),
                            ),
                        )

    if malformed_chunks and not parsed_chunks:
        raise ValueError("Google stream returned only malformed JSON chunks.")

    response = _build_response(
        text="".join(text_chunks),
        text_signature=text_signature,
        thinking="".join(thinking_chunks),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        model=model,
    )
    if finish_reason is None:
        raise StreamInterruptedError(response)
    return response


def _build_response(
    *,
    text: str,
    text_signature: str = "",
    thinking: str = "",
    tool_calls: list[ToolCall],
    usage: dict[str, MutablePlainTree],
    finish_reason: str | None,
    model: _GeminiModel,
) -> ModelResponse:
    """Build a ``ModelResponse`` from Gemini's parsed stream fields."""
    output_tokens = from_plain(usage.get("candidatesTokenCount"), int, default=0)
    cache_read = from_plain(usage.get("cachedContentTokenCount"), int, default=0)
    # ``promptTokenCount`` is cache-inclusive; store the non-cached remainder so
    # ``TokenCount.input_tokens`` is disjoint from ``cache_read_tokens``.
    input_tokens = max(
        0,
        from_plain(usage.get("promptTokenCount"), int, default=0) - cache_read,
    )
    tokens = TokenCount(
        request=input_tokens,
        response=output_tokens,
        cache_read=cache_read,
    )
    # Gemini doesn't expose a stable message id - synthesize one.
    message_id = f"gemini_{uuid.uuid4().hex[:16]}"
    return ModelResponse(
        message=AssistantMessage(
            text=text,
            thought_signature=text_signature,
            thinking_blocks=({"type": "thinking", "thinking": thinking},)
            if thinking
            else (),
            tool_calls=tuple(tool_calls),
        ),
        tokens=tokens,
        stop_reason=normalize_stop_reason(
            finish_reason,
            kind="google",
            has_tool_use=bool(tool_calls),
        ),
        message_id=message_id,
        request_id=message_id,
        spend=model.spend(tokens),
    )
