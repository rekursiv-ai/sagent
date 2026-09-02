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
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, cast, override

import asyncio
import base64
import json
import logging
import math
import os
import uuid


if TYPE_CHECKING:
    import httpx2

    import sagent.lib.image as image_lib
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold
    image_lib = lazy_import("sagent.lib.image")

from sagent.catalog import google as google_catalog
from sagent.lib.custom_json import (
    IntCodec,
    MutableJSON,
    MutableJSONValue,
    json_unfreeze,
)
from sagent.providers.lib.errors import (
    error_status_code,
    is_context_overflow_text,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.lib.perloop import PerLoop
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.types.cost import TokenCount
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.providers import (
    ModelRole,
    resolve,
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

_STREAM_IDLE_TIMEOUT = 600.0  # config-globals: ignore -- stream idle timeout dial
_API_BASE: Final = "https://generativelanguage.googleapis.com/v1beta"


# Image / wire byte limits shared by every Gemini model (verified Jun 2026):
#   - No per-image pixel cap: "There isn't a specific limit to the number of
#     pixels in an image" -- larger images are tiled into 768x768 tiles
#     server-side (https://ai.google.dev/gemini-api/docs/image-understanding;
#     Firebase AI Logic input-file-requirements). So ``max_image_dim=0`` (no
#     client resize) and ``max_image_bytes=0`` (no per-image cap).
#   - The only documented limit is the 20 MB TOTAL inline request size
#     (text + system + inline bytes), so ``max_request_bytes=20 MB``; the
#     byte-aware compaction gate enforces it across the whole request.


class Google:
    """Google provider - creates Gemini model backends."""

    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    DEFAULT_UTILITY_MODEL = "gemini-2.5-flash-lite"

    # Model limits and pricing.
    # ModelLimits: https://ai.google.dev/gemini-api/docs/models
    # Pricing: https://ai.google.dev/gemini-api/docs/pricing
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = google_catalog.models()
    """Per-model capability, shared by every Gemini transport."""

    TRANSPORT: ClassVar[ModelCapability] = google_catalog.api()
    """What this transport lets through; subclasses declare their own."""

    @property
    def ROLES(self) -> Mapping[ModelRole, str]:  # noqa: N802
        """Role name to base id; ``utility`` falls back to the default."""
        return MappingProxyType(
            {
                "default": self.DEFAULT_MODEL,
                "utility": self.DEFAULT_UTILITY_MODEL or self.DEFAULT_MODEL,
            }
        )

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
          model_id: Model ID; ``None`` uses ``DEFAULT_MODEL``.

        Returns:
          model: Gemini model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``CAPABILITIES``.

        """
        mid = model_id if model_id is not None else "default"
        capability, settings = resolve(
            mid, models=self.CAPABILITIES, roles=self.ROLES, transport=self.TRANSPORT
        )
        return _GeminiModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    def utility_model(self) -> _GeminiModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Backend for ``DEFAULT_UTILITY_MODEL``.

        """
        return self.model("utility")


class _GeminiModel(ModelDefaults):
    """Gemini model backend."""

    def __init__(
        self,
        provider: Google,
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
        """Local estimate via ``len(text) // 4`` (Gemini's heuristic)."""
        return len(text) // 4

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate via Gemini's tile formula (``tiles * 258``)."""
        # Tile size undocumented; using OpenAI's 512x512 as proxy.
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        tiles = math.ceil(dims[0] / 512) * math.ceil(dims[1] / 512)
        return tiles * 258

    @override
    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Call ``:countTokens`` for the exact server-side count."""
        url = f"{_API_BASE}/models/{self.capability.model_id}:countTokens"
        body = _build_request(request, self.capability, self.settings, self.limits)
        client = await self._get_client()
        r = await client.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._provider.api_key,
            },
            timeout=60.0,
        )
        if 400 <= r.status_code < 500:
            # Byte limit first, uniform with the streaming path: a 413 must
            # route to byte-overflow recovery, not be mis-read as token
            # overflow because its body happens to say "too large".
            raise_if_request_too_large(r.status_code, r.text)
            msg = r.text.lower()
            if "too large" in msg or "too long" in msg or "exceeds the maximum" in msg:
                raise PromptTooLongError(r.text)
        r.raise_for_status()
        return IntCodec.coerce(cast(MutableJSON, r.json()).get("totalTokens"), 0)

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
          ValueError: Server returns ``400`` for non-overflow reasons.

        """
        url = f"{_API_BASE}/models/{self.capability.model_id}:streamGenerateContent?alt=sse"
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
                err_body = (await r.aread()).decode(errors="replace")
                raise_if_request_too_large(r.status_code, err_body)
                msg = err_body.lower()
                if (
                    "too large" in msg
                    or "too long" in msg
                    or "exceeds the maximum" in msg
                ):
                    raise PromptTooLongError(err_body)
                if r.status_code == 400:
                    raise ValueError(f"Google API 400: {err_body}")
            r.raise_for_status()
            return await _consume_gemini_stream(r, publish=publish, model=self)


def _strip_additional_properties(schema: MutableJSONValue) -> MutableJSONValue:
    """Remove ``additionalProperties`` recursively for Gemini tool schemas."""
    if isinstance(schema, dict):
        schema_map = cast(MutableJSON, schema)
        return cast(
            MutableJSONValue,
            {
                k: _strip_additional_properties(v)
                for k, v in schema_map.items()
                if k != "additionalProperties"
            },
        )
    if isinstance(schema, list):
        return cast(
            MutableJSONValue,
            [_strip_additional_properties(item) for item in schema],
        )
    return schema


def _build_request(
    request: ModelRequest,
    capability: ModelCapability,
    settings: ModelSettings,
    limits: ModelLimits,
) -> MutableJSON:
    """Convert history entries to the Gemini API request body.

    Gemini groups consecutive ``functionResponse`` parts into one
    ``role=user`` content. Tool results with image attachments emit
    ``functionResponse`` + ``inlineData`` siblings in that same user
    content per Gemini's rules.
    """
    # Build tool_use_id → function name mapping from prior model responses
    # so we can echo the right name when emitting functionResponse parts.
    call_names: dict[str, str] = {
        tc.id: tc.name
        for entry in request.messages
        if isinstance(entry, AssistantMessage)
        for tc in entry.tool_calls
    }

    contents: list[MutableJSON] = []
    pending_tool_parts: list[MutableJSON] = []
    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            parts: list[MutableJSON] = []
            if entry.text:
                parts.append({"text": entry.text})
            for att in entry.attachments:
                block = _attachment_part(
                    att, limits.max_image_edge_px, limits.max_image_bytes
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
                contents.append(cast(MutableJSON, {"role": "user", "parts": parts}))
        elif isinstance(entry, AssistantMessage):
            _flush_tool_parts(contents, pending_tool_parts)
            model_parts: list[MutableJSON] = []
            if entry.text:
                text_part = cast(MutableJSON, {"text": entry.text})
                # Gemini 3.x requires the model's thought signature echoed back
                # on its parts in subsequent requests, else the API rejects the
                # continuation. Omitted when empty (older models / no thinking).
                if entry.thought_signature:
                    text_part["thoughtSignature"] = entry.thought_signature
                model_parts.append(text_part)
            for tc in entry.tool_calls:
                fc_part = cast(
                    MutableJSON,
                    {
                        "functionCall": {
                            "name": tc.name,
                            "args": dict(tc.args),
                        },
                    },
                )
                if tc.thought_signature:
                    fc_part["thoughtSignature"] = tc.thought_signature
                model_parts.append(fc_part)
            if model_parts:
                contents.append(
                    cast(MutableJSON, {"role": "model", "parts": model_parts})
                )
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
                }
            )
            for att in entry.attachments:
                block = _attachment_part(
                    att, limits.max_image_edge_px, limits.max_image_bytes
                )
                if block is not None:
                    pending_tool_parts.append(block)
    _flush_tool_parts(contents, pending_tool_parts)

    thinking_config = _thinking_config(capability, settings)
    gen_config: MutableJSON = {}
    if thinking_config is None:
        gen_config["temperature"] = request.temperature
    if request.max_response_tokens is not None:
        gen_config["maxOutputTokens"] = request.max_response_tokens
    if thinking_config is not None:
        gen_config["thinkingConfig"] = cast(MutableJSONValue, thinking_config)
    body: MutableJSON = cast(
        MutableJSON,
        {
            "contents": contents,
            "generationConfig": gen_config,
        },
    )
    if request.system:
        body["systemInstruction"] = cast(
            MutableJSONValue,
            {"parts": [{"text": request.system}]},
        )
    if request.tools:
        body["tools"] = cast(
            MutableJSONValue,
            [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": _strip_additional_properties(
                                cast(
                                    MutableJSONValue, json_unfreeze(t.directive_schema)
                                )
                            ),
                        }
                        for t in request.tools
                    ],
                },
            ],
        )
    return body


def _thinking_config(
    capability: ModelCapability, settings: ModelSettings
) -> MutableJSON | None:
    """Return Gemini thinking config, or ``None`` when thinking is off."""
    # gemini-1.5 rejects ``thinkingConfig`` outright, so an off row must send
    # no key at all rather than a zero budget.
    if settings.thinking_budget == "none" or "none" not in capability.thinking_budget:
        return None
    include = settings.thinking_output == "text"
    if settings.thinking_budget == "auto":
        return {"includeThoughts": include, "thinkingBudget": -1}
    budget = google_catalog.thinking_budget(settings.thinking_effort)
    return {"includeThoughts": include, "thinkingBudget": int(budget)}


def _attachment_part(
    att: object,
    max_image_dim: int,
    max_image_bytes: int,
) -> MutableJSON | None:
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
        raw, mime = image_lib.resize(
            raw, max_dim=max_image_dim, max_bytes=max_image_bytes
        )
    b64 = base64.b64encode(raw).decode()
    return {"inlineData": {"mimeType": mime, "data": b64}}


def _flush_tool_parts(contents: list[MutableJSON], pending: list[MutableJSON]) -> None:
    """Emit buffered tool-response parts as a user message, then clear."""
    if pending:
        contents.append({"role": "user", "parts": list(pending)})
        pending.clear()


async def _consume_gemini_stream(
    r: httpx2.Response,
    *,
    publish: Callable[[RuntimeEvent], None] | None = None,
    model: _GeminiModel,
    chunk_unwrap: Callable[[MutableJSON], MutableJSON] | None = None,
) -> ModelResponse:
    """Parse SSE stream from :streamGenerateContent?alt=sse.

    Each ``data:`` line is a full GenerateContentResponse JSON object
    with partial content; we accumulate text and tool calls across
    events. ``publish`` receives a ``RuntimeEvent`` per chunk; ``None``
    disables streaming.
    """
    text_chunks: list[str] = []
    text_signature: str = ""
    thinking_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: MutableJSON = {}
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
                event = cast(MutableJSON, json.loads(data_str))
            except json.JSONDecodeError as exc:
                malformed_chunks += 1
                logger.warning("Google stream malformed JSON chunk: %s", exc)
                continue
            parsed_chunks += 1
            if chunk_unwrap is not None:
                event = chunk_unwrap(event)
            event_usage = event.get("usageMetadata")
            if isinstance(event_usage, dict):
                usage = cast(MutableJSON, event_usage)
            candidates = cast(list[MutableJSON], event.get("candidates") or [])
            if not candidates:
                continue
            first = candidates[0]
            fr = first.get("finishReason")
            if isinstance(fr, str) and fr:
                finish_reason = fr
            content = cast(MutableJSON, first.get("content") or {})
            parts = cast(list[MutableJSON], content.get("parts") or [])
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
                    fc = cast(MutableJSON, part["functionCall"])
                    fc_name = fc.get("name") or ""
                    fc_args = cast(MutableJSON, fc.get("args") or {})
                    if isinstance(fc_name, str):
                        tc_id = f"call_{uuid.uuid4().hex[:24]}"
                        tool_calls.append(
                            ToolCall(
                                id=tc_id,
                                name=fc_name,
                                args=cast(Mapping[str, object], fc_args),
                                thought_signature=cast(
                                    str, part.get("thoughtSignature", "")
                                ),
                            )
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
    usage: MutableJSON,
    finish_reason: str | None,
    model: _GeminiModel,
) -> ModelResponse:
    """Build a ``ModelResponse`` from Gemini's parsed stream fields."""
    output_tokens = IntCodec.coerce(usage.get("candidatesTokenCount"), 0)
    cache_read = IntCodec.coerce(usage.get("cachedContentTokenCount"), 0)
    # ``promptTokenCount`` is cache-inclusive; store the non-cached remainder so
    # ``TokenCount.input_tokens`` is disjoint from ``cache_read_tokens``.
    input_tokens = max(
        0, IntCodec.coerce(usage.get("promptTokenCount"), 0) - cache_read
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
