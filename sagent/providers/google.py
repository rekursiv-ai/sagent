"""Google provider (Gemini models).

Usage::

    from sagent.providers import Google

    provider = Google.from_key("AIza...")
    # or: export GOOGLE_API_KEY=AIza... and use Google.from_env()
    flash = provider.model("gemini-3-flash-preview")
    response = await flash.buffer(request)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast

import asyncio
import base64
import json
import logging
import math
import os
import uuid


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
from sagent.providers.lib.stop_reason import normalize_stop_reason


logger = logging.getLogger(__name__)

_STREAM_IDLE_TIMEOUT = 600.0
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class Google:
    """Google provider - creates Gemini model backends."""

    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    DEFAULT_UTILITY_MODEL = "gemini-3-flash-preview"

    # Model limits and pricing.
    # Limits: https://ai.google.dev/gemini-api/docs/models
    # Pricing: https://ai.google.dev/gemini-api/docs/pricing
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the Gemini API docs for the model's
    # input token limit and max output tokens.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "gemini-3-flash-preview": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.50,
                response=3.00,
                cache_read=0.05,
            ),
        ),
        "gemini-3.1-pro-preview": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=2.00,
                response=12.00,
                cache_read=0.20,
            ),
        ),
        "gemini-2.0-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.10,
                response=0.40,
                cache_read=0.025,
            ),
        ),
        "gemini-2.5-flash-lite": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.10,
                response=0.40,
                cache_read=0.025,
            ),
        ),
        "gemini-2.5-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.30,
                response=2.50,
                cache_read=0.075,
            ),
        ),
        "gemini-2.5-pro": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.25,
                response=10.0,
                cache_read=0.31,
            ),
        ),
        "gemini-1.5-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=0.075,
                response=0.3,
                cache_read=0.01875,
            ),
        ),
        "gemini-1.5-pro": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(
                request=1.25,
                response=5.0,
                cache_read=0.3125,
            ),
        ),
    }

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_key(cls, api_key: str) -> Google:
        """Create provider from an API key.

        Args:
          api_key: Google API key (``AIza...``).

        Returns:
          provider: Google provider instance.

        """
        return cls(api_key=api_key)

    @classmethod
    def from_env(cls) -> Google:
        """Create provider from GOOGLE_API_KEY env var.

        Returns:
          provider: Google provider instance.

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
        max_request_tokens: int | None = None,
    ) -> _GeminiModel:
        """Create a model backend. ``None`` → ``DEFAULT_MODEL``.

        Args:
          model_id: Model ID (e.g. ``"gemini-3-flash-preview"``). ``None`` for default.
          max_request_tokens: Max request tokens; ``None`` uses the provider default.

        Returns:
          model: Model backend implementing Model.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        # Fail fast -- every supported model must be in KNOWN_MODELS.
        profile = self.KNOWN_MODELS.get(mid)
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for Google. Known models: {known}",
            )
        return _GeminiModel(
            provider=self,
            model_id=mid,
            profile=profile,
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens
            ),
        )

    def utility_model(self) -> _GeminiModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Model backend for ``DEFAULT_UTILITY_MODEL``.

        """
        return self.model(self.DEFAULT_UTILITY_MODEL)


class _GeminiModel:
    """Gemini model backend."""

    def __init__(
        self,
        provider: Google,
        model_id: str,
        profile: ModelProfile,
        max_request_tokens: int,
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._profile = profile
        self._max_request_tokens = max_request_tokens
        # Persistent client for connection reuse.
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reused ``AsyncClient``. Lazy-created."""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient()
            return self._client

    @property
    def max_request_tokens(self) -> int:
        """Max input tokens for this model."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Gemini model identifier string."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Max output tokens for this model."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming responses."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether this backend supports extended thinking."""
        return False

    @property
    def supports_effort(self) -> bool:
        """Whether this backend supports effort-level control."""
        return False

    @property
    def supports_cache_control(self) -> bool:
        """Whether this backend supports prompt caching directives."""
        return False

    @property
    def supports_context_management(self) -> bool:
        """Whether this backend supports context-window management."""
        return False

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether this backend supports persistent server-side retry."""
        return False

    @property
    def supports_account_auth(self) -> bool:
        """Whether this backend uses account-based authentication."""
        return False

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate token count for text using 4 chars/token heuristic.

        Args:
          text: Input text to estimate.

        Returns:
          count: Approximate token count.

        """
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        return self._profile.pricing

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate token count for an image using Gemini's tile formula.

        Args:
          data: Raw image bytes.

        Returns:
          count: Approximate token count (258 per 512×512 tile).

        """
        # Gemini uses 258 tokens per tile (tile-based, similar to OpenAI).
        # https://discuss.ai.google.dev/t/gemini-pro-image-pricing-by-tile-or-fixed/40839
        # Exact tile size undocumented; using OpenAI's 512x512 as proxy.
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        tiles = math.ceil(dims[0] / 512) * math.ceil(dims[1] / 512)
        return tiles * 258

    @property
    def max_image_dim(self) -> int:
        """Max image dimension in pixels (width or height)."""
        return 3072

    @property
    def max_image_bytes(self) -> int:
        """Max image size in bytes (20 MiB)."""
        return 20 * 1024 * 1024

    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error indicates context-window overflow.

        Args:
          error: Exception raised by an API call.

        Returns:
          overflow: ``True`` if the error message indicates the prompt was too long.

        """
        msg = str(error).lower()
        return "too large" in msg or "too long" in msg or "exceeds the maximum" in msg

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """No provider-specific transient cases beyond status codes.

        Gemini errors are ``httpx.HTTPStatusError`` with a status code;
        the shared status-code path in ``retry.py`` covers them.
        """
        del error
        return False

    async def buffer(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Send a buffered request to Gemini.

        Args:
          request: Model request.

        Returns:
          response: Translated model response.

        """
        url = f"{_API_BASE}/models/{self._model_id}:generateContent"
        body = _build_request(request, self.max_image_dim, self.max_image_bytes)
        client = await self._get_client()
        r = await client.post(
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._provider.api_key,
            },
            timeout=120.0,
        )
        if r.status_code == 400:
            msg = r.text.lower()
            if "too large" in msg or "too long" in msg or "context" in msg:
                raise PromptTooLongError(r.text)
            raise ValueError(f"Google API 400: {r.text}")
        r.raise_for_status()
        resp = _parse_response(r.json(), self._profile.pricing)
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
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream via :streamGenerateContent (alt=sse).

        Args:
          request: Model request.
          on_text: Optional callback invoked with each text chunk as it arrives.
          on_thinking: Optional callback for thinking chunks. Gemini's
              streaming response surfaces thinking parts inline; this
              wraps them so the renderer can show them as they arrive.

        Returns:
          response: Translated model response assembled from streamed chunks.

        """
        del on_thinking  # gemini streams thinking inline as text; no separate hook
        url = f"{_API_BASE}/models/{self._model_id}:streamGenerateContent?alt=sse"
        body = _build_request(request, self.max_image_dim, self.max_image_bytes)
        client = await self._get_client()
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._provider.api_key,
            },
            # Idle detection is handled by the asyncio watchdog
            # inside ``_consume_gemini_stream``; give httpx a
            # generous read deadline.
            timeout=httpx.Timeout(_STREAM_IDLE_TIMEOUT, connect=30.0),
        ) as r:
            if r.status_code == 400:
                err_body = (await r.aread()).decode(errors="replace")
                msg = err_body.lower()
                if "too large" in msg or "too long" in msg or "context" in msg:
                    raise PromptTooLongError(err_body)
                raise ValueError(f"Google API 400: {err_body}")
            r.raise_for_status()
            return await _consume_gemini_stream(
                r, on_text=on_text, pricing=self._profile.pricing
            )


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
    max_image_dim: int = 3072,
    max_image_bytes: int = 20 * 1024 * 1024,
) -> MutableJSON:
    """Convert ModelRequest to Gemini API format."""
    # Build tool_use_id → function name mapping from model responses.
    call_names: dict[str, str] = {}
    for msg in request.messages:
        if msg.descriptor == "multipart/x-model-message":
            for part in cast(tuple[Message, ...], msg.content):
                if part.descriptor == "multipart/x-tool-call":
                    qid = get_queue_id(part)
                    if qid:
                        call_names[qid] = get_tool_name(part)

    contents: list[MutableJSON] = []
    pending_tool_parts: list[MutableJSON] = []
    for msg in request.messages:
        if msg.descriptor == "text/x-user-message":
            _flush_tool_parts(contents, pending_tool_parts)
            contents.append(
                cast(MutableJSON, {"role": "user", "parts": [{"text": msg.content}]})
            )
        elif msg.descriptor == "multipart/x-user-message":
            _flush_tool_parts(contents, pending_tool_parts)
            parts: list[MutableJSON] = []
            for part in cast(tuple[Message, ...], msg.content):
                if part.descriptor == "text/plain":
                    parts.append({"text": cast(str, part.content)})
                elif is_image(part.descriptor) or (
                    part.descriptor == "application/pdf"
                ):
                    raw = cast(bytes, part.content)
                    mime = part.descriptor
                    if is_image(mime):
                        raw, mime = image_lib.resize(
                            raw, max_dim=max_image_dim, max_bytes=max_image_bytes
                        )
                    b64 = base64.b64encode(raw).decode()
                    parts.append({"inlineData": {"mimeType": mime, "data": b64}})
            if not parts:
                parts.append({"text": ""})
            contents.append(cast(MutableJSON, {"role": "user", "parts": parts}))
        elif msg.descriptor == "multipart/x-model-message":
            _flush_tool_parts(contents, pending_tool_parts)
            msg_parts = cast(tuple[Message, ...], msg.content)
            parts = []
            for part in msg_parts:
                if part.descriptor == "text/plain":
                    parts.append({"text": cast(str, part.content)})
                elif part.descriptor == "multipart/x-tool-call":
                    directive = get_directive(part)
                    parts.append(
                        {
                            "functionCall": {
                                "name": get_tool_name(part),
                                "args": json_unfreeze(directive),
                            },
                        }
                    )
            if parts:
                contents.append(cast(MutableJSON, {"role": "model", "parts": parts}))
        elif msg.descriptor == "multipart/x-tool-result":
            qid = get_queue_id(msg)
            func_name = call_names.get(qid, qid)
            parts_tr = cast(tuple[Message, ...], msg.content)
            text = "\n".join(
                str(p.content)
                for p in parts_tr
                if p.descriptor in ("text/plain", "text/x-error")
            )
            image_parts_res: list[Message] = [
                p
                for p in parts_tr
                if is_image(p.descriptor) or p.descriptor == "application/pdf"
            ]
            pending_tool_parts.append(
                {
                    "functionResponse": {
                        "name": func_name,
                        "response": {"content": text},
                    },
                }
            )
            # Gemini accepts inlineData parts alongside functionResponse
            # in the same user-role message.
            for part in image_parts_res:
                raw = cast(bytes, part.content)
                mime_type = part.descriptor
                if is_image(part.descriptor):
                    raw, mime_type = image_lib.resize(
                        raw, max_dim=max_image_dim, max_bytes=max_image_bytes
                    )
                b64 = base64.b64encode(raw).decode()
                pending_tool_parts.append(
                    {"inlineData": {"mimeType": mime_type, "data": b64}}
                )
    _flush_tool_parts(contents, pending_tool_parts)

    gen_config: MutableJSON = cast(MutableJSON, {"temperature": request.temperature})
    if request.max_response_tokens is not None:
        gen_config["maxOutputTokens"] = request.max_response_tokens
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
            {
                "parts": [{"text": request.system}],
            },
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
                                json_unfreeze(t.directive_schema)
                            ),
                        }
                        for t in request.tools
                    ],
                },
            ],
        )
    return body


def _flush_tool_parts(contents: list[MutableJSON], pending: list[MutableJSON]) -> None:
    """Emit buffered tool-response parts as a user message, then clear.

    Gemini batches consecutive functionResponse (and tool-result
    inlineData) parts into a single user-role content.
    """
    if pending:
        contents.append({"role": "user", "parts": list(pending)})
        pending.clear()


async def _consume_gemini_stream(
    r: httpx.Response,
    *,
    on_text: Callable[[str], None] | None,
    pricing: Pricing,
    chunk_unwrap: Callable[[MutableJSON], MutableJSON] | None = None,
) -> ModelResponse:
    """Parse SSE stream from :streamGenerateContent?alt=sse.

    Each ``data:`` line is a full GenerateContentResponse JSON object
    with partial content; we accumulate text and tool calls across
    events.

    Args:
      r: The streaming httpx response.
      on_text: Optional callback invoked with each text delta.
      pricing: Pricing struct used to compute final cost.
      chunk_unwrap: Optional transform applied to each parsed chunk
        before consumption. Used by the Code Assist subscription
        provider, whose chunks are wrapped as ``{"response": <Gemini-
        chunk>, "traceId": ..., "consumedCredits": ...}``; passing
        ``lambda c: c.get("response") or {}`` unwraps to the standard
        Gemini shape this function expects.

    """
    text_chunks: list[str] = []
    tool_parts: list[Message] = []
    usage: MutableJSON = {}
    finish_reason: str | None = None

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
            except json.JSONDecodeError:
                continue
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
                    chunk = part.get("text")
                    if isinstance(chunk, str):
                        text_chunks.append(chunk)
                        if on_text is not None:
                            on_text(chunk)
                elif "functionCall" in part:
                    fc = cast(MutableJSON, part["functionCall"])
                    fc_name = fc.get("name") or ""
                    fc_args = cast(MutableJSON, fc.get("args") or {})
                    if isinstance(fc_name, str):
                        tc_id = f"call_{uuid.uuid4().hex[:24]}"
                        tool_parts.append(
                            tool_call_message(tc_id, fc_name, json_freeze(fc_args))
                        )

    input_tokens = int_val(usage.get("promptTokenCount"), 0)
    output_tokens = int_val(usage.get("candidatesTokenCount"), 0)
    cache_read = int_val(usage.get("cachedContentTokenCount"), 0)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        max(0, input_tokens - cache_read),
        output_tokens,
        cache_read=cache_read,
    )
    message_id = f"gemini_{uuid.uuid4().hex[:16]}"
    msg_parts: list[Message] = []
    if text_chunks:
        msg_parts.append(TextMessage("".join(text_chunks), "text/plain"))
    msg_parts.extend(tool_parts)
    has_tool_use = bool(tool_parts)
    all_parts_g: list[Message] = [
        TextMessage(message_id, "text/x-queue-id"),
        *msg_parts,
    ]
    return ModelResponse(
        content=MultipartMessage(
            tuple(all_parts_g),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            finish_reason,
            kind="google",
            has_tool_use=has_tool_use,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )


def _parse_response(data: MutableJSON, pricing: Pricing) -> ModelResponse:
    """Convert Gemini response to ModelResponse."""
    candidates_raw = cast(list[MutableJSON], data.get("candidates") or [])
    if not candidates_raw:
        raise ValueError("Gemini response contains no candidates.")
    candidate = candidates_raw[0]
    content_obj = cast(MutableJSON, candidate.get("content") or {})
    raw_parts = cast(list[MutableJSON], content_obj.get("parts") or [])
    msg_parts: list[Message] = []

    for part in raw_parts:
        if "text" in part:
            msg_parts.append(TextMessage(str(part["text"]), "text/plain"))
        elif "functionCall" in part:
            fc = cast(MutableJSON, part["functionCall"])
            fc_name = str(fc.get("name") or "")
            tc_id = f"call_{uuid.uuid4().hex[:24]}"
            msg_parts.append(
                tool_call_message(
                    tc_id, fc_name, json_freeze(cast(MutableJSON, fc.get("args") or {}))
                )
            )

    usage = cast(MutableJSON, data.get("usageMetadata") or {})
    input_tokens = int_val(usage.get("promptTokenCount"), 0)
    output_tokens = int_val(usage.get("candidatesTokenCount"), 0)
    cache_read = int_val(usage.get("cachedContentTokenCount"), 0)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        max(0, input_tokens - cache_read),
        output_tokens,
        cache_read=cache_read,
    )
    # Gemini doesn't expose a stable message id - synthesize one.
    message_id = f"gemini_{uuid.uuid4().hex[:16]}"
    finish_reason = candidate.get("finishReason")
    has_tool_use = any(p.descriptor == "multipart/x-tool-call" for p in msg_parts)
    all_parts_nr: list[Message] = [
        TextMessage(message_id, "text/x-queue-id"),
        *msg_parts,
    ]
    return ModelResponse(
        content=MultipartMessage(
            tuple(all_parts_nr),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            finish_reason if isinstance(finish_reason, str) else None,
            kind="google",
            has_tool_use=has_tool_use,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )
