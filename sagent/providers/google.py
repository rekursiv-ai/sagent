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
    from wrapt import lazy_import

    httpx = lazy_import("httpx")  # 100ms cold
    image_lib = lazy_import("sagent.lib.image")

from sagent.lib import token_count
from sagent.lib.json import (
    MutableJSON,
    MutableJSONValue,
    int_val,
    json_unfreeze,
)
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.thinking import ThinkingCapability, valid_thinking_states
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
    TokenCount,
    UsageSnapshot,
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

_STREAM_IDLE_TIMEOUT = 600.0
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_GOOGLE_THINKING_BUDGETS = {
    "min": 1_024,
    "low": 4_096,
    "medium": 8_192,
    "high": 16_384,
    "xhigh": 20_480,
    "max": 24_576,
}


class Google:
    """Google provider - creates Gemini model backends."""

    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    DEFAULT_UTILITY_MODEL = "gemini-2.5-flash-lite"

    # Model limits and pricing.
    # Limits: https://ai.google.dev/gemini-api/docs/models
    # Pricing: https://ai.google.dev/gemini-api/docs/pricing
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "gemini-3-flash-preview": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(request=0.50, response=3.00, cache_read=0.05),
        ),
        "gemini-3.1-pro-preview": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(request=2.00, response=12.00, cache_read=0.20),
        ),
        "gemini-2.0-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(request=0.10, response=0.40, cache_read=0.025),
        ),
        "gemini-2.5-flash-lite": ModelProfile(
            max_request_tokens=1_048_576,
            max_response_tokens=65_536,
            pricing=Pricing(request=0.10, response=0.40, cache_read=0.025),
        ),
        "gemini-2.5-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(request=0.30, response=2.50, cache_read=0.075),
        ),
        "gemini-2.5-pro": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(request=1.25, response=10.0, cache_read=0.31),
        ),
        "gemini-1.5-flash": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(request=0.075, response=0.3, cache_read=0.01875),
            supports_thinking=False,
        ),
        "gemini-1.5-pro": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=65_536,
            pricing=Pricing(request=1.25, response=5.0, cache_read=0.3125),
            supports_thinking=False,
        ),
    }

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
        max_request_tokens: int | None = None,
    ) -> _GeminiModel:
        """Create a model backend.

        Args:
          model_id: Model ID; ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override max input tokens.

        Returns:
          model: Gemini model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
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
          model: Backend for ``DEFAULT_UTILITY_MODEL``.

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

    async def close(self) -> None:
        """Close the reusable HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def max_request_tokens(self) -> int:
        """Maximum input tokens the model accepts."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Provider-specific model identifier."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Maximum output tokens the model can generate."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """Whether the model supports token-by-token streaming."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether the model supports extended thinking."""
        return self._profile.supports_thinking

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        """Gemini surfaces readable thought parts; no server-side redaction."""
        return valid_thinking_states(
            ThinkingCapability(supports_thinking=self.supports_thinking),
        )

    @property
    def supports_effort(self) -> bool:
        """Whether the model accepts an effort hint."""
        return self._profile.supports_thinking

    @property
    def valid_efforts(self) -> tuple[str, ...]:
        """Gemini effort levels (mapped to ``thinkingBudget``)."""
        if not self.supports_effort:
            return ()
        return ("min", "low", "medium", "high", "xhigh", "max")

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt caching."""
        return False

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """Gemini API has no equivalent of OpenAI's processing tier."""
        return ()

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """Gemini API exposes no fast-latency path."""
        return ()

    @property
    def supports_context_management(self) -> bool:
        """Whether the provider manages context overflow internally."""
        return False

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether the provider retries internally on transient failures."""
        return False

    @property
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account-based authentication."""
        return False

    def approx_text_tokens(self, text: str) -> int:
        """Local estimate via ``len(text) // 4`` (Gemini's heuristic)."""
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        return self._profile.pricing

    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate via Gemini's tile formula (``tiles * 258``)."""
        # Tile size undocumented; using OpenAI's 512x512 as proxy.
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        tiles = math.ceil(dims[0] / 512) * math.ceil(dims[1] / 512)
        return tiles * 258

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Delegate to the local heuristic; a single-string roundtrip would cost more than the gain."""
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Delegate to the local heuristic (Google's published tile formula)."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Call ``:countTokens`` for the exact server-side count."""
        url = f"{_API_BASE}/models/{self._model_id}:countTokens"
        body = _build_request(request, self.max_image_dim, self.max_image_bytes)
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
            msg = r.text.lower()
            if "too large" in msg or "too long" in msg or "exceeds the maximum" in msg:
                raise PromptTooLongError(r.text)
        r.raise_for_status()
        return int_val(cast(MutableJSON, r.json()).get("totalTokens"), 0)

    @property
    def max_image_dim(self) -> int:
        """Maximum image dimension (pixels) accepted by the API."""
        return 3072

    @property
    def max_image_bytes(self) -> int:
        """Maximum image size (bytes) accepted by the API."""
        return 20 * 1024 * 1024

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify an error as a context-window overflow.

        Args:
          error: Exception raised by the provider call.

        Returns:
          overflow: True when ``error`` indicates context overflow.

        """
        msg = str(error).lower()
        return "too large" in msg or "too long" in msg or "exceeds the maximum" in msg

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Classify an error using Google-specific retry heuristics.

        Args:
          error: Exception raised by the provider call.

        Returns:
          retryable: Always ``False`` (status-code dispatch covers retries).

        """
        del error
        return False

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Gemini exposes no per-window rate-limit telemetry."""
        return None

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request via the streaming path with no callbacks.

        The non-streaming ``:generateContent`` endpoint has a hard 120 s
        client timeout that fails on large compaction prompts; routing
        through ``:streamGenerateContent`` uses an idle-based timeout
        that scales with response time.

        Args:
          request: Fully-built model request.

        Returns:
          response: Parsed ``ModelResponse`` with usage and cost filled in.

        Raises:
          PromptTooLongError: Server reports context overflow.
          ValueError: Server returns ``400`` for non-overflow reasons.

        """
        return await self.stream(request, None)

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
            timeout=httpx.Timeout(_STREAM_IDLE_TIMEOUT, connect=30.0),
        ) as r:
            if 400 <= r.status_code < 500:
                err_body = (await r.aread()).decode(errors="replace")
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
            return await _consume_gemini_stream(
                r,
                publish=publish,
                pricing=self._profile.pricing,
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
                block = _attachment_part(att, max_image_dim, max_image_bytes)
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
                block = _attachment_part(att, max_image_dim, max_image_bytes)
                if block is not None:
                    pending_tool_parts.append(block)
    _flush_tool_parts(contents, pending_tool_parts)

    thinking_config = _thinking_config(request)
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


def _thinking_config(request: ModelRequest) -> MutableJSON | None:
    """Return Gemini thinking config for a sagent request."""
    if request.effort is not None:
        budget = _GOOGLE_THINKING_BUDGETS.get(request.effort)
        if budget is None:
            valid = ", ".join(sorted(_GOOGLE_THINKING_BUDGETS))
            raise ValueError(
                f"Invalid Google effort {request.effort!r}. Valid efforts: {valid}.",
            )
        return {"includeThoughts": True, "thinkingBudget": budget}
    if request.thinking in ("adaptive", "enabled"):
        return {"includeThoughts": True, "thinkingBudget": -1}
    return None


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
    r: httpx.Response,
    *,
    publish: Callable[[RuntimeEvent], None] | None = None,
    pricing: Pricing,
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
                        text_signature = cast(str, part["thoughtSignature"])
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
        pricing=pricing,
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
    pricing: Pricing,
) -> ModelResponse:
    """Build a ``ModelResponse`` from Gemini's parsed stream fields."""
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
    return ModelResponse(
        message=AssistantMessage(
            text=text,
            thought_signature=text_signature,
            thinking_blocks=({"type": "thinking", "thinking": thinking},)
            if thinking
            else (),
            tool_calls=tuple(tool_calls),
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            finish_reason,
            kind="google",
            has_tool_use=bool(tool_calls),
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )
