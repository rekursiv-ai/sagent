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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar, Self, cast

import asyncio
import base64
import contextlib
import dataclasses
import json
import logging
import math
import os


if TYPE_CHECKING:
    import httpx
    import tiktoken

    import sagent.lib.image as image_lib
else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")  # 100ms cold
    image_lib = lazy_import("sagent.lib.image")
    tiktoken = lazy_import("tiktoken")  # 30ms cold

from sagent.lib import debug_log, token_count
from sagent.lib.custom_json import (
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
from sagent.providers.lib.errors import (
    error_status_code,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.usage import openai_usage
from sagent.thinking import ThinkingCapability, valid_thinking_states
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
    TokenCount,
    UsageSnapshot,
    base_model_id,
    latency_from_model_id,
    strip_latency_tags,
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

# Map sagent's effort vocabulary onto the effort levels OpenAI's reasoning
# models actually accept (``minimal``/``low``/``medium``/``high``). The agent
# advertises a wider set (``none``..``max``) for cross-provider uniformity, so
# both OpenAI transports -- chat-completions (``reasoning_effort`` here) and the
# subscription Responses API (``reasoning.effort`` in ``openai_sub``) -- must
# funnel through this single table. Sending a raw ``none``/``xhigh``/``max`` is
# a 400 on the wire, and two transports disagreeing is a contract drift bug.
OPENAI_REASONING_EFFORT: dict[str, str] = {
    "none": "minimal",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class OpenAICompat:
    """Base provider for OpenAI chat-completions compatible endpoints."""

    DEFAULT_MODEL: ClassVar[str] = ""
    DEFAULT_UTILITY_MODEL: ClassVar[str] = ""
    ENV_VAR: ClassVar[str] = ""
    BASE_URL: ClassVar[str] = ""
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {}
    MODEL_CLASS: ClassVar[type[OpenAICompatModel]]

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL

    @classmethod
    def from_key(cls, api_key: str, *, base_url: str | None = None) -> Self:
        """Build provider from an API key.

        Args:
          api_key: API key for the chat-completions endpoint.
          base_url: Override for the provider's ``BASE_URL``.

        Returns:
          provider: Configured provider instance.

        """
        return cls(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls, *, base_url: str | None = None) -> Self:
        """Build provider from the class's ``ENV_VAR``.

        Args:
          base_url: Override for the provider's ``BASE_URL``.

        Returns:
          provider: Configured provider instance.

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
          max_request_tokens: Override max input tokens.

        Returns:
          model: Chat-completions model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``, or it
              carries a ``+fast`` tag the backend has no latency mode for.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        # Strip only latency tags for the lookup: catalogs here carry no
        # ``+1m`` variants, so a foreign context tag stays an error.
        profile = self.KNOWN_MODELS.get(mid) or self.KNOWN_MODELS.get(
            strip_latency_tags(mid),
        )
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for {type(self).__name__}."
                f" Known models: {known}",
            )
        model = self.MODEL_CLASS(
            provider=self,
            model_id=mid,
            profile=profile,
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens
            ),
        )
        if (
            latency_from_model_id(mid) is not None
            and "fast" not in model.valid_latency_modes
        ):
            raise ValueError(
                f"Model {mid!r} for {type(self).__name__} does not support"
                " fast mode (+fast)",
            )
        return model

    def utility_model(self) -> OpenAICompatModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Backend for the cheapest/fastest known model.

        """
        mid = self.DEFAULT_UTILITY_MODEL or self.DEFAULT_MODEL
        return self.model(mid)


def _is_context_overflow_text(msg: str) -> bool:
    """True if the error body text describes a context-window overflow.

    Callers MUST run ``is_request_too_large`` first and short-circuit on a
    byte wire-limit: the ``"input too large"`` phrase below is byte-ambiguous,
    so this helper is only sound when the byte case has already been excluded
    (see :meth:`OpenAICompatModel.is_context_overflow`).
    """
    with contextlib.suppress(json.JSONDecodeError):
        raw_body = json.loads(msg)
        if isinstance(raw_body, Mapping):
            body = cast(Mapping[str, object], raw_body)
            raw_error = body.get("error")
            if isinstance(raw_error, Mapping):
                error = cast(Mapping[str, object], raw_error)
                code = error.get("code")
                if code is not None:
                    return code == "context_length_exceeded"
    lower = msg.lower()
    if "context_length_exceeded" in lower or "prompt too long" in lower:
        return True
    if "maximum context length" in lower or "input too large" in lower:
        return True
    if "context window" in lower and (
        "exceed" in lower or "overflow" in lower or "maximum" in lower
    ):
        return True
    return "model context" in lower and "exceed" in lower


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
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._last_usage: UsageSnapshot | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reused ``AsyncClient`` (per-model). Lazy-created."""
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
    def _wire_model_id(self) -> str:
        """Model id sent on the wire, stripped of any ``+1m``/``+200k`` tag.

        The window suffix is a sagent-local budget selector; the OpenAI API
        rejects it as an unknown model. Capability lookups and metadata key off
        the base id too, so the wire name must drop the tag.
        """
        return base_model_id(self._model_id)

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
        """Whether the model exposes a reasoning/thinking field."""
        return self._reasoning_field is not None

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        """OpenAI-compat vendors surface readable reasoning, no redaction.

        Vendors exposing a ``reasoning_content`` field (Moonshot,
        MiniMax, DashScope) return readable text; plain chat-completions
        (no reasoning field) supports only ``off-hide``. None expose a
        server-side redaction mode.
        """
        return valid_thinking_states(
            ThinkingCapability(supports_thinking=self.supports_thinking),
        )

    @property
    def supports_effort(self) -> bool:
        """Whether the model accepts a ``reasoning_effort`` hint."""
        return self._is_effort_model(self._model_id)

    @property
    def valid_efforts(self) -> tuple[str, ...]:
        """Effort levels accepted by reasoning models; empty otherwise."""
        if not self.supports_effort:
            return ()
        return ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt caching."""
        return False

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """OpenAI-compat vendors (Moonshot, MiniMax, DashScope) have no tier knob."""
        return ()

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """OpenAI-compat vendors expose no fast-latency path."""
        return ()

    def effective_service_tier(self, request: ModelRequest) -> str | None:
        """Resolve the wire ``service_tier``, folding in ``latency="fast"``.

        OpenAI has no separate fast-mode field; the fast path is just the
        ``priority`` processing tier. ``latency="fast"`` therefore maps to
        ``service_tier="priority"`` and wins over an explicit
        ``service_tier`` when both are set.

        Cost note: OpenAI's per-tier pricing is not modeled here (no public
        per-tier rate data wired into ``Pricing``), so a ``priority``
        request is currently billed at the model's standard rates. Anthropic
        fast mode, by contrast, is server-authoritative via ``usage.speed``.

        Args:
          request: Outgoing model request.

        Returns:
          tier: Service tier to send, or ``None`` to omit the field.

        """
        if request.latency == "fast" and "fast" in self.valid_latency_modes:
            return "priority"
        if request.service_tier is not None and (
            request.service_tier in self.valid_service_tiers
        ):
            return request.service_tier
        return None

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
        """Local estimate via ``len(text) // 4``."""
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        return self._profile.pricing

    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate via OpenAI's tile formula (``85 + tiles * 170``).

        References:
          https://platform.openai.com/docs/guides/vision/calculating-costs

        """
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        tiles = math.ceil(dims[0] / 512) * math.ceil(dims[1] / 512)
        return 85 + tiles * 170

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Local tiktoken count when an encoding for ``model_id`` exists.

        Falls back to :meth:`approx_text_tokens` for non-OpenAI models
        whose ids tiktoken doesn't recognize (Kimi, Qwen, MiniMax, etc.).
        """
        enc = self._tiktoken_encoding()
        if enc is None:
            return self.approx_text_tokens(text)
        return len(enc.encode(text))

    async def actual_image_tokens(self, data: bytes) -> int:
        """Delegate to the local heuristic (OpenAI's published tile formula)."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Local tiktoken-driven walk; falls back to approx without an encoding."""
        enc = self._tiktoken_encoding()
        if enc is None:
            return self.approx_request_tokens(request)
        return token_count.approx_request_tokens(
            request, _TiktokenEstimator(encoding=enc, image_fallback=self)
        )

    def _tiktoken_encoding(self) -> tiktoken.Encoding | None:
        """Return tiktoken's encoding for this model, or ``None`` if unknown."""
        try:
            return tiktoken.encoding_for_model(self._wire_model_id)
        except KeyError:
            return None

    @property
    def max_image_dim(self) -> int:
        """Maximum image edge (pixels) accepted, from the model profile."""
        return self._profile.max_image_dim

    @property
    def max_image_bytes(self) -> int:
        """Maximum size (bytes) of a single image, from the model profile."""
        return self._profile.max_image_bytes

    @property
    def max_request_bytes(self) -> int:
        """Maximum request-body size (bytes), from the model profile."""
        return self._profile.max_request_bytes

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
        return _is_context_overflow_text(str(error))

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """No provider-specific transient cases beyond status codes.

        Args:
          error: Exception raised by the provider call.

        Returns:
          retryable: Always ``False`` (status-code dispatch covers retries).

        """
        del error
        return False

    @property
    def _endpoint(self) -> str:
        """Chat-completions URL for the wrapped provider."""
        return f"{self._provider.base_url.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers (Authorization + JSON content-type)."""
        return {
            "Authorization": f"Bearer {self._provider.api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, request: ModelRequest, *, stream: bool) -> MutableJSON:
        """Build the chat-completions request body."""
        body: MutableJSON = cast(
            MutableJSON,
            {
                "model": self._wire_model_id,
                "messages": build_messages(
                    request, self.max_image_dim, self.max_image_bytes
                ),
                "temperature": request.temperature,
            },
        )
        if request.max_response_tokens is not None:
            # OpenAI reasoning models (gpt-5 / o-series) reject ``max_tokens``
            # with a 400 and require ``max_completion_tokens``; the same model
            # set is gated by ``supports_effort``. Other compat vendors
            # (Moonshot, MiniMax, DashScope) still take ``max_tokens``.
            field = "max_completion_tokens" if self.supports_effort else "max_tokens"
            body[field] = request.max_response_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = cast(MutableJSONValue, {"include_usage": True})
        if request.effort is not None and self.supports_effort:
            # Map through the shared table: the wire accepts only a subset of
            # sagent's advertised efforts. An unknown value should be impossible
            # (the agent's effort setter validates against ``valid_efforts``), so
            # warn rather than silently run at the most expensive level.
            mapped = OPENAI_REASONING_EFFORT.get(request.effort)
            if mapped is None:
                logger.warning(
                    "unknown reasoning effort %r; defaulting to 'high'",
                    request.effort,
                )
                mapped = "high"
            body["reasoning_effort"] = mapped
        tier = self.effective_service_tier(request)
        if tier is not None:
            body["service_tier"] = tier
        debug_log.trace(
            "api_call",
            kind="openai_chat",
            model=self._wire_model_id,
            latency=request.latency,
            service_tier=tier,
        )
        if request.tools:
            body["tools"] = cast(
                MutableJSONValue,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": json_unfreeze(t.directive_schema),
                        },
                    }
                    for t in request.tools
                ],
            )
        return self._transform_body(body, request)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request via the streaming path with no callbacks.

        The non-streaming POST has a fixed client timeout that fails
        on large compaction prompts; the streaming path uses an idle-
        based timeout that scales with response time.

        Args:
          request: Fully-built model request.

        Returns:
          response: Parsed ``ModelResponse``.

        Raises:
          PromptTooLongError: Server reports context overflow.

        """
        return await self.stream(request, None)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Stream a chat-completions SSE response.

        An asyncio watchdog resets on every delivered chunk so a server
        that goes silent mid-stream trips ``asyncio.TimeoutError`` after
        ``_STREAM_IDLE_TIMEOUT`` rather than hanging for the full HTTP
        read deadline.

        Args:
          request: Fully-built model request.
          publish: Called per streamed ``RuntimeEvent``; ``None`` disables
              live streaming.

        Returns:
          response: Parsed ``ModelResponse``.

        Raises:
          PromptTooLongError: Server reports context overflow.

        """
        body = self._build_body(request, stream=True)
        client = await self._get_client()
        async with client.stream(
            "POST",
            self._endpoint,
            json=body,
            headers={**self._headers, "Accept": "text/event-stream"},
            timeout=httpx.Timeout(_STREAM_IDLE_TIMEOUT, connect=30.0),
        ) as r:
            if 400 <= r.status_code < 500:
                err_body = (await r.aread()).decode(errors="replace")
                raise_if_request_too_large(r.status_code, err_body)
                if _is_context_overflow_text(err_body):
                    raise PromptTooLongError(err_body)
            r.raise_for_status()
            self._last_usage = openai_usage(r.headers)
            return await consume_stream(
                r,
                publish=publish,
                pricing=self._profile.pricing,
                reasoning_field=self._reasoning_field,
            )

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Return normalized usage from the latest response's headers."""
        return self._last_usage


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _TiktokenEstimator:
    """``TokenEstimator`` adapter using ``tiktoken`` for text and a fallback for images.

    Attributes:
      encoding: tiktoken encoding for the active model.
      image_fallback: ``OpenAICompatModel`` whose ``approx_image_tokens``
          provides image-token estimates (tiktoken doesn't handle images).

    """

    encoding: tiktoken.Encoding
    image_fallback: OpenAICompatModel

    def approx_text_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def approx_image_tokens(self, data: bytes) -> int:
        return self.image_fallback.approx_image_tokens(data)


def build_messages(
    request: ModelRequest,
    max_image_dim: int = 0,
    max_image_bytes: int = 0,
) -> list[MutableJSON]:
    """Convert history entries to OpenAI chat-completions format.

    Tool-result images can't ride inside a ``role=tool`` message under
    chat completions; they're hoisted into a synthetic follow-up
    ``role=user`` message. Consecutive tool result images batch into a
    single such follow-up to preserve the "tool messages must be
    contiguous" rule.

    Args:
      request: Model request with history and optional system prompt.
      max_image_dim: Maximum image dimension (pixels); larger inputs are
          resized before encoding.
      max_image_bytes: Maximum image size in bytes after resize.

    Returns:
      messages: Chat-completions wire-format messages.

    """
    ids = IdRemapper("call_")
    messages: list[MutableJSON] = []
    pending_images: list[MutableJSON] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            _flush_images(messages, pending_images)
            messages.append(_build_user_message(entry, max_image_dim, max_image_bytes))
        elif isinstance(entry, AssistantMessage):
            _flush_images(messages, pending_images)
            tool_calls_wire: list[MutableJSON] = [
                {
                    "id": ids.map(tc.id),
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(dict(tc.args)),
                    },
                }
                for tc in entry.tool_calls
            ]
            m: MutableJSON = {
                "role": "assistant",
                "content": entry.text or None,
            }
            if tool_calls_wire:
                m["tool_calls"] = cast(MutableJSONValue, tool_calls_wire)
            messages.append(m)
        else:
            # ToolResult: role=tool with text content; image attachments
            # become a follow-up role=user message (chat-completions has
            # no in-tool-message image support).
            content = entry.content
            if entry.is_error and content:
                content = f"[Error] {content}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ids.map(entry.call_id),
                    "content": content,
                }
            )
            for att in entry.attachments:
                if _is_image_mime(att.descriptor):
                    raw, mime_type = image_lib.resize(
                        att.data,
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
    """Emit buffered image blocks as a synthetic user message, then clear."""
    if pending:
        messages.append({"role": "user", "content": list(pending)})
        pending.clear()


def _build_user_message(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
) -> MutableJSON:
    """Build a chat-completions user message from a UserMessage."""
    image_atts = [att for att in entry.attachments if _is_image_mime(att.descriptor)]
    non_image_atts = [
        att for att in entry.attachments if not _is_image_mime(att.descriptor)
    ]
    for att in non_image_atts:
        logger.warning(
            "OpenAI-compat: skipping non-image attachment (mime=%s)",
            att.descriptor,
        )
    if not image_atts:
        return {"role": "user", "content": entry.text}
    blocks: list[MutableJSON] = []
    for att in image_atts:
        raw, mime = image_lib.resize(
            att.data,
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
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    return cast(MutableJSON, {"role": "user", "content": blocks})


def _is_image_mime(descriptor: str) -> bool:
    """True for image content types (no descriptor registry needed)."""
    return descriptor.startswith("image/")


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


async def consume_stream(
    r: httpx.Response,
    *,
    publish: Callable[[RuntimeEvent], None] | None,
    pricing: Pricing,
    reasoning_field: str | None,
) -> ModelResponse:
    """Parse an SSE stream into an assembled ``ModelResponse``.

    Tool-call arguments are streamed as sparse per-index deltas; we
    accumulate then json.loads at the end. ``reasoning_field`` (when
    set) captures the provider's reasoning/thinking delta as a string.

    Args:
      r: Open streaming response from the provider.
      publish: Called per streamed ``RuntimeEvent``; ``None`` disables
          live streaming.
      pricing: Per-token price schedule for cost computation.
      reasoning_field: Provider-specific reasoning-text field name,
          or ``None`` when the provider does not surface reasoning.

    Returns:
      response: Assembled ``ModelResponse`` with usage and cost filled in.

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
    saw_done = False
    async with asyncio.timeout_at(deadline) as watchdog:
        async for raw_line in r.aiter_lines():
            watchdog.reschedule(loop.time() + _STREAM_IDLE_TIMEOUT)
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                saw_done = True
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
                if publish is not None:
                    publish(ModelResponsePartial(content_chunk))
            if reasoning_field:
                think_chunk = delta.get(reasoning_field)
                if isinstance(think_chunk, str) and think_chunk:
                    thinking_parts.append(think_chunk)
                    if publish is not None:
                        publish(ModelResponseThinking(think_chunk))
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

    thinking_blocks: list[Mapping[str, object]] = []
    if thinking_parts:
        thinking_blocks.append(
            {"type": "reasoning", "text": "".join(thinking_parts)},
        )
    tool_calls: list[ToolCall] = []
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
        tool_calls.append(
            ToolCall(id=tc_id, name=tc_name, args=cast(Mapping[str, object], args))
        )
    asst = AssistantMessage(
        text="".join(text_parts),
        thinking_blocks=tuple(thinking_blocks),
        tool_calls=tuple(tool_calls),
    )

    total_input, output_tokens, cache_read = _extract_usage(usage)
    # OpenAI-compatible APIs report a cache-inclusive prompt total; store the
    # non-cached remainder so ``TokenCount.input_tokens`` is disjoint from the
    # cache pools (the convention the cost contract and consumers rely on).
    input_tokens = max(0, total_input - cache_read)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        input_tokens,
        output_tokens,
        cache_read=cache_read,
    )
    response = ModelResponse(
        message=asst,
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            finish_reason,
            kind="openai",
            has_tool_use=bool(tool_calls),
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )
    if not saw_done:
        raise StreamInterruptedError(response)
    return response


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
