"""Anthropic provider (API key).

Thin API-key client. Speaks history entries
(``UserMessage`` / ``AssistantMessage`` / ``ToolResult``) directly.

Usage::

    from sagent.providers import Anthropic

    provider = Anthropic.from_key("sk-ant-...")
    # or: Anthropic.from_env()  (reads ANTHROPIC_API_KEY)
    sonnet = provider.model("claude-sonnet-4-6")
    response = await sonnet.buffer(request)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import time


if TYPE_CHECKING:
    from anthropic._models import FinalRequestOptions
    from anthropic._streaming import AsyncStream
    from anthropic.lib.streaming import AsyncMessageStream
    from anthropic.types.raw_message_stream_event import RawMessageStreamEvent

    import anthropic
    import httpx

    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    anthropic = lazy_import("anthropic")  # 569ms cold
    httpx = lazy_import("httpx")  # 168ms cold
    AsyncMessageStream = lazy_import("anthropic.lib.streaming").AsyncMessageStream
    AsyncStream = lazy_import("anthropic._streaming").AsyncStream
    image_lib = lazy_import("sagent.lib.image")

from sagent.lib import debug_log, token_count
from sagent.lib.json import MutableJSON, MutableJSONValue, json_unfreeze
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    find_response_not_read,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
    TokenCount,
    base_model_id,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)

_STREAM_IDLE_TIMEOUT = 600.0


class _AnthropicTransportClient(Protocol):
    async def send(self, request: httpx.Request, *, stream: bool) -> httpx.Response: ...


class _AnthropicRawStreamSDK(Protocol):
    _client: _AnthropicTransportClient

    def _build_request(self, options: FinalRequestOptions) -> httpx.Request: ...

    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> anthropic.APIStatusError: ...


_CONTEXT_1M_BETA = "context-1m-2025-08-07"
_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
_REDACT_THINKING_BETA = "redact-thinking-2026-02-12"
_FAST_MODE_BETA = "fast-mode-2026-02-01"
_DEFAULT_API_TARGET_INPUT_TOKENS = 40_000

# Models whose Pricing carries non-zero fast-mode rates AND whose API
# accepts ``speed="fast"``. Same set on both API-key and subscription
# transports; the CLI transport can't pass the knob.
_FAST_MODE_MODELS = frozenset(
    {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
    }
)


def supports_fast_mode(model_id: str) -> bool:
    """True when the model accepts ``speed="fast"`` on the Anthropic API.

    Anthropic fast mode is an inference-acceleration feature: it runs the
    same weights through a faster serving path for higher output
    tokens/sec, at premium pricing, and is wire-distinct from the
    ``service_tier`` capacity hint (both can be set on one request). This
    differs from OpenAI, which has no separate fast field -- its fast path
    is just ``service_tier="priority"``. The cross-provider ``latency``
    hint hides that asymmetry from callers.
    """
    return base_model_id(model_id) in _FAST_MODE_MODELS


# Models that support server-side ``clear_tool_uses_20250919``. Per
# Anthropic docs: Sonnet 4/4.5, Haiku 4.5, Opus 4/4.1/4.5. We treat
# the +1m variants identically (same base model).
_CONTEXT_MANAGEMENT_MODELS = frozenset(
    {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    }
)


def supports_native_context_management(model_id: str) -> bool:
    """True when the model accepts the ``clear_tool_uses_20250919`` beta."""
    return base_model_id(model_id) in _CONTEXT_MANAGEMENT_MODELS


def context_betas(model_id: str) -> list[str]:
    """Return beta headers required by the requested context window.

    Args:
      model_id: Model identifier, possibly with a context-window suffix.

    Returns:
      betas: Beta header strings to include in the request.

    """
    betas: list[str] = []
    if model_id.lower().endswith("+1m"):
        betas.append(_CONTEXT_1M_BETA)
    if supports_native_context_management(model_id):
        betas.append(_CONTEXT_MANAGEMENT_BETA)
    return betas


def build_context_management(
    *,
    has_thinking: bool = False,
    cache_cold: bool = False,
    server_side_context_management: bool = False,
    trigger_tokens: int = 0,
    target_input_tokens: int = _DEFAULT_API_TARGET_INPUT_TOKENS,
    tools: Sequence[Tool] = (),
) -> dict[str, list[MutableJSON]] | None:
    """Build ``context_management`` body for the Anthropic API.

    Args:
      has_thinking: Whether thinking mode is active.
      cache_cold: Whether the prompt cache TTL likely expired.
      server_side_context_management: Opt in to Anthropic's server-side
          ``clear_tool_uses_20250919`` beta. Off by default.
      trigger_tokens: Threshold at which the server starts clearing.
      target_input_tokens: Desired post-clear token count.
      tools: Advertised tools; ``Tool.clearable_results`` determines which
          tool results may be cleared.

    Returns:
      context_management: Context-management config, or ``None`` if no edits
          apply.

    """
    edits: list[MutableJSON] = []
    if has_thinking:
        keep: MutableJSON | str = (
            {"type": "thinking_turns", "value": 1} if cache_cold else "all"
        )
        edits.append({"type": "clear_thinking_20251015", "keep": keep})
    if server_side_context_management and trigger_tokens > 0:
        clearable = [t.name for t in tools if getattr(t, "clearable_results", False)]
        unclearable = [
            t.name for t in tools if not getattr(t, "clearable_results", False)
        ]
        clear_at_least = max(trigger_tokens - target_input_tokens, 1_000)
        edits.append(
            cast(
                MutableJSON,
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": trigger_tokens},
                    "clear_at_least": {
                        "type": "input_tokens",
                        "value": clear_at_least,
                    },
                    "clear_tool_inputs": clearable,
                    "exclude_tools": unclearable,
                },
            ),
        )
    return {"edits": edits} if edits else None


# Model limits and pricing.
# Limits & pricing: https://docs.anthropic.com/en/docs/about-claude/models
# Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
# Model list: anthropic.Anthropic().models.list()
#
# To add a new model: check the Anthropic docs for context window
# and max output tokens, then add a ModelProfile + pricing entry.
# Opus 4.8 fast mode is $10/$50 per MTok (2x standard). Opus 4.6 and
# 4.7 fast mode is $30/$150 per MTok (6x standard). All Opus models
# share the same standard rate so we keep one base ``_OPUS`` and split
# only the fast-mode rates. ``_OPUS`` (4.8 fast rates) is also reused by
# non-fast Opus models (e.g. 4.5); that is safe because billing keys on
# the server's ``usage.speed=="fast"`` and those models have
# ``valid_latency_modes=()`` so they never request -- nor are billed --
# fast.
# Source: https://docs.anthropic.com/en/docs/build-with-claude/fast-mode
_OPUS = Pricing(
    request=5.0,
    response=25.0,
    cache_write=6.25,
    cache_read=0.5,
    fast_request=10.0,
    fast_response=50.0,
)
_OPUS_FAST_4_6_7 = Pricing(
    request=5.0,
    response=25.0,
    cache_write=6.25,
    cache_read=0.5,
    fast_request=30.0,
    fast_response=150.0,
)
_SONNET = Pricing(
    request=3.0,
    response=15.0,
    cache_write=3.75,
    cache_read=0.3,
)
_HAIKU = Pricing(
    request=1.0,
    response=5.0,
    cache_write=1.25,
    cache_read=0.1,
)


class Anthropic:
    """Anthropic provider - API key auth.

    The base class exposes hooks that alternative auth implementations can override:
    ``get_sdk``, ``build_system``, ``extra_headers``, ``extra_body``,
    ``handle_auth_error``, ``subscription``.
    """

    # Latest model we roll to when ``model_id`` is None. Bump on release.
    DEFAULT_MODEL = "claude-opus-4-8+1m"
    DEFAULT_UTILITY_MODEL = "claude-haiku-4-5"

    # ``chars_per_token`` measured via ``messages.count_tokens`` on a 2.6M-char
    # mixed code+JSON+thinking session (de89f75430bf). Three tokenizer
    # generations cluster: opus-4-7 (2.83), opus/sonnet-4.6-4.5 (3.66),
    # sonnet-4.5 / haiku-4.5 (4.83). Pure-English content tokenizes higher;
    # these defaults err toward overcount for mixed agent traffic, which is
    # the safe direction for the compaction trigger.
    # opus-4-8 inherits 4-7's value (2.83) pending its own measurement.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "claude-opus-4-8": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=2.83,
        ),
        "claude-opus-4-8+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=2.83,
        ),
        "claude-opus-4-7": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS_FAST_4_6_7,
            chars_per_token=2.83,
        ),
        "claude-opus-4-7+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS_FAST_4_6_7,
            chars_per_token=2.83,
        ),
        "claude-opus-4-6": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS_FAST_4_6_7,
            chars_per_token=3.66,
        ),
        "claude-opus-4-6+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS_FAST_4_6_7,
            chars_per_token=3.66,
        ),
        "claude-opus-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=3.66,
        ),
        "claude-opus-4-5+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=3.66,
        ),
        "claude-sonnet-4-6": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
            chars_per_token=3.66,
        ),
        "claude-sonnet-4-6+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
            chars_per_token=3.66,
        ),
        "claude-sonnet-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
            chars_per_token=4.83,
        ),
        "claude-sonnet-4-5+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
            chars_per_token=4.83,
        ),
        "claude-haiku-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=64_000,
            pricing=_HAIKU,
            supports_thinking=False,
            chars_per_token=4.83,
        ),
    }

    def __init__(
        self,
        *,
        api_key: str,
        server_side_context_management: bool = False,
        redact_thinking: bool = False,
    ) -> None:
        self._api_key = api_key
        self._server_side_context_management = server_side_context_management
        self._redact_thinking = redact_thinking
        self._lock = asyncio.Lock()
        self._sdk: anthropic.AsyncAnthropic | None = None

    @property
    def server_side_context_management(self) -> bool:
        """Whether Anthropic's ``clear_tool_uses_20250919`` beta is enabled."""
        return self._server_side_context_management

    @classmethod
    def from_key(
        cls,
        api_key: str,
        *,
        server_side_context_management: bool = False,
        redact_thinking: bool = False,
    ) -> Anthropic:
        """Create provider from an API key.

        Args:
          api_key: Anthropic API key (``sk-ant-...``).
          server_side_context_management: Opt in to Anthropic's
              ``clear_tool_uses_20250919`` beta. Defaults to off; sagent's
              own client-side compactor handles budget. See the runtime
              prompt section "Context management" for the rationale.
          redact_thinking: Request redacted thinking blocks from Anthropic.

        Returns:
          provider: Configured Anthropic provider instance.

        """
        return cls(
            api_key=api_key,
            server_side_context_management=server_side_context_management,
            redact_thinking=redact_thinking,
        )

    @classmethod
    def from_env(
        cls,
        *,
        server_side_context_management: bool = False,
        redact_thinking: bool = False,
    ) -> Anthropic:
        """Create provider from ``ANTHROPIC_API_KEY`` env var.

        Args:
          server_side_context_management: Opt in to Anthropic's
              ``clear_tool_uses_20250919`` beta. Defaults to off.
          redact_thinking: Request redacted thinking blocks from Anthropic.

        Returns:
          provider: Configured Anthropic provider instance.

        Raises:
          RuntimeError: If the API key is not configured.

        """
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("Anthropic API key not configured.")
        return cls(
            api_key=key,
            server_side_context_management=server_side_context_management,
            redact_thinking=redact_thinking,
        )

    def model(
        self, model_id: str | None = None, max_request_tokens: int | None = None
    ) -> _AnthropicModel:
        """Create a model backend.

        Args:
          model_id: Model ID. ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override max input tokens. ``None`` uses profile default.

        Returns:
          model: Anthropic model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        # Try exact match first, then strip +1m context tag.
        # Fail fast on unknown model IDs.
        profile = self.KNOWN_MODELS.get(mid) or self.KNOWN_MODELS.get(
            base_model_id(mid),
        )
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for Anthropic. Known models: {known}",
            )
        return _AnthropicModel(
            provider=self,
            model_id=mid,
            profile=profile,
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens
            ),
        )

    def utility_model(self) -> _AnthropicModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Backend for ``DEFAULT_UTILITY_MODEL``.

        """
        return self.model(self.DEFAULT_UTILITY_MODEL)

    # -- Hooks (subclasses override) -----------------------------------

    @property
    def subscription(self) -> bool:
        """Whether this provider bills against a subscription."""
        return False

    async def get_sdk(self) -> anthropic.AsyncAnthropic:
        """Get or create the underlying Anthropic SDK client.

        Returns:
          client: Shared ``AsyncAnthropic`` instance.

        """
        async with self._lock:
            if self._sdk is None:
                self._sdk = anthropic.AsyncAnthropic(api_key=self._api_key)
            return self._sdk

    async def close_sdk(self) -> None:
        """Close and clear the shared Anthropic SDK client."""
        async with self._lock:
            if self._sdk is None:
                return
            sdk = self._sdk
            self._sdk = None
        close = getattr(sdk, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    def build_system(
        self,
        system: str | None,
        messages: list[anthropic.types.MessageParam] | None = None,
        *,
        cache_ttl: str = "5m",
    ) -> list[anthropic.types.TextBlockParam] | str | anthropic.NotGiven:
        """Build the system prompt for an API call.

        Args:
          system: System prompt text, or ``None`` to omit.
          messages: Conversation messages (used by subscription subclass).
          cache_ttl: Prompt-cache TTL (``5m`` or ``1h``) for providers using blocks.

        Returns:
          system_param: System prompt in API format, or ``NOT_GIVEN``.

        """
        del messages, cache_ttl  # unused in plain API mode
        if system is not None:
            return system
        return anthropic.NOT_GIVEN

    def extra_headers(self, model_id: str) -> dict[str, str]:
        """Return per-request extra headers.

        Args:
          model_id: Active model id used to derive beta opt-ins.

        Returns:
          headers: Header dict (``anthropic-beta`` set when betas apply).

        """
        betas = context_betas(model_id)
        if self._redact_thinking:
            betas.append(_REDACT_THINKING_BETA)
        return {"anthropic-beta": ",".join(betas)} if betas else {}

    def extra_body(
        self,
        *,
        has_thinking: bool,
        cache_cold: bool,
        trigger_tokens: int,
        tools: Sequence[Tool],
    ) -> MutableJSON | None:
        """Return per-request extra body fields.

        Args:
          has_thinking: True when the request enables extended thinking.
          cache_cold: True when the cache TTL likely expired.
          trigger_tokens: Threshold at which server-side clearing starts.
          tools: Advertised tools, used to derive clearing policy.

        Returns:
          body: Extra body payload, or ``None`` to skip.

        """
        cm = build_context_management(
            has_thinking=has_thinking,
            cache_cold=cache_cold,
            server_side_context_management=self._server_side_context_management,
            trigger_tokens=trigger_tokens,
            tools=tools,
        )
        if cm is None:
            return None
        return {"context_management": cast(MutableJSONValue, cm)}

    async def handle_auth_error(self) -> None:
        """Handle a 401 from the API. No-op for API-key auth."""
        return


_RE_ANTHROPIC_TOKENS = re.compile(r"(\d[\d,]*)\s*tokens?\s*>\s*(\d[\d,]*)")

# Statusless Anthropic errors with body-declared types in this set are
# transient retryables that the shared status-code classifier misses.
_RETRYABLE_BODY_TYPES = frozenset(
    {"api_error", "overloaded_error", "rate_limit_error", "server_error"}
)


def _is_prompt_too_long_text(
    msg: str,
    *,
    error_body: Mapping[str, object] | None = None,
) -> bool:
    """True if the error describes a context-window overflow.

    Prefers the structured ``error.type``/``error.message`` fields on
    ``error_body`` when present: an Anthropic ``invalid_request_error``
    whose ``message`` text mentions overflow phrases is canonical. When
    ``error_body`` is absent, falls back to substring matching against
    the stringified error -- which is fragile and matches benign 400s
    that incidentally mention "context window" (e.g. tool-schema
    validation errors). Callers with access to the parsed SDK body
    should always pass ``error_body``.

    Args:
      msg: Stringified error or message text.
      error_body: Parsed Anthropic error body (``e.body``), when available.

    Returns:
      overflow: True when the error indicates a prompt-too-long condition.

    """
    if error_body is not None:
        nested = error_body.get("error")
        if isinstance(nested, Mapping):
            nested_map = cast(Mapping[str, object], nested)
            error_type = nested_map.get("type")
            if error_type != "invalid_request_error":
                return False
            inner = nested_map.get("message")
            return isinstance(inner, str) and _matches_overflow_phrase(inner)
    return _matches_overflow_phrase(msg)


def _matches_overflow_phrase(msg: str) -> bool:
    """Substring fallback for Anthropic overflow phrases."""
    lower = msg.lower()
    if "too long" in lower or "too_long" in lower:
        return True
    return "context window" in lower and (
        "exceed" in lower or "overflow" in lower or "maximum" in lower
    )


def _api_status_body(error: anthropic.APIStatusError) -> Mapping[str, object] | None:
    """Return the parsed Anthropic error body, if the SDK exposed one."""
    body = getattr(error, "body", None)
    return cast(Mapping[str, object], body) if isinstance(body, Mapping) else None


def _raise_if_prompt_too_long(e: anthropic.APIStatusError) -> None:
    """Re-raise as PromptTooLongError if this is a prompt-too-long error."""
    raw = str(e)
    if not _is_prompt_too_long_text(raw, error_body=_api_status_body(e)):
        return
    actual, limit = None, None
    m = _RE_ANTHROPIC_TOKENS.search(raw)
    if m:
        actual = int(m.group(1).replace(",", ""))
        limit = int(m.group(2).replace(",", ""))
    raise PromptTooLongError(raw, actual_tokens=actual, limit_tokens=limit) from e


def _anthropic_error_type(body: Mapping[object, object]) -> object:
    """Extract the Anthropic error type from flat or nested error bodies."""
    error_type = body.get("type")
    if error_type != "error":
        return error_type
    nested = body.get("error")
    if not isinstance(nested, Mapping):
        return error_type
    return cast(Mapping[object, object], nested).get("type")


def _tool_names_from_kwargs(kwargs: dict[str, object]) -> list[str | None]:
    """Extract tool names from a request kwargs dict for logging."""
    raw: object = kwargs.get("tools") or []
    if not isinstance(raw, list):
        return []
    out: list[str | None] = []
    for t in cast(list[object], raw):
        if isinstance(t, dict):
            name = cast(dict[str, object], t).get("name")
            out.append(name if isinstance(name, str) else None)
        else:
            out.append(None)
    return out


def _guard_stream_interrupt(
    resp: ModelResponse,
    *,
    kind: str,
    model_id: str,
) -> None:
    """Raise ``StreamInterruptedError`` if a ``model_tool_use`` response arrived
    without any ``ToolCall``s.

    Gates on actual content rather than the API's ``stop_reason``, which is
    unreliable. When violated, the tool block was almost certainly dropped
    mid-stream; retry usually recovers. The partial response is carried on
    the error so the retry layer can fall back gracefully.
    """
    has_tool_calls = bool(resp.message.tool_calls)
    if resp.stop_reason == "model_tool_use" and not has_tool_calls:
        text = resp.message.text
        debug_log.trace_error(
            "stream_interrupted",
            kind=kind,
            model=model_id,
            has_text=bool(text.strip()),
            text_len=len(text),
            thinking_blocks=len(resp.message.thinking_blocks),
            request_id=resp.request_id,
            message_id=resp.message_id,
        )
        raise StreamInterruptedError(resp)


def _request_id(e: BaseException) -> str | None:
    """Best-effort request-id extractor for an anthropic error."""
    rid = getattr(e, "request_id", None)
    if rid:
        return rid
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            return headers.get("request-id") or headers.get("x-request-id")
        except Exception:  # noqa: BLE001 -- best-effort, must not mask the original error
            return None
    return None


class _AnthropicModel:
    """Claude model backend - translates ModelRequest/ModelResponse."""

    def __init__(
        self,
        provider: Anthropic,
        model_id: str,
        profile: ModelProfile,
        max_request_tokens: int = 200_000,
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._profile = profile
        self._max_request_tokens = max_request_tokens
        self._last_response_time = time.time()

    @property
    def _cache_cold(self) -> bool:
        """True when prompt cache TTL likely expired (>1h idle)."""
        return time.time() - self._last_response_time > 3600.0

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
    def supports_effort(self) -> bool:
        """Whether the model accepts an effort hint."""
        return True

    @property
    def supports_cache_control(self) -> bool:
        """Whether the provider supports prompt caching."""
        return True

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """Anthropic Messages API accepts ``auto`` (default) or ``standard_only``.

        ``auto`` uses Priority Tier capacity when available, falling back
        to standard; ``standard_only`` opts a single request out of any
        Priority commitment. See https://platform.claude.com/docs/en/api/service-tiers.
        """
        return ("auto", "standard_only")

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """``latency="fast"`` maps to ``speed="fast"`` on supported Opus models.

        Anthropic fast mode is a distinct inference-acceleration field
        (sent via ``extra_body`` plus the ``fast-mode-2026-02-01`` beta),
        orthogonal to ``service_tier`` -- both can be set on one request.
        OpenAI, by contrast, has no separate field; there ``latency="fast"``
        resolves to ``service_tier="priority"``.
        """
        return ("fast",) if supports_fast_mode(self._model_id) else ()

    @property
    def supports_context_management(self) -> bool:
        """Whether the provider manages context overflow internally."""
        return self._provider.server_side_context_management

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether the provider retries internally on transient failures."""
        return True

    @property
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account-based authentication."""
        return self._provider.subscription

    def approx_text_tokens(self, text: str) -> int:
        """Local estimate via ``chars_per_token``.

        Args:
          text: Text to score.

        Returns:
          tokens: Approximate input token count.

        """
        return int(len(text) / self._profile.chars_per_token)

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing schedule for this model."""
        return self._profile.pricing

    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate from image dimensions (Anthropic's formula).

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Approximate input token count (``width*height/750``).

        References:
          https://docs.anthropic.com/en/docs/build-with-claude/vision#calculate-image-costs

        """
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // 750 if dims is not None else 0

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Delegate to the local heuristic.

        Anthropic's ``messages.count_tokens`` endpoint operates on
        full message lists, not bare strings, so a single-string
        roundtrip would cost a request and a tail-padded tokenization
        guess. Local heuristic is the practical truth source here.
        """
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Delegate to the local heuristic (Anthropic's published formula)."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Call the server's ``messages.count_tokens`` for an exact count."""
        messages = _build_messages(request, self.max_image_dim, self.max_image_bytes)
        sdk = await self._provider.get_sdk()
        kwargs = self._build_kwargs(request, messages)
        # ``count_tokens`` accepts a subset of ``messages.create``'s kwargs.
        for k in ("max_tokens", "temperature", "stop_sequences", "service_tier"):
            kwargs.pop(k, None)
        try:
            result = await sdk.messages.count_tokens(**kwargs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- dynamic kwargs
        except anthropic.AuthenticationError:
            await self._provider.handle_auth_error()
            sdk = await self._provider.get_sdk()
            kwargs["system"] = self._provider.build_system(
                request.system, messages, cache_ttl=request.cache_ttl
            )
            result = await sdk.messages.count_tokens(**kwargs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- dynamic kwargs
        return result.input_tokens

    @property
    def max_image_dim(self) -> int:
        """Maximum image dimension (pixels) accepted by the API."""
        return 8000

    @property
    def max_image_bytes(self) -> int:
        """Maximum image size (bytes) accepted by the API."""
        return 5 * 1024 * 1024

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify an error as a context-window overflow.

        The body text is the canonical signal. The HTTP status code
        carries less information: Anthropic returns 400 with body
        ``"prompt is too long"`` and 413 with body ``"Request size
        exceeds model context window"``, but other status codes
        (uncommon but observed in production) can carry the same body
        text. Trust the body, ignore the status.

        Args:
          error: Exception raised by the provider call.

        Returns:
          overflow: True when ``error`` indicates context overflow.

        """
        if isinstance(error, PromptTooLongError):
            return True
        if not isinstance(error, anthropic.APIStatusError):
            return False
        return _is_prompt_too_long_text(str(error), error_body=_api_status_body(error))

    async def close(self) -> None:
        """Close the shared provider SDK owned by this model."""
        await self._provider.close_sdk()

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Statusless Anthropic errors that still declare retryability.

        Args:
          error: Exception raised by the provider call.

        Returns:
          retryable: True when the response body declares a known
              retryable ``type`` (e.g. ``overloaded_error``).

        """
        body = getattr(error, "body", None)
        if not isinstance(body, Mapping):
            return False
        error_type = _anthropic_error_type(cast(Mapping[object, object], body))
        return isinstance(error_type, str) and error_type in _RETRYABLE_BODY_TYPES

    def _build_kwargs(
        self,
        request: ModelRequest,
        messages: list[anthropic.types.MessageParam],
    ) -> dict[str, object]:
        """Build kwargs for the Anthropic messages API."""
        thinking = request.thinking if self.supports_thinking else None
        has_thinking = thinking in ("adaptive", "enabled")
        max_tok = request.max_response_tokens or self.max_response_tokens
        kwargs: dict[str, object] = {
            "model": base_model_id(self._model_id),
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": request.temperature,
            "system": self._provider.build_system(
                request.system, messages, cache_ttl=request.cache_ttl
            ),
        }
        if thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1.0
        elif thinking == "enabled":
            # Anthropic counts thinking against ``max_tokens``, so the
            # total must leave room for the visible reply. Double the
            # request, but clamp to the model cap (opus-4-8's cap equals
            # the API ceiling, where the raw double overflows). Split the
            # total so ``budget_tokens`` stays strictly below ``max_tokens``.
            total = min(max_tok * 2, self.max_response_tokens)
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": total // 2,
            }
            kwargs["max_tokens"] = total
            kwargs["temperature"] = 1.0
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": json_unfreeze(t.directive_schema),
                }
                for t in request.tools
            ]
        if request.effort is not None:
            kwargs["output_config"] = {"effort": request.effort}
        if (
            request.service_tier is not None
            and request.service_tier in self.valid_service_tiers
        ):
            kwargs["service_tier"] = request.service_tier
        body = self._provider.extra_body(
            has_thinking=has_thinking,
            cache_cold=self._cache_cold,
            trigger_tokens=(
                self.max_request_tokens // 2
                if supports_native_context_management(self._model_id)
                else 0
            ),
            tools=request.tools or (),
        )
        if body is not None:
            kwargs["extra_body"] = body
        headers = self._provider.extra_headers(self._model_id)
        if request.latency == "fast":
            # ``speed`` is a first-class parameter only on the SDK's beta
            # endpoints (``beta.messages.*``); the stream path here is a
            # custom raw POST to ``/v1/messages`` (see ``_raw_message_stream``),
            # so we inject ``speed`` via ``extra_body`` plus the research-
            # preview beta header. That is the wire-equivalent of calling
            # ``beta.messages.create(speed="fast", betas=[...])`` -- do not
            # drop this injection without moving the stream onto the beta SDK.
            # Reject early on unsupported models so the agent is told rather
            # than silently billed standard speed.
            if not supports_fast_mode(self._model_id):
                raise ValueError(
                    f"Model {self._model_id!r} does not support fast mode "
                    f"(latency='fast').",
                )
            body = cast(dict[str, object], kwargs.get("extra_body") or {})
            body["speed"] = "fast"
            kwargs["extra_body"] = body
            betas = headers.get("anthropic-beta", "")
            beta_list = [b for b in betas.split(",") if b]
            if _FAST_MODE_BETA not in beta_list:
                beta_list.append(_FAST_MODE_BETA)
            headers = {**headers, "anthropic-beta": ",".join(beta_list)}
        if headers:
            kwargs["extra_headers"] = headers
        return kwargs

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a request via the streaming path with no callbacks.

        The Anthropic SDK rejects non-streaming requests whose prompt
        size may exceed a 10-minute wall, so the buffered path always
        routes through ``stream`` to keep large compaction calls valid.

        Args:
          request: Fully-built model request.

        Returns:
          response: Parsed ``ModelResponse`` with usage and cost filled in.

        Raises:
          PromptTooLongError: Server reports context overflow.

        """
        return await self.stream(request, on_text=None, on_thinking=None)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream a request, calling ``on_text`` / ``on_thinking`` per chunk.

        Args:
          request: Fully-built model request.
          on_text: Called per text chunk; ``None`` disables text streaming.
          on_thinking: Called per thinking chunk; ``None`` disables it.

        Returns:
          response: Parsed ``ModelResponse`` with usage and cost filled in.

        Raises:
          PromptTooLongError: Server reports context overflow.

        """
        messages = _build_messages(request, self.max_image_dim, self.max_image_bytes)
        sdk = await self._provider.get_sdk()
        kwargs = self._build_kwargs(request, messages)
        extra_body = cast(Mapping[str, object], kwargs.get("extra_body") or {})
        extra_headers = cast(Mapping[str, str], kwargs.get("extra_headers") or {})
        debug_log.trace(
            "api_call",
            kind="stream",
            model=self._model_id,
            roles=debug_log.role_sequence(messages),
            n_messages=len(messages),
            n_tools=len(kwargs.get("tools") or []),  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- kwargs.tools always list
            thinking=kwargs.get("thinking"),
            speed=extra_body.get("speed"),
            fast_beta=_FAST_MODE_BETA in extra_headers.get("anthropic-beta", ""),
        )
        try:
            raw = await _stream_impl(sdk, kwargs, on_text, on_thinking)
        except anthropic.AuthenticationError:
            await self._provider.handle_auth_error()
            sdk = await self._provider.get_sdk()
            kwargs["system"] = self._provider.build_system(
                request.system, messages, cache_ttl=request.cache_ttl
            )
            raw = await _stream_impl(sdk, kwargs, on_text, on_thinking)
        except anthropic.APIStatusError as e:
            # Do NOT wrap a status-bearing error as StreamingResponseNotReadError
            # even when it chains a ResponseNotRead: that would turn a
            # retryable 429/529 fatal. The status carries the retry signal;
            # the unread-body crash is handled downstream in
            # ``retry.py::_response_body_excerpt``. Let the error re-raise so
            # the classifier sees the status.
            if not _is_prompt_too_long_text(str(e), error_body=_api_status_body(e)):
                raise
            debug_log.trace_error(
                "bad_request",
                kind="stream",
                model=self._model_id,
                error=str(e),
                status=getattr(e, "status_code", None),
                request_id=_request_id(e),
                roles=debug_log.role_sequence(messages),
                messages=debug_log.summarize_messages(messages),
                tools=_tool_names_from_kwargs(kwargs),
                thinking=kwargs.get("thinking"),
                system_preview=str(kwargs.get("system", ""))[:400],
            )
            _raise_if_prompt_too_long(e)
            raise
        except Exception as e:
            # A bare or non-status ``ResponseNotRead`` (directly or chained)
            # is neither an APIStatusError (so the overflow branch misses
            # it) nor a transport error (so the retry classifier deems it
            # fatal); re-wrap into a user-facing error with remediation.
            not_read = find_response_not_read(e)
            if not_read is not None:
                raise StreamingResponseNotReadError(
                    provider_name="Anthropic", cause=not_read
                ) from e
            raise
        resp = _parse_response(raw, self._profile.pricing)
        self._last_response_time = time.time()
        _guard_stream_interrupt(resp, kind="stream", model_id=self._model_id)
        return resp


async def _stream_impl(
    sdk: anthropic.AsyncAnthropic,
    kwargs: dict[str, object],
    on_text: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None = None,
) -> anthropic.types.Message:
    """Run the streaming call and return the final message.

    Routes ``text_delta`` events to ``on_text`` and ``thinking_delta``
    events to ``on_thinking`` as they arrive.
    """
    stream = await _raw_message_stream(sdk, kwargs)
    async with AsyncMessageStream(stream, output_format=anthropic.NOT_GIVEN) as s:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STREAM_IDLE_TIMEOUT
        async with asyncio.timeout_at(deadline) as watchdog:
            async for event in s:
                watchdog.reschedule(
                    loop.time() + _STREAM_IDLE_TIMEOUT,
                )
                delta = getattr(event, "delta", None)
                if delta is None:
                    continue
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta" and on_text is not None:
                    on_text(delta.text)
                elif delta_type == "thinking_delta" and on_thinking is not None:
                    on_thinking(delta.thinking)
            return await s.get_final_message()


async def _raw_message_stream(
    sdk: anthropic.AsyncAnthropic,
    kwargs: dict[str, object],
) -> AsyncStream[RawMessageStreamEvent]:
    body = {key: value for key, value in kwargs.items() if not _is_request_option(key)}
    body["stream"] = True
    body = {
        key: value
        for key, value in body.items()
        if not isinstance(value, anthropic.NotGiven)
    }
    options = _final_request_options(kwargs, body)
    raw_sdk = cast(_AnthropicRawStreamSDK, sdk)
    request = _build_raw_anthropic_request(raw_sdk, options)
    response = await _send_raw_anthropic_request(raw_sdk, request)
    if response.status_code >= 400:
        await _raise_anthropic_status_error(raw_sdk, response)
    return AsyncStream(
        cast_to=cast(
            "type[RawMessageStreamEvent]", anthropic.types.RawMessageStreamEvent
        ),
        response=response,
        client=sdk,
    )


def _final_request_options(
    kwargs: dict[str, object],
    body: dict[str, object],
) -> FinalRequestOptions:
    return anthropic._models.FinalRequestOptions.construct(  # noqa: SLF001 -- SDK request builder is the only way to preserve auth/default headers while reading stream errors safely.
        method="post",
        url="/v1/messages",
        json_data=body,
        headers=cast(Any, kwargs.get("extra_headers")),
        extra_json=cast(Any, kwargs.get("extra_body")),
        timeout=cast(Any, kwargs.get("timeout", anthropic.NOT_GIVEN)),
    )


def _build_raw_anthropic_request(
    sdk: _AnthropicRawStreamSDK,
    options: FinalRequestOptions,
) -> httpx.Request:
    return sdk._build_request(options)  # noqa: SLF001 -- only the SDK request builder preserves auth/default headers for raw stream error handling.


async def _send_raw_anthropic_request(
    sdk: _AnthropicRawStreamSDK,
    request: httpx.Request,
) -> httpx.Response:
    return await sdk._client.send(request, stream=True)  # noqa: SLF001 -- raw stream status handling must use the SDK-owned authenticated client.


async def _raise_anthropic_status_error(
    sdk: _AnthropicRawStreamSDK,
    response: httpx.Response,
) -> None:
    body_text = (await response.aread()).decode(errors="replace")
    body: object = body_text
    with contextlib.suppress(json.JSONDecodeError):
        body = json.loads(body_text)
    message = body_text or f"Error code: {response.status_code}"
    raise sdk._make_status_error(message, body=body, response=response)  # noqa: SLF001 -- provider must preserve the SDK's status exception taxonomy.


def _is_request_option(key: str) -> bool:
    return key in {"extra_headers", "extra_body", "extra_query", "timeout"}


def _cache_mark(ttl: str) -> dict[str, str]:
    """Build the ``cache_control`` marker for the requested TTL."""
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _add_cache_breakpoint(
    messages: list[anthropic.types.MessageParam],
    ttl: str = "5m",
) -> None:
    """Add cache_control to the last content block of the last message."""
    mark = _cache_mark(ttl)
    last = messages[-1]
    raw_content: object = last.get("content")
    if isinstance(raw_content, str):
        messages[-1] = cast(
            anthropic.types.MessageParam,
            {
                **last,
                "content": [
                    {
                        "type": "text",
                        "text": raw_content,
                        "cache_control": mark,
                    },
                ],
            },
        )
    else:
        blocks: list[object] = list(cast(list[object], raw_content))
        for i in range(len(blocks) - 1, -1, -1):
            btype = (
                cast(dict[str, object], blocks[i]).get("type", "")
                if isinstance(blocks[i], dict)
                else ""
            )
            if btype not in ("thinking", "redacted_thinking"):
                blocks[i] = {
                    **cast(dict[str, object], blocks[i]),
                    "cache_control": mark,
                }
                break
        if blocks:
            messages[-1] = cast(
                anthropic.types.MessageParam,
                {**last, "content": blocks},
            )


def _build_messages(
    request: ModelRequest,
    max_image_dim: int = 8000,
    max_image_bytes: int = 5 * 1024 * 1024,
) -> list[anthropic.types.MessageParam]:
    """Convert history entries to Anthropic message format.

    Tool-call IDs are remapped to ``toolu_N`` so history from other
    providers (OpenAI ``fc_*``, etc.) is accepted by the Anthropic API.

    Anthropic uses alternating user/assistant messages with content
    blocks. Tool results are content blocks inside user messages;
    consecutive tool results batch into one user message.
    """
    ids = IdRemapper("toolu_")
    messages: list[anthropic.types.MessageParam] = []
    pending_tool_results: list[dict[str, object]] = []

    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            blocks = _user_blocks(entry, max_image_dim, max_image_bytes)
            # Coalesce: when a user message lands mid-cohort (preempt
            # with detached stubs), append its blocks into the same
            # role=user wire message that holds the pending tool_results.
            # Emitting a separate role=user message breaks Anthropic's
            # strict alternation and triggers HTTP 400.
            if pending_tool_results:
                pending_tool_results.extend(cast(list[dict[str, object]], blocks))
                _flush_tool_results(messages, pending_tool_results)
            elif blocks:
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "user", "content": blocks},
                    )
                )
        elif isinstance(entry, AssistantMessage):
            _flush_tool_results(messages, pending_tool_results)
            blocks = _assistant_blocks(entry, ids)
            if blocks:
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "assistant", "content": blocks},
                    )
                )
        else:
            # TapeEvent is the closed union {UserMessage, AssistantMessage,
            # ToolResult}; the two branches above consume the first two.
            pending_tool_results.append(
                _tool_result_block(entry, ids, max_image_dim, max_image_bytes)
            )

    _flush_tool_results(messages, pending_tool_results)

    if messages:
        _add_cache_breakpoint(messages, request.cache_ttl)

    return messages


def _user_blocks(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
) -> list[object]:
    """Build Anthropic content blocks from a UserMessage."""
    blocks: list[object] = []
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    for att in entry.attachments:
        block = _attachment_block(att, max_image_dim, max_image_bytes)
        if block is not None:
            blocks.append(block)
    return blocks


def _assistant_blocks(
    entry: AssistantMessage,
    ids: IdRemapper,
) -> list[dict[str, object]]:
    """Build Anthropic content blocks from an AssistantMessage.

    Thinking blocks are emitted verbatim (the wire dict from
    ``block.model_dump()`` stored on the message), with two exceptions:

    1. A ``thinking`` block whose ``signature`` is set but whose ``thinking``
       body is empty has lost its signed payload and cannot re-validate
       server-side (Anthropic answers HTTP 400 ``thinking blocks ... cannot be
       modified``). Such orphans are elided.
    2. Non-native thinking-block types (e.g. ``{"type":"reasoning"}`` from
       OpenAI / Moonshot / MiniMax / OpenAI-subscription) cannot be translated
       and would trip Anthropic's content-block validator after a cross-
       provider session switch. They are dropped silently; the underlying
       reasoning is opaque to other providers and there is no faithful re-
       encoding.

    ``redacted_thinking`` has no client-visible body to lose and always passes
    through. Anthropic rejects assistant messages whose final block is
    thinking, so we append a placeholder text block when no text or tool_use
    follows.
    """
    blocks: list[dict[str, object]] = [
        dict(tb)
        for tb in entry.thinking_blocks
        if _is_native_thinking(tb) and not _is_orphan_thinking(tb)
    ]
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    blocks.extend(_tool_use_block(tc, ids) for tc in entry.tool_calls)
    if blocks and blocks[-1].get("type") in ("thinking", "redacted_thinking"):
        blocks.append({"type": "text", "text": "."})
    return blocks


def _is_orphan_thinking(block: Mapping[str, object]) -> bool:
    """True for signed ``thinking`` blocks whose signed body is gone."""
    return (
        block.get("type") == "thinking"
        and bool(block.get("signature"))
        and not block.get("thinking")
    )


def _is_native_thinking(block: Mapping[str, object]) -> bool:
    """True for Anthropic-native thinking-block types the API accepts."""
    return block.get("type") in ("thinking", "redacted_thinking")


def _tool_use_block(tc: ToolCall, ids: IdRemapper) -> dict[str, object]:
    """Wire-shape a ``ToolCall`` for Anthropic's tool_use content block."""
    return {
        "type": "tool_use",
        "id": ids.map(tc.id),
        "name": tc.name,
        "input": dict(tc.args),
    }


def _tool_result_block(
    entry: ToolResult,
    ids: IdRemapper,
    max_image_dim: int,
    max_image_bytes: int,
) -> dict[str, object]:
    """Build a single tool_result block for a ToolResult.

    Image attachments inline as image blocks alongside the text.
    """
    image_attachments = [
        att for att in entry.attachments if _is_image_mime(att.descriptor)
    ]
    tool_result_content: object
    if image_attachments:
        wire_blocks: list[dict[str, object]] = []
        if entry.content:
            wire_blocks.append({"type": "text", "text": entry.content})
        for att in image_attachments:
            block = _attachment_block(att, max_image_dim, max_image_bytes)
            if block is not None:
                wire_blocks.append(block)
        tool_result_content = wire_blocks
    else:
        tool_result_content = entry.content
    return {
        "type": "tool_result",
        "tool_use_id": ids.map(entry.call_id),
        "content": tool_result_content,
        "is_error": entry.is_error,
    }


def _attachment_block(
    att: object,
    max_image_dim: int,
    max_image_bytes: int,
) -> dict[str, object] | None:
    """Translate a ``BytesMessage`` attachment to an Anthropic block."""
    data = getattr(att, "data", None)
    descriptor = getattr(att, "descriptor", "")
    if not isinstance(data, bytes) or not isinstance(descriptor, str):
        return None
    is_image = _is_image_mime(descriptor)
    is_pdf = descriptor == "application/pdf"
    if not (is_image or is_pdf):
        logger.warning(
            "Anthropic: skipping attachment with unsupported mime=%s",
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
    return {
        "type": "image" if is_image else "document",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": b64,
        },
    }


def _is_image_mime(descriptor: str) -> bool:
    """True for image content types (no descriptor registry needed)."""
    return descriptor.startswith("image/")


def _flush_tool_results(
    messages: list[anthropic.types.MessageParam],
    pending: list[dict[str, object]],
) -> None:
    """Emit buffered tool_result blocks as a user message, then clear."""
    if not pending:
        return
    messages.append(
        cast(
            anthropic.types.MessageParam,
            {"role": "user", "content": list(pending)},
        )
    )
    pending.clear()


def _is_valid_tool_name(name: str) -> bool:
    """Return True when ``name`` is a plausible registered tool name.

    Filters template placeholders the model can echo back from the
    Anthropic server's injected tool-use spec (e.g. ``$FUNCTION_NAME``,
    ``$TOOL_NAME``). Real registered tools always have Python-identifier
    names; ``$FUNCTION_NAME`` is not an identifier so ``isidentifier()``
    alone rejects it.
    """
    return name.isidentifier()


def _parse_response(raw: anthropic.types.Message, pricing: Pricing) -> ModelResponse:
    """Convert Anthropic Message to ModelResponse with AssistantMessage."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[Mapping[str, object]] = []

    for block in raw.content:
        if isinstance(block, anthropic.types.TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, anthropic.types.ToolUseBlock):
            if not _is_valid_tool_name(block.name):
                # Anthropic's API injects its tool-use spec (containing
                # ``$FUNCTION_NAME`` etc.) into the model's hidden
                # context when tools are registered. If the user asks
                # the model to dump its context, the model echoes the
                # spec verbatim and the server re-parses that echo as a
                # real ``tool_use`` block. Without this filter the
                # unmatched call leaks to ``_run_tool_and_post``
                # ("Unknown tool") and the orphan tool_use poisons the
                # next request (Anthropic HTTP 400).
                logger.warning(
                    "anthropic: skipping tool_use with placeholder name %r",
                    block.name,
                )
                continue
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    args=cast(Mapping[str, object], dict(block.input)),
                )
            )
        elif isinstance(
            block,
            anthropic.types.ThinkingBlock | anthropic.types.RedactedThinkingBlock,
        ):
            thinking_blocks.append(block.model_dump())

    message = AssistantMessage(
        text="".join(text_parts),
        thinking_blocks=tuple(thinking_blocks),
        tool_calls=tuple(tool_calls),
    )

    cache_write = getattr(raw.usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(raw.usage, "cache_read_input_tokens", 0) or 0
    # Server-authoritative: a request that opted into fast mode but fell
    # back to standard speed reports ``usage.speed == "standard"`` and is
    # billed at standard rates.
    served_fast = getattr(raw.usage, "speed", None) == "fast"
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        cache_write,
        cache_read,
        fast=served_fast,
    )
    debug_log.trace(
        "api_response",
        kind="anthropic",
        usage_speed=getattr(raw.usage, "speed", None),
        billed_fast=served_fast,
        input_cost=in_cost,
        output_cost=out_cost,
    )
    return ModelResponse(
        message=message,
        tokens=TokenCount(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            raw.stop_reason,
            kind="anthropic",
            has_tool_use=bool(tool_calls),
        ),
        stop_sequence=raw.stop_sequence,
        message_id=raw.id or "",
        request_id=getattr(raw, "_request_id", "") or "",
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )
