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

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar, cast

import asyncio
import base64
import logging
import os
import re
import time


if TYPE_CHECKING:
    import anthropic

    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    anthropic = lazy_import("anthropic")  # 569ms cold
    image_lib = lazy_import("sagent.lib.image")

from sagent.lib import debug_log, token_count
from sagent.lib.json import MutableJSON, json_unfreeze
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.types.exceptions import (
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.history import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ModelRequest, ModelResponse, TokenCount


logger = logging.getLogger(__name__)

_STREAM_IDLE_TIMEOUT = 600.0

_CONTEXT_TAGS = ("+1m", "+200k")
_CONTEXT_1M_BETA = "context-1m-2025-08-07"
_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"

# Models that support server-side ``clear_tool_uses_20250919``. Per
# Anthropic docs: Sonnet 4/4.5, Haiku 4.5, Opus 4/4.1/4.5. We treat
# the +1m variants identically (same base model).
_CONTEXT_MANAGEMENT_MODELS = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    }
)


def _strip_context_tag(model_id: str) -> str:
    """Strip a trailing window-size tag so the wire id is canonical."""
    lower = model_id.lower()
    for tag in _CONTEXT_TAGS:
        if lower.endswith(tag):
            return model_id[: -len(tag)]
    return model_id


def supports_native_context_management(model_id: str) -> bool:
    """True when the model accepts the ``clear_tool_uses_20250919`` beta."""
    return _strip_context_tag(model_id) in _CONTEXT_MANAGEMENT_MODELS


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


# Model limits and pricing.
# Limits & pricing: https://docs.anthropic.com/en/docs/about-claude/models
# Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
# Model list: anthropic.Anthropic().models.list()
#
# To add a new model: check the Anthropic docs for context window
# and max output tokens, then add a ModelProfile + pricing entry.
_OPUS = Pricing(
    request=5.0,
    response=25.0,
    cache_write=6.25,
    cache_read=0.5,
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
    DEFAULT_MODEL = "claude-opus-4-7+1m"
    DEFAULT_UTILITY_MODEL = "claude-haiku-4-5"

    # ``chars_per_token`` measured via ``messages.count_tokens`` on a 2.6M-char
    # mixed code+JSON+thinking session (de89f75430bf). Three tokenizer
    # generations cluster: opus-4-7 (2.83), opus/sonnet-4.6-4.5 (3.66),
    # sonnet-4.5 / haiku-4.5 (4.83). Pure-English content tokenizes higher;
    # these defaults err toward overcount for mixed agent traffic, which is
    # the safe direction for the compaction trigger.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "claude-opus-4-7": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=2.83,
        ),
        "claude-opus-4-7+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=2.83,
        ),
        "claude-opus-4-6": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
            chars_per_token=3.66,
        ),
        "claude-opus-4-6+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
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

    def __init__(self, *, api_key: str) -> None:
        self._api_key = api_key
        self._lock = asyncio.Lock()
        self._sdk: anthropic.AsyncAnthropic | None = None

    @classmethod
    def from_key(cls, api_key: str) -> Anthropic:
        """Create provider from an API key.

        Args:
          api_key: Anthropic API key (``sk-ant-...``).

        Returns:
          provider: Configured Anthropic provider instance.

        """
        return cls(api_key=api_key)

    @classmethod
    def from_env(cls) -> Anthropic:
        """Create provider from ``ANTHROPIC_API_KEY`` env var.

        Returns:
          provider: Configured Anthropic provider instance.

        Raises:
          RuntimeError: If the API key is not configured.

        """
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("Anthropic API key not configured.")
        return cls(api_key=key)

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
            _strip_context_tag(mid),
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
        if self._sdk is not None:
            return self._sdk
        async with self._lock:
            if self._sdk is None:
                self._sdk = anthropic.AsyncAnthropic(api_key=self._api_key)
            return self._sdk

    def build_system(
        self,
        system: str | None,
        messages: list[anthropic.types.MessageParam] | None = None,
    ) -> list[anthropic.types.TextBlockParam] | str | anthropic.NotGiven:
        """Build the system prompt for an API call.

        Args:
          system: System prompt text, or ``None`` to omit.
          messages: Conversation messages (used by subscription subclass).

        Returns:
          system_param: System prompt in API format, or ``NOT_GIVEN``.

        """
        del messages  # unused in plain API mode
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
        return {"anthropic-beta": ",".join(betas)} if betas else {}

    def extra_body(
        self,
        *,
        has_thinking: bool,
        cache_cold: bool,
    ) -> MutableJSON | None:
        """Return per-request extra body fields.

        Args:
          has_thinking: True when the request enables extended thinking.
          cache_cold: True when the cache TTL likely expired.

        Returns:
          body: Extra body payload, or ``None`` to skip.

        """
        del has_thinking, cache_cold
        return None

    async def handle_auth_error(self) -> None:
        """Handle a 401 from the API. No-op for API-key auth."""
        return


_RE_ANTHROPIC_TOKENS = re.compile(r"(\d[\d,]*)\s*tokens?\s*>\s*(\d[\d,]*)")

# Statusless Anthropic errors with body-declared types in this set are
# transient retryables that the shared status-code classifier misses.
_RETRYABLE_BODY_TYPES = frozenset(
    {"api_error", "overloaded_error", "rate_limit_error", "server_error"}
)


def _is_prompt_too_long_text(msg: str) -> bool:
    """True if the error body text describes a context-window overflow."""
    lower = msg.lower()
    return "too long" in lower or "too_long" in lower or "context window" in lower


def _raise_if_prompt_too_long(e: anthropic.APIStatusError) -> None:
    """Re-raise as PromptTooLongError if this is a prompt-too-long error."""
    raw = str(e)
    if not _is_prompt_too_long_text(raw):
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
    def supports_context_management(self) -> bool:
        """Whether the provider manages context overflow internally."""
        return self._provider.subscription

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
            kwargs["system"] = self._provider.build_system(request.system, messages)
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
        return _is_prompt_too_long_text(str(error))

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
            "model": _strip_context_tag(self._model_id),
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": request.temperature,
            "system": self._provider.build_system(request.system, messages),
        }
        if thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1.0
        elif thinking == "enabled":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": max_tok,
            }
            kwargs["max_tokens"] = max_tok * 2
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
        )
        if supports_native_context_management(self._model_id):
            # ``clear_tool_uses_20250919``: server-side clearing of old
            # tool results when the prompt exceeds ``trigger``. ``keep``
            # preserves the most recent N tool uses (matches our
            # ``microcompact_keep_recent`` default). ``clear_at_least``
            # prevents cache-busting for trivially small clears.
            cm_config = {
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": 100_000},
                        "keep": {"type": "tool_uses", "value": 5},
                        "clear_at_least": {"type": "input_tokens", "value": 1_000},
                    }
                ],
            }
            body = {**(body or {}), "context_management": cm_config}
        if body is not None:
            kwargs["extra_body"] = body
        headers = self._provider.extra_headers(self._model_id)
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
        debug_log.trace(
            "api_call",
            kind="stream",
            model=self._model_id,
            roles=debug_log.role_sequence(messages),
            n_messages=len(messages),
            n_tools=len(kwargs.get("tools") or []),  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- kwargs.tools always list
            thinking=kwargs.get("thinking"),
        )
        try:
            raw = await _stream_impl(sdk, kwargs, on_text, on_thinking)
        except anthropic.AuthenticationError:
            await self._provider.handle_auth_error()
            sdk = await self._provider.get_sdk()
            kwargs["system"] = self._provider.build_system(request.system, messages)
            raw = await _stream_impl(sdk, kwargs, on_text, on_thinking)
        except anthropic.APIStatusError as e:
            if not _is_prompt_too_long_text(str(e)):
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
    async with sdk.messages.stream(**kwargs) as s:  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- dynamic kwargs
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
        if isinstance(entry, UserMessage):
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
            # HistoryEntry is the closed union {UserMessage, AssistantMessage,
            # ToolResult}; the two branches above consume the first two.
            pending_tool_results.append(
                _tool_result_block(entry, ids, max_image_dim, max_image_bytes)
            )

    _flush_tool_results(messages, pending_tool_results)

    if messages:
        _add_cache_breakpoint(messages, request.cache_ttl)

    return messages


def _user_blocks(
    entry: UserMessage,
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
    ``block.model_dump()`` stored on the message). Anthropic rejects
    assistant messages whose final block is thinking, so we append a
    placeholder text block when no text or tool_use follows.
    """
    blocks: list[dict[str, object]] = [dict(tb) for tb in entry.thinking_blocks]
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    blocks.extend(_tool_use_block(tc, ids) for tc in entry.tool_calls)
    if blocks and blocks[-1].get("type") in ("thinking", "redacted_thinking"):
        blocks.append({"type": "text", "text": "."})
    return blocks


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


def _parse_response(raw: anthropic.types.Message, pricing: Pricing) -> ModelResponse:
    """Convert Anthropic Message to ModelResponse with AssistantMessage."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[Mapping[str, object]] = []

    for block in raw.content:
        if isinstance(block, anthropic.types.TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, anthropic.types.ToolUseBlock):
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
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        cache_write,
        cache_read,
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
