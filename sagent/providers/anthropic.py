"""Anthropic provider (API key).

Thin API-key client matching OpenAI/Google shape.

Usage::

    from sagent.providers import Anthropic

    provider = Anthropic.from_key("sk-ant-...")
    # or: Anthropic.from_env()  (reads ANTHROPIC_API_KEY)
    sonnet = provider.model("claude-sonnet-4-6")
    response = await sonnet.buffer(request)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast

import asyncio
import base64
import logging
import re
import time


if TYPE_CHECKING:
    import anthropic

    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    anthropic = lazy_import("anthropic")  # 569ms cold
    image_lib = lazy_import("sagent.lib.image")

from sagent.custom_exceptions import (
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib import apikey, debug_log
from sagent.lib.descriptors import has_error, is_image
from sagent.lib.json import (
    JSONValue,
    MutableJSON,
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


_CONTEXT_TAGS = ("+1m", "+200k")
_CONTEXT_1M_BETA = "context-1m-2025-08-07"


def _strip_context_tag(model_id: str) -> str:
    """Strip a trailing window-size tag so the wire id is canonical."""
    lower = model_id.lower()
    for tag in _CONTEXT_TAGS:
        if lower.endswith(tag):
            return model_id[: -len(tag)]
    return model_id


def context_betas(model_id: str) -> list[str]:
    """Return beta headers required by the requested context window.

    Args:
      model_id: Model identifier, possibly with a context-window suffix.

    Returns:
      betas: Beta header strings to include in the request.

    """
    if model_id.lower().endswith("+1m"):
        return [_CONTEXT_1M_BETA]
    return []


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
    DEFAULT_MODEL = "claude-opus-4-6+1m"
    DEFAULT_UTILITY_MODEL = "claude-haiku-4-5"

    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "claude-opus-4-7": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-opus-4-7+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-opus-4-6": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-opus-4-6+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-opus-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-opus-4-5+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_OPUS,
        ),
        "claude-sonnet-4-6": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
        ),
        "claude-sonnet-4-6+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
        ),
        "claude-sonnet-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
        ),
        "claude-sonnet-4-5+1m": ModelProfile(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            pricing=_SONNET,
        ),
        "claude-haiku-4-5": ModelProfile(
            max_request_tokens=200_000,
            max_response_tokens=64_000,
            pricing=_HAIKU,
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
          api_key: Anthropic API key.

        Returns:
          provider: Anthropic provider instance.

        """
        return cls(api_key=api_key)

    @classmethod
    def from_env(cls) -> Anthropic:
        """Create provider from ANTHROPIC_API_KEY env var.

        Returns:
          provider: Anthropic provider instance.

        Raises:
          RuntimeError: If the API key is not configured.

        """
        key = apikey.get("ANTHROPIC_API_KEY")
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
          model: Anthropic model backend for ``DEFAULT_UTILITY_MODEL``.

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
          client: Async Anthropic SDK client.

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
          model_id: Model identifier for context-window beta selection.

        Returns:
          headers: Extra HTTP headers to include.

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
          has_thinking: Whether thinking mode is active.
          cache_cold: Whether the prompt cache TTL likely expired.

        Returns:
          body: Extra body fields, or ``None``.

        """
        del has_thinking, cache_cold
        return None

    async def handle_auth_error(self) -> None:
        """Handle a 401 from the API. No-op for API-key auth."""
        return


_RE_ANTHROPIC_TOKENS = re.compile(r"(\d[\d,]*)\s*tokens?\s*>\s*(\d[\d,]*)")


def _raise_if_prompt_too_long(e: anthropic.BadRequestError) -> None:
    """Re-raise as PromptTooLongError if this is a prompt-too-long error."""
    msg = str(e).lower()  # anthropic types are annotation-only here
    if "too long" in msg or "too_long" in msg:
        actual, limit = None, None
        m = _RE_ANTHROPIC_TOKENS.search(str(e))
        if m:
            actual = int(m.group(1).replace(",", ""))
            limit = int(m.group(2).replace(",", ""))
        raise PromptTooLongError(
            str(e), actual_tokens=actual, limit_tokens=limit
        ) from e


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
    """Raise ``StreamInterruptedError`` if a tool_use response arrived
    without any ``ToolUseBlock``s.

    Gates on *actual content* rather than the API's ``stop_reason``
    field, which is unreliable (``stop_reason === 'model_tool_use'`` is not
    always set correctly). When this invariant is violated the tool
    block was almost certainly dropped mid-stream; retrying the request
    usually recovers it. The parsed partial response is carried on the
    error so the retry layer can fall back gracefully after exhausting
    attempts.
    """
    parts = cast(tuple[Message, ...], resp.content.content)
    has_tool_calls = any(p.descriptor == "multipart/x-tool-call" for p in parts)
    if resp.stop_reason == "model_tool_use" and not has_tool_calls:
        text_parts = [
            cast(str, p.content) for p in parts if p.descriptor == "text/plain"
        ]
        thinking_count = sum(
            1 for p in parts if p.descriptor == "application/x-thinking-structured"
        )
        text = "".join(text_parts)
        debug_log.trace_error(
            "stream_interrupted",
            kind=kind,
            model=model_id,
            has_text=bool(text.strip()),
            text_len=len(text),
            thinking_blocks=thinking_count,
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


# ==================================================================
# Model backend
# ==================================================================


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
        """Maximum input tokens for this model."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Model identifier string."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Maximum output tokens for this model."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """Whether this model supports streaming responses."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether this model supports thinking mode."""
        return True

    @property
    def supports_effort(self) -> bool:
        """Whether this model supports effort configuration."""
        return True

    @property
    def supports_cache_control(self) -> bool:
        """Whether this model supports prompt caching."""
        return True

    @property
    def supports_context_management(self) -> bool:
        """Whether server-side context management is available."""
        return self._provider.subscription

    @property
    def supports_persistent_retry(self) -> bool:
        """Whether retries with persistent message IDs are supported."""
        return True

    @property
    def supports_account_auth(self) -> bool:
        """Whether this model uses account auth."""
        return self._provider.subscription

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate token count for text using 4 chars/token heuristic.

        Args:
          text: Input text to estimate.

        Returns:
          tokens: Estimated token count.

        """
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        return self._profile.pricing

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate token count for an image using Anthropic's formula.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Estimated token count (width*height/750), or 0 if
              dimensions cannot be read.

        """
        # Anthropic: ~width*height/750 tokens.
        # https://docs.anthropic.com/en/docs/build-with-claude/vision#calculate-image-costs
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // 750 if dims is not None else 0

    @property
    def max_image_dim(self) -> int:
        """Maximum image dimension (width or height) in pixels."""
        return 8000

    @property
    def max_image_bytes(self) -> int:
        """Maximum image size in bytes (5 MiB)."""
        return 5 * 1024 * 1024

    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error indicates a context-window overflow.

        Args:
          error: Exception raised by the API call.

        Returns:
          overflow: ``True`` if the error is a context-window overflow.

        """
        if not isinstance(error, anthropic.BadRequestError):
            return False
        msg = str(error).lower()
        return "too long" in msg or "too_long" in msg or "context" in msg

    def _build_kwargs(
        self,
        request: ModelRequest,
        messages: list[anthropic.types.MessageParam],
    ) -> dict[str, object]:
        """Build kwargs for the Anthropic messages API."""
        has_thinking = request.thinking in ("adaptive", "enabled")
        max_tok = request.max_response_tokens or self.max_response_tokens
        kwargs: dict[str, object] = {
            "model": _strip_context_tag(self._model_id),
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": request.temperature,
            "system": self._provider.build_system(request.system, messages),
        }
        if request.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1.0
        elif request.thinking == "enabled":
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
        body = self._provider.extra_body(
            has_thinking=has_thinking,
            cache_cold=self._cache_cold,
        )
        if body is not None:
            kwargs["extra_body"] = body
        headers = self._provider.extra_headers(self._model_id)
        if headers:
            kwargs["extra_headers"] = headers
        return kwargs

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a buffered request to Claude and return a ModelResponse.

        Args:
          request: Model request to send.

        Returns:
          response: Parsed model response.

        """
        messages = _build_messages(request, self.max_image_dim, self.max_image_bytes)
        sdk = await self._provider.get_sdk()
        kwargs = self._build_kwargs(request, messages)
        logger.debug(
            "API call: model=%s, messages=%d",
            self._model_id,
            len(messages),
        )
        debug_log.trace(
            "api_call",
            kind="buffer",
            model=self._model_id,
            roles=debug_log.role_sequence(messages),
            n_messages=len(messages),
            n_tools=len(kwargs.get("tools") or []),  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- kwargs.tools always list
            thinking=kwargs.get("thinking"),
        )
        try:
            raw = cast(
                anthropic.types.Message,
                await sdk.messages.create(**kwargs),  # pyright: ignore[reportArgumentType, reportCallIssue]  # ty: ignore[no-matching-overload] -- dynamic kwargs
            )
        except anthropic.AuthenticationError:
            await self._provider.handle_auth_error()
            sdk = await self._provider.get_sdk()
            kwargs["system"] = self._provider.build_system(request.system, messages)
            raw = cast(
                anthropic.types.Message,
                await sdk.messages.create(**kwargs),  # pyright: ignore[reportArgumentType, reportCallIssue]  # ty: ignore[no-matching-overload] -- dynamic kwargs
            )
        except anthropic.BadRequestError as e:
            debug_log.trace_error(
                "bad_request",
                kind="buffer",
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
        logger.debug(
            "API response: tokens=%d/%d, stop=%s",
            resp.tokens.input_tokens,
            resp.tokens.output_tokens,
            resp.stop_reason,
        )
        _guard_stream_interrupt(resp, kind="buffer", model_id=self._model_id)
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream a request, calling on_text for each text chunk.

        Args:
          request: Model request to send.
          on_text: Callback invoked with each streamed text delta.

        Returns:
          response: Parsed model response after the stream completes.

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
            raw = await _stream_impl(sdk, kwargs, on_text)
        except anthropic.AuthenticationError:
            await self._provider.handle_auth_error()
            sdk = await self._provider.get_sdk()
            kwargs["system"] = self._provider.build_system(request.system, messages)
            raw = await _stream_impl(sdk, kwargs, on_text)
        except anthropic.BadRequestError as e:
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
) -> anthropic.types.Message:
    """Run the streaming call and return the final message.

    Returns:
      message: Final assembled ``anthropic.types.Message`` after the
          stream has closed.

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
                if (
                    delta is not None
                    and getattr(delta, "type", "") == "text_delta"
                    and on_text is not None
                ):
                    on_text(delta.text)
            return await s.get_final_message()


# -- Translation helpers -----------------------------------------------


_CACHE_MARK: dict[str, str] = {"type": "ephemeral"}


def _add_cache_breakpoint(
    messages: list[anthropic.types.MessageParam],
) -> None:
    """Add cache_control to the last content block of the last message.

    Places exactly one marker per request so the API can cache the
    conversation prefix.
    """
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
                        "cache_control": _CACHE_MARK,
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
                    "cache_control": _CACHE_MARK,
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
    """Convert Message list to Anthropic message format.

    Tool-call IDs are remapped to ``toolu_N`` so history from other
    providers (OpenAI ``fc_*``, etc.) is accepted by the Anthropic API.

    Anthropic uses alternating user/assistant messages with content
    blocks. Tool results are content blocks inside user messages.
    Consecutive tool results are batched into one user message.
    """
    ids = IdRemapper("toolu_")
    messages: list[anthropic.types.MessageParam] = []
    pending_tool_results: list[dict[str, object]] = []

    for msg in request.messages:
        if msg.descriptor == "text/x-user-message":
            if pending_tool_results and msg.content:
                pending_tool_results.append({"type": "text", "text": msg.content})
                continue
            _flush_tool_results(messages, pending_tool_results)
            if msg.content:
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "user", "content": msg.content},
                    )
                )
        elif msg.descriptor == "multipart/x-user-message":
            _flush_tool_results(messages, pending_tool_results)
            blocks: list[object] = []
            for part in cast(tuple[Message, ...], msg.content):
                if part.descriptor == "text/plain":
                    blocks.append({"type": "text", "text": cast(str, part.content)})
                elif is_image(part.descriptor) or (
                    part.descriptor == "application/pdf"
                ):
                    raw = cast(bytes, part.content)
                    img = is_image(part.descriptor)
                    mime = part.descriptor
                    if img:
                        raw, mime = image_lib.resize(
                            raw, max_dim=max_image_dim, max_bytes=max_image_bytes
                        )
                    b64 = base64.b64encode(raw).decode()
                    blocks.append(
                        {
                            "type": "image" if img else "document",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        }
                    )
                else:
                    logger.warning(
                        "Anthropic: skipping user attachment with unsupported mime=%s",
                        part.descriptor,
                    )
            if blocks:
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "user", "content": blocks},
                    )
                )
        elif msg.descriptor == "multipart/x-model-message":
            _flush_tool_results(messages, pending_tool_results)
            parts_mm = cast(tuple[Message, ...], msg.content)
            if not parts_mm:
                continue
            content: list[dict[str, object]] = []
            for part in parts_mm:
                if part.descriptor == "application/x-thinking-structured":
                    content.append(
                        cast(
                            dict[str, object],
                            json_unfreeze(cast(JSONValue, part.content)),
                        )
                    )
                elif part.descriptor == "text/plain":
                    content.append({"type": "text", "text": cast(str, part.content)})
                elif part.descriptor == "multipart/x-tool-call":
                    directive = get_directive(part)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": ids.map(get_queue_id(part)),
                            "name": get_tool_name(part),
                            "input": json_unfreeze(directive),
                        }
                    )
            # Anthropic rejects assistant messages whose final block is
            # thinking/redacted_thinking. Append a placeholder text block
            # when no text or tool_use follows the thinking blocks.
            if content and content[-1].get("type") in ("thinking", "redacted_thinking"):
                content.append({"type": "text", "text": "."})
            if content:
                messages.append(
                    cast(
                        anthropic.types.MessageParam,
                        {"role": "assistant", "content": content},
                    )
                )
        elif msg.descriptor == "multipart/x-tool-result":
            # tool_result content is a string when there are no image
            # parts, or a list of text/image blocks when there are.
            # Anthropic supports image blocks inside tool_result natively;
            # PDFs are rasterized upstream (in Read) so we only handle
            # image/* here.
            parts_tr = cast(tuple[Message, ...], msg.content)
            is_error = has_error(msg)
            tool_result_content: object
            text_parts_tr = [
                str(p.content)
                for p in parts_tr
                if p.descriptor in ("text/plain", "text/x-error")
            ]
            image_parts_tr = [p for p in parts_tr if is_image(p.descriptor)]
            if image_parts_tr:
                blocks_tr: list[dict[str, object]] = []
                if text_parts_tr:
                    blocks_tr.append({"type": "text", "text": "\n".join(text_parts_tr)})
                for part in image_parts_tr:
                    raw, mime_type = image_lib.resize(
                        cast(bytes, part.content),
                        max_dim=max_image_dim,
                        max_bytes=max_image_bytes,
                    )
                    b64 = base64.b64encode(raw).decode()
                    blocks_tr.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        }
                    )
                tool_result_content = blocks_tr
            else:
                tool_result_content = "\n".join(text_parts_tr)
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": ids.map(get_queue_id(msg)),
                    "content": tool_result_content,
                    "is_error": is_error,
                }
            )

    _flush_tool_results(messages, pending_tool_results)

    if messages:
        _add_cache_breakpoint(messages)

    return messages


def _flush_tool_results(
    messages: list[anthropic.types.MessageParam],
    pending: list[dict[str, object]],
) -> None:
    """Emit buffered tool_result blocks as a user message, then clear.

    Consecutive tool results must batch into one user message to keep
    the API's role-alternation invariant happy.
    """
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
    """Convert Anthropic Message to our ModelResponse."""
    parts: list[Message] = []

    for block in raw.content:
        if isinstance(block, anthropic.types.TextBlock):
            parts.append(TextMessage(block.text, "text/plain"))
        elif isinstance(block, anthropic.types.ToolUseBlock):
            parts.append(
                tool_call_message(
                    block.id,
                    block.name,
                    json_freeze(cast(MutableJSON, dict(block.input))),
                )
            )
        elif isinstance(
            block,
            anthropic.types.ThinkingBlock | anthropic.types.RedactedThinkingBlock,
        ):
            parts.append(
                JsonMessage(
                    json_freeze(block.model_dump()),
                    "application/x-thinking-structured",
                )
            )

    msg_id = raw.id or ""
    all_parts: list[Message] = []
    if msg_id:
        all_parts.append(TextMessage(msg_id, "text/x-queue-id"))
    all_parts.extend(parts)
    content = MultipartMessage(
        tuple(all_parts),
        "multipart/x-model-message",
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
    has_tool_use = any(p.descriptor == "multipart/x-tool-call" for p in parts)
    return ModelResponse(
        content=content,
        tokens=TokenCount(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            raw.stop_reason,
            kind="anthropic",
            has_tool_use=has_tool_use,
        ),
        stop_sequence=raw.stop_sequence,
        message_id=raw.id or "",
        request_id=getattr(raw, "_request_id", "") or "",
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )
