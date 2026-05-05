"""SelfHosted provider - drives HuggingFace causal LMs from sagent.

Loads a HuggingFace ``AutoModelForCausalLM`` and ``AutoTokenizer`` from either
a model name or a local cache path, then exposes the ``Model`` interface
consumed by :class:`sagent.agent.Agent`.

Usage::

    from sagent.providers import SelfHosted

    provider = SelfHosted.from_hf("Qwen/Qwen3.6-27B")
    model = provider.model()              # same snapshot
    response = await model.buffer(request)

Tool-calling:
  - Qwen3 chat template renders tool declarations and asks the model
    to emit ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.
  - DeepSeek-V3 / Kimi-K2 emit JSON inside ``<｜tool▁calls▁begin｜>``
    blocks. Both formats are parsed here.

Models can be downloaded, for example, via,

hf download Qwen/Qwen3.6-27B --local-dir /opt/models/qwen3.6-27b

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, runtime_checkable

import asyncio
import importlib
import json
import logging
import os
import re
import time
import uuid

from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.json import (
    MutableJSON,
    MutableJSONValue,
    json_freeze,
    json_unfreeze,
)
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    get_tool_name,
    tool_call_message,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.stop_reason import normalize_stop_reason


if TYPE_CHECKING:
    from torch import Tensor, nn

    import torch
    import transformers as transformers_lib

    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    # ``transformers`` and ``torch`` add several seconds to importing
    # ``sagent.providers``. The CLI dispatches providers by
    # name, so hosted-provider users must not pay that cost just because
    # ``SelfHosted`` lives in the same package.
    transformers_lib = lazy_import("transformers")
    torch = lazy_import("torch")
    image_lib = lazy_import("sagent.lib.image")


logger = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    eos_token_id: int

    def apply_chat_template(self, messages: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str: ...


class _GenerateModel(Protocol):
    def generate(self, input_ids: Tensor, **kwargs: object) -> Tensor: ...


@runtime_checkable
class _HasInputIds(Protocol):
    input_ids: object


@runtime_checkable
class _HasData(Protocol):
    data: object


@dataclass(frozen=True, kw_only=True, slots=True)
class _RenderedPrompt:
    input_ids: Tensor
    attention_mask: Tensor | None


def _parse_dtype(name: str | None) -> torch.dtype | None:
    """Parse a torch dtype name from environment configuration."""
    if not name:
        return None
    normalized = name.removeprefix("torch.")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtypes.get(normalized)
    if dtype is None:
        valid = ", ".join(sorted(dtypes))
        raise ValueError(
            f"Unsupported {SelfHosted.DTYPE_ENV_VAR}={name!r}; use {valid}."
        )
    return dtype


def _parse_bool_env(name: str, value: str | None) -> bool:
    """Parse a boolean environment flag."""
    if value is None or value == "":
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported {name}={value!r}; expected one of "
        "1/0, true/false, yes/no, on/off."
    )


def _context_window(config: MutableJSON, default: int) -> int:
    """Extract the configured context window from an HF config dict."""
    max_pos = config.get("max_position_embeddings")
    if isinstance(max_pos, int):
        return max_pos
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_max_pos = cast(MutableJSON, text_config).get("max_position_embeddings")
        if isinstance(text_max_pos, int):
            return text_max_pos
    return default


def _default_device() -> str | None:
    """Choose the fastest local torch device available by default."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


class SelfHosted:
    """Self-hosted model provider backed by HuggingFace transformers."""

    DEFAULT_MODEL: ClassVar[str] = "Qwen/Qwen3.6-27B"
    ENV_VAR: ClassVar[str] = "SAGENT_SELFHOSTED_MODEL"
    DEVICE_ENV_VAR: ClassVar[str] = "SAGENT_SELFHOSTED_DEVICE"
    DTYPE_ENV_VAR: ClassVar[str] = "SAGENT_SELFHOSTED_DTYPE"
    COMPILE_ENV_VAR: ClassVar[str] = "SAGENT_SELFHOSTED_COMPILE"
    DEFAULT_MAX_REQUEST_TOKENS: ClassVar[int] = 32_768
    DEFAULT_MAX_RESPONSE_TOKENS: ClassVar[int] = 4_096

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: _Tokenizer,
        model_id: str,
        max_request_tokens: int,
        max_response_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._max_request_tokens = max_request_tokens
        self._max_response_tokens = (
            max_response_tokens or self.DEFAULT_MAX_RESPONSE_TOKENS
        )

    @classmethod
    def from_key(
        cls,
        path_or_repo: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        compile_model: bool | None = None,
    ) -> SelfHosted:
        """Build a provider from a HuggingFace repo ID or local snapshot path.

        Args:
          path_or_repo: HuggingFace repo ID or local HF snapshot path.
          device: Target device for model tensors.
          dtype: Data type for model parameters.
          compile_model: Whether to wrap the model with ``torch.compile``.

        Returns:
          provider: Configured self-hosted provider.

        """
        if device is None:
            device = os.environ.get(cls.DEVICE_ENV_VAR) or _default_device()
        if dtype is None:
            dtype = _parse_dtype(os.environ.get(cls.DTYPE_ENV_VAR))
        if compile_model is None:
            compile_model = _parse_bool_env(
                cls.COMPILE_ENV_VAR,
                os.environ.get(cls.COMPILE_ENV_VAR),
            )
        path = str(Path(path_or_repo).expanduser())
        return cls.from_hf(
            path,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
        )

    @classmethod
    def from_env(cls) -> SelfHosted:
        """Build a provider from environment configuration.

        ``SAGENT_SELFHOSTED_MODEL`` may be a HuggingFace repo ID or local
        snapshot path. If unset, Qwen 3.6 is used. Optional
        ``SAGENT_SELFHOSTED_DEVICE``, ``SAGENT_SELFHOSTED_DTYPE``, and
        ``SAGENT_SELFHOSTED_COMPILE`` values are forwarded to ``from_hf``.
        Supported dtype values are ``float16``, ``bfloat16``, and ``float32``.

        Returns:
          provider: Configured self-hosted provider.

        Raises:
          ValueError: If ``SAGENT_SELFHOSTED_DTYPE`` is unsupported.

        """
        path = os.environ.get(cls.ENV_VAR) or cls.DEFAULT_MODEL
        device = os.environ.get(cls.DEVICE_ENV_VAR) or _default_device()
        dtype = _parse_dtype(os.environ.get(cls.DTYPE_ENV_VAR))
        compile_model = _parse_bool_env(
            cls.COMPILE_ENV_VAR,
            os.environ.get(cls.COMPILE_ENV_VAR),
        )
        return cls.from_hf(
            path,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
        )

    @property
    def native_model(self) -> nn.Module:
        """Return the underlying model."""
        return self._model

    @property
    def tokenizer(self) -> _Tokenizer:
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def hosted_model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @property
    def hosted_max_request_tokens(self) -> int:
        """Return the maximum request tokens."""
        return self._max_request_tokens

    @property
    def hosted_max_response_tokens(self) -> int:
        """Return the maximum response tokens."""
        return self._max_response_tokens

    @classmethod
    def from_hf(
        cls,
        path_or_repo: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        compile_model: bool = False,
    ) -> SelfHosted:
        """Build a SelfHosted provider from a HuggingFace repo ID or path.

        Loads the model through ``transformers.AutoModelForCausalLM`` and the
        tokenizer through ``transformers.AutoTokenizer``. ``path_or_repo`` may
        be a HuggingFace model ID or a local snapshot/cache path.

        Args:
          path_or_repo: HuggingFace repo ID or local HF snapshot path.
          device: Target device for model tensors.
          dtype: Data type for model parameters.
          trust_remote_code: Whether to allow model repository Python code.
          compile_model: Whether to wrap the model with ``torch.compile``.

        Returns:
          provider: Configured SelfHosted instance.

        """
        model_id = str(path_or_repo)
        start = time.perf_counter()
        logger.info(
            "Loading SelfHosted model model_id=%s device=%s dtype=%s "
            "trust_remote_code=%s",
            model_id,
            device,
            dtype,
            trust_remote_code,
        )
        if os.environ.get("SAGENT_SELFHOSTED_DISABLE_DEEPGEMM") == "1":
            _disable_transformers_deepgemm()
            logger.info("DeepGEMM disabled via env; using Triton FP8 fallback")
        load_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        config = cast(
            MutableJSON,
            transformers_lib.AutoConfig.from_pretrained(  # pyright: ignore[reportUnknownMemberType] -- no stubs  # ty: ignore[possibly-missing-attribute] -- lazy import; ty can't resolve
                model_id,
                trust_remote_code=trust_remote_code,
            ).to_dict(),
        )
        max_request_tokens = _context_window(config, cls.DEFAULT_MAX_REQUEST_TOKENS)
        logger.debug(
            "Loaded SelfHosted config model_id=%s max_request_tokens=%d",
            model_id,
            max_request_tokens,
        )
        # device="auto" delegates placement to accelerate's device_map,
        # sharding weights across all visible GPUs. Required for models
        # that exceed a single GPU's VRAM.
        shard_auto = isinstance(device, str) and device == "auto"
        if shard_auto:
            load_kwargs["device_map"] = "auto"
        model = cast(
            "nn.Module",
            transformers_lib.AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType] -- no stubs  # ty: ignore[possibly-missing-attribute] -- lazy import; ty can't resolve
                model_id,
                **load_kwargs,
            ),
        )
        if device is not None and not shard_auto:
            model = model.to(device)
        model.eval()
        if compile_model:
            compile_start = time.perf_counter()
            logger.info(
                "Compiling SelfHosted model model_id=%s device=%s",
                model_id,
                _module_device(model),
            )
            model = _compile_model(model)
            logger.info(
                "Compiled SelfHosted model model_id=%s elapsed_sec=%.2f",
                model_id,
                time.perf_counter() - compile_start,
            )
        tokenizer = cast(
            _Tokenizer,
            transformers_lib.AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType] -- no stubs  # ty: ignore[possibly-missing-attribute] -- lazy import; ty can't resolve
                model_id,
                trust_remote_code=trust_remote_code,
            ),
        )
        logger.info(
            "Loaded SelfHosted model model_id=%s device=%s max_request_tokens=%d "
            "elapsed_sec=%.2f",
            model_id,
            _module_device(model),
            max_request_tokens,
            time.perf_counter() - start,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            max_request_tokens=max_request_tokens,
        )

    def model(
        self, model_id: str | None = None, /, max_request_tokens: int | None = None
    ) -> SelfHostedModel:
        """Return the bound model.

        ``model_id`` must match the loaded model or be ``None``.
        ``max_request_tokens`` is accepted for Protocol conformance but
        ignored; the context window is fixed at load time.

        Args:
          model_id: Must match the loaded model or be ``None``.
          max_request_tokens: Ignored; accepted for Protocol conformance.

        Returns:
          model: Bound model instance.

        Raises:
          ValueError: If ``model_id`` doesn't match the loaded model.

        """
        del max_request_tokens
        if model_id is not None and model_id != self._model_id:
            raise ValueError(
                f"SelfHosted provider is bound to {self._model_id!r}, "
                f"got {model_id!r}. Rebuild via SelfHosted.from_hf().",
            )
        return SelfHostedModel(provider=self)

    def utility_model(self) -> SelfHostedModel:
        """Return the default model instance.

        Returns:
          model: Bound model using the loaded model ID.

        """
        return self.model()


class _ProviderLike(Protocol):
    @property
    def hosted_max_request_tokens(self) -> int: ...
    @property
    def hosted_model_id(self) -> str: ...
    @property
    def hosted_max_response_tokens(self) -> int: ...
    @property
    def native_model(self) -> nn.Module: ...
    @property
    def tokenizer(self) -> _Tokenizer: ...


class SelfHostedModel:
    """``Model`` backend for a self-hosted HuggingFace model."""

    def __init__(self, *, provider: _ProviderLike) -> None:
        self._provider = provider

    @property
    def max_request_tokens(self) -> int:
        """Return the maximum request tokens."""
        return self._provider.hosted_max_request_tokens

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._provider.hosted_model_id

    @property
    def max_response_tokens(self) -> int:
        """Return the maximum response tokens."""
        return self._provider.hosted_max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """Return whether streaming is supported."""
        # Naive streaming would tokenize then emit one token at a time.
        # Not implemented for the first cut; ``stream`` falls back to
        # buffering and calling ``on_text`` once.
        return False

    @property
    def supports_thinking(self) -> bool:
        """Return whether thinking mode is supported."""
        return False

    @property
    def supports_effort(self) -> bool:
        """Return whether effort control is supported."""
        return True

    @property
    def supports_cache_control(self) -> bool:
        """Return whether cache control is supported."""
        return False

    @property
    def supports_context_management(self) -> bool:
        """Return whether context management is supported."""
        return False

    @property
    def supports_persistent_retry(self) -> bool:
        """Return whether persistent retry is supported."""
        return False

    @property
    def supports_account_auth(self) -> bool:
        """Return whether account auth is supported."""
        return False

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate the token count for a text string.

        Args:
          text: Input text.

        Returns:
          count: Estimated number of tokens.

        """
        return len(text) // 4

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate the token count for an image.

        Args:
          data: Raw image bytes.

        Returns:
          count: Estimated number of tokens.

        """
        # Based on Qwen3.6-27B vision encoder (docs/qwen3_27b.md):
        # patch_size=16, spatial_merge_size=2 → 32x32 pixels per token.
        # Other self-hosted models will differ; override as needed.
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // (32 * 32) if dims is not None else 0

    @property
    def max_image_dim(self) -> int:
        """Return the maximum image dimension in pixels."""
        return 2000

    @property
    def max_image_bytes(self) -> int:
        """Return the maximum image size in bytes."""
        return 5 * 1024 * 1024

    def is_context_overflow(self, error: Exception) -> bool:
        """Return whether the error indicates a context overflow.

        Args:
          error: Exception from the model.

        Returns:
          is_overflow: Always ``False`` for self-hosted models.

        """
        del error
        return False

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Render messages via chat template, generate, and parse tool calls.

        Args:
          request: Model request with messages and tool declarations.

        Returns:
          response: Model response with text and/or tool-call messages.

        """
        rendered = self._render_prompt(request)
        model = self._provider.native_model
        device = next(iter(model.parameters())).device
        input_ids = rendered.input_ids.to(device)
        attention_mask = (
            rendered.attention_mask.to(device)
            if rendered.attention_mask is not None
            else torch.ones_like(input_ids)
        )
        max_new = min(
            request.max_response_tokens or self.max_response_tokens,
            max(1, self._provider.hosted_max_request_tokens - input_ids.shape[-1]),
        )
        generate_kwargs: dict[str, object] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "eos_token_id": self._provider.tokenizer.eos_token_id,
            "pad_token_id": self._provider.tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = request.temperature
        start = time.perf_counter()
        logger.debug(
            "SelfHosted generate start model_id=%s device=%s input_tokens=%d "
            "max_new_tokens=%d tools=%d attention_mask=%s",
            self.model_id,
            device,
            int(input_ids.shape[-1]),
            max_new,
            len(request.tools or []),
            rendered.attention_mask is not None,
        )
        out = await asyncio.to_thread(_generate, model, input_ids, generate_kwargs)
        elapsed_sec = time.perf_counter() - start
        new_tokens = out[0, input_ids.shape[-1] :]
        output_tokens = int(new_tokens.shape[-1])
        text = self._provider.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )
        tool_msgs, cleaned_text = _extract_tool_calls(
            text,
            allowed_tools={t.name for t in request.tools or []},
        )
        # HF ``generate`` doesn't return a finish_reason; infer from
        # whether we exhausted the budget.
        hit_cap = output_tokens >= max_new
        finish_reason = "length" if hit_cap else "stop"
        logger.debug(
            "SelfHosted generate end model_id=%s output_tokens=%d "
            "finish_reason=%s tool_calls=%d elapsed_sec=%.2f "
            "output_tokens_per_sec=%.2f",
            self.model_id,
            output_tokens,
            finish_reason,
            len(tool_msgs),
            elapsed_sec,
            output_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
        )
        msg_parts: list[Message] = []
        if cleaned_text.strip():
            msg_parts.append(TextMessage(cleaned_text, "text/plain"))
        msg_parts.extend(tool_msgs)
        has_tool_use = bool(tool_msgs)
        return ModelResponse(
            content=MultipartMessage(
                tuple(msg_parts),
                "multipart/x-model-message",
            ),
            tokens=TokenCount(
                input_tokens=int(input_ids.shape[-1]),
                output_tokens=output_tokens,
            ),
            stop_reason=normalize_stop_reason(
                finish_reason,
                kind="openai",  # use OpenAI vocab - same stop/length semantics
                has_tool_use=has_tool_use,
            ),
        )

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Buffer the response then emit text in one shot.

        Args:
          request: Model request with messages and tool declarations.
          on_text: Callback invoked with each text part of the response.

        Returns:
          response: Model response with text and/or tool-call messages.

        """
        resp = await self.buffer(request)
        if on_text is not None:
            for part in cast(tuple[Message, ...], resp.content.content):
                if part.descriptor == "text/plain":
                    on_text(cast(str, part.content))
        return resp

    def _render(self, request: ModelRequest) -> Tensor:
        """Render a ``ModelRequest`` into token ids via the chat template.

        Tool declarations are passed to ``apply_chat_template`` when the
        tokenizer supports the ``tools=`` kwarg; otherwise they're
        rendered as a system-preamble JSON block so the model at least
        sees the schema.
        """
        return self._render_prompt(request).input_ids

    def _render_prompt(self, request: ModelRequest) -> _RenderedPrompt:
        """Render a ``ModelRequest`` and preserve tokenizer masks."""
        tokenizer = self._provider.tokenizer
        messages = _build_chat_messages(request)
        kwargs: MutableJSON = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if request.effort is not None:
            kwargs["enable_thinking"] = request.effort != "none"
        if request.tools:
            kwargs["tools"] = [_tool_schema(t) for t in request.tools]
        rendered = _apply_chat_template(
            tokenizer, messages, kwargs, request.tools or []
        )
        ids_tensor = cast("Tensor", _input_ids(rendered))
        if ids_tensor.ndim == 1:
            ids_tensor = ids_tensor.unsqueeze(0)
        attention_mask = cast("Tensor | None", _attention_mask(rendered))
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return _RenderedPrompt(input_ids=ids_tensor, attention_mask=attention_mask)


def _apply_chat_template(
    tokenizer: _Tokenizer,
    messages: list[MutableJSON],
    kwargs: MutableJSON,
    tools: list[Tool],
) -> object:
    """Apply a chat template with graceful HF feature fallback."""
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError) as e:
        last_error = e

    if "enable_thinking" in kwargs:
        logger.debug(
            "SelfHosted tokenizer rejected enable_thinking; retrying without it."
        )
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("enable_thinking")
        try:
            return tokenizer.apply_chat_template(messages, **retry_kwargs)
        except (TypeError, ValueError) as e:
            last_error = e
            kwargs = cast(MutableJSON, retry_kwargs)

    if "tools" in kwargs:
        logger.debug(
            "SelfHosted tokenizer rejected native tools; retrying with inlined schema."
        )
        _inline_tool_preamble(messages, tools)
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("tools")
        try:
            return tokenizer.apply_chat_template(messages, **retry_kwargs)
        except (TypeError, ValueError) as e:
            last_error = e
            if "enable_thinking" not in retry_kwargs:
                raise
            retry_kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **retry_kwargs)
    raise last_error


def _generate(
    model: nn.Module,
    input_ids: Tensor,
    generate_kwargs: dict[str, object],
) -> Tensor:
    """Run blocking HF generation under inference mode."""
    with torch.inference_mode():
        return cast(_GenerateModel, model).generate(input_ids, **generate_kwargs)


def _compile_model(model: nn.Module) -> nn.Module:
    """Wrap a model with torch.compile."""
    return cast("nn.Module", cast(Callable[[object], object], torch.compile)(model))


def _disable_transformers_deepgemm() -> None:
    """Force-route transformers' FP8 dispatch through the Triton fallback.

    DeepGEMM's FP8 kernel ships with layout recipes only for sm_90
    (Hopper) and sm_100 (Blackwell DC); sm_120 (consumer Blackwell, e.g.
    RTX 5090) trips an "Unknown recipe" C++ assertion at runtime.
    Transformers' dispatch in ``w8a8_fp8_matmul`` only catches
    ``ImportError``, so the assertion is not recoverable from Python.
    Setting the module's private ``_deepgemm_available`` flag to False
    causes ``_load_deepgemm_kernel`` to raise ``ImportError`` on its
    next call, which the dispatch catches and falls back to the
    universal Triton FP8 kernel (slower but functional).
    """
    finegrained_fp8 = importlib.import_module(
        "transformers.integrations.finegrained_fp8"
    )
    # Monkey-patch the module-level flag; no public knob exists upstream.
    setattr(finegrained_fp8, "_deepgemm_available", False)  # noqa: B010


def _module_device(model: nn.Module) -> str:
    """Return the first parameter device for diagnostics."""
    param = next(iter(model.parameters()), None)
    return str(param.device) if param is not None else "unknown"


def _inline_tool_preamble(messages: list[MutableJSON], tools: list[Tool]) -> None:
    """Inline tool schemas for templates without native tool support."""
    if not tools:
        return
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = (
            str(messages[0]["content"]) + "\n\n" + _tool_preamble(tools)
        )
    else:
        messages.insert(
            0,
            {"role": "system", "content": _tool_preamble(tools)},
        )


def _input_ids(rendered: object) -> object:
    """Extract token ids from chat-template return values."""
    if isinstance(rendered, Mapping):
        return cast(Mapping[str, object], rendered)["input_ids"]
    if isinstance(rendered, _HasInputIds):
        return rendered.input_ids
    if isinstance(rendered, _HasData) and isinstance(rendered.data, Mapping):
        return cast(Mapping[str, object], rendered.data)["input_ids"]
    return rendered


def _attention_mask(rendered: object) -> object | None:
    """Extract an attention mask from chat-template return values."""
    if isinstance(rendered, Mapping):
        return cast(Mapping[str, object], rendered).get("attention_mask")
    if isinstance(rendered, _HasData) and isinstance(rendered.data, Mapping):
        return cast(Mapping[str, object], rendered.data).get("attention_mask")
    return getattr(rendered, "attention_mask", None)


# -- chat-template helpers --------------------------------------------


def _build_chat_messages(request: ModelRequest) -> list[MutableJSON]:
    """Translate internal Message list to HF chat-template format."""
    ids = IdRemapper("call_")
    messages: list[MutableJSON] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for msg in request.messages:
        if msg.descriptor == "text/x-user-message":
            messages.append({"role": "user", "content": cast(str, msg.content)})
        elif msg.descriptor == "multipart/x-user-message":
            text = "\n".join(
                str(p.content)
                for p in cast(tuple[Message, ...], msg.content)
                if p.descriptor == "text/plain"
            )
            messages.append({"role": "user", "content": text})
        elif msg.descriptor == "multipart/x-model-message":
            parts_mm = cast(tuple[Message, ...], msg.content)
            text_parts = [
                str(p.content) for p in parts_mm if p.descriptor == "text/plain"
            ]
            tool_calls_hf: list[MutableJSON] = []
            for p in parts_mm:
                if p.descriptor == "multipart/x-tool-call":
                    directive = get_directive(p)
                    tool_calls_hf.append(
                        {
                            "id": ids.map(get_queue_id(p)),
                            "type": "function",
                            "function": {
                                "name": get_tool_name(p),
                                "arguments": json.dumps(json_unfreeze(directive)),
                            },
                        }
                    )
            entry: MutableJSON = {
                "role": "assistant",
                "content": "\n".join(text_parts),
            }
            if tool_calls_hf:
                entry["tool_calls"] = cast(MutableJSONValue, tool_calls_hf)
            messages.append(entry)
        elif msg.descriptor == "multipart/x-tool-result":
            parts_tr = cast(tuple[Message, ...], msg.content)
            text = "\n".join(
                str(p.content)
                for p in parts_tr
                if p.descriptor in ("text/plain", "text/x-error")
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ids.map(get_queue_id(msg)),
                    "content": text,
                }
            )
    return messages


def _tool_schema(tool: Tool) -> MutableJSON:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": json_unfreeze(tool.directive_schema),
        },
    }


def _tool_preamble(tools: list[Tool]) -> str:
    schemas = [_tool_schema(t) for t in tools]
    return (
        "You have access to the following tools. To call a tool, emit "
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>.\n'
        + json.dumps(schemas, indent=2)
    )


# -- tool-call parsing -------------------------------------------------
#
# Qwen3: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
# DSV3 / Kimi-K2: ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`` with
# nested function descriptors. We accept both and any substring that
# looks like ``<tool_call>...</tool_call>``; everything outside the
# call blocks becomes the free-text response.

_QWEN_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_DS_BLOCK = re.compile(
    r"<\u2502tool\u2581calls\u2581begin\u2502>(.*?)<\u2502tool\u2581calls\u2581end\u2502>",
    re.DOTALL,
)
_DS_ONE = re.compile(
    r"<\u2502tool\u2581call\u2581begin\u2502>(.*?)<\u2502tool\u2581call\u2581end\u2502>",
    re.DOTALL,
)


def _extract_tool_calls(
    text: str,
    *,
    allowed_tools: set[str] | None = None,
) -> tuple[list[Message], str]:
    """Return (tool_call_messages, remaining_text)."""
    calls: list[Message] = []

    def qwen_repl(match: re.Match[str]) -> str:
        tc = _parse_qwen_tool_call(match.group(1))
        if tc is None:
            logger.warning("SelfHosted preserved malformed Qwen tool call.")
            return match.group(0)
        if not _tool_allowed(tc, allowed_tools):
            logger.warning(
                "SelfHosted preserved unadvertised tool call tool=%s.",
                get_tool_name(tc),
            )
            return match.group(0)
        calls.append(tc)
        return ""

    cleaned = _QWEN_TOOL_CALL.sub(qwen_repl, text)

    def deepseek_repl(block: re.Match[str]) -> str:
        block_calls: list[Message] = []
        for inner in _DS_ONE.finditer(block.group(1)):
            tc = _parse_deepseek_tool_call(inner.group(1))
            if tc is None:
                logger.warning("SelfHosted preserved malformed DeepSeek tool call.")
                return block.group(0)
            if not _tool_allowed(tc, allowed_tools):
                logger.warning(
                    "SelfHosted preserved unadvertised tool call tool=%s.",
                    get_tool_name(tc),
                )
                return block.group(0)
            block_calls.append(tc)
        if not block_calls:
            logger.warning("SelfHosted preserved empty DeepSeek tool-call block.")
            return block.group(0)
        calls.extend(block_calls)
        return ""

    cleaned = _DS_BLOCK.sub(deepseek_repl, cleaned)
    return calls, cleaned.strip()


def _tool_allowed(tool_call: Message, allowed_tools: set[str] | None) -> bool:
    """Return whether a parsed tool call was advertised to the model."""
    return allowed_tools is None or get_tool_name(tool_call).lower() in {
        tool.lower() for tool in allowed_tools
    }


def _parse_qwen_tool_call(raw: str) -> Message | None:
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload_d = cast(MutableJSON, payload)
    name = cast(str | None, payload_d.get("name"))
    raw_args = payload_d.get("arguments") or {}
    if not isinstance(name, str) or not name or not isinstance(raw_args, dict):
        return None
    tc_id = str(uuid.uuid4())[:12]
    return tool_call_message(tc_id, name, json_freeze(cast(MutableJSON, raw_args)))


def _parse_deepseek_tool_call(raw: str) -> Message | None:
    r"""DeepSeek/Kimi format: ``function\n{name}\n```json\n{args}\n``` ``.

    Best-effort: strip leading tag text, find the first JSON object
    with a ``name`` field, and an ``arguments`` sibling or the first
    code-fenced block.
    """
    text = raw.strip()
    name_match = re.search(r"function\s*\n\s*([A-Za-z0-9_]+)", text)
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if name_match is None or json_match is None:
        return None
    try:
        args = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    tc_name = name_match.group(1)
    tc_id = str(uuid.uuid4())[:12]
    return tool_call_message(tc_id, tc_name, json_freeze(cast(MutableJSON, args)))
