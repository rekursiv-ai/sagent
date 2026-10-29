"""Hugging Face causal LMs, loaded into this process and run with transformers.

A model is named by its Hugging Face repo id or local snapshot, with options
after ``+`` in any order: a device (``cuda``, ``mps``, ``cpu``, or ``auto`` to
shard across every GPU), a dtype (``bfloat16``, ``float16``, ``float32``),
``compile``, and a context size (``large``, ``200k``). Lookup reads nothing: the
config, which states the window, is read when the model's settings are first
asked for, and the weights load on the first request. Both are shared by every
model the provider hands out for that repo.

The model is told about tools by its chat template, and its calls are read
back out of the text: Qwen's ``<tool_call>`` JSON, and DeepSeek's and Kimi's
``<｜tool▁calls▁begin｜>`` blocks.

Download a model into the shared cache (``HF_HOME``, provisioned by ops/env),
not a directory of its own, so every checkout and user reuses one copy:

  hf download Qwen/Qwen3.6-27B

Typical usage example:

  async with SelfHosted() as local:                   # Qwen/Qwen3.6-27B
      qwen = local["Qwen/Qwen3.6-27B+bfloat16+cuda"]
      async for event in qwen.stream(request):        # Loads the weights.
          ...
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, cast, override

import asyncio
import contextlib
import threading

from treekle import from_plain, json_dumps, mutable

from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    literal_members,
)
from sagent.model.context import ContextSizeName
from sagent.model.cost import PriceCatalog, PriceCondition, TokenCount, TokenPrice
from sagent.model.exceptions import UnknownModelError
from sagent.model.types import Response, TextChunk
from sagent.providers.openai import wire
from sagent.providers.resources import SharedResources
from sagent.providers.selfhosted.calls import read_calls
from sagent.providers.table import ModelTags
from sagent.session.types import AssistantMessage, Thinking, ToolCall, ToolResult
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from torch import Tensor, nn

    import torch
    import transformers

    from sagent.model.types import Request
    from sagent.providers.types import ProviderSharedResources, UsageWindow
    from sagent.session.types import Attachment
    from sagent.tools.types import Tool
else:
    from wrapt import lazy_import

    # Seconds to import; a caller that never runs a local model never pays.
    torch = lazy_import("torch")
    transformers = lazy_import("transformers")


__all__ = [
    "SelfHosted",
    "SelfHostedModel",
]


class SelfHosted(AsyncResource):
    """Hugging Face models, run in this process.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      model: What a role (``default``, ``utility``) names: a repo id or local
        snapshot, with options.
      trust_remote_code: Whether a repo's own Python may run on load.

    """

    section: ClassVar[str] = "selfhosted"
    """The catalog section whose models and settings this provider serves."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        model: str = "Qwen/Qwen3.6-27B",
        trust_remote_code: bool = False,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self._default = model
        self._trust_remote_code = trust_remote_code
        self._windows: dict[str, int] = {}
        self._loaded: dict[_Spec, _Loaded] = {}
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Return the name the catalog knows this provider by."""
        return self._name

    @property
    def resources(self) -> ProviderSharedResources:
        """Return what this provider shares with others."""
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return ``()``: a local model has no rate limits."""
        return ()

    def __getitem__(self, model: str) -> SelfHostedModel:
        roles = ("default", "best", "everyday", "utility")
        # A catalog asks each provider in turn who serves a bare ``opus``; taken
        # as a repo id, it would load from the Hub instead of failing the lookup.
        if model not in roles and "/" not in model.partition("+")[0]:
            raise UnknownModelError(
                f"{model!r} is no Hugging Face repo id or path; give ORG/NAME or a path.",
            )
        spec, size = _spec(self._default if model in roles else model)
        return SelfHostedModel(provider=self, spec=spec, size=size)

    def __iter__(self) -> Iterator[str]:
        return iter((self._default,))

    @override
    async def aclose(self) -> None:
        self._loaded = {}
        if self._owns_resources:
            await self._resources.aclose()

    def window(self, path: str) -> int:
        """Return the tokens ``path``'s model takes in, read from its config once.

        Args:
          path: A repo id or local snapshot.

        Returns:
          tokens: The config's ``max_position_embeddings``.

        Raises:
          ValueError: The config states no window.

        """
        window = self._windows.get(path)
        if window is None:
            window = self._windows[path] = _window(
                path,
                trust_remote_code=self._trust_remote_code,
            )
        return window

    async def load(self, spec: _Spec) -> _Loaded:
        """Return the weights ``spec`` names, loading them on first use.

        Args:
          spec: The model and its load options.

        Returns:
          loaded: The weights and tokenizer, shared by every model of ``spec``.

        """
        async with self._lock:
            loaded = self._loaded.get(spec)
            if loaded is None:
                loaded = self._loaded[spec] = await asyncio.to_thread(
                    _load,
                    spec,
                    trust_remote_code=self._trust_remote_code,
                )
        return loaded


class SelfHostedModel(AsyncResource):
    """One local model; its capability and settings are built when first asked for."""

    temperature: ClassVar[float] = 1.0
    """The sampling temperature; ``0`` decodes greedily."""

    def __init__(
        self,
        *,
        provider: SelfHosted,
        spec: _Spec,
        size: ContextSizeName | int = "fullcontext",
    ) -> None:
        self._provider = provider
        self._spec = spec
        self._size: ContextSizeName | int = size
        self._capability: ModelCapability | None = None
        self._settings: ModelSettings | None = None

    @property
    def provider(self) -> SelfHosted:
        """Return the provider serving this model."""
        return self._provider

    @property
    def capability(self) -> ModelCapability:
        """Return every setting this model offers, built from the config on first call."""
        if self._capability is None:
            window = self.provider.window(self._spec.path)
            # In-process weights cost nothing, and no server rolls history.
            prices = PriceCatalog(
                prices={PriceCondition(): TokenPrice()},
                window=window,
            )
            self._capability = ModelCapability(
                model_id=self._spec.path,
                # ``stream`` caps each reply to the room the prompt leaves.
                limits=ModelLimits(
                    max_request_tokens=window,
                    max_response_tokens=4_096,
                    truncates_at_context=True,
                ),
                prices=prices,
                thinking_effort=frozenset({"none", "low", "medium", "high"}),
                default=ModelSettings(context_size=prices["auto", "fullcontext"]),
            )
        return self._capability

    @property
    def settings(self) -> ModelSettings:
        """Return the settings chosen, at the size the name gave."""
        if self._settings is None:
            capability = self.capability
            self._settings = replace(
                capability.default,
                context_size=capability.prices["auto", self._size],
            )
        return self._settings

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        """Choose ``settings``; raise unless the capability offers each choice."""
        self.capability.check(settings)
        self._settings = settings

    async def count(self, request: Request) -> int:
        """Return the tokens ``request`` renders to, as the model's tokenizer counts."""
        loaded = await self._loaded()
        return _render(loaded.tokenizer, request, settings=self.settings).shape[-1]

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return ``0``: the chat template renders text alone, so no attachment is sent."""
        del attachment
        return 0

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply, whole, then the finished ``Response``.

        Args:
          request: What the model is asked.

        Yields:
          event: The reply's text as one ``TextChunk``, then one ``Response``.

        """
        loaded = await self._loaded()
        ids = _render(loaded.tokenizer, request, settings=self.settings)
        prompt = ids.shape[-1]
        limits = self.capability.limits
        budget = min(
            request.response_tokens(limits),
            max(1, limits.max_request_tokens - prompt),
        )
        new = await _generated(
            loaded,
            ids,
            max_new_tokens=budget,
            temperature=self.temperature,
        )
        text = loaded.tokenizer.decode(new, skip_special_tokens=True)
        calls, said = read_calls(
            text,
            offered=[tool.name for tool in request.tools],
        )
        if said:
            yield TextChunk(text=said)
        records = (*((AssistantMessage(content=said),) if said else ()), *calls)
        tokens = TokenCount(request=prompt, response=new.shape[-1])
        # ``generate`` says no finish reason; a reply that spent its budget ran out.
        stop_reason = (
            "max_tokens"
            if tokens.response >= budget
            else "tool_use"
            if calls
            else "end_turn"
        )
        yield Response(
            records=records,
            tokens=tokens,
            spend=self.capability.prices[
                "auto",
                tokens.prompt,
                datetime.now(UTC).date(),
            ]
            * tokens,
            stop_reason=stop_reason,
        )

    async def _loaded(self) -> _Loaded:
        """Return the weights."""
        return await self.provider.load(self._spec)


class _Tokenizer(Protocol):
    """What this module asks of a Hugging Face tokenizer."""

    eos_token_id: int

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> object:
        """Return the conversation rendered, as ids when ``tokenize`` is set."""
        ...

    def decode(self, token_ids: Tensor, *, skip_special_tokens: bool) -> str:
        """Return ``token_ids`` as text."""
        ...


class _Generator(Protocol):
    """What this module asks of a Hugging Face causal LM."""

    def generate(self, inputs: Tensor, **kwargs: object) -> Tensor:
        """Return ``inputs`` and the tokens generated after them."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yield the weights, the first of which says the device."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _Spec:
    """A model named with its load options."""

    path: str
    device: Literal["", "auto", "cpu", "cuda", "mps"] = ""
    """Where the weights go; empty picks the fastest device present."""

    dtype: Literal["", "bfloat16", "float16", "float32"] = ""
    """The weights' type; empty keeps the checkpoint's."""

    compile: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class _Loaded:
    """A model's weights and tokenizer, in memory."""

    model: _Generator
    tokenizer: _Tokenizer
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Held while the model generates: it serves one reply at a time."""


def _spec(name: str) -> tuple[_Spec, ContextSizeName | int]:
    """Return what ``name`` (``repo+option+option``) asks to load, and the size it names."""
    path, *options = name.split("+")
    if not path:
        raise ValueError("A local model's name must start with a repo id or path.")
    devices = ("auto", "cpu", "cuda", "mps")
    dtypes = ("bfloat16", "float16", "float32")
    spec = _Spec(path=str(Path(path).expanduser()))
    size: ContextSizeName | int = "fullcontext"
    seen: set[str] = set()
    for raw in options:
        option = raw.removeprefix("torch.")
        # A load option first: the device ``auto`` is a tier's word too.
        named = (
            None
            if option in (*devices, *dtypes, "compile", "no-compile")
            else ModelTags.parse(f"{path}+{raw}").context
        )
        kind = (
            "device"
            if option in devices
            else "dtype"
            if option in dtypes
            else "compile"
            if option in ("compile", "no-compile")
            else "size"
            if named is not None
            else ""
        )
        if not kind:
            known = ", ".join(
                (
                    *devices,
                    *dtypes,
                    "compile",
                    "no-compile",
                    *literal_members(ContextSizeName),
                    "200k",
                ),
            )
            raise ValueError(f"Unknown option {raw!r} in {name!r}; use {known}.")
        if kind in seen:
            raise ValueError(f"The {kind} is given twice in {name!r}.")
        seen.add(kind)
        if named is not None:
            size = named
        elif kind == "device":
            spec = replace(
                spec,
                device=cast(Literal["auto", "cpu", "cuda", "mps"], option),
            )
        elif kind == "dtype":
            spec = replace(
                spec,
                dtype=cast(Literal["bfloat16", "float16", "float32"], option),
            )
        else:
            spec = replace(spec, compile=option == "compile")
    return spec, size


def _window(path: str, *, trust_remote_code: bool) -> int:
    """Return the tokens ``path``'s model takes in, as its config states."""
    config = dict[str, object](
        transformers.AutoConfig.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
        ).to_dict(),
    )
    text = from_plain(config.get("text_config"), dict[str, object], default={})
    stated = config.get("max_position_embeddings", text.get("max_position_embeddings"))
    if stated is None:
        raise ValueError(f"{path!r}'s config states no max_position_embeddings.")
    return from_plain(stated, int)


def _load(spec: _Spec, *, trust_remote_code: bool) -> _Loaded:
    """Load ``spec``'s weights and tokenizer; this blocks for minutes."""
    device = spec.device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else ""
    )
    options: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if spec.dtype:
        options["dtype"] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[spec.dtype]
    # ``auto`` lets accelerate shard the weights across every visible GPU, for
    # a model larger than one GPU holds.
    if device == "auto":
        options["device_map"] = "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(spec.path, **options)
    if device and device != "auto":
        model = model.to(device)
    model.eval()
    compiled = (
        cast("Callable[[object], object]", torch.compile)(model)
        if spec.compile
        else model
    )
    return _Loaded(
        model=cast(_Generator, compiled),
        tokenizer=cast(
            _Tokenizer,
            transformers.AutoTokenizer.from_pretrained(
                spec.path,
                trust_remote_code=trust_remote_code,
            ),
        ),
    )


def _render(
    tokenizer: _Tokenizer,
    request: Request,
    *,
    settings: ModelSettings,
) -> Tensor:
    """Return the request rendered by the model's chat template, as token ids."""
    messages = _messages(request)
    options: dict[str, object] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "enable_thinking": settings.thinking_effort != "none",
    }
    if request.tools:
        options["tools"] = [_schema(tool) for tool in request.tools]
    rendered = _templated(tokenizer, messages, options=options, tools=request.tools)
    ids = cast("Tensor", _field(rendered, "input_ids"))
    return ids.unsqueeze(0) if ids.ndim == 1 else ids


# A template that knows neither switch rejects it; each is dropped in turn, and a
# template that cannot take tools reads their schemas from the system prompt.
def _templated(
    tokenizer: _Tokenizer,
    messages: list[dict[str, object]],
    *,
    options: dict[str, object],
    tools: Sequence[Tool],
) -> object:
    """Apply the chat template, dropping each feature it rejects."""
    attempts: list[tuple[list[dict[str, object]], dict[str, object]]] = [
        (messages, options),
    ]
    bare = {key: value for key, value in options.items() if key != "enable_thinking"}
    attempts.append((messages, bare))
    if tools:
        inlined = _with_preamble(messages, tools=tools)
        without_tools = {key: value for key, value in bare.items() if key != "tools"}
        attempts.append(
            (inlined, {**without_tools, "enable_thinking": options["enable_thinking"]}),
        )
        attempts.append((inlined, without_tools))
    for conversation, kwargs in attempts[:-1]:
        with contextlib.suppress(TypeError, ValueError):
            return tokenizer.apply_chat_template(conversation, **kwargs)
    conversation, kwargs = attempts[-1]
    return tokenizer.apply_chat_template(conversation, **kwargs)


def _field(rendered: object, name: str) -> object:
    """Return ``name`` from what a chat template returned: a tensor, or a batch."""
    if isinstance(rendered, Mapping):
        return cast("Mapping[str, object]", rendered)[name]
    return getattr(rendered, name, rendered)


# A cancelled caller stops the model at its next token and waits for it, so neither the
# GPU nor the next reply is left behind a reply no one reads.
async def _generated(
    loaded: _Loaded,
    ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
) -> Tensor:
    """Generate the reply's tokens, one reply at a time; return only the new ones."""
    stop = threading.Event()
    async with loaded.lock:
        generating = asyncio.ensure_future(
            asyncio.to_thread(
                _generate,
                loaded,
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=stop,
            ),
        )
        try:
            return await asyncio.shield(generating)
        except asyncio.CancelledError:
            stop.set()
            await asyncio.wait({generating})
            raise


def _generate(
    loaded: _Loaded,
    ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    stop: threading.Event,
) -> Tensor:
    """Generate the reply's tokens, blocking; return only the new ones."""
    device = next(iter(loaded.model.parameters())).device
    ids = ids.to(device)
    options = _options(
        ids,
        device=device,
        eos=loaded.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=stop,
    )
    with torch.inference_mode():
        out = loaded.model.generate(ids, **options)
    return out[0, ids.shape[-1] :]


def _options(
    ids: Tensor,
    *,
    device: torch.device,
    eos: int,
    max_new_tokens: int,
    temperature: float,
    stop: threading.Event,
) -> dict[str, object]:
    """Return what ``generate`` is asked for one reply."""
    options: dict[str, object] = {
        "attention_mask": torch.ones_like(ids),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos,
        "pad_token_id": eos,
        # A plain list: ``generate`` merges it into its own criteria.
        "stopping_criteria": [_Stopped(stop)],
    }
    if device.type == "mps":
        options["use_cache"] = False
    if temperature > 0:
        options["do_sample"] = True
        options["temperature"] = temperature
    return options


class _Stopped:
    """A stopping criterion: done, every row, once ``event`` is set."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids: Tensor, scores: object, **kwargs: object) -> Tensor:
        del scores, kwargs
        return torch.full(
            (input_ids.shape[0],),
            self._event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


def _messages(request: Request) -> list[dict[str, object]]:
    """Return the system prompt and records as chat-template messages, text only."""
    ids: dict[str, str] = {}
    messages: list[dict[str, object]] = []
    if system := "\n\n".join(text for text in request.system if text):
        messages.append({"role": "system", "content": system})
    turn: dict[str, object] | None = None
    for record in wire.flatten(request.records):
        if isinstance(record, (AssistantMessage, ToolCall)):
            if turn is None:
                turn = {"role": "assistant", "content": ""}
                messages.append(turn)
            if isinstance(record, AssistantMessage):
                turn["content"] = f"{turn['content']}{record.content or ''}"
            else:
                calls = cast("list[object]", turn.setdefault("tool_calls", []))
                calls.append(_call(record, ids=ids))
            continue
        if isinstance(record, Thinking):
            continue
        said = wire.user_content(record)
        if said is not None:
            turn = None
            messages.append({"role": "user", "content": said[0]})
        elif isinstance(record, ToolResult):
            turn = None
            text = record.text()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ids.setdefault(record.call_id, f"call_{len(ids)}"),
                    "content": f"[Error] {text}"
                    if record.is_error is True and text
                    else text,
                },
            )
    return messages


def _call(call: ToolCall, *, ids: dict[str, str]) -> dict[str, object]:
    return {
        "id": ids.setdefault(call.call_id, f"call_{len(ids)}"),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json_dumps(mutable(call.arguments)),
        },
    }


def _schema(tool: Tool) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": mutable(tool.schema),
        },
    }


def _with_preamble(
    messages: list[dict[str, object]],
    *,
    tools: Sequence[Tool],
) -> list[dict[str, object]]:
    """Return ``messages`` with the tools' schemas told in the system prompt."""
    preamble = (
        "You have access to the following tools. To call a tool, emit "
        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>.\n'
        + json_dumps([_schema(tool) for tool in tools], indent=2)
    )
    if messages and messages[0]["role"] == "system":
        head = {**messages[0], "content": f"{messages[0]['content']}\n\n{preamble}"}
        return [head, *messages[1:]]
    return [{"role": "system", "content": preamble}, *messages]
