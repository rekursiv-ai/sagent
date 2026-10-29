"""Tests for the self-hosted provider, with a stand-in model and tokenizer.

No weights load: ``_load`` is replaced by one that returns a tokenizer that
renders a fixed prompt and a model that "generates" scripted tokens, so each
test runs in milliseconds and touches neither transformers nor a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import asyncio
import threading
import time

from torch import Tensor, nn

import pytest
import torch

from sagent.model.exceptions import UnknownModelError
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.resources import SharedResources
from sagent.providers.selfhosted import server
from sagent.providers.selfhosted.server import SelfHosted, SelfHostedModel
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    Thinking,
    TokenUsage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.read import Read


_REQUEST = Request(system=("Be terse.",), records=(UserMessage(content="hi"),))


@dataclass(slots=True, kw_only=True)
class _Tokenizer:
    """Renders every conversation to the same prompt, and decodes to ``text``."""

    text: str = "Hello."
    prompt: tuple[int, ...] = (5, 6, 7)
    rejects: frozenset[str] = frozenset()
    """Template options that raise, as a template that lacks them does."""

    eos_token_id: int = 2
    rendered: list[tuple[list[dict[str, object]], dict[str, object]]] = field(
        default_factory=list[tuple[list[dict[str, object]], dict[str, object]]],
    )

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> object:
        self.rendered.append((conversation, kwargs))
        if self.rejects & kwargs.keys():
            raise TypeError(
                f"unexpected keyword {sorted(self.rejects & kwargs.keys())}",
            )
        return {"input_ids": torch.tensor(self.prompt), "attention_mask": None}

    def decode(self, token_ids: Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        assert token_ids.ndim == 1
        return self.text


@dataclass(slots=True, kw_only=True)
class _Generator:
    """Appends ``new`` tokens to the prompt, after ``delay_sec`` or until stopped."""

    new: int = 2
    delay_sec: float = 0.0
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    active: int = 0
    most_active: int = 0
    started: threading.Event = field(default_factory=threading.Event)
    guard: threading.Lock = field(default_factory=threading.Lock)

    def generate(self, inputs: Tensor, **kwargs: object) -> Tensor:
        with self.guard:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        try:
            self.calls.append(kwargs)
            self.started.set()
            stops = cast(
                "list[Callable[..., Tensor]]",
                kwargs.get("stopping_criteria", []),
            )
            deadline = time.monotonic() + self.delay_sec
            while time.monotonic() < deadline:
                if any(bool(stop(inputs, None).all()) for stop in stops):
                    break
                time.sleep(0.001)
            added = torch.arange(self.new).unsqueeze(0) + 100
            return torch.cat([inputs, added], dim=-1)
        finally:
            with self.guard:
                self.active -= 1

    def parameters(self) -> Iterator[nn.Parameter]:
        yield nn.Parameter(torch.zeros(1))


@pytest.fixture
def configs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace reading a config with one stating a 40-token window; record each read."""
    read: list[str] = []

    def window(path: str, *, trust_remote_code: bool) -> int:
        del trust_remote_code
        read.append(path)
        return 40

    monkeypatch.setattr(server, "_window", window)
    return read


@pytest.fixture
def loads(
    monkeypatch: pytest.MonkeyPatch,
    configs: list[str],
) -> list[server._Spec]:
    """Replace loading with a stand-in, and record each spec loaded."""
    del configs
    loaded: list[server._Spec] = []

    def load(spec: server._Spec, *, trust_remote_code: bool) -> server._Loaded:
        del trust_remote_code
        loaded.append(spec)
        return server._Loaded(model=_Generator(), tokenizer=_Tokenizer())

    monkeypatch.setattr(server, "_load", load)
    return loaded


def _served(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: _Tokenizer | None = None,
    model: _Generator | None = None,
    window: int = 40,
    name: str = "default",
) -> SelfHostedModel:
    """Return a model whose weights are ``model`` and ``tokenizer``."""
    loaded = server._Loaded(
        model=model or _Generator(),
        tokenizer=tokenizer or _Tokenizer(),
    )

    def load(spec: server._Spec, *, trust_remote_code: bool) -> server._Loaded:
        del spec, trust_remote_code
        return loaded

    def stated(path: str, *, trust_remote_code: bool) -> int:
        del path, trust_remote_code
        return window

    monkeypatch.setattr(server, "_load", load)
    monkeypatch.setattr(server, "_window", stated)
    return SelfHosted()[name]


async def _reply(
    model: SelfHostedModel,
    request: Request = _REQUEST,
) -> list[TextChunk | Response]:
    return [event async for event in model.stream(request)]


# ---- naming and loading ------------------------------------------------------


def test_a_role_names_the_providers_model_and_lookup_loads_nothing(
    configs: list[str],
    loads: list[server._Spec],
) -> None:
    local = SelfHosted(model="Org/Model+cpu")
    assert local.name == "selfhosted"
    assert SelfHosted(name="gpu").name == "gpu"
    assert list(local) == ["Org/Model+cpu"]
    assert local.usage == ()
    looked_up = [local[role] for role in ("default", "best", "everyday", "utility")]
    assert configs == []
    assert {model.capability.model_id for model in looked_up} == {"Org/Model"}
    assert local["Other/Model"].capability.model_id == "Other/Model"
    # One config read per repo, when its settings are first asked for.
    assert configs == ["Org/Model", "Other/Model"]
    assert loads == []


@pytest.mark.parametrize(
    ("name", "context"),
    [
        ("default", 40),
        ("Org/M+smallcontext", 40),
        ("Org/M+cpu+30", 30),
        ("Org/M+2k", 40),
    ],
)
def test_the_window_is_the_configs_and_a_name_may_size_it(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    context: int,
) -> None:
    model = _served(monkeypatch, name=name)
    assert model.capability.limits.max_request_tokens == 40
    assert model.settings.context_size.tokens == context


@pytest.mark.parametrize("name", ["opus", "typo+cuda", ""])
def test_a_name_that_is_no_repo_id_or_path_is_unknown(name: str) -> None:
    """Else a catalog looking for who serves ``opus`` would load it from the Hub."""
    with pytest.raises(UnknownModelError, match="no Hugging Face repo id or path"):
        SelfHosted()[name]


@pytest.mark.parametrize(
    ("name", "spec"),
    [
        ("Org/M", server._Spec(path="Org/M")),
        (
            "Org/M+bfloat16+cuda",
            server._Spec(path="Org/M", device="cuda", dtype="bfloat16"),
        ),
        (
            "Org/M+torch.float16+compile+auto",
            server._Spec(path="Org/M", device="auto", dtype="float16", compile=True),
        ),
        ("Org/M+no-compile+mps", server._Spec(path="Org/M", device="mps")),
        (
            "~/w+cpu+float32",
            server._Spec(
                path=str(Path("~/w").expanduser()),
                device="cpu",
                dtype="float32",
            ),
        ),
    ],
)
def test_a_name_carries_its_load_options_in_any_order(
    name: str,
    spec: server._Spec,
) -> None:
    assert server._spec(name) == (spec, "fullcontext")


@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("Org/M+fullcontext", "fullcontext"),
        ("Org/M+200k+cuda", 200_000),
        ("Org/M+SmallContext", "smallcontext"),
        ("Org/M+midcontext", "midcontext"),
    ],
)
def test_a_name_may_carry_a_size_beside_its_load_options(
    name: str,
    size: str | int,
) -> None:
    spec, named = server._spec(name)
    assert (spec.path, named) == ("Org/M", size)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "must start with a repo id"),
        ("+cuda", "must start with a repo id"),
        ("Org/M+cuda+cpu", "device is given twice"),
        ("Org/M+bfloat16+float16", "dtype is given twice"),
        ("Org/M+compile+no-compile", "compile is given twice"),
        ("Org/M+fullcontext+200k", "size is given twice"),
        ("Org/M+fast", "Unknown option 'fast'"),
        ("Org/M+", "Unknown option ''"),
    ],
)
def test_a_bad_name_is_refused(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        server._spec(name)


@pytest.mark.asyncio
async def test_shared_resources_outlive_the_provider() -> None:
    async with SharedResources() as shared:
        local = SelfHosted(resources=shared)
        assert local.resources is shared
        http = shared.http
        await local.aclose()
        assert not http.is_closed
    owned = SelfHosted()
    http = owned.resources.http
    await owned.aclose()
    assert http.is_closed


@pytest.mark.asyncio
async def test_weights_load_once_per_name_and_are_shared(
    loads: list[server._Spec],
) -> None:
    local = SelfHosted()
    first, second = local["default"], local["default"]
    other = local["Other/Model"]
    await asyncio.gather(
        first.count(_REQUEST),
        second.count(_REQUEST),
        other.count(_REQUEST),
    )
    assert loads == [
        server._Spec(path="Qwen/Qwen3.6-27B"),
        server._Spec(path="Other/Model"),
    ]
    # Closing drops the weights; the next request loads them again.
    await local.aclose()
    await first.count(_REQUEST)
    assert len(loads) == 3


# ---- the reply ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_the_reply_whole_then_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _served(monkeypatch)
    events = await _reply(model)
    assert events[0] == TextChunk(text="Hello.")
    response = events[1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hello."),)
    assert (response.tokens.request, response.tokens.response) == (3, 2)
    assert response.spend.total == 0.0
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_a_reply_that_calls_a_tool_names_it_as_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = 'Reading. <tool_call>{"name": "read", "arguments": {"file_path": "a"}}</tool_call>'
    model = _served(monkeypatch, tokenizer=_Tokenizer(text=text))
    response = (
        await _reply(model, Request(records=_REQUEST.records, tools=(Read(),)))
    )[-1]
    assert isinstance(response, Response)
    (said, call) = response.records
    assert said == AssistantMessage(content="Reading.")
    # The runtime looks the tool up by its exact name.
    assert isinstance(call, ToolCall)
    assert (call.name, dict(call.arguments)) == ("Read", {"file_path": "a"})
    assert response.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_a_reply_of_only_a_call_shows_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = '<tool_call>{"name": "Read", "arguments": {}}</tool_call>'
    model = _served(monkeypatch, tokenizer=_Tokenizer(text=text))
    events = await _reply(model, Request(tools=(Read(),)))
    assert len(events) == 1
    response = events[0]
    assert isinstance(response, Response)
    assert [type(record) for record in response.records] == [ToolCall]


@pytest.mark.asyncio
async def test_a_reply_that_spends_its_budget_ran_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = '<tool_call>{"name": "Read", "arguments": {}}</tool_call>'
    # A window of 5 leaves 2 tokens after the 3-token prompt; the model writes 2.
    model = _served(monkeypatch, tokenizer=_Tokenizer(text=text), window=5)
    response = (await _reply(model, Request(tools=(Read(),))))[-1]
    assert isinstance(response, Response)
    assert response.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_the_reply_is_capped_within_what_the_window_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator()
    model = _served(monkeypatch, model=generator, window=1_000)
    await _reply(model)
    assert generator.calls[0]["max_new_tokens"] == min(4_096, 1_000 - 3)


@pytest.mark.asyncio
async def test_a_request_may_ask_for_less_than_the_window_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator()
    model = _served(monkeypatch, model=generator, window=1_000)
    await _reply(model, Request(max_response_tokens=50))
    assert generator.calls[0]["max_new_tokens"] == 50


@pytest.mark.asyncio
async def test_generation_is_sampled_and_stops_at_the_end_of_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator()
    model = _served(monkeypatch, model=generator)
    await _reply(model)
    (options,) = generator.calls
    assert options["eos_token_id"] == 2
    assert options["pad_token_id"] == 2
    assert options["do_sample"] is True
    assert options["temperature"] == 1.0
    assert "use_cache" not in options
    mask = options["attention_mask"]
    assert isinstance(mask, Tensor)
    assert torch.equal(mask, torch.ones(1, 3, dtype=torch.long))


@pytest.mark.asyncio
async def test_a_temperature_of_zero_decodes_greedily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator()
    model = _served(monkeypatch, model=generator)
    monkeypatch.setattr(SelfHostedModel, "temperature", 0.0)
    await _reply(model)
    assert "do_sample" not in generator.calls[0]
    assert "temperature" not in generator.calls[0]


def test_generation_on_mps_runs_without_the_kv_cache() -> None:
    ids = torch.tensor([[1, 2]])
    options = server._options(
        ids,
        device=torch.device("mps"),
        eos=0,
        max_new_tokens=3,
        temperature=0.0,
        stop=threading.Event(),
    )
    assert options["use_cache"] is False
    assert "use_cache" not in server._options(
        ids,
        device=torch.device("cpu"),
        eos=0,
        max_new_tokens=3,
        temperature=0.0,
        stop=threading.Event(),
    )


@pytest.mark.asyncio
async def test_count_is_the_rendered_prompts_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _served(monkeypatch, tokenizer=_Tokenizer(prompt=(1, 2, 3, 4, 5)))
    assert await model.count(_REQUEST) == 5


@pytest.mark.asyncio
async def test_one_model_generates_one_reply_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator(delay_sec=0.02)
    model = _served(monkeypatch, model=generator)
    await asyncio.gather(_reply(model), _reply(model), _reply(model))
    assert len(generator.calls) == 3
    assert generator.most_active == 1


@pytest.mark.asyncio
async def test_a_cancelled_reply_stops_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _Generator(delay_sec=30.0)
    model = _served(monkeypatch, model=generator)
    replying = asyncio.create_task(_reply(model))
    await asyncio.to_thread(generator.started.wait, 5.0)
    started = time.monotonic()
    replying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replying
    # Cancelled, it waits for the model to stop, and the model stopped at once.
    assert generator.active == 0
    assert time.monotonic() - started < 5.0


# ---- the prompt --------------------------------------------------------------


def test_the_conversation_renders_as_chat_template_messages() -> None:
    request = Request(
        system=("Be terse.", "", "Mind the tests."),
        records=(
            UserMessage(content="look"),
            TokenUsage(),
            Thinking(content="hidden"),
            AssistantMessage(content="Reading"),
            AssistantMessage(content=" both."),
            ToolCall(call_id="toolu_a", name="Read", arguments={"file_path": "a"}),
            ToolCall(call_id="toolu_b", name="Read", arguments={"file_path": "b"}),
            UncategorizedToolResult(call_id="toolu_a", content="x = 1"),
            UncategorizedToolResult(call_id="toolu_b", content="gone", is_error=True),
            UncategorizedToolResult(call_id="toolu_c", is_error=True),
            AssistantMessage(content="Done."),
        ),
    )
    assert server._messages(request) == [
        {"role": "system", "content": "Be terse.\n\nMind the tests."},
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": "Reading both.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "a"}'},
                },
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "b"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "x = 1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "[Error] gone"},
        {"role": "tool", "tool_call_id": "call_2", "content": ""},
        {"role": "assistant", "content": "Done."},
    ]
    assert server._messages(Request()) == []


@pytest.mark.asyncio
async def test_the_template_is_asked_for_thinking_and_the_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    model = _served(monkeypatch, tokenizer=tokenizer)
    model.settings = replace(model.settings, thinking_effort="high")
    await model.count(Request(records=_REQUEST.records, tools=(Read(),)))
    ((_, options),) = tokenizer.rendered
    read = Read()
    assert options == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "enable_thinking": True,
        "tools": [server._schema(read)],
    }
    assert server._schema(read) == {
        "type": "function",
        "function": {
            "name": "Read",
            "description": read.description,
            "parameters": dict(read.schema),
        },
    }


@pytest.mark.parametrize(
    ("rejects", "tried"),
    [
        (frozenset({"enable_thinking"}), [{"enable_thinking", "tools"}, {"tools"}]),
        (
            frozenset({"tools"}),
            [{"enable_thinking", "tools"}, {"tools"}, {"enable_thinking"}],
        ),
        (
            frozenset({"enable_thinking", "tools"}),
            [{"enable_thinking", "tools"}, {"tools"}, {"enable_thinking"}, set[str]()],
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_template_that_lacks_a_feature_is_asked_without_it(
    monkeypatch: pytest.MonkeyPatch,
    rejects: frozenset[str],
    tried: list[set[str]],
) -> None:
    tokenizer = _Tokenizer(rejects=rejects)
    model = _served(monkeypatch, tokenizer=tokenizer)
    assert await model.count(Request(system=("Sys.",), tools=(Read(),))) == 3
    assert [
        set(options) & {"enable_thinking", "tools"} for _, options in tokenizer.rendered
    ] == tried
    conversation, options = tokenizer.rendered[-1]
    if "tools" not in options:
        # Without native tools, their schemas ride the system prompt.
        system = conversation[0]["content"]
        assert isinstance(system, str)
        assert system.startswith("Sys.\n\nYou have access to the following tools.")
        assert '"name": "Read"' in system


@pytest.mark.asyncio
async def test_a_template_that_fails_every_way_raises_its_last_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer(rejects=frozenset({"tokenize"}))
    model = _served(monkeypatch, tokenizer=tokenizer)
    with pytest.raises(TypeError, match="tokenize"):
        await model.count(Request(tools=(Read(),)))
    with pytest.raises(TypeError, match="tokenize"):
        await model.count(Request())
    assert len(tokenizer.rendered) == 4 + 2


def test_a_tool_preamble_opens_a_system_prompt_when_there_is_none() -> None:
    messages = server._with_preamble(
        [{"role": "user", "content": "hi"}],
        tools=(Read(),),
    )
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith(
        "You have access to the following tools.",
    )
    assert messages[1] == {"role": "user", "content": "hi"}


@pytest.mark.parametrize(
    "rendered",
    [
        {"input_ids": torch.tensor([[1, 2]])},
        SimpleNamespace(input_ids=torch.tensor([1, 2])),
        torch.tensor([1, 2]),
    ],
)
def test_the_prompt_ids_are_read_from_whatever_the_template_returns(
    rendered: object,
) -> None:
    ids = server._field(rendered, "input_ids")
    assert isinstance(ids, Tensor)
    assert torch.equal(ids.reshape(1, -1), torch.tensor([[1, 2]]))


# ---- loading weights ---------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _Hub:
    """Stands in for transformers and torch, recording what loading asked."""

    config: Mapping[str, object] = field(default_factory=dict[str, object])
    mps: bool = False
    cuda: bool = False
    asked: list[tuple[str, object]] = field(default_factory=list[tuple[str, object]])

    def transformers(self) -> SimpleNamespace:
        return SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=self._config),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self._model),
            AutoTokenizer=SimpleNamespace(from_pretrained=self._tokenizer),
        )

    def torch(self) -> SimpleNamespace:
        return SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: self.mps),
            ),
            cuda=SimpleNamespace(is_available=lambda: self.cuda),
            compile=self._compile,
            bfloat16=torch.bfloat16,
            float16=torch.float16,
            float32=torch.float32,
        )

    def _config(self, path: str, **options: object) -> SimpleNamespace:
        self.asked.append(("config", (path, options)))
        return SimpleNamespace(to_dict=lambda: dict(self.config))

    def _model(self, path: str, **options: object) -> _Weights:
        self.asked.append(("model", (path, options)))
        return _Weights(asked=self.asked)

    def _tokenizer(self, path: str, **options: object) -> str:
        self.asked.append(("tokenizer", (path, options)))
        return "tokenizer"

    def _compile(self, model: object) -> str:
        self.asked.append(("compile", model))
        return "compiled"


@dataclass(slots=True, kw_only=True)
class _Weights:
    """A loaded model, recording where it is moved and that it is put in eval."""

    asked: list[tuple[str, object]]

    def to(self, device: str) -> _Weights:
        self.asked.append(("to", device))
        return self

    def eval(self) -> None:
        self.asked.append(("eval", None))


def _load(
    monkeypatch: pytest.MonkeyPatch,
    hub: _Hub,
    name: str,
    *,
    trust: bool = False,
) -> server._Loaded:
    monkeypatch.setattr(server, "transformers", hub.transformers())
    monkeypatch.setattr(server, "torch", hub.torch())
    spec, _ = server._spec(name)
    return server._load(spec, trust_remote_code=trust)


@pytest.mark.parametrize(
    ("config", "window"),
    [
        ({"max_position_embeddings": 262_144}, 262_144),
        ({"text_config": {"max_position_embeddings": 65_536}}, 65_536),
    ],
)
def test_the_window_is_read_from_the_models_config(
    monkeypatch: pytest.MonkeyPatch,
    config: Mapping[str, object],
    window: int,
) -> None:
    hub = _Hub(config=config)
    monkeypatch.setattr(server, "transformers", hub.transformers())
    assert server._window("Org/M", trust_remote_code=True) == window
    assert hub.asked == [("config", ("Org/M", {"trust_remote_code": True}))]


def test_a_config_that_states_no_window_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "transformers", _Hub().transformers())
    with pytest.raises(ValueError, match="states no max_position_embeddings"):
        server._window("Org/M", trust_remote_code=False)


@pytest.mark.parametrize(
    "case",
    [
        ("Org/M", True, True, "mps", {}),
        ("Org/M", False, True, "cuda", {}),
        ("Org/M", False, False, None, {}),
        ("Org/M+cpu", True, True, "cpu", {}),
        ("Org/M+auto", False, True, None, {"device_map": "auto"}),
        ("Org/M+bfloat16", False, False, None, {"dtype": torch.bfloat16}),
        ("Org/M+float16", False, False, None, {"dtype": torch.float16}),
        ("Org/M+float32", False, False, None, {"dtype": torch.float32}),
    ],
)
def test_weights_load_onto_the_fastest_device_there_is(
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, bool, bool, str | None, dict[str, object]],
) -> None:
    name, mps, cuda, moved, options = case
    hub = _Hub(mps=mps, cuda=cuda)
    loaded = _load(monkeypatch, hub, name, trust=True)
    assert ("model", ("Org/M", {"trust_remote_code": True, **options})) in hub.asked
    assert [device for kind, device in hub.asked if kind == "to"] == (
        [moved] if moved is not None else []
    )
    assert ("eval", None) in hub.asked
    assert "config" not in [kind for kind, _ in hub.asked]
    assert ("tokenizer", ("Org/M", {"trust_remote_code": True})) in hub.asked
    assert loaded.tokenizer == "tokenizer"


def test_a_compiled_model_is_the_one_served(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = _Hub()
    assert _load(monkeypatch, hub, "Org/M+compile").model == "compiled"
    assert [kind for kind, _ in hub.asked].count("compile") == 1


def test_an_attachment_costs_nothing_for_it_is_never_sent() -> None:
    attached = Attachment(mime_descriptor="image/png", data=b"x" * 40)
    assert SelfHosted()["default"].attachment_tokens(attached) == 0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
