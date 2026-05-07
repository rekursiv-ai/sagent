"""Tests for sagent.providers.selfhosted tool-call parsing + chat rendering.

End-to-end model loading lives behind ``@pytest.mark.integration``
(requires transformers + real HF weights). The parsing and chat-format
helpers are exercised here without model instantiation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override
from unittest.mock import Mock

import logging

import pytest


if TYPE_CHECKING:
    from torch import nn

    import torch
else:
    torch = pytest.importorskip(
        "torch",
        reason="SelfHosted tests require the selfhosted extra.",
    )
    nn = pytest.importorskip(
        "torch.nn",
        reason="SelfHosted tests require the selfhosted extra.",
    )

from sagent.custom_types import (
    BytesMessage,
    JsonMessage,
    Message,
    ModelRequest,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import get_directive, get_tool_name
from sagent.providers.selfhosted import (
    SelfHosted,
    SelfHostedModel,
    _build_chat_messages,
    _default_device,
    _disable_generate_cache,
    _extract_tool_calls,
    _parse_deepseek_tool_call,
    _parse_model_spec,
    _parse_qwen_tool_call,
    _RenderedPrompt,
    _tool_preamble,
)


class _FakeTool:
    name: str = "bash"
    tool_id: str = "application/x-tool-bash"
    description: str = "Run a bash command."
    supports_microcompaction: bool = False

    def __init__(self) -> None:
        self.directive_schema = json_freeze(
            {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            }
        )

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str | None:
        return None

    async def run(self, msg: Message) -> Message:
        del msg
        return TextMessage("", "text/plain")


class TestToolCallParsing:
    def test_qwen_single_call(self):
        text = 'text before <tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call> after'
        calls, cleaned = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].descriptor == "multipart/x-tool-call"
        assert get_tool_name(calls[0]) == "bash"
        assert get_directive(calls[0]) == {"cmd": "ls"}
        assert "<tool_call>" not in cleaned

    def test_qwen_multiple_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
        )
        calls, cleaned = _extract_tool_calls(text)
        assert [get_tool_name(c) for c in calls] == ["a", "b"]
        assert get_directive(calls[1]) == {"x": 1}
        assert cleaned == ""

    def test_qwen_malformed_ignored(self):
        text = "<tool_call>not-json</tool_call> ok"
        calls, cleaned = _extract_tool_calls(text)
        assert calls == []
        assert cleaned == "<tool_call>not-json</tool_call> ok"

    def test_qwen_unknown_tool_is_not_dispatched(self) -> None:
        text = '<tool_call>{"name": "made_up", "arguments": {}}</tool_call>'

        calls, cleaned = _extract_tool_calls(text, allowed_tools={"bash"})

        assert calls == []
        assert cleaned == text

    def test_qwen_tool_allowlist_matches_cli_tool_names(self) -> None:
        text = '<tool_call>{"name": "bash", "arguments": {"cmd": "pwd"}}</tool_call>'

        calls, cleaned = _extract_tool_calls(text, allowed_tools={"Bash"})

        assert len(calls) == 1
        assert get_tool_name(calls[0]) == "bash"
        assert cleaned == ""

    def test_deepseek_block(self):
        text = (
            "<\u2502tool\u2581calls\u2581begin\u2502>"
            "<\u2502tool\u2581call\u2581begin\u2502>"
            'function\nbash\n```json\n{"cmd": "ls"}\n```'
            "<\u2502tool\u2581call\u2581end\u2502>"
            "<\u2502tool\u2581calls\u2581end\u2502>"
        )
        calls, cleaned = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].descriptor == "multipart/x-tool-call"
        assert get_tool_name(calls[0]) == "bash"
        assert get_directive(calls[0]) == {"cmd": "ls"}
        assert cleaned == ""

    def test_deepseek_unknown_tool_is_not_dispatched(self) -> None:
        text = (
            "<\u2502tool\u2581calls\u2581begin\u2502>"
            "<\u2502tool\u2581call\u2581begin\u2502>"
            "function\nmade_up\n```json\n{}\n```"
            "<\u2502tool\u2581call\u2581end\u2502>"
            "<\u2502tool\u2581calls\u2581end\u2502>"
        )

        calls, cleaned = _extract_tool_calls(text, allowed_tools={"bash"})

        assert calls == []
        assert cleaned == text

    def test_qwen_parse_direct(self):
        tc = _parse_qwen_tool_call('{"name": "f", "arguments": {"a": 1}}')
        assert tc is not None
        assert tc.descriptor == "multipart/x-tool-call"
        assert get_tool_name(tc) == "f"
        assert get_directive(tc) == {"a": 1}

    def test_deepseek_parse_direct(self):
        raw = 'function\nbash\n```json\n{"cmd": "ls"}\n```'
        tc = _parse_deepseek_tool_call(raw)
        assert tc is not None
        assert tc.descriptor == "multipart/x-tool-call"
        assert get_tool_name(tc) == "bash"

    def test_plain_text_no_calls(self):
        calls, cleaned = _extract_tool_calls("just a response")
        assert calls == []
        assert cleaned == "just a response"


class _FakeTensor:
    def __init__(self, ndim: int = 1) -> None:
        self.ndim = ndim
        self.unsqueeze_dim: int | None = None

    def unsqueeze(self, dim: int) -> _FakeTensor:
        self.unsqueeze_dim = dim
        self.ndim = 2
        return self


class _FakeTokenizer:
    eos_token_id = 2

    def __init__(
        self,
        *,
        fail_on_tools: bool = False,
        fail_on_effort: bool = False,
        return_mapping: bool = False,
    ) -> None:
        self.fail_on_tools = fail_on_tools
        self.fail_on_effort = fail_on_effort
        self.return_mapping = return_mapping
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages: object, **kwargs: object) -> object:
        messages_list = cast(list[dict[str, object]], messages)
        self.calls.append((messages_list, dict(kwargs)))
        if self.fail_on_tools and "tools" in kwargs:
            raise TypeError("tools unsupported")
        if self.fail_on_effort and "enable_thinking" in kwargs:
            raise TypeError("effort unsupported")
        tensor = _FakeTensor()
        return {"input_ids": tensor} if self.return_mapping else tensor

    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "decoded"


class _FakeBatchEncoding:
    def __init__(self) -> None:
        self.input_ids = _FakeTensor()
        self.attention_mask = _FakeTensor()


class _FakeBatchEncodingTokenizer(_FakeTokenizer):
    @override
    def apply_chat_template(self, messages: object, **kwargs: object) -> object:
        messages_list = cast(list[dict[str, object]], messages)
        self.calls.append((messages_list, dict(kwargs)))
        return _FakeBatchEncoding()


class _FakeProvider:
    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.hosted_model_id = "local"
        self.hosted_max_request_tokens = 128
        self.hosted_max_response_tokens = 16

    @property
    def native_model(self) -> nn.Module:
        return nn.Module()


class _GenerateCaptureModel:
    def __init__(self, *, device: str = "cpu") -> None:
        self.weight = torch.zeros((), device=device)
        self.kwargs: dict[str, object] = {}
        self.grad_enabled = True

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter((self.weight,))

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.kwargs = kwargs
        self.grad_enabled = torch.is_grad_enabled()
        return torch.cat(
            (input_ids, torch.tensor([[3]], device=input_ids.device)),
            dim=1,
        )


class _CaptureProvider(_FakeProvider):
    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__(_FakeTokenizer())
        self.capture_model = _GenerateCaptureModel(device=device)

    @property
    @override
    def native_model(self) -> nn.Module:
        return cast(nn.Module, self.capture_model)


class TestChatRender:
    def test_system_and_user(self):
        req = ModelRequest(
            messages=[
                TextMessage("hi", "text/x-user-message"),
            ],
            system="be terse",
        )
        msgs = cast(list[Any], _build_chat_messages(req))
        assert msgs[0] == {"role": "system", "content": "be terse"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_assistant_tool_calls(self):
        req = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("running", "text/plain"),
                        MultipartMessage(
                            (
                                TextMessage("t1", "text/x-queue-id"),
                                JsonMessage(
                                    json_freeze({"cmd": "ls"}),
                                    "application/x-tool-bash",
                                ),
                            ),
                            "multipart/x-tool-call",
                        ),
                    ),
                    "multipart/x-model-message",
                ),
            ],
        )
        msgs = cast(list[Any], _build_chat_messages(req))
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "bash"
        # arguments is a JSON string (HF convention).
        assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_tool_result(self):
        req = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("t1", "text/x-queue-id"),
                        TextMessage("done", "text/plain"),
                    ),
                    "multipart/x-tool-result",
                ),
            ],
        )
        msgs = cast(list[Any], _build_chat_messages(req))
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["content"] == "done"
        assert msgs[0]["tool_call_id"] == "call_0"

    def test_tool_preamble_contains_schema(self):
        preamble = _tool_preamble([_FakeTool()])
        assert "bash" in preamble
        assert '"required"' in preamble

    def test_multipart_user_keeps_text_parts(self) -> None:
        req = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("one", "text/plain"),
                        BytesMessage(b"image", "image/png"),
                        TextMessage("two", "text/plain"),
                    ),
                    "multipart/x-user-message",
                )
            ]
        )
        assert _build_chat_messages(req) == [{"role": "user", "content": "one\ntwo"}]

    def test_tool_result_keeps_error_parts(self) -> None:
        req = ModelRequest(
            messages=[
                MultipartMessage(
                    (
                        TextMessage("q1", "text/x-queue-id"),
                        TextMessage("ok", "text/plain"),
                        TextMessage("bad", "text/x-error"),
                    ),
                    "multipart/x-tool-result",
                )
            ]
        )
        assert _build_chat_messages(req)[0]["content"] == "ok\nbad"


class TestSelfHostedProvider:
    def test_default_device_prefers_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.backends.mps.is_available",
            Mock(return_value=True),
        )
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.cuda.is_available",
            Mock(return_value=True),
        )

        assert _default_device() == "mps"

    def test_default_device_uses_cuda_without_mps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.backends.mps.is_available",
            Mock(return_value=False),
        )
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.cuda.is_available",
            Mock(return_value=True),
        )

        assert _default_device() == "cuda"

    def test_default_device_none_without_accelerator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.backends.mps.is_available",
            Mock(return_value=False),
        )
        monkeypatch.setattr(
            "sagent.providers.selfhosted.torch.cuda.is_available",
            Mock(return_value=False),
        )

        assert _default_device() is None

    def test_from_hf_loads_transformers_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = nn.Module()
        tokenizer = _FakeTokenizer()

        class FakeConfig:
            def to_dict(self) -> dict[str, object]:
                return {
                    "model_type": "qwen3_5",
                    "max_position_embeddings": 8192,
                }

        class FakeAutoConfig:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
                assert model_id == "Qwen/Qwen3.6-27B"
                assert kwargs == {"trust_remote_code": False}
                return FakeConfig()

        class FakeAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> nn.Module:
                assert model_id == "Qwen/Qwen3.6-27B"
                assert kwargs == {"trust_remote_code": False}
                return model

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> _FakeTokenizer:
                assert model_id == "Qwen/Qwen3.6-27B"
                assert kwargs == {"trust_remote_code": False}
                return tokenizer

        monkeypatch.setattr(
            "sagent.providers.selfhosted.transformers_lib",
            type(
                "FakeTransformers",
                (),
                {
                    "AutoConfig": FakeAutoConfig,
                    "AutoModelForCausalLM": FakeAutoModelForCausalLM,
                    "AutoTokenizer": FakeAutoTokenizer,
                },
            ),
        )

        provider = SelfHosted.from_hf("Qwen/Qwen3.6-27B")

        assert provider.native_model is model
        assert provider.tokenizer is tokenizer
        assert provider.hosted_model_id == "Qwen/Qwen3.6-27B"
        assert provider.hosted_max_request_tokens == 8192

    def test_from_hf_compiles_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = nn.Module()
        compiled_model = nn.Module()
        tokenizer = _FakeTokenizer()

        class FakeConfig:
            def to_dict(self) -> dict[str, object]:
                return {"max_position_embeddings": 8192}

        class FakeAutoConfig:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
                del model_id, kwargs
                return FakeConfig()

        class FakeAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> nn.Module:
                del model_id, kwargs
                return model

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> _FakeTokenizer:
                del model_id, kwargs
                return tokenizer

        compile_mock = Mock(return_value=compiled_model)
        monkeypatch.setattr(
            "sagent.providers.selfhosted.transformers_lib",
            type(
                "FakeTransformers",
                (),
                {
                    "AutoConfig": FakeAutoConfig,
                    "AutoModelForCausalLM": FakeAutoModelForCausalLM,
                    "AutoTokenizer": FakeAutoTokenizer,
                },
            ),
        )
        monkeypatch.setattr(
            "sagent.providers.selfhosted._compile_model",
            compile_mock,
        )

        provider = SelfHosted.from_hf("Qwen/Qwen3.6-27B", compile_model=True)

        compile_mock.assert_called_once_with(model)
        assert provider.native_model is compiled_model

    def test_parse_model_spec_accepts_options_in_any_order(self) -> None:
        left = _parse_model_spec("Qwen/Qwen3.6-27B+bfloat16+cuda")
        right = _parse_model_spec("Qwen/Qwen3.6-27B+cuda+bfloat16")

        assert left == right
        assert left.path_or_repo == "Qwen/Qwen3.6-27B"
        assert left.device == "cuda"
        assert left.dtype is torch.bfloat16
        assert not left.compile_model

    def test_parse_model_spec_accepts_compile_and_torch_dtype(self) -> None:
        spec = _parse_model_spec("Qwen/Qwen3.6-27B+torch.float16+auto+compile")

        assert spec.path_or_repo == "Qwen/Qwen3.6-27B"
        assert spec.device == "auto"
        assert spec.dtype is torch.float16
        assert spec.compile_model

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("+cuda+bfloat16", "must start"),
            ("Qwen/Qwen3.6-27B+", "must not be empty"),
            ("Qwen/Qwen3.6-27B++cuda", "must not be empty"),
            ("Qwen/Qwen3.6-27B+gpu", "Unsupported"),
            ("Qwen/Qwen3.6-27B+bfloat16+float16", "Duplicate"),
            ("Qwen/Qwen3.6-27B+cuda+mps", "Duplicate"),
            ("Qwen/Qwen3.6-27B+compile+no-compile", "Duplicate"),
        ],
    )
    def test_parse_model_spec_rejects_invalid_options(
        self, spec: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _parse_model_spec(spec)

    def test_from_key_loads_hf_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id=str(tmp_path),
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setattr(SelfHosted, "from_hf", load)
        monkeypatch.setattr(
            "sagent.providers.selfhosted._default_device",
            Mock(return_value=None),
        )

        assert SelfHosted.from_key(str(tmp_path)) is provider
        load.assert_called_once_with(
            str(tmp_path),
            device=None,
            dtype=None,
            compile_model=False,
        )

    def test_from_key_uses_default_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id=str(tmp_path),
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setattr(SelfHosted, "from_hf", load)
        monkeypatch.setattr(
            "sagent.providers.selfhosted._default_device",
            Mock(return_value="mps"),
        )

        assert SelfHosted.from_key(str(tmp_path)) is provider
        load.assert_called_once_with(
            str(tmp_path),
            device="mps",
            dtype=None,
            compile_model=False,
        )

    def test_from_key_reads_inline_device_and_dtype(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="Qwen/Qwen3.6-27B",
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setattr(SelfHosted, "from_hf", load)

        assert SelfHosted.from_key("Qwen/Qwen3.6-27B+cuda+bfloat16") is provider
        load.assert_called_once_with(
            "Qwen/Qwen3.6-27B",
            device="cuda",
            dtype=torch.bfloat16,
            compile_model=False,
        )

    def test_from_key_reads_inline_compile(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="Qwen/Qwen3.6-27B",
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setattr(SelfHosted, "from_hf", load)
        monkeypatch.setattr(
            "sagent.providers.selfhosted._default_device",
            Mock(return_value=None),
        )

        assert SelfHosted.from_key("Qwen/Qwen3.6-27B+compile") is provider
        load.assert_called_once_with(
            "Qwen/Qwen3.6-27B",
            device=None,
            dtype=None,
            compile_model=True,
        )

    def test_from_key_explicit_args_override_inline_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="Qwen/Qwen3.6-27B",
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setattr(SelfHosted, "from_hf", load)

        assert (
            SelfHosted.from_key(
                "Qwen/Qwen3.6-27B+cuda+bfloat16+compile",
                device="cpu",
                dtype=torch.float32,
                compile_model=False,
            )
            is provider
        )
        load.assert_called_once_with(
            "Qwen/Qwen3.6-27B",
            device="cpu",
            dtype=torch.float32,
            compile_model=False,
        )

    def test_from_key_ignores_selfhosted_env_vars(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="Qwen/Qwen3.6-27B",
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setenv("SAGENT_SELFHOSTED_DEVICE", "mps")
        monkeypatch.setenv("SAGENT_SELFHOSTED_DTYPE", "float16")
        monkeypatch.setenv("SAGENT_SELFHOSTED_COMPILE", "1")
        monkeypatch.setattr(SelfHosted, "from_hf", load)
        monkeypatch.setattr(
            "sagent.providers.selfhosted._default_device",
            Mock(return_value=None),
        )

        assert SelfHosted.from_key("Qwen/Qwen3.6-27B") is provider
        load.assert_called_once_with(
            "Qwen/Qwen3.6-27B",
            device=None,
            dtype=None,
            compile_model=False,
        )

    def test_from_env_uses_default_model_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id=SelfHosted.DEFAULT_MODEL,
            max_request_tokens=128,
        )
        load = Mock(return_value=provider)
        monkeypatch.setenv("SAGENT_SELFHOSTED_MODEL", "other")
        monkeypatch.setenv("SAGENT_SELFHOSTED_DEVICE", "mps")
        monkeypatch.setenv("SAGENT_SELFHOSTED_DTYPE", "float16")
        monkeypatch.setenv("SAGENT_SELFHOSTED_COMPILE", "1")
        monkeypatch.setattr(SelfHosted, "from_hf", load)
        monkeypatch.setattr(
            "sagent.providers.selfhosted._default_device",
            Mock(return_value=None),
        )

        assert SelfHosted.from_env() is provider
        load.assert_called_once_with(
            SelfHosted.DEFAULT_MODEL,
            device=None,
            dtype=None,
            compile_model=False,
        )

    def test_properties_and_model_binding(self) -> None:
        tokenizer = _FakeTokenizer()
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=tokenizer,
            model_id="local",
            max_request_tokens=128,
            max_response_tokens=32,
        )
        assert provider.native_model is not None
        assert provider.tokenizer is tokenizer
        assert provider.hosted_model_id == "local"
        assert provider.hosted_max_request_tokens == 128
        assert provider.hosted_max_response_tokens == 32
        assert isinstance(provider.model(), SelfHostedModel)
        assert provider.model().model_id == "local"
        assert provider.utility_model().model_id == "local"

    def test_default_response_token_limit(self) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="local",
            max_request_tokens=128,
        )
        assert provider.hosted_max_response_tokens == 4096

    def test_model_rejects_mismatched_id(self) -> None:
        provider = SelfHosted(
            model=nn.Module(),
            tokenizer=_FakeTokenizer(),
            model_id="local",
            max_request_tokens=128,
        )
        with pytest.raises(ValueError, match="bound to 'local'"):
            provider.model("other")


class TestSelfHostedModel:
    def test_capability_properties(self) -> None:
        model = SelfHostedModel(provider=_FakeProvider(_FakeTokenizer()))
        assert model.max_request_tokens == 128
        assert model.model_id == "local"
        assert model.max_response_tokens == 16
        assert not model.supports_streaming
        assert not model.supports_thinking
        assert model.supports_effort
        assert not model.supports_cache_control
        assert not model.supports_context_management
        assert not model.supports_persistent_retry
        assert not model.supports_account_auth
        assert model.max_image_dim == 2000
        assert model.max_image_bytes == 5 * 1024 * 1024
        assert not model.is_context_overflow(RuntimeError("boom"))
        assert model.estimate_text_token_count("abcdefgh") == 2

    def test_render_passes_tool_schema(self) -> None:
        tokenizer = _FakeTokenizer()
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))
        ids = model._render(
            ModelRequest(
                messages=[TextMessage("hi", "text/x-user-message")], tools=[_FakeTool()]
            )
        )
        assert ids.ndim == 2
        fake = cast(_FakeTensor, ids)
        assert fake.unsqueeze_dim == 0
        assert "tools" in tokenizer.calls[0][1]

    def test_render_effort_none_disables_thinking(self) -> None:
        tokenizer = _FakeTokenizer()
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        model._render(
            ModelRequest(
                messages=[TextMessage("hi", "text/x-user-message")],
                effort="none",
            )
        )

        assert tokenizer.calls[0][1]["enable_thinking"] is False

    def test_render_effort_enables_thinking(self) -> None:
        tokenizer = _FakeTokenizer()
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        model._render(
            ModelRequest(
                messages=[TextMessage("hi", "text/x-user-message")],
                effort="high",
            )
        )

        assert tokenizer.calls[0][1]["enable_thinking"] is True

    def test_render_retries_without_unsupported_effort(self) -> None:
        tokenizer = _FakeTokenizer(fail_on_effort=True)
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        model._render(
            ModelRequest(
                messages=[TextMessage("hi", "text/x-user-message")],
                effort="none",
            )
        )

        assert tokenizer.calls[0][1]["enable_thinking"] is False
        assert "enable_thinking" not in tokenizer.calls[1][1]

    def test_render_accepts_tokenizer_mapping(self) -> None:
        tokenizer = _FakeTokenizer(return_mapping=True)
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        ids = model._render(
            ModelRequest(messages=[TextMessage("hi", "text/x-user-message")])
        )

        assert ids.ndim == 2

    def test_render_accepts_batch_encoding(self) -> None:
        tokenizer = _FakeBatchEncodingTokenizer()
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        ids = model._render(
            ModelRequest(messages=[TextMessage("hi", "text/x-user-message")])
        )

        assert ids.ndim == 2

    def test_render_preserves_batch_encoding_attention_mask(self) -> None:
        tokenizer = _FakeBatchEncodingTokenizer()
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))

        rendered = model._render_prompt(
            ModelRequest(messages=[TextMessage("hi", "text/x-user-message")])
        )

        assert rendered.attention_mask is not None
        assert rendered.attention_mask.ndim == 2

    def test_render_inlines_tool_schema_when_template_rejects_tools(self) -> None:
        tokenizer = _FakeTokenizer(fail_on_tools=True)
        model = SelfHostedModel(provider=_FakeProvider(tokenizer))
        model._render(
            ModelRequest(
                messages=[TextMessage("hi", "text/x-user-message")],
                system="sys",
                tools=[_FakeTool()],
            )
        )
        retry_messages = tokenizer.calls[1][0]
        assert "You have access" in str(retry_messages[0]["content"])
        assert "tools" not in tokenizer.calls[1][1]

    @pytest.mark.anyio
    async def test_stream_emits_text_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = MultipartMessage(
            (
                TextMessage("hello", "text/plain"),
                MultipartMessage((), "multipart/x-tool-call"),
            ),
            "multipart/x-model-message",
        )

        async def fake_buffer(self: SelfHostedModel, request: ModelRequest) -> object:
            del self, request
            return type("Resp", (), {"content": response})()

        monkeypatch.setattr(SelfHostedModel, "buffer", fake_buffer)
        seen: list[str] = []
        resp = await SelfHostedModel(provider=_FakeProvider(_FakeTokenizer())).stream(
            ModelRequest(messages=[]),
            on_text=seen.append,
        )
        assert resp.content is response
        assert seen == ["hello"]

    @pytest.mark.anyio
    async def test_buffer_sends_attention_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _CaptureProvider()
        model = SelfHostedModel(provider=provider)
        monkeypatch.setattr(
            model,
            "_render_prompt",
            Mock(
                return_value=_RenderedPrompt(
                    input_ids=torch.tensor([[1, 2]]),
                    attention_mask=None,
                )
            ),
        )

        await model.buffer(ModelRequest(messages=[]))

        attention_mask = provider.capture_model.kwargs["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)
        assert torch.equal(attention_mask, torch.ones((1, 2), dtype=torch.int64))
        assert not provider.capture_model.grad_enabled

    @pytest.mark.anyio
    async def test_buffer_uses_rendered_attention_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _CaptureProvider()
        model = SelfHostedModel(provider=provider)
        mask = torch.tensor([[1, 0]])
        monkeypatch.setattr(
            model,
            "_render_prompt",
            Mock(
                return_value=_RenderedPrompt(
                    input_ids=torch.tensor([[1, 2]]),
                    attention_mask=mask,
                )
            ),
        )

        await model.buffer(ModelRequest(messages=[]))

        assert provider.capture_model.kwargs["attention_mask"] is mask

    def test_generate_cache_is_disabled_on_mps(self) -> None:
        assert _disable_generate_cache("mps")
        assert _disable_generate_cache("mps:0")
        assert not _disable_generate_cache("cpu")
        assert not _disable_generate_cache("cuda:0")

    @pytest.mark.anyio
    async def test_buffer_offloads_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _CaptureProvider()
        model = SelfHostedModel(provider=provider)
        seen: list[object] = []

        async def fake_to_thread(
            fn: Callable[..., object], *args: object, **kwargs: object
        ) -> object:
            seen.append(fn)
            return fn(*args, **kwargs)

        monkeypatch.setattr(
            "sagent.providers.selfhosted.asyncio.to_thread",
            fake_to_thread,
        )
        monkeypatch.setattr(
            model,
            "_render_prompt",
            Mock(
                return_value=_RenderedPrompt(
                    input_ids=torch.tensor([[1, 2]]),
                    attention_mask=None,
                )
            ),
        )

        await model.buffer(ModelRequest(messages=[]))

        assert seen

    @pytest.mark.anyio
    async def test_buffer_logs_output_tokens_per_second(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = SelfHostedModel(provider=_CaptureProvider())
        monkeypatch.setattr(
            model,
            "_render_prompt",
            Mock(
                return_value=_RenderedPrompt(
                    input_ids=torch.tensor([[1, 2]]),
                    attention_mask=None,
                )
            ),
        )

        with caplog.at_level(logging.DEBUG):
            await model.buffer(ModelRequest(messages=[]))

        assert "output_tokens_per_sec=" in caplog.text


class TestToolCallInvalidBranches:
    def test_invalid_qwen_payloads(self) -> None:
        assert _parse_qwen_tool_call("[]") is None
        assert _parse_qwen_tool_call('{"name": "", "arguments": {}}') is None
        assert _parse_qwen_tool_call('{"name": "x", "arguments": [1]}') is None

    def test_invalid_deepseek_payloads(self) -> None:
        assert _parse_deepseek_tool_call("no function") is None
        assert _parse_deepseek_tool_call("function\nbash\n```json\n[]\n```") is None
        assert (
            _parse_deepseek_tool_call("function\nbash\n```json\nnot-json\n```") is None
        )
