"""Tests for ``providers.selfhosted``: tool-call parsing + chat-message builder.

Avoids loading torch/transformers by exercising only the pure helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import json

from sagent.lib.custom_json import MutableJSON
from sagent.providers.selfhosted import (
    SelfHostedModel,
    _attention_mask,
    _build_chat_messages,
    _context_window,
    _disable_generate_cache,
    _extract_tool_calls,
    _inline_tool_preamble,
    _input_ids,
    _parse_deepseek_tool_call,
    _parse_qwen_tool_call,
    _tool_allowed,
    _tool_preamble,
    _tool_schema,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


def test_context_window_top_level_max_position() -> None:
    config = cast(MutableJSON, {"max_position_embeddings": 8192})
    assert _context_window(config, default=1024) == 8192


def test_context_window_nested_text_config() -> None:
    config = cast(MutableJSON, {"text_config": {"max_position_embeddings": 4096}})
    assert _context_window(config, default=1024) == 4096


def test_context_window_falls_back_to_default() -> None:
    assert _context_window(cast(MutableJSON, {}), default=2048) == 2048


def test_context_window_non_integer_top_level_falls_back() -> None:
    config = cast(MutableJSON, {"max_position_embeddings": "weird"})
    assert _context_window(config, default=512) == 512


def test_extract_qwen_tool_call_basic() -> None:
    text = (
        'pre-text <tool_call>{"name": "Bash", "arguments": {"cmd": "ls"}}</tool_call>'
        " post"
    )
    calls, cleaned = _extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert dict(calls[0].args) == {"cmd": "ls"}
    # Tool call section is stripped from the cleaned text.
    assert "<tool_call>" not in cleaned


def test_extract_qwen_tool_call_no_args_key_defaults_to_empty() -> None:
    text = '<tool_call>{"name": "Echo"}</tool_call>'
    calls, _cleaned = _extract_tool_calls(text)
    assert len(calls) == 1
    assert dict(calls[0].args) == {}


def test_extract_qwen_malformed_preserved() -> None:
    text = "<tool_call>not json</tool_call>"
    calls, cleaned = _extract_tool_calls(text)
    assert calls == []
    # Malformed call is preserved verbatim in the output.
    assert "<tool_call>" in cleaned


def test_extract_qwen_unadvertised_tool_preserved() -> None:
    text = '<tool_call>{"name": "Forbidden", "arguments": {}}</tool_call>'
    calls, cleaned = _extract_tool_calls(text, allowed_tools={"Bash"})
    assert calls == []
    assert "<tool_call>" in cleaned


def test_extract_qwen_allowed_tools_case_insensitive() -> None:
    text = '<tool_call>{"name": "bash", "arguments": {}}</tool_call>'
    calls, _cleaned = _extract_tool_calls(text, allowed_tools={"Bash"})
    assert len(calls) == 1


def test_extract_no_tool_calls_returns_stripped_text() -> None:
    calls, cleaned = _extract_tool_calls("just text  ")
    assert calls == []
    assert cleaned == "just text"


def test_extract_deepseek_tool_call_block() -> None:
    text = (
        "intro\n"
        "<│tool▁calls▁begin│>"
        "<│tool▁call▁begin│>function\n"
        "Bash\n```json\n"
        '{"cmd": "ls"}'
        "\n```\n"
        "<│tool▁call▁end│>"
        "<│tool▁calls▁end│>"
    )
    calls, cleaned = _extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "Bash"
    assert dict(calls[0].args) == {"cmd": "ls"}
    assert "tool" not in cleaned or "calls" not in cleaned


def test_extract_deepseek_invalid_json_preserved() -> None:
    text = (
        "<│tool▁calls▁begin│>"
        "<│tool▁call▁begin│>function\n"
        "Bash\n```json\n"
        "not json"
        "\n```\n"
        "<│tool▁call▁end│>"
        "<│tool▁calls▁end│>"
    )
    calls, cleaned = _extract_tool_calls(text)
    assert calls == []
    # Malformed block stays in the output.
    assert "tool" in cleaned


def test_tool_allowed_none_means_all_allowed() -> None:
    tc = ToolCall(id="x", name="Bash", args={})
    assert _tool_allowed(tc, None) is True


def test_tool_allowed_case_insensitive_match() -> None:
    tc = ToolCall(id="x", name="bASh", args={})
    assert _tool_allowed(tc, {"Bash"}) is True


def test_tool_allowed_not_in_set() -> None:
    tc = ToolCall(id="x", name="Forbidden", args={})
    assert _tool_allowed(tc, {"Bash"}) is False


def test_build_chat_messages_user_only() -> None:
    req = ModelRequest(messages=[UserMessage(text="hi")])
    msgs = _build_chat_messages(req)
    assert msgs == [{"role": "user", "content": "hi"}]


def test_build_chat_messages_with_system() -> None:
    req = ModelRequest(messages=[UserMessage(text="hi")], system="be brief")
    msgs = _build_chat_messages(req)
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_build_chat_messages_assistant_with_tool_call_remaps_id() -> None:
    asst = AssistantMessage(
        text="thinking",
        tool_calls=(ToolCall(id="orig-1", name="Bash", args={"cmd": "ls"}),),
    )
    msgs = _build_chat_messages(ModelRequest(messages=[asst]))
    asst_msg = msgs[0]
    assert asst_msg["role"] == "assistant"
    tcs = cast(list[MutableJSON], asst_msg["tool_calls"])
    assert tcs[0]["id"] == "call_0"
    assert cast(MutableJSON, tcs[0]["function"])["name"] == "Bash"
    assert cast(MutableJSON, tcs[0]["function"])["arguments"] == json.dumps(
        {"cmd": "ls"}
    )


def test_build_chat_messages_tool_result_emits_tool_role() -> None:
    asst = AssistantMessage(tool_calls=(ToolCall(id="orig-1", name="N", args={}),))
    res = ToolResult(call_id="orig-1", content="done")
    msgs = _build_chat_messages(ModelRequest(messages=[asst, res]))
    tool_msg = msgs[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_0"
    assert tool_msg["content"] == "done"


def test_build_chat_messages_tool_result_error_prefixed() -> None:
    res = ToolResult(call_id="x", content="oops", is_error=True)
    msgs = _build_chat_messages(ModelRequest(messages=[res]))
    assert msgs[-1]["content"] == "[Error] oops"


def test_inline_tool_preamble_no_tools_noop() -> None:
    msgs: list[MutableJSON] = [{"role": "user", "content": "hi"}]
    _inline_tool_preamble(msgs, [])
    assert msgs == [{"role": "user", "content": "hi"}]


def test_inline_tool_preamble_no_system_prepends_one() -> None:
    msgs: list[MutableJSON] = [{"role": "user", "content": "hi"}]

    class _StubTool:
        name: str = "Bash"
        description: str = "shell"
        directive_schema: Mapping[str, object] = MappingProxyType({"type": "object"})

    _inline_tool_preamble(
        msgs,
        cast("list[Tool]", [_StubTool()]),
    )
    assert msgs[0]["role"] == "system"
    assert "tool_call" in str(msgs[0]["content"])


def test_inline_tool_preamble_extends_existing_system_message() -> None:
    msgs: list[MutableJSON] = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "hi"},
    ]

    class _StubTool:
        name: str = "Bash"
        description: str = "shell"
        directive_schema: Mapping[str, object] = MappingProxyType({"type": "object"})

    _inline_tool_preamble(msgs, cast("list[Tool]", [_StubTool()]))
    assert "existing system" in str(msgs[0]["content"])
    assert "tool_call" in str(msgs[0]["content"])


class _StubBash:
    name: str = "Bash"
    description: str = "Run shell"
    directive_schema: Mapping[str, object] = MappingProxyType({"type": "object"})


def test_tool_schema_wraps_as_function() -> None:
    schema = _tool_schema(cast("Tool", _StubBash()))
    assert schema["type"] == "function"
    func = cast(MutableJSON, schema["function"])
    assert func["name"] == "Bash"
    assert func["description"] == "Run shell"


def test_tool_preamble_contains_marker_and_tool_name() -> None:
    out = _tool_preamble(cast("list[Tool]", [_StubBash()]))
    assert "<tool_call>" in out
    assert "Bash" in out


def test_parse_qwen_tool_call_valid() -> None:
    tc = _parse_qwen_tool_call('{"name": "Bash", "arguments": {"cmd": "ls"}}')
    assert tc is not None
    assert tc.name == "Bash"


def test_parse_qwen_tool_call_invalid_json_returns_none() -> None:
    assert _parse_qwen_tool_call("not json") is None


def test_parse_qwen_tool_call_non_dict_returns_none() -> None:
    assert _parse_qwen_tool_call('"just a string"') is None


def test_parse_qwen_tool_call_missing_name_returns_none() -> None:
    assert _parse_qwen_tool_call('{"arguments": {}}') is None


def test_parse_qwen_tool_call_non_dict_args_returns_none() -> None:
    assert _parse_qwen_tool_call('{"name": "Bash", "arguments": "x"}') is None


def test_parse_deepseek_tool_call_valid() -> None:
    raw = 'function\nBash\n```json\n{"cmd": "ls"}\n```'
    tc = _parse_deepseek_tool_call(raw)
    assert tc is not None
    assert tc.name == "Bash"
    assert dict(tc.args) == {"cmd": "ls"}


def test_parse_deepseek_tool_call_missing_name_returns_none() -> None:
    raw = '```json\n{"cmd": "ls"}\n```'
    assert _parse_deepseek_tool_call(raw) is None


def test_parse_deepseek_tool_call_invalid_json_returns_none() -> None:
    raw = "function\nBash\n```json\nnot json\n```"
    assert _parse_deepseek_tool_call(raw) is None


def test_parse_deepseek_tool_call_non_dict_args_returns_none() -> None:
    raw = 'function\nBash\n```json\n"x"\n```'
    assert _parse_deepseek_tool_call(raw) is None


def test_disable_generate_cache_mps_true() -> None:
    assert _disable_generate_cache("mps") is True
    assert _disable_generate_cache("mps:0") is True


def test_disable_generate_cache_cuda_false() -> None:
    assert _disable_generate_cache("cuda") is False
    assert _disable_generate_cache("cpu") is False


def test_input_ids_from_mapping_with_key() -> None:
    assert _input_ids({"input_ids": [1, 2, 3]}) == [1, 2, 3]


def test_input_ids_from_object_with_attr() -> None:
    class _Obj:
        def __init__(self) -> None:
            self.input_ids = [4, 5, 6]

    assert _input_ids(_Obj()) == [4, 5, 6]


def test_input_ids_from_nested_data_attr() -> None:
    class _Obj:
        def __init__(self) -> None:
            self.data = {"input_ids": [7]}

    assert _input_ids(_Obj()) == [7]


def test_input_ids_fallthrough_returns_value() -> None:
    sentinel = object()
    assert _input_ids(sentinel) is sentinel


def test_attention_mask_from_mapping() -> None:
    assert _attention_mask({"attention_mask": [1, 1]}) == [1, 1]


def test_attention_mask_absent_returns_none() -> None:
    # Plain object without ``attention_mask`` → None.
    assert _attention_mask(object()) is None


class _StubProvider:
    """Minimal provider stand-in to exercise ``SelfHostedModel`` properties."""

    hosted_max_request_tokens: int = 1234
    hosted_model_id: str = "stub/qwen"
    hosted_max_response_tokens: int = 567


def test_self_hosted_model_properties() -> None:
    # Structural-only stub; the model only reads ``hosted_*`` properties
    # here so the missing ``native_model`` / ``tokenizer`` are never touched.
    stub = _StubProvider()
    m = SelfHostedModel(provider=stub)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- partial protocol stub
    assert m.max_request_tokens == 1234
    assert m.model_id == "stub/qwen"
    assert m.max_response_tokens == 567
    assert m.supports_streaming is False
    assert m.supports_thinking is False
    assert m.supports_effort is True
    assert m.supports_cache_control is False
    assert m.supports_context_management is False
    assert m.supports_persistent_retry is False
    assert m.supports_account_auth is False
    # A local model has no provider-imposed image/wire caps -- all three are
    # the 0=unlimited sentinel (consistent: no borrowed pixel cap either).
    assert m.max_image_dim == 0
    assert m.max_image_bytes == 0
    assert m.max_request_bytes == 0
    assert m.approx_text_tokens("a" * 16) == 4
    assert m.is_context_overflow(RuntimeError("x")) is False
    assert m.is_retryable_provider_error(RuntimeError("x")) is False
    assert m.pricing.request == 0.0


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
