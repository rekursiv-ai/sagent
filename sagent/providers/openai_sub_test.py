"""Tests for ``providers.openai_sub``: builders, JWT helpers, credential I/O."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

import base64
import json

import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.custom_types import ModelRequest
from sagent.lib.json import JSONValue
from sagent.providers import OpenAI
from sagent.providers.lib.cost import ModelProfile
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.openai_sub import (
    OpenAISubscription,
    _build_input,
    _build_tool,
    _build_tool_result_item,
    _build_tools,
    _jwt_claim,
    _jwt_exp,
    _jwt_payload,
    _parse_tool_arguments,
    _subscription_profile,
)


# Minimal ``Tool``-shaped stub for the builders (Protocol consumers).
class _StubTool:
    name: str = "Bash"
    tool_id: str = "application/x-tool-bash"
    description: str = "Run shell commands"
    directive_schema: Mapping[str, JSONValue] = MappingProxyType({"type": "object"})
    supports_microcompaction: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


def _stub_request_messages(
    *items: UserMessage | AssistantMessage | ToolResult,
) -> ModelRequest:
    # The builders only iterate ``request.messages``; build a stand-in.
    return ModelRequest(messages=list(items))


def test_subscription_profile_clamps_request_tokens() -> None:
    p = ModelProfile(max_request_tokens=1_000_000, max_response_tokens=1_000_000)
    clamped = _subscription_profile(p)
    # Wire contract is 272_000 / 32_000 -- see openai_sub._SUBSCRIPTION_MAX_*.
    assert clamped.max_request_tokens == 272_000
    assert clamped.max_response_tokens == 32_000
    assert clamped.pricing == p.pricing


def test_subscription_profile_keeps_small_limits() -> None:
    p = ModelProfile(max_request_tokens=100_000, max_response_tokens=10_000)
    clamped = _subscription_profile(p)
    assert clamped.max_request_tokens == 100_000
    assert clamped.max_response_tokens == 10_000


def test_build_tool_shape() -> None:
    out = _build_tool(_StubTool())
    assert out["type"] == "function"
    assert out["name"] == "Bash"
    assert out.get("description") == "Run shell commands"
    assert out["parameters"] == {"type": "object"}
    assert out["strict"] is None


def test_build_tools_list_passthrough() -> None:
    out = _build_tools([_StubTool(), _StubTool()])
    assert len(out) == 2
    assert all(o["name"] == "Bash" for o in out)


def _items_as_list(req: ModelRequest) -> list[Mapping[str, object]]:
    """Cast the Responses-API TypedDict union output to a uniform mapping list."""
    return cast(list[Mapping[str, object]], _build_input(req))


def test_build_input_user_message() -> None:
    items = _items_as_list(_stub_request_messages(UserMessage(text="hello")))
    assert items == [{"role": "user", "content": "hello"}]


def test_build_input_assistant_text_and_tool_call() -> None:
    asst = AssistantMessage(
        text="thinking",
        tool_calls=(ToolCall(id="ext-1", name="Bash", args={"cmd": "ls"}),),
    )
    items = _items_as_list(_stub_request_messages(asst))
    assert items[0] == {"role": "assistant", "content": "thinking"}
    call_item = items[1]
    assert call_item["type"] == "function_call"
    assert call_item["call_id"] == "fc_0"
    assert call_item["id"] == "fc_0"
    assert call_item["name"] == "Bash"
    assert call_item["arguments"] == json.dumps({"cmd": "ls"})


def test_build_input_tool_result_pair_matches_call_id() -> None:
    asst = AssistantMessage(tool_calls=(ToolCall(id="ext-1", name="N", args={}),))
    res = ToolResult(call_id="ext-1", content="done")
    items = _items_as_list(_stub_request_messages(asst, res))
    out_item = items[-1]
    assert out_item["type"] == "function_call_output"
    assert out_item["call_id"] == "fc_0"
    assert out_item["output"] == "done"


def test_build_tool_result_item_error_prefixes_marker() -> None:
    res = ToolResult(call_id="cid", content="boom", is_error=True)
    out = _build_tool_result_item(res, IdRemapper("fc_"))
    assert out["output"] == "[Error] boom"


def test_build_assistant_items_no_text_only_tool_calls() -> None:
    """Empty assistant text + tool call → single function_call item."""
    asst = AssistantMessage(tool_calls=(ToolCall(id="x", name="N", args={}),))
    items = _items_as_list(_stub_request_messages(asst))
    assert len(items) == 1
    assert items[0]["type"] == "function_call"


def test_parse_tool_arguments_prefers_done_when_set() -> None:
    out = _parse_tool_arguments(
        delta_args='{"a": 1}',
        done_args='{"b": 2}',
        tool_name="N",
        call_id="cid",
    )
    assert out == {"b": 2}


def test_parse_tool_arguments_falls_back_to_delta() -> None:
    out = _parse_tool_arguments(
        delta_args='{"a": 1}',
        done_args="",
        tool_name="N",
        call_id="cid",
    )
    assert out == {"a": 1}


def test_parse_tool_arguments_both_empty_returns_empty_dict() -> None:
    out = _parse_tool_arguments("", "", tool_name="N", call_id="cid")
    assert out == {}


def test_parse_tool_arguments_invalid_json_skipped() -> None:
    out = _parse_tool_arguments(
        delta_args="not json",
        done_args='{"x": 1}',
        tool_name="N",
        call_id="cid",
    )
    assert out == {"x": 1}


def test_parse_tool_arguments_done_empty_object_keeps_delta() -> None:
    # ``done`` is parsed but falsy; truthy ``delta`` wins via ``if done``.
    out = _parse_tool_arguments(
        delta_args='{"a": 1}',
        done_args="{}",
        tool_name="N",
        call_id="cid",
    )
    assert out == {"a": 1}


def _make_jwt(payload: dict[str, object]) -> str:
    header_b = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=")
    body_b = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header_b.decode()}.{body_b.decode()}.sig"


def test_jwt_payload_round_trip() -> None:
    token = _make_jwt({"exp": 1234, "https://api.openai.com/auth": {"x": "y"}})
    assert _jwt_payload(token).get("exp") == 1234


def test_jwt_payload_short_token_returns_empty() -> None:
    assert _jwt_payload("only_one_part") == {}


def test_jwt_payload_invalid_base64_returns_empty() -> None:
    # Crafted body that decodes to non-JSON.
    bad = "h." + base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode() + ".s"
    assert _jwt_payload(bad) == {}


def test_jwt_exp_extracts_value() -> None:
    token = _make_jwt({"exp": 9001})
    assert _jwt_exp(token) == 9001.0


def test_jwt_exp_missing_returns_zero() -> None:
    token = _make_jwt({})
    assert _jwt_exp(token) == 0.0


def test_jwt_claim_nested_value() -> None:
    token = _make_jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acc-123"}}
    )
    assert _jwt_claim(token, "https://api.openai.com/auth", "chatgpt_account_id") == (
        "acc-123"
    )


def test_jwt_claim_namespace_missing_returns_empty() -> None:
    token = _make_jwt({})
    assert _jwt_claim(token, "ns", "key") == ""


def _write_creds_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_creds(
    access: str, *, expires_at: float = 0.0
) -> OpenAISubscription.Credentials:
    return OpenAISubscription.Credentials(
        access_token=access,
        refresh_token=_FAKE_REFRESH,
        account_id=_FAKE_ACCOUNT,
        expires_at=expires_at,
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    access = _make_jwt({"exp": 1700.0})
    target = tmp_path / "auth.json"
    OpenAISubscription.save(_make_creds(access, expires_at=1700.0), path=target)
    loaded = OpenAISubscription.load(path=target)
    assert loaded["access_token"] == access
    assert loaded["refresh_token"] == _FAKE_REFRESH
    assert loaded["account_id"] == _FAKE_ACCOUNT
    # ``expires_at`` is recomputed from the JWT exp claim on load.
    assert loaded["expires_at"] == 1700.0


def test_save_preserves_existing_fields(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    _write_creds_file(target, {"keep_me": True, "tokens": {"old": "x"}})
    creds = _make_creds(_make_jwt({"exp": 0.0}))
    OpenAISubscription.save(creds, path=target)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["keep_me"] is True
    assert on_disk["auth_mode"] == "chatgpt"
    assert on_disk["tokens"]["access_token"] == creds["access_token"]
    # Old fields under ``tokens`` survive.
    assert on_disk["tokens"]["old"] == "x"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenAISubscription.load(path=tmp_path / "nope.json")


def test_save_writes_id_token_when_present(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    creds = _make_creds(_make_jwt({"exp": 0.0}))
    sentinel = "fake-jwt-marker"
    creds["id_token"] = sentinel
    OpenAISubscription.save(creds, path=target)
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["tokens"]["id_token"] == sentinel


# Test-only token values; not credentials. Held in module variables so the
# bandit-style S105/S106 rules don't fire on inline literals at constructor
# call sites.
_FAKE_ACCESS = ""
_FAKE_REFRESH = "rt-test"
_FAKE_ACCOUNT = "acc"


def _make_provider(*, expires_at: float = 9999.0) -> OpenAISubscription:
    return OpenAISubscription(
        access_token=_FAKE_ACCESS or _make_jwt({"exp": expires_at}),
        refresh_token=_FAKE_REFRESH,
        account_id=_FAKE_ACCOUNT,
        expires_at=expires_at,
    )


def test_subscription_from_key_returns_plain_openai() -> None:
    """``from_key`` delegates to the API-key ``OpenAI`` class."""
    p = OpenAISubscription.from_key("sk-test")
    assert isinstance(p, OpenAI)
    assert not isinstance(p, OpenAISubscription)


def test_subscription_model_unknown_id_raises() -> None:
    p = _make_provider()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-model")


def test_subscription_model_uses_default_when_unset() -> None:
    m = _make_provider().model()
    assert m.model_id == OpenAISubscription.DEFAULT_MODEL


def test_subscription_utility_model_uses_utility_default() -> None:
    m = _make_provider().utility_model()
    assert m.model_id == OpenAISubscription.DEFAULT_UTILITY_MODEL


def test_subscription_model_clamps_against_wire_contract() -> None:
    m = _make_provider().model("gpt-5.5")
    # KNOWN_MODELS is clamped via ``_subscription_profile``.
    assert m.max_request_tokens == 272_000
    assert m.max_response_tokens == 32_000


def test_subscription_model_overrides_supports_thinking_false() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.supports_thinking is False
    assert m.supports_account_auth is True


def test_subscription_context_overflow_detection() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_context_overflow(RuntimeError("context_length_exceeded")) is True
    assert m.is_context_overflow(RuntimeError("exceeds the context window")) is True
    assert m.is_context_overflow(RuntimeError("other failure")) is False


def test_subscription_retryable_provider_error() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_retryable_provider_error(RuntimeError("You can retry your request"))
    assert not m.is_retryable_provider_error(RuntimeError("permanent failure"))


def test_subscription_expired_property_true_when_past() -> None:
    p = _make_provider(expires_at=0.0)
    assert p.expired is True


def test_subscription_expired_property_false_when_far_future() -> None:
    p = _make_provider(expires_at=9_999_999_999.0)
    assert p.expired is False


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
