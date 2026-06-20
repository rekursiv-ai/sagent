"""Tests for ``providers.openai_sub``: builders, JWT helpers, credential I/O."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import base64
import io
import json
import time

import httpx
import openai
import pytest

from sagent.lib.custom_json import JSONValue
from sagent.providers import OpenAI, openai_sub
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    find_response_not_read,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.openai_sub import (
    OpenAISubscription,
    _build_input,
    _build_tool,
    _build_tool_result_item,
    _build_tools,
    _build_user_item,
    _consume_stream,
    _jwt_claim,
    _jwt_exp,
    _jwt_payload,
    _parse_tool_arguments,
    _subscription_profile,
)
from sagent.types.exceptions import AuthRefreshError, UserFacingError
from sagent.types.model import ModelRequest, StreamInterruptedError
from sagent.types.runtime import (
    AssistantMessage,
    BytesMessage,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


# Minimal ``Tool``-shaped stub for the builders (Protocol consumers).
class _StubTool:
    name: str = "Bash"
    tool_id: str = "application/x-tool-bash"
    description: str = "Run shell commands"
    directive_schema: Mapping[str, JSONValue] = MappingProxyType({"type": "object"})
    clearable_results: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


class _NeverYieldingStream:
    """Responses stream that stays open forever unless the provider closes it."""

    def __init__(self) -> None:
        self.closed = False
        self.entered = asyncio.Event()

    def __aiter__(self) -> _NeverYieldingStream:
        return self

    async def __anext__(self) -> object:
        self.entered.set()
        await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _DelayedStream:
    """Responses stream that yields events slowly but within the idle budget."""

    def __init__(self, events: list[object], *, delay_sec: float) -> None:
        self._events = events
        self._delay_sec = delay_sec

    def __aiter__(self) -> _DelayedStream:
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        await asyncio.sleep(self._delay_sec)
        return self._events.pop(0)


class _TextDeltaEvent:
    """Small stand-in for OpenAI's text-delta event class."""

    def __init__(self, delta: str) -> None:
        self.delta = delta


class _ReasoningDeltaEvent:
    """Small stand-in for OpenAI's reasoning-delta event classes."""

    def __init__(self, delta: str) -> None:
        self.delta = delta


class _CompletedEvent:
    """Small stand-in for OpenAI's completed event class."""

    def __init__(self) -> None:
        self.response = _CompletedResponse()


class _CompletedResponse:
    """Small stand-in for OpenAI's completed response payload."""

    id: str = "resp_123"
    status: str = "completed"
    usage: object | None = None


class _ResponseErrorEvent:
    """Small stand-in for OpenAI's stream error event."""

    code: str = "rate_limit"
    message: str = "too many requests"
    param: str = "input"


class _FailedEvent:
    """Small stand-in for OpenAI's failed terminal event."""

    def __init__(self) -> None:
        self.response = _FailedResponse()


class _FailedResponse:
    """Small stand-in for a failed OpenAI response payload."""

    id: str = "resp_failed"
    status: str = "failed"
    error = type(
        "Error",
        (),
        {"code": "server_error", "message": "backend failed"},
    )()
    incomplete_details = None


class _IncompleteEvent:
    """Small stand-in for OpenAI's incomplete terminal event."""

    def __init__(self) -> None:
        self.response = _IncompleteResponse()


class _IncompleteResponse:
    """Small stand-in for an incomplete OpenAI response payload."""

    id: str = "resp_incomplete"
    status: str = "incomplete"
    error = None
    incomplete_details = type("Incomplete", (), {"reason": "max_output_tokens"})()


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


# 1×1 transparent PNG -- smallest valid PNG ``image_lib.resize`` will accept.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAen"
    b"63NgAAAAASUVORK5CYII="
)


def test_build_input_preserves_user_image_attachment() -> None:
    """User-side image attachments survive as Responses ``input_image`` blocks.

    The subscription path historically dropped ``attachments`` because the
    builder mapped ``UserMessage`` to a bare-string ``content``; vision turns
    crossing the Codex backend silently regressed.
    """
    req = _stub_request_messages(
        UserMessage(
            text="what is this?",
            attachments=(BytesMessage(data=_TINY_PNG, descriptor="image/png"),),
        ),
    )
    items = _items_as_list(req)
    assert len(items) == 1
    content = items[0]["content"]
    assert isinstance(content, list)
    types = [b.get("type") for b in cast(list[Mapping[str, object]], content)]
    assert types == ["input_text", "input_image"]
    image_block = cast(list[Mapping[str, object]], content)[1]
    image_url = cast(str, image_block["image_url"])
    assert image_url.startswith("data:image/")
    assert ";base64," in image_url


def test_build_user_item_text_only_keeps_bare_string_content() -> None:
    """No attachments → no allocation overhead, simple ``content=str`` shape."""
    item = _build_user_item(
        UserMessage(text="hi"), max_image_dim=2048, max_image_bytes=20 * 1024 * 1024
    )
    assert item == {"role": "user", "content": "hi"}


def test_build_user_item_drops_non_image_attachment_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PDF + other non-image attachments are skipped with a warning.

    The Responses API has no analogue of Anthropic's PDF block; opaque drop
    would silently lose user intent, so the path logs each skipped descriptor.
    """
    with caplog.at_level("WARNING", logger="sagent.providers.openai_sub"):
        item = _build_user_item(
            UserMessage(
                text="see attached",
                attachments=(BytesMessage(data=b"%PDF", descriptor="application/pdf"),),
            ),
            max_image_dim=2048,
            max_image_bytes=20 * 1024 * 1024,
        )
    assert item == {"role": "user", "content": "see attached"}
    assert any("application/pdf" in r.message for r in caplog.records)


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
    assert call_item["status"] == "completed"


def test_build_input_tool_result_pair_matches_call_id() -> None:
    asst = AssistantMessage(tool_calls=(ToolCall(id="ext-1", name="N", args={}),))
    res = ToolResult(call_id="ext-1", content="done")
    items = _items_as_list(_stub_request_messages(asst, res))
    out_item = items[-1]
    assert out_item["type"] == "function_call_output"
    assert out_item["call_id"] == "fc_0"
    assert out_item["output"] == "done"
    assert out_item["status"] == "completed"


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


def test_login_manual_advertises_localhost_redirect_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual login must send the ``localhost`` redirect_uri Hydra allow-lists.

    ``127.0.0.1`` is rejected with ``authorize_hydra_invalid_request``.
    """
    monkeypatch.setattr(openai_sub, "DEFAULT_CREDENTIALS_PATH", tmp_path / "auth.json")
    access = _make_jwt(
        {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-123",
                "chatgpt_plan_type": "pro",
            },
        }
    )
    token_resp = MagicMock()
    token_resp.status_code = 200
    token_resp.json.return_value = {
        "access_token": access,
        "refresh_token": "refresh-xyz",
        "expires_in": 3600,
    }
    captured: dict[str, object] = {}

    def _post(_url: str, *, data: dict[str, str], **_kw: object) -> MagicMock:
        captured["redirect_uri"] = data["redirect_uri"]
        return token_resp

    http_client = MagicMock()
    http_client.__enter__.return_value.post.side_effect = _post

    out = io.StringIO()
    with (
        patch(
            "sagent.providers.openai_sub.httpx.Client",
            return_value=http_client,
        ),
        patch.object(
            openai_sub,
            "parse_manual_auth_code",
            return_value="the-code",
        ),
        patch("builtins.input", return_value="the-code#state"),
    ):
        OpenAISubscription.login(out, manual=True)

    printed = out.getvalue()
    assert "localhost" in printed
    assert "127.0.0.1" not in printed
    assert captured["redirect_uri"] == "http://localhost:1455/auth/callback"


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
_FRESH_REFRESH = "rt-fresh"
_STALE_REFRESH = "rt-stale"


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


def test_subscription_default_model_inherits_from_openai() -> None:
    """``OpenAISubscription`` defers to ``OpenAI.DEFAULT_MODEL``."""
    assert OpenAISubscription.DEFAULT_MODEL == OpenAI.DEFAULT_MODEL


def test_subscription_default_utility_model_inherits_from_openai() -> None:
    """``OpenAISubscription`` defers to ``OpenAI.DEFAULT_UTILITY_MODEL``."""
    assert OpenAISubscription.DEFAULT_UTILITY_MODEL == OpenAI.DEFAULT_UTILITY_MODEL


def test_subscription_utility_model_uses_utility_default() -> None:
    m = _make_provider().utility_model()
    assert m.model_id == OpenAISubscription.DEFAULT_UTILITY_MODEL


def test_subscription_model_clamps_against_wire_contract() -> None:
    m = _make_provider().model("gpt-5.5")
    # KNOWN_MODELS is clamped via ``_subscription_profile``.
    assert m.max_request_tokens == 272_000
    assert m.max_response_tokens == 32_000


def test_subscription_model_supports_thinking_via_reasoning_effort() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.supports_thinking is True
    assert m.supports_account_auth is True


def test_subscription_valid_service_tiers_priority_only() -> None:
    # Codex ``/fast`` slash command sets service_tier="priority"; the
    # subscription endpoint accepts no other values.
    m = _make_provider().model("gpt-5.5")
    assert m.valid_service_tiers == ("priority",)


def test_subscription_valid_latency_modes_fast() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.valid_latency_modes == ("fast",)


@pytest.mark.anyio
async def test_subscription_close_releases_sdk_and_http_client() -> None:
    """``close`` tears down BOTH the provider's SDK and the HTTP client.

    The base ``OpenAICompatModel.close`` closes the shared HTTP client;
    the subscription path additionally has an ``AsyncOpenAI`` SDK owned
    by the *provider*, so the model's override delegates to
    ``provider.close_sdk()``. Regression guard for the partial-teardown
    leak the ``Model.close`` contract surfaced.
    """
    provider = _make_provider()
    model = provider.model("gpt-5.5")
    sdk = MagicMock()
    sdk.close = AsyncMock()
    provider._sdk = sdk
    provider._sdk_token = "dummy-token"  # noqa: S105 -- test fixture, not a secret

    await model.close()

    sdk.close.assert_awaited_once()
    assert provider._sdk is None
    assert provider._sdk_token is None


def test_subscription_fast_latency_resolves_to_priority_tier() -> None:
    m = _make_provider().model("gpt-5.5")
    tier = m.effective_service_tier(
        ModelRequest(messages=[UserMessage(text="x")], latency="fast")
    )
    assert tier == "priority"


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


async def _reasoning_effort_for(request: ModelRequest) -> str:
    provider = _make_provider()
    sdk = MagicMock()
    sdk.responses = MagicMock()
    sdk.responses.create = AsyncMock(
        return_value=_DelayedStream([_CompletedEvent()], delay_sec=0.0)
    )
    with (
        patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)),
        patch(
            "sagent.providers.openai_sub.oai_responses.ResponseCompletedEvent",
            _CompletedEvent,
        ),
    ):
        model = provider.model("gpt-5.5")
        await model.stream(request)
    await_args = sdk.responses.create.await_args
    assert await_args is not None
    reasoning = await_args.kwargs["reasoning"]
    return str(reasoning.effort)


@pytest.mark.anyio
async def test_subscription_stream_maps_adaptive_thinking_to_reasoning_effort() -> None:
    effort = await _reasoning_effort_for(
        ModelRequest(messages=[UserMessage(text="hi")], thinking="adaptive")
    )
    assert effort == "medium"


@pytest.mark.anyio
async def test_subscription_stream_maps_sagent_max_effort_to_openai_high() -> None:
    effort = await _reasoning_effort_for(
        ModelRequest(messages=[UserMessage(text="hi")], effort="max")
    )
    assert effort == "high"


async def _reasoning_for(request: ModelRequest) -> object:
    """Return the ``reasoning`` object passed to ``responses.create``."""
    provider = _make_provider()
    sdk = MagicMock()
    sdk.responses = MagicMock()
    sdk.responses.create = AsyncMock(
        return_value=_DelayedStream([_CompletedEvent()], delay_sec=0.0)
    )
    with (
        patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)),
        patch(
            "sagent.providers.openai_sub.oai_responses.ResponseCompletedEvent",
            _CompletedEvent,
        ),
    ):
        model = provider.model("gpt-5.5")
        await model.stream(request)
    await_args = sdk.responses.create.await_args
    assert await_args is not None
    return await_args.kwargs["reasoning"]


@pytest.mark.anyio
async def test_subscription_stream_requests_reasoning_summary_when_thinking() -> None:
    """Thinking on must request a reasoning summary, else no text streams.

    The OpenAI Responses API only emits reasoning-text deltas when
    ``reasoning.summary`` is set; without it the model reasons silently
    and ``on_thinking`` never fires. Requesting thinking must therefore
    set ``summary`` so the reasoning surface is actually populated.
    """
    reasoning = await _reasoning_for(
        ModelRequest(messages=[UserMessage(text="hi")], thinking="adaptive")
    )
    assert getattr(reasoning, "summary", None) == "auto"


@pytest.mark.anyio
async def test_subscription_stream_omits_reasoning_summary_without_thinking() -> None:
    """No thinking requested -> no reasoning object (and no summary)."""
    reasoning = await _reasoning_for(
        ModelRequest(messages=[UserMessage(text="hi")], thinking=None)
    )
    assert getattr(reasoning, "summary", None) is None


class TestHandleAuthError:
    """After ``/login`` writes fresh creds, the live provider must reload.

    Bug: ``do_login`` (``repl/run_repl.py``) calls
    ``provider.handle_auth_error()`` to hot-reload in-memory tokens
    from disk after a successful re-auth. Without this hook on
    :class:`OpenAISubscription`, the live instance keeps its stale
    ``_access_token`` / ``_refresh_token`` after ``/login``; the next
    model call either sends the stale access token (-> API 401) or
    posts the stale refresh token (-> 400 ->
    :class:`AuthRefreshError`). Either way ``/login`` is a no-op until
    process restart.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk(self, tmp_path: Path) -> None:
        """Fresh creds on disk replace stale in-memory creds."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=fresh_access,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        with patch(
            "sagent.providers.openai_sub.DEFAULT_CREDENTIALS_PATH",
            cred_file,
        ):
            await provider.handle_auth_error()
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_same(self, tmp_path: Path) -> None:
        """Disk matches in-memory -> fall through to network refresh."""
        in_memory_access = _make_jwt({"exp": 0.0})
        cred_file = tmp_path / "auth.json"
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=in_memory_access,
                refresh_token=_STALE_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=0.0,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=in_memory_access,
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "access_token": refreshed_access,
                "refresh_token": _FRESH_REFRESH,
                "expires_in": 3600,
            },
        )
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_missing(self, tmp_path: Path) -> None:
        """Missing disk file -> caught -> fall through to network refresh."""
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        mock_resp = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "access_token": refreshed_access,
                "refresh_token": _FRESH_REFRESH,
                "expires_in": 3600,
            },
        )
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub.DEFAULT_CREDENTIALS_PATH",
                tmp_path / "does_not_exist.json",
            ),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH


class TestEnsureValidRace:
    """``_ensure_valid`` must reload disk before refreshing.

    Multi-process race: process A's token expires; A refreshes,
    rotating the refresh_token and saving fresh creds to disk.
    Process B's in-memory refresh_token is now revoked. When B's
    token expires, ``_ensure_valid`` must check disk first; if disk
    has fresher creds, adopt them without posting B's stale
    refresh_token (which would 400). Mirrors ``handle_auth_error``'s
    disk-first pattern.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk_when_sibling_refreshed(
        self, tmp_path: Path
    ) -> None:
        """Fresh disk creds preempt the network refresh entirely."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token=fresh_access,
                refresh_token=_FRESH_REFRESH,
                account_id=_FAKE_ACCOUNT,
                expires_at=time.time() + 3600,
            ),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            account_id=_FAKE_ACCOUNT,
            expires_at=time.time() - 100,
        )

        # If ``_refresh`` is reached, the test fails fast -- the disk
        # reload should preempt it.
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=AssertionError(
                "_ensure_valid called _refresh instead of reloading from disk"
            ),
        )
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub.DEFAULT_CREDENTIALS_PATH",
                cred_file,
            ),
        ):
            token = await provider._ensure_valid()

        assert token == fresh_access
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH


class TestStreamResponseNotRead:
    """Unread SDK streaming errors must not leak raw httpx exceptions."""

    @pytest.mark.anyio
    async def test_create_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        sdk.responses.create = AsyncMock(side_effect=httpx.ResponseNotRead())

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert not isinstance(raised.value, httpx.ResponseNotRead)
        assert "OpenAI subscription streaming request failed" in str(raised.value)

    @pytest.mark.anyio
    async def test_wrapped_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        err = RuntimeError("SDK failed")
        err.__cause__ = httpx.ResponseNotRead()
        sdk.responses.create = AsyncMock(side_effect=err)

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert "OpenAI subscription streaming request failed" in str(raised.value)

    def test_response_not_read_context_is_detected(self) -> None:
        err = RuntimeError("SDK failed")
        err.__context__ = httpx.ResponseNotRead()

        assert find_response_not_read(err) is not None


class TestStreamAuthRetry:
    """Mid-call 401 must trigger ``handle_auth_error`` + one-shot retry.

    Bug: a stale in-memory bearer (rotated server-side between our
    local expiry check and the request arriving) surfaces as
    ``openai.AuthenticationError`` from ``sdk.responses.create``.
    Without the catch + reload + retry wrapper, the error bubbles
    raw to the user; with it, the runtime reloads disk creds (or
    force-refreshes) and retries once before giving up.
    """

    @pytest.mark.anyio
    async def test_auth_error_triggers_reload_and_retries_once(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex")
        response = httpx.Response(401, request=request)
        auth_err = openai.AuthenticationError(
            "Unauthorized", response=response, body=None
        )
        # Return DIFFERENT SDKs from get_sdk so the test can prove the
        # retry call landed on a freshly-built SDK (with a rotated
        # bearer baked into default_headers), not the cached one.
        sdk_stale = MagicMock()
        sdk_stale.responses = MagicMock()
        sdk_stale.responses.create = AsyncMock(side_effect=auth_err)
        sdk_fresh = MagicMock()
        sdk_fresh.responses = MagicMock()
        sdk_fresh.responses.create = AsyncMock(side_effect=auth_err)
        get_sdk = AsyncMock(side_effect=[sdk_stale, sdk_fresh])
        with (
            patch.object(provider, "get_sdk", get_sdk),
            patch.object(provider, "handle_auth_error", AsyncMock()) as ha,
        ):
            model = provider.model("gpt-5.5")
            req = ModelRequest(messages=[UserMessage(text="hi")])
            with pytest.raises(openai.AuthenticationError):
                await model.stream(req)
        ha.assert_awaited_once()
        # Each SDK saw exactly one create call: the original attempt on
        # the stale SDK, the retry on the freshly-fetched one.
        assert sdk_stale.responses.create.await_count == 1
        assert sdk_fresh.responses.create.await_count == 1
        assert get_sdk.await_count == 2


class TestStreamIdleTimeout:
    """Silent Responses streams must not make the runtime wait forever."""

    @pytest.mark.anyio
    async def test_silent_stream_times_out_and_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = _NeverYieldingStream()
        monkeypatch.setattr(
            "sagent.providers.openai_sub._STREAM_IDLE_TIMEOUT",
            0.01,
        )

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                _consume_stream(
                    stream,
                    pricing=Pricing(),
                    publish=None,
                ),
                timeout=0.2,
            )

        assert stream.closed is True

    @pytest.mark.anyio
    async def test_stream_events_reschedule_idle_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub._STREAM_IDLE_TIMEOUT",
            0.05,
        )
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        stream = _DelayedStream(
            [_TextDeltaEvent("he"), _TextDeltaEvent("llo"), _CompletedEvent()],
            delay_sec=0.03,
        )

        response = await asyncio.wait_for(
            _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            ),
            timeout=0.2,
        )

        assert response.message.text == "hello"
        assert response.message_id == "resp_123"

    @pytest.mark.anyio
    async def test_truncated_stream_raises_interrupted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        stream = _DelayedStream([_TextDeltaEvent("partial")], delay_sec=0.0)

        with pytest.raises(StreamInterruptedError) as raised:
            await _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            )

        assert raised.value.response.message.text == "partial"
        assert raised.value.response.stop_reason == "model_finished"

    @pytest.mark.anyio
    async def test_response_error_event_is_user_facing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseErrorEvent",
            _ResponseErrorEvent,
        )
        stream = _DelayedStream([_ResponseErrorEvent()], delay_sec=0.0)

        with pytest.raises(UserFacingError) as raised:
            await _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            )

        msg = str(raised.value)
        assert "too many requests" in msg
        assert "code=rate_limit" in msg
        assert "param=input" in msg

    @pytest.mark.anyio
    async def test_response_failed_event_is_user_facing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseFailedEvent",
            _FailedEvent,
        )
        stream = _DelayedStream([_FailedEvent()], delay_sec=0.0)

        with pytest.raises(UserFacingError) as raised:
            await _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            )

        msg = str(raised.value)
        assert "status=failed" in msg
        assert "response_id=resp_failed" in msg
        assert "code=server_error" in msg
        assert "backend failed" in msg

    @pytest.mark.anyio
    async def test_response_incomplete_event_is_user_facing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseIncompleteEvent",
            _IncompleteEvent,
        )
        stream = _DelayedStream([_IncompleteEvent()], delay_sec=0.0)

        with pytest.raises(UserFacingError) as raised:
            await _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            )

        msg = str(raised.value)
        assert "status=incomplete" in msg
        assert "response_id=resp_incomplete" in msg
        assert "reason=max_output_tokens" in msg

    @pytest.mark.anyio
    async def test_stream_routes_reasoning_deltas_to_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseReasoningTextDeltaEvent",
            _ReasoningDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseReasoningSummaryTextDeltaEvent",
            _ReasoningDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai_sub.oai_responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        thinking_chunks: list[str] = []

        def _sink(ev: RuntimeEvent) -> None:
            if isinstance(ev, ModelResponseThinking):
                thinking_chunks.append(ev.text)

        stream = _DelayedStream(
            [
                _ReasoningDeltaEvent("think "),
                _TextDeltaEvent("answer"),
                _ReasoningDeltaEvent("more"),
                _CompletedEvent(),
            ],
            delay_sec=0.0,
        )

        response = await _consume_stream(
            stream,
            pricing=Pricing(),
            publish=_sink,
        )

        assert thinking_chunks == ["think ", "more"]
        assert response.message.text == "answer"
        assert response.message.thinking_blocks == (
            {"type": "reasoning", "text": "think more"},
        )

    @pytest.mark.anyio
    async def test_cancelled_stream_closes(self) -> None:
        stream = _NeverYieldingStream()
        task = asyncio.create_task(
            _consume_stream(
                stream,
                pricing=Pricing(),
                publish=None,
            ),
        )
        await asyncio.wait_for(stream.entered.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert stream.closed is True


class TestRefreshErrors:
    """``_refresh`` must surface auth failures as :class:`AuthRefreshError`.

    Codex's ``auth.openai.com/oauth/token`` endpoint returns 400/401 when
    the refresh token has been rotated, revoked, or expired. The raw
    ``httpx.HTTPStatusError`` is useless to the user -- it leaks the
    OAuth URL into the terminal. Convert it into a typed, user-facing
    error with actionable text so the renderer can present "Run /login"
    without dumping a traceback.
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [400, 401])
    async def test_refresh_4xx_raises_auth_refresh_error(self, status: int) -> None:
        """400/401 on the token endpoint -> :class:`AuthRefreshError`."""
        provider = _make_provider(expires_at=0.0)
        request = httpx.Request("POST", "https://auth.openai.com/oauth/token")
        response = httpx.Response(status, request=request, text="invalid_grant")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(AuthRefreshError) as excinfo,
        ):
            await provider._refresh()

        msg = str(excinfo.value)
        assert "/login" in msg, (
            f"AuthRefreshError must guide the user to /login; got: {msg!r}"
        )
        assert "HTTPStatusError" not in msg

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_refresh_5xx_does_not_raise_auth_refresh_error(
        self, status: int
    ) -> None:
        """Server-side failures bubble as plain ``httpx.HTTPStatusError``."""
        provider = _make_provider(expires_at=0.0)
        request = httpx.Request("POST", "https://auth.openai.com/oauth/token")
        response = httpx.Response(status, request=request, text="upstream down")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider._refresh()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
