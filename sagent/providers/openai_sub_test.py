from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, patch

import base64
import io
import json
import time

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails

import pytest

from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import json_freeze
from sagent.lib.message import (
    get_directive,
    get_tool_name,
    tool_call_message,
)
from sagent.providers import openai_sub as os_
from sagent.providers.lib.cost import Pricing
from sagent.providers.openai_sub import (
    OpenAISubscription,
    _build_input,
    _build_tools,
    _consume_stream,
)


def _user(text: str) -> Message:
    return TextMessage(text, "text/x-user-message")


def _assistant(
    text: str = "",
    *,
    tool_calls: list[Message] | None = None,
) -> Message:
    parts: list[Message] = []
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    return MultipartMessage(tuple(parts), "multipart/x-model-message")


def _tool_result(queue_id: str, text: str) -> Message:
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            TextMessage(text, "text/plain"),
        ),
        "multipart/x-tool-result",
    )


def _resp_text(resp: ModelResponse) -> str | None:
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "text/plain" and isinstance(p, TextMessage):
            return p.content
    return None


def _resp_tool_calls(resp: ModelResponse) -> list[Message]:
    if not isinstance(resp.content, MultipartMessage):
        return []
    return [p for p in resp.content.content if p.descriptor == "multipart/x-tool-call"]


_AT = "test-at"
_RT = "test-rt"
_ACCT = "acct-123"


def _make_provider(**overrides: Any) -> OpenAISubscription:
    creds = cast(
        OpenAISubscription.Credentials,
        {
            "access_token": _AT,
            "refresh_token": _RT,
            "account_id": _ACCT,
            "expires_at": 1e12,
            **overrides,
        },
    )
    return OpenAISubscription.from_credentials(creds)


_DEFAULT_PRICING = Pricing(request=2.5, response=15.0, cache_read=0.25)


class _AsyncIterator:
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._i = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> object:
        if self._i >= len(self._items):
            raise StopAsyncIteration
        val = self._items[self._i]
        self._i += 1
        return val


class TestFromCredentials:
    def test_basic(self) -> None:
        p = _make_provider()
        assert p._access_token == _AT
        assert p._refresh_token == _RT
        assert p._account_id == _ACCT

    def test_missing_access_token(self) -> None:
        with pytest.raises(KeyError):
            OpenAISubscription.from_credentials(
                {"refresh_token": "rt", "account_id": "a", "expires_at": 0},  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type,missing-typed-dict-key] -- deliberately incomplete credentials
            )

    def test_missing_refresh_token(self) -> None:
        with pytest.raises(KeyError):
            OpenAISubscription.from_credentials(
                {"access_token": "at", "account_id": "a", "expires_at": 0},  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type,missing-typed-dict-key] -- deliberately incomplete credentials
            )

    def test_missing_account_id(self) -> None:
        with pytest.raises(KeyError):
            OpenAISubscription.from_credentials(
                {"access_token": "at", "refresh_token": "rt", "expires_at": 0},  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type,missing-typed-dict-key] -- deliberately incomplete credentials
            )

    def test_missing_expires_at(self) -> None:
        with pytest.raises(KeyError):
            OpenAISubscription.from_credentials(
                {"access_token": "at", "refresh_token": "rt", "account_id": "a"},  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type,missing-typed-dict-key] -- deliberately incomplete credentials
            )


class TestModel:
    def test_default_model_id(self) -> None:
        m = _make_provider().model()
        assert m.model_id == "gpt-5.5"

    def test_custom_model_id(self) -> None:
        m = _make_provider().model("gpt-5.4-mini")
        assert m.model_id == "gpt-5.4-mini"

    def test_default_max_request_tokens(self) -> None:
        m = _make_provider().model()
        assert m.max_request_tokens == 1_050_000

    def test_custom_max_request_tokens(self) -> None:
        m = _make_provider().model("gpt-5.4", max_request_tokens=64_000)
        assert m.max_request_tokens == 64_000

    def test_max_response_tokens(self) -> None:
        m = _make_provider().model()
        assert m.max_response_tokens == 128_000


class TestUtilityModel:
    def test_returns_default_utility(self) -> None:
        m = _make_provider().utility_model()
        assert m.model_id == "gpt-5.4-mini"


class TestModelProperties:
    def test_supports_streaming(self) -> None:
        assert _make_provider().model().supports_streaming is True

    def test_supports_thinking(self) -> None:
        assert _make_provider().model().supports_thinking is False

    def test_supports_effort_gpt5(self) -> None:
        assert _make_provider().model("gpt-5.4").supports_effort is True

    def test_supports_effort_o_series(self) -> None:
        assert _make_provider().model("o3-mini").supports_effort is True

    def test_no_effort_for_gpt4(self) -> None:
        assert _make_provider().model("gpt-4o").supports_effort is False

    def test_supports_cache_control(self) -> None:
        assert _make_provider().model().supports_cache_control is False

    def test_supports_context_management(self) -> None:
        assert _make_provider().model().supports_context_management is False

    def test_supports_persistent_retry(self) -> None:
        assert _make_provider().model().supports_persistent_retry is False

    def test_supports_account_auth(self) -> None:
        assert _make_provider().model().supports_account_auth is True

    @pytest.mark.anyio
    async def test_stream_omits_unsupported_request_knobs(self) -> None:
        provider = _make_provider()
        create = AsyncMock(
            return_value=_AsyncIterator(
                [
                    _mock_completed_event(
                        response_id="resp-knobs",
                        status="completed",
                        input_tokens=1,
                        output_tokens=1,
                    )
                ]
            )
        )
        provider._sdk = MagicMock(responses=MagicMock(create=create))
        provider._sdk_token = _AT
        request = ModelRequest(
            messages=[_user("hello")],
            max_response_tokens=123,
            temperature=0.25,
            effort="high",
        )

        await provider.model("gpt-5.4").stream(request)

        assert create.await_args is not None
        kwargs = create.await_args.kwargs
        assert "max_output_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert kwargs["reasoning"].effort == "high"


class TestExpired:
    def test_not_expired(self) -> None:
        p = _make_provider(expires_at=time.time() + 3600)
        assert not p.expired

    def test_expired(self) -> None:
        p = _make_provider(expires_at=time.time() - 1)
        assert p.expired

    def test_within_buffer(self) -> None:
        p = _make_provider(expires_at=time.time() + 100)
        assert p.expired


class TestRefresh:
    @pytest.mark.anyio
    async def test_refresh_updates_tokens(self, tmp_path: Path) -> None:
        p = _make_provider(expires_at=time.time() - 1)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "expires_in": 7200,
        }
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        auth_path = tmp_path / "auth.json"
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub._DEFAULT_PATH",
                auth_path,
            ),
        ):
            await p._refresh()

        assert p._access_token == "new-at"  # noqa: S105 -- test credential
        assert p._refresh_token == "new-rt"  # noqa: S105 -- test credential
        assert p._expires_at > time.time()

    @pytest.mark.anyio
    async def test_refresh_persists_to_disk(self, tmp_path: Path) -> None:
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": "old",
                        "refresh_token": "old-rt",
                        "account_id": "acct-123",
                    },
                }
            )
        )
        p = _make_provider(expires_at=time.time() - 1)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-at",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }
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
                "sagent.providers.openai_sub._DEFAULT_PATH",
                auth_path,
            ),
        ):
            await p._refresh()

        saved = json.loads(auth_path.read_text())
        assert saved["tokens"]["access_token"] == "new-at"  # noqa: S105 -- test credential
        assert saved["tokens"]["refresh_token"] == "new-rt"  # noqa: S105 -- test credential

    @pytest.mark.anyio
    async def test_ensure_valid_no_refresh_needed(self) -> None:
        p = _make_provider(expires_at=time.time() + 3600)
        token = await p._ensure_valid()
        assert token == _AT

    @pytest.mark.anyio
    async def test_ensure_valid_triggers_refresh(self, tmp_path: Path) -> None:
        p = _make_provider(expires_at=time.time() - 1)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "refreshed",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        auth_path = tmp_path / "auth.json"
        with (
            patch(
                "sagent.providers.openai_sub.httpx.AsyncClient",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub._DEFAULT_PATH",
                auth_path,
            ),
        ):
            token = await p._ensure_valid()
        assert token == "refreshed"  # noqa: S105 -- test credential


class TestIsContextOverflow:
    def test_context_length_exceeded(self) -> None:
        m = _make_provider().model()
        assert m.is_context_overflow(RuntimeError("context_length_exceeded"))

    def test_maximum_context_length(self) -> None:
        m = _make_provider().model()
        assert m.is_context_overflow(RuntimeError("maximum context length"))

    def test_exceeds_context_window(self) -> None:
        m = _make_provider().model()
        assert m.is_context_overflow(RuntimeError("exceeds the context window"))

    def test_unrelated(self) -> None:
        m = _make_provider().model()
        assert not m.is_context_overflow(RuntimeError("rate limit"))


class TestBuildInput:
    def test_user_message(self) -> None:
        request = ModelRequest(messages=[_user("hello")])
        items = cast(list[dict[str, Any]], _build_input(request))
        assert items == [{"role": "user", "content": "hello"}]

    def test_user_multipart(self) -> None:
        msg = MultipartMessage(
            (
                TextMessage("part1", "text/plain"),
                TextMessage("part2", "text/plain"),
            ),
            "multipart/x-user-message",
        )
        request = ModelRequest(messages=[msg])
        items = cast(list[dict[str, Any]], _build_input(request))
        assert items[0]["content"] == "part1\npart2"

    def test_assistant_with_tool_calls(self) -> None:
        request = ModelRequest(
            messages=[
                _assistant(
                    "Calling.",
                    tool_calls=[
                        tool_call_message("t1", "bash", json_freeze({"command": "ls"}))
                    ],
                ),
            ],
        )
        items = cast(list[dict[str, Any]], _build_input(request))
        assert items[0] == {"role": "assistant", "content": "Calling."}
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "bash"
        assert json.loads(items[1]["arguments"]) == {"command": "ls"}

    def test_tool_result(self) -> None:
        request = ModelRequest(
            messages=[_tool_result("t1", "file list")],
        )
        items = cast(list[dict[str, Any]], _build_input(request))
        assert items[0]["type"] == "function_call_output"
        assert items[0]["output"] == "file list"

    def test_full_conversation(self) -> None:
        request = ModelRequest(
            messages=[
                _user("run ls"),
                _assistant(
                    tool_calls=[
                        tool_call_message("t1", "bash", json_freeze({"command": "ls"}))
                    ],
                ),
                _tool_result("t1", "file1 file2"),
            ],
        )
        items = cast(list[dict[str, Any]], _build_input(request))
        assert items[0]["role"] == "user"
        assert items[1]["type"] == "function_call"
        assert items[2]["type"] == "function_call_output"


class _FakeTool:
    def __init__(self) -> None:
        self.name = "bash"
        self.tool_id = "application/x-tool-bash"
        self.description = "Run bash."
        self.directive_schema = json_freeze({"type": "object", "properties": {}})
        self.supports_microcompaction = False

    def summary(self, msg: Message) -> str:
        del msg
        return ""

    def prompt(self) -> str | None:
        return None

    async def run(self, msg: Message) -> Message:
        del msg
        return TextMessage("", "text/plain")


class TestBuildTools:
    def test_basic(self) -> None:
        tools = _build_tools([_FakeTool()])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "bash"
        assert tools[0].get("description") == "Run bash."


class TestConsumeStream:
    @pytest.mark.anyio
    async def test_text_assembly(self) -> None:
        events = [
            _mock_text_delta("Hello"),
            _mock_text_delta(" world"),
            _mock_completed_event(
                response_id="resp-1",
                status="completed",
                input_tokens=10,
                output_tokens=5,
                cached_tokens=2,
            ),
        ]
        stream = _AsyncIterator(events)
        resp = await _consume_stream(
            stream,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- test double for AsyncResponseStream
            pricing=_DEFAULT_PRICING,
            on_text=None,
        )
        assert _resp_text(resp) == "Hello world"
        assert resp.tokens.input_tokens == 10
        assert resp.tokens.output_tokens == 5
        assert resp.tokens.cache_read_tokens == 2
        assert resp.message_id == "resp-1"
        assert resp.stop_reason == "model_finished"

    @pytest.mark.anyio
    async def test_on_text_callback(self) -> None:

        events = [
            _mock_text_delta("Hi"),
            _mock_text_delta("!"),
            _mock_completed_event(
                response_id="resp-2",
                status="completed",
                input_tokens=5,
                output_tokens=2,
            ),
        ]
        chunks: list[str] = []
        stream = _AsyncIterator(events)
        await _consume_stream(
            stream,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- test double for AsyncResponseStream
            pricing=_DEFAULT_PRICING,
            on_text=chunks.append,
        )
        assert chunks == ["Hi", "!"]

    @pytest.mark.anyio
    async def test_tool_call_extraction(self) -> None:

        events = [
            _mock_function_args_delta("fc_1", '{"command":'),
            _mock_function_args_delta("fc_1", '"ls"}'),
            _mock_output_item_done(
                item_id="fc_1",
                call_id="call_abc",
                name="bash",
            ),
            _mock_completed_event(
                response_id="resp-3",
                status="completed",
                input_tokens=20,
                output_tokens=10,
            ),
        ]
        stream = _AsyncIterator(events)
        resp = await _consume_stream(
            stream,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- test double for AsyncResponseStream
            pricing=_DEFAULT_PRICING,
            on_text=None,
        )
        tcs = _resp_tool_calls(resp)
        assert len(tcs) == 1
        assert get_tool_name(tcs[0]) == "bash"
        assert get_directive(tcs[0]) == json_freeze({"command": "ls"})
        assert resp.stop_reason == "model_tool_use"

    @pytest.mark.anyio
    async def test_cost_calculation(self) -> None:

        events = [
            _mock_completed_event(
                response_id="resp-4",
                status="completed",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                cached_tokens=500_000,
            ),
        ]
        stream = _AsyncIterator(events)
        resp = await _consume_stream(
            stream,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- test double for AsyncResponseStream
            pricing=_DEFAULT_PRICING,
            on_text=None,
        )
        expected_input = (500_000 * 2.5 + 500_000 * 0.25) / 1_000_000
        expected_output = 1_000_000 * 15.0 / 1_000_000
        assert resp.input_cost == pytest.approx(expected_input)
        assert resp.output_cost == pytest.approx(expected_output)
        assert resp.total_cost == pytest.approx(expected_input + expected_output)

    @pytest.mark.anyio
    async def test_incomplete_status_maps_to_length(self) -> None:

        events = [
            _mock_text_delta("partial"),
            _mock_completed_event(
                response_id="resp-5",
                status="incomplete",
                input_tokens=5,
                output_tokens=5,
            ),
        ]
        stream = _AsyncIterator(events)
        resp = await _consume_stream(
            stream,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type] -- test double for AsyncResponseStream
            pricing=_DEFAULT_PRICING,
            on_text=None,
        )
        assert resp.stop_reason == "max_tokens"


class TestLoadSave:
    def test_load(self, tmp_path: Path) -> None:
        p = tmp_path / "auth.json"
        p.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": _make_jwt(exp=9_000_000_000),
                        "refresh_token": "rt",
                        "account_id": "acct",
                    },
                }
            )
        )
        creds = OpenAISubscription.load(path=p)
        assert creds["refresh_token"] == "rt"  # noqa: S105 -- test credential
        assert creds["account_id"] == "acct"
        assert creds["expires_at"] == 9_000_000_000.0

    def test_save_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "auth.json"
        OpenAISubscription.save(
            OpenAISubscription.Credentials(
                access_token="at",  # noqa: S106 -- test credential
                refresh_token="rt",  # noqa: S106 -- test credential
                account_id="acct",
                expires_at=1e9,
            ),
            path=p,
        )
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["auth_mode"] == "chatgpt"
        assert data["tokens"]["access_token"] == "at"  # noqa: S105 -- test credential

    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "auth.json"
        original = OpenAISubscription.Credentials(
            access_token=_make_jwt(exp=9_000_000_000),
            refresh_token="rt",  # noqa: S106 -- test credential
            account_id="acct",
            expires_at=9_000_000_000.0,
        )
        OpenAISubscription.save(original, path=p)
        loaded = OpenAISubscription.load(path=p)
        assert loaded["refresh_token"] == original["refresh_token"]
        assert loaded["account_id"] == original["account_id"]
        assert loaded["expires_at"] == original["expires_at"]

    def test_load_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            OpenAISubscription.load(path=Path("/nonexistent/auth.json"))


class TestLogin:
    def test_manual_uses_pasted_redirect_url(self) -> None:
        access_token = _make_jwt(
            claims={
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct-manual",
                    "chatgpt_plan_type": "plus",
                }
            },
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": access_token,
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }
        mock_http = MagicMock()
        mock_http.post.return_value = mock_resp
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        redirect_url = (
            "http://127.0.0.1:1455/auth/callback?code=manual-code&state=state"
        )

        with (
            patch(
                "sagent.providers.openai_sub.httpx.Client",
                return_value=mock_http,
            ),
            patch(
                "sagent.providers.openai_sub.secrets.token_urlsafe",
                return_value="state",
            ),
            patch(
                "sagent.providers.openai_sub.AuthCodeListener",
            ) as mock_listener,
            patch("builtins.input", return_value=redirect_url),
            patch.object(OpenAISubscription, "save") as mock_save,
        ):
            OpenAISubscription.login(output=io.StringIO(), manual=True)

        mock_listener.assert_not_called()
        mock_save.assert_called_once()
        body = mock_http.post.call_args.kwargs["data"]
        assert body["code"] == "manual-code"
        assert body["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"


class TestOpenAIAccountRouting:
    def test_save_default_writes_legacy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        legacy = tmp_path / "auth.json"
        monkeypatch.setattr(os_, "_DEFAULT_PATH", legacy)
        os_.OpenAISubscription.save(
            os_.OpenAISubscription.Credentials(
                access_token="at",  # noqa: S106 -- test fixture
                refresh_token="rt",  # noqa: S106 -- test fixture
                account_id="acct",
                expires_at=1.0,
            ),
        )
        assert legacy.exists()

    def test_save_named_writes_per_account_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        legacy = tmp_path / "auth.json"
        monkeypatch.setattr(os_, "_DEFAULT_PATH", legacy)
        os_.OpenAISubscription.save(
            os_.OpenAISubscription.Credentials(
                access_token="at",  # noqa: S106 -- test fixture
                refresh_token="rt",  # noqa: S106 -- test fixture
                account_id="acct",
                expires_at=1.0,
            ),
            account="work",
        )
        assert not legacy.exists()
        assert (tmp_path / "auth-work.json").exists()

    def test_default_and_named_dont_collide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        legacy = tmp_path / "auth.json"
        monkeypatch.setattr(os_, "_DEFAULT_PATH", legacy)
        os_.OpenAISubscription.save(
            os_.OpenAISubscription.Credentials(
                access_token=_make_jwt(),
                refresh_token="rt-default",  # noqa: S106 -- test fixture
                account_id="acct-default",
                expires_at=9e9,
            ),
        )
        os_.OpenAISubscription.save(
            os_.OpenAISubscription.Credentials(
                access_token=_make_jwt(),
                refresh_token="rt-work",  # noqa: S106 -- test fixture
                account_id="acct-work",
                expires_at=9e9,
            ),
            account="work",
        )
        assert os_.OpenAISubscription.load()["account_id"] == "acct-default"
        assert os_.OpenAISubscription.load(account="work")["account_id"] == "acct-work"


def _make_jwt(*, exp: float = 9e9, claims: dict[str, object] | None = None) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=")
    payload_data = {"exp": exp, **(claims or {})}
    payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=")
    return f"{header.decode()}.{payload.decode()}.sig"


def _mock_text_delta(text: str) -> Any:
    return ResponseTextDeltaEvent.model_construct(
        delta=text,
        content_index=0,
        item_id="item",
        output_index=0,
        sequence_number=0,
        type="response.output_text.delta",
        logprobs=[],
    )


def _mock_function_args_delta(item_id: str, delta: str) -> Any:
    return ResponseFunctionCallArgumentsDeltaEvent.model_construct(
        delta=delta,
        item_id=item_id,
        output_index=0,
        sequence_number=0,
        type="response.function_call_arguments.delta",
    )


def _mock_output_item_done(
    *,
    item_id: str,
    call_id: str,
    name: str,
) -> Any:
    item = ResponseFunctionToolCall.model_construct(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments="",
        id=item_id,
    )
    return ResponseOutputItemDoneEvent.model_construct(
        item=item,
        output_index=0,
        sequence_number=0,
        type="response.output_item.done",
    )


def _mock_completed_event(
    *,
    response_id: str,
    status: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> Any:
    details = (
        InputTokensDetails.model_construct(cached_tokens=cached_tokens)
        if cached_tokens
        else None
    )
    usage = ResponseUsage.model_construct(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=details,
        output_tokens_details=None,
    )
    resp = Response.model_construct(
        id=response_id,
        usage=usage,
        status=status,
        created_at=0,
        model="gpt-5.4",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return ResponseCompletedEvent.model_construct(
        response=resp,
        sequence_number=0,
        type="response.completed",
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
