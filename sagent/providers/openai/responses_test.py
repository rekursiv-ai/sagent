"""Responses transport regressions exercised at the HTTP boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

import asyncio
import base64
import dataclasses
import json
import os

import httpx2
import pytest

from sagent.agent.retry import is_rate_limited, is_retryable
from sagent.agent.session_io import _entry_from_json, _entry_to_json
from sagent.catalog import openai as openai_catalog
from sagent.lib.custom_json import DictCodec, JSONValue, MutableJSON
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.responses import (
    _build_input,
    _build_tool,
    _build_tool_result_item,
    _build_tools,
    _build_user_item,
    _consume_stream,
    _OpenAIResponsesModel,
    _parse_tool_arguments,
)
from sagent.types.capability import (
    ModelCapability,
    ModelSettings,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.exceptions import UserFacingError
from sagent.types.model import (
    ModelRequest,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AssistantMessage,
    BytesMessage,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


@dataclass(slots=True, kw_only=True)
class _Wire:
    requests: list[dict[str, object]] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    async def send(self, request: httpx2.Request) -> httpx2.Response:
        self.paths.append(request.url.path)
        self.requests.append(DictCodec.coerce(json.loads(request.content)))
        if request.url.path != "/v1/responses":
            return httpx2.Response(
                404, request=request, json={"error": {"message": "Responses required"}}
            )
        return httpx2.Response(
            200,
            request=request,
            headers={
                "content-type": "text/event-stream",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "25",
                "x-ratelimit-reset-requests": "2ms",
            },
            content=_response_stream(),
        )


def _response_stream() -> bytes:
    events: list[MutableJSON] = [
        {
            "type": "response.output_text.delta",
            "delta": "OK",
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 0,
        },
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-4o",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 2,
                    "total_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        },
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    result = _Wire()

    async def send(
        _client: httpx2.AsyncClient, request: httpx2.Request, **_kwargs: object
    ) -> httpx2.Response:
        return await result.send(request)

    monkeypatch.setattr(httpx2.AsyncClient, "send", send)
    return result


@pytest.mark.anyio
@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-6-astra"])
async def test_api_stream_uses_responses_with_supported_knobs(
    wire: _Wire, model_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "debug.jsonl"
    monkeypatch.setenv("SAGENT_DEBUG", "1")
    monkeypatch.setenv("SAGENT_DEBUG_LOG", str(log_path))
    provider = OpenAI.from_key("test-key")
    model = provider.model(model_id)
    if model_id == "gpt-6-astra":
        model.settings.thinking_effort = "high"
    try:
        response = await model.stream(
            ModelRequest(
                messages=[UserMessage(text="hello")],
                tools=[_StubTool()],
                max_response_tokens=123,
                temperature=0.25,
            )
        )
    finally:
        await provider.close_sdk()
    assert response.message.text == "OK"
    assert response.tokens.request == 8
    assert response.tokens.cache_read == 4
    assert wire.paths == ["/v1/responses"]
    records = [
        DictCodec.coerce(json.loads(line)) for line in log_path.read_text().splitlines()
    ]
    assert any(
        record.get("event") == "api_call"
        and record.get("kind") == "openai_responses"
        and record.get("model") == model_id
        for record in records
    )
    body = wire.requests[0]
    assert body["max_output_tokens"] == 123
    assert body["store"] is False
    if model_id == "gpt-4o":
        assert body["temperature"] == 0.25
        assert "reasoning" not in body
    else:
        assert "temperature" not in body
        assert DictCodec.coerce(body["reasoning"])["effort"] == "high"
        assert body["include"] == ["reasoning.encrypted_content"]


@pytest.mark.anyio
async def test_api_responses_preserves_quota_headers(wire: _Wire) -> None:
    provider = OpenAI.from_key("test-key")
    model = provider.model("gpt-4o")
    try:
        await model.stream(ModelRequest(messages=[UserMessage(text="hello")]))
        snapshot = model.usage_snapshot()
    finally:
        await provider.close_sdk()
    assert wire.paths == ["/v1/responses"]
    assert snapshot is not None
    assert snapshot.windows[0].utilization == 0.75
    assert snapshot.windows[0].resets_at is not None


def test_responses_preserves_image_bearing_tool_results() -> None:
    request = ModelRequest(
        messages=[
            AssistantMessage(
                tool_calls=(ToolCall(id="foreign", name="Read", args={}),)
            ),
            ToolResult(
                call_id="foreign",
                content="image",
                attachments=(BytesMessage(data=_TINY_PNG, descriptor="image/png"),),
            ),
        ]
    )
    items = _build_input(request)
    output = DictCodec.coerce(items[-1])["output"]
    assert isinstance(output, list)
    assert "data:image/" in json.dumps(output)
    assert (
        DictCodec.coerce(items[0])["call_id"] == DictCodec.coerce(items[-1])["call_id"]
    )


@pytest.mark.parametrize("model_id", ["o1", "o3-mini"])
def test_o_series_catalog_has_only_supported_efforts(model_id: str) -> None:
    assert OpenAI.from_key("test-key").model(
        model_id
    ).capability.thinking_effort == frozenset({"low", "medium", "high"})


@pytest.mark.parametrize("model_id", ["gpt-5.4-pro", "gpt-5.5-pro"])
def test_pro_catalog_has_only_supported_efforts(model_id: str) -> None:
    assert OpenAI.from_key("test-key").model(
        model_id
    ).capability.thinking_effort == frozenset({"medium", "high", "xhigh"})


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.2", "gpt-5.3-codex", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"],
)
def test_earlier_gpt5_catalog_keeps_native_efforts(model_id: str) -> None:
    model = OpenAI.from_key("test-key").model(model_id)
    assert model.capability.thinking_effort == frozenset(
        {"none", "low", "medium", "high", "xhigh"}
    )
    assert openai_catalog.reasoning_effort("xhigh", model_id=model_id) == "xhigh"


def _free_model() -> _OpenAIResponsesModel:
    """A model whose every rate is zero -- cost is not what these assert."""
    return _OpenAIResponsesModel(
        provider=OpenAI.from_key("test-key"),
        capability=ModelCapability(
            prices=PriceCatalog({PriceCatalogProduct(): TokenPrice()})
        ),
        settings=ModelSettings(),
    )


def _priced_model() -> _OpenAIResponsesModel:
    """$1/Mtok request with the usual 1.25x cache-write multiplier."""
    return _OpenAIResponsesModel(
        provider=OpenAI.from_key("test-key"),
        capability=ModelCapability(
            prices=PriceCatalog(
                {PriceCatalogProduct(): TokenPrice(request=1.0, cache_write=1.25)}
            )
        ),
        settings=ModelSettings(),
    )


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

    def __init__(self, response: object | None = None) -> None:
        self.response = response or _CompletedResponse()


class _CompletedResponse:
    """Small stand-in for OpenAI's completed response payload."""

    def __init__(self, usage: object | None = None) -> None:
        self.id = "resp_123"
        self.status = "completed"
        self.usage = usage


class _InputTokenDetails:
    """SDK-shaped details object whose ``to_dict`` exposes new fields."""

    def __init__(self, *, cached_tokens: int, cache_write_tokens: int) -> None:
        self.cached_tokens = cached_tokens
        self._cache_write_tokens = cache_write_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self._cache_write_tokens,
        }


class _Usage:
    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = _InputTokenDetails(
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )


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

    def __init__(self, reason: str = "max_output_tokens") -> None:
        self.response = _IncompleteResponse(reason)


class _IncompleteResponse:
    """Small stand-in for an incomplete OpenAI response payload."""

    id: str = "resp_incomplete"
    status: str = "incomplete"
    error = None

    def __init__(self, reason: str) -> None:
        self.incomplete_details = type("Incomplete", (), {"reason": reason})()


class _ReasoningOutputItem:
    type: str = "reasoning"

    def to_dict(self, **_: object) -> dict[str, object]:
        return {
            "id": "rs_123",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "encrypted-reasoning",
            "status": "completed",
        }


class _ReasoningOutputDoneEvent:
    def __init__(self) -> None:
        self.item = _ReasoningOutputItem()


class _RefusalDeltaEvent:
    def __init__(self, delta: str) -> None:
        self.delta = delta


class _RefusalDoneEvent:
    def __init__(self, refusal: str) -> None:
        self.refusal = refusal


def _stub_request_messages(
    *items: UserMessage | AssistantMessage | ToolResult,
) -> ModelRequest:
    # The builders only iterate ``request.messages``; build a stand-in.
    return ModelRequest(messages=list(items))


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
    image_url = str(image_block["image_url"])
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
    with caplog.at_level("WARNING", logger="sagent.providers.openai.responses"):
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


def test_build_assistant_items_replays_encrypted_reasoning() -> None:
    reasoning: dict[str, object] = {
        "id": "rs_123",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "encrypted-reasoning",
        "status": "completed",
    }
    asst = AssistantMessage(text="answer", thinking_blocks=(reasoning,))
    items = _items_as_list(_stub_request_messages(asst))
    assert items == [
        reasoning,
        {"role": "assistant", "content": "answer"},
    ]


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


@pytest.mark.parametrize("done_args", ["[]", "null", "42", '"text"'])
def test_parse_tool_arguments_non_object_keeps_delta(done_args: str) -> None:
    assert _parse_tool_arguments(
        '{"nested": {"items": [1, true, null]}}',
        done_args,
        tool_name="N",
        call_id="cid",
    ) == {"nested": {"items": [1, True, None]}}


def test_parse_tool_arguments_done_empty_object_keeps_delta() -> None:
    # ``done`` is parsed but falsy; truthy ``delta`` wins via ``if done``.
    out = _parse_tool_arguments(
        delta_args='{"a": 1}',
        done_args="{}",
        tool_name="N",
        call_id="cid",
    )
    assert out == {"a": 1}


class TestStreamIdleTimeout:
    """Silent Responses streams must not make the runtime wait forever."""

    @pytest.mark.anyio
    async def test_silent_stream_times_out_and_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = _NeverYieldingStream()
        monkeypatch.setattr(
            "sagent.providers.openai.responses._STREAM_IDLE_TIMEOUT",
            0.01,
        )

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                _consume_stream(
                    stream,
                    model=_free_model(),
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
            "sagent.providers.openai.responses._STREAM_IDLE_TIMEOUT",
            0.05,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        stream = _DelayedStream(
            [_TextDeltaEvent("he"), _TextDeltaEvent("llo"), _CompletedEvent()],
            delay_sec=0.03,
        )

        response = await asyncio.wait_for(
            _consume_stream(
                stream,
                model=_free_model(),
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
            "sagent.providers.openai.responses.responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        stream = _DelayedStream([_TextDeltaEvent("partial")], delay_sec=0.0)

        with pytest.raises(StreamInterruptedError) as raised:
            await _consume_stream(
                stream,
                model=_free_model(),
                publish=None,
            )

        assert raised.value.response.message.text == "partial"
        assert raised.value.response.stop_reason == "model_finished"

    @pytest.mark.anyio
    async def test_response_error_event_is_user_facing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseErrorEvent",
            _ResponseErrorEvent,
        )
        stream = _DelayedStream([_ResponseErrorEvent()], delay_sec=0.0)

        with pytest.raises(UserFacingError) as raised:
            await _consume_stream(
                stream,
                model=_free_model(),
                publish=None,
            )

        msg = str(raised.value)
        assert "too many requests" in msg
        assert "code=rate_limit" in msg
        assert "param=input" in msg
        model = OpenAI.from_key("test-key").model("gpt-5.6-sol")
        assert is_retryable(raised.value, model) is True
        assert is_rate_limited(raised.value) is True

    @pytest.mark.anyio
    async def test_error_event_closes_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-stream error event must close the stream before propagating.

        ``ResponseErrorEvent``/``ResponseFailedEvent`` raise from inside the
        ``async for``; if the only cleanup is the cancellation/timeout ``except``
        plus the not-completed fallthrough, the raised error skips both and the
        SSE connection leaks. The stream must be closed on EVERY non-completed
        exit.
        """
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseErrorEvent",
            _ResponseErrorEvent,
        )

        class _ClosableStream:
            def __init__(self, events: list[object]) -> None:
                self._events = events
                self.closed = False

            def __aiter__(self) -> _ClosableStream:
                return self

            async def __anext__(self) -> object:
                if not self._events:
                    raise StopAsyncIteration
                return self._events.pop(0)

            async def aclose(self) -> None:
                self.closed = True

        stream = _ClosableStream([_ResponseErrorEvent()])
        with pytest.raises(UserFacingError):
            await _consume_stream(stream, model=_free_model(), publish=None)
        assert stream.closed, "error-event path leaked the stream (aclose never called)"

    @pytest.mark.anyio
    async def test_response_failed_event_is_user_facing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseFailedEvent",
            _FailedEvent,
        )
        stream = _DelayedStream([_FailedEvent()], delay_sec=0.0)

        with pytest.raises(UserFacingError) as raised:
            await _consume_stream(
                stream,
                model=_free_model(),
                publish=None,
            )

        msg = str(raised.value)
        assert "status=failed" in msg
        assert "response_id=resp_failed" in msg
        assert "code=server_error" in msg
        assert "backend failed" in msg
        model = OpenAI.from_key("test-key").model("gpt-5.6-sol")
        assert is_retryable(raised.value, model) is True

    @pytest.mark.anyio
    async def test_response_incomplete_event_returns_partial_length_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseIncompleteEvent",
            _IncompleteEvent,
        )
        stream = _DelayedStream(
            [_TextDeltaEvent("partial answer"), _IncompleteEvent()],
            delay_sec=0.0,
        )

        response = await _consume_stream(
            stream,
            model=_free_model(),
            publish=None,
        )

        assert response.message.text == "partial answer"
        assert response.message_id == "resp_incomplete"
        assert response.stop_reason == "max_tokens"

    @pytest.mark.anyio
    async def test_response_incomplete_content_filter_is_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseIncompleteEvent",
            _IncompleteEvent,
        )
        stream = _DelayedStream(
            [_TextDeltaEvent("partial answer"), _IncompleteEvent("content_filter")],
            delay_sec=0.0,
        )

        response = await _consume_stream(
            stream,
            model=_free_model(),
            publish=None,
        )

        assert response.message.text == "partial answer"
        assert response.stop_reason == "model_refusal"

    @pytest.mark.anyio
    async def test_completed_usage_tracks_and_bills_cache_write_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        event = _CompletedEvent(
            _CompletedResponse(
                _Usage(
                    input_tokens=1309,
                    output_tokens=2,
                    cache_write_tokens=1306,
                )
            )
        )
        response = await _consume_stream(
            _DelayedStream([event], delay_sec=0.0),
            model=_priced_model(),
            publish=None,
        )
        assert response.tokens.request == 3
        assert response.tokens.cache_write == 1306
        assert (
            response.spend.request
            + response.spend.cache_write
            + response.spend.cache_read
        ) == pytest.approx((3 + 1306 * 1.25) / 1_000_000)

    @pytest.mark.anyio
    async def test_stream_preserves_and_replays_encrypted_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseOutputItemDoneEvent",
            _ReasoningOutputDoneEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        response = await _consume_stream(
            _DelayedStream(
                [_ReasoningOutputDoneEvent(), _CompletedEvent()],
                delay_sec=0.0,
            ),
            model=_priced_model(),
            publish=None,
        )
        encrypted = next(
            block
            for block in response.message.thinking_blocks
            if block.get("encrypted_content") == "encrypted-reasoning"
        )
        replay = _items_as_list(
            ModelRequest(messages=[AssistantMessage(thinking_blocks=(encrypted,))])
        )
        assert replay == [encrypted]

    @pytest.mark.anyio
    async def test_stream_preserves_refusal_without_duplication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseRefusalDeltaEvent",
            _RefusalDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseRefusalDoneEvent",
            _RefusalDoneEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseCompletedEvent",
            _CompletedEvent,
        )
        response = await _consume_stream(
            _DelayedStream(
                [
                    _RefusalDeltaEvent("I can’t "),
                    _RefusalDeltaEvent("help with that."),
                    _RefusalDoneEvent("I can’t help with that."),
                    _CompletedEvent(),
                ],
                delay_sec=0.0,
            ),
            model=_priced_model(),
            publish=None,
        )
        assert response.message.text == "I can’t help with that."
        assert response.stop_reason == "model_refusal"

    @pytest.mark.anyio
    async def test_stream_routes_reasoning_deltas_to_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseTextDeltaEvent",
            _TextDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseReasoningTextDeltaEvent",
            _ReasoningDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseReasoningSummaryTextDeltaEvent",
            _ReasoningDeltaEvent,
        )
        monkeypatch.setattr(
            "sagent.providers.openai.responses.responses.ResponseCompletedEvent",
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
            model=_priced_model(),
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
                model=_priced_model(),
                publish=None,
            ),
        )
        await asyncio.wait_for(stream.entered.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert stream.closed is True


class _VerifyTool(_StubTool):
    name = "verify"
    description = "Verify the computed integer n."
    directive_schema = MappingProxyType(
        {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        }
    )


@pytest.mark.network_openai
@pytest.mark.anyio
async def test_api_astra_reasoning_tool_roundtrip_and_legacy_replay() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    provider = OpenAI.from_env()
    model = provider.model("gpt-6-astra")
    model.settings.thinking_effort = "high"
    user = UserMessage(
        text="Find the smallest positive integer n with n mod 17=12, n mod 19=7, n mod 23=5. Compute it before calling verify(n). After verification, do not call tools again; report the result."
    )
    try:
        first = await model.stream(
            ModelRequest(
                messages=[user], tools=[_VerifyTool()], max_response_tokens=2048
            )
        )
        assert any(
            isinstance(block.get("encrypted_content"), str)
            and block["encrypted_content"]
            for block in first.message.thinking_blocks
        )
        assert first.message.tool_calls
        call = first.message.tool_calls[0]
        assert call.name == "verify"
        number = call.args["n"]
        assert isinstance(number, int)
        assert (number % 17, number % 19, number % 23) == (12, 7, 5)
        restored = _entry_from_json(
            DictCodec.coerce(json.loads(json.dumps(_entry_to_json(first.message))))
        )
        assert isinstance(restored, AssistantMessage)
        assert restored == first.message
        result = ToolResult(call_id=call.id, content="Verified successfully.")
        for history in _replay_variants(restored):
            response = await model.stream(
                ModelRequest(
                    messages=[user, history, result],
                    tools=[_VerifyTool()],
                    max_response_tokens=2048,
                )
            )
            assert response.message.text
            assert not response.message.tool_calls
            assert response.tokens.response > 0
    finally:
        await provider.close_sdk()


def _replay_variants(message: AssistantMessage) -> Iterator[AssistantMessage]:
    yield message
    yield dataclasses.replace(
        message, thinking_blocks=({"type": "reasoning", "text": "legacy reasoning"},)
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
