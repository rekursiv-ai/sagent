"""Tests for the OpenAI-compatible base provider, against a fake HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, override

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.openai import chat, compat, token_count
from sagent.providers.openai.compat import OpenAICompatible, OpenAIModel, models
from sagent.providers.resources import SharedResources
from sagent.providers.types import UsageWindow
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    Thinking,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping


_REQUEST = Request(system=("Be terse.",), records=(UserMessage(content="hi"),))


class _Vendor(OpenAICompatible):
    """A vendor as a subclass names one: its catalog, endpoint, and key."""

    section: ClassVar[str] = "moonshot"
    package: ClassVar[str] = "sagent.providers.moonshot"
    document: ClassVar[str] = "moonshot.yaml"
    url: ClassVar[str] = "https://api.vendor.test/v1"
    key_variable: ClassVar[str] = "VENDOR_API_KEY"
    reasoning_field: ClassVar[str] = "reasoning_content"


class _Renewing(_Vendor):
    """A vendor whose login renews once after a 401."""

    def __init__(self, *, resources: SharedResources) -> None:
        super().__init__(resources=resources)
        self.renewals: list[str] = []

    @override
    async def credentials(self) -> dict[str, str]:
        return {"authorization": f"Bearer token-{len(self.renewals)}"}

    @override
    async def reauthenticate(self) -> bool:
        self.renewals.append("renewed")
        return True


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VENDOR_API_KEY", "sk-vendor")


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record each backoff instead of waiting it out."""
    waits: list[float] = []

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)

    monkeypatch.setattr(compat, "asyncio", SimpleNamespace(sleep=sleep))
    return waits


def _sse(*chunks: Mapping[str, object]) -> bytes:
    lines = "".join(f"data: {json_dumps(chunk)}\n\n" for chunk in chunks)
    return f"{lines}data: [DONE]\n\n".encode()


def _said(text: str) -> tuple[dict[str, object], ...]:
    return (
        {"choices": [{"delta": {"reasoning_content": "Hmm."}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    )


def _ok(
    text: str = "Hello",
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx2.Response:
    return httpx2.Response(200, headers=headers, content=_sse(*_said(text)))


def _failed(
    status: int,
    message: str,
    *,
    code: str = "",
    headers: Mapping[str, str] | None = None,
) -> httpx2.Response:
    body = {"error": {"message": message, "code": code, "type": ""}}
    return httpx2.Response(status, headers=headers, content=json_dumps(body).encode())


async def _stream(
    model: OpenAIModel,
    request: Request = _REQUEST,
) -> list[TextChunk | Response]:
    return [event async for event in model.stream(request)]


def _body(request: httpx2.Request) -> dict[str, object]:
    return json_loads(request.content, dict[str, object])


# ---- the provider ------------------------------------------------------------


def test_the_provider_serves_its_sections_models_by_any_name() -> None:
    vendor = _Vendor()
    assert vendor.name == "moonshot"
    assert _Vendor(name="kimi-work").name == "kimi-work"
    assert list(vendor) == list(
        models(_Vendor.package, document=_Vendor.document, section="moonshot"),
    )
    model = vendor["default"]
    assert model.provider is vendor
    assert model.capability.model_id == "kimi-3.0"
    assert vendor.usage == ()


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("kimi-2.6", "kimi-2.6"),  # Its exact name.
        ("kimi-k2.6", "kimi-2.6"),  # The vendor's id.
        ("utility", "kimi-2.6"),  # A role.
        ("default", "kimi-3.0"),
        ("kimi", "kimi-3.0"),  # A family: its newest.
        ("kimi-code", "kimi-code-2.7"),
        ("kimi-code-2", "kimi-code-2.7"),  # A family and major.
    ],
)
def test_a_model_is_found_by_any_of_its_names(name: str, model_id: str) -> None:
    assert _Vendor()[name].capability.model_id == model_id


@pytest.mark.asyncio
async def test_credentials_are_the_key_given_else_the_keys_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await _Vendor().credentials() == {"authorization": "Bearer sk-vendor"}
    assert await _Vendor(api_key="sk-given").credentials() == {
        "authorization": "Bearer sk-given",
    }
    monkeypatch.delenv("VENDOR_API_KEY")
    with pytest.raises(AuthError, match="VENDOR_API_KEY"):
        await _Vendor().credentials()
    monkeypatch.setattr(_Vendor, "key_variable", "")
    assert await _Vendor().credentials() == {}
    assert await _Vendor().reauthenticate() is False


@pytest.mark.asyncio
async def test_the_endpoint_is_the_base_url_given_else_the_vendors() -> None:
    assert await _Vendor().endpoint() == "https://api.vendor.test/v1"
    assert (
        await _Vendor(base_url="http://box:8000/v1").endpoint() == "http://box:8000/v1"
    )


@pytest.mark.asyncio
async def test_closing_closes_only_resources_the_provider_made() -> None:
    owned = _Vendor()
    http = owned.resources.http
    await owned.aclose()
    assert http.is_closed
    async with SharedResources() as shared:
        async with _Vendor(resources=shared):
            http = shared.http
        assert not http.is_closed


# ---- streaming ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_posts_the_request_and_yields_text_then_the_response() -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        vendor = _Vendor(resources=shared, base_url="http://box:8000/v1/")
        kimi = vendor["default"]
        events = await _stream(kimi)
    (sent,) = server.requests
    assert (sent.method, str(sent.url)) == (
        "POST",
        "http://box:8000/v1/chat/completions",
    )
    assert sent.headers["authorization"] == "Bearer sk-vendor"
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["accept"] == "text/event-stream"
    assert _body(sent) == vendor.encode(_REQUEST, model=kimi)
    assert _body(sent) == chat.encode(
        _REQUEST,
        capability=kimi.capability,
        settings=kimi.settings,
        reasoning_field="reasoning_content",
        temperature=1.0,
    )
    assert events[:2] == [
        TextChunk(text="Hmm.", thinking=True),
        TextChunk(text="Hello", record=1),
    ]
    assert len(events) == 3
    response = events[2]
    assert isinstance(response, Response)
    assert response.records == (
        Thinking(content="Hmm."),
        AssistantMessage(content="Hello"),
    )
    assert (response.tokens.request, response.tokens.response) == (3, 1)


@pytest.mark.asyncio
async def test_a_request_carries_the_headers_its_provider_adds() -> None:
    class Tagged(_Vendor):
        @override
        def headers(self, request: Request) -> dict[str, str]:
            return {"x-session": str(len(request.records))}

    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        await _stream(Tagged(resources=shared)["default"])
    assert server.requests[0].headers["x-session"] == "1"
    assert _Vendor().headers(_REQUEST) == {}


@pytest.mark.asyncio
async def test_each_reply_updates_the_accounts_windows() -> None:
    limits = {"x-ratelimit-limit-requests": "4", "x-ratelimit-remaining-requests": "3"}
    server = _Server(replies=[_ok(headers=limits), _ok()])
    async with _Served(server) as shared:
        vendor = _Vendor(resources=shared)
        await _stream(vendor["default"])
        window = UsageWindow(label="requests", utilization=0.25)
        assert vendor.usage == (window,)
        # A reply that reports no window leaves the last report standing.
        await _stream(vendor["default"])
        assert vendor.usage == (window,)


@pytest.mark.asyncio
async def test_an_error_reply_raises_what_it_means(slept: list[float]) -> None:
    server = _Server(
        replies=[
            _failed(400, "x", code="context_length_exceeded"),
            httpx2.Response(400, content=b"<html>Bad Request</html>"),
        ],
    )
    async with _Served(server) as shared:
        kimi = _Vendor(resources=shared)["default"]
        with pytest.raises(ContextOverflowError):
            await _stream(kimi)
        with pytest.raises(RuntimeError, match=r"400 : <html>Bad Request</html>"):
            await _stream(kimi)
    assert len(server.requests) == 2
    assert slept == []


@pytest.mark.asyncio
async def test_a_failure_whose_body_is_cut_short_still_means_its_status(
    slept: list[float],
) -> None:
    cut = httpx2.Response(503, content=b'{"error": ')
    server = _Server(replies=[cut] * 5)
    async with _Served(server) as shared:
        with pytest.raises(OverloadedError, match=r"\{\"error\": "):
            await _stream(_Vendor(resources=shared)["default"])
    assert len(server.requests) == 5
    # Retried as an overload is, not failed as a parse error.
    assert slept == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_a_short_failure_before_the_first_event_is_retried(
    slept: list[float],
) -> None:
    server = _Server(
        replies=[
            _failed(503, "Busy"),
            _failed(429, "Slow", headers={"retry-after": "2"}),
            _ok(),
        ],
    )
    async with _Served(server) as shared:
        events = await _stream(_Vendor(resources=shared)["default"])
    assert TextChunk(text="Hello", record=1) in events
    assert len(server.requests) == 3
    # Doubling backoff, or the server's advice when it is longer.
    assert slept == [0.5, 2.0]


@pytest.mark.asyncio
async def test_retrying_stops_after_five_attempts(slept: list[float]) -> None:
    server = _Server(replies=[_failed(503, "Busy")] * 5)
    async with _Served(server) as shared:
        with pytest.raises(OverloadedError, match="Busy"):
            await _stream(_Vendor(resources=shared)["default"])
    assert len(server.requests) == 5
    assert slept == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_a_long_advised_wait_is_left_to_the_caller(slept: list[float]) -> None:
    server = _Server(replies=[_failed(429, "Slow", headers={"retry-after": "31"})])
    async with _Served(server) as shared:
        with pytest.raises(RateLimitedError) as raised:
            await _stream(_Vendor(resources=shared)["default"])
    assert raised.value.retry_after_sec == 31.0
    assert len(server.requests) == 1
    assert slept == []


@pytest.mark.asyncio
async def test_a_dropped_connection_is_a_disconnect_and_is_retried(
    slept: list[float],
) -> None:
    refused = httpx2.ConnectError("Connection refused")
    server = _Server(replies=[refused, _ok()])
    async with _Served(server) as shared:
        events = await _stream(_Vendor(resources=shared)["default"])
    assert TextChunk(text="Hello", record=1) in events
    assert slept == [0.5]
    server = _Server(replies=[refused] * 5)
    async with _Served(server) as shared:
        with pytest.raises(DisconnectedError, match="Connection refused"):
            await _stream(_Vendor(resources=shared)["default"])
    assert len(server.requests) == 5


@pytest.mark.asyncio
async def test_a_failure_after_text_has_streamed_is_not_retried(
    slept: list[float],
) -> None:
    broken = httpx2.Response(200, content=_broken(_said("Hal")[:2]))
    server = _Server(replies=[broken, _ok()])
    async with _Served(server) as shared:
        kimi = _Vendor(resources=shared)["default"]
        seen: list[TextChunk | Response] = []
        with pytest.raises(DisconnectedError, match="reset"):
            await _drain(kimi, into=seen)
    assert seen == [
        TextChunk(text="Hmm.", thinking=True),
        TextChunk(text="Hal", record=1),
    ]
    assert len(server.requests) == 1
    assert slept == []


@pytest.mark.asyncio
async def test_a_stream_that_ends_unfinished_is_a_disconnect(
    slept: list[float],
) -> None:
    cut = httpx2.Response(200, content=_sse(*_said("Hal")[:2]))
    server = _Server(replies=[cut])
    async with _Served(server) as shared:
        with pytest.raises(DisconnectedError, match="before it was finished"):
            await _stream(_Vendor(resources=shared)["default"])
    assert slept == []


@pytest.mark.asyncio
async def test_a_401_renews_the_login_once_and_asks_again() -> None:
    server = _Server(
        replies=[
            _failed(401, "Expired"),
            _ok(),
            _failed(401, "Expired"),
            _failed(401, "Still"),
        ],
    )
    async with _Served(server) as shared:
        renewing = _Renewing(resources=shared)
        model = renewing["default"]
        assert TextChunk(text="Hello", record=1) in await _stream(model)
        with pytest.raises(AuthError, match="Still"):
            await _stream(model)
    assert [r.headers["authorization"] for r in server.requests] == [
        "Bearer token-0",
        "Bearer token-1",
        "Bearer token-1",
        "Bearer token-2",
    ]
    assert renewing.renewals == ["renewed", "renewed"]


@pytest.mark.asyncio
async def test_a_401_with_nothing_to_renew_is_an_auth_error() -> None:
    server = _Server(replies=[_failed(401, "Bad key")])
    async with _Served(server) as shared:
        with pytest.raises(AuthError, match="Bad key"):
            await _stream(_Vendor(resources=shared)["default"])
    assert len(server.requests) == 1


# ---- counting and replies ----------------------------------------------------


@pytest.mark.asyncio
async def test_count_estimates_from_the_rows_characters_per_token() -> None:
    kimi = _Vendor()["default"]
    request = Request(system=("x" * 40,), records=(UserMessage(content="y" * 40),))
    tokens = await kimi.count(request)
    assert tokens == int(
        len("x" * 40 + "\n" + "y" * 40) / kimi.capability.approx_chars_per_token,
    )
    assert tokens > 0


@pytest.mark.asyncio
async def test_reply_posts_json_and_returns_the_json_answered() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b'{"answer": 42}')])
    async with _Served(server) as shared:
        vendor = _Vendor(resources=shared)
        answer = await vendor.reply("/things", body={"q": 1}, headers={"x-extra": "1"})
    assert answer == {"answer": 42}
    (sent,) = server.requests
    assert str(sent.url) == "https://api.vendor.test/v1/things"
    assert sent.headers["x-extra"] == "1"
    assert sent.headers["accept"] == "*/*"
    assert _body(sent) == {"q": 1}


@dataclass(slots=True, kw_only=True)
class _Server:
    """Answers each request with the next scripted reply, and keeps the requests."""

    replies: list[httpx2.Response | Exception]
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _Served(SharedResources):
    """Shared resources whose HTTP client answers from ``handler``, never the network."""

    def __init__(self, handler: Callable[[httpx2.Request], httpx2.Response]) -> None:
        super().__init__()
        self._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    @property
    @override
    def http(self) -> httpx2.AsyncClient:
        return self._client

    @override
    async def aclose(self) -> None:
        await super().aclose()
        await self._client.aclose()


async def _drain(model: OpenAIModel, *, into: list[TextChunk | Response]) -> None:
    """Collect what ``model`` streams into ``into`` until the stream fails."""
    stream = model.stream(_REQUEST)
    while True:
        into.append(await anext(stream))


async def _broken(chunks: tuple[Mapping[str, object], ...]) -> AsyncIterator[bytes]:
    """Stream ``chunks``, then drop the connection."""
    yield "".join(f"data: {json_dumps(chunk)}\n\n" for chunk in chunks).encode()
    raise httpx2.ReadError("Connection reset by peer")


def test_an_attachment_is_billed_by_openai_s_formula() -> None:
    model = _Vendor()["default"]
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360f8cf000000030101005c0d6a1c0000000049454e44ae426082",
    )
    attached = Attachment(mime_descriptor="image/png", data=png)
    assert model.attachment_tokens(attached) == token_count.attachment_tokens(
        attached,
        capability=model.capability,
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
