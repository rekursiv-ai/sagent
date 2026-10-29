"""Tests for Gemini over the Generative Language API, against a fake HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, override

from treekle import ReadError, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.cost import TokenCount
from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    UnknownModelError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.google import api, wire
from sagent.providers.google.api import Google, models
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, Attachment, UserMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from sagent.model.capability import ModelCapability


_REQUEST = Request(system=("Be terse.",), records=(UserMessage(content="hi"),))
_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record each backoff instead of waiting it out."""
    waits: list[float] = []

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)

    monkeypatch.setattr(api, "asyncio", SimpleNamespace(sleep=sleep))
    return waits


# ---- credentials -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("google", "gemini", "key"),
    [("AIza-g", "AIza-m", "AIza-g"), ("", "AIza-m", "AIza-m")],
)
async def test_the_key_is_googles_variable_else_geminis(
    monkeypatch: pytest.MonkeyPatch,
    google: str,
    gemini: str,
    key: str,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", google)
    monkeypatch.setenv("GEMINI_API_KEY", gemini)
    assert await Google().credentials() == {"x-goog-api-key": key}


@pytest.mark.asyncio
async def test_no_key_is_an_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(
        AuthError,
        match=r"^Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set\.$",
    ):
        await Google().credentials()


# ---- stream ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_posts_the_request_and_yields_text_then_the_response() -> None:
    server = _Server(replies=[_ok("He", "llo")])
    async with _Served(server) as shared:
        pro = _thinking(Google(resources=shared)["gemini-pro"])
        events = await _stream(pro.stream(_REQUEST))
    (sent,) = server.requests
    assert (sent.method, str(sent.url)) == (
        "POST",
        f"{_BASE}/gemini-3.1-pro-preview:streamGenerateContent?alt=sse",
    )
    assert sent.headers["x-goog-api-key"] == "AIza-test"
    assert sent.headers["content-type"] == "application/json"
    assert _body(sent) == wire.encode(
        _REQUEST,
        capability=pro.capability,
        settings=pro.settings,
    )
    assert events[:2] == [TextChunk(text="He"), TextChunk(text="llo")]
    response = events[2]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hello"),)
    assert (response.stop_reason, response.tokens) == (
        "end_turn",
        TokenCount(request=7, response=2),
    )


@pytest.mark.asyncio
async def test_a_failure_before_the_first_event_is_retried_with_backoff(
    slept: list[float],
) -> None:
    server = _Server(
        replies=[
            _failed(503, "UNAVAILABLE"),
            httpx2.ConnectError("refused"),
            _failed(429, "RESOURCE_EXHAUSTED", headers={"retry-after": "2"}),
            _ok("fine"),
        ],
    )
    async with _Served(server) as shared:
        events = await _stream(
            Google(resources=shared)["gemini-flash"].stream(_REQUEST),
        )
    assert slept == [0.5, 1.0, 2.0]
    assert len(server.requests) == 4
    assert events[0] == TextChunk(text="fine")


@pytest.mark.asyncio
async def test_retrying_stops_after_five_attempts(slept: list[float]) -> None:
    server = _Server(replies=[_failed(500, "INTERNAL") for _ in range(6)])
    async with _Served(server) as shared:
        with pytest.raises(OverloadedError):
            await _stream(Google(resources=shared)["gemini-flash"].stream(_REQUEST))
    assert slept == [0.5, 1.0, 2.0, 4.0]
    assert len(server.requests) == 5


@pytest.mark.asyncio
async def test_a_long_advised_wait_is_left_to_the_caller(slept: list[float]) -> None:
    server = _Server(
        replies=[_failed(429, "RESOURCE_EXHAUSTED", headers={"retry-after": "31"})],
    )
    async with _Served(server) as shared:
        with pytest.raises(RateLimitedError) as raised:
            await _stream(Google(resources=shared)["gemini-flash"].stream(_REQUEST))
    assert raised.value.retry_after_sec == 31.0
    assert slept == []


@pytest.mark.asyncio
async def test_a_wait_of_thirty_seconds_is_still_retried(slept: list[float]) -> None:
    server = _Server(
        replies=[
            _failed(429, "RESOURCE_EXHAUSTED", headers={"retry-after": "30"}),
            _ok("fine"),
        ],
    )
    async with _Served(server) as shared:
        await _stream(Google(resources=shared)["gemini-flash"].stream(_REQUEST))
    assert slept == [30.0]


@pytest.mark.asyncio
async def test_a_drop_after_text_is_not_retried(
    slept: list[float],
) -> None:
    started = _sse(_event("He"))
    server = _Server(replies=[httpx2.Response(200, content=_broken(started)), _ok("x")])
    async with _Served(server) as shared:
        pro = _thinking(Google(resources=shared)["gemini-pro"])
        seen: list[TextChunk | Response] = []
        with pytest.raises(DisconnectedError, match="reset by peer"):
            await _collect(pro.stream(_REQUEST), into=seen)
    assert seen == [TextChunk(text="He")]
    assert slept == []
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_a_drop_before_any_text_is_retried_then_raised(
    slept: list[float],
) -> None:
    server = _Server(replies=[httpx2.ReadError("gone") for _ in range(5)])
    async with _Served(server) as shared:
        with pytest.raises(DisconnectedError, match="gone"):
            await _stream(Google(resources=shared)["gemini-pro"].stream(_REQUEST))
    assert len(slept) == 4


@pytest.mark.asyncio
async def test_an_unfinished_stream_is_a_dropped_connection() -> None:
    reply = httpx2.Response(
        200,
        content=_sse({"candidates": [{"content": {"parts": [{"text": "a"}]}}]}),
    )
    server = _Server(replies=[reply])
    async with _Served(server) as shared:
        with pytest.raises(DisconnectedError, match="before its reply finished"):
            await _stream(Google(resources=shared)["gemini-pro"].stream(_REQUEST))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "kind", "message"),
    [
        (
            httpx2.Response(
                400,
                content=b'{"error": {"message": "The input token count (2000000)'
                b' exceeds the maximum number of tokens allowed (1048576)."}}',
            ),
            ContextOverflowError,
            "input token count",
        ),
        (
            httpx2.Response(429, content=b'[{"error": {"message": "capacity"}}]'),
            RateLimitedError,
            "capacity",
        ),
        (httpx2.Response(418, content=b"short and stout"), RuntimeError, "stout"),
        (httpx2.Response(418, content=b"\xff bad"), RuntimeError, "\ufffd bad"),
        (httpx2.Response(400, content=b"{not json"), RuntimeError, "not json"),
        (httpx2.Response(401, content=b"{}"), AuthError, r"^\{\}$"),
    ],
)
async def test_an_error_reply_is_read_whatever_its_body(
    reply: httpx2.Response,
    kind: type[Exception],
    message: str,
    slept: list[float],
) -> None:
    server = _Server(replies=[reply] * 5)
    async with _Served(server) as shared:
        with pytest.raises(kind, match=message):
            await _stream(Google(resources=shared)["gemini-pro"].stream(_REQUEST))
    del slept


@pytest.mark.asyncio
async def test_a_401_is_not_retried_with_an_api_key() -> None:
    server = _Server(replies=[_failed(401, "UNAUTHENTICATED")])
    async with _Served(server) as shared:
        with pytest.raises(AuthError):
            await _stream(Google(resources=shared)["gemini-pro"].stream(_REQUEST))
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_a_provider_that_can_renew_retries_a_401_once() -> None:
    server = _Server(
        replies=[
            _failed(401, "UNAUTHENTICATED"),
            _ok("fine"),
            _failed(401, "UNAUTHENTICATED"),
            _failed(401, "UNAUTHENTICATED"),
        ],
    )
    async with _Served(server) as shared:
        provider = _Renewing(resources=shared)
        events = await _stream(provider["gemini-pro"].stream(_REQUEST))
        with pytest.raises(AuthError):
            await _stream(provider["gemini-pro"].stream(_REQUEST))
    assert events[0] == TextChunk(text="fine")
    assert [r.headers["x-goog-api-key"] for r in server.requests] == [
        "key-1",
        "key-2",
        "key-2",
        "key-3",
    ]
    assert provider.rejected == ["key-1", "key-2"]


@pytest.mark.asyncio
async def test_a_dropped_stream_keeps_its_advised_wait() -> None:
    async with _Served(_Server(replies=[])) as shared:
        provider = _Dropping(resources=shared)
        seen: list[TextChunk | Response] = []
        with pytest.raises(DisconnectedError, match="flaky") as raised:
            await _collect(provider["gemini-pro"].stream(_REQUEST), into=seen)
    assert raised.value.retry_after_sec == 7.0
    assert seen == [TextChunk(text="so far")]


# ---- count -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_asks_the_server() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b'{"totalTokens": 42}')])
    async with _Served(server) as shared:
        pro = _thinking(Google(resources=shared)["gemini-pro"])
        assert await pro.count(_REQUEST) == 42
    (sent,) = server.requests
    assert str(sent.url) == f"{_BASE}/gemini-3.1-pro-preview:countTokens"
    body = wire.encode(_REQUEST, capability=pro.capability, settings=pro.settings)
    assert _body(sent) == {
        "generateContentRequest": {"model": "models/gemini-3.1-pro-preview", **body},
    }


@pytest.mark.asyncio
async def test_a_count_reply_that_is_no_object_is_refused() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b"[1]")])
    async with _Served(server) as shared:
        with pytest.raises(ReadError):
            await Google(resources=shared)["gemini-pro"].count(_REQUEST)


@pytest.mark.asyncio
async def test_the_envelope_is_asked_for_the_model_streamed_to() -> None:
    server = _Server(replies=[_ok("x")])
    async with _Served(server) as shared:
        provider = _Enveloping(resources=shared)
        await _stream(provider["gemini-flash"].stream(_REQUEST))
    assert provider.models == ["gemini-3.8-flash"]
    assert _body(server.requests[0])["wrapped"] is True


@pytest.mark.asyncio
async def test_count_names_a_model_without_a_vendor_id_by_its_own() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b'{"totalTokens": 1}')])
    async with _Served(server) as shared:
        provider = Google(resources=shared)
        pro = provider["gemini-pro"]
        bare = api.GoogleModel(
            provider=provider,
            capability=_without_vendor_id(pro),
            settings=pro.settings,
        )
        await bare.count(_REQUEST)
    assert str(server.requests[0].url) == f"{_BASE}/gemini-pro-3.1:countTokens"


@pytest.mark.asyncio
async def test_a_stream_names_a_model_without_a_vendor_id_by_its_own() -> None:
    server = _Server(replies=[_ok("x")])
    async with _Served(server) as shared:
        provider = Google(resources=shared)
        pro = provider["gemini-pro"]
        bare = api.GoogleModel(
            provider=provider,
            capability=_without_vendor_id(pro),
            settings=pro.settings,
        )
        await _stream(bare.stream(_REQUEST))
    assert str(server.requests[0].url).startswith(f"{_BASE}/gemini-pro-3.1:")


# ---- the provider ------------------------------------------------------------


def test_the_provider_is_named_by_its_section_unless_named() -> None:
    assert (Google.section, Google().name, Google(name="work").name) == (
        "google",
        "google",
        "work",
    )


def test_the_provider_reports_no_rate_limits() -> None:
    assert Google().usage == ()


def test_the_provider_lists_its_models_and_resolves_roles() -> None:
    provider = Google()
    assert list(provider) == list(models("google"))
    assert "gemini-pro-3.1" in list(provider)
    model = provider["default"]
    assert (model.provider, model.capability.model_id) == (provider, "gemini-argon-4.0")
    with pytest.raises(UnknownModelError):
        provider["gpt-5"]


def test_the_api_serves_google_s_current_models_under_their_ids() -> None:
    """As ai.google.dev's pricing and deprecations pages list them on 2026-10-27.

    ``gemini-3.7-flash`` and ``gemini-3.5-flash`` only route to 3.8 and 3.6 now.
    """
    served = {name: models("google")[name].wire_model_id for name in models("google")}
    assert served == {
        "gemini-argon-4.0": "gemini-4-argon",
        "gemini-pro-3.1": "gemini-3.1-pro-preview",
        "gemini-pro-2.5": "gemini-2.5-pro",
        "gemini-flash-3.8": "gemini-3.8-flash",
        "gemini-flash-3.6": "gemini-3.6-flash",
        "gemini-flash-3.0": "gemini-3-flash-preview",
        "gemini-flash-2.5": "gemini-2.5-flash",
        "gemini-flash-lite-3.5": "gemini-3.5-flash-lite",
        "gemini-flash-lite-3.1": "gemini-3.1-flash-lite",
        "gemini-flash-lite-2.5": "gemini-2.5-flash-lite",
    }
    assert models("google")["default"].model_id == "gemini-argon-4.0"


@pytest.mark.parametrize(
    ("day", "rates"),
    [("2026-11-30", (2.0, 10.0, 0.1)), ("2026-12-01", (4.0, 20.0, 0.2))],
)
def test_argon_s_introductory_rates_double_on_december_first(
    day: str,
    rates: tuple[float, float, float],
) -> None:
    argon = models("google")["gemini-argon"]
    price = argon.prices["auto", 1, date.fromisoformat(day)]
    assert (price.request, price.response, price.cache_read) == rates
    assert (argon.limits.max_request_tokens, argon.limits.max_response_tokens) == (
        1_048_576,
        65_536,
    )
    assert argon.thinking_effort == {"low", "medium", "high"}


def test_the_catalog_is_read_once_per_section() -> None:
    assert models("google") is models("google")
    assert models("google") is not models("google-cli")


@pytest.mark.asyncio
async def test_a_provider_closes_only_resources_it_made() -> None:
    shared = _Closing()
    await Google(resources=shared).aclose()
    assert not shared.closed
    owned = Google()
    resources = owned.resources
    assert isinstance(resources, SharedResources)
    client = resources.http
    await owned.aclose()
    assert client.is_closed


# ---- helpers -----------------------------------------------------------------


def _event(text: str, *, finish: str = "") -> dict[str, object]:
    candidate: dict[str, object] = {"content": {"parts": [{"text": text}]}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


def _sse(*events: Mapping[str, object]) -> bytes:
    return "".join(f"data: {json_dumps(event)}\n\n" for event in events).encode()


def _ok(*texts: str) -> httpx2.Response:
    events = [_event(text) for text in texts]
    events.append(
        {
            **_event("", finish="STOP"),
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
        },
    )
    return httpx2.Response(200, content=_sse(*events))


def _failed(
    status: int,
    kind: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx2.Response:
    body = {"error": {"code": status, "message": kind.lower(), "status": kind}}
    return httpx2.Response(status, headers=headers, content=json_dumps(body).encode())


async def _broken(head: bytes) -> AsyncIterator[bytes]:
    """Stream ``head``, then drop the connection."""
    yield head
    raise httpx2.ReadError("Connection reset by peer")


async def _stream(
    events: AsyncIterator[TextChunk | Response],
) -> list[TextChunk | Response]:
    return [event async for event in events]


async def _collect(
    events: AsyncIterator[TextChunk | Response],
    *,
    into: list[TextChunk | Response],
) -> None:
    """Keep each event in ``into`` as it comes, so a failure leaves those before it."""
    while (event := await anext(events, None)) is not None:
        into.append(event)


def _body(request: httpx2.Request) -> dict[str, object]:
    return json_loads(request.content, dict[str, object])


def _thinking(model: api.GoogleModel) -> api.GoogleModel:
    """Return ``model`` set to think, so its request needs its capability."""
    model.settings = replace(
        model.settings,
        thinking_budget="fixed",
        thinking_effort="high",
    )
    return model


def _without_vendor_id(model: api.GoogleModel) -> ModelCapability:
    return replace(model.capability, wire_model_id="")


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


class _Renewing(Google):
    """A provider whose key rotates whenever the server refuses one."""

    def __init__(self, *, resources: SharedResources) -> None:
        super().__init__(resources=resources)
        self.key = 1
        self.rejected: list[str] = []

    @override
    async def credentials(self) -> dict[str, str]:
        return {"x-goog-api-key": f"key-{self.key}"}

    @override
    async def reauthenticate(self, *, rejected: Mapping[str, str]) -> bool:
        self.rejected.append(rejected["x-goog-api-key"])
        self.key += 1
        return True


class _Enveloping(Google):
    """A provider that wraps each body, remembering the model it was for."""

    def __init__(self, *, resources: SharedResources) -> None:
        super().__init__(resources=resources)
        self.models: list[str] = []

    @override
    async def envelope(
        self,
        body: Mapping[str, object],
        *,
        model: str,
        request: Request,
    ) -> Mapping[str, object]:
        del request
        self.models.append(model)
        return {**body, "wrapped": True}


class _Dropping(Google):
    """A provider whose stream says a little, then drops with an advised wait."""

    @override
    async def events(
        self,
        body: Mapping[str, object],
        *,
        model: str,
        request: Request,
    ) -> AsyncIterator[dict[str, object]]:
        del body, model, request
        yield _event("so far")
        raise DisconnectedError("flaky", retry_after_sec=7.0)


class _Closing(SharedResources):
    """Shared resources that remember being closed."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    @override
    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


# The level each model thinks at when sent none (ai.google.dev, "Gemini
# thinking", read 2026-10-27); 3.1 Pro is Gemini 3 Pro's ``high``. 2.5 Flash-Lite
# does not think unasked; ``agy`` serves 3.8 Flash at high.
@pytest.mark.parametrize(
    ("section", "model", "effort"),
    [
        ("google", "gemini-argon-4.0", "medium"),
        ("google", "gemini-pro-3.1", "high"),
        ("google", "gemini-flash-3.8", "medium"),
        ("google", "gemini-flash-3.6", "medium"),
        ("google", "gemini-flash-3.0", "high"),
        ("google", "gemini-flash-lite-3.5", "min"),
        ("google", "gemini-flash-lite-3.1", "min"),
        ("google", "gemini-flash-lite-2.5", "none"),
        ("google-cli", "gemini-flash-3.8", "high"),
    ],
)
def test_a_model_starts_at_the_level_it_thinks_at_unasked(
    section: str,
    model: str,
    effort: str,
) -> None:
    assert models(section)[model].default.thinking_effort == effort


def test_an_attachment_is_billed_by_the_vendor_s_formula() -> None:
    model = Google()["default"]
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360f8cf000000030101005c0d6a1c0000000049454e44ae426082",
    )
    attached = Attachment(mime_descriptor="image/png", data=png)
    assert model.attachment_tokens(attached) == wire.attachment_tokens(
        attached,
        limits=model.capability.limits,
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
