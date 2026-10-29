"""Tests for Claude over the Messages API, against a fake HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, override

from treekle import from_plain, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import (
    AuthError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    UnknownModelError,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.anthropic import api, wire
from sagent.providers.anthropic.api import Anthropic, models
from sagent.providers.catalog import listings
from sagent.providers.resources import SharedResources
from sagent.providers.types import Login, UsageWindow
from sagent.session.types import AssistantMessage, Attachment, UserMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from sagent.model.capability import ThinkingBudget
    from sagent.model.cost import ServiceTier


_REQUEST = Request(system=("Be terse.",), records=(UserMessage(content="hi"),))


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record each backoff instead of waiting it out."""
    waits: list[float] = []

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)

    monkeypatch.setattr(api, "asyncio", SimpleNamespace(sleep=sleep))
    return waits


def _sse(*events: Mapping[str, object]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json_dumps(event)}\n\n" for event in events
    ).encode()


def _said(text: str, **usage: object) -> tuple[dict[str, object], ...]:
    """Return the events of a reply that says ``text``."""
    return (
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1, **usage},
        },
        {"type": "message_stop"},
    )


def _ok(
    text: str = "Hello",
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx2.Response:
    return httpx2.Response(200, headers=headers, content=_sse(*_said(text)))


def _failed(
    status: int,
    kind: str,
    message: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return httpx2.Response(status, headers=headers, content=json_dumps(body).encode())


async def _stream(
    model: api.AnthropicModel,
    request: Request = _REQUEST,
) -> list[object]:
    return [event async for event in model.stream(request)]


def _body(request: httpx2.Request) -> dict[str, object]:
    return json_loads(request.content, dict[str, object])


# ---- credentials and the request ---------------------------------------------


@pytest.mark.asyncio
async def test_credentials_are_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await Anthropic().credentials() == {"x-api-key": "sk-test"}
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(AuthError, match=r"^ANTHROPIC_API_KEY is not set\.$"):
        await Anthropic().credentials()


@pytest.mark.asyncio
async def test_stream_posts_the_request_and_yields_text_then_the_response() -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        opus = Anthropic(resources=shared)["opus"]
        events = await _stream(opus)
    (sent,) = server.requests
    assert (sent.method, str(sent.url)) == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert sent.headers["x-api-key"] == "sk-test"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.headers["content-type"] == "application/json"
    # Thinking, on unasked, keeps its old turns by the context-management beta.
    assert sent.headers["anthropic-beta"] == "context-management-2025-06-27"
    body, _ = wire.encode(
        _REQUEST,
        capability=opus.capability,
        settings=opus.settings,
        system=wire.system_blocks(_REQUEST.system, settings=opus.settings),
    )
    assert _body(sent) == body
    assert events[0] == TextChunk(text="Hello")
    assert len(events) == 2
    response = events[1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hello"),)
    assert (response.tokens.request, response.tokens.response) == (3, 1)


@pytest.mark.asyncio
async def test_the_betas_a_request_needs_ride_one_header() -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        opus = Anthropic(resources=shared)["opus"]
        opus.settings = replace(
            opus.settings,
            thinking_budget="auto",
            thinking_output="redacted",
            service_tier="priority",
        )
        await _stream(opus)
    assert server.requests[0].headers["anthropic-beta"] == (
        "redact-thinking-2026-02-12,context-management-2025-06-27,fast-mode-2026-02-01"
    )


@pytest.mark.asyncio
async def test_each_reply_updates_the_accounts_windows() -> None:
    limits = {
        "anthropic-ratelimit-unified-5h-utilization": "0.25",
        "anthropic-ratelimit-unified-5h-status": "allowed",
    }
    server = _Server(replies=[_ok(headers=limits), _ok()])
    async with _Served(server) as shared:
        anthropic = Anthropic(resources=shared)
        assert anthropic.usage == ()
        await _stream(anthropic["opus"])
        window = UsageWindow(label="5h", utilization=0.25)
        assert anthropic.usage == (window,)
        # A reply that reports no window leaves the last report standing.
        await _stream(anthropic["opus"])
        assert anthropic.usage == (window,)


# ---- failures ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_error_reply_raises_what_it_means(slept: list[float]) -> None:
    server = _Server(
        replies=[
            _failed(400, "invalid_request_error", "prompt is too long: 300000 tokens"),
            httpx2.Response(400, content=b"<html>Bad Request</html>"),
            httpx2.Response(400, content=b"{truncated"),
        ],
    )
    async with _Served(server) as shared:
        opus = Anthropic(resources=shared)["opus"]
        with pytest.raises(ContextOverflowError) as overflow:
            await _stream(opus)
        # The message is the error's own, read out of the JSON body.
        assert overflow.value.args == ("prompt is too long: 300000 tokens",)
        with pytest.raises(RuntimeError, match=r"400 : <html>Bad Request</html>"):
            await _stream(opus)
        # A body that only looks like JSON is read as text.
        with pytest.raises(RuntimeError, match=r"400 : \{truncated"):
            await _stream(opus)
    assert len(server.requests) == 3
    assert slept == []


@pytest.mark.asyncio
async def test_a_short_failure_before_the_first_event_is_retried(
    slept: list[float],
) -> None:
    server = _Server(
        replies=[
            _failed(529, "overloaded_error", "Overloaded"),
            _failed(429, "rate_limit_error", "slow", headers={"retry-after": "2"}),
            _ok(),
        ],
    )
    async with _Served(server) as shared:
        events = await _stream(Anthropic(resources=shared)["opus"])
    assert events[0] == TextChunk(text="Hello")
    assert len(server.requests) == 3
    # Doubling backoff, or the server's advice when it is longer.
    assert slept == [0.5, 2.0]


@pytest.mark.asyncio
async def test_retrying_stops_after_five_attempts(slept: list[float]) -> None:
    server = _Server(replies=[_failed(529, "overloaded_error", "Overloaded")] * 5)
    async with _Served(server) as shared:
        with pytest.raises(OverloadedError, match="Overloaded"):
            await _stream(Anthropic(resources=shared)["opus"])
    assert len(server.requests) == 5
    assert slept == [0.5, 1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_a_long_advised_wait_is_left_to_the_caller(slept: list[float]) -> None:
    server = _Server(
        replies=[
            _failed(429, "rate_limit_error", "slow", headers={"retry-after": "31"}),
        ],
    )
    async with _Served(server) as shared:
        with pytest.raises(RateLimitedError) as raised:
            await _stream(Anthropic(resources=shared)["opus"])
    assert raised.value.retry_after_sec == 31.0
    assert len(server.requests) == 1
    assert slept == []
    # Thirty seconds is still short enough to wait out here.
    server = _Server(
        replies=[
            _failed(429, "rate_limit_error", "slow", headers={"retry-after": "30"}),
            _ok(),
        ],
    )
    async with _Served(server) as shared:
        await _stream(Anthropic(resources=shared)["opus"])
    assert slept == [30.0]


@pytest.mark.asyncio
async def test_a_dropped_connection_is_a_disconnect_and_is_retried(
    slept: list[float],
) -> None:
    refused = httpx2.ConnectError("Connection refused")
    server = _Server(replies=[refused, _ok()])
    async with _Served(server) as shared:
        events = await _stream(Anthropic(resources=shared)["opus"])
    assert events[0] == TextChunk(text="Hello")
    assert slept == [0.5]
    server = _Server(replies=[refused] * 5)
    async with _Served(server) as shared:
        with pytest.raises(DisconnectedError, match="Connection refused"):
            await _stream(Anthropic(resources=shared)["opus"])
    assert len(server.requests) == 5


@pytest.mark.asyncio
async def test_a_failure_after_text_has_streamed_is_not_retried(
    slept: list[float],
) -> None:
    """Text the caller already showed cannot be taken back."""
    server = _Server(
        replies=[httpx2.Response(200, content=_broken(_said("Partial")[:3]))],
    )
    async with _Served(server) as shared:
        stream = Anthropic(resources=shared)["opus"].stream(_REQUEST)
        assert await anext(stream) == TextChunk(text="Partial")
        with pytest.raises(DisconnectedError, match="reset"):
            await anext(stream)
    assert len(server.requests) == 1
    assert slept == []


@pytest.mark.asyncio
async def test_a_rejected_api_key_is_not_renewed(slept: list[float]) -> None:
    server = _Server(
        replies=[_failed(401, "authentication_error", "invalid x-api-key")],
    )
    async with _Served(server) as shared:
        anthropic = Anthropic(resources=shared)
        assert await anthropic.reauthenticate() is False
        with pytest.raises(AuthError, match="invalid x-api-key"):
            await _stream(anthropic["opus"])
    assert len(server.requests) == 1
    assert slept == []


@pytest.mark.asyncio
async def test_a_401_renews_credentials_and_retries_once(slept: list[float]) -> None:
    server = _Server(
        replies=[_failed(401, "authentication_error", "expired"), _ok()],
    )
    async with _Served(server) as shared:
        renewing = _Renewing(resources=shared)
        await _stream(renewing["opus"])
    assert [sent.headers["x-api-key"] for sent in server.requests] == ["old", "new"]
    server = _Server(replies=[_failed(401, "authentication_error", "revoked")] * 2)
    async with _Served(server) as shared:
        renewing = _Renewing(resources=shared)
        with pytest.raises(AuthError, match="revoked"):
            await _stream(renewing["opus"])
    assert renewing.renewals == 1
    assert len(server.requests) == 2
    assert slept == []


@pytest.mark.asyncio
async def test_a_providers_own_betas_lead_and_none_repeats() -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        opus = _Renewing(resources=shared)["opus"]
        opus.settings = replace(opus.settings, thinking_budget="auto")
        await _stream(opus)
    assert server.requests[0].headers["anthropic-beta"] == (
        "context-management-2025-06-27,interleaved-thinking-2025-05-14"
    )


# ---- count -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "betas"),
    [
        ("priority", "context-management-2025-06-27,fast-mode-2026-02-01"),
        ("default", "context-management-2025-06-27"),
    ],
)
@pytest.mark.asyncio
async def test_count_asks_the_server_without_generation_knobs(
    tier: ServiceTier,
    betas: str,
) -> None:
    server = _Server(
        replies=[
            httpx2.Response(200, content=json_dumps({"input_tokens": 42}).encode()),
        ],
    )
    async with _Served(server) as shared:
        opus = Anthropic(resources=shared)["opus"]
        opus.settings = replace(
            opus.settings,
            thinking_budget="auto",
            thinking_effort="high",
            service_tier=tier,
        )
        assert await opus.count(_REQUEST) == 42
    (sent,) = server.requests
    assert str(sent.url) == "https://api.anthropic.com/v1/messages/count_tokens"
    body = _body(sent)
    assert body.keys() == {
        "model",
        "messages",
        "system",
        "thinking",
        "context_management",
    }
    assert body["model"] == "claude-opus-5-5"
    assert sent.headers["anthropic-beta"] == betas


# ---- the prompt cache --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cache_lapses_only_after_an_answer_left_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1_000.0]
    monkeypatch.setattr(api, "time", SimpleNamespace(time=lambda: now[0]))
    server = _Server(replies=[_ok(), _ok(), _ok()])
    async with _Served(server) as shared:
        opus = Anthropic(resources=shared)["opus"]
        opus.settings = replace(opus.settings, thinking_budget="auto")
        assert opus.settings.cache_ttl_sec == 300.0
        await _stream(opus)  # Never answered before, so nothing has lapsed.
        now[0] += 300.0
        await _stream(opus)  # Idle exactly the cache's lifetime.
        now[0] += 301.0
        await _stream(opus)
    kept = [_kept_thinking(_body(sent)) for sent in server.requests]
    assert kept == ["all", "all", {"type": "thinking_turns", "value": 1}]


# ---- models ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("opus", "opus-5.5"),
        ("opus-4", "opus-4.8"),
        ("opus-5.0", "opus-5.0"),
        ("claude-opus-4-8", "opus-4.8"),
        ("default", "opus-5.5"),
        ("best", "fable-5.1"),
        ("everyday", "sonnet-5.5"),
        ("utility", "haiku-5.5"),
    ],
)
def test_a_model_answers_to_its_name_role_family_or_vendor_id(
    name: str,
    model_id: str,
) -> None:
    model = Anthropic()[name]
    assert model.capability.model_id == model_id
    assert model.provider.name == "anthropic"


def test_a_context_tag_selects_its_size() -> None:
    model = Anthropic()["opus+200k"]
    assert model.settings.context_size.tokens == 200_000
    assert model.capability.limits.max_request_tokens == 1_000_000
    with pytest.raises(UnknownModelError, match="'gpt-5'"):
        Anthropic()["gpt-5"]


def test_iterating_lists_the_sections_models() -> None:
    names = list(Anthropic())
    assert names[:3] == ["fable-5.1", "fable-5.0", "mythos-5.1"]
    assert len(names) == 16
    assert "default" not in names


def test_a_provider_is_named_for_its_section_unless_named() -> None:
    assert Anthropic().name == "anthropic"
    assert Anthropic(name="work").name == "work"
    assert not isinstance(Anthropic(), Login)  # An API key needs no sign-in.


@pytest.mark.asyncio
async def test_a_provider_closes_only_resources_it_made() -> None:
    own = Anthropic()
    client = own.resources.http
    await own.aclose()
    assert client.is_closed
    async with SharedResources() as shared:
        client = shared.http
        await Anthropic(resources=shared).aclose()
        assert not client.is_closed


# ---- the catalog's facts -----------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("fable-5.1", "June 2026"),
        ("opus-5.0", "May 2026"),
        ("opus-4.6", "May 2025"),
        ("sonnet-4.6", "August 2025"),
        ("haiku-4.5", "February 2025"),
    ],
)
def test_every_row_states_its_published_cutoff(model_id: str, cutoff: str) -> None:
    assert models("anthropic")[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("model_id", "rates"),
    [
        ("fable-5.1", (10.0, 50.0, 12.5, 20.0, 0.25)),
        ("opus-5.5", (4.0, 20.0, 5.0, 8.0, 0.2)),
        ("sonnet-4.6", (3.0, 15.0, 3.75, 6.0, 0.3)),
        ("haiku-5.5", (0.1, 0.5, 0.125, 0.2, 0.01)),
    ],
)
def test_a_row_bills_the_published_standard_card(
    model_id: str,
    rates: tuple[float, float, float, float, float],
) -> None:
    card = models("anthropic")[model_id].prices["auto", 1, date(2026, 10, 22)]
    assert (
        card.request,
        card.response,
        card.cache_write,
        card.cache_write_1h,
        card.cache_read,
    ) == rates


def test_only_the_fast_mode_opus_models_offer_priority() -> None:
    table = models("anthropic")
    fast = {name for name, row in table.items() if "priority" in row.service_tier}
    assert fast == {"opus-5.5", "opus-5.0", "opus-4.8"}
    card = table["opus-5.0"].prices["priority", 1, date(2026, 10, 22)]
    assert (card.request, card.cache_write, card.cache_read) == (10.0, 12.5, 1.0)


def test_dated_and_banded_cards_bill_their_published_rates() -> None:
    table = models("anthropic")
    sonnet = table["sonnet-5.5"].prices
    before, after = date(2026, 10, 6), date(2026, 10, 7)
    assert sonnet["auto", 1, before].cache_read == 0.2
    assert sonnet["auto", 1, after].cache_read == 0.1
    haiku = table["haiku-5.5"].prices
    at = date(2026, 10, 22)
    assert haiku["auto", 100_000, at].request == 0.1
    assert haiku["auto", 100_001, at].request == 0.5


@pytest.mark.parametrize("model_id", ["fable-5.1", "opus-5.5", "sonnet-5.0"])
def test_1m_models_serve_1m(model_id: str) -> None:
    limits = models("anthropic")[model_id].limits
    assert limits.max_request_tokens == 1_000_000
    assert limits.max_response_tokens == 128_000


@pytest.mark.parametrize("model_id", ["opus-4.5", "sonnet-4.5", "haiku-4.5"])
def test_pre_4_6_models_have_a_200k_window(model_id: str) -> None:
    limits = models("anthropic")[model_id].limits
    assert limits.max_request_tokens == 200_000
    assert limits.max_response_tokens == 64_000


def test_small_is_haikus_cheaper_band_and_opus_has_none() -> None:
    assert Anthropic()["haiku-5.5+smallcontext"].settings.context_size.tokens == 100_000
    assert (
        Anthropic()["opus-5.5+smallcontext"].settings.context_size.tokens == 1_000_000
    )


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        # ``enabled`` is rejected from 4.7 on: adaptive only.
        ("fable-5.1", {"none", "auto"}),
        ("opus-4.8", {"none", "auto"}),
        ("opus-4.6", {"none", "auto", "fixed"}),
        # The 4.5 generation rejects ``adaptive``.
        ("opus-4.5", {"none", "fixed"}),
        ("haiku-4.5", {"none", "fixed"}),
    ],
)
def test_each_generation_offers_its_measured_budget_modes(
    model_id: str,
    budget: set[ThinkingBudget],
) -> None:
    assert models("anthropic")[model_id].thinking_budget == budget


def test_xhigh_effort_is_a_4_7_and_later_level() -> None:
    table = models("anthropic")
    assert table["fable-5.1"].thinking_effort == {
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }
    assert "xhigh" not in table["opus-4.6"].thinking_effort


@pytest.mark.parametrize(
    ("model_id", "edge_px", "chars_per_token"),
    [("opus-5.5", 2576, 2.38), ("opus-4.6", 1568, 3.12), ("sonnet-4.5", 1568, 3.12)],
)
def test_image_limits_and_tokenizer_follow_the_generation(
    model_id: str,
    edge_px: int,
    chars_per_token: float,
) -> None:
    row = models("anthropic")[model_id]
    assert row.approx_chars_per_token == chars_per_token
    assert row.limits.max_image_edge_px == edge_px
    # 10 MB base64-encoded on the wire is 7.5 MB raw.
    assert row.limits.max_image_bytes == 7_500_000
    assert row.limits.max_request_bytes == 32 * 1024 * 1024


def test_the_api_offers_both_cache_lifetimes_and_retries_itself() -> None:
    row = models("anthropic")["opus"]
    assert row.cache_ttl_sec == {300.0, 3600.0}
    assert row.manage_context_server_side == {False, True}
    assert row.service_tier == {"auto", "default", "priority"}
    assert row.thinking_output == {"none", "text", "redacted"}
    assert row.retries_internally
    assert not row.account_auth


def test_the_catalog_is_read_once_per_section() -> None:
    assert models("anthropic") is models("anthropic")
    assert models("anthropic") is not models("anthropic-cli")


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


class _Renewing(Anthropic):
    """A provider whose key the server rejects until it is renewed."""

    betas: ClassVar[tuple[str, ...]] = (
        "context-management-2025-06-27",
        "interleaved-thinking-2025-05-14",
    )

    def __init__(self, *, resources: SharedResources) -> None:
        super().__init__(resources=resources)
        self.renewals = 0

    @override
    async def credentials(self) -> dict[str, str]:
        return {"x-api-key": "new" if self.renewals else "old"}

    @override
    async def reauthenticate(self) -> bool:
        self.renewals += 1
        return True


async def _broken(events: tuple[Mapping[str, object], ...]) -> AsyncIterator[bytes]:
    """Stream ``events``, then drop the connection."""
    yield _sse(*events)
    raise httpx2.ReadError("Connection reset by peer")


def _kept_thinking(body: Mapping[str, object]) -> object:
    management = from_plain(
        body["context_management"],
        dict[str, list[dict[str, object]]],
    )
    (edit,) = management["edits"]
    return edit["keep"]


# Claude Code 2.1.309's own choice, captured on 2026-10-27 with no settings: the
# ``output_config.effort`` it sent each model. The 4.5 Sonnet and Haiku take none.
@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("fable-5.1", "high"),
        ("fable-5.0", "high"),
        ("mythos-5.1", "high"),
        ("mythos-5.0", "high"),
        ("opus-5.5", "medium"),
        ("opus-5.0", "high"),
        ("opus-4.8", "high"),
        ("opus-4.7", "xhigh"),
        ("opus-4.6", "high"),
        ("opus-4.5", "high"),
        ("sonnet-5.5", "medium"),
        ("sonnet-5.0", "high"),
        ("sonnet-4.6", "high"),
        ("sonnet-4.5", "none"),
        ("haiku-5.5", "medium"),
        ("haiku-4.5", "none"),
    ],
)
def test_a_model_starts_at_the_effort_claude_code_gives_it(
    model: str,
    effort: str,
) -> None:
    for section in _messages_api_sections():
        capability, settings = models(section).resolve(model)
        assert (capability.model_id, settings.thinking_effort) == (model, effort)


# The thinking Claude Code 2.1.309 sent each model, captured likewise: adaptive,
# except a 31,999-token budget to the 4.5 models.
@pytest.mark.parametrize(
    ("model", "budget"),
    [
        ("fable-5.1", "auto"),
        ("opus-5.5", "auto"),
        ("opus-4.6", "auto"),
        ("sonnet-4.6", "auto"),
        ("haiku-5.5", "auto"),
        ("opus-4.5", "fixed"),
        ("sonnet-4.5", "fixed"),
        ("haiku-4.5", "fixed"),
    ],
)
def test_a_model_thinks_as_claude_code_has_it(model: str, budget: str) -> None:
    for section in _messages_api_sections():
        _, settings = models(section).resolve(model)
        assert settings.thinking_budget == budget


def _messages_api_sections() -> list[str]:
    """Return each catalog section served over the Messages API, as ``Anthropic`` is."""
    sections = [
        listing.section
        for listing in listings()
        if isinstance(listing.factory, type) and issubclass(listing.factory, Anthropic)
    ]
    assert "anthropic" in sections
    return sections


def test_an_attachment_is_billed_by_the_vendor_s_formula() -> None:
    model = Anthropic()["default"]
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
