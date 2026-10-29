"""Tests for OpenAI over a ChatGPT login, against a fake HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, override

import base64
import re
import time

from treekle import from_plain, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.model.types import Request, Response
from sagent.providers.openai import responses, token_count, wire
from sagent.providers.openai.lease import LeasedToken
from sagent.providers.openai.sub import OpenAISub
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, UserMessage


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_REQUEST = Request(system=("Be terse.",), records=(UserMessage(content="hi"),))


def _jwt(name: str, *, expires_in_sec: float = 3_600.0) -> str:
    def part(value: object) -> str:
        return (
            base64.urlsafe_b64encode(json_dumps(value).encode()).rstrip(b"=").decode()
        )

    return f"{part({'alg': 'none'})}.{part({'sub': name, 'exp': time.time() + expires_in_sec})}.sig"


@pytest.fixture
def auth(tmp_path: Path) -> Path:
    path = tmp_path / "auth.json"
    tokens = {
        "access_token": _jwt("first"),
        "refresh_token": "r1",
        "account_id": "acct-1",
    }
    path.write_text(json_dumps({"auth_mode": "chatgpt", "tokens": tokens}))
    return path


def _ok() -> httpx2.Response:
    events = (
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hi"}],
            },
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    )
    headers = {
        "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-used-percent": "5",
    }
    content = "".join(f"data: {json_dumps(event)}\n\n" for event in events).encode()
    return httpx2.Response(200, headers=headers, content=content)


def test_the_subscription_serves_its_sections_clamped_models() -> None:
    sub = OpenAISub()
    assert sub.name == "openai-sub"
    luna = sub["utility"]
    assert luna.capability.account_auth is True
    assert luna.capability.limits.max_request_tokens == 272_000
    assert luna.capability.limits.max_response_tokens == 32_000
    assert sorted(luna.capability.service_tier) == ["auto", "priority"]
    assert sorted(sub["gpt-nano-5.4"].capability.service_tier) == ["auto"]


@pytest.mark.asyncio
async def test_stream_reaches_the_codex_endpoint_as_codex_does(auth: Path) -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        sub = OpenAISub(resources=shared, credentials=auth)
        luna = sub["utility"]
        events = [event async for event in luna.stream(_REQUEST)]
        assert [window.label for window in sub.usage] == ["7d"]
    (sent,) = server.requests
    assert str(sent.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert sent.headers["authorization"] == f"Bearer {_access(auth)}"
    assert sent.headers["chatgpt-account-id"] == "acct-1"
    # A first-party originator, or ChatGPT's usage page files the requests under
    # "Other" (codex-rs default_client.rs, is_first_party_originator).
    assert sent.headers["originator"] == "codex_cli_rs"
    assert re.fullmatch(
        rf"codex_cli_rs/{re.escape(OpenAISub.codex_version)} \([^;()]+; [^;()]+\) \S+",
        sent.headers["user-agent"],
    )
    # Only the underscored name caches; Codex 0.153's ``session-id`` cached
    # nothing when probed on 2026-10-27.
    assert sent.headers["session_id"] == wire.cache_key(_REQUEST)
    body = json_loads(sent.content, dict[str, object])
    assert body == responses.encode(
        _REQUEST,
        capability=luna.capability,
        settings=luna.settings,
    )
    assert body["prompt_cache_key"] == sent.headers["session_id"]
    response = events[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hi"),)


# As Codex 0.153.0 sent it, captured on 2026-10-27.
@pytest.mark.asyncio
async def test_a_request_names_its_kind_as_codex_does(auth: Path) -> None:
    server = _Server(replies=[_ok(), _ok()])
    async with _Served(server) as shared:
        luna = OpenAISub(resources=shared, credentials=auth)["utility"]
        [event async for event in luna.stream(_REQUEST)]
        [event async for event in luna.stream(replace(_REQUEST, purpose="compaction"))]
    kinds = [
        json_loads(sent.headers["x-codex-turn-metadata"], dict[str, object])[
            "request_kind"
        ]
        for sent in server.requests
    ]
    assert kinds == ["turn", "compaction"]


def test_the_turn_metadata_names_the_session_window_and_compaction(
    auth: Path,
) -> None:
    asked = replace(
        _REQUEST,
        purpose="compaction",
        compaction="auto",
        session="s-1",
        agent="s-1",
        name="/root",
        window=2,
        turn=3,
        step=1,
        turn_started_ms=1_793_135_007_400,
    )
    sent = OpenAISub(credentials=auth).headers(asked)
    assert sent["session-id"] == "s-1"
    assert sent["x-codex-window-id"] == "s-1:2"
    assert sent["x-client-request-id"] == asked.call_id
    assert "x-openai-subagent" not in sent
    assert json_loads(sent["x-codex-turn-metadata"], dict[str, object]) == {
        "session_id": "s-1",
        "thread_id": "s-1",
        "agent_name": "/root",
        "turn_id": asked.prompt_id,
        "root_turn_id": asked.prompt_id,
        "turn_started_at_unix_ms": 1_793_135_007_400,
        "window_id": "s-1:2",
        "window_number": 2,
        "request_kind": "compaction",
        "compaction": {"trigger": "auto"},
    }


# As Codex 0.153.0 sent a spawned agent's calls, captured on 2026-10-27.
def test_a_spawned_agents_calls_name_its_thread_and_its_parents(auth: Path) -> None:
    asked = replace(
        _REQUEST,
        session="s-1",
        agent="a-2",
        parent="s-1",
        parent_turn=4,
        root_prompt="r-9",
        name="/root/helper",
        turn=1,
    )
    sent = OpenAISub(credentials=auth).headers(asked)
    assert sent["session-id"] == "s-1"
    assert sent["x-openai-subagent"] == "collab_spawn"
    assert sent["x-codex-parent-thread-id"] == "s-1"
    assert sent["x-codex-window-id"] == "a-2:0"
    assert json_loads(sent["x-codex-turn-metadata"], dict[str, object]) == {
        "session_id": "s-1",
        "thread_id": "a-2",
        "agent_name": "/root/helper",
        "turn_id": asked.prompt_id,
        "root_turn_id": "r-9",
        "turn_started_at_unix_ms": 0,
        "window_id": "a-2:0",
        "window_number": 0,
        "request_kind": "turn",
        "parent_thread_id": "s-1",
        "parent_turn_id": Request(agent="s-1", turn=4).prompt_id,
        "subagent_kind": "thread_spawn",
        "thread_source": "subagent",
    }


# Codex replays the token a reply sets on every later request of that turn
# (codex-rs core/src/client.rs, ``x-codex-turn-state``).
@pytest.mark.asyncio
async def test_a_turns_routing_token_is_sent_back_within_that_turn_only(
    auth: Path,
) -> None:
    routed = _ok()
    routed.headers["x-codex-turn-state"] = "tok-1"
    server = _Server(replies=[routed, _ok(), _ok()])
    turn = replace(_REQUEST, session="s-1", agent="s-1", turn=1)
    async with _Served(server) as shared:
        luna = OpenAISub(resources=shared, credentials=auth)["utility"]
        for asked in (turn, turn, replace(turn, turn=2)):
            [event async for event in luna.stream(asked)]
    sent = [r.headers.get("x-codex-turn-state") for r in server.requests]
    assert sent == [None, "tok-1", None]


@pytest.mark.asyncio
async def test_no_temperature_is_sent_even_to_a_model_that_cannot_reason(
    auth: Path,
) -> None:
    server = _Server(replies=[_ok()])
    async with _Served(server) as shared:
        model = OpenAISub(resources=shared, credentials=auth)["gpt-4.1"]
        [event async for event in model.stream(_REQUEST)]
    assert "temperature" not in json_loads(
        server.requests[0].content,
        dict[str, object],
    )


@pytest.mark.asyncio
async def test_a_401_renews_the_login_and_asks_again(auth: Path) -> None:
    renewed = _jwt("renewed")
    server = _Server(
        replies=[
            httpx2.Response(401, content=b'{"error": {"message": "expired"}}'),
            httpx2.Response(
                200,
                content=json_dumps(
                    {"access_token": renewed, "refresh_token": "r2"},
                ).encode(),
            ),
            _ok(),
        ],
    )
    async with _Served(server) as shared:
        model = OpenAISub(resources=shared, credentials=auth)["utility"]
        events = [event async for event in model.stream(_REQUEST)]
    assert isinstance(events[-1], Response)
    first, refresh, again = server.requests
    assert str(refresh.url) == "https://auth.openai.com/oauth/token"
    assert again.headers["authorization"] == f"Bearer {renewed}"
    assert first.headers["authorization"] != again.headers["authorization"]


@pytest.mark.asyncio
async def test_count_estimates_with_the_models_tokenizer(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def get_encoding(name: str) -> SimpleNamespace:
        asked.append(name)
        return SimpleNamespace(encode_ordinary=list)

    monkeypatch.setattr(
        token_count,
        "tiktoken",
        SimpleNamespace(get_encoding=get_encoding),
    )
    async with SharedResources() as shared:
        sub = OpenAISub(resources=shared, credentials=auth)
        assert await sub["utility"].count(_REQUEST) == len("Be terse.\nhi")
    assert asked == ["o200k_base"]


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [True, False])
async def test_login_signs_in_through_the_codex_login(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
    headless: bool,
) -> None:
    signed: list[bool] = []

    async def sign_in(self: object, *, headless: bool = False) -> None:
        del self
        signed.append(headless)

    sub = OpenAISub(credentials=auth)
    monkeypatch.setattr(type(sub._login), "sign_in", sign_in)
    await sub.login(headless=headless)
    assert signed == [headless]
    await sub.aclose()


@pytest.mark.asyncio
async def test_a_borrowing_subscription_asks_its_lender_never_a_file() -> None:
    lent: list[str] = []

    async def lend(*, rejected: str) -> LeasedToken:
        lent.append(rejected)
        return LeasedToken(
            access_token=f"lent-{len(lent)}",
            account_id="acct-lent",
            expires_at=time.time() + 3_600.0,
        )

    refused = httpx2.Response(401, content=b'{"error": {"message": "expired"}}')
    server = _Server(replies=[refused, _ok()])
    async with _Served(server) as shared:
        sub = OpenAISub(resources=shared, lease=lend)
        events = [event async for event in sub["utility"].stream(_REQUEST)]
        with pytest.raises(AuthError, match="lent by another process"):
            await sub.login()
    assert isinstance(events[-1], Response)
    assert [sent.headers["authorization"] for sent in server.requests] == [
        "Bearer lent-1",
        "Bearer lent-2",
    ]
    assert server.requests[1].headers["chatgpt-account-id"] == "acct-lent"
    # The refused token is named, so the lender knows to renew it.
    assert lent == ["", "lent-1"]


def _access(auth: Path) -> str:
    tokens = json_loads(auth.read_text(), dict[str, object])["tokens"]
    return from_plain(tokens, dict[str, str])["access_token"]


@dataclass(slots=True, kw_only=True)
class _Server:
    """Answers each request with the next scripted reply, and keeps the requests."""

    replies: list[httpx2.Response]
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.replies.pop(0)


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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
