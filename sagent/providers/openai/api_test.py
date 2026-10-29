"""Tests for OpenAI over the Responses API, against a fake HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, override

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.types import Request, Response
from sagent.providers.openai import responses
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.sub import OpenAISub
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, UserMessage
from sagent.tools.read import Read


if TYPE_CHECKING:
    from collections.abc import Callable


_REQUEST = Request(
    system=("Be terse.",),
    records=(UserMessage(content="hi"),),
    tools=(Read(),),
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")


def _sse(*events: dict[str, object]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json_dumps(e)}\n\n" for e in events
    ).encode()


def test_the_provider_serves_openais_models_by_any_name() -> None:
    openai = OpenAI()
    assert openai.name == "openai"
    assert "luna-6.0" in list(openai)
    assert openai["utility"].capability.wire_model_id == "gpt-6-luna"
    # The API offers every window and tier the model has.
    assert openai["sol-6.0+smallcontext"].settings.context_size.tokens == 272_000
    assert "flex" in openai["sol-6.0"].capability.service_tier


def test_a_model_that_cannot_reason_is_sent_the_sampling_temperature() -> None:
    openai = OpenAI()
    gpt = openai["gpt-4.1"]
    assert openai.encode(_REQUEST, model=gpt)["temperature"] == 1.0
    assert "temperature" not in openai.encode(_REQUEST, model=openai["luna-6.0"])


@pytest.mark.asyncio
async def test_stream_posts_a_responses_request_and_reads_its_events() -> None:
    reply = _sse(
        {"type": "response.output_text.delta", "delta": "Hi"},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hi"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 9, "output_tokens": 1},
            },
        },
    )
    server = _Server(replies=[httpx2.Response(200, content=reply)])
    async with _Served(server) as shared:
        luna = OpenAI(resources=shared)["luna-6.0"]
        events = [event async for event in luna.stream(_REQUEST)]
    (sent,) = server.requests
    assert str(sent.url) == "https://api.openai.com/v1/responses"
    assert sent.headers["authorization"] == "Bearer sk-openai"
    assert json_loads(sent.content, dict[str, object]) == responses.encode(
        _REQUEST,
        capability=luna.capability,
        settings=luna.settings,
        temperature=1.0,
    )
    response = events[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="Hi"),)
    assert response.tokens.request == 9


@pytest.mark.asyncio
async def test_count_asks_the_server_with_only_what_it_counts() -> None:
    answer = json_dumps(
        {"object": "response.input_tokens", "input_tokens": 123},
    ).encode()
    server = _Server(replies=[httpx2.Response(200, content=answer)])
    async with _Served(server) as shared:
        luna = OpenAI(resources=shared)["luna-6.0"]
        luna.settings = replace(luna.settings, thinking_effort="low")
        assert await luna.count(_REQUEST) == 123
    (sent,) = server.requests
    assert str(sent.url) == "https://api.openai.com/v1/responses/input_tokens"
    body = json_loads(sent.content, dict[str, object])
    encoded = responses.encode(
        _REQUEST,
        capability=luna.capability,
        settings=luna.settings,
    )
    assert sorted(body) == [
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "tools",
    ]
    assert body == {key: encoded[key] for key in body}


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


# Where Codex 0.153.0 serves a model, the effort it picks unasked: the
# ``default_reasoning_level`` of /backend-api/codex/models, else of its bundled
# models.json. Elsewhere the effort the API reported using when sent none.
# Both read on 2026-10-27.
@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("astra-6.0", "medium"),
        ("sol-6.1", "low"),
        ("sol-6.0", "medium"),
        ("luna-6.0", "medium"),
        ("sol-5.6", "low"),
        ("gpt-5.6", "medium"),
        ("luna-5.6", "medium"),
        ("terra-5.6", "medium"),
        ("gpt-5.5", "medium"),
        ("gpt-pro-5.5", "high"),
        ("gpt-5.4", "none"),
        ("gpt-pro-5.4", "medium"),
        ("gpt-mini-5.4", "none"),
        ("gpt-nano-5.4", "none"),
        ("gpt-5.2", "none"),
        ("gpt-pro-5.2", "medium"),
        ("gpt-5.1", "none"),
        ("gpt-4.1", "none"),
    ],
)
def test_a_model_starts_at_the_effort_codex_gives_it(model: str, effort: str) -> None:
    for provider in (OpenAI(), OpenAISub()):
        if model in provider:
            assert provider[model].settings.thinking_effort == effort


def test_a_1m_window_takes_input_only_up_to_the_room_its_reply_needs() -> None:
    """Measured on 2026-10-27 (gpt-5.6-luna, gpt-6-luna): 921,013 input tokens were
    served and 921,913 refused as past the window, whatever ``max_output_tokens``.
    """
    for model in (
        "astra-6.0",
        "sol-6.0",
        "luna-5.6",
        "gpt-5.5",
        "gpt-pro-5.5",
        "gpt-5.4",
    ):
        limits = OpenAI()[model].capability.limits
        assert (limits.max_request_tokens, limits.max_context_tokens) == (
            921_000,
            1_050_000,
        ), model


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
