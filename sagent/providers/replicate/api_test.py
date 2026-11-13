"""Tests for native text prediction lifecycle and credential-safe polling."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import asyncio

from treekle import ReadError, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError, DisconnectedError
from sagent.model.types import Request, Response
from sagent.providers.catalog import ModelCatalog
from sagent.providers.ollama.api import Ollama
from sagent.providers.openai import wire
from sagent.providers.replicate import api
from sagent.providers.replicate.api import Replicate
from sagent.providers.resources import SharedResources
from sagent.providers.table import ModelTable
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    SystemMessage,
    Thinking,
    ToolCall,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.session.types import SessionRecord


@pytest.mark.asyncio
async def test_prediction_polls_to_completion_without_following_sse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(
        api,
        "asyncio",
        SimpleNamespace(timeout=asyncio.timeout, sleep=sleep),
    )
    provider, seen = _provider(
        {
            "status": "starting",
            "urls": {
                "get": "https://api.replicate.com/v1/predictions/pA-_1",
                "stream": "https://attacker.example/steal",
            },
        },
        {
            "status": "processing",
            "urls": {"get": "https://api.replicate.com/v1/predictions/p1"},
        },
        {
            "status": "succeeded",
            "output": ["hello", " world"],
            "metrics": {"input_token_count": 2, "output_token_count": 2},
        },
    )
    async with provider:
        events = [
            event
            async for event in provider["default"].stream(
                Request(records=(UserMessage(content="hi"),), max_response_tokens=23),
            )
        ]
    assert waits == [0.5]
    assert [request.method for request in seen] == ["POST", "GET", "GET"]
    assert all(request.headers["authorization"] == "Bearer test" for request in seen)
    assert str(seen[0].url) == "https://api.replicate.com/v1/predictions"
    assert all(request.headers["Cancel-After"] == "10m" for request in seen)
    assert all(
        request.headers["content-type"] == "application/json" for request in seen
    )
    assert [request.content for request in seen[1:]] == [b"", b""]
    assert json_loads(seen[0].content) == {
        "version": "meta/meta-llama-3-8b-instruct",
        "input": {"prompt": "User: hi", "system_prompt": "", "max_tokens": 23},
    }
    response = events[-1]
    assert isinstance(response, Response)
    assert response.records == (AssistantMessage(content="hello world"),)
    assert response.tokens.request == response.tokens.response == 2


@pytest.mark.asyncio
async def test_prediction_can_complete_without_polling() -> None:
    provider, seen = _provider(
        {
            "status": "succeeded",
            "output": "hello",
            "metrics": {"input_token_count": 1, "output_token_count": 1},
        },
    )
    async with provider:
        assert (
            len([event async for event in provider["default"].stream(Request())]) == 2
        )
    assert len(seen) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/steal",
        "http://api.replicate.com/v1/predictions/p1",
        "https://api.replicate.com.evil/v1/predictions/p1",
        "https://api.replicate.com@evil.example/v1/predictions/p1",
        "https://api.replicate.com/v1/predictions/p1?forward=evil",
        "https://api.replicate.com/v1/predictions/p1#fragment",
        "https://api.replicate.com/v1/predictions/../secret",
        "https://api.replicate.com:8443/v1/predictions/p1",
    ],
)
@pytest.mark.asyncio
async def test_untrusted_poll_url_is_rejected_before_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="trusted"):
        await Replicate().prediction(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        {"status": "failed", "error": "wrapper failed"},
        {"status": "canceled"},
        {"status": "unknown"},
    ],
)
async def test_unsuccessful_predictions_raise(value: dict[str, object]) -> None:
    provider, _ = _provider(value)
    with pytest.raises(RuntimeError, match="Replicate prediction"):
        async with provider:
            await anext(provider.events({}, model="m", headers={}))


@pytest.mark.asyncio
async def test_missing_text_is_not_a_successful_reply() -> None:
    provider, _ = _provider({"status": "succeeded", "output": None})
    with pytest.raises(DisconnectedError, match="no text"):
        async with provider:
            await anext(provider.events({}, model="m", headers={}))


@pytest.mark.asyncio
async def test_missing_counts_cannot_silently_report_free_generation() -> None:
    provider, _ = _provider({"status": "succeeded", "output": "hello"})
    with pytest.raises(RuntimeError, match="token counts"):
        async with provider:
            await anext(provider.events({}, model="m", headers={}))


@pytest.mark.parametrize("status", [300, 307, 401])
@pytest.mark.asyncio
async def test_http_errors_forward_status_body_and_headers_without_redirects(
    status: int,
) -> None:
    reply = httpx2.Response(
        status,
        content=b"unauthorized",
        headers={"location": "https://attacker.example/steal", "retry-after": "7"},
    )
    provider, seen = _provider(reply, {"status": "succeeded"})
    error = AuthError("refused")
    with (
        patch.object(wire, "to_exception", return_value=error) as classify,
        pytest.raises(AuthError, match="refused") as raised,
    ):
        async with provider:
            await provider.prediction("https://api.replicate.com/v1/predictions/p1")
    assert raised.value is error
    classify.assert_called_once_with(status, body="unauthorized", headers=reply.headers)
    assert [str(request.url) for request in seen] == [
        "https://api.replicate.com/v1/predictions/p1",
    ]


@pytest.mark.parametrize("reply", [{"output": "hello"}, {"status": 7}])
@pytest.mark.asyncio
async def test_prediction_validates_status_at_the_response_boundary(
    reply: dict[str, object],
) -> None:
    provider, _ = _provider(reply)
    with pytest.raises(ReadError):
        async with provider:
            await provider.prediction("https://api.replicate.com/v1/predictions/p1")


def test_caller_must_supply_wrapper_capabilities_and_prices() -> None:
    with pytest.raises(ValueError, match="models=ModelTable"):
        Replicate()["default"]
    assert list(Replicate()) == []


@pytest.mark.parametrize(
    "records",
    [
        (ToolCall(call_id="1", name="Read"),),
        (
            UserMessage(
                content="hi",
                attachments=(Attachment(mime_descriptor="image/png", data=b"png"),),
            ),
        ),
        (
            AgentToAgentMessage(
                sender="lead",
                content="hi",
                attachments=(Attachment(mime_descriptor="image/png", data=b"png"),),
            ),
        ),
    ],
)
def test_text_wrapper_rejects_unsupported_inputs(
    records: Sequence[SessionRecord],
) -> None:
    provider, _ = _provider()
    with pytest.raises(ValueError, match="do not support"):
        provider.encode(Request(records=records), model=provider["default"])


def test_text_wrapper_encodes_system_and_history() -> None:
    provider, _ = _provider()
    model = provider["default"]
    body = provider.encode(
        Request(
            system=("system",),
            records=(
                UserMessage(content="hi"),
                Thinking(content="private deliberation"),
                AssistantMessage(content="hello"),
                # Another agent's line and a harness note are the user's side too.
                AgentToAgentMessage(sender="lead", content="status?"),
                AgentToAgentMessage(content="unsigned"),
                SystemMessage(content="Saved."),
            ),
            max_response_tokens=23,
        ),
        model=model,
    )
    assert body == {
        "version": "meta/meta-llama-3-8b-instruct",
        "input": {
            "prompt": "User: hi\n\nAssistant: hello\n\nUser: [from lead]: status?"
            "\n\nUser: unsigned\n\nUser: <system-reminder>\nSaved.\n</system-reminder>",
            "system_prompt": "system",
            "max_tokens": 23,
        },
    }


@pytest.mark.asyncio
async def test_catalog_accepts_a_caller_priced_wrapper() -> None:
    row = replace(
        Ollama.models()["default"],
        wire_model_id="meta/meta-llama-3-8b-instruct",
    )
    table = ModelTable(rows=(row,), roles={"default": row.model_id})
    async with ModelCatalog(
        factories={"replicate": partial(Replicate, models=table)},
    ) as catalog:
        model = catalog.model("replicate/default", current="replicate")
        assert model.capability.wire_model_id == "meta/meta-llama-3-8b-instruct"
        assert list(catalog["replicate"]) == [row.model_id]


def _provider(
    *replies: dict[str, object] | httpx2.Response,
) -> tuple[Replicate, list[httpx2.Request]]:
    row = replace(
        Ollama.models()["default"],
        wire_model_id="meta/meta-llama-3-8b-instruct",
    )
    seen: list[httpx2.Request] = []
    pending = list(replies)

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        reply = pending.pop(0)
        return (
            reply
            if isinstance(reply, httpx2.Response)
            else httpx2.Response(200, content=json_dumps(reply).encode())
        )

    resources = SharedResources()
    resources._http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(answer),
        follow_redirects=True,
    )
    provider = Replicate(
        api_key="test",
        models=ModelTable(rows=(row,), roles={"default": row.model_id}),
    )
    provider._resources = resources
    return provider, seen


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
