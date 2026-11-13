"""Tests for native Vertex Gemini requests, signatures, endpoints and ADC tokens."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Self

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.lib.posix.child import Child
from sagent.model.exceptions import AuthError
from sagent.model.types import Request, Response
from sagent.providers.resources import SharedResources
from sagent.providers.table import ModelTable
from sagent.providers.vertex.api import Vertex
from sagent.session.types import ToolCall, UserMessage


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.asyncio
async def test_vertex_uses_native_global_endpoint_and_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project123")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    provider = Vertex(api_key="token")
    assert (
        provider.endpoint("streamGenerateContent", model="gemini-3.8-flash")
        == "https://aiplatform.googleapis.com/v1/projects/project123/locations/global/publishers/google/models/gemini-3.8-flash:streamGenerateContent"
    )
    assert await provider.credentials() == {"authorization": "Bearer token"}
    assert not await provider.reauthenticate(rejected={})
    assert list(provider) == ["gemini-flash-3.8"]
    assert provider["default"].capability.wire_model_id == "gemini-3.8-flash"


@pytest.mark.parametrize(
    ("project", "location", "message"),
    [
        ("", "global", "GOOGLE_CLOUD_PROJECT"),
        ("project", "../bad", "GOOGLE_CLOUD_LOCATION"),
    ],
)
def test_invalid_cloud_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    project: str,
    location: str,
    message: str,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)
    monkeypatch.delenv("VERTEX_BASE_URL", raising=False)
    with pytest.raises(AuthError, match=message):
        Vertex().endpoint("countTokens", model="x")


def test_regional_and_explicit_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("VERTEX_BASE_URL", raising=False)
    assert (
        Vertex()
        .endpoint("countTokens", model="x")
        .startswith("https://us-central1-aiplatform.googleapis.com/")
    )
    with pytest.raises(ValueError, match="Regional Vertex pricing"):
        Vertex()["default"]
    assert (
        Vertex(base_url="https://proxy.example/v1/").endpoint("countTokens", model="x")
        == "https://proxy.example/v1/models/x:countTokens"
    )
    # Gemini's variable names another endpoint, which Vertex never follows.
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://gemini.example/v1beta")
    monkeypatch.setenv("VERTEX_BASE_URL", "https://gateway.example/v1")
    assert (
        Vertex().endpoint("countTokens", model="x")
        == "https://gateway.example/v1/models/x:countTokens"
    )
    # Only the trailing slash is dropped, never a path's last letter.
    assert (
        Vertex(base_url="https://proxy.example/X/").endpoint("countTokens", model="x")
        == "https://proxy.example/X/models/x:countTokens"
    )


@pytest.mark.asyncio
async def test_vertex_stream_preserves_tool_signatures_and_counts_native_body() -> None:
    row = replace(
        Vertex.models()["default"],
        model_id="custom",
        wire_model_id="deployment",
    )
    table = ModelTable(rows=(row,), roles={"default": "custom"})
    seen: list[httpx2.Request] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if "countTokens" in str(request.url):
            return httpx2.Response(200, content=b'{"totalTokens": 7}')
        event = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "Read",
                                    "args": {"path": "a.py"},
                                },
                                "thoughtSignature": "opaque-signature",
                            },
                        ],
                    },
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
        }
        return httpx2.Response(200, content=f"data: {json_dumps(event)}\n\n".encode())

    async with _resources(answer) as resources:
        provider = Vertex(
            resources=resources,
            api_key="token",
            base_url="https://vertex.example/v1",
            models=table,
        )
        assert list(provider) == ["custom"]
        model = provider["default"]
        request = Request(records=(UserMessage(content="hi"),))
        assert await model.count(request) == 7
        events = [event async for event in model.stream(request)]
        response = events[-1]
        assert isinstance(response, Response)
        call = response.records[0]
        assert isinstance(call, ToolCall)
        assert call.extra["thoughtSignature"] == "opaque-signature"
        replay = model.encode(Request(records=(call,)))
        assert "opaque-signature" in json_dumps(replay)
    assert json_loads(seen[0].content, dict[str, object]).get("contents") is not None
    assert "generateContentRequest" not in json_loads(
        seen[0].content,
        dict[str, object],
    )
    assert seen[1].headers["authorization"] == "Bearer token"


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "output"), [(0, b"token\n"), (1, b""), (0, b"")])
async def test_vertex_command_authentication(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    output: bytes,
) -> None:
    monkeypatch.delenv("VERTEX_ACCESS_TOKEN", raising=False)
    called: list[tuple[str, ...]] = []

    async def start(*argv: str, **options: object) -> _TokenChild:
        del options
        called.append(argv)
        return _TokenChild(code=code, output=output)

    monkeypatch.setattr(Child, "start", start)
    provider = Vertex()
    if code == 0 and output:
        assert await provider.credentials() == {"authorization": "Bearer token"}
    else:
        with pytest.raises(AuthError, match="token command failed"):
            await provider.credentials()
    assert called == [("gcloud", "auth", "application-default", "print-access-token")]
    assert await provider.reauthenticate(rejected={})


@pytest.mark.asyncio
async def test_vertex_command_start_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERTEX_ACCESS_TOKEN", raising=False)

    async def failed(*argv: str, **options: object) -> _TokenChild:
        del argv, options
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(Child, "start", failed)
    with pytest.raises(AuthError, match="configure gcloud ADC"):
        await Vertex().credentials()


def _resources(handler: Callable[[httpx2.Request], httpx2.Response]) -> SharedResources:
    resources = SharedResources()
    resources._http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return resources


class _TokenProcess:
    def __init__(self, *, code: int, output: bytes) -> None:
        self.returncode = code
        self.output = output

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.output, b""


class _TokenChild:
    def __init__(self, *, code: int, output: bytes) -> None:
        self.process = _TokenProcess(code=code, output=output)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
