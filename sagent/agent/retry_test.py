"""Tests for retry classification and send-with-retry behavior."""

from __future__ import annotations

from collections.abc import Callable

import asyncio

from openai import APIError

import httpx
import pytest

from sagent.agent.retry import send_with_retry
from sagent.custom_types import (
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    TextMessage,
    TokenCount,
)
from sagent.lib.message import response_text


pytestmark = pytest.mark.anyio

TRANSCRIPT_ERROR = (
    "An error occurred while processing your request. You can retry your "
    "request, or contact us through our help center at help.openai.com if "
    "the error persists. Please include the request ID "
    "53a68618-891f-4243-9b55-48e8c9f79dbb in your message."
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _model_response(text: str = "ok") -> ModelResponse:
    return ModelResponse(
        content=MultipartMessage(
            (TextMessage(text, "text/plain"),),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(input_tokens=10, output_tokens=5),
        stop_reason="model_finished",
    )


class _StatuslessOpenAIStreamModel:
    model_id: str = "mock"
    max_request_tokens: int = 1_000_000
    max_response_tokens: int = 8_000
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    supports_streaming: bool = True
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    supports_context_management: bool = False
    max_image_dim: int = 0
    max_image_bytes: int = 0
    pricing: Pricing = Pricing()

    def __init__(self) -> None:
        self.stream_calls = 0

    def estimate_text_token_count(self, text: str) -> int:
        return len(text) // 4

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 0

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        return _model_response()

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_thinking
        self.stream_calls += 1
        if self.stream_calls == 1:
            raise APIError(
                message=TRANSCRIPT_ERROR,
                request=httpx.Request("POST", "https://api.openai.test/responses"),
                body={"message": TRANSCRIPT_ERROR},
            )
        if on_text is not None:
            on_text("ok")
        return _model_response()

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        return "you can retry" in str(error).lower()


async def test_statusless_openai_stream_api_error_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StatuslessOpenAIStreamModel()
    events: list[dict[str, object]] = []

    async def no_sleep(delay: float) -> None:
        del delay

    chunks: list[str] = []

    def on_text(chunk: str) -> None:
        chunks.append(chunk)

    def log_event(event: str, **payload: object) -> None:
        events.append({"event": event, **payload})

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    response = await send_with_retry(
        model,
        ModelRequest(messages=[], system="", tools=None, max_response_tokens=100),
        on_text=on_text,
        max_attempts=2,
        persistent_retry=False,
        log_event=log_event,
    )

    assert response_text(response.content) == "ok"
    assert chunks[-1] == "ok"
    assert "retrying" in chunks[0]
    assert model.stream_calls == 2
    assert [event["event"] for event in events] == ["retry"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
