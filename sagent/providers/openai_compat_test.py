"""Tests for sagent.providers.openai_compat (shared base)."""

from __future__ import annotations

from typing import Any, ClassVar, override
from unittest.mock import AsyncMock, MagicMock, patch

import math

import pytest

from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Provider,
    TextMessage,
)
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.openai_compat import (
    OpenAICompat,
    consume_stream,
    parse_response,
)


_PRICING = Pricing(request=1.0, response=2.0, cache_read=0.5)


def _user(text: str) -> Message:
    return TextMessage(text, "text/x-user-message")


def _text(resp: ModelResponse) -> str | None:
    """Extract plain text from a ModelResponse."""
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "text/plain" and isinstance(p, TextMessage):
            return p.content
    return None


def _thinking(resp: ModelResponse) -> str | None:
    """Extract thinking text from a ModelResponse."""
    if not isinstance(resp.content, MultipartMessage):
        return None
    for p in resp.content.content:
        if p.descriptor == "text/x-thinking" and isinstance(p, TextMessage):
            return p.content
    return None


# -- pricing ----------------------------------------------------------


class TestComputeCost:
    def test_basic(self) -> None:
        in_cost, out_cost, total = compute_cost(
            _PRICING,
            input_tokens=1000,
            output_tokens=2000,
        )
        assert math.isclose(in_cost, 0.001)
        assert math.isclose(out_cost, 0.004)
        assert math.isclose(total, 0.005)

    def test_cache_discount(self) -> None:
        in_cost, _, _ = compute_cost(
            _PRICING,
            input_tokens=400,
            output_tokens=0,
            cache_read=600,
        )
        # 400 uncached @ 1.0/Mtok + 600 cached @ 0.5/Mtok = 0.0007.
        assert math.isclose(in_cost, 0.0007)


# -- subclass plumbing ------------------------------------------------


class _Dummy(OpenAICompat):
    DEFAULT_MODEL: ClassVar[str] = "dummy-1"
    ENV_VAR: ClassVar[str] = "DUMMY_KEY"
    BASE_URL: ClassVar[str] = "https://api.example.test/v1"
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        "dummy-1": ModelProfile(
            max_request_tokens=128_000, max_response_tokens=16_384, pricing=_PRICING
        ),
    }


class TestBase:
    def test_from_key_and_endpoint(self) -> None:
        p: Provider = _Dummy.from_key("k")
        m = p.model()
        assert m.model_id == "dummy-1"
        assert m._endpoint == "https://api.example.test/v1/chat/completions"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DUMMY_KEY", "sekret")
        p = _Dummy.from_env()
        assert p.api_key == "sekret"

    def test_from_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DUMMY_KEY", raising=False)
        with pytest.raises(RuntimeError, match="API key not configured"):
            _Dummy.from_env()

    def test_base_url_override(self) -> None:
        p = _Dummy.from_key("k", base_url="http://localhost:8000/v1")
        m = p.model()
        assert m._endpoint == "http://localhost:8000/v1/chat/completions"

    def test_known_model_profile(self) -> None:
        p = _Dummy.from_key("k")
        m = p.model("dummy-1")
        assert m._profile.pricing == _PRICING

    def test_unknown_model_raises(self) -> None:
        p = _Dummy.from_key("k")
        with pytest.raises(ValueError, match="Unknown model 'unknown'"):
            p.model("unknown")


# -- reasoning / thinking surface ------------------------------------


class TestReasoningContent:
    def test_parse_reasoning_content(self) -> None:
        data: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "thinking...",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        resp = parse_response(
            data, pricing=_PRICING, reasoning_field="reasoning_content"
        )
        assert _text(resp) == "answer"
        assert _thinking(resp) == "thinking..."

    def test_parse_reasoning_disabled(self) -> None:
        data: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "thinking...",
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {},
        }
        resp = parse_response(data, pricing=_PRICING, reasoning_field=None)
        assert _thinking(resp) is None

    @pytest.mark.anyio
    async def test_stream_reasoning_content(self) -> None:
        lines = [
            'data: {"id":"1","choices":[{"delta":{"reasoning_content":"hmm"}}]}',
            'data: {"id":"1","choices":[{"delta":{"content":"ok"}}]}',
            'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{}}',
            "data: [DONE]",
        ]

        async def _aiter() -> Any:
            for ln in lines:
                yield ln

        fake_resp = MagicMock()
        fake_resp.aiter_lines = _aiter
        resp = await consume_stream(
            fake_resp,
            on_text=None,
            pricing=_PRICING,
            reasoning_field="reasoning_content",
        )
        assert _text(resp) == "ok"
        assert _thinking(resp) == "hmm"


class TestStopReasonNormalization:
    """End-to-end: provider wire format → canonical agent vocabulary."""

    def test_buffered_stop_to_model_finished(self) -> None:
        data: dict[str, Any] = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }
        resp = parse_response(data, pricing=_PRICING, reasoning_field=None)
        assert resp.stop_reason == "model_finished"

    def test_buffered_length_to_max_tokens(self) -> None:
        """OpenAI's truncation signal must reach the agent as canonical."""
        data: dict[str, Any] = {
            "choices": [{"message": {"content": "trunc"}, "finish_reason": "length"}],
            "usage": {},
        }
        resp = parse_response(data, pricing=_PRICING, reasoning_field=None)
        assert resp.stop_reason == "max_tokens"

    def test_buffered_content_filter_to_model_refusal(self) -> None:
        data: dict[str, Any] = {
            "choices": [
                {"message": {"content": ""}, "finish_reason": "content_filter"},
            ],
            "usage": {},
        }
        resp = parse_response(data, pricing=_PRICING, reasoning_field=None)
        assert resp.stop_reason == "model_refusal"

    @pytest.mark.anyio
    async def test_stream_length_to_max_tokens(self) -> None:
        """SSE finish_reason=length normalizes through ``consume_stream``."""
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"part"}}]}',
            'data: {"id":"1","choices":[{"delta":{},"finish_reason":"length"}],"usage":{}}',
            "data: [DONE]",
        ]

        async def _aiter() -> Any:
            for ln in lines:
                yield ln

        fake_resp = MagicMock()
        fake_resp.aiter_lines = _aiter
        resp = await consume_stream(
            fake_resp, on_text=None, pricing=_PRICING, reasoning_field=None
        )
        assert _text(resp) == "part"
        assert resp.stop_reason == "max_tokens"


# -- transform_body / effort hook -------------------------------------


class _EffortDummy(_Dummy):
    class _Model(_Dummy.MODEL_CLASS):  # ty: ignore[unsupported-base] -- test: subclassing dynamic class attr
        @override
        def _is_effort_model(self, model_id: str) -> bool:
            return model_id == "dummy-1"

    MODEL_CLASS = _Model


class TestEffortHook:
    def test_reasoning_effort_injected(self) -> None:
        m = _EffortDummy.from_key("k").model()
        req = ModelRequest(messages=[_user("hi")], effort="high")
        body = m._build_body(req, stream=False)
        assert body["reasoning_effort"] == "high"

    def test_effort_ignored_when_unsupported(self) -> None:
        m = _Dummy.from_key("k").model()
        req = ModelRequest(messages=[_user("hi")], effort="high")
        body = m._build_body(req, stream=False)
        assert "reasoning_effort" not in body


# -- buffer wired through the base class (no subclass needed) ---------


class TestBufferRoundtrip:
    @pytest.mark.anyio
    async def test_buffer(self) -> None:
        m = _Dummy.from_key("k").model()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "sagent.providers.openai_compat.httpx.AsyncClient",
            return_value=mock_client,
        ):
            resp = await m.buffer(ModelRequest(messages=[_user("hi")]))
        assert _text(resp) == "Hi!"
        # Endpoint built from BASE_URL.
        assert mock_client.post.call_args.args[0].endswith("/chat/completions")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
