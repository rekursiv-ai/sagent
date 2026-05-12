"""Tests for ``custom_types``: adapter-layer dataclasses + protocols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    HistoryEntry,
    Tool,
    UserMessage,
)
from sagent.custom_types import (
    ContextBudget,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    Pricing,
    TokenCount,
)
from sagent.testing import MockModelCaps


@dataclass(slots=True, kw_only=True)
class _MiniModel(MockModelCaps):
    """Minimal model satisfying the ``Model`` protocol."""

    model_id: str = "mini"
    max_request_tokens: int = 200_000
    responses: list[AssistantMessage] = field(default_factory=list)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(message=AssistantMessage(text=""))

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_text, on_thinking
        return ModelResponse(message=AssistantMessage(text=""))

    async def runtime_stream(
        self,
        history: list[HistoryEntry],
        system: str,
        tools: list[Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_text, on_thinking
        return AssistantMessage(text="")


def test_pricing_defaults_are_zero() -> None:
    p = Pricing()
    assert p.request == 0.0
    assert p.response == 0.0
    assert p.cache_write == 0.0
    assert p.cache_read == 0.0


def test_pricing_kw_only_construction() -> None:
    p = Pricing(request=3.0, response=15.0, cache_write=3.75, cache_read=0.30)
    assert p.request == 3.0
    assert p.cache_write == 3.75


def test_pricing_is_hashable() -> None:
    """Frozen dataclass; usable as a dict key."""
    p = Pricing()
    assert hash(p) == hash(Pricing())


def test_token_count_defaults_zero() -> None:
    t = TokenCount()
    assert t.input_tokens == 0
    assert t.output_tokens == 0
    assert t.cache_creation_tokens == 0
    assert t.cache_read_tokens == 0


def test_token_count_add_sums_all_fields() -> None:
    a = TokenCount(
        input_tokens=10,
        output_tokens=20,
        cache_creation_tokens=5,
        cache_read_tokens=2,
    )
    b = TokenCount(
        input_tokens=1,
        output_tokens=2,
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    c = a + b
    assert c.input_tokens == 11
    assert c.output_tokens == 22
    assert c.cache_creation_tokens == 8
    assert c.cache_read_tokens == 6


def test_token_count_sub_yields_diff() -> None:
    a = TokenCount(input_tokens=10, output_tokens=20)
    b = TokenCount(input_tokens=3, output_tokens=5)
    c = a - b
    assert c.input_tokens == 7
    assert c.output_tokens == 15


def test_token_count_is_hashable() -> None:
    """Frozen dataclass; usable as a dict key."""
    t = TokenCount()
    assert hash(t) == hash(TokenCount())


def test_model_request_defaults() -> None:
    req = ModelRequest(messages=[UserMessage(text="hi")])
    assert len(req.messages) == 1
    assert req.system is None
    assert req.tools is None
    assert req.max_response_tokens is None
    assert req.temperature == 1.0
    assert req.thinking is None
    assert req.effort is None
    assert req.cache_ttl == "5m"
    assert req.stop_sequences == ()


def test_model_request_explicit_fields() -> None:
    req = ModelRequest(
        messages=[],
        system="sys",
        tools=None,
        max_response_tokens=2048,
        temperature=0.5,
        thinking="adaptive",
        effort="high",
        cache_ttl="1h",
        stop_sequences=("STOP",),
    )
    assert req.system == "sys"
    assert req.max_response_tokens == 2048
    assert req.temperature == 0.5
    assert req.thinking == "adaptive"
    assert req.effort == "high"
    assert req.cache_ttl == "1h"
    assert req.stop_sequences == ("STOP",)


def test_model_response_defaults() -> None:
    r = ModelResponse(message=AssistantMessage(text="ok"))
    assert r.message.text == "ok"
    assert isinstance(r.tokens, TokenCount)
    assert r.stop_reason == "model_finished"
    assert r.stop_sequence is None
    assert r.message_id == ""
    assert r.request_id == ""
    assert r.input_cost == 0.0
    assert r.output_cost == 0.0
    assert r.total_cost == 0.0


def test_model_response_with_costs() -> None:
    r = ModelResponse(
        message=AssistantMessage(text="x"),
        tokens=TokenCount(input_tokens=10),
        stop_reason="end_turn",
        message_id="m-1",
        input_cost=0.01,
        output_cost=0.02,
        total_cost=0.03,
    )
    assert r.tokens.input_tokens == 10
    assert r.message_id == "m-1"
    assert r.total_cost == pytest.approx(0.03)


def test_context_budget_post_init_rejects_zero_request_tokens() -> None:
    with pytest.raises(ValueError, match="max_request_tokens"):
        ContextBudget(max_request_tokens=0, max_response_tokens=1024)


def test_context_budget_post_init_rejects_zero_response_tokens() -> None:
    with pytest.raises(ValueError, match="max_response_tokens"):
        ContextBudget(max_request_tokens=1000, max_response_tokens=0)


def test_context_budget_post_init_rejects_buffer_geq_request() -> None:
    with pytest.raises(ValueError, match="buffer_tokens"):
        ContextBudget(
            max_request_tokens=100,
            max_response_tokens=10,
            buffer_tokens=100,
        )


def test_context_budget_post_init_rejects_zero_chars_per_token() -> None:
    with pytest.raises(ValueError, match="chars_per_token"):
        ContextBudget(
            max_request_tokens=1000,
            max_response_tokens=10,
            chars_per_token=0,
        )


def test_context_budget_from_model_uses_proportional_defaults() -> None:
    b = ContextBudget.from_model(_MiniModel())
    assert b.max_request_tokens == 200_000
    assert b.max_response_tokens == 8_192
    assert b.chars_per_token == 4
    assert b.buffer_tokens == max(200_000 // 15, 8_000)
    assert b.reattach_count == 5


def test_context_budget_from_model_floors_for_small_models() -> None:
    """Small models exercise the ``max(..., floor)`` branches.

    The buffer-tokens floor of 8000 only validates when
    ``max_request_tokens`` exceeds it, so we pick a request limit
    just above the floor.
    """
    b = ContextBudget.from_model(_MiniModel(max_request_tokens=10_000))
    assert b.buffer_tokens == 8_000
    assert b.reattach_max_chars == 4 * 2_000
    assert b.reattach_budget == 4 * 10_000


def test_model_spec_defaults_account_none() -> None:
    spec = ModelSpec(provider="Anthropic", auth="api", model_id="claude-3-5")
    assert spec.provider == "Anthropic"
    assert spec.auth == "api"
    assert spec.model_id == "claude-3-5"
    assert spec.account is None


def test_model_spec_with_account_override() -> None:
    spec = ModelSpec(
        provider="Anthropic", auth="sub", model_id="claude-3-5", account="josh@x"
    )
    assert spec.account == "josh@x"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
