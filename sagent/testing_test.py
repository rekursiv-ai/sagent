"""Tests for ``testing``: ``FakeAgent`` + ``with_fake_agent`` + capabilities."""

from __future__ import annotations

import asyncio

import pytest

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.runtime import (
    AssistantMessage,
    ModelResponseComplete,
    ToolResult,
    UserMessage,
)
from sagent.custom_types import Pricing
from sagent.testing import FakeAgent, MockModelCaps, with_fake_agent
from sagent.tools.core import (
    ToolState,
    current_agent_var,
    tool_state_var,
)


def test_mock_model_caps_static_flags() -> None:
    """Capability flags expose the documented defaults."""
    m = MockModelCaps()
    assert m.max_response_tokens == 8_192
    assert m.supports_streaming is True
    assert m.supports_thinking is False
    assert m.supports_effort is False
    assert m.supports_cache_control is False
    assert m.supports_context_management is False
    assert m.supports_persistent_retry is False
    assert m.supports_account_auth is False
    assert m.max_image_dim == 8_000
    assert m.max_image_bytes == 5 * 1024 * 1024


def test_mock_model_caps_pricing_zero() -> None:
    """Pricing defaults to all-zero."""
    m = MockModelCaps()
    assert isinstance(m.pricing, Pricing)
    assert m.pricing.request == 0.0


def test_mock_model_caps_estimate_text() -> None:
    m = MockModelCaps()
    assert m.estimate_text_token_count("") == 0
    assert m.estimate_text_token_count("12345678") == 2


def test_mock_model_caps_estimate_image_is_constant() -> None:
    m = MockModelCaps()
    assert m.estimate_image_token_count(b"") == 256
    assert m.estimate_image_token_count(b"\x89PNG") == 256


def test_mock_model_caps_is_context_overflow_default_false() -> None:
    m = MockModelCaps()
    assert m.is_context_overflow(RuntimeError("x")) is False


def test_mock_model_caps_is_retryable_provider_error_default_false() -> None:
    m = MockModelCaps()
    assert m.is_retryable_provider_error(RuntimeError("x")) is False


def test_fake_agent_default_state() -> None:
    a = FakeAgent()
    assert isinstance(a.tool_state, ToolState)
    assert a.events == []
    assert dict(a.background) == {}


@pytest.mark.asyncio
async def test_fake_agent_null_model_stream_returns_empty() -> None:
    """The internal ``_NullModel`` wired into the default runtime is callable."""
    a = FakeAgent()

    def _on_text(_t: str) -> None:
        return None

    msg = await a.runtime.model.stream(
        a.runtime.history,
        a.runtime.system,
        list(a.runtime.tools_map.values()),
        _on_text,
        _on_text,
    )
    assert msg.text == ""


def test_fake_agent_runtime_observer_records_events() -> None:
    a = FakeAgent()
    msg = AssistantMessage(text="hi")
    a.runtime.publish(ModelResponseComplete(message=msg))
    a.runtime.publish(UserMessage(text="ping"))
    assert len(a.events) == 2
    completes = a.events_of(ModelResponseComplete)
    assert len(completes) == 1
    assert completes[0].message.text == "hi"


def test_fake_agent_events_of_filters_by_type() -> None:
    a = FakeAgent()
    a.runtime.publish(UserMessage(text="one"))
    a.runtime.publish(ToolResult(call_id="c", content="ok"))
    a.runtime.publish(UserMessage(text="two"))
    users = a.events_of(UserMessage)
    assert [u.text for u in users] == ["one", "two"]


def test_fake_agent_register_and_cancel_background() -> None:
    a = FakeAgent()

    async def _make_entry() -> BackgroundTaskEntry:
        task = asyncio.create_task(asyncio.sleep(0))
        await task
        return BackgroundTaskEntry(
            task=task,
            tool_name="t",
            queue_id="q1",
            started=0.0,
        )

    entry = asyncio.new_event_loop().run_until_complete(_make_entry())
    a.register_background("q1", entry)
    assert "q1" in a.background
    a.cancel_background("q1")
    assert "q1" not in a.background


def test_fake_agent_cancel_background_unknown_id_is_noop() -> None:
    a = FakeAgent()
    a.cancel_background("does-not-exist")
    assert dict(a.background) == {}


def test_with_fake_agent_installs_in_context_vars() -> None:
    with with_fake_agent() as agent:
        assert current_agent_var.get() is agent
        assert tool_state_var.get() is agent.tool_state


def test_with_fake_agent_resets_context_vars() -> None:
    prev_state = ToolState()
    state_token = tool_state_var.set(prev_state)
    try:
        with with_fake_agent():
            pass
        # On exit, the original ToolState is restored.
        assert tool_state_var.get() is prev_state
    finally:
        tool_state_var.reset(state_token)


def test_with_fake_agent_accepts_custom_tool_state() -> None:
    custom = ToolState()
    custom.bash_cwd = "/var/x"
    with with_fake_agent(tool_state=custom) as agent:
        assert agent.tool_state is custom
        assert tool_state_var.get() is custom


def test_with_fake_agent_does_not_leak_on_exception() -> None:
    """ContextVars are reset even when the body raises."""
    state_before = tool_state_var.get(None)
    with pytest.raises(ValueError, match="boom"), with_fake_agent():
        raise ValueError("boom")
    assert tool_state_var.get(None) is state_before


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
