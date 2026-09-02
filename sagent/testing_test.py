"""Tests for ``testing``: ``FakeAgent`` + ``with_fake_agent`` + capabilities."""

from __future__ import annotations

import asyncio

import pytest

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import (
    ToolState,
    current_agent_var,
    tool_state_var,
)
from sagent.testing import FakeAgent, MockModelCaps, with_fake_agent
from sagent.types.cost import PriceCatalogProduct, TokenPrice
from sagent.types.runtime import (
    AssistantMessage,
    Halt,
    Kill,
    ModelResponseComplete,
    Quit,
    ToolResult,
    UserMessage,
)


def test_mock_model_caps_static_flags() -> None:
    """Capability flags expose the documented defaults."""
    m = MockModelCaps()
    assert m.limits.max_response_tokens == 8_192
    assert m.capability.thinking_budget == frozenset({"none"})
    assert m.capability.thinking_effort == frozenset({"none"})
    assert m.capability.cache_ttl_sec == 0.0
    assert m.capability.retries_internally is False
    assert m.capability.account_auth is False
    assert m.limits.max_image_edge_px == 8_000
    assert m.limits.max_image_bytes == 5 * 1024 * 1024


def test_mock_model_caps_pricing_zero() -> None:
    """Every rate defaults to zero, so mocks never fabricate spend."""
    m = MockModelCaps()
    assert m.capability.prices[PriceCatalogProduct()] == TokenPrice()


def test_mock_model_caps_estimate_text() -> None:
    m = MockModelCaps()
    assert m.approx_text_tokens("") == 0
    assert m.approx_text_tokens("12345678") == 2


def test_mock_model_caps_estimate_image_is_constant() -> None:
    m = MockModelCaps()
    assert m.approx_image_tokens(b"") == 256
    assert m.approx_image_tokens(b"\x89PNG") == 256


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


def test_fake_agent_background_merges_runtime_detached() -> None:
    a = FakeAgent()
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        a.runtime.detached["det-1"] = task
        a._tool_registry["det-1"] = ("Read", 123.0)

        merged = a.background

        assert merged["job-1"].kind == "detached"
        assert merged["job-1"].queue_id == "job-1"
        assert merged["job-1"].call_id == "det-1"
        assert merged["job-1"].tool_name == "Read"
        assert merged["job-1"].started == 123.0
        _ = task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


def test_fake_agent_background_uses_defaults_for_unregistered_detached() -> None:
    a = FakeAgent()
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        a.runtime.detached["det-1"] = task

        merged = a.background

        assert merged["job-1"].tool_name == "?"
        assert merged["job-1"].call_id == "det-1"
        assert merged["job-1"].started > 0.0
        _ = task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_fake_agent_null_model_stream_returns_empty() -> None:
    """The internal ``_NullModel`` wired into the default runtime is callable."""
    a = FakeAgent()

    msg = await a.runtime.model.stream(
        a.runtime.context().messages,
        lambda _ev: None,
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


@pytest.mark.asyncio
async def test_fake_agent_shutdown_pushes_quit() -> None:
    a = FakeAgent()

    a.shutdown()
    items = await asyncio.wait_for(a.runtime.inbox.drain(), timeout=0.1)

    assert isinstance(items[-1], Quit)
    assert a.events == []


@pytest.mark.asyncio
async def test_fake_agent_force_shutdown_cancels_visible_background() -> None:
    a = FakeAgent()
    task = asyncio.create_task(asyncio.sleep(60))
    try:
        a.register_background(
            "q1",
            BackgroundTaskEntry(
                task=task,
                tool_name="Tool",
                queue_id="q1",
                started=0.0,
            ),
        )

        a.shutdown(force=True)
        items = await asyncio.wait_for(a.runtime.inbox.drain(), timeout=0.1)
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled()
        assert [type(item) for item in items] == [Kill, Quit]
        assert a.events == []
    finally:
        _ = task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_fake_agent_force_shutdown_keeps_hidden_background_running() -> None:
    a = FakeAgent()
    task = asyncio.create_task(asyncio.sleep(60))
    try:
        a.register_background(
            "q1",
            BackgroundTaskEntry(
                task=task,
                tool_name="Tool",
                queue_id="q1",
                started=0.0,
                hidden=True,
            ),
        )

        a.shutdown(force=True)

        assert not task.cancelled()
    finally:
        _ = task.cancel()
        await asyncio.gather(task, return_exceptions=True)


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


@pytest.mark.asyncio
async def test_null_model_satisfies_runtime_model_protocol() -> None:
    """The default fake runtime's model accepts the runtime ``stream`` shape.

    Guards against regressions that swap the runtime ``Model`` Protocol
    (lean ``stream(history, publish) -> AssistantMessage``) for the rich
    provider ``types.model.Model`` surface and leave ``_NullModel``
    stranded. The runtime calls ``model.stream`` with exactly these two
    positional args; this test invokes the same shape end-to-end.
    """
    a = FakeAgent()
    ctx = a.runtime.context()

    msg = await a.runtime.model.stream(
        ctx.messages,
        lambda _ev: None,
    )
    assert isinstance(msg, AssistantMessage)
    assert msg.text == ""


@pytest.mark.asyncio
async def test_fake_agent_halt_pushes_to_inbox_not_observers() -> None:
    """``halt`` queues a ``Halt`` on the inbox without invoking observers.

    Real ``Agent.halt`` (``agent/agent.py:883-885``) pushes to
    ``runtime.inbox`` so the runtime dispatch loop drives the halt
    state machine. ``FakeAgent`` mirrors that contract: observers stay
    reserved for events the runtime itself publishes so tests can
    distinguish runtime-sourced halts from stub calls.
    """
    a = FakeAgent()

    a.halt()
    items = await asyncio.wait_for(a.runtime.inbox.drain(), timeout=0.1)

    assert [type(item) for item in items] == [Halt]
    assert a.events == []


def test_with_fake_agent_accepts_prebuilt_agent() -> None:
    """``with_fake_agent(agent=...)`` installs the caller's fake verbatim."""
    prebuilt = FakeAgent()
    prebuilt.tool_state.bash_cwd = "/srv/x"
    with with_fake_agent(agent=prebuilt) as active:
        assert active is prebuilt
        assert current_agent_var.get() is prebuilt
        assert tool_state_var.get() is prebuilt.tool_state


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
