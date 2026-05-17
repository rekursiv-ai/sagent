"""Tests for ``tools.agent_send``: routing messages between live agents."""

from __future__ import annotations

import asyncio

import pytest

from sagent.testing import FakeAgent, with_fake_agent
from sagent.tools import agent_send as send_module
from sagent.tools.agent_send import AgentSend
from sagent.tools.core import agent_label_var, agent_registry
from sagent.types.history import ToolResult, UserMessage


def test_metadata_basics() -> None:
    t = AgentSend()
    assert t.name == "AgentSend"
    assert t.tool_id == "application/x-tool-agentsend"
    assert t.supports_microcompaction is False
    assert t.summary_result(ToolResult(call_id="", content="")) is None


def test_summary_short_and_long() -> None:
    t = AgentSend()
    assert t.summary({"to": "Bob", "content": "hi"}) == "AgentSend → Bob: hi"
    long = "x" * 60
    s = t.summary({"to": "Bob", "content": long})
    assert s.endswith("...")
    # Preview is truncated to 40 chars (37 + ...).
    assert len(s.split(": ", 1)[1]) == 40


def test_summary_without_to() -> None:
    t = AgentSend()
    assert t.summary({}) == "AgentSend"


def test_prompt_lists_other_agents() -> None:
    t = AgentSend()
    token_self = agent_label_var.set("Me")
    agent_registry["Me"] = FakeAgent()
    agent_registry["Bob"] = FakeAgent()
    agent_registry["Alice"] = FakeAgent()
    try:
        p = t.prompt()
    finally:
        agent_registry.pop("Me", None)
        agent_registry.pop("Bob", None)
        agent_registry.pop("Alice", None)
        agent_label_var.reset(token_self)
    assert "Alice" in p
    assert "Bob" in p
    assert "Me" not in p


def test_prompt_empty_when_alone() -> None:
    t = AgentSend()
    token = agent_label_var.set("Me")
    agent_registry["Me"] = FakeAgent()
    try:
        p = t.prompt()
    finally:
        agent_registry.pop("Me", None)
        agent_label_var.reset(token)
    assert p == ""


@pytest.mark.asyncio
async def test_run_requires_to() -> None:
    t = AgentSend()
    with with_fake_agent():
        result = await t.run({"to": "", "content": "hi"})
    assert result.is_error
    assert "'to' is required" in result.content


@pytest.mark.asyncio
async def test_run_requires_content() -> None:
    t = AgentSend()
    with with_fake_agent():
        result = await t.run({"to": "Bob", "content": ""})
    assert result.is_error
    assert "'content' is required" in result.content


@pytest.mark.asyncio
async def test_run_unknown_agent() -> None:
    t = AgentSend()
    with with_fake_agent():
        result = await t.run({"to": "Ghost", "content": "hi"})
    assert result.is_error
    assert "Unknown agent" in result.content


@pytest.mark.asyncio
async def test_run_delivers_message() -> None:
    t = AgentSend()
    target = FakeAgent()
    agent_registry["Bob"] = target
    token = agent_label_var.set("Me")
    try:
        with with_fake_agent():
            result = await t.run({"to": "Bob", "content": "hello"})
    finally:
        agent_registry.pop("Bob", None)
        agent_label_var.reset(token)
    assert not result.is_error
    assert "Delivered to Bob" in result.content
    # Drain the inbox -- the runtime's GatedDeque is async so use drain().
    items = await target.runtime.inbox.drain()
    assert any(
        isinstance(i, UserMessage) and "hello" in i.text and "[from Me]" in i.text
        for i in items
    )


@pytest.mark.asyncio
async def test_run_delay_schedules_call_later(monkeypatch: pytest.MonkeyPatch) -> None:
    t = AgentSend()
    target = FakeAgent()
    agent_registry["Bob"] = target
    token = agent_label_var.set("Me")

    calls: list[tuple[float, object, tuple[object, ...]]] = []

    class _FakeLoop:
        def call_later(self, delay: float, fn: object, *args: object) -> None:
            calls.append((delay, fn, args))

    fake_loop = _FakeLoop()

    def _get_loop() -> _FakeLoop:
        return fake_loop

    monkeypatch.setattr("asyncio.get_running_loop", _get_loop)
    try:
        with with_fake_agent():
            result = await t.run({"to": "Bob", "content": "later", "delay": 5})
    finally:
        agent_registry.pop("Bob", None)
        agent_label_var.reset(token)
    assert not result.is_error
    assert "Scheduled for Bob in 5s" in result.content
    assert calls
    assert calls[0][0] == 5
    # ``_deliver`` is the scheduled callable.
    assert calls[0][1] is send_module._deliver


def test_deliver_into_live_inbox() -> None:
    target = FakeAgent()
    send_module._deliver(target, "Me", "ping", 7)
    drained = (
        asyncio.get_event_loop_policy()
        .new_event_loop()
        .run_until_complete(target.runtime.inbox.drain())
    )
    assert any(
        isinstance(i, UserMessage)
        and "ping" in i.text
        and "[from Me, 7s ago]" in i.text
        for i in drained
    )


def test_deliver_dead_target_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        send_module._deliver(None, "Me", "x", 3)
    assert any("Delayed message to dead agent" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
