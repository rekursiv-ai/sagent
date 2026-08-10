"""Tests for ``tools.agent_send``: routing messages between live agents."""

from __future__ import annotations

import asyncio

import pytest

from sagent.agent.state import agent_label_var, agent_registry
from sagent.testing import FakeAgent, with_fake_agent
from sagent.tools import agent_send as send_module
from sagent.tools.agent_send import AgentSend
from sagent.types.runtime import (
    AgentSendMessage,
)


def test_metadata_basics() -> None:
    t = AgentSend()
    assert t.name == "AgentSend"
    assert t.tool_id == "application/x-tool-agentsend"


def test_summary_short_and_long() -> None:
    t = AgentSend()
    assert t.summary({"to": "Bob", "content": "hi"}) == "AgentSend → Bob: hi"
    # Uncapped: the renderer wraps and line-caps; the label is not clipped.
    long = "x" * 60
    s = t.summary({"to": "Bob", "content": long})
    assert s == f"AgentSend \u2192 Bob: {long}"


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
    # ``Me`` surfaces as the self-identity anchor (``Your agent label is
    # 'Me'.``) but is excluded from the addressable peers listing.
    assert "Your agent label is 'Me'." in p
    assert "Active agents you can message: Alice, Bob" in p


def test_prompt_self_only_returns_identity() -> None:
    t = AgentSend()
    token = agent_label_var.set("Me")
    agent_registry["Me"] = FakeAgent()
    try:
        p = t.prompt()
    finally:
        agent_registry.pop("Me", None)
        agent_label_var.reset(token)
    assert p == "Your agent label is 'Me'."


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
        isinstance(i, AgentSendMessage) and i.source == "Me" and i.text == "hello"
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
    agent_registry["DelayTarget"] = target
    try:
        send_module._deliver("DelayTarget", "Me", "ping", 7)
    finally:
        agent_registry.pop("DelayTarget", None)
    drained = asyncio.new_event_loop().run_until_complete(target.runtime.inbox.drain())
    # The documentation (assets/default/tools_agentsend.md:6 and the
    # description tooltip) claims the delivered message "automatically
    # notes delay time". The drained payload must mention both the
    # delay window and the original content so the recipient knows
    # how stale a scheduled reminder is.
    #
    # The delivery posts an ``AgentSendMessage`` (preempting) -- the
    # ``call_later`` delay timer alone supplies the "wait before
    # delivery" semantic. A deferred message here would double-defer:
    # the runtime parks deferred messages until ``AgentIdle``, masking
    # the wake-up the delay was meant to provide.
    matches = [
        i for i in drained if isinstance(i, AgentSendMessage) and i.source == "Me"
    ]
    assert matches, drained
    body = matches[0].text
    assert "ping" in body
    assert "7s" in body


def test_deliver_dead_target_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        # ``"Ghost"`` is not in ``agent_registry``; delivery must
        # log a warning rather than silently push into thin air.
        send_module._deliver("Ghost", "Me", "x", 3)
    assert any("Delayed message to dead agent" in rec.message for rec in caplog.records)


def test_deliver_reresolves_target_at_delivery_time() -> None:
    """Re-resolution beats stale-handle delivery.

    Schedule against label X holding agent A; before the timer fires,
    A dies and B is registered under the same label. The delayed
    message must reach B (the *current* holder), not A's defunct
    inbox.
    """
    original = FakeAgent()
    replacement = FakeAgent()
    agent_registry["Rebind"] = original
    # Simulate the schedule→drop→rebind sequence between
    # ``call_later`` and the timer callback.
    agent_registry["Rebind"] = replacement
    try:
        send_module._deliver("Rebind", "Me", "after-rebind", 1)
    finally:
        agent_registry.pop("Rebind", None)
    # ``drain()`` blocks on an empty queue; inspect the underlying
    # queue size directly so the assertion can verify the stale
    # handle stayed empty without hanging.
    assert replacement.runtime.inbox._queue.qsize() == 1, (
        "delivery must reach the current registry holder"
    )
    assert original.runtime.inbox._queue.qsize() == 0, (
        "delivery must not land on the stale handle"
    )


@pytest.mark.asyncio
async def test_run_negative_delay_is_error() -> None:
    """Negative ``delay`` bypasses the ``delay > 0`` gate; reject it.

    Schema declares ``minimum: 0``; runtime must enforce.
    """
    t = AgentSend()
    target = FakeAgent()
    agent_registry["Bob"] = target
    try:
        with with_fake_agent():
            result = await t.run({"to": "Bob", "content": "hi", "delay": -1})
    finally:
        agent_registry.pop("Bob", None)
    assert result.is_error
    assert "delay" in result.content


@pytest.mark.asyncio
async def test_self_send_nudge_fires_when_to_matches_sender() -> None:
    """Undelayed self-send surfaces the soft loop-detection nudge.

    Pin the intended trigger: sender label_var is set AND ``to`` matches.
    The nudge protects against LLM self-loop bugs (see pinger incident);
    this test fixes the contract so the fallthrough-default fix below
    can't accidentally suppress the legitimate case.
    """
    t = AgentSend()
    me = FakeAgent()
    agent_registry["Me"] = me
    token = agent_label_var.set("Me")
    try:
        with with_fake_agent():
            result = await t.run({"to": "Me", "content": "hello self"})
    finally:
        agent_registry.pop("Me", None)
        agent_label_var.reset(token)
    assert not result.is_error
    assert "sending a message to yourself" in result.content


@pytest.mark.asyncio
async def test_no_nudge_when_sender_label_unset() -> None:
    """No nudge when the caller's ``agent_label_var`` isn't set.

    Before the guard, ``agent_label_var.get("unknown")`` produced the
    sentinel string ``"unknown"``. A target agent registered under the
    literal label ``"unknown"`` would falsely match the unset sender
    and trigger the self-send nudge -- a phantom warning for a routine
    cross-agent message. Pin the guard: nudge fires only when label_var
    is genuinely set AND equals ``to``.
    """
    t = AgentSend()
    target = FakeAgent()
    # ``unknown`` is the literal value of the prior nudge's default
    # sender sentinel; verify it no longer self-matches.
    agent_registry["unknown"] = target
    try:
        # No ``agent_label_var.set(...)`` -- the sender label_var is
        # at its default empty value, modelling a tool call from a
        # context that didn't establish identity (test harnesses,
        # FakeAgent, root before serve_forever).
        with with_fake_agent():
            result = await t.run({"to": "unknown", "content": "hi"})
    finally:
        agent_registry.pop("unknown", None)
    assert not result.is_error
    assert "sending a message to yourself" not in result.content, (
        f"unset sender must not phantom-match an 'unknown' target; got {result.content!r}"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
