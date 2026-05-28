"""Tests for ``bin.slack``: deterministic routing, commands, rendering.

All tests stub the Slack SDK -- no network calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import argparse
import asyncio
import contextlib
import json
import os
import time

import pytest

from sagent.agent import Agent as RealAgent
from sagent.bin.slack import (
    SlackAdapter,
    _AgentSlack,
    _extract_agent_mention,
    _extract_channel_from_text,
    _extract_event,
    _flush_log,
    _latest_session_dir,
    _list_personas,
    _load_manifest,
    _make_log_forwarder,
    _new_session_dir,
    _render_event,
    _resolve_tokens,
    _save_manifest,
    _strip_mention,
    load_persona,
    log_tap,
    parse_slack_args,
)
from sagent.lib.json import MutableJSON
from sagent.testing import FakeAgent
from sagent.tools.core import agent_registry
from sagent.tools.slack import Slack
from sagent.types.model import ModelSpec
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolLabel,
    ToolResult,
    UserMessage,
)


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Wipe the global agent_registry around every test."""
    yield
    agent_registry.clear()


def _register(*names: str) -> dict[str, FakeAgent]:
    """Insert one ``FakeAgent`` per name into the global registry."""
    agents: dict[str, FakeAgent] = {}
    for n in names:
        agent = FakeAgent()
        agent_registry[n] = agent
        agents[n] = agent
    return agents


def _drain_inbox(agent: FakeAgent) -> list[RuntimeEvent]:
    """Synchronously drain the agent runtime inbox for assertions."""
    out: list[RuntimeEvent] = []
    queue = agent.runtime.inbox._queue
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _inbox_size(agent: FakeAgent) -> int:
    return agent.runtime.inbox._queue.qsize()


class _SpySlack:
    """Records ``send()`` calls; substituted for the real Slack tool."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []
        # Pluggable mock for ``create_channel`` used in ``_resolve_router_log_channel``.
        self.create_channel: AsyncMock = AsyncMock(return_value="id=C_DEFAULT")

    async def send(
        self,
        channel: str,
        text: str,
        thread_ts: str = "",
    ) -> str:
        self.sent.append((channel, text, thread_ts))
        return f"Sent. ts=1.0 channel={channel}"

    @property
    def last_text(self) -> str:
        return self.sent[-1][1]


def _make_adapter(
    *,
    persona_dir: Path | None = None,
    tmp_path: Path | None = None,
) -> tuple[SlackAdapter, _SpySlack]:
    """Build a ``SlackAdapter`` with all external dependencies stubbed out."""
    adapter: SlackAdapter = object.__new__(SlackAdapter)
    adapter._bot_token = "xoxb-fake"  # noqa: S105 -- test credential
    adapter._self_user_id = "UBOT"
    adapter._persona_dir = persona_dir or Path("/tmp/no-personas")  # noqa: S108 -- test fallback path
    adapter._session_dir = tmp_path or Path("/tmp/no-session")  # noqa: S108 -- test fallback path
    adapter._log_prefix = ""
    adapter._log_channels = {}
    adapter._log_channel_owners = {}
    adapter._thread_owners = {}
    adapter._active_agents = {}
    adapter._tasks = []
    adapter._bg_tasks = set()
    adapter._user_names = {}
    adapter._router_log_channel = ""
    adapter._sent_messages = {}
    spy = _SpySlack()
    adapter._slack = spy  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAttributeAccessIssue]  -- test spy duck-types Slack.send only
    return adapter, spy


def _msg_event(
    text: str,
    *,
    channel: str = "C123",
    user: str = "UHUMAN",
    ts: str = "1.0",
    thread_ts: str = "",
    bot_id: str = "",
    username: str = "",
    subtype: str | None = None,
) -> MutableJSON:
    ev: MutableJSON = {
        "type": "message",
        "text": text,
        "channel": channel,
        "user": user,
        "ts": ts,
    }
    if thread_ts:
        ev["thread_ts"] = thread_ts
    if bot_id:
        ev["bot_id"] = bot_id
    if username:
        ev["username"] = username
    if subtype is not None:
        ev["subtype"] = subtype
    return ev


def _reaction_event(
    reaction: str,
    *,
    channel: str = "C123",
    msg_ts: str = "1.0",
    user: str = "UHUMAN",
    item_type: str = "message",
    item_user: str = "",
) -> MutableJSON:
    ev: MutableJSON = {
        "type": "reaction_added",
        "user": user,
        "reaction": reaction,
        "item": {
            "type": item_type,
            "channel": channel,
            "ts": msg_ts,
        },
    }
    if item_user:
        ev["item_user"] = item_user
    return ev


class TestExtractEvent:
    def test_valid_envelope(self) -> None:
        payload: MutableJSON = {"event": {"type": "app_mention", "text": "hi"}}
        ev = _extract_event(payload)
        assert ev is not None
        assert ev["type"] == "app_mention"

    def test_missing_event_key(self) -> None:
        assert _extract_event({"foo": "bar"}) is None

    def test_non_dict_payload(self) -> None:
        assert _extract_event(None) is None

    def test_non_dict_event(self) -> None:
        assert _extract_event({"event": "string"}) is None


class TestStripMention:
    def test_removes_bot_mention(self) -> None:
        assert _strip_mention("<@U123> hello", "U123") == "hello"

    def test_no_user_id(self) -> None:
        assert _strip_mention("  hello  ", "") == "hello"

    def test_multiple_mentions(self) -> None:
        assert _strip_mention("<@U1> <@U1> hi", "U1") == "hi"


class TestExtractAgentMention:
    def test_matches_registered_agent(self) -> None:
        _ = _register("Sara")
        assert _extract_agent_mention("Sara check the deploy") == "Sara"

    def test_case_insensitive(self) -> None:
        _ = _register("Sara")
        assert _extract_agent_mention("sara check the deploy") == "Sara"

    def test_no_match(self) -> None:
        assert _extract_agent_mention("hello world") is None


class TestLoadPersona:
    def test_loads_named(self, tmp_path: Path) -> None:
        _ = (tmp_path / "sre.md").write_text("You are an SRE.")
        assert load_persona(tmp_path, "sre") == "You are an SRE."

    def test_falls_back_to_default(self, tmp_path: Path) -> None:
        _ = (tmp_path / "default.md").write_text("Default persona.")
        assert load_persona(tmp_path, "nonexistent") == "Default persona."

    def test_generates_when_missing(self, tmp_path: Path) -> None:
        result = load_persona(tmp_path, "ghost")
        assert "ghost" in result


class TestListPersonas:
    def test_lists_md_files(self, tmp_path: Path) -> None:
        _ = (tmp_path / "sre.md").write_text("")
        _ = (tmp_path / "pm.md").write_text("")
        _ = (tmp_path / "notes.txt").write_text("")
        assert _list_personas(tmp_path) == ["pm", "sre"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert _list_personas(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert _list_personas(tmp_path / "nope") == []


class TestAgentSlack:
    def test_prompt_lists_peers(self) -> None:
        _ = _register("Sara", "Bob")
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        p = tool.prompt()
        assert "Bob" in p
        assert "Sara" not in p

    def test_prompt_empty_when_alone(self) -> None:
        _ = _register("Sara")
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        assert tool.prompt() == ""

    def test_prompt_empty_no_agents(self) -> None:
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        assert tool.prompt() == ""


class TestRouteHumanMessages:
    @pytest.mark.anyio
    async def test_ignores_own_messages(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("hi", user="UBOT"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_ignores_non_message_types(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event("hi")
        ev["type"] = "reaction_added"
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_ignores_message_subtypes(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("hi", subtype="channel_join"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_route_to_log_channel_owner(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._log_channel_owners["C_LOG"] = "Sara"
        await adapter._route(_msg_event("check this", channel="C_LOG"))
        items = _drain_inbox(agents["Sara"])
        assert len(items) == 1
        assert isinstance(items[0], UserMessage)
        assert "check this" in items[0].text

    @pytest.mark.anyio
    async def test_route_by_agent_name_mention(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        await adapter._route(_msg_event("Sara do the thing"))
        assert _inbox_size(agents["Sara"]) > 0
        assert _inbox_size(agents["Bob"]) == 0

    @pytest.mark.anyio
    async def test_agent_mention_sets_thread_owner(self) -> None:
        adapter, _ = _make_adapter()
        _ = _register("Sara")
        await adapter._route(_msg_event("Sara do X", ts="1.0"))
        assert adapter._thread_owners[("C123", "1.0")] == "Sara"

    @pytest.mark.anyio
    async def test_thread_continuation(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        adapter._thread_owners[("C123", "1.0")] = "Sara"
        await adapter._route(_msg_event("follow up", thread_ts="1.0", ts="2.0"))
        assert _inbox_size(agents["Sara"]) > 0
        assert _inbox_size(agents["Bob"]) == 0

    @pytest.mark.anyio
    async def test_single_agent_default(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("do something"))
        assert _inbox_size(agents["Sara"]) > 0

    @pytest.mark.anyio
    async def test_single_agent_default_sets_thread_owner(self) -> None:
        adapter, _ = _make_adapter()
        _ = _register("Sara")
        await adapter._route(_msg_event("do something", ts="5.0"))
        assert adapter._thread_owners[("C123", "5.0")] == "Sara"

    @pytest.mark.anyio
    async def test_ambiguous_agents_replies(self) -> None:
        adapter, spy = _make_adapter()
        agents = _register("Sara", "Bob")
        await adapter._route(_msg_event("do something"))
        assert _inbox_size(agents["Sara"]) == 0
        assert _inbox_size(agents["Bob"]) == 0
        assert len(spy.sent) == 1
        assert "Bob" in spy.last_text
        assert "Sara" in spy.last_text

    @pytest.mark.anyio
    async def test_no_agents_replies_with_guidance(self, tmp_path: Path) -> None:
        _ = (tmp_path / "sre.md").write_text("")
        adapter, spy = _make_adapter(persona_dir=tmp_path)
        await adapter._route(_msg_event("hello"))
        assert len(spy.sent) == 1
        assert "create" in spy.last_text.lower()
        assert "sre" in spy.last_text


class TestRouteAgentMessages:
    @pytest.mark.anyio
    async def test_agent_message_routes_to_mentioned_agent(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        ev = _msg_event(
            "Bob can you check this?",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Bob"]) > 0
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_agent_message_no_self_routing(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event(
            "Sara thinking out loud",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_agent_thread_continuation_skips_self(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        adapter._thread_owners[("C123", "1.0")] = "Sara"
        ev = _msg_event(
            "here's my reply",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            thread_ts="1.0",
            ts="2.0",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_agent_thread_continuation_routes_to_owner(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        adapter._thread_owners[("C123", "1.0")] = "Sara"
        ev = _msg_event(
            "here's my answer",
            bot_id="B1",
            username="Bob",
            subtype="bot_message",
            thread_ts="1.0",
            ts="2.0",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) > 0

    @pytest.mark.anyio
    async def test_agent_log_channel_skips_self(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._log_channel_owners["C_LOG"] = "Sara"
        ev = _msg_event(
            "my own log",
            channel="C_LOG",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_foreign_bot_ignored(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event(
            "Sara deploy now",
            bot_id="B_OTHER",
            username="deploy-bot",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_agent_unroutable_message_dropped(self) -> None:
        adapter, spy = _make_adapter()
        agents = _register("Sara", "Bob")
        ev = _msg_event(
            "just thinking",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0
        assert _inbox_size(agents["Bob"]) == 0
        assert len(spy.sent) == 0

    @pytest.mark.anyio
    async def test_agent_message_no_commands(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara")
        ev = _msg_event(
            "help",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert len(spy.sent) == 0


class TestCommands:
    @pytest.mark.anyio
    async def test_help(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("help", "C1", "1.0")
        assert result is True
        assert "create" in spy.last_text

    @pytest.mark.anyio
    async def test_list_no_agents(self, tmp_path: Path) -> None:
        _ = (tmp_path / "sre.md").write_text("")
        adapter, spy = _make_adapter(persona_dir=tmp_path)
        result = await adapter._try_command("list", "C1", "1.0")
        assert result is True
        assert "No active" in spy.last_text
        assert "sre" in spy.last_text

    @pytest.mark.anyio
    async def test_list_with_agents(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara", "Bob")
        result = await adapter._try_command("list", "C1", "1.0")
        assert result is True
        assert "Sara" in spy.last_text
        assert "Bob" in spy.last_text

    @pytest.mark.anyio
    async def test_create_duplicate(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara")
        result = await adapter._try_command("create Sara", "C1", "1.0")
        assert result is True
        assert "already exists" in spy.last_text

    @pytest.mark.anyio
    async def test_stop_nonexistent(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("stop Ghost", "C1", "1.0")
        assert result is True
        assert "No agent" in spy.last_text

    @pytest.mark.anyio
    async def test_stop_existing(self, tmp_path: Path) -> None:
        adapter, spy = _make_adapter(tmp_path=tmp_path)
        _ = _register("Sara")
        result = await adapter._try_command("stop Sara", "C1", "1.0")
        assert result is True
        assert "stopped" in spy.last_text

    @pytest.mark.anyio
    async def test_unknown_not_a_command(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("deploy everything", "C1", "1.0")
        assert result is False
        assert len(spy.sent) == 0

    @pytest.mark.anyio
    async def test_create_with_custom_label(self, tmp_path: Path) -> None:
        _ = (tmp_path / "sre.md").write_text("You are an SRE.")
        adapter, _ = _make_adapter(persona_dir=tmp_path)
        mock = AsyncMock()
        adapter.spawn_agent = mock  # ty: ignore[invalid-assignment] -- test mock
        result = await adapter._try_command("create sre as ops", "C1", "1.0")
        assert result is True
        mock.assert_called_once_with("ops", "You are an SRE.")

    @pytest.mark.anyio
    async def test_create_loads_persona(self, tmp_path: Path) -> None:
        _ = (tmp_path / "pm.md").write_text("You are a PM.")
        adapter, _ = _make_adapter(persona_dir=tmp_path)
        mock = AsyncMock()
        adapter.spawn_agent = mock  # ty: ignore[invalid-assignment] -- test mock
        result = await adapter._try_command("create pm", "C1", "1.0")
        assert result is True
        mock.assert_called_once_with("pm", "You are a PM.")


class TestRouteCommandIntegration:
    @pytest.mark.anyio
    async def test_help_via_route(self) -> None:
        adapter, spy = _make_adapter()
        await adapter._route(_msg_event("help"))
        assert len(spy.sent) == 1
        assert "Commands:" in spy.last_text

    @pytest.mark.anyio
    async def test_list_via_route(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara")
        await adapter._route(_msg_event("list"))
        assert len(spy.sent) == 1
        assert "Sara" in spy.last_text

    @pytest.mark.anyio
    async def test_command_skipped_when_agent_mentioned(self) -> None:
        """'help' routes to agent named 'help' if one exists (step 2 > step 4)."""
        agents = _register("help")
        adapter, spy = _make_adapter()
        await adapter._route(_msg_event("help me with this"))
        assert _inbox_size(agents["help"]) > 0
        assert len(spy.sent) == 0


class TestRouteReactions:
    @pytest.mark.anyio
    async def test_reaction_routed_to_agent(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hello world", "0.5")
        await adapter._route(_reaction_event("heart"))
        items = _drain_inbox(agents["Sara"])
        assert len(items) == 1
        assert isinstance(items[0], UserMessage)
        assert ":heart:" in items[0].text
        assert "hello world" in items[0].text

    @pytest.mark.anyio
    async def test_reaction_includes_user_name(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._user_names["UHUMAN"] = "Josh"
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "0.5")
        await adapter._route(_reaction_event("thumbsup"))
        items = _drain_inbox(agents["Sara"])
        assert isinstance(items[0], UserMessage)
        assert "Josh" in items[0].text
        assert ":thumbsup:" in items[0].text

    @pytest.mark.anyio
    async def test_reaction_uses_cached_thread_ts(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "2.0")] = ("Sara", "reply text", "1.0")
        await adapter._route(_reaction_event("heart", msg_ts="2.0"))
        items = _drain_inbox(agents["Sara"])
        assert isinstance(items[0], UserMessage)
        assert "thread_ts=1.0" in items[0].text

    @pytest.mark.anyio
    async def test_reaction_ignored_when_not_cached(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_reaction_event("heart"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_reaction_ignored_for_non_message_item(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart", item_type="file"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_reaction_ignored_when_agent_stopped(self) -> None:
        adapter, _ = _make_adapter()
        # Sara cached but not in registry.
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart"))
        # No crash, no routing.

    @pytest.mark.anyio
    async def test_self_reaction_ignored(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart", user="UBOT"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_reaction_to_non_bot_message_ignored(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart", item_user="UOTHER"))
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_reaction_missing_fields_ignored(self) -> None:
        adapter, _ = _make_adapter()
        _ = _register("Sara")
        # No item.
        await adapter._route(
            {"type": "reaction_added", "user": "UHUMAN", "reaction": "heart"}
        )
        # No reaction.
        await adapter._route(
            {
                "type": "reaction_added",
                "user": "UHUMAN",
                "item": {"type": "message", "channel": "C123", "ts": "1.0"},
            }
        )
        # No user.
        await adapter._route(
            {
                "type": "reaction_added",
                "reaction": "heart",
                "item": {"type": "message", "channel": "C123", "ts": "1.0"},
            }
        )

    @pytest.mark.anyio
    async def test_agent_message_cached_on_route(self) -> None:
        adapter, _ = _make_adapter()
        _ = _register("Sara", "Bob")
        ev = _msg_event(
            "here is my response",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            ts="3.0",
            thread_ts="1.0",
            user="",
        )
        await adapter._route(ev)
        cached = adapter._sent_messages.get(("C123", "3.0"))
        assert cached is not None
        agent_name, text, thread_ts = cached
        assert agent_name == "Sara"
        assert "response" in text
        assert thread_ts == "1.0"

    @pytest.mark.anyio
    async def test_reaction_empty_msg_text(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "", "1.0")
        await adapter._route(_reaction_event("heart"))
        items = _drain_inbox(agents["Sara"])
        assert len(items) == 1
        assert isinstance(items[0], UserMessage)
        assert ":heart:" in items[0].text
        assert '""' not in items[0].text


class TestRenderEvent:
    def test_user_message(self) -> None:
        assert _render_event(UserMessage(text="hello")) == "━━ input ━━\nhello"

    def test_empty_user_message(self) -> None:
        assert _render_event(UserMessage(text="   ")) is None

    def test_thinking(self) -> None:
        assert _render_event(ModelResponseThinking(text="pondering...")) == (
            "💭 pondering..."
        )

    def test_thinking_empty(self) -> None:
        assert _render_event(ModelResponseThinking(text="")) is None

    def test_thinking_truncated(self) -> None:
        out = _render_event(ModelResponseThinking(text="x" * 3000))
        assert out is not None
        assert out.endswith("…")
        assert len(out) < 3010

    def test_tool_label(self) -> None:
        assert _render_event(ToolLabel(call_id="c1", text="Bash ls")) == "  Bash ls"

    def test_tool_result_with_summary(self) -> None:
        ev = ToolResult(call_id="c1", content="raw", summary="2 files")
        out = _render_event(ev)
        assert out == "  → 2 files"

    def test_tool_result_error(self) -> None:
        out = _render_event(ToolResult(call_id="c1", content="failed", is_error=True))
        assert out == "  ✗ failed"

    def test_tool_result_diff(self) -> None:
        ev = ToolResult(
            call_id="c1",
            content="ok",
            diff="@@ -1,1 +1,1 @@\n-old\n+new",
        )
        out = _render_event(ev)
        assert out is not None
        assert "```diff" in out
        assert "-old" in out
        assert "+new" in out

    def test_tool_result_empty_returns_none(self) -> None:
        assert _render_event(ToolResult(call_id="c1", content="")) is None

    def test_model_error(self) -> None:
        out = _render_event(ModelResponseError(exception=RuntimeError("oops")))
        assert out == "✗ RuntimeError: oops"

    def test_partial_chunks_skipped(self) -> None:
        assert _render_event(ModelResponsePartial(text="chunk")) is None

    def test_assistant_message(self) -> None:
        # ``AssistantMessage`` isn't in the ``RuntimeEvent`` union but the
        # renderer accepts it for the cases where the agent runtime
        # publishes assistant messages as part of an observer fanout.
        ev: RuntimeEvent = AssistantMessage(text="hello")  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAssignmentType]  -- see comment
        assert _render_event(ev) == "hello"

    def test_assistant_message_empty(self) -> None:
        ev: RuntimeEvent = AssistantMessage(text="   ")  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAssignmentType]  -- see comment above
        assert _render_event(ev) is None

    def test_tool_result_with_hint(self) -> None:
        ev = ToolResult(call_id="c1", content="ok", hint="be careful")
        assert _render_event(ev) == "  hint: be careful"

    def test_tool_result_truncates_long_content(self) -> None:
        ev = ToolResult(call_id="c1", content="x" * 3000)
        out = _render_event(ev)
        assert out is not None
        assert out.endswith("…")
        assert len(out) < 3020


class TestSessionDirs:
    def test_new_session_dir_creates_timestamped(self, tmp_path: Path) -> None:
        d = _new_session_dir(tmp_path)
        assert d.is_dir()
        assert d.parent == tmp_path

    def test_latest_session_dir_none_for_missing(self, tmp_path: Path) -> None:
        assert _latest_session_dir(tmp_path / "nope") is None

    def test_latest_session_dir_picks_most_recent(self, tmp_path: Path) -> None:
        older = tmp_path / "20200101_000000"
        older.mkdir()
        _ = (older / "manifest.json").write_text("{}")
        newer = tmp_path / "20300101_000000"
        newer.mkdir()
        _ = (newer / "manifest.json").write_text("{}")
        # Newer mtime wins regardless of name order.
        os.utime(older, (1.0, 1.0))
        os.utime(newer, (time.time(), time.time()))
        assert _latest_session_dir(tmp_path) == newer

    def test_latest_session_dir_skips_no_manifest(self, tmp_path: Path) -> None:
        empty = tmp_path / "session"
        empty.mkdir()
        assert _latest_session_dir(tmp_path) is None


class TestManifest:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        agents = {"Sara": {"persona": "sre", "system": "you are sre"}}
        _save_manifest(tmp_path, agents)
        assert _load_manifest(tmp_path) == agents

    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        assert _load_manifest(tmp_path) == {}

    def test_load_migrates_legacy_string_format(self, tmp_path: Path) -> None:
        _ = (tmp_path / "manifest.json").write_text(json.dumps({"Sara": "sre"}))
        result = _load_manifest(tmp_path)
        assert result == {"Sara": {"persona": "sre", "system": ""}}


class TestExtractChannelFromText:
    def test_extracts_channel_id(self) -> None:
        # Function looks for a bare ``channel=...`` token (whitespace
        # separated). The trailing ``]`` is stripped.
        assert _extract_channel_from_text("channel=C123] hello") == "C123"

    def test_no_channel_token(self) -> None:
        assert _extract_channel_from_text("no marker here") == ""


class TestExtractAgentMentionWhitespace:
    def test_empty_after_strip(self) -> None:
        # First word stripped of punctuation is empty -> early break.
        _ = _register("Sara")
        assert _extract_agent_mention(",,,") is None


class TestReactionRouting:
    @pytest.mark.anyio
    async def test_reaction_missing_channel_in_item(self) -> None:
        adapter, _ = _make_adapter()
        _ = _register("Sara")
        ev: MutableJSON = {
            "type": "reaction_added",
            "user": "UHUMAN",
            "reaction": "heart",
            "item": {"type": "message", "ts": "1.0"},
        }
        # No channel -> early return; no crash.
        await adapter._route(ev)


class TestResolveUser:
    @pytest.mark.anyio
    async def test_returns_cached(self) -> None:
        adapter, _ = _make_adapter()
        adapter._user_names["U1"] = "Alice"
        assert await adapter._resolve_user("U1") == "Alice"

    @pytest.mark.anyio
    async def test_resolves_via_web(self) -> None:
        adapter, _ = _make_adapter()
        mock_web = MagicMock()
        mock_web.users_info = AsyncMock(
            return_value={"user": {"profile": {"display_name": "Josh"}}},
        )
        adapter._web = mock_web
        assert await adapter._resolve_user("U1") == "Josh"
        # Cached on subsequent call.
        assert await adapter._resolve_user("U1") == "Josh"
        assert mock_web.users_info.await_count == 1

    @pytest.mark.anyio
    async def test_falls_back_to_real_name(self) -> None:
        adapter, _ = _make_adapter()
        mock_web = MagicMock()
        mock_web.users_info = AsyncMock(
            return_value={"user": {"profile": {}, "real_name": "Real Josh"}},
        )
        adapter._web = mock_web
        assert await adapter._resolve_user("U1") == "Real Josh"

    @pytest.mark.anyio
    async def test_falls_back_to_user_id_on_error(self) -> None:
        adapter, _ = _make_adapter()
        mock_web = MagicMock()
        mock_web.users_info = AsyncMock(side_effect=OSError("boom"))
        adapter._web = mock_web
        assert await adapter._resolve_user("U1") == "U1"


class TestLogRoute:
    @pytest.mark.anyio
    async def test_logs_without_channel(self) -> None:
        adapter, spy = _make_adapter()
        adapter._log_route("decision text")
        assert spy.sent == []

    @pytest.mark.anyio
    async def test_forwards_to_router_log_channel(self) -> None:
        adapter, spy = _make_adapter()
        adapter._router_log_channel = "C_ROUTER"
        adapter._log_route("decision text")
        # Drain background task.
        await asyncio.gather(*adapter._bg_tasks, return_exceptions=True)
        assert any(s[0] == "C_ROUTER" and "decision" in s[1] for s in spy.sent)


class TestResolveRouterLogChannel:
    @pytest.mark.anyio
    async def test_passthrough_if_channel_id(self) -> None:
        adapter, _ = _make_adapter()
        adapter._router_log_channel = "CABC123"
        await adapter._resolve_router_log_channel()
        assert adapter._router_log_channel == "CABC123"

    @pytest.mark.anyio
    async def test_finds_existing_channel(self) -> None:
        adapter, _ = _make_adapter()
        adapter._router_log_channel = "router-log"
        with patch.object(
            SlackAdapter, "_find_channel", new=AsyncMock(return_value="C_FOUND")
        ):
            await adapter._resolve_router_log_channel()
        assert adapter._router_log_channel == "C_FOUND"

    @pytest.mark.anyio
    async def test_creates_when_missing(self) -> None:
        adapter, spy = _make_adapter()
        adapter._router_log_channel = "router-log"
        spy.create_channel = AsyncMock(return_value="id=C_NEW")
        with patch.object(
            SlackAdapter, "_find_channel", new=AsyncMock(return_value=None)
        ):
            await adapter._resolve_router_log_channel()
        assert adapter._router_log_channel == "C_NEW"

    @pytest.mark.anyio
    async def test_create_returns_tool_result_clears_channel(self) -> None:
        adapter, spy = _make_adapter()
        adapter._router_log_channel = "router-log"
        spy.create_channel = AsyncMock(
            return_value=ToolResult(call_id="", content="forbidden", is_error=True),
        )
        with patch.object(
            SlackAdapter, "_find_channel", new=AsyncMock(return_value=None)
        ):
            await adapter._resolve_router_log_channel()
        assert adapter._router_log_channel == ""

    @pytest.mark.anyio
    async def test_create_unexpected_result_clears_channel(self) -> None:
        adapter, spy = _make_adapter()
        adapter._router_log_channel = "router-log"
        spy.create_channel = AsyncMock(return_value="weird")
        with patch.object(
            SlackAdapter, "_find_channel", new=AsyncMock(return_value=None)
        ):
            await adapter._resolve_router_log_channel()
        assert adapter._router_log_channel == ""


class TestHandle:
    @pytest.mark.anyio
    async def test_acks_and_ignores_non_event(self) -> None:
        adapter, _ = _make_adapter()
        client = MagicMock()
        client.send_socket_mode_response = AsyncMock()
        req = MagicMock()
        req.type = "hello"
        req.envelope_id = "env1"
        await adapter._handle(client, req)
        client.send_socket_mode_response.assert_awaited_once()

    @pytest.mark.anyio
    async def test_dispatches_event(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        client = MagicMock()
        client.send_socket_mode_response = AsyncMock()
        req = MagicMock()
        req.type = "events_api"
        req.envelope_id = "env1"
        req.payload = {"event": _msg_event("Sara go")}
        await adapter._handle(client, req)
        assert _inbox_size(agents["Sara"]) > 0

    @pytest.mark.anyio
    async def test_missing_event_returns(self) -> None:
        adapter, _ = _make_adapter()
        client = MagicMock()
        client.send_socket_mode_response = AsyncMock()
        req = MagicMock()
        req.type = "events_api"
        req.envelope_id = "env1"
        req.payload = {}
        await adapter._handle(client, req)  # No crash.

    @pytest.mark.anyio
    async def test_route_exception_is_logged(self) -> None:
        adapter, _ = _make_adapter()
        client = MagicMock()
        client.send_socket_mode_response = AsyncMock()
        req = MagicMock()
        req.type = "events_api"
        req.envelope_id = "env1"
        req.payload = {"event": {"type": "message", "text": "hi"}}
        with patch.object(
            SlackAdapter, "_route", new=AsyncMock(side_effect=RuntimeError("nope"))
        ):
            await adapter._handle(client, req)  # No crash; exception swallowed.


class TestEnsureLogChannel:
    @pytest.mark.anyio
    async def test_returns_cached(self) -> None:
        adapter, _ = _make_adapter()
        adapter._log_channels["sara-log"] = "C_CACHED"
        cid = await adapter.ensure_log_channel("sara")
        assert cid == "C_CACHED"

    @pytest.mark.anyio
    async def test_creates_new(self) -> None:
        adapter, _ = _make_adapter()
        with patch.object(
            Slack, "create_channel", new=AsyncMock(return_value="id=C_NEW")
        ):
            cid = await adapter.ensure_log_channel("sara")
        assert cid == "C_NEW"
        assert adapter._log_channel_owners["C_NEW"] == "sara"

    @pytest.mark.anyio
    async def test_falls_back_to_find_when_create_fails(self) -> None:
        adapter, _ = _make_adapter()
        err = ToolResult(call_id="", content="exists", is_error=True)
        with (
            patch.object(Slack, "create_channel", new=AsyncMock(return_value=err)),
            patch.object(
                SlackAdapter, "_find_channel", new=AsyncMock(return_value="C_FOUND")
            ),
        ):
            cid = await adapter.ensure_log_channel("sara")
        assert cid == "C_FOUND"
        assert adapter._log_channel_owners["C_FOUND"] == "sara"

    @pytest.mark.anyio
    async def test_returns_none_when_unfindable(self) -> None:
        adapter, _ = _make_adapter()
        err = ToolResult(call_id="", content="exists", is_error=True)
        with (
            patch.object(Slack, "create_channel", new=AsyncMock(return_value=err)),
            patch.object(
                SlackAdapter, "_find_channel", new=AsyncMock(return_value=None)
            ),
        ):
            cid = await adapter.ensure_log_channel("sara")
        assert cid is None

    @pytest.mark.anyio
    async def test_unexpected_create_response_returns_none(self) -> None:
        adapter, _ = _make_adapter()
        with patch.object(Slack, "create_channel", new=AsyncMock(return_value="weird")):
            cid = await adapter.ensure_log_channel("sara")
        assert cid is None

    @pytest.mark.anyio
    async def test_creates_and_syncs_members(self) -> None:
        adapter, _ = _make_adapter()
        sync_mock = AsyncMock()
        with (
            patch.object(
                Slack, "create_channel", new=AsyncMock(return_value="id=C_NEW")
            ),
            patch.object(SlackAdapter, "_sync_members", new=sync_mock),
        ):
            cid = await adapter.ensure_log_channel("sara", source_channel="C_SRC")
        assert cid == "C_NEW"
        sync_mock.assert_awaited_once()


class TestFindChannel:
    @pytest.mark.anyio
    async def test_finds_by_name(self) -> None:
        adapter, _ = _make_adapter()
        listing = "C100  #sara-log  (members=2)\nC200  #other  (members=5)"
        with patch.object(Slack, "_list_channels", new=AsyncMock(return_value=listing)):
            cid = await adapter._find_channel("sara-log")
        assert cid == "C100"

    @pytest.mark.anyio
    async def test_returns_none_when_listing_fails(self) -> None:
        adapter, _ = _make_adapter()
        err = ToolResult(call_id="", content="forbidden", is_error=True)
        with patch.object(Slack, "_list_channels", new=AsyncMock(return_value=err)):
            cid = await adapter._find_channel("sara-log")
        assert cid is None

    @pytest.mark.anyio
    async def test_returns_none_when_not_found(self) -> None:
        adapter, _ = _make_adapter()
        listing = "C100  #other  (members=2)"
        with patch.object(Slack, "_list_channels", new=AsyncMock(return_value=listing)):
            cid = await adapter._find_channel("missing")
        assert cid is None


class TestSyncMembers:
    @pytest.mark.anyio
    async def test_invites_non_bot_members(self) -> None:
        adapter, _ = _make_adapter()
        invite_mock = AsyncMock()
        with (
            patch.object(
                SlackAdapter,
                "_list_members",
                new=AsyncMock(return_value=["UBOT", "U1", "U2"]),
            ),
            patch.object(SlackAdapter, "_invite", new=invite_mock),
        ):
            await adapter._sync_members("C_SRC", "C_DST")
        # UBOT (self) excluded; U1 and U2 invited.
        assert invite_mock.await_count == 2


class TestListMembers:
    @pytest.mark.anyio
    async def test_returns_members_on_success(self) -> None:
        adapter, _ = _make_adapter()
        response = MagicMock()
        response.is_success = True
        response.json = MagicMock(return_value={"ok": True, "members": ["U1", "U2"]})
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=response)
        with patch("httpx.AsyncClient", return_value=client):
            members = await adapter._list_members("C_SRC")
        assert members == ["U1", "U2"]

    @pytest.mark.anyio
    async def test_returns_empty_on_failure(self) -> None:
        adapter, _ = _make_adapter()
        response = MagicMock()
        response.is_success = False
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=response)
        with patch("httpx.AsyncClient", return_value=client):
            members = await adapter._list_members("C_SRC")
        assert members == []

    @pytest.mark.anyio
    async def test_returns_empty_on_not_ok(self) -> None:
        adapter, _ = _make_adapter()
        response = MagicMock()
        response.is_success = True
        response.json = MagicMock(return_value={"ok": False})
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=response)
        with patch("httpx.AsyncClient", return_value=client):
            members = await adapter._list_members("C_SRC")
        assert members == []


class TestInvite:
    @pytest.mark.anyio
    async def test_posts_invite(self) -> None:
        adapter, _ = _make_adapter()
        post = AsyncMock()
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = post
        with patch("httpx.AsyncClient", return_value=client):
            await adapter._invite("C_DST", "U1")
        post.assert_awaited_once()


class TestMakeLogForwarder:
    def test_forwards_rendered_events(self) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        fwd = _make_log_forwarder(queue)
        fwd(UserMessage(text="hi"))
        assert queue.qsize() == 1
        assert queue.get_nowait() == "━━ input ━━\nhi"

    def test_skips_unrendered_events(self) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        fwd = _make_log_forwarder(queue)
        fwd(ModelResponsePartial(text="chunk"))
        assert queue.empty()


class TestFlushLog:
    @pytest.mark.anyio
    async def test_sends_single_chunk(self) -> None:
        spy = _SpySlack()
        await _flush_log(["one", "two"], "C1", spy)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]  -- spy duck-types Slack
        assert len(spy.sent) == 1
        assert spy.sent[0][1] == "one\ntwo"

    @pytest.mark.anyio
    async def test_splits_when_over_limit(self) -> None:
        spy = _SpySlack()
        big_line = "x" * 2000
        await _flush_log([big_line, big_line, big_line], "C1", spy)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]  -- spy
        # 3 × 2001 bytes > 3900 -> at least 2 sends.
        assert len(spy.sent) >= 2

    @pytest.mark.anyio
    async def test_empty_buffer_no_send(self) -> None:
        spy = _SpySlack()
        await _flush_log([], "C1", spy)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]  -- spy
        assert spy.sent == []


class _FakeLogChannelAdapter:
    """Minimal adapter for ``log_tap`` tests."""

    bot_token = "xoxb-fake"  # noqa: S105 -- test credential

    def __init__(self, channel_id: str | None = "C_LOG") -> None:
        self._channel_id = channel_id
        self.ensure_calls: list[tuple[str, str]] = []

    async def ensure_log_channel(
        self,
        agent_name: str,
        source_channel: str = "",
    ) -> str | None:
        self.ensure_calls.append((agent_name, source_channel))
        return self._channel_id


class TestLogTap:
    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_flushes_buffer_on_sentinel(self) -> None:
        adapter = _FakeLogChannelAdapter()
        events: asyncio.Queue[str | None] = asyncio.Queue()
        # Use a bare ``channel=...`` token so ``_extract_channel_from_text``
        # picks it up.
        events.put_nowait("channel=C_SRC] one")
        events.put_nowait("two")
        events.put_nowait(None)
        with patch.object(Slack, "send", new=AsyncMock(return_value="ok")) as send_mock:
            task = asyncio.create_task(log_tap(events, "sara", adapter))
            # Drain by letting the loop iterate until the queue is empty
            # and the sentinel's flush has fired.
            for _ in range(50):
                await asyncio.sleep(0)
                if send_mock.await_count >= 1:
                    break
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert send_mock.await_count >= 1
        # source_channel should have been extracted from the first line.
        assert adapter.ensure_calls[0][1] == "C_SRC"

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_skips_when_no_channel(self) -> None:
        adapter = _FakeLogChannelAdapter(channel_id=None)
        events: asyncio.Queue[str | None] = asyncio.Queue()
        events.put_nowait("hello")
        events.put_nowait(None)
        with patch.object(Slack, "send", new=AsyncMock()) as send_mock:
            task = asyncio.create_task(log_tap(events, "sara", adapter))
            for _ in range(50):
                await asyncio.sleep(0)
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # Adapter returned None, so nothing should ever be sent.
        assert send_mock.await_count == 0


class TestParseSlackArgs:
    def test_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        args, remaining = parse_slack_args(parser, [])
        assert args.app_token == ""
        assert args.bot_token == ""
        assert args.resume is False
        assert remaining == []

    def test_explicit_tokens(self) -> None:
        parser = argparse.ArgumentParser()
        args, _ = parse_slack_args(
            parser,
            [
                "--app-token",
                "A1",
                "--bot-token",
                "B1",
                "--continue",
            ],
        )
        assert args.app_token == "A1"  # noqa: S105 -- test value
        assert args.bot_token == "B1"  # noqa: S105 -- test value
        assert args.resume is True


class TestResolveTokens:
    def test_uses_explicit_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        ns = argparse.Namespace(
            app_token="A1",  # noqa: S106 -- test value
            bot_token="B1",  # noqa: S106 -- test value
        )
        assert _resolve_tokens(ns) == ("A1", "B1")

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_APP_TOKEN", "ENV_A")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "ENV_B")
        ns = argparse.Namespace(app_token="", bot_token="")
        assert _resolve_tokens(ns) == ("ENV_A", "ENV_B")

    def test_exits_on_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        ns = argparse.Namespace(app_token="", bot_token="")
        with pytest.raises(SystemExit):
            _ = _resolve_tokens(ns)


class TestStopAgent:
    def test_stop_agent_removes_from_manifest(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter(tmp_path=tmp_path)
        adapter._active_agents["Sara"] = {"persona": "x", "system": "y"}
        adapter.stop_agent("Sara")
        assert "Sara" not in adapter._active_agents

    def test_stop_agent_unknown_is_noop(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter(tmp_path=tmp_path)
        adapter.stop_agent("Ghost")  # No crash.


class TestAdapterConstruction:
    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_start_invokes_auth_and_connect(self) -> None:
        adapter, _ = _make_adapter()
        adapter._web = MagicMock()
        adapter._web.auth_test = AsyncMock(return_value={"user_id": "UNEW"})
        adapter._socket = MagicMock()
        adapter._socket.socket_mode_request_listeners = []
        adapter._socket.connect = AsyncMock()
        task = asyncio.create_task(adapter.start())
        for _ in range(50):
            await asyncio.sleep(0)
            if adapter.bot_user_id:
                break
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert adapter.bot_user_id == "UNEW"

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_start_with_router_log_channel_resolves(self) -> None:
        adapter, _ = _make_adapter()
        adapter._router_log_channel = "CABC123"
        adapter._web = MagicMock()
        adapter._web.auth_test = AsyncMock(return_value={"user_id": "UNEW"})
        adapter._socket = MagicMock()
        adapter._socket.socket_mode_request_listeners = []
        adapter._socket.connect = AsyncMock()
        task = asyncio.create_task(adapter.start())
        for _ in range(50):
            await asyncio.sleep(0)
            if adapter.bot_user_id:
                break
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert adapter.bot_user_id == "UNEW"

    @pytest.mark.anyio
    @pytest.mark.real_sleep
    async def test_spawn_agent_writes_manifest(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter(tmp_path=tmp_path)
        adapter._model = MagicMock()
        adapter._model_spec = MagicMock()
        adapter._compactor = None
        adapter._effort = None
        adapter._max_tool_call_rounds = None
        adapter._max_budget_usd = None
        fake_child = MagicMock()
        fake_child.name = "Sara"
        # ``serve_forever`` waits until cancel.
        serve_evt = asyncio.Event()

        async def _serve() -> None:
            await serve_evt.wait()

        fake_child.serve_forever = _serve
        fake_child.shutdown = MagicMock()
        fake_child.runtime.observers = []

        with (
            patch(
                "sagent.bin.slack.Agent",
                MagicMock(return_value=fake_child),
            ),
            patch(
                "sagent.bin.slack.resolve_tools",
                MagicMock(return_value=[]),
            ),
        ):
            await adapter.spawn_agent("Sara", "you are Sara")
            assert "Sara" in agent_registry
            assert "Sara" in adapter._active_agents
            manifest_path = tmp_path / "manifest.json"
            assert manifest_path.exists()
            # Let _run_child get past the try-finally setup.
            for _ in range(20):
                await asyncio.sleep(0)
            # Now signal serve_forever to exit so the finally blocks run.
            serve_evt.set()
            for _ in range(20):
                await asyncio.sleep(0)
            # Tear down any remaining tasks.
            for t in adapter._tasks:
                _ = t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    @pytest.mark.anyio
    async def test_stop_agent_shuts_down_real_agent(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter(tmp_path=tmp_path)
        fake = MagicMock(spec=RealAgent)
        agent_registry["Sara"] = fake
        adapter._active_agents["Sara"] = {"persona": "x", "system": "y"}
        adapter.stop_agent("Sara")
        fake.shutdown.assert_called_once()
        assert "Sara" not in adapter._active_agents

    @pytest.mark.anyio
    async def test_init_stores_attributes(self, tmp_path: Path) -> None:
        model = MagicMock()
        adapter = SlackAdapter(
            app_token="xapp-fake",  # noqa: S106 -- test credential
            bot_token="xoxb-fake",  # noqa: S106 -- test credential
            model=model,
            model_spec=ModelSpec(
                provider="OpenAI",
                auth="env",
                model_id="gpt",
                account="",
            ),
            persona_dir=tmp_path,
            session_dir=tmp_path / "session",
            compactor=None,
            log_prefix="agent-",
            router_log_channel="ch",
            effort="high",
            max_tool_call_rounds=5,
            max_budget_usd=1.0,
        )
        assert adapter.bot_token == "xoxb-fake"  # noqa: S105 -- test value
        assert adapter.bot_user_id == ""
        assert adapter._log_prefix == "agent-"
        assert adapter._router_log_channel == "ch"


class TestRouteEdgeCases:
    @pytest.mark.anyio
    async def test_empty_text_skipped(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara")
        await adapter._route(_msg_event("   "))
        assert spy.sent == []

    @pytest.mark.anyio
    async def test_empty_channel_skipped(self) -> None:
        adapter, spy = _make_adapter()
        _ = _register("Sara")
        await adapter._route(_msg_event("hi", channel=""))
        assert spy.sent == []

    @pytest.mark.anyio
    async def test_app_mention_etype_skipped(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event("hi")
        ev["type"] = "app_mention"
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_bot_with_unknown_subtype_skipped(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event(
            "weird",
            bot_id="B1",
            username="Sara",
            subtype="file_share",
            user="",
        )
        await adapter._route(ev)
        assert _inbox_size(agents["Sara"]) == 0

    @pytest.mark.anyio
    async def test_empty_command_string(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("", "C1", "1.0")
        assert result is False
        assert spy.sent == []


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
