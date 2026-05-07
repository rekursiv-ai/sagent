"""Tests for bin/slack.py -- deterministic Slack routing.

All tests mock the Slack SDK and LLM layers -- no network calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from sagent.bin.slack import (
    SlackAdapter,
    _AgentSlack,
    _extract_agent_mention,
    _extract_event,
    _list_personas,
    _render_event,
    _strip_mention,
    load_persona,
)
from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.json import JSON
from sagent.tools.core import agent_registry


# -- Fake agent for registry ------------------------------------------------


class _FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inbox: Deque[Message] = Deque()


def _register(*names: str) -> dict[str, _FakeAgent]:
    agents: dict[str, _FakeAgent] = {}
    for n in names:
        agent = _FakeAgent(n)
        agent_registry[n] = agent
        agents[n] = agent
    return agents


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    yield
    agent_registry.clear()


# -- Spy Slack tool ----------------------------------------------------------


class _SpySlack:
    """Records send() calls for assertion without touching the network."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    async def send(
        self,
        channel: str,
        text: str,
        thread_ts: str = "",
    ) -> str | Message:
        self.sent.append((channel, text, thread_ts))
        return f"Sent. ts=1.0 channel={channel}"

    @property
    def last_text(self) -> str:
        return self.sent[-1][1]


# -- Adapter factory -------------------------------------------------------


def _make_adapter(
    *,
    persona_dir: Path | None = None,
    tmp_path: Path | None = None,
) -> tuple[SlackAdapter, _SpySlack]:
    """Build a SlackAdapter with all external deps stubbed out."""
    adapter = object.__new__(SlackAdapter)
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
    adapter._slack = spy  # pyright: ignore[reportAttributeAccessIssue] -- test mock: injecting spy into private attr
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
) -> dict[str, Any]:
    ev: dict[str, Any] = {
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


# -- Pure helper tests -----------------------------------------------------


class TestExtractEvent:
    def test_valid_envelope(self) -> None:
        payload = cast(dict[str, Any], {"event": {"type": "app_mention", "text": "hi"}})
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
        _register("Sara")
        assert _extract_agent_mention("Sara check the deploy") == "Sara"

    def test_case_insensitive(self) -> None:
        _register("Sara")
        assert _extract_agent_mention("sara check the deploy") == "Sara"

    def test_no_match(self) -> None:
        assert _extract_agent_mention("hello world") is None


class TestLoadPersona:
    def test_loads_named(self, tmp_path: Path) -> None:
        (tmp_path / "sre.md").write_text("You are an SRE.")
        assert load_persona(tmp_path, "sre") == "You are an SRE."

    def test_falls_back_to_default(self, tmp_path: Path) -> None:
        (tmp_path / "default.md").write_text("Default persona.")
        assert load_persona(tmp_path, "nonexistent") == "Default persona."

    def test_generates_when_missing(self, tmp_path: Path) -> None:
        result = load_persona(tmp_path, "ghost")
        assert "ghost" in result


class TestListPersonas:
    def test_lists_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "sre.md").write_text("")
        (tmp_path / "pm.md").write_text("")
        (tmp_path / "notes.txt").write_text("")
        assert _list_personas(tmp_path) == ["pm", "sre"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert _list_personas(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert _list_personas(tmp_path / "nope") == []


# -- _AgentSlack discovery -------------------------------------------------


class TestAgentSlack:
    def test_prompt_lists_peers(self) -> None:
        _register("Sara", "Bob")
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        p = tool.prompt()
        assert "Bob" in p
        assert "Sara" not in p

    def test_prompt_empty_when_alone(self) -> None:
        _register("Sara")
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        assert tool.prompt() == ""

    def test_prompt_empty_no_agents(self) -> None:
        tool = _AgentSlack(token="x", username="Sara")  # noqa: S106 -- test credential
        assert tool.prompt() == ""


# -- Routing: basic cases --------------------------------------------------


class TestRouteHumanMessages:
    @pytest.mark.anyio
    async def test_ignores_own_messages(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("hi", user="UBOT"))
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_ignores_non_message_types(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        ev = _msg_event("hi")
        ev["type"] = "reaction_added"
        await adapter._route(ev)
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_ignores_message_subtypes(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("hi", subtype="channel_join"))
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_route_to_log_channel_owner(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._log_channel_owners["C_LOG"] = "Sara"
        await adapter._route(_msg_event("check this", channel="C_LOG"))
        items = agents["Sara"].inbox.drain()
        assert len(items) == 1
        assert "check this" in str(items[0].content)

    @pytest.mark.anyio
    async def test_route_by_agent_name_mention(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        await adapter._route(_msg_event("Sara do the thing"))
        assert not agents["Sara"].inbox.empty()
        assert agents["Bob"].inbox.empty()

    @pytest.mark.anyio
    async def test_agent_mention_sets_thread_owner(self) -> None:
        adapter, _ = _make_adapter()
        _register("Sara")
        await adapter._route(_msg_event("Sara do X", ts="1.0"))
        assert adapter._thread_owners[("C123", "1.0")] == "Sara"

    @pytest.mark.anyio
    async def test_thread_continuation(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara", "Bob")
        adapter._thread_owners[("C123", "1.0")] = "Sara"
        await adapter._route(_msg_event("follow up", thread_ts="1.0", ts="2.0"))
        assert not agents["Sara"].inbox.empty()
        assert agents["Bob"].inbox.empty()

    @pytest.mark.anyio
    async def test_single_agent_default(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_msg_event("do something"))
        assert not agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_single_agent_default_sets_thread_owner(self) -> None:
        adapter, _ = _make_adapter()
        _register("Sara")
        await adapter._route(_msg_event("do something", ts="5.0"))
        assert adapter._thread_owners[("C123", "5.0")] == "Sara"

    @pytest.mark.anyio
    async def test_ambiguous_agents_replies(self) -> None:
        adapter, spy = _make_adapter()
        agents = _register("Sara", "Bob")
        await adapter._route(_msg_event("do something"))
        assert agents["Sara"].inbox.empty()
        assert agents["Bob"].inbox.empty()
        assert len(spy.sent) == 1
        assert "Bob" in spy.last_text
        assert "Sara" in spy.last_text

    @pytest.mark.anyio
    async def test_no_agents_replies_with_guidance(self, tmp_path: Path) -> None:
        (tmp_path / "sre.md").write_text("")
        adapter, spy = _make_adapter(persona_dir=tmp_path)
        await adapter._route(_msg_event("hello"))
        assert len(spy.sent) == 1
        assert "create" in spy.last_text.lower()
        assert "sre" in spy.last_text


# -- Routing: agent-to-agent -----------------------------------------------


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
        assert not agents["Bob"].inbox.empty()
        assert agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()

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
        assert not agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()
        assert agents["Bob"].inbox.empty()
        assert len(spy.sent) == 0

    @pytest.mark.anyio
    async def test_agent_message_no_commands(self) -> None:
        adapter, spy = _make_adapter()
        _register("Sara")
        ev = _msg_event(
            "help",
            bot_id="B1",
            username="Sara",
            subtype="bot_message",
            user="",
        )
        await adapter._route(ev)
        assert len(spy.sent) == 0


# -- Command handling ------------------------------------------------------


class TestCommands:
    @pytest.mark.anyio
    async def test_help(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("help", "C1", "1.0")
        assert result is True
        assert "create" in spy.last_text

    @pytest.mark.anyio
    async def test_list_no_agents(self, tmp_path: Path) -> None:
        (tmp_path / "sre.md").write_text("")
        adapter, spy = _make_adapter(persona_dir=tmp_path)
        result = await adapter._try_command("list", "C1", "1.0")
        assert result is True
        assert "No active" in spy.last_text
        assert "sre" in spy.last_text

    @pytest.mark.anyio
    async def test_list_with_agents(self) -> None:
        adapter, spy = _make_adapter()
        _register("Sara", "Bob")
        result = await adapter._try_command("list", "C1", "1.0")
        assert result is True
        assert "Sara" in spy.last_text
        assert "Bob" in spy.last_text

    @pytest.mark.anyio
    async def test_create_duplicate(self) -> None:
        adapter, spy = _make_adapter()
        _register("Sara")
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
        adapter, _ = _make_adapter(tmp_path=tmp_path)
        agents = _register("Sara")
        result = await adapter._try_command("stop Sara", "C1", "1.0")
        assert result is True
        items = agents["Sara"].inbox.drain()
        assert any(item.descriptor == "text/x-quit" for item in items)

    @pytest.mark.anyio
    async def test_unknown_not_a_command(self) -> None:
        adapter, spy = _make_adapter()
        result = await adapter._try_command("deploy everything", "C1", "1.0")
        assert result is False
        assert len(spy.sent) == 0

    @pytest.mark.anyio
    async def test_create_with_custom_label(self, tmp_path: Path) -> None:
        (tmp_path / "sre.md").write_text("You are an SRE.")
        adapter, _ = _make_adapter(persona_dir=tmp_path)
        mock = AsyncMock()
        adapter.spawn_agent = mock  # ty: ignore[invalid-assignment] -- test mock
        result = await adapter._try_command("create sre as ops", "C1", "1.0")
        assert result is True
        mock.assert_called_once_with("ops", "You are an SRE.")

    @pytest.mark.anyio
    async def test_create_loads_persona(self, tmp_path: Path) -> None:
        (tmp_path / "pm.md").write_text("You are a PM.")
        adapter, _ = _make_adapter(persona_dir=tmp_path)
        mock = AsyncMock()
        adapter.spawn_agent = mock  # ty: ignore[invalid-assignment] -- test mock
        result = await adapter._try_command("create pm", "C1", "1.0")
        assert result is True
        mock.assert_called_once_with("pm", "You are a PM.")


# -- Integration: full route → command → reply flow ------------------------


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
        _register("Sara")
        await adapter._route(_msg_event("list"))
        assert len(spy.sent) == 1
        assert "Sara" in spy.last_text

    @pytest.mark.anyio
    async def test_command_skipped_when_agent_mentioned(self) -> None:
        """'help' routes to agent named 'help' if one exists (step 2 > step 4)."""
        agents = _register("help")
        adapter, spy = _make_adapter()
        await adapter._route(_msg_event("help me with this"))
        assert not agents["help"].inbox.empty()
        assert len(spy.sent) == 0


# -- Reaction event helper ---------------------------------------------------


def _reaction_event(
    reaction: str,
    *,
    channel: str = "C123",
    msg_ts: str = "1.0",
    user: str = "UHUMAN",
    item_type: str = "message",
    item_user: str = "",
) -> dict[str, Any]:
    ev: dict[str, Any] = {
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


# -- Routing: reactions ----------------------------------------------------


class TestRouteReactions:
    @pytest.mark.anyio
    async def test_reaction_routed_to_agent(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hello world", "0.5")
        await adapter._route(_reaction_event("heart"))
        items = agents["Sara"].inbox.drain()
        assert len(items) == 1
        assert ":heart:" in str(items[0].content)
        assert "hello world" in str(items[0].content)

    @pytest.mark.anyio
    async def test_reaction_includes_user_name(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._user_names["UHUMAN"] = "Josh"
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "0.5")
        await adapter._route(_reaction_event("thumbsup"))
        items = agents["Sara"].inbox.drain()
        assert "Josh" in str(items[0].content)
        assert ":thumbsup:" in str(items[0].content)

    @pytest.mark.anyio
    async def test_reaction_uses_cached_thread_ts(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "2.0")] = ("Sara", "reply text", "1.0")
        await adapter._route(_reaction_event("heart", msg_ts="2.0"))
        items = agents["Sara"].inbox.drain()
        assert "thread_ts=1.0" in str(items[0].content)

    @pytest.mark.anyio
    async def test_reaction_ignored_when_not_cached(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        await adapter._route(_reaction_event("heart"))
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_reaction_ignored_for_non_message_item(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart", item_type="file"))
        assert agents["Sara"].inbox.empty()

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
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_reaction_to_non_bot_message_ignored(self) -> None:
        adapter, _ = _make_adapter()
        agents = _register("Sara")
        adapter._sent_messages[("C123", "1.0")] = ("Sara", "hi", "1.0")
        await adapter._route(_reaction_event("heart", item_user="UOTHER"))
        assert agents["Sara"].inbox.empty()

    @pytest.mark.anyio
    async def test_reaction_missing_fields_ignored(self) -> None:
        adapter, _ = _make_adapter()
        _register("Sara")
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
        _register("Sara", "Bob")
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
        items = agents["Sara"].inbox.drain()
        assert len(items) == 1
        assert ":heart:" in str(items[0].content)
        assert '""' not in str(items[0].content)


# -- Log rendering --------------------------------------------------------


class TestRenderEvent:
    def test_plain_text(self) -> None:
        ev = TextMessage("hello", "text/plain")
        assert _render_event(ev) == "hello"

    def test_error(self) -> None:
        ev = TextMessage("oops", "text/x-error")
        assert _render_event(ev) == "✗ oops"

    def test_thinking(self) -> None:
        ev = TextMessage("pondering...", "text/x-thinking")
        assert _render_event(ev) == "💭 pondering..."

    def test_thinking_empty(self) -> None:
        ev = TextMessage("", "text/x-thinking")
        assert _render_event(ev) is None

    def test_tool_label(self) -> None:
        ev = TextMessage("Bash ls", "text/x-tool-label")
        assert _render_event(ev) == "  Bash ls"

    def test_tool_result_with_content(self) -> None:
        parts = (
            TextMessage("file.txt", "text/plain"),
            TextMessage("", "text/x-hint-tool-use-nudge"),
        )
        ev = MultipartMessage(parts, "multipart/x-tool-result")
        result = _render_event(ev)
        assert result is not None
        assert "file.txt" in result

    def test_tool_result_error(self) -> None:
        parts = (TextMessage("failed", "text/x-error"),)
        ev = MultipartMessage(parts, "multipart/x-tool-result")
        result = _render_event(ev)
        assert result is not None
        assert "✗ failed" in result

    def test_tool_result_diff(self) -> None:
        parts = (
            TextMessage("Replaced 1 occurrence(s)", "text/plain"),
            TextMessage("@@ -1,1 +1,1 @@\n-old\n+new", "text/x-diff"),
        )
        ev = MultipartMessage(parts, "multipart/x-tool-result")
        result = _render_event(ev)
        assert result is not None
        assert "```diff" in result
        assert "-old" in result
        assert "+new" in result

    def test_user_input(self) -> None:
        ev = TextMessage("user msg", "text/x-user-injected")
        result = _render_event(ev)
        assert result is not None
        assert "user msg" in result

    def test_skips_done(self) -> None:
        ev = JsonMessage(cast(JSON, {}), "application/x-done")
        assert _render_event(ev) is None

    def test_skips_status_changed(self) -> None:
        ev = TextMessage("idle", "text/x-signal-status-changed")
        assert _render_event(ev) is None

    def test_interrupted(self) -> None:
        ev = TextMessage("", "text/x-interrupted")
        assert _render_event(ev) == "[interrupted]"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
