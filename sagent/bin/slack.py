#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"

Slack service: deterministic message routing to persistent agents.

Connects to Slack via Socket Mode, routes messages to persistent child
agents using deterministic rules (log channels, @mentions, thread
ownership). Agent lifecycle is managed via simple text commands.

Commands (in any message to the bot):
    create <persona> [as <label>]  -- spawn agent from persona file
    list                           -- active agents and available personas
    stop <name>                    -- stop an agent
    help                           -- show commands

Routing priority:
    1. Log channel -> owning agent
    2. Agent name at start of message -> that agent
    3. Thread continuation -> same agent as thread starter
    4. Single active agent -> route to it
    5. Otherwise -> reply with guidance

Setup
=====

1. Create a Slack app at https://api.slack.com/apps.
2. Enable Socket Mode (Basic Information -> Socket Mode).
3. Generate an app-level token with connections:write scope
   (xapp-...).
4. Under OAuth & Permissions add bot scopes: chat:write,
   chat:write.customize, channels:manage, channels:read,
   channels:history, app_mentions:read, im:history,
   im:read, im:write, users:read, groups:history,
   reactions:read.
   Install to workspace (xoxb-...).
5. Under Event Subscriptions subscribe to app_mention,
   message.im, and reaction_added.

Usage
=====

::

    export SLACK_APP_TOKEN=xapp-...
    export SLACK_BOT_TOKEN=xoxb-...
    ./slack.py
    ./slack.py --provider Google --auth env --model gemini-3.1-pro-preview
    ./slack.py --persona-dir ./personas
'''
# fmt: on

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time as _time

from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

import httpx2

from sagent.agent import Agent
from sagent.agent.state import agent_registry
from sagent.bin.cli import (
    DEFAULT_TOOLS,
    parse_agent_args,
    resolve_tools,
)
from sagent.compaction.summary import SummaryCompactor
from sagent.lib.custom_json import MutableJSON
from sagent.lib.userdirs import data_dir
from sagent.providers import build_provider
from sagent.tools.slack import Slack
from sagent.types.model import ModelRecipe
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponseError,
    ModelResponseThinking,
    RuntimeEvent,
    ToolLabel,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest

    from sagent.types.model import Model

logger = logging.getLogger(__name__)

# Agent-sent messages retained for reaction lookups. Reactions land on
# recent messages, so the tail is what matters; the cache is a lookup
# table, not a transcript.
_SENT_MESSAGE_CACHE: Final = 512

_AGENT_TOOL_NAMES = [
    name for name in DEFAULT_TOOLS if name not in ("AgentSpawn", "AgentSend")
]


def _extract_event(payload: MutableJSON | None) -> MutableJSON | None:
    """Pull the inner ``event`` dict from a Socket Mode envelope."""
    if not isinstance(payload, dict):
        return None
    ev = payload.get("event")
    if not isinstance(ev, dict):
        return None
    return cast(MutableJSON, ev)


def _strip_mention(text: str, self_user_id: str) -> str:
    """Remove the bot's ``<@U...>`` mention from text."""
    if not self_user_id:
        return text.strip()
    return text.replace(f"<@{self_user_id}>", "").strip()


def _extract_agent_mention(text: str) -> str | None:
    """Return the agent label that ``text`` opens with, or ``None``."""
    for word in text.split():
        clean = word.rstrip(",:;.!?")
        if not clean:
            break
        if clean in agent_registry:
            return clean
        lower = clean.lower()
        for label in agent_registry:
            if label.lower() == lower:
                return label
        break
    return None


def _manifest_path(session_dir: Path) -> Path:
    """Return the manifest.json path inside a session directory."""
    return session_dir / "manifest.json"


def _new_session_dir(root: Path) -> Path:
    """Create a timestamped session directory under ``root``."""
    name = _time.strftime("%Y%m%d_%H%M%S")
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_session_dir(root: Path) -> Path | None:
    """Return the most recently modified session directory under ``root``."""
    if not root.is_dir():
        return None
    candidates = sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "manifest.json").exists()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _save_manifest(
    session_dir: Path,
    agents: dict[str, dict[str, str]],
) -> None:
    """Persist ``{label: {persona, system}}`` to ``manifest.json``."""
    session_dir.mkdir(parents=True, exist_ok=True)
    _ = _manifest_path(session_dir).write_text(
        json.dumps(agents, indent=2),
        encoding="utf-8",
    )


def _load_manifest(session_dir: Path) -> dict[str, dict[str, str]]:
    """Load ``{label: {persona, system}}`` from ``manifest.json``."""
    path = _manifest_path(session_dir)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            result[k] = {"persona": v, "system": ""}
        else:
            result[k] = dict(v)
    return result


def load_persona(persona_dir: Path, persona_name: str) -> str:
    """Load a persona markdown file, falling back to ``default.md``.

    Args:
      persona_dir: Directory containing persona ``.md`` files.
      persona_name: Stem name of the persona file to load.

    Returns:
      persona_text: Persona content, or a synthesised fallback string.

    """
    path = persona_dir / f"{persona_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    default = persona_dir / "default.md"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return f"You are {persona_name}."


def _list_personas(persona_dir: Path) -> list[str]:
    """List available persona names (stems of ``.md`` files)."""
    if not persona_dir.is_dir():
        return []
    return sorted(p.stem for p in persona_dir.glob("*.md"))


class _AgentSlack(Slack):
    """Slack tool that advertises active peer agents in its prompt."""

    @override
    def prompt(self) -> str:
        """Return tool-prompt text listing peer agents this one can @-mention."""
        others = sorted(k for k in agent_registry if k != self._username)
        if not others:
            return ""
        names = ", ".join(others)
        return (
            f"Active agents you can message by @-mentioning at the start"
            f" of your Slack message: {names}"
        )


class SlackAdapter:
    """Socket Mode listener with deterministic routing and agent lifecycle."""

    def __init__(
        self,
        *,
        app_token: str,
        bot_token: str,
        model: Model,
        model_recipe: ModelRecipe,
        persona_dir: Path,
        session_dir: Path,
        compactor: SummaryCompactor | None,
        log_prefix: str,
        router_log_channel: str = "",
        effort: str | None = None,
        max_tool_call_rounds: int | None = None,
        max_budget_usd: float | None = None,
    ) -> None:
        self._web = AsyncWebClient(token=bot_token)
        self._socket = SocketModeClient(
            app_token=app_token,
            web_client=self._web,
        )
        self._bot_token = bot_token
        self._slack = Slack(token=bot_token)
        self._model = model
        self._model_recipe = model_recipe
        self._persona_dir = persona_dir
        self._session_dir = session_dir
        self._compactor = compactor
        self._log_prefix = log_prefix
        self._router_log_channel = router_log_channel
        self._effort = effort
        self._max_tool_call_rounds = max_tool_call_rounds
        self._max_budget_usd = max_budget_usd
        self._self_user_id: str = ""
        self._log_channels: dict[str, str] = {}
        self._log_channel_owners: dict[str, str] = {}
        self._thread_owners: dict[tuple[str, str], str] = {}
        self._active_agents: dict[str, dict[str, str]] = {}
        # Set + done-callback rather than a list: ``stop_agent`` never
        # pruned it, so every create/stop cycle leaked a task object for
        # the life of the adapter.
        self._tasks: set[asyncio.Task[None]] = set()
        self._bg_tasks: set[asyncio.Task[object]] = set()
        self._user_names: dict[str, str] = {}
        # Cache of agent-sent messages: (channel, ts) → (agent, text, thread_ts).
        # Bounded: reactions arrive against recent messages, so an
        # unbounded dict retained every bot message for the life of a
        # service that runs for weeks.
        self._sent_messages: OrderedDict[tuple[str, str], tuple[str, str, str]] = (
            OrderedDict()
        )

    @property
    def bot_user_id(self) -> str:
        """Return the bot's Slack user ID."""
        return self._self_user_id

    @property
    def bot_token(self) -> str:
        """Return the Slack bot token."""
        return self._bot_token

    async def _resolve_user(self, user_id: str) -> str:
        """Resolve a Slack user ID to a display name, with caching."""
        if user_id in self._user_names:
            return self._user_names[user_id]
        try:
            resp = await self._web.users_info(user=user_id)  # pyright: ignore[reportUnknownMemberType] -- slack_sdk stubs
            user_obj = cast(MutableJSON, resp.get("user") or {})
            profile = cast(MutableJSON, user_obj.get("profile") or {})
            name = str(
                profile.get("display_name")
                or user_obj.get("real_name")
                or user_obj.get("name")
                or user_id
            )
            self._user_names[user_id] = name
        except (KeyError, AttributeError, OSError, SlackApiError):
            self._user_names[user_id] = user_id
        return self._user_names[user_id]

    def _log_route(self, text: str) -> None:
        """Log a routing decision to logger and optionally to a Slack channel."""
        logger.info("%s", text)
        if self._router_log_channel:
            task = asyncio.create_task(
                self._slack.send(self._router_log_channel, text),
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def start(self) -> None:
        """Connect to Socket Mode and listen for events until cancelled."""
        auth_test = cast(Callable[[], Awaitable[object]], self._web.auth_test)
        auth = cast(MutableJSON, await auth_test())
        self._self_user_id = str(auth.get("user_id", ""))
        logger.info("Bot user_id=%s", self._self_user_id)
        if self._router_log_channel:
            await self._resolve_router_log_channel()
        self._socket.socket_mode_request_listeners.append(self._handle)
        await self._socket.connect()
        await asyncio.Event().wait()

    async def _resolve_router_log_channel(self) -> None:
        """Resolve a channel name to an ID, creating the channel if needed."""
        name = self._router_log_channel.lstrip("#")
        # Already a channel ID (starts with C).
        if name.startswith("C") and name[1:].isalnum():
            return
        # Look up by name.
        cid = await self._find_channel(name)
        if cid:
            self._router_log_channel = cid
            logger.info("Router log channel: #%s (%s)", name, cid)
            return
        # Create it.
        result = await self._slack.create_channel(name)
        if isinstance(result, ToolResult):
            logger.warning("Could not create/find #%s: %s", name, result.content)
            self._router_log_channel = ""
            return
        if result.startswith("id="):
            self._router_log_channel = result[3:]
            logger.info(
                "Created router log channel: #%s (%s)", name, self._router_log_channel
            )
        else:
            self._router_log_channel = ""

    async def _handle(
        self,
        client: AsyncBaseSocketModeClient,
        req: SocketModeRequest,
    ) -> None:
        """Ack the Socket Mode request and dispatch the event to ``_route``."""
        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id),
        )
        if req.type != "events_api":
            return
        payload = cast(MutableJSON, req.payload)
        event = _extract_event(payload)
        if event is None:
            return
        try:
            await self._route(event)
        except Exception:
            logger.exception("Error routing event: %s", event.get("type"))

    async def _route(self, event: MutableJSON) -> None:
        """Route one Slack event to its target agent or to a command handler."""
        if event.get("user") == self._self_user_id:
            return

        etype = event.get("type")

        if etype == "reaction_added":
            await self._route_reaction(event)
            return

        text = str(event.get("text") or "").strip()
        channel = str(event.get("channel") or "")
        if not channel or not text:
            return

        if etype != "message":
            return

        subtype = event.get("subtype")

        # Identify agent-originated messages by username.
        sender_agent: str | None = None
        if event.get("bot_id"):
            if subtype not in (None, "bot_message"):
                return
            username = str(event.get("username") or "")
            if username in agent_registry:
                sender_agent = username
            else:
                return  # foreign bot
        elif subtype is not None:
            return

        ts = str(event.get("ts") or "")
        thread_ts = str(event.get("thread_ts") or ts)
        clean = _strip_mention(text, self._self_user_id)
        user_id = str(event.get("user") or event.get("username") or "someone")
        user = await self._resolve_user(user_id)
        source = f"[channel={channel} thread_ts={thread_ts} user={user_id}]"
        formatted = f"{source} {clean}"

        # Cache agent messages for reaction lookups.
        if sender_agent and ts:
            self._sent_messages[(channel, ts)] = (sender_agent, clean, thread_ts)
            while len(self._sent_messages) > _SENT_MESSAGE_CACHE:
                _ = self._sent_messages.popitem(last=False)

        # 1. Log channel → owning agent (skip self-routing).
        if channel in self._log_channel_owners:
            owner = self._log_channel_owners[channel]
            if owner in agent_registry and owner != sender_agent:
                self._log_route(f"[{user}->{owner}] log-channel {clean}")
                agent_registry[owner].runtime.inbox.push_back(
                    UserMessage(text=formatted),
                )
            return

        # 2. Explicit agent name at start of message.
        target = _extract_agent_mention(clean)
        if target and target in agent_registry and target != sender_agent:
            thread_key = (channel, thread_ts)
            if thread_key not in self._thread_owners:
                self._thread_owners[thread_key] = target
            self._log_route(f"[{user}->{target}] mention {clean}")
            agent_registry[target].runtime.inbox.push_back(
                UserMessage(text=formatted),
            )
            return

        # 3. Thread continuation.
        thread_key = (channel, thread_ts)
        if thread_key in self._thread_owners:
            owner = self._thread_owners[thread_key]
            if owner in agent_registry and owner != sender_agent:
                self._log_route(f"[{user}->{owner}] thread {clean}")
                agent_registry[owner].runtime.inbox.push_back(
                    UserMessage(text=formatted),
                )
                return

        # Agent messages that don't match steps 1-3: drop.
        if sender_agent:
            self._log_route(f"[{sender_agent}] dropped {clean}")
            return

        # 4. Try as command (human messages only).
        if await self._try_command(clean, channel, thread_ts):
            self._log_route(f"[{user}] command {clean}")
            return

        # 5. Single agent default.
        agents = list(agent_registry)
        if len(agents) == 1:
            self._thread_owners[(channel, thread_ts)] = agents[0]
            self._log_route(f"[{user}->{agents[0]}] default {clean}")
            agent_registry[agents[0]].runtime.inbox.push_back(
                UserMessage(text=formatted),
            )
            return

        # 6. Ambiguous or no agents.
        self._log_route(f"[{user}] unrouted {clean}")
        if agents:
            names = ", ".join(sorted(agents))
            await self._reply(
                channel,
                thread_ts,
                f"Active agents: {names}. Mention one by name.",
            )
        else:
            personas = ", ".join(_list_personas(self._persona_dir)) or "(none)"
            await self._reply(
                channel,
                thread_ts,
                f"No agents active. `create <persona>` to start one."
                f"\nPersonas: {personas}",
            )

    async def _route_reaction(self, event: MutableJSON) -> None:
        """Route a reaction_added event to the agent whose message was reacted to."""
        user_id = str(event.get("user") or "")
        reaction = str(event.get("reaction") or "")
        raw_item = event.get("item")
        if not isinstance(raw_item, dict) or not reaction or not user_id:
            return
        item = cast(MutableJSON, raw_item)

        if item.get("type") != "message":
            return

        channel = str(item.get("channel") or "")
        msg_ts = str(item.get("ts") or "")
        if not channel or not msg_ts:
            return

        # Quick filter: skip reactions to non-bot messages.
        item_user = str(event.get("item_user") or "")
        if item_user and item_user != self._self_user_id:
            return

        # Look up the cached agent message.
        cached = self._sent_messages.get((channel, msg_ts))
        if cached is None:
            return

        agent_name, msg_text, thread_ts = cached
        if agent_name not in agent_registry:
            return

        user = await self._resolve_user(user_id)

        # Format the reaction as a user input to the agent.
        source = f"[channel={channel} thread_ts={thread_ts} user={user_id}]"
        if msg_text:
            formatted = (
                f"{source} {user} reacted with :{reaction}:"
                f' to your message: "{msg_text}"'
            )
        else:
            formatted = f"{source} {user} reacted with :{reaction}: to your message"

        self._log_route(f"[{user}->{agent_name}] reaction :{reaction}:")
        agent_registry[agent_name].runtime.inbox.push_back(
            UserMessage(text=formatted),
        )

    # -- Command handling --------------------------------------------------

    async def _try_command(
        self,
        text: str,
        channel: str,
        thread_ts: str,
    ) -> bool:
        """Dispatch ``text`` as a slash-free command; return True if handled."""
        parts = text.split()
        if not parts:
            return False
        cmd = parts[0].lower()

        if cmd == "help":
            await self._reply(
                channel,
                thread_ts,
                "Commands:\n"
                "• `create <persona>` -- spawn agent"
                " (`create <p> as <label>` for custom name)\n"
                "• `list` -- active agents and available personas\n"
                "• `stop <name>` -- stop an agent\n"
                "• `help` -- this message",
            )
            return True

        if cmd == "list":
            lines: list[str] = []
            if agent_registry:
                lines.append("*Active agents:*")
                lines.extend(f"  • {label}" for label in sorted(agent_registry))
            else:
                lines.append("No active agents.")
            personas = _list_personas(self._persona_dir)
            if personas:
                lines.append(f"*Personas:* {', '.join(personas)}")
            await self._reply(channel, thread_ts, "\n".join(lines))
            return True

        if cmd == "create" and len(parts) >= 2:
            persona_name = parts[1]
            label = persona_name
            if len(parts) >= 4 and parts[2].lower() == "as":
                label = parts[3]
            if label in agent_registry:
                await self._reply(
                    channel,
                    thread_ts,
                    f"`{label}` already exists.",
                )
                return True
            system_text = load_persona(self._persona_dir, persona_name)
            await self.spawn_agent(label, system_text)
            await self._reply(channel, thread_ts, f"Agent `{label}` created.")
            return True

        if cmd == "stop" and len(parts) >= 2:
            label = parts[1]
            if label not in agent_registry:
                await self._reply(
                    channel,
                    thread_ts,
                    f"No agent `{label}`.",
                )
                return True
            self.stop_agent(label)
            await self._reply(channel, thread_ts, f"Agent `{label}` stopped.")
            return True

        return False

    # -- Agent lifecycle ---------------------------------------------------

    async def spawn_agent(
        self,
        label: str,
        system_text: str,
    ) -> None:
        """Create a persistent child agent under ``label`` with the given persona.

        Args:
          label: Unique agent name registered in ``agent_registry``.
          system_text: System prompt that defines the agent's identity.

        """
        base_tools = resolve_tools(_AGENT_TOOL_NAMES)
        tools = [*base_tools, _AgentSlack(token=self._bot_token, username=label)]
        child = Agent(
            name=label,
            description=f"Persistent agent: {label}",
            model=self._model,
            model_recipe=self._model_recipe,
            system=system_text,
            tools=tools,
            compactor=self._compactor,
            effort=self._effort,
            max_tool_call_rounds=self._max_tool_call_rounds,
            max_budget_usd=self._max_budget_usd,
            session_dir=self._session_dir / label,
            persistent_retry=True,
        )
        child._lifecycle = "serviced"  # noqa: SLF001 -- cross-layer serviced flag
        child._is_subagent = True  # noqa: SLF001 -- cross-layer subagent flag
        agent_registry[label] = child
        self._active_agents[label] = {"persona": "custom", "system": system_text}
        _save_manifest(self._session_dir, self._active_agents)

        log_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _run_child(c: Agent = child) -> None:
            try:
                fwd = _make_log_forwarder(log_queue)
                c.runtime.observers.append(fwd)
                try:
                    await c.serve_forever()
                finally:
                    if fwd in c.runtime.observers:
                        c.runtime.observers.remove(fwd)
                    log_queue.put_nowait(None)
            finally:
                _ = agent_registry.pop(c.name, None)

        for coro in (_run_child(), log_tap(log_queue, label, self)):
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def stop_agent(self, label: str) -> None:
        """Shut down the agent registered under ``label`` and persist manifest."""
        agent = agent_registry.get(label)
        if agent is not None and isinstance(agent, Agent):
            agent.shutdown()
        _ = self._active_agents.pop(label, None)
        _save_manifest(self._session_dir, self._active_agents)

    async def _reply(self, channel: str, thread_ts: str, text: str) -> None:
        """Post ``text`` into ``channel`` (threaded on ``thread_ts``)."""
        _ = await self._slack.send(channel, text, thread_ts)

    # -- Log channel management --------------------------------------------

    async def ensure_log_channel(
        self,
        agent_name: str,
        source_channel: str = "",
    ) -> str | None:
        """Create or find the per-agent log channel and sync its membership.

        Args:
          agent_name: Agent label used to derive the channel name.
          source_channel: Channel ID whose members are synced into the log channel.

        Returns:
          channel_id: Slack channel ID, or ``None`` on failure.

        """
        ch_name = f"{self._log_prefix}{agent_name}-log".lower()
        if ch_name in self._log_channels:
            return self._log_channels[ch_name]
        slack = Slack(token=self._bot_token)
        result = await slack.create_channel(ch_name)
        if isinstance(result, ToolResult):
            cid = await self._find_channel(ch_name)
            if cid:
                self._log_channels[ch_name] = cid
                self._log_channel_owners[cid] = agent_name
                return cid
            logger.warning(
                "Failed to create/find #%s: %s",
                ch_name,
                result.content,
            )
            return None
        if result.startswith("id="):
            cid = result[3:]
            self._log_channels[ch_name] = cid
            self._log_channel_owners[cid] = agent_name
            if source_channel:
                await self._sync_members(source_channel, cid)
            return cid
        return None

    async def _sync_members(
        self,
        source_channel: str,
        target_channel: str,
    ) -> None:
        """Invite every non-bot member of ``source_channel`` into ``target_channel``."""
        members = await self._list_members(source_channel)
        for uid in members:
            if uid == self._self_user_id:
                continue
            await self._invite(target_channel, uid)

    async def _list_members(self, channel: str) -> list[str]:
        """Return user IDs of members in ``channel``, or ``[]`` on failure."""
        url = "https://slack.com/api/conversations.members"
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        params = {"channel": channel, "limit": 1000}
        async with httpx2.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, headers=headers, params=params)
            if not r.is_success:
                return []
            body = r.json()
        if not body.get("ok"):
            return []
        members = cast(list[object], body.get("members") or [])
        return [str(m) for m in members]

    async def _invite(self, channel: str, user: str) -> None:
        """Invite ``user`` to ``channel`` via the Slack web API."""
        url = "https://slack.com/api/conversations.invite"
        headers = {
            "Authorization": f"Bearer {self._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        async with httpx2.AsyncClient(timeout=30.0) as client:
            _ = await client.post(
                url,
                headers=headers,
                content=json.dumps({"channel": channel, "users": user}),
            )

    async def _find_channel(self, name: str) -> str | None:
        """Look up a Slack channel ID by its bare name."""
        slack = Slack(token=self._bot_token)
        result = await slack._list_channels(1000)  # noqa: SLF001 -- no public channel-list API on Slack wrapper
        if isinstance(result, ToolResult):
            return None
        for line in result.splitlines():
            if f"#{name}" in line:
                return line.split()[0]
        return None


def _extract_channel_from_text(text: str) -> str:
    """Return the ``channel=<id>`` token value from ``text``, or ``""``."""
    for part in text.split():
        if part.startswith("channel="):
            return part[8:].rstrip("]")
    return ""


# Buffered characters that trigger a flush mid-session. Roughly one
# Slack message, so a live agent's log channel stays current instead of
# filling only when the agent exits.
_LOG_FLUSH_CHARS: Final = 3_500


async def _flush_log(
    buffer: list[str],
    channel_id: str,
    slack: Slack,
    *,
    msg_limit: int = 3900,
) -> None:
    """Flush buffered log lines to Slack, splitting to fit ``msg_limit``.

    Splits at line boundaries where it can and WITHIN a line when it
    must: one rendered tool result can exceed the whole per-message
    limit on its own, and appending it whole made Slack reject the send
    rather than deliver a shortened one.
    """
    chunk: list[str] = []
    chunk_len = 0
    for line in buffer:
        for piece in _split_line(line, msg_limit):
            piece_len = len(piece) + 1  # +1 for newline join
            if chunk and chunk_len + piece_len > msg_limit:
                _ = await slack.send(channel_id, "\n".join(chunk))
                chunk = []
                chunk_len = 0
            chunk.append(piece)
            chunk_len += piece_len
    if chunk:
        _ = await slack.send(channel_id, "\n".join(chunk))


def _split_line(line: str, msg_limit: int) -> list[str]:
    """Break one line into pieces that each fit within ``msg_limit``."""
    if len(line) < msg_limit:
        return [line]
    return [line[i : i + msg_limit - 1] for i in range(0, len(line), msg_limit - 1)]


def _make_log_forwarder(
    queue: asyncio.Queue[str | None],
) -> Callable[[RuntimeEvent], None]:
    """Build an observer that renders ``RuntimeEvent`` payloads to log lines."""

    def _fwd(ev: RuntimeEvent) -> None:
        rendered = _render_event(ev)
        if rendered is not None:
            queue.put_nowait(rendered)

    return _fwd


def _render_event(event: RuntimeEvent) -> str | None:
    """Convert a ``RuntimeEvent`` to a log-friendly line, or ``None`` to skip."""
    if isinstance(event, UserMessage):
        text = event.text.strip()
        return f"━━ input ━━\n{text}" if text else None

    if isinstance(event, AssistantMessage):
        text = event.text.strip()
        return text or None

    if isinstance(event, ModelResponseThinking):
        text = event.text.strip()
        if not text:
            return None
        return f"💭 {text}"

    if isinstance(event, ToolLabel):
        return f"  {event.text.strip()}"

    if isinstance(event, ToolResult):
        if event.is_error:
            return f"  ✗ {event.content.strip()}"
        lines: list[str] = []
        if event.diff.strip():
            lines.append(f"```diff\n{event.diff.strip()}\n```")
        if event.hint.strip():
            lines.append(f"  hint: {event.hint.strip()}")
        if event.summary.strip():
            lines.append(f"  → {event.summary.strip()}")
        if not lines and event.content.strip():
            # No length clamp: ``_flush_log`` chunks the buffer at line
            # boundaries under Slack's per-message limit, so an oversize
            # body is split across messages rather than lost.
            lines.append(f"  → {event.content.strip()}")
        return "\n".join(lines) if lines else None

    if isinstance(event, ModelResponseError):
        return f"✗ {type(event.exception).__name__}: {event.exception}"

    # Streaming chunks (ModelResponsePartial) are intentionally skipped --
    # the final AssistantMessage carries the full text and avoids log spam.
    return None


class LogChannelAdapter(Protocol):
    """Minimal adapter surface required by :func:`log_tap`."""

    @property
    def bot_token(self) -> str:
        """Slack bot token used to construct outgoing Slack clients."""
        ...

    async def ensure_log_channel(
        self,
        agent_name: str,
        source_channel: str = "",
    ) -> str | None:
        """Create or find the per-agent log channel and return its ID."""
        ...


async def log_tap(
    events: asyncio.Queue[str | None],
    agent_name: str,
    adapter: LogChannelAdapter,
) -> None:
    """Forward rendered log lines from ``events`` to the agent's log channel.

    Args:
      events: Queue producing rendered lines; ``None`` flushes the buffer.
      agent_name: Owning agent label, used to resolve the log channel.
      adapter: Adapter that creates/finds the per-agent log channel.

    """
    channel_id: str | None = None
    slack: Slack | None = None
    source_channel: str = ""
    buffer: list[str] = []
    while True:
        rendered = await events.get()
        if rendered is None:
            # Terminal sentinel: the producer has ended, so flush what is
            # left and RETURN. Continuing left one tap task alive per
            # agent for the life of the process.
            if buffer and channel_id and slack:
                await _flush_log(buffer, channel_id, slack)
            buffer.clear()
            return
        if not source_channel:
            source_channel = _extract_channel_from_text(rendered)
        if channel_id is None:
            channel_id = await adapter.ensure_log_channel(
                agent_name,
                source_channel=source_channel,
            )
            if channel_id is None:
                continue
            slack = Slack(token=adapter.bot_token)
        buffer.append(rendered)
        # Flush as soon as a message's worth has accumulated. A serviced
        # agent runs for days, so waiting for the terminal sentinel meant
        # its log channel stayed empty for the whole session while the
        # buffer grew without bound.
        if slack is not None and sum(len(x) + 1 for x in buffer) >= _LOG_FLUSH_CHARS:
            await _flush_log(buffer, channel_id, slack)
            buffer.clear()


def parse_slack_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Add shared Slack service flags and delegate to ``parse_agent_args``.

    These flags cover Slack tokens, persona, logging, session, and agent-model
    configuration for the service entry point.

    Args:
      parser: Argparse parser to extend in place.
      argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
      parsed: Tuple of ``(namespace, remaining_args)`` from ``parse_known_args``.

    """
    _ = parser.add_argument(
        "--app-token",
        default="",
        help="Slack app token (xapp-...). Default: $SLACK_APP_TOKEN.",
    )
    _ = parser.add_argument(
        "--bot-token",
        default="",
        help="Slack bot token (xoxb-...). Default: $SLACK_BOT_TOKEN.",
    )
    _ = parser.add_argument(
        "--persona-dir",
        default=str(Path(__file__).resolve().parent.parent / "assets" / "slack"),
        help="Directory of persona .md files.",
    )
    _ = parser.add_argument(
        "--log-prefix",
        default="",
        help="Prefix for log channel names (e.g. 'agent-' -> #agent-sara-log).",
    )
    _ = parser.add_argument(
        "--router-log-channel",
        dest="router_log_channel",
        default="router-log",
        help="Channel name for router decision logs (default: router-log). Created if needed.",
    )
    _ = parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory.",
    )
    _ = parser.add_argument(
        "--session-dir",
        dest="session_dir",
        default=str(data_dir() / "rekursiv-ai" / "sagent" / "slack"),
        help="Directory for session persistence (default: ~/.sagent/slack).",
    )
    _ = parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        default=False,
        help="Resume agents from a previous session.",
    )
    return parse_agent_args(parser, argv)


def _resolve_tokens(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(app_token, bot_token)`` from CLI flags or env, exiting on miss."""
    app = args.app_token or os.environ.get("SLACK_APP_TOKEN", "")
    bot = args.bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
    missing: list[str] = []
    if not app:
        missing.append("app token (--app-token or $SLACK_APP_TOKEN)")
    if not bot:
        missing.append("bot token (--bot-token or $SLACK_BOT_TOKEN)")
    if missing:
        _ = sys.stderr.write(f"Missing: {', '.join(missing)}\n")
        sys.exit(1)
    return cast(tuple[str, str], (app, bot))


async def _run(args: argparse.Namespace) -> None:
    """Wire up the adapter and any resumed agents, then serve until shutdown."""
    if args.cwd:
        os.chdir(args.cwd)

    app_token, bot_token = _resolve_tokens(args)
    provider = build_provider(args.provider, args.auth, account=args.account)
    model = provider.model(args.model)
    model_recipe = ModelRecipe(
        provider=args.provider,
        auth=args.auth,
        model_id=model.spec.tagged_model_id,
        account=args.account,
    )
    compactor = SummaryCompactor() if args.compact else None
    persona_dir = Path(args.persona_dir)

    _ = sys.stderr.write(f"[{args.provider}] {model.spec.tagged_model_id}\n")
    _ = sys.stderr.write(f"[personas] {persona_dir}\n")

    session_root = Path(args.session_dir)
    session_root.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one-time sync mkdir is negligible
    if args.resume:
        candidate = _latest_session_dir(session_root)
        if candidate is None:
            _ = sys.stderr.write("No previous session found.\n")
            sys.exit(1)
        session_dir = candidate
        _ = sys.stderr.write(f"[continue] {session_dir.name}\n")
    else:
        session_dir = _new_session_dir(session_root)

    adapter = SlackAdapter(
        app_token=app_token,
        bot_token=bot_token,
        model=model,
        model_recipe=model_recipe,
        persona_dir=persona_dir,
        session_dir=session_dir,
        compactor=compactor,
        log_prefix=args.log_prefix,
        router_log_channel=args.router_log_channel,
        effort=args.effort,
        max_tool_call_rounds=args.max_tool_call_rounds,
        max_budget_usd=args.max_budget_usd,
    )

    # Resume agents from previous session.
    if args.resume:
        manifest = _load_manifest(session_dir)
        for label, info in manifest.items():
            saved_system = info.get("system", "")
            persona_name = info.get("persona", "default")
            system_text = saved_system or load_persona(persona_dir, persona_name)
            _ = sys.stderr.write(f"[resume] {label} (persona={persona_name})\n")
            await adapter.spawn_agent(label, system_text)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        if stop.is_set():
            os._exit(1)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    adapter_task = asyncio.create_task(adapter.start())
    logger.info("Slack service started")
    await stop.wait()

    # Shutdown.
    for agent in list(agent_registry.values()):
        if isinstance(agent, Agent):
            agent.shutdown()
    _ = adapter_task.cancel()
    logger.info("Shutdown complete")


def main() -> int:
    """Parse args, run the Slack service, and return the process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, remaining = parse_slack_args(parser)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        _ = sys.stderr.write("\n[interrupted]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
