#!/bin/sh
# ruff: noqa: EXE003, D300  -- Polyglot: #!/bin/sh + triple-single-quotes are intentional.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --no-sync python3 "$0" "$@"

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

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

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

import httpx

from sagent.agent import Agent
from sagent.agent.agent import QUIT_SENTINEL
from sagent.bin.cli import (
    DEFAULT_TOOLS,
    parse_agent_args,
    resolve_tools,
)
from sagent.compactor import SummaryCompactor
from sagent.custom_types import Message, ModelSpec, is_message
from sagent.lib import apikey
from sagent.lib.json import MutableJSON
from sagent.providers import build_provider
from sagent.tools.core import agent_registry
from sagent.tools.slack import Slack


if TYPE_CHECKING:
    from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest

    from sagent.custom_types import Model


logger = logging.getLogger(__name__)

_AGENT_TOOL_NAMES = [
    name for name in DEFAULT_TOOLS if name not in ("AgentSpawn", "AgentSend")
]


# -- Slack helpers ---------------------------------------------------------


def _extract_event(payload: MutableJSON | None) -> MutableJSON | None:
    if not isinstance(payload, dict):
        return None
    ev = payload.get("event")
    if not isinstance(ev, dict):
        return None
    return cast(MutableJSON, ev)


def _strip_mention(text: str, self_user_id: str) -> str:
    if not self_user_id:
        return text.strip()
    return text.replace(f"<@{self_user_id}>", "").strip()


def _extract_agent_mention(text: str) -> str | None:
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


# -- Session persistence ---------------------------------------------------


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / "manifest.json"


def _new_session_dir(root: Path) -> Path:
    name = _time.strftime("%Y%m%d_%H%M%S")
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_session_dir(root: Path) -> Path | None:
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
    session_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(session_dir).write_text(
        json.dumps(agents, indent=2),
        encoding="utf-8",
    )


def _load_manifest(session_dir: Path) -> dict[str, dict[str, str]]:
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


# -- Persona loading -------------------------------------------------------


def load_persona(persona_dir: Path, persona_name: str) -> str:
    path = persona_dir / f"{persona_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    default = persona_dir / "default.md"
    if default.exists():
        return default.read_text(encoding="utf-8")
    return f"You are {persona_name}."


def _list_personas(persona_dir: Path) -> list[str]:
    if not persona_dir.is_dir():
        return []
    return sorted(p.stem for p in persona_dir.glob("*.md"))


# -- Agent-aware Slack tool ------------------------------------------------


class _AgentSlack(Slack):
    """Slack tool that advertises active peer agents in its prompt."""

    @override
    def prompt(self) -> str:
        others = sorted(k for k in agent_registry if k != self._username)
        if not others:
            return ""
        names = ", ".join(others)
        return (
            f"Active agents you can message by @-mentioning at the start"
            f" of your Slack message: {names}"
        )


# -- Slack adapter ---------------------------------------------------------


class SlackAdapter:
    """Socket Mode listener with deterministic routing and agent lifecycle."""

    def __init__(
        self,
        *,
        app_token: str,
        bot_token: str,
        model: Model,
        model_spec: ModelSpec,
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
        self._model_spec = model_spec
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
        self._tasks: list[asyncio.Task[None]] = []
        self._bg_tasks: set[asyncio.Task[object]] = set()
        self._user_names: dict[str, str] = {}
        # Cache of agent-sent messages: (channel, ts) → (agent, text, thread_ts).
        self._sent_messages: dict[tuple[str, str], tuple[str, str, str]] = {}

    @property
    def bot_user_id(self) -> str:
        return self._self_user_id

    @property
    def bot_token(self) -> str:
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
        if is_message(result):
            logger.warning("Could not create/find #%s: %s", name, result.content)
            self._router_log_channel = ""
            return
        result_str = str(result)
        if result_str.startswith("id="):
            self._router_log_channel = result_str[3:]
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
            self._sent_messages[(channel, ts)] = (sender_agent, clean[:200], thread_ts)

        # 1. Log channel → owning agent (skip self-routing).
        if channel in self._log_channel_owners:
            owner = self._log_channel_owners[channel]
            if owner in agent_registry and owner != sender_agent:
                self._log_route(f"[{user}->{owner}] log-channel {clean[:200]}")
                agent_registry[owner].inbox.put(formatted)
            return

        # 2. Explicit agent name at start of message.
        target = _extract_agent_mention(clean)
        if target and target in agent_registry and target != sender_agent:
            thread_key = (channel, thread_ts)
            if thread_key not in self._thread_owners:
                self._thread_owners[thread_key] = target
            self._log_route(f"[{user}->{target}] mention {clean[:200]}")
            agent_registry[target].inbox.put(formatted)
            return

        # 3. Thread continuation.
        thread_key = (channel, thread_ts)
        if thread_key in self._thread_owners:
            owner = self._thread_owners[thread_key]
            if owner in agent_registry and owner != sender_agent:
                self._log_route(f"[{user}->{owner}] thread {clean[:200]}")
                agent_registry[owner].inbox.put(formatted)
                return

        # Agent messages that don't match steps 1-3: drop.
        if sender_agent:
            self._log_route(f"[{sender_agent}] dropped {clean[:200]}")
            return

        # 4. Try as command (human messages only).
        if await self._try_command(clean, channel, thread_ts):
            self._log_route(f"[{user}] command {clean[:200]}")
            return

        # 5. Single agent default.
        agents = list(agent_registry)
        if len(agents) == 1:
            self._thread_owners[(channel, thread_ts)] = agents[0]
            self._log_route(f"[{user}->{agents[0]}] default {clean[:200]}")
            agent_registry[agents[0]].inbox.put(formatted)
            return

        # 6. Ambiguous or no agents.
        self._log_route(f"[{user}] unrouted {clean[:200]}")
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
        agent_registry[agent_name].inbox.put(formatted)

    # -- Command handling --------------------------------------------------

    async def _try_command(
        self,
        text: str,
        channel: str,
        thread_ts: str,
    ) -> bool:
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
        base_tools = resolve_tools(_AGENT_TOOL_NAMES)
        tools = [*base_tools, _AgentSlack(token=self._bot_token, username=label)]
        child = Agent(
            name=label,
            description=f"Persistent agent: {label}",
            model=self._model,
            model_spec=self._model_spec,
            system=system_text,
            tools=tools,
            compactor=self._compactor,
            effort=self._effort,
            max_tool_call_rounds=self._max_tool_call_rounds,
            max_budget_usd=self._max_budget_usd,
            session_dir=self._session_dir / label,
            persistent_retry=True,
        )
        child._persistent = True  # noqa: SLF001 -- cross-layer flag on child agent
        agent_registry[label] = child
        self._active_agents[label] = {"persona": "custom", "system": system_text}
        _save_manifest(self._session_dir, self._active_agents)

        events: asyncio.Queue[Message | None] = asyncio.Queue()

        async def _run_child(
            c: Agent = child,
            e: asyncio.Queue[Message | None] = events,
        ) -> None:
            try:
                async for event in c.run_continuous():
                    e.put_nowait(event)
            finally:
                agent_registry.pop(c.name, None)

        self._tasks.append(asyncio.create_task(_run_child()))
        self._tasks.append(asyncio.create_task(log_tap(events, label, self)))

    def stop_agent(self, label: str) -> None:
        agent = agent_registry.get(label)
        if agent:
            agent.inbox.put(QUIT_SENTINEL)
        self._active_agents.pop(label, None)
        _save_manifest(self._session_dir, self._active_agents)

    async def _reply(self, channel: str, thread_ts: str, text: str) -> None:
        await self._slack.send(channel, text, thread_ts)

    # -- Log channel management --------------------------------------------

    async def ensure_log_channel(
        self,
        agent_name: str,
        source_channel: str = "",
    ) -> str | None:
        ch_name = f"{self._log_prefix}{agent_name}-log".lower()
        if ch_name in self._log_channels:
            return self._log_channels[ch_name]
        slack = Slack(token=self._bot_token)
        result = await slack.create_channel(ch_name)
        if is_message(result):
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
        result_str = str(result)
        if result_str.startswith("id="):
            cid = result_str[3:]
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
        members = await self._list_members(source_channel)
        for uid in members:
            if uid == self._self_user_id:
                continue
            await self._invite(target_channel, uid)

    async def _list_members(self, channel: str) -> list[str]:
        url = "https://slack.com/api/conversations.members"
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        params = {"channel": channel, "limit": 1000}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, headers=headers, params=params)
            if not r.is_success:
                return []
            body = r.json()
        if not body.get("ok"):
            return []
        members: list[object] = body.get("members") or []
        return [str(m) for m in members]

    async def _invite(self, channel: str, user: str) -> None:
        url = "https://slack.com/api/conversations.invite"
        headers = {
            "Authorization": f"Bearer {self._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                url,
                headers=headers,
                content=json.dumps({"channel": channel, "users": user}),
            )

    async def _find_channel(self, name: str) -> str | None:
        slack = Slack(token=self._bot_token)
        result = await slack._list_channels(1000)  # noqa: SLF001 -- no public channel-list API on Slack wrapper
        if is_message(result):
            return None
        for line in str(result).splitlines():
            if f"#{name}" in line:
                return line.split()[0]
        return None


# -- Log tap ---------------------------------------------------------------


def _extract_channel_from_text(text: str) -> str:
    for part in text.split():
        if part.startswith("channel="):
            return part[8:].rstrip("]")
    return ""


_SLACK_MSG_LIMIT = 3900


async def _flush_log(
    buffer: list[str],
    channel_id: str,
    slack: Slack,
) -> None:
    """Flush buffered log lines to Slack, splitting at line boundaries."""
    chunk: list[str] = []
    chunk_len = 0
    for line in buffer:
        line_len = len(line) + 1  # +1 for newline join
        if chunk and chunk_len + line_len > _SLACK_MSG_LIMIT:
            await slack.send(channel_id, "\n".join(chunk))
            chunk = []
            chunk_len = 0
        chunk.append(line)
        chunk_len += line_len
    if chunk:
        await slack.send(channel_id, "\n".join(chunk))


def _render_event(event: Message) -> str | None:
    """Convert an agent event to log-friendly text, or None to skip."""
    desc = event.descriptor

    if desc == "text/plain":
        text = str(event.content).strip()
        return text or None

    if desc == "text/x-error":
        return f"✗ {str(event.content).strip()}"

    if desc in ("text/x-user-injected", "text/x-signal-user-input"):
        text = str(event.content).strip()
        return f"━━ input ━━\n{text}" if text else None

    if desc == "text/x-tool-label":
        return f"  {str(event.content).strip()}"

    if desc == "multipart/x-tool-result":
        parts = cast(tuple[Message, ...], event.content)
        lines: list[str] = []
        for p in parts:
            if p.descriptor == "text/x-error":
                lines.append(f"  ✗ {str(p.content).strip()}")
            elif p.descriptor == "text/x-hint-tool-use-nudge" and p.content:
                lines.append(f"  hint: {str(p.content).strip()}")
            elif p.descriptor == "text/x-diff" and p.content:
                diff = str(p.content).strip()
                if diff:
                    lines.append(f"```diff\n{diff}\n```")
            elif p.descriptor.startswith("text/") and p.content:
                text = str(p.content).strip()
                if text:
                    if len(text) > 2000:
                        text = text[:2000] + "…"
                    lines.append(f"  → {text}")
        return "\n".join(lines) if lines else None

    if desc == "text/x-interrupted":
        return "[interrupted]"

    if desc == "multipart/x-child-event":
        parts = cast(tuple[Message, ...], event.content)
        if len(parts) < 2:
            return None
        label = str(parts[0].content)
        inner = parts[1]
        inner_rendered = _render_event(inner)
        if inner_rendered:
            pfx = f"[{label}]"
            return "\n".join(f"  {pfx} {ln}" for ln in inner_rendered.splitlines())
        return None

    if desc == "text/x-thinking":
        text = str(event.content).strip()
        if not text:
            return None
        if len(text) > 2000:
            text = text[:2000] + "…"
        return f"💭 {text}"

    # Skip status changes and done signals.
    if desc in (
        "text/x-signal-status-changed",
        "application/x-done",
    ):
        return None

    # Fallback: any other text/* descriptor.
    if desc.startswith("text/"):
        text = str(event.content).strip()
        return text or None

    return None


async def log_tap(
    events: asyncio.Queue[Message | None],
    agent_name: str,
    adapter: SlackAdapter,
) -> None:
    channel_id: str | None = None
    slack: Slack | None = None
    source_channel: str = ""
    buffer: list[str] = []
    while True:
        event = await events.get()
        if event is None:
            if buffer and channel_id and slack:
                await _flush_log(buffer, channel_id, slack)
            buffer.clear()
            continue
        rendered = _render_event(event)
        if rendered is None:
            continue
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


# -- Wiring ----------------------------------------------------------------


def parse_slack_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Add shared Slack service flags and delegate to ``parse_agent_args``.

    These flags cover Slack tokens, persona, logging, session, and agent-model
    configuration for the service entry point.
    """
    parser.add_argument(
        "--app-token",
        default="",
        help="Slack app token (xapp-...). Default: $SLACK_APP_TOKEN.",
    )
    parser.add_argument(
        "--bot-token",
        default="",
        help="Slack bot token (xoxb-...). Default: $SLACK_BOT_TOKEN.",
    )
    parser.add_argument(
        "--persona-dir",
        default=str(Path(__file__).resolve().parent.parent / "assets" / "slack"),
        help="Directory of persona .md files.",
    )
    parser.add_argument(
        "--log-prefix",
        default="",
        help="Prefix for log channel names (e.g. 'agent-' -> #agent-sara-log).",
    )
    parser.add_argument(
        "--router-log-channel",
        dest="router_log_channel",
        default="router-log",
        help="Channel name for router decision logs (default: router-log). Created if needed.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory.",
    )
    parser.add_argument(
        "--session-dir",
        dest="session_dir",
        default=str(Path.home() / ".sagent" / "slack"),
        help="Directory for session persistence (default: ~/.sagent/slack).",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        default=False,
        help="Resume agents from a previous session.",
    )
    return parse_agent_args(parser, argv)


def _resolve_tokens(args: argparse.Namespace) -> tuple[str, str]:
    app = args.app_token or apikey.get("SLACK_APP_TOKEN")
    bot = args.bot_token or apikey.get("SLACK_BOT_TOKEN")
    missing: list[str] = []
    if not app:
        missing.append("app token (--app-token or $SLACK_APP_TOKEN)")
    if not bot:
        missing.append("bot token (--bot-token or $SLACK_BOT_TOKEN)")
    if missing:
        sys.stderr.write(f"Missing: {', '.join(missing)}\n")
        sys.exit(1)
    return app, bot


async def _run(args: argparse.Namespace) -> None:
    if args.cwd:
        os.chdir(args.cwd)

    app_token, bot_token = _resolve_tokens(args)
    provider = build_provider(args.provider, args.auth, account=args.account)
    model = provider.model(args.model)
    model_spec = ModelSpec(
        provider=args.provider,
        auth=args.auth,
        model_id=model.model_id,
        account=args.account,
    )
    compactor = SummaryCompactor() if args.compact else None
    persona_dir = Path(args.persona_dir)

    sys.stderr.write(f"[{args.provider}] {model.model_id}\n")
    sys.stderr.write(f"[personas] {persona_dir}\n")

    session_root = Path(args.session_dir)
    session_root.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one-time sync mkdir is negligible
    if args.resume:
        session_dir = _latest_session_dir(session_root)
        if session_dir is None:
            sys.stderr.write("No previous session found.\n")
            sys.exit(1)
        sys.stderr.write(f"[continue] {session_dir.name}\n")
    else:
        session_dir = _new_session_dir(session_root)

    adapter = SlackAdapter(
        app_token=app_token,
        bot_token=bot_token,
        model=model,
        model_spec=model_spec,
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
            sys.stderr.write(f"[resume] {label} (persona={persona_name})\n")
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
        agent.inbox.put(QUIT_SENTINEL)
    adapter_task.cancel()
    logger.info("Shutdown complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Slack service -- deterministic multi-agent routing.",
    )
    args, remaining = parse_slack_args(parser)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        sys.stderr.write("\n[interrupted]\n")


if __name__ == "__main__":
    main()
