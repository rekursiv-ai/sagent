"""Slack integration: send/read messages for interactive participation.

Uses Slack's Web API with a bot user token (``xoxb-...``). Set up a
bot user at https://api.slack.com/apps, grant scopes (``chat:write``,
``chat:write.customize``, ``channels:history``, ``channels:read``,
``channels:manage``, ``users:read``, ``groups:history`` for private
channels, ``im:history`` for DMs), and install to your workspace.

Supported operations:

- ``send`` -- post a message to a channel or DM (threads supported)
- ``list_channels`` -- enumerate channels the bot is in
- ``list_messages`` -- recent messages from a channel
- ``read_thread`` -- all replies under a parent message ts
- ``list_users`` -- enumerate workspace users
- ``create_channel`` -- create a new channel
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import asyncio
import json
import logging

from sagent.custom_types import Message, TextMessage, is_message
from sagent.lib.json import JSON, MutableJSON, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.lib.web.fetch import FetchError, fetch
from sagent.tools.core import load_tool_description


logger = logging.getLogger(__name__)


_API_BASE = "https://slack.com/api"
_DEFAULT_TIMEOUT = 30.0


_OPERATIONS = (
    "send",
    "list_channels",
    "list_messages",
    "read_thread",
    "list_users",
    "create_channel",
)


async def _slack_call(
    method: str,
    params: Mapping[str, str | int],
    token: str,
    post: bool = False,
) -> MutableJSON | Message:
    """Call a Slack Web API method. Returns parsed JSON body.

    POST methods use a JSON body + ``Authorization: Bearer``; GET
    methods send params in the query string. Slack returns
    ``{"ok": true, ...}`` or ``{"ok": false, "error": "..."}``.
    """
    url = f"{_API_BASE}/{method}"
    headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
    try:
        if post:
            headers["Content-Type"] = "application/json; charset=utf-8"
            raw = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty can't narrow to_thread through overloads
                bytes,
                await asyncio.to_thread(
                    fetch,
                    url=url,
                    method="POST",
                    json=dict(params),
                    headers=headers,
                    timeout_sec=_DEFAULT_TIMEOUT,
                ),
            )
        else:
            raw = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty can't narrow to_thread through overloads
                bytes,
                await asyncio.to_thread(
                    fetch,
                    url=url,
                    params=dict(params),
                    headers=headers,
                    timeout_sec=_DEFAULT_TIMEOUT,
                ),
            )
    except FetchError as e:
        return TextMessage(
            f"Slack HTTP {e.status}: {e.body[:200].decode(errors='replace')}",
            "text/x-error",
        )
    body = cast(MutableJSON, json.loads(raw))
    if not body.get("ok"):
        return TextMessage(
            f"Slack API {method} failed: {body.get('error', 'unknown')}",
            "text/x-error",
        )
    return body


class Slack:
    """Tool: Slack interaction via Web API.

    Args:
      token: Bot user token (``xoxb-...``).
      username: Display name override for ``chat.postMessage``
        (requires ``chat:write.customize`` scope).
      icon_url: Avatar URL override for ``chat.postMessage``.

    """

    name: str = "Slack"
    tool_id: str = "application/x-tool-slack"
    description: str = load_tool_description("Slack")
    supports_microcompaction: bool = False
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(_OPERATIONS)},
                "channel": {"type": "string"},
                "channel_name": {"type": "string"},
                "text": {"type": "string"},
                "thread_ts": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["operation"],
        }
    )

    def __init__(
        self,
        *,
        token: str,
        username: str = "",
        icon_url: str = "",
    ) -> None:
        self._token = token
        self._username = username
        self._icon_url = icon_url

    def summary(self, msg: Message) -> str:
        """Return a short label for this Slack operation.

        Args:
          msg: Tool call message.

        Returns:
          label: "Slack <operation>:<channel>".

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        channel = str(directive.get("channel", ""))
        suffix = f":{channel}" if channel else ""
        return f"Slack {operation}{suffix}"

    def prompt(self) -> str:
        """Return per-request system prompt text.

        Returns:
          prompt: Always empty for this tool.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Dispatch the requested Slack operation.

        Args:
          msg: Tool call message with ``operation`` and operation-specific fields.

        Returns:
          result: Operation result or error message.

        """
        directive = get_directive(msg)
        operation = str(directive.get("operation", ""))
        result = await self._dispatch(
            operation=operation,
            channel=str(directive.get("channel", "")),
            channel_name=str(directive.get("channel_name", "")),
            text=str(directive.get("text", "")),
            thread_ts=str(directive.get("thread_ts", "")),
            limit=int_val(directive.get("limit"), 25),
        )
        if is_message(result):
            return result
        return TextMessage(result, "text/plain")

    async def _dispatch(
        self,
        *,
        operation: str,
        channel: str,
        channel_name: str,
        text: str,
        thread_ts: str,
        limit: int,
    ) -> str | Message:
        if operation == "send":
            return await self._send(channel, text=text, thread_ts=thread_ts)
        if operation == "list_channels":
            return await self._list_channels(limit)
        if operation == "list_messages":
            return await self._list_messages(channel, limit=limit)
        if operation == "read_thread":
            return await self._read_thread(channel, thread_ts=thread_ts, limit=limit)
        if operation == "list_users":
            return await self._list_users(limit)
        if operation == "create_channel":
            return await self.create_channel(channel_name)
        return TextMessage(f"Unknown operation: {operation}", "text/x-error")

    async def _send(
        self,
        channel: str,
        text: str,
        thread_ts: str,
    ) -> str | Message:
        if not channel or not text:
            return TextMessage("'channel' and 'text' required.", "text/x-error")
        params: dict[str, str] = {"channel": channel, "text": text}
        if thread_ts:
            params["thread_ts"] = thread_ts
        if self._username:
            params["username"] = self._username
        if self._icon_url:
            params["icon_url"] = self._icon_url
        body = await _slack_call(
            "chat.postMessage",
            params=params,
            token=self._token,
            post=True,
        )
        if is_message(body):
            return body
        sender = self._username or "bot"
        logger.info("[%s->%s] %s", sender, channel, text[:200])
        return f"Sent. ts={body.get('ts')} channel={body.get('channel')}"

    async def _list_channels(self, limit: int) -> str | Message:
        params = {
            "limit": max(1, min(1000, limit)),
            "exclude_archived": "true",
        }
        body = await _slack_call("conversations.list", params=params, token=self._token)
        if is_message(body):
            return body
        channels = cast(list[MutableJSON], body.get("channels") or [])
        if not channels:
            return "(no channels)"
        return "\n".join(
            f"{c.get('id')}  #{c.get('name')}  (members={c.get('num_members', '?')})"
            for c in channels
        )

    async def _list_messages(
        self,
        channel: str,
        limit: int,
    ) -> str | Message:
        if not channel:
            return TextMessage("'channel' required.", "text/x-error")
        params = {"channel": channel, "limit": max(1, min(200, limit))}
        body = await _slack_call(
            "conversations.history", params=params, token=self._token
        )
        if is_message(body):
            return body
        messages = cast(list[MutableJSON], body.get("messages") or [])
        logger.info("[list_messages] channel=%s count=%d", channel, len(messages))
        return _render_messages(messages)

    async def _read_thread(
        self,
        channel: str,
        thread_ts: str,
        limit: int,
    ) -> str | Message:
        if not channel or not thread_ts:
            return TextMessage("'channel' and 'thread_ts' required.", "text/x-error")
        params = {
            "channel": channel,
            "ts": thread_ts,
            "limit": max(1, min(200, limit)),
        }
        body = await _slack_call(
            "conversations.replies", params=params, token=self._token
        )
        if is_message(body):
            return body
        messages = cast(list[MutableJSON], body.get("messages") or [])
        logger.info(
            "[read_thread] channel=%s thread=%s count=%d",
            channel,
            thread_ts,
            len(messages),
        )
        return _render_messages(messages)

    async def _list_users(self, limit: int) -> str | Message:
        params = {"limit": max(1, min(1000, limit))}
        body = await _slack_call("users.list", params=params, token=self._token)
        if is_message(body):
            return body
        members = cast(list[MutableJSON], body.get("members") or [])
        if not members:
            return "(no users)"
        return "\n".join(
            f"{m.get('id')}  @{m.get('name')}  ({m.get('real_name', '')})"
            for m in members
            if not m.get("deleted")
        )

    async def create_channel(self, channel_name: str) -> str | Message:
        """Create a Slack channel.

        Args:
          channel_name: Name for the new channel.

        Returns:
          result: ``"id=C..."`` on success, or error message.

        """
        if not channel_name:
            return TextMessage("'channel_name' required.", "text/x-error")
        body = await _slack_call(
            "conversations.create",
            params={"name": channel_name},
            token=self._token,
            post=True,
        )
        if is_message(body):
            return body
        ch = cast(MutableJSON, body.get("channel") or {})
        return f"id={ch.get('id')}"

    async def send(
        self,
        channel: str,
        text: str,
        thread_ts: str = "",
    ) -> str | Message:
        """Post a message to a Slack channel.

        Args:
          channel: Channel ID to post to.
          text: Message text.
          thread_ts: Parent message timestamp for threading.

        Returns:
          result: Confirmation string or error message.

        """
        return await self._send(channel, text=text, thread_ts=thread_ts)


def _render_messages(messages: list[MutableJSON]) -> str:
    if not messages:
        return "(no messages)"
    lines: list[str] = []
    for m in messages:
        user = m.get("username") or m.get("user") or m.get("bot_id") or "?"
        ts = m.get("ts", "?")
        text = m.get("text", "")
        lines.append(f"[{ts}] <{user}> {text}")
        reactions = cast(list[MutableJSON], m.get("reactions") or [])
        if reactions:
            parts = [f":{r.get('name', '?')}:x{r.get('count', 0)}" for r in reactions]
            lines.append(f"  reactions: {' '.join(parts)}")
    return "\n".join(lines)
