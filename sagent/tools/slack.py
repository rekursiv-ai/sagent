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
from typing import Final, cast

import asyncio
import json
import logging

from wesearch.errors import FetchError
from wesearch.fetch import RequestParams, fetch

from sagent.lib.custom_json import JSON, MutableJSON, int_val, json_freeze
from sagent.tools.core import load_tool_description
from sagent.types.runtime import ToolResult


logger = logging.getLogger(__name__)

_API_BASE: Final = "https://slack.com/api"
_DEFAULT_TIMEOUT = 30.0  # config-globals: ignore -- request timeout dial, retunable

_OPERATIONS: Final = (
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
) -> MutableJSON | ToolResult:
    """Call a Slack Web API method and parse the JSON response.

    POST methods use a JSON body + ``Authorization: Bearer``; GET
    methods send params in the query string. Slack returns
    ``{"ok": true, ...}`` or ``{"ok": false, "error": "..."}``.

    Args:
      method: Slack Web API method (e.g. ``chat.postMessage``).
      params: Request parameters (body for POST, query for GET).
      token: Bot user token (``xoxb-...``).
      post: When True, send as ``POST`` with JSON body.

    Returns:
      body: Parsed JSON body on success, or a ``ToolResult`` error.

    """
    url = f"{_API_BASE}/{method}"
    headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
    try:
        if post:
            headers["Content-Type"] = "application/json; charset=utf-8"
            raw = await asyncio.to_thread(
                fetch,
                url=url,
                request=RequestParams(
                    method="POST",
                    json=dict(params),
                    headers=headers,
                    timeout_sec=_DEFAULT_TIMEOUT,
                ),
            )
        else:
            raw = await asyncio.to_thread(
                fetch,
                url=url,
                request=RequestParams(
                    params=dict(params),
                    headers=headers,
                    timeout_sec=_DEFAULT_TIMEOUT,
                ),
            )
    except FetchError as e:
        return ToolResult(
            call_id="",
            content=(f"Slack HTTP {e.status}: {e.body[:200].decode(errors='replace')}"),
            is_error=True,
        )
    body = cast(MutableJSON, json.loads(raw[0]))
    if not body.get("ok"):
        return ToolResult(
            call_id="",
            content=f"Slack API {method} failed: {body.get('error', 'unknown')}",
            is_error=True,
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
    clearable_results: bool = False
    description: str = load_tool_description("Slack")
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

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Tool arguments.

        Returns:
          label: Human-readable summary string.

        """
        operation = str(args.get("operation", ""))
        channel = str(args.get("channel", ""))
        suffix = f":{channel}" if channel else ""
        return f"Slack {operation}{suffix}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for Slack.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network call, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Dispatch the requested Slack operation.

        Args:
          args: Tool arguments containing ``operation`` and op-specific fields.

        Returns:
          result: Operation output or an error message.

        """
        operation = str(args.get("operation", ""))
        result = await self._dispatch(
            operation=operation,
            channel=str(args.get("channel", "")),
            channel_name=str(args.get("channel_name", "")),
            text=str(args.get("text", "")),
            thread_ts=str(args.get("thread_ts", "")),
            limit=int_val(args.get("limit"), 25),
        )
        if isinstance(result, ToolResult):
            return result
        return ToolResult(call_id="", content=result)

    async def _dispatch(
        self,
        *,
        operation: str,
        channel: str,
        channel_name: str,
        text: str,
        thread_ts: str,
        limit: int,
    ) -> str | ToolResult:
        """Route ``operation`` to the matching private helper."""
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
        return ToolResult(
            call_id="",
            content=f"Unknown operation: {operation}",
            is_error=True,
        )

    async def _send(
        self,
        channel: str,
        text: str,
        thread_ts: str,
    ) -> str | ToolResult:
        """Post one message to ``channel`` via ``chat.postMessage``."""
        if not channel or not text:
            return ToolResult(
                call_id="",
                content="'channel' and 'text' required.",
                is_error=True,
            )
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
        if isinstance(body, ToolResult):
            return body
        sender = self._username or "bot"
        logger.info("[%s->%s] %s", sender, channel, text[:200])
        return f"Sent. ts={body.get('ts')} channel={body.get('channel')}"

    async def _list_channels(self, limit: int) -> str | ToolResult:
        """Enumerate non-archived channels via ``conversations.list``."""
        params = {
            "limit": max(1, min(1000, limit)),
            "exclude_archived": "true",
        }
        body = await _slack_call("conversations.list", params=params, token=self._token)
        if isinstance(body, ToolResult):
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
    ) -> str | ToolResult:
        """Render the last ``limit`` messages via ``conversations.history``."""
        if not channel:
            return ToolResult(
                call_id="",
                content="'channel' required.",
                is_error=True,
            )
        params = {"channel": channel, "limit": max(1, min(200, limit))}
        body = await _slack_call(
            "conversations.history", params=params, token=self._token
        )
        if isinstance(body, ToolResult):
            return body
        messages = cast(list[MutableJSON], body.get("messages") or [])
        logger.info("[list_messages] channel=%s count=%d", channel, len(messages))
        return _render_messages(messages)

    async def _read_thread(
        self,
        channel: str,
        thread_ts: str,
        limit: int,
    ) -> str | ToolResult:
        """Render replies for ``thread_ts`` via ``conversations.replies``."""
        if not channel or not thread_ts:
            return ToolResult(
                call_id="",
                content="'channel' and 'thread_ts' required.",
                is_error=True,
            )
        params = {
            "channel": channel,
            "ts": thread_ts,
            "limit": max(1, min(200, limit)),
        }
        body = await _slack_call(
            "conversations.replies", params=params, token=self._token
        )
        if isinstance(body, ToolResult):
            return body
        messages = cast(list[MutableJSON], body.get("messages") or [])
        logger.info(
            "[read_thread] channel=%s thread=%s count=%d",
            channel,
            thread_ts,
            len(messages),
        )
        return _render_messages(messages)

    async def _list_users(self, limit: int) -> str | ToolResult:
        """Enumerate non-deleted users via ``users.list``."""
        params = {"limit": max(1, min(1000, limit))}
        body = await _slack_call("users.list", params=params, token=self._token)
        if isinstance(body, ToolResult):
            return body
        members = cast(list[MutableJSON], body.get("members") or [])
        if not members:
            return "(no users)"
        return "\n".join(
            f"{m.get('id')}  @{m.get('name')}  ({m.get('real_name', '')})"
            for m in members
            if not m.get("deleted")
        )

    async def create_channel(self, channel_name: str) -> str | ToolResult:
        """Create a Slack channel.

        Args:
          channel_name: Channel name to create (no ``#`` prefix).

        Returns:
          result: ``id=<channel-id>`` confirmation, or a ``ToolResult`` error.

        """
        if not channel_name:
            return ToolResult(
                call_id="",
                content="'channel_name' required.",
                is_error=True,
            )
        body = await _slack_call(
            "conversations.create",
            params={"name": channel_name},
            token=self._token,
            post=True,
        )
        if isinstance(body, ToolResult):
            return body
        ch = cast(MutableJSON, body.get("channel") or {})
        return f"id={ch.get('id')}"

    async def send(
        self,
        channel: str,
        text: str,
        thread_ts: str = "",
    ) -> str | ToolResult:
        """Post a message to a Slack channel.

        Args:
          channel: Channel id or DM id.
          text: Message body to post.
          thread_ts: Optional parent ``ts`` to thread the reply under.

        Returns:
          result: ``Sent. ts=... channel=...`` line, or a ``ToolResult`` error.

        """
        return await self._send(channel, text=text, thread_ts=thread_ts)


def _render_messages(messages: list[MutableJSON]) -> str:
    """Render messages as ``[ts] <user> text`` lines with reaction tails."""
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
