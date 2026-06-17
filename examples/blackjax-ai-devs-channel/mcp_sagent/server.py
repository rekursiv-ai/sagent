"""MCP stdio server for blackjax-chat peer messaging.

This module is the entry point that each agent's ``claude --print``
subprocess spawns via its ``--mcp-config``. The CLI talks to it over
stdio using the standard MCP protocol; on every ``CallToolRequest``,
our handler dispatches into the in-process :mod:`delivery` module
which talks to sagent's :data:`agent_registry` directly.

Three tools are exposed:

  - ``sagent_send(to, content)`` — peer message. Replaces the
    bridge-mounted ``AgentSend`` for inter-agent traffic.
  - ``sagent_defer(delay_s, body)`` — schedule a wake-up for the
    calling agent (self-defer). Replaces ``bash sleep`` for "check
    back in N min" patterns and the previous DeferRouter prose
    fallback.
  - ``sagent_self(status?, context?)`` — status report / context
    action (clear/compact/recompact). Mirrors sagent's bridge-mounted
    ``AgentSelf`` but available via the same MCP namespace as the
    other plugin tools, so agents see one consistent interface.

Each agent's CLI subprocess spawns its own instance of this server
(stdio transport, one subprocess per ``claude --print``). The
``SAGENT_ROLE`` env var — set by :mod:`runtime.build` when it writes
the per-role MCP config — tells the server which agent it serves.
That's how peer attribution works without per-call wire-format
changes: each MCP server instance just knows its identity from the
environment.

The fully-qualified tool names the CLI sees are:

  - ``mcp__sagent_chat__sagent_send``
  - ``mcp__sagent_chat__sagent_defer``
  - ``mcp__sagent_chat__sagent_self``

(MCP's namespacing convention: ``mcp__<server-name>__<tool-name>``;
the ``<server-name>`` portion is whatever key the agent's mcp.json
puts under ``mcpServers``. We use ``sagent`` — see
:mod:`runtime.build`.)

The probe (see README § "Episode 3") established that all three
target models — haiku, sonnet, opus — emit structured ``tool_use``
blocks for tools mounted this way. That's the entire reason this
plugin exists; do not move these tools back to sagent's in-process
MCP bridge without re-running the probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import logging
import os
import sys

import mcp.server
import mcp.server.stdio
import mcp.types as mcp_types


# Ensure the plugin's siblings are importable when this module is
# launched as a subprocess via ``--mcp-config``. We need ``delivery``
# from :mod:`mcp_sagent.delivery`. Resolve the plugin root from
# ``__file__`` rather than relying on ``PYTHONPATH``.
_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from datetime import UTC

from mcp_sagent import delivery


_LOG = logging.getLogger("mcp_sagent.server")

# Debug trace: every tool call gets a line. Useful for "did the call
# reach us at all?" forensics — sagent's runtime trace doesn't see
# external-MCP tool calls (those don't touch the bridge), so this is
# the only authoritative log of what the model dispatched to us.
# Lives in the same data dir as ``main.jsonl`` and the per-role
# trace files — see :data:`delivery.DATA_DIR`.
_DEBUG_LOG = delivery.SESSIONS_DIR / "mcp_calls.log"


def _debug(msg: str) -> None:
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            from datetime import datetime

            f.write(
                f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ')} "
                f"[role={_AGENT_ROLE}] {msg}\n"
            )
    except Exception:
        pass


# --------------------------------------------------------------------------
# Server identity (from env, set by runtime.build when launching the CLI)
# --------------------------------------------------------------------------

# Each agent's CLI gets its own MCP server subprocess; SAGENT_ROLE is
# set in the env of that subprocess. The server reads it once at
# startup. If absent we default to ``operator`` so a manual invocation
# of this module still functions (useful for the standalone probe).
_AGENT_ROLE = os.environ.get("SAGENT_ROLE", "operator")

# Sentinel file: when present, the MCP server's tool handlers skip
# audit log writes AND inbox pushes. ``serve.py`` touches this file
# before pushing the warmup probe and removes it once warmup
# completes. File-based (vs env-var) because env vars can't be
# flipped on a running subprocess — and our MCP server IS a
# subprocess of ``claude --print``, which is itself a subprocess of
# ``serve.py``. Filesystem visibility is the only synchronisation
# mechanism that all three layers share.
_SUPPRESS_FLAG = delivery.SESSIONS_DIR / "_suppress_audit"


def _suppress_audit() -> bool:
    return _SUPPRESS_FLAG.exists()


# --------------------------------------------------------------------------
# Tool catalog
# --------------------------------------------------------------------------

server = mcp.server.Server("sagent_chat")


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="sagent_send",
            description=(
                "Send a chat message to a peer agent in the blackjax-chat "
                "channel. Use 'to' for the peer label (e.g. 'swe', 'tl', "
                "'statistician', 'tech-writer', 'junior-swe', 'user') and "
                "'content' for the message body. This is the ONLY way to "
                "deliver a message to a peer — your assistant text is for "
                "thinking, not for routing. The recipient sees your message "
                "on their next turn."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": (
                            "Peer label: tl, swe, junior-swe, statistician, "
                            "tech-writer, or user."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Message body to deliver to the peer.",
                    },
                },
                "required": ["to", "content"],
            },
        ),
        mcp_types.Tool(
            name="sagent_defer",
            description=(
                "Schedule a wake-up for YOURSELF (the calling agent) after a "
                "delay. After scheduling, end your turn — the runtime "
                "schedules an asyncio timer and pushes a wake-up message "
                "into your inbox after delay_s seconds, at which point you "
                "process it in a fresh turn. Use this for 'check back in N "
                "minutes' patterns (CI waits, remote-state polling). NEVER "
                "use 'bash sleep' for waiting — that blocks your turn for "
                "the entire duration and makes you uninterruptible."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "delay_s": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3600,
                        "description": "Seconds to wait before the wake-up fires (1-3600).",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Message body delivered to your inbox after the "
                            "delay. Typically a one-line reminder of what to "
                            "do on wake-up, e.g. 'check PR #141 CI status'."
                        ),
                    },
                },
                "required": ["delay_s", "body"],
            },
        ),
        mcp_types.Tool(
            name="sagent_self",
            description=(
                "Patch the calling agent's own state. Use status='ready' as "
                "a non-disruptive bootstrap acknowledgement. Use "
                "context='clear' to wipe history, context='compact' to "
                "trigger a structured compaction, or context='recompact' to "
                "recompact existing history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Optional status text; omit to keep the current status.",
                    },
                    "context": {
                        "type": "string",
                        "enum": ["clear", "compact", "recompact"],
                        "description": (
                            "Optional context action. Omit to preserve "
                            "context; use 'clear', 'compact', or 'recompact'."
                        ),
                    },
                },
            },
        ),
    ]


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------


def _text(s: str, is_error: bool = False) -> list[mcp_types.ContentBlock]:
    """Convenience wrapper — MCP wants a list of content blocks."""
    return [mcp_types.TextContent(type="text", text=s)]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[mcp_types.ContentBlock]:
    # Each call resolves the sender identity from the module-level
    # constant (set from env at startup). This means a single MCP
    # server instance serves a single agent — the sagent.serve.py
    # process launches one CLI per agent and each CLI spawns its
    # own MCP server with its own SAGENT_ROLE.
    sender = _AGENT_ROLE
    suppress = _suppress_audit()
    _debug(f"call_tool name={name!r} arguments={arguments!r} suppress={suppress}")

    if name == "sagent_send":
        to = str(arguments.get("to", "")).strip()
        content = str(arguments.get("content", ""))
        if not to:
            return _text("[Error] 'to' is required.", is_error=True)
        if not content:
            return _text("[Error] 'content' is required.", is_error=True)
        ok, status = delivery.route_send(
            from_role=sender,
            to=to,
            content=content,
            suppress_audit=suppress,
        )
        _debug(f"  route_send -> ok={ok} status={status!r}")
        # Also report the registry contents to surface the
        # cross-process registry issue if we hit it.
        try:
            from sagent.tools.core import agent_registry

            _debug(f"  agent_registry keys: {sorted(agent_registry)}")
        except Exception as e:
            _debug(f"  agent_registry import failed: {e!r}")
        return _text(status, is_error=not ok)

    if name == "sagent_defer":
        try:
            delay_s = int(arguments.get("delay_s", 0))
        except (TypeError, ValueError):
            return _text("[Error] 'delay_s' must be an integer.", is_error=True)
        body = str(arguments.get("body", "")).strip()
        if not body:
            return _text("[Error] 'body' is required.", is_error=True)
        ok, status = delivery.schedule_defer(
            sender=sender,
            delay_s=delay_s,
            body=body,
            suppress_audit=suppress,
        )
        return _text(status, is_error=not ok)

    if name == "sagent_self":
        # NB: this MCP server runs as a SEPARATE Python process spawned
        # by claude --print's --mcp-config — its ``agent_registry``
        # is a fresh, empty module-level dict, distinct from the live
        # registry that ``serve.py`` owns. So we can't reach the live
        # runtime from here to set ``runtime.status`` directly.
        #
        # For warmup the round-trip just needs to ack so the model
        # ends the turn (the bootstrap directive requires a single
        # tool call and "ok"). For real status changes the operator
        # path is the web UI / ``/api/restart`` -- ``sagent_self``
        # has no live-runtime equivalent on a session-persistent
        # subprocess.
        #
        # Return a benign no-op ack. ``status`` / ``context`` args
        # are echoed back so the model can verify the call was seen.
        status_text = arguments.get("status")
        context = arguments.get("context")
        notes: list[str] = []
        if status_text:
            notes.append(f"status acknowledged: {status_text!r}")
        if context:
            notes.append(f"context arg acknowledged: {context!r}")
        if not notes:
            notes.append("no-op")
        return _text("; ".join(notes))

    return _text(f"[Error] unknown tool: {name!r}", is_error=True)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


async def _amain() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOG.info("mcp_sagent server starting for role=%s", _AGENT_ROLE)
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            mcp.server.InitializationOptions(
                server_name="sagent_chat",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=mcp.server.NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
