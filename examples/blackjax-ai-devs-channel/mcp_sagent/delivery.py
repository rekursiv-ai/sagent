"""Peer-message delivery + audit-log writes for blackjax-chat.

The MCP server runs as a subprocess of ``claude --print``, which is a
subprocess of ``serve.py``. They are three separate Python processes,
so ``serve.py``'s in-process ``agent_registry`` is INVISIBLE to the
MCP server (its own module instance has an empty registry). Direct
delivery via ``target.runtime.inbox.push_back`` is therefore not an
option from the MCP server's perspective.

We bridge the gap with HTTP: the MCP server POSTs to ``serve.py``'s
loopback endpoints (``/api/post`` and ``/api/defer``) which DO have
access to the live registry. The audit log write also happens
server-side so a single ``main.jsonl`` is the source of truth and
the ``_SUPPRESS_FLAG`` sentinel is checked exactly once per call.

  - :func:`route_send`  → ``POST /api/post`` with ``{from, to, body}``.
  - :func:`schedule_defer` → ``POST /api/defer`` with ``{from, to, body, delay_s}``.
  - :func:`append_record` is kept as a process-local helper for the
    sentinel-suppressed paths where we want to write directly (not
    currently used after the HTTP refactor; kept for tests).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fcntl
import json
import logging
import os
import urllib.error
import urllib.request


_LOG = logging.getLogger(__name__)


# Data directory: holds ``main.jsonl`` + the ``sessions/`` tree
# (per-role traces, MCP configs, sentinel, debug log). Decoupled from
# the plugin's source location so deployments can co-locate audit
# data with sibling runtimes (e.g. ``channel/main.jsonl``) for unified
# end-of-day merges. Resolution order:
#
#   1. ``SAGENT_DATA_DIR`` env var if set (canonical — set by
#      ``serve.py`` at startup; propagated to MCP-server subprocesses
#      via the per-role mcp.json ``env:`` block).
#   2. Plugin source dir (default for casual/test runs).
#
# Module-level singleton: resolved once at import. Tests that need to
# override should monkeypatch :data:`MAIN_JSONL_PATH` directly (and
# any caller computing its own sessions path from the env should
# re-read the env var, not this constant).
def _resolve_data_dir() -> Path:
    env = os.environ.get("SAGENT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


DATA_DIR = _resolve_data_dir()
MAIN_JSONL_PATH = DATA_DIR / "main.jsonl"
SESSIONS_DIR = DATA_DIR / "sessions"


# Base URL of the ``serve.py`` HTTP loopback. Set by
# :mod:`mcp_sagent.config_factory` in the per-role mcp.json env so the
# MCP server subprocess inherits it. Defaults to ``127.0.0.1:8767`` for
# probe / standalone use.
_SERVE_URL = os.environ.get(
    "SAGENT_HTTP_URL",
    "http://127.0.0.1:8767",
).rstrip("/")


def iso8601_z(dt: datetime | None = None) -> str:
    """UTC ISO-8601 with millisecond precision + ``Z`` suffix."""
    if dt is None:
        dt = datetime.now(UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def append_record(
    *,
    from_role: str,
    to: Iterable[str],
    body: str,
    ts: str | None = None,
    path: Path | None = None,
) -> None:
    """Append one chat/-compatible audit record with an exclusive POSIX lock.

    Process-local — used by :mod:`serve` directly and by tests. The
    MCP server does NOT call this directly (it goes through HTTP so
    serve.py owns the write and the sentinel check).
    """
    if path is None:
        path = MAIN_JSONL_PATH
    record = {
        "ts": ts or iso8601_z(),
        "from": from_role,
        "to": list(to),
        "body": body,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_user_message(*, to_role: str, body: str, ts: str | None = None) -> None:
    """Record a human-originated message to ``main.jsonl``.

    Used by serve.py's ``/api/post`` when ``from='user'``.
    """
    append_record(from_role="user", to=[to_role], body=body, ts=ts)


# --------------------------------------------------------------------------
# HTTP bridges (used by the MCP server's tool handlers)
# --------------------------------------------------------------------------


def _http_post(path: str, payload: dict[str, Any]) -> tuple[bool, str, int]:
    """POST JSON to ``<serve_url><path>``. Returns ``(ok, body_or_error, status_code)``."""
    url = f"{_SERVE_URL}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, body, resp.status
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(exc)
        return False, body, exc.code
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", 0


def route_send(
    *,
    from_role: str,
    to: str,
    content: str,
    suppress_audit: bool = False,
) -> tuple[bool, str]:
    """Deliver a peer message via ``serve.py``'s ``/api/post``.

    Returns ``(ok, human_readable_status)`` for the MCP tool's
    ``ToolResult``. ``suppress_audit`` is ignored here — the sentinel
    file is the single source of truth and is checked server-side.

    The HTTP roundtrip is local (127.0.0.1) and small (<1KB typical
    body), so latency is negligible compared to model-call latency.
    """
    ok, body, status = _http_post(
        "/api/post",
        {"from": from_role, "to": to, "body": content},
    )
    if ok:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        if parsed.get("ok"):
            return True, f"Delivered to {to}."
        return False, body or f"HTTP {status}"
    return False, f"delivery failed (HTTP {status}): {body}"


def schedule_defer(
    *,
    sender: str,
    delay_s: int,
    body: str,
    suppress_audit: bool = False,
) -> tuple[bool, str]:
    """Schedule a self-wake-up via ``serve.py``'s ``/api/defer``.

    Returns ``(ok, status)``. Sender is the calling agent (recipient
    of the future wake-up). Validation (delay range, registry lookup)
    lives server-side.
    """
    if delay_s < 1 or delay_s > 3600:
        return False, f"delay_s must be in [1, 3600], got {delay_s}"
    ok, body_resp, status = _http_post(
        "/api/defer",
        {"from": sender, "to": sender, "body": body, "delay_s": delay_s},
    )
    if ok:
        try:
            parsed = json.loads(body_resp)
        except json.JSONDecodeError:
            parsed = {}
        if parsed.get("ok"):
            return True, f"Scheduled wake-up for @{sender} in {delay_s}s."
        return False, body_resp or f"HTTP {status}"
    return False, f"defer scheduling failed (HTTP {status}): {body_resp}"


def cancel_all_deferred() -> int:
    """No-op in the HTTP-bridged model — defers are server-side asyncio.

    Kept as a test-stable API; the test suite for the deprecated
    in-process scheduler still references it.
    """
    return 0
