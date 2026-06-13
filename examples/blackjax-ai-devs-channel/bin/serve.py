"""Live serve: bring up all 5 agents + an HTTP/web viewer on 127.0.0.1.

Run from the plugin root:

    python examples/blackjax-ai-devs-channel/bin/serve.py [--port 8767] [--host 127.0.0.1]

What this does
--------------

1. Constructs the 5 role agents via the per-role ``build()`` factories.
   Each agent's CLI subprocess is launched with ``--mcp-config`` that
   includes the plugin's MCP server (``mcp_sagent/server.py``) with
   ``SAGENT_ROLE=<label>`` in its env — so peer messaging, self-defer,
   and status-report all flow through ``mcp__sagent_chat__*`` tool calls
   instead of sagent's bridge-mounted ``AgentSend``/``AgentSelf``.
2. Marks each agent ``_persistent=True`` and registers each in
   ``sagent.tools.core.agent_registry`` keyed by role label.
3. Adds ``user`` + ``system`` FakeAgents so peer messages addressed to
   them satisfy the "unknown agent" validation. The web UI is the
   user-visible surface for ``user``-targeted messages (read from
   ``main.jsonl``).
4. Starts ``serve_forever()`` for each agent as a background task.
5. Touches the ``_suppress_audit`` sentinel, fires the bootstrap probe
   (one MCP-bridge warm-up turn per agent), waits for AgentIdle, then
   removes the sentinel — so the warmup turn writes zero audit log
   records and pushes zero peer messages.
6. Starts a Starlette+uvicorn HTTP server on the configured host:port.

Shutdown via SIGTERM or Ctrl-C: the HTTP server stops first, then
each agent's runtime drains gracefully.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import argparse
import asyncio
import json
import logging
import os
import signal
import sys


_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

# Pulls in the data-dir resolution (env-configurable). All file paths
# below derive from ``delivery.DATA_DIR`` / ``delivery.SESSIONS_DIR``
# so a single ``SAGENT_DATA_DIR`` env var co-locates the audit log,
# trace files, per-role mcp.json, sentinel, and debug log — useful
# for end-of-day merges that union this plugin's main.jsonl with the
# sibling chat/ runtime's main.jsonl in one directory.
from datetime import UTC

# v2.1-β: imported at module scope (rather than inside
# ``_run_materializer_tripwire``) so test fixtures can monkeypatch it.
# When the parent ``sagent`` install is older than the materializer
# module, the import silently no-ops and the tripwire treats it as
# "v2 fallback". Operators on older sagent: upgrade to pick up v2.1.
from typing import Any

from mcp_sagent import delivery


arun_canary_against_live_cli: Any
try:
    from sagent.providers.anthropic_cli_session import (
        arun_canary_against_live_cli as _live_canary,
    )

    arun_canary_against_live_cli = _live_canary
except ImportError:
    arun_canary_against_live_cli = None


_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8767
_MAIN_JSONL = delivery.MAIN_JSONL_PATH

# Sentinel file: when present, the plugin's MCP server skips audit
# log writes AND peer inbox pushes. Set during the bootstrap warmup
# window. Path must match :data:`mcp_sagent.server._SUPPRESS_FLAG`.
_SUPPRESS_FLAG = delivery.SESSIONS_DIR / "_suppress_audit"

_LOG = logging.getLogger("blackjax_chat.serve")


# --------------------------------------------------------------------------
# Agent bring-up
# --------------------------------------------------------------------------


def _build_all_agents():
    """Construct + return the 5 role agents, each marked persistent + registered.

    Each agent's ``claude --print`` subprocess is wired via
    ``--mcp-config`` to the plugin's MCP server with ``SAGENT_ROLE=<label>``
    in the server's env — the sagent_send/sagent_defer/sagent_self tools
    appear in the catalog as ``mcp__sagent_chat__*`` and are the only path
    for peer messaging. No ``MentionRouter`` or ``DeferRouter``
    observer is installed; the prose-fallback machinery from the
    previous attempt is replaced by structural tool calls.
    """
    from roles.junior_swe import build as build_junior
    from roles.statistician import build as build_stat
    from roles.swe import build as build_swe
    from roles.tech_writer import build as build_tw
    from roles.tl import build as build_tl
    from runtime import trace_writer

    from sagent.testing import FakeAgent
    from sagent.tools.core import agent_registry

    builders = {
        "tl": build_tl,
        "swe": build_swe,
        "junior-swe": build_junior,
        "statistician": build_stat,
        "tech-writer": build_tw,
    }
    agents = {}
    for label, builder in builders.items():
        agent = builder()
        agent._persistent = True
        agent_registry[label] = agent
        trace_writer.install_on(agent, label)
        agents[label] = agent

    # ``user`` is a mailbox-without-listener: ``sagent_send(to='user', …)``
    # lands here. The FakeAgent's inbox accumulates undelivered notes
    # harmlessly; the web UI shows them via the audit log.
    agent_registry["user"] = FakeAgent()
    # ``system`` is reserved for diagnostic probes / future eventing.
    agent_registry["system"] = FakeAgent()

    return agents


async def _serve_agents_forever(agents):
    """Spawn one ``serve_forever`` task per agent; return the gather handle."""
    tasks = [
        asyncio.create_task(agent.serve_forever(), name=f"serve-{label}")
        for label, agent in agents.items()
    ]
    return tasks


_MATERIALIZER_TRIPWIRE_ENV = "SAGENT_CLI_OWN_SESSION"


async def _run_materializer_tripwire() -> None:
    """Startup tripwire for the CLI session materializer (default-on since
    v2.1-β graduated 2026-06-09).

    Materializer mode is the default: sagent owns the on-disk session
    JSONL. Operators can opt out by setting
    ``SAGENT_CLI_OWN_SESSION=0`` (or ``false``/``no``) — that skips
    the tripwire entirely and ``_build_all_agents`` falls back to v2
    CLI-owned mode.

    When NOT opted out, the tripwire spawns a 1-turn canary against
    ``claude --print``, schema-checks claude's JSONL output, runs a
    structural round-trip diff against what the materializer would
    have written, and:

    - On clean verdict: leaves the env unchanged (default-on stays
      on); ``_build_all_agents`` passes ``materialize_session=True``
      to every provider.
    - On drift / canary unavailable / canary raised: sets
      ``SAGENT_CLI_OWN_SESSION=0`` so ``_build_all_agents`` falls
      back to v2 CLI-owned mode. The findings are logged at WARNING
      level so the operator sees the cause and can pin the CLI
      version or update the materializer.

    Always logs the verdict (one INFO line on success, one WARNING
    line per finding on drift) so the boot transcript shows whether
    materialization mode is live for this run.
    """
    opt_out = os.environ.get(_MATERIALIZER_TRIPWIRE_ENV, "").lower() in (
        "0",
        "false",
        "no",
    )
    if opt_out:
        _LOG.info(
            "materializer tripwire: %s=0 — v2 CLI-owned mode (operator opt-out)",
            _MATERIALIZER_TRIPWIRE_ENV,
        )
        return

    if arun_canary_against_live_cli is None:
        _LOG.warning(
            "materializer tripwire: import unavailable; falling back to v2 mode"
        )
        os.environ[_MATERIALIZER_TRIPWIRE_ENV] = "0"
        return

    _LOG.info("materializer tripwire: spawning canary against claude --print…")
    try:
        result = await arun_canary_against_live_cli()
    except Exception as exc:
        _LOG.warning(
            "materializer tripwire: canary raised %s: %s; falling back to v2 mode",
            type(exc).__name__,
            exc,
        )
        os.environ[_MATERIALIZER_TRIPWIRE_ENV] = "0"
        return

    if result.is_safe:
        _LOG.info(
            "materializer tripwire: PASS — sagent will own the session JSONL "
            "for this boot (default-on, %s unset/non-zero)",
            _MATERIALIZER_TRIPWIRE_ENV,
        )
        return

    _LOG.warning(
        "materializer tripwire: FAIL — %d finding(s); falling back to v2 mode",
        len(result.findings),
    )
    for finding in result.findings:
        _LOG.warning("  drift @ %s: %s", finding.location, finding.detail)
    os.environ[_MATERIALIZER_TRIPWIRE_ENV] = "0"


def _rehydrate_agents_from_jsonl(agents) -> None:
    """Seed each agent's tape from its on-disk claude JSONL (materialize-mode
    resume-from-memory).

    After a ``serve.py`` restart, sagent's in-memory tape is empty (the
    plugin does not persist sagent's own ``session.jsonl``). In materialize
    mode that is fatal to continuity: on the second turn the materializer
    rewrites the on-disk JSONL from the short fresh tape and CLOBBERS the
    full pre-restart history. To preserve continuity we reconstruct the
    tape from the claude JSONL itself -- the materializer module ships the
    inverse parser -- so the resolved view matches the prior conversation
    and the materializer rewrites it faithfully (no clobber).

    Steps per role agent: ``parse_jsonl_to_messages`` -> repair any dangling
    tool-call pairing (truncated mid-tool writes) -> wrap each message as a
    ``ReferrableTapeEvent`` -> ``runtime.replay_tape`` ->
    ``model.seed_session(len(messages))`` so the provider knows the on-disk
    JSONL already holds that tape prefix (new entries go via stdin, not
    re-fed; the next spawn uses ``--resume``).

    No-op when materialization is off (``SAGENT_CLI_OWN_SESSION`` opted out),
    where ``claude --resume`` already owns and resumes the on-disk history
    natively. Best-effort and never crashes boot: a per-agent failure logs a
    warning and that agent starts fresh.
    """
    opt_out = os.environ.get(_MATERIALIZER_TRIPWIRE_ENV, "").lower() in (
        "0",
        "false",
        "no",
    )
    if opt_out:
        _LOG.info("rehydrate: v2 CLI-owned mode — claude --resume owns history")
        return
    try:
        from sagent.agent.session_io import repair_dangling_tool_calls
        from sagent.providers.anthropic_cli_session import (
            parse_jsonl_to_messages,
            session_jsonl_path,
        )
        from sagent.types.tape import ReferrableTapeEvent, TapeRef
    except ImportError as exc:
        _LOG.warning("rehydrate: import unavailable (%s); agents start fresh", exc)
        return

    cwd = Path.cwd()
    for label, agent in agents.items():
        # ``agents`` holds only the 5 real role agents (the ``user`` /
        # ``system`` FakeAgents live in ``agent_registry``, not here), so
        # ``agent.model`` is always the ``_AnthropicCLIModel``. ``model``
        # is dynamically typed (``Any``) because the generic ``Model``
        # protocol doesn't carry the session-persistence surface
        # (``session_id`` / ``seed_session``) — both are public API on
        # the AnthropicCLI model.
        model: Any = agent.model
        session_id = getattr(model, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            continue
        try:
            path = session_jsonl_path(session_id, cwd=cwd)
            if not path.exists():
                _LOG.info("rehydrate %s: no prior JSONL — fresh start", label)
                continue
            parsed: list[Any] = list(parse_jsonl_to_messages(path))
            messages = repair_dangling_tool_calls(parsed)
            if not messages:
                _LOG.info("rehydrate %s: JSONL parsed empty — fresh start", label)
                continue
            records = [
                ReferrableTapeEvent(
                    ref=TapeRef(session_id=session_id, ordinal=i),
                    event=message,
                )
                for i, message in enumerate(messages)
            ]
            agent.runtime.replay_tape(records)
            # Mark the on-disk JSONL as the already-synced prefix so the
            # materializer rewrites it faithfully and new entries go via
            # stdin rather than being re-fed.
            model.seed_session(len(messages))
            _LOG.info(
                "rehydrate %s: seeded %d messages from %s",
                label,
                len(messages),
                path.name,
            )
        except Exception as exc:
            _LOG.warning(
                "rehydrate %s: failed (%s: %s); agent starts fresh",
                label,
                type(exc).__name__,
                exc,
            )


# --------------------------------------------------------------------------
# Startup warmup — fire one MCP-tool-call-using turn per agent before
# accepting user traffic, so the first real user message doesn't hit the
# AnthropicCLI's "first-turn fumbles MCP tool calls" pattern observed in
# spike sigint_probe3 and race_repro_v2.
# --------------------------------------------------------------------------


_WARMUP_PROMPT = (
    "BOOTSTRAP PROBE. You MUST call the tool "
    "`mcp__sagent_chat__sagent_self` with arguments "
    '`{"status": "ready"}` right now, as your FIRST and ONLY '
    "action in this turn. Do not respond with text in place of the "
    "tool call — describing what you would do does not count and "
    "the bootstrap will be considered failed. After the tool returns "
    "its result, end the turn with the single word `ok` as a "
    "non-empty text content block.\n\n"
    "This is a one-time startup probe, not a real request. Future "
    "messages from peers and users are real work and must be "
    "handled normally — never reuse this bootstrap pattern as a "
    "reply template (e.g. do not reply to a real message with "
    "`sagent_send(content='ready')` or `'hello, ready'`)."
)


async def _warmup_agents(agents, *, timeout_s: float = 90.0) -> dict[str, bool]:
    """Send a warmup directive to each agent in parallel; wait for AgentIdle.

    Returns a ``{label: success}`` map. ``success=True`` means the agent
    reached AgentIdle within ``timeout_s`` after the warmup push;
    ``False`` means it timed out (we proceed anyway — a slow warmup
    shouldn't block the whole server, and the audit log will surface
    any agent that never speaks).

    The MCP server's audit-log writes AND peer inbox pushes are
    suppressed by touching the ``_SUPPRESS_FLAG`` sentinel before the
    warmup push and removing it after every agent has reached
    AgentIdle (or timed out). The MCP server reads this file's
    existence on every ``CallToolRequest``, so toggling propagates
    across all per-agent MCP server subprocesses without needing to
    restart any of them. The bootstrap probe uses ``sagent_self``
    (a status-report tool that does not touch any peer inbox) so the
    warmup turn is silent to peers by construction.

    The trailing "acknowledge with 'ok'" instruction is deliberate —
    Anthropic's API rejects assistant messages with empty text
    content blocks (``400 messages: text content blocks must be
    non-empty``), so a single-word ack on the warmup turn guarantees
    the history shape is valid for subsequent turns.
    """
    from sagent.types.runtime import AgentIdle, UserMessage

    try:
        idle_events: dict[str, asyncio.Event] = {}
        observers: list[Any] = []
        for label, agent in agents.items():
            evt = asyncio.Event()
            idle_events[label] = evt

            def _watcher(ev, _evt=evt):
                if isinstance(ev, AgentIdle):
                    _evt.set()

            agent.runtime.observers.append(_watcher)
            observers.append((agent, _watcher))

        # Touch the suppression sentinel BEFORE pushing the prompt so
        # the very first MCP CallToolRequest fired by the bootstrap
        # turn sees an empty audit log + a silent inbox.
        _SUPPRESS_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _SUPPRESS_FLAG.touch()

        for label, agent in agents.items():
            agent.runtime.inbox.push_back(UserMessage(text=_WARMUP_PROMPT))

        async def _wait_one(label: str) -> tuple[str, bool]:
            try:
                await asyncio.wait_for(idle_events[label].wait(), timeout_s)
                return label, True
            except TimeoutError:
                return label, False

        results = await asyncio.gather(*[_wait_one(l) for l in agents])
        for agent, watcher in observers:
            try:
                agent.runtime.observers.remove(watcher)
            except ValueError:
                pass
        # DO NOT call Agent.clear() here. It works in principle (preempt
        # + wipe history + reset per-tool recall) but in the
        # AnthropicCLI path, the history shrink trips
        # ``_should_respawn`` in the provider, which kills the warm
        # CLI subprocess and respawns a fresh one — undoing exactly
        # the MCP-bridge warmup we just paid for. Leave the warmup
        # turn in history; the new prompt's explicit "do NOT reuse
        # this pattern" framing + the use of ``sagent_self`` (silent
        # to peers) means even a parrot-after-respawn doesn't leak.
        return dict(results)
    finally:
        # Re-enable audit logging + peer deliveries for real traffic.
        _SUPPRESS_FLAG.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# HTTP routes
# --------------------------------------------------------------------------


_DEBUG_HTML_PATH = _PLUGIN_DIR / "web" / "debug.html"
_INDEX_HTML_PATH = _PLUGIN_DIR / "web" / "index.html"


def _event_search_text(ev: dict[str, Any]) -> tuple[str, str]:
    """Return ``(kind, flattened-searchable-text)`` for one trace event.

    Mirrors ``chat/chat:_event_search_text`` but uses sagent's event
    field names. Trace events come from ``trace_writer.TraceWriter``
    which serializes RuntimeEvent dataclasses by ``_event`` + their
    fields (e.g. ``text``, ``tool_calls``, ``message``, ``source``).
    """
    kind = ev.get("_event") or "?"
    parts: list[str] = []
    # Top-level text on AssistantMessage / AgentSendMessage /
    # UserMessage / ModelResponsePartial events.
    txt = ev.get("text")
    if isinstance(txt, str) and txt:
        parts.append(txt)
    # Nested message payload (ModelResponseComplete).
    msg = ev.get("message")
    if isinstance(msg, dict):
        mt = msg.get("text")
        if isinstance(mt, str) and mt:
            parts.append(mt)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    name = str(tc.get("name", ""))
                    args = tc.get("args") or tc.get("arguments") or {}
                    try:
                        args_str = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_str = str(args)
                    parts.append(f"{name} {args_str}")
    # ToolLabel events carry an inline label.
    if "label" in ev and isinstance(ev["label"], str):
        parts.append(ev["label"])
    # Channel records embedded in sagent events (rare but possible).
    src = ev.get("source")
    if isinstance(src, str) and src:
        parts.append(src)
    return kind, "  ".join(parts)


def _snippet(text: str, q: str, width: int = 180) -> str:
    """One-line window of ``text`` centred on the first match of ``q``."""
    low = text.lower()
    i = low.find(q.lower())
    if i < 0:
        return text[:width].replace("\n", " ")
    start = max(0, i - 50)
    end = min(len(text), i + len(q) + (width - 50))
    s = text[start:end].replace("\n", " ")
    return ("…" if start > 0 else "") + s + ("…" if end < len(text) else "")


def _iter_trace_files():
    """Yield ``(role, path)`` for every ``<role>.trace.jsonl`` under sessions/."""
    sessions = delivery.SESSIONS_DIR
    if not sessions.exists():
        return
    for p in sorted(sessions.glob("*.trace.jsonl")):
        yield p.name[: -len(".trace.jsonl")], p


def _read_jsonl(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _search_all(q: str, scope: str, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """Grep ``main.jsonl`` and/or every ``sessions/*.trace.jsonl`` for ``q``."""
    results: list[dict[str, Any]] = []
    if not q:
        return results, False
    ql = q.lower()
    if scope in ("messages", "all"):
        for i, m in enumerate(_read_jsonl(_MAIN_JSONL)):
            body = m.get("body", "") or ""
            if ql in body.lower():
                results.append(
                    {
                        "source": "message",
                        "idx": i,
                        "ts": m.get("ts", ""),
                        "from": m.get("from", ""),
                        "to": m.get("to", []),
                        "snippet": _snippet(body, q),
                    }
                )
                if len(results) >= limit:
                    return results, True
    if scope in ("traces", "all"):
        for role, p in _iter_trace_files():
            for i, ev in enumerate(_read_jsonl(p)):
                kind, text = _event_search_text(ev)
                if ql in text.lower():
                    results.append(
                        {
                            "source": "trace",
                            "role": role,
                            "idx": i,
                            "ts": ev.get("_ts", ""),
                            "kind": kind,
                            "snippet": _snippet(text, q),
                        }
                    )
                    if len(results) >= limit:
                        return results, True
    return results, False


def _diagnose_agent(label: str, agent) -> dict[str, Any]:
    """Compute status + diagnosis for one agent, mirroring chat/'s _agents_status fields.

    Sagent's single-process model maps the chat/ status set (dead, hung,
    stuck, working, idle) like this:
      - dead   : never (if HTTP is up, the agent's asyncio task is alive)
      - working: model_call is not None
      - hung   : model_call active AND last assistant turn >90s ago (or never)
      - stuck  : inbox has queued messages AND no model_call AND no recent activity
      - idle   : model_call None and inbox empty
    """
    from datetime import datetime

    in_flight_call = agent.runtime.model_call is not None
    inbox_pending = 0
    try:
        inbox_pending = agent.runtime.inbox._queue.qsize()
    except AttributeError:
        pass

    # Compute last_assistant_ts + age_sec from history tail.
    last_ts_iso: str | None = None
    age_sec: int | None = None
    for m in reversed(agent.history):
        if type(m).__name__ == "AssistantMessage":
            ts = getattr(m, "timestamp", None)
            if isinstance(ts, (int, float)):
                last_ts_iso = (
                    datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.")
                    + f"{int((ts % 1) * 1000):03d}Z"
                )
                age_sec = int(datetime.now(UTC).timestamp() - ts)
            break

    # Recent trace events (last 6) — sourced from the trace file rather
    # than agent.runtime.events to match chat/'s "tail the trace" check.
    try:
        from runtime import trace_writer

        tp = trace_writer.trace_path_for(label)
        events = _read_jsonl(tp)
    except Exception:
        events = []
    recent = []
    for ev in events[-6:]:
        kind, text = _event_search_text(ev)
        recent.append(
            {
                "ts": ev.get("_ts", ""),
                "kind": kind,
                "summary": text[:200],
            }
        )

    # Inflight detection: was there a ModelCallStarted with no matching
    # ModelIdle/ModelResponseComplete/ModelResponseError in the recent tail?
    in_turn = False
    started_idx = -1
    ended_idx = -1
    for i, ev in enumerate(events[-100:]):
        k = ev.get("_event") or "?"
        if k == "ModelCallStarted":
            started_idx = i
        elif k in ("ModelIdle", "ModelResponseComplete", "ModelResponseError"):
            ended_idx = i
    in_turn = started_idx > ended_idx

    inflight: str | None = None
    if in_turn:
        # The most recent ToolLabel published since the turn began is our
        # best proxy for "what tool is currently running".
        for ev in reversed(events[-100:]):
            if (ev.get("_event") or "?") == "ToolLabel":
                inflight = str(ev.get("text") or ev.get("label") or "")
                break
        if inflight is None:
            inflight = "model thinking"

    # Last completed turn outcome.
    last_result: dict[str, Any] | None = None
    for ev in reversed(events):
        k = ev.get("_event") or "?"
        if k == "ModelResponseComplete":
            last_result = {"ok": True, "ts": ev.get("_ts", "")}
            break
        if k == "ModelResponseError":
            last_result = {"ok": False, "ts": ev.get("_ts", "")}
            break

    # Status verdict.
    if in_flight_call and (age_sec is None or age_sec > 90):
        status = "hung"
    elif in_flight_call:
        status = "working"
    elif inbox_pending > 0 and (age_sec is None or age_sec > 120):
        status = "stuck"
    elif age_sec is not None and age_sec < 60:
        status = "working"
    else:
        status = "idle"

    # One-line diagnosis.
    if status == "hung":
        tail = f" — in-flight {inflight}" if inflight else ""
        diagnosis = (
            f"Model call in flight for {age_sec or '?'}s with no progress{tail}. "
            f"May be a slow CLI subprocess or a stuck tool; consider restart."
        )
    elif status == "stuck":
        diagnosis = (
            f"Inbox has {inbox_pending} message(s) queued and no recent activity "
            f"({age_sec or '?'}s since last assistant turn). Likely a runtime drain stall."
        )
    elif status == "working":
        if in_flight_call:
            diagnosis = f"Model call active ({age_sec or '?'}s since last reply)."
        else:
            diagnosis = f"Recently active ({age_sec}s since last assistant turn)."
    else:
        diagnosis = "Last turn complete, inbox empty — waiting for work."

    model_id = ""
    try:
        model_id = agent.model.model_id
    except AttributeError:
        pass

    # Pending preview from the inbox.
    # asyncio.Queue doesn't expose its underlying deque publicly; we peek
    # at the private ``_queue`` (collections.deque) for the preview.
    # Best-effort and read-only — never mutated.
    pending_preview: list[dict[str, Any]] = []
    try:
        deque_items = list(agent.runtime.inbox._queue._queue)
        for item in deque_items[:4]:
            text = (getattr(item, "text", "") or "").replace("\n", " ")[:140]
            src = getattr(item, "source", "") or type(item).__name__
            pending_preview.append(
                {
                    "from": src,
                    "ts": "",
                    "snippet": text,
                }
            )
    except (AttributeError, TypeError):
        pass

    return {
        "role": label,
        "alive": True,
        "pid": None,  # not meaningful in single-process model
        "status": status,
        "diagnosis": diagnosis,
        "in_turn": in_turn,
        "inflight": inflight,
        "blocked_on": None,
        "wchan": None,
        "last_result": last_result,
        "last_ts": last_ts_iso,
        "trace_mtime": None,
        "age_sec": age_sec,
        "pending": inbox_pending,
        "pending_preview": pending_preview,
        "recent": recent,
        "model_id": model_id,
        "total_cost_usd": float(agent.total_cost_usd),
    }


def _build_http_app(agents):
    """Construct the Starlette app.

    Endpoints (mirroring ``chat serve``'s API for parity with the
    pre-existing operator tooling):

      GET  /                     Discord-style viewer HTML
      GET  /debug                Debug console (agents + search tabs)
      GET  /api/roles            Known role labels (static)
      GET  /api/agents           Per-agent liveness + activity + diagnosis
      GET  /api/messages         Recent main.jsonl records;
                                 ``?since=ts&limit=N`` or
                                 ``?around=N&ctx=K`` (debug "show context")
      GET  /api/trace/<role>     Per-role runtime event JSONL; supports
                                 ``?around=N&ctx=K`` window
      GET  /api/search           Full-text search across messages + traces;
                                 ``?q=&scope=traces|messages|all&limit=N``
      POST /api/post             User-ingress; body: ``{to, body}``
      POST /api/restart          Soft restart (clear history) of one agent;
                                 ``{role, skip_backlog}``; loopback-only

    Backwards-compatible aliases kept for the migration window:
      GET  /messages = /api/messages
      GET  /agents   = /api/agents
      POST /send     = /api/post
    """
    from mcp_sagent import delivery
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from starlette.routing import Route

    # Static role labels mirror chat/chat:39-42's ``KNOWN_ROLES``.
    _KNOWN_ROLES = [
        "user",
        "tl",
        "swe",
        "junior-swe",
        "statistician",
        "tech-writer",
        "system",
    ]

    async def index(_request: Request) -> Response:
        if not _INDEX_HTML_PATH.exists():
            return JSONResponse(
                {"error": f"missing {_INDEX_HTML_PATH}"}, status_code=404
            )
        return HTMLResponse(_INDEX_HTML_PATH.read_text(encoding="utf-8"))

    async def debug_index(_request: Request) -> Response:
        if not _DEBUG_HTML_PATH.exists():
            return JSONResponse(
                {"error": f"missing {_DEBUG_HTML_PATH}"}, status_code=404
            )
        return HTMLResponse(_DEBUG_HTML_PATH.read_text(encoding="utf-8"))

    async def list_roles(_request: Request) -> Response:
        return JSONResponse({"roles": _KNOWN_ROLES})

    async def search(request: Request) -> Response:
        qp = request.query_params
        q = (qp.get("q") or "").strip()
        scope = qp.get("scope") or "traces"
        if scope not in ("traces", "messages", "all"):
            scope = "traces"
        try:
            limit = max(1, min(1000, int(qp.get("limit", "300"))))
        except ValueError:
            limit = 300
        results, truncated = _search_all(q, scope, limit)
        return JSONResponse(
            {
                "q": q,
                "scope": scope,
                "results": results,
                "truncated": truncated,
            }
        )

    async def restart(request: Request) -> Response:
        # Soft restart: in the single-process model we can't actually kill
        # one agent's subprocess and respawn it cleanly without losing the
        # other four. The closest semantic equivalent is to wipe the
        # agent's history (forces a fresh ``claude --print`` respawn via
        # the HotSpare's ``_should_respawn`` check) + optionally drain
        # its inbox (``skip_backlog``).
        client_host = (request.client.host if request.client else "") or ""
        if client_host not in ("127.0.0.1", "::1", "localhost", ""):
            return JSONResponse(
                {"error": "restart disabled: not a loopback client"},
                status_code=403,
            )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"bad json: {exc}"}, status_code=400)
        role = str(payload.get("role") or "").strip()
        skip = bool(payload.get("skip_backlog"))
        if role not in agents:
            return JSONResponse(
                {"error": f"unknown role {role!r}", "known": sorted(agents)},
                status_code=404,
            )
        agent = agents[role]
        notes: list[str] = []
        if skip:
            # GatedDeque wraps asyncio.Queue (sagent runtime.py:552).
            # The Queue's underlying ``_queue`` is a ``collections.deque``;
            # we drop it directly. Won't unblock any awaiting consumer,
            # but the subsequent ``Agent.clear()`` resets the model_call /
            # cohort / detached state so the runtime returns to a clean
            # baseline.
            try:
                q = agent.runtime.inbox._queue
                drained = 0
                while not q.empty():
                    try:
                        q.get_nowait()
                        drained += 1
                    except Exception:
                        break
                notes.append(f"drained {drained} pending inbox item(s)")
            except AttributeError:
                notes.append("inbox drain unsupported on this sagent version")
        # ``Agent.clear()`` preempts the in-flight model_call + wipes
        # history + resets the per-tool recall cache. The HotSpare's
        # ``_should_respawn`` check then triggers a fresh ``claude --print``
        # subprocess on the next model call (history.length < last_sent_index).
        try:
            await agent.clear()
            notes.append("history cleared; CLI subprocess will respawn on next turn")
            ok = True
        except Exception as exc:
            notes.append(f"clear() failed: {type(exc).__name__}: {exc}")
            ok = False
        return JSONResponse(
            {
                "ok": ok,
                "role": role,
                "skip_backlog": skip,
                "output": "\n".join(notes),
            }
        )

    def _read_all_records() -> list[dict[str, Any]]:
        if not _MAIN_JSONL.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(_MAIN_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    async def list_messages(request: Request) -> Response:
        qp = request.query_params
        records = _read_all_records()

        # ``around=N&ctx=K`` returns a window of K records before and after
        # index N, plus offset/total/hit. Used by the debug page's
        # "show context" tab — identical contract to chat serve.
        if "around" in qp:
            total = len(records)
            try:
                around = int(qp["around"])
            except ValueError:
                around = total - 1
            try:
                ctx = max(0, min(50, int(qp.get("ctx", "5"))))
            except ValueError:
                ctx = 5
            start = max(0, around - ctx)
            end = min(total, around + ctx + 1)
            return JSONResponse(
                {
                    "records": records[start:end],
                    "offset": start,
                    "total": total,
                    "hit": around,
                }
            )

        since = qp.get("since")
        try:
            limit = max(1, min(2000, int(qp.get("limit", "200"))))
        except ValueError:
            limit = 200
        if since:
            records = [r for r in records if r.get("ts", "") > since]
        if len(records) > limit:
            records = records[-limit:]
        return JSONResponse({"records": records})

    async def list_agents(_request: Request) -> Response:
        # Per-agent liveness, activity, diagnosis, recent trace, inflight
        # tool, and pending-queue preview. Returns the **list** shape that
        # debug.html's render loop iterates over (chat-serve parity).
        from datetime import datetime

        items = [_diagnose_agent(label, agent) for label, agent in agents.items()]
        items.sort(key=lambda d: d["role"])
        now = datetime.now(UTC)
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        return JSONResponse({"agents": items, "now": now_iso})

    async def get_trace(request: Request) -> Response:
        """Per-role runtime-event trace.

        Mirrors chat serve's ``GET /api/trace/<role>``:

          GET /api/trace/<role>              → last 500 events
          GET /api/trace/<role>?around=N&ctx=K → window of K events on each side
                                                  of index N + offset/total/hit
        """
        from runtime import trace_writer

        role = request.path_params["role"]
        if role not in agents:
            return JSONResponse(
                {"error": f"unknown role {role!r}", "known": sorted(agents)},
                status_code=404,
            )
        path = trace_writer.trace_path_for(role)
        if not path.exists():
            return JSONResponse({"events": [], "total": 0})

        events: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # ``total`` is ALWAYS the on-disk count (post-fix 2026-06-03).
        # The previous shape returned ``total = len(events)`` AFTER
        # tail-truncation to the limit, so once a trace file grew past
        # ``limit`` events, the reported total froze at ``limit`` and
        # the frontend's "did anything new arrive?" check
        # (``total > traceTotal``) never fired -- panel stuck.
        total_on_disk = len(events)
        qp = request.query_params
        if "around" in qp:
            try:
                around = int(qp["around"])
            except ValueError:
                around = total_on_disk - 1
            try:
                ctx = max(0, min(200, int(qp.get("ctx", "14"))))
            except ValueError:
                ctx = 14
            start = max(0, around - ctx)
            end = min(total_on_disk, around + ctx + 1)
            return JSONResponse(
                {
                    "events": events[start:end],
                    "offset": start,
                    "total": total_on_disk,
                    "hit": around,
                }
            )

        # Default: tail of the last N events. Cap raised from 500
        # → 2000 so a busy multi-hour TL session fits comfortably.
        try:
            limit = max(1, min(20000, int(qp.get("limit", "2000"))))
        except ValueError:
            limit = 2000
        sliced = events[-limit:] if total_on_disk > limit else events
        return JSONResponse(
            {
                "events": sliced,
                "total": total_on_disk,
                "returned": len(sliced),
            }
        )

    async def post(request: Request) -> Response:
        """Operator ingress + cross-process peer routing.

        Body shape: ``{to, body, from?}``.

          - ``from`` absent or ``"user"``: operator ingress. Writes a
            ``user → [to]`` audit record and pushes ``UserMessage``
            into the target's inbox (existing web-UI behavior).
          - ``from`` is a known agent label: peer routing called by
            the plugin's MCP server. Writes a ``<from> → [to]`` audit
            record and pushes ``AgentSendMessage(source=<from>)`` into
            the target's inbox. This is how the out-of-process MCP
            server (subprocess of ``claude --print``) reaches the
            in-process ``agent_registry`` that ``serve.py`` owns.
            Respects the ``_SUPPRESS_FLAG`` sentinel for warmup-window
            silence.
        """
        from sagent.tools.core import agent_registry
        from sagent.types.runtime import AgentSendMessage, UserMessage

        payload = await request.json()
        to = str(payload.get("to", "")).strip()
        body = str(payload.get("body", ""))
        from_role = str(payload.get("from", "user")).strip() or "user"
        # ``urgent`` flag (default False) applies to BOTH ingress paths:
        #
        # * Operator messages (``from='user'``) → ``UserMessage(urgent=…)``:
        #   default False at the plugin layer means operator typing
        #   buffers instead of interrupting. Back-to-back operator
        #   messages don't waste the recipient's in-flight compute on
        #   routine follow-ups. The web UI exposes an "interrupt"
        #   toggle / Ctrl+Enter shortcut so the operator can opt into
        #   immediate preempt when the next message genuinely needs to
        #   land before the recipient finishes.
        # * Peer messages (``from=<peer>``) → ``AgentSendMessage(urgent=…)``:
        #   default False, same rationale. Peers explicitly pass
        #   ``urgent=True`` via the ``sagent_send`` MCP tool for STOP /
        #   pivot / same-sender-correction shapes.
        urgent = bool(payload.get("urgent", False))
        if not to or not body:
            return JSONResponse(
                {"error": "both 'to' and 'body' are required"}, status_code=400
            )
        target = agent_registry.get(to)
        if target is None:
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )
        # User-as-recipient is a "mailbox without listener" — audit
        # log is the visible surface; no inbox push.
        suppress = _SUPPRESS_FLAG.exists()
        if from_role == "user":
            if to == "user":
                return JSONResponse(
                    {"error": "user→user is meaningless; pick an agent"},
                    status_code=400,
                )
            if not suppress:
                delivery.append_user_message(to_role=to, body=body)
            target.runtime.inbox.push_back(
                UserMessage(text=body, urgent=urgent),
            )
        else:
            # Peer routing path called by the MCP server.
            if not suppress:
                delivery.append_record(from_role=from_role, to=[to], body=body)
            if to != "user":
                target.runtime.inbox.push_back(
                    AgentSendMessage(
                        source=from_role,
                        text=body,
                        urgent=urgent,
                    ),
                )
        return JSONResponse({"ok": True, "to": to, "from": from_role})

    async def defer(request: Request) -> Response:
        """Operator-side + MCP-server-side wake-up scheduling.

        Body: ``{to, body, delay_s, from?}``. Defaults ``from="operator"``.

        Schedules an ``asyncio.call_later`` that pushes
        ``AgentSendMessage(source=<from>, text='[defer +Ns] <body>')``
        into the target's inbox after ``delay_s`` seconds. Audit log
        receives one record at schedule time with body
        ``[defer +Ns scheduled] <body>``. Sentinel-respecting same as
        ``/api/post``.
        """
        from sagent.tools.core import agent_registry
        from sagent.types.runtime import AgentSendMessage

        client_host = (request.client.host if request.client else "") or ""
        if client_host not in ("127.0.0.1", "::1", "localhost", ""):
            return JSONResponse(
                {"error": "defer disabled: not a loopback client"},
                status_code=403,
            )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"bad json: {exc}"}, status_code=400)
        to = str(payload.get("to", "")).strip()
        body = str(payload.get("body", "")).strip()
        from_role = str(payload.get("from", "operator")).strip() or "operator"
        try:
            delay_s = int(payload.get("delay_s", 0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "delay_s must be an int"}, status_code=400)
        if not to or not body:
            return JSONResponse(
                {"error": "both 'to' and 'body' are required"}, status_code=400
            )
        if delay_s < 1 or delay_s > 86400:
            return JSONResponse(
                {"error": "delay_s must be in [1, 86400]"}, status_code=400
            )
        target = agent_registry.get(to)
        if target is None or to == "user":
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )

        marked = f"[defer +{delay_s}s] {body}"

        def _fire() -> None:
            tgt = agent_registry.get(to)
            if tgt is None:
                _LOG.warning("defer: target @%s gone from registry; dropping", to)
                return
            tgt.runtime.inbox.push_back(
                AgentSendMessage(source=from_role, text=marked),
            )

        asyncio.get_running_loop().call_later(delay_s, _fire)
        if not _SUPPRESS_FLAG.exists():
            delivery.append_record(
                from_role=from_role,
                to=[to],
                body=f"[defer +{delay_s}s scheduled] {body}",
            )
        return JSONResponse(
            {"ok": True, "to": to, "from": from_role, "delay_s": delay_s}
        )

    return Starlette(
        debug=False,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/debug", debug_index, methods=["GET"]),
            Route("/debug.html", debug_index, methods=["GET"]),
            # Canonical /api/* paths (chat-serve parity).
            Route("/api/roles", list_roles, methods=["GET"]),
            Route("/api/agents", list_agents, methods=["GET"]),
            Route("/api/messages", list_messages, methods=["GET"]),
            Route("/api/trace/{role}", get_trace, methods=["GET"]),
            Route("/api/search", search, methods=["GET"]),
            Route("/api/post", post, methods=["POST"]),
            Route("/api/defer", defer, methods=["POST"]),
            Route("/api/restart", restart, methods=["POST"]),
            # Backwards-compat aliases for the inline viewer's current
            # poll URLs; can be removed after the viewer is updated.
            Route("/messages", list_messages, methods=["GET"]),
            Route("/agents", list_agents, methods=["GET"]),
            Route("/send", post, methods=["POST"]),
        ],
    )


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------


async def _amain(host: str, port: int) -> int:
    import os

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Expose the port to the per-role MCP server entries built inside
    # ``_build_all_agents``: each role embeds ``SAGENT_HTTP_URL=http://
    # 127.0.0.1:<port>`` into the spawned MCP server's env, and that
    # URL is where ``mcp_sagent.delivery`` POSTs ``/api/post`` /
    # ``/api/defer``. The MCP server runs in a separate Python
    # process; this env var is the only handshake.
    os.environ["SAGENT_HTTP_PORT"] = str(port)

    # v2.1-β startup tripwire. When the operator has opted into
    # sagent-owned session JSONLs (``SAGENT_CLI_OWN_SESSION=1``), run
    # a 1-turn canary through ``claude --print`` first, compare its
    # JSONL output against what the materializer would have written,
    # and only KEEP the env flag if the verdict is clean. A drift
    # finding (unknown entry type, missing required field, structural
    # round-trip diff) clears the env var so ``_build_all_agents``
    # falls back to v2 CLI-owned mode for this boot.
    await _run_materializer_tripwire()

    agents = _build_all_agents()
    _LOG.info("brought up %d agents: %s", len(agents), sorted(agents))

    # Materialize-mode resume-from-memory: seed each agent's tape from its
    # on-disk claude JSONL BEFORE it starts serving (and before warmup), so
    # the materializer rewrites the full prior history instead of clobbering
    # it from a fresh (empty) tape on the second turn after a restart. The
    # persisted "memory" is the claude JSONL; the materializer's parser
    # reconstructs the tape from it. No-op in v2 (opt-out) mode, where
    # ``claude --resume`` owns and resumes the history natively.
    _rehydrate_agents_from_jsonl(agents)

    serve_tasks = await _serve_agents_forever(agents)

    # Startup warmup — empirically required for the AnthropicCLI
    # provider. Without it, the very first model_call after the
    # ``claude --print`` subprocess spawns hangs indefinitely
    # (observed 2026-06-01: >5 min no reply), the same first-turn
    # MCP-tool-fumble pattern we documented in the Phase 0 spike.
    # The warmup is a "hello → AgentSend(to=user, content='ready')"
    # round-trip per agent, structured so the model is conditioned on
    # the SAME ``AgentSend(to=user)`` template we want it to use for
    # real replies. The cost is one ``hello, ready`` record per agent
    # in ``main.jsonl`` at startup — visible noise but small and
    # easily filtered.
    _LOG.info("warming up agents (≤90s)…")
    warmup_results = await _warmup_agents(agents)
    ok = [l for l, v in warmup_results.items() if v]
    timed_out = [l for l, v in warmup_results.items() if not v]
    _LOG.info("warmup done: ready=%s timed_out=%s", sorted(ok), sorted(timed_out))

    app = _build_http_app(agents)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    _LOG.info("starting HTTP server on http://%s:%d", host, port)
    http_task = asyncio.create_task(server.serve(), name="http")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    _LOG.info("shutdown signal received; stopping HTTP server")
    server.should_exit = True
    await http_task
    _LOG.info("stopping agents")
    for agent in agents.values():
        agent.shutdown()
    await asyncio.gather(*serve_tasks, return_exceptions=True)
    _LOG.info("clean exit")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=_DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print(
            f"refusing to bind public host {args.host!r}; only 127.0.0.1/::1/localhost allowed",
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(_amain(args.host, args.port))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
