#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Interactive LLM agent CLI.

``--provider`` is a class name from ``sagent.providers``; ``--auth`` is
the suffix of a zero-arg ``from_<auth>`` classmethod on that class.
Dispatch is ``getattr(providers, provider).from_<auth>()`` - no
registry, no string aliases.

Usage::

    # Anthropic API key (reads ANTHROPIC_API_KEY)
    ./cli.py --provider Anthropic

    # OpenAI
    ./cli.py --provider OpenAI --model gpt-5.5

    # Google
    ./cli.py --provider Google --auth env --model gemini-3.1-pro-preview

    # Moonshot / DashScope / MiniMax (OpenAI chat-completions compatible)
    ./cli.py --provider Moonshot --model kimi-k2.6
    ./cli.py --provider DashScope --model qwen3.6-plus
    ./cli.py --provider MiniMax --model MiniMax-M2.7

    # Model + window tag (append +1m or +200k; 200K is the default)
    ./cli.py --model claude-sonnet-4-6+1m
    ./cli.py --session ~/.sessions/my
    ./cli.py --resume       # pick from past sessions for this cwd
    ./cli.py --continue     # resume the most recent for this cwd

    # Advisor strategy: Sonnet as executor, Opus as advisor.
    # See https://claude.com/blog/the-advisor-strategy
    ./cli.py --model claude-sonnet-4-6 --advisor claude-opus-4-7
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import cast

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import signal
import sys

from sagent import (
    providers,
    sessions,
    tools,
    types,
)
from sagent.agent import Agent
from sagent.agent.session_io import (
    SessionMeta,
    append_session,
    load_session,
    serialize_tool_state,
)
from sagent.compactor import SummaryCompactor
from sagent.lib.json import MutableJSON, json_unfreeze
from sagent.prompt import build_system
from sagent.providers import build_provider
from sagent.repl import run_repl
from sagent.tools.advisor import Advisor
from sagent.tools.core import set_recipe


_DEFAULT_PROVIDER = "Anthropic"
_DEFAULT_AUTH = "env"

DEFAULT_TOOLS = [
    "AgentSpawn",
    "AgentSend",
    "AgentSelf",
    # "BackgroundTask",
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "List",
    "WebSearch",
    "WebFetch",
    "PaperSearch",
    "PaperDetails",
    "PaperAuthor",
    "PaperFetch",
    "PlayAudio",
    "Skill",
]


def resolve_tools(names: list[str]) -> list[types.tools.Tool]:
    """Instantiate tools by class name from the ``tools`` module.

    Bash is constructed last with ``peers=`` set to its sibling tools
    so its ``[bash-lint]`` feature can suggest dedicated replacements
    for invocations like ``grep``/``cat``/``find``/``sed -i``.

    Args:
      names: Tool class names to look up in the tools submodule.

    Returns:
      tools: Instantiated tool objects in the requested order.

    Raises:
      SystemExit: If a tool name is not found in the tools module.

    """
    if names == ["none"]:
        return []
    non_bash: dict[str, types.tools.Tool] = {}
    for name in names:
        if name == "Bash":
            continue
        cls = getattr(tools, name, None)
        if cls is None:
            raise SystemExit(f"unknown tool: {name!r}")
        non_bash[name] = cls()
    resolved: list[types.tools.Tool] = []
    peers = tuple(non_bash.values())
    for name in names:
        if name == "Bash":
            resolved.append(tools.Bash(peers=peers))
        else:
            resolved.append(non_bash[name])
    return resolved


def _resolve_session_dir(args: argparse.Namespace) -> str | None:
    """Pick the session directory per --session / --resume / --continue."""
    if args.session is not None:
        return str(args.session)
    cwd = Path.cwd()
    if args.continue_:
        return _resolve_continue(cwd)
    if args.continue_all:
        return _resolve_continue_all()
    if args.resume is not None:
        if args.resume is True:
            return _resolve_resume(cwd)
        return _resolve_resume_hash(str(args.resume), cwd)
    if args.resume_all:
        return _resolve_resume_all()
    return str(sessions.new_session_dir(cwd))


def _resolve_continue(cwd: Path) -> str:
    """Resume the most recent session for ``cwd``, or start fresh."""
    latest = sessions.latest_session(cwd)
    if latest is not None:
        sys.stderr.write(f"[resume] {latest.path}\n")
        return str(latest.path)
    sys.stderr.write("[resume] no prior sessions for this cwd; starting fresh.\n")
    return str(sessions.new_session_dir(cwd))


def _resolve_resume_hash(session_hash: str, cwd: Path) -> str:
    """Resume a session by hash prefix (directory name match)."""
    for s in sessions.list_sessions(cwd):
        if s.path.name.startswith(session_hash):
            sys.stderr.write(f"[resume] {s.path}\n")
            return str(s.path)
    for s in sessions.list_all_sessions():
        if s.path.name.startswith(session_hash):
            sys.stderr.write(f"[resume] {s.path}\n")
            return str(s.path)
    sys.stderr.write(
        f"[resume] no session matching {session_hash!r}; starting fresh.\n"
    )
    return str(sessions.new_session_dir(cwd))


def _resolve_resume(cwd: Path) -> str:
    """Show interactive session picker, or start fresh on no selection."""
    avail = sessions.list_sessions(cwd)
    if not avail:
        sys.stderr.write("[resume] no prior sessions; starting fresh.\n")
        return str(sessions.new_session_dir(cwd))
    choice = sessions.pick_session(avail)
    if choice is not None:
        sys.stderr.write(f"[resume] {choice.path}\n")
        return str(choice.path)
    sys.stderr.write("[resume] no selection; starting fresh.\n")
    return str(sessions.new_session_dir(cwd))


def _resolve_continue_all() -> str:
    """Resume the most recent session across all projects, or start fresh."""
    all_sessions = sessions.list_all_sessions()
    if all_sessions:
        sys.stderr.write(f"[resume] {all_sessions[0].path}\n")
        return str(all_sessions[0].path)
    sys.stderr.write("[resume] no prior sessions; starting fresh.\n")
    return str(sessions.new_session_dir(Path.cwd()))


def _resolve_resume_all() -> str:
    """Show interactive picker across all projects, or start fresh on no selection."""
    avail = sessions.list_all_sessions()
    if not avail:
        sys.stderr.write("[resume] no prior sessions; starting fresh.\n")
        return str(sessions.new_session_dir(Path.cwd()))
    choice = sessions.pick_session(avail)
    if choice is not None:
        sys.stderr.write(f"[resume] {choice.path}\n")
        return str(choice.path)
    sys.stderr.write("[resume] no selection; starting fresh.\n")
    return str(sessions.new_session_dir(Path.cwd()))


def parse_agent_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Add shared agent flags to ``parser`` and parse ``argv``.

    Args:
      parser: Argparse parser to extend in place.
      argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
      parsed: Tuple of ``(namespace, remaining_args)`` from ``parse_known_args``.

    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument(
        "--provider",
        default=_DEFAULT_PROVIDER,
        help=(
            "Provider class name from ``sagent.providers`` "
            f"(default: {_DEFAULT_PROVIDER})."
        ),
    )
    parser.add_argument(
        "--auth",
        default=_DEFAULT_AUTH,
        help=(
            "Auth method suffix - dispatches to ``<Provider>.from_<auth>()``. "
            f"Default: {_DEFAULT_AUTH}."
        ),
    )
    parser.add_argument(
        "--account",
        default=None,
        metavar="NAME",
        help=(
            "Optional credential account name for providers that support named"
            " credential stores. Ignored by providers that do not."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "For login: print an auth URL and paste the returned code instead"
            " of waiting for a browser callback."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (default per provider). Append +1m / +200k to set window.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Additional system prompt instructions.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Active prompt recipe (yaml). Either a bare name resolved"
            " under ``assets/<name>.yaml`` (e.g. ``sagent``, ``bare``)"
            " or a filesystem path to a yaml. Default: sagent."
        ),
    )
    parser.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic compaction (default: on; --no-compact to disable).",
    )
    parser.add_argument(
        "--effort",
        default=None,
        help=(
            "Anthropic effort level (low|medium|high|xhigh|max)."
            " Unset = API default. Non-Anthropic providers will raise."
        ),
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Tool class names to load from the tools submodule"
            f" (default: {' '.join(DEFAULT_TOOLS)})."
        ),
    )
    parser.add_argument(
        "--add-dir",
        dest="add_dir",
        nargs="+",
        default=[],
        metavar="DIR",
        help=(
            "Additional directories whose AGENTS.md files extend the prompt beyond cwd."
        ),
    )
    parser.add_argument(
        "--max-budget-usd",
        dest="max_budget_usd",
        type=float,
        default=None,
        metavar="USD",
        help="Maximum dollar amount to spend on API calls.",
    )
    parser.add_argument(
        "--max-tool-call-rounds",
        dest="max_tool_call_rounds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Per-prompt model-request cap. Default: unlimited (interactive"
            " sessions shouldn't need one). Set a finite N to bound"
            " batch/automated runs."
        ),
    )
    parser.add_argument(
        "--max-request-tokens",
        dest="max_request_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Maximum request tokens for one model call. Default: model limit.",
    )
    parser.add_argument(
        "--max-response-tokens",
        dest="max_response_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Maximum response tokens for one model call. Default: model limit.",
    )
    args, remaining = parser.parse_known_args(raw_argv)
    args.provider_explicit = _flag_present(raw_argv, "--provider")
    args.auth_explicit = _flag_present(raw_argv, "--auth")
    args.account_explicit = _flag_present(raw_argv, "--account")
    args.model_explicit = _flag_present(raw_argv, "--model")
    return args, remaining


def _flag_present(argv: list[str], flag: str) -> bool:
    """Return True when ``flag`` appears as ``--flag`` or ``--flag=value``."""
    return any(tok == flag or tok.startswith(flag + "=") for tok in argv)


def _parse_cli_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Add CLI-specific flags and delegate to ``parse_agent_args``."""
    parser.add_argument(
        "--session",
        default=None,
        help="Session directory for persistence. Overrides --resume/--continue.",
    )
    parser.add_argument(
        "--no-session-persistence",
        dest="no_session",
        action="store_true",
        help="Disable session persistence. Sessions are not saved to disk.",
    )
    parser.add_argument(
        "--output-format",
        dest="output_format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format: text (default), json (final result), stream-json (events as NDJSON).",
    )
    parser.add_argument(
        "--input-format",
        dest="input_format",
        choices=["text", "stream-json"],
        default="text",
        help=(
            "Input format for headless mode: text (default) or stream-json"
            ' (NDJSON of {"prompt": ...} objects, joined with blank lines).'
        ),
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="HASH",
        help="Resume a session. No arg: interactive picker. With HASH: resume that session.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        action="store_true",
        help="Resume the most recent session for this cwd.",
    )
    parser.add_argument(
        "--resume-all",
        dest="resume_all",
        action="store_true",
        help="Interactive picker over past sessions across all projects.",
    )
    parser.add_argument(
        "--continue-all",
        dest="continue_all",
        action="store_true",
        help="Resume the most recent session across all projects.",
    )
    parser.add_argument(
        "--name",
        default="Agent",
        help="Agent name (default: Agent).",
    )
    parser.add_argument(
        "--history",
        default=None,
        metavar="PATH",
        help="File for input history (default: ~/.sagent_history).",
    )
    parser.add_argument(
        "--advisor",
        default=None,
        metavar="MODEL",
        help=(
            "Model ID for an advisor sub-agent the executor can"
            " consult when stuck. Typically Opus while --model is"
            " Sonnet or Haiku. Default: no advisor."
        ),
    )
    parser.add_argument(
        "--advisor-max-uses",
        dest="advisor_max_uses",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap advisor invocations for this session."
            " Default: unlimited (bounded by --max-tool-call-rounds)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Set Sagent log level. REPL mode writes logs to file;"
            " headless mode writes to stderr. Overrides SAGENT_LOG_LEVEL."
        ),
    )
    return parse_agent_args(parser, argv)


def _build_provider_model(
    args: argparse.Namespace,
) -> tuple[types.providers.Provider, types.model.Model, str]:
    """Build the provider/model pair requested by CLI flags."""
    auth = str(args.auth)
    model_id = cast(str | None, args.model)
    model_lookup = model_id
    if args.provider == "SelfHosted":
        auth = model_id or "env"
        model_lookup = None
    provider = build_provider(str(args.provider), auth, account=args.account)
    model = provider.model(model_lookup)
    return provider, model, auth


def _apply_resume_model_defaults(args: argparse.Namespace, meta: SessionMeta) -> None:
    """Layer persisted model metadata under explicit CLI model flags."""
    if meta.provider and not bool(getattr(args, "provider_explicit", False)):
        args.provider = meta.provider
    if meta.auth and not bool(getattr(args, "auth_explicit", False)):
        args.auth = meta.auth
    if meta.account and not bool(getattr(args, "account_explicit", False)):
        args.account = meta.account
    if not bool(getattr(args, "model_explicit", False)):
        args.model = meta.model_id or args.model
        if (
            bool(getattr(args, "provider_explicit", False))
            and args.model is not None
            and not _provider_knows_model(str(args.provider), str(args.model))
        ):
            args.model = None


def _provider_knows_model(provider_name: str, model_id: str) -> bool:
    """Return True when the named provider's catalog includes ``model_id``."""
    cls = getattr(providers, provider_name, None)
    if cls is None:
        return False
    known = getattr(cls, "KNOWN_MODELS", None)
    return isinstance(known, dict) and model_id in known


def _configure_logging(level: str | None) -> None:
    """Configure CLI logging from flag or environment (headless / pre-mode)."""
    raw = level or os.environ.get("SAGENT_LOG_LEVEL")
    if not raw:
        return
    name = raw.upper()
    value = getattr(logging, name, None)
    if not isinstance(value, int):
        valid = ", ".join(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
        raise SystemExit(f"invalid log level {raw!r}; use {valid}")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sagent").setLevel(value)
    logging.getLogger("sagent").setLevel(value)


def _install_repl_logging(
    level: str | None = None,
    *,
    session_dir: str | Path | None = None,
) -> None:
    """REPL mode: never write logs to stderr.

    Python's default ``lastResort`` handler emits ``WARNING+`` records
    to stderr, which corrupts prompt-toolkit's display (any stderr write
    overlays the rendered UI). Policy: stderr is for headless mode
    only. This function:

      - Replaces ``logging.lastResort`` with ``NullHandler`` so the
        implicit fallback is silent.
      - Removes any pre-installed stderr/stdout-bound handlers on the
        root logger (e.g. from a prior ``basicConfig`` call).
      - Routes records to ``SAGENT_LOG_FILE`` or, by default,
        ``<session_dir>/repl.log`` via a ``FileHandler`` so the user
        can ``tail -f`` for diagnostics without breaking the REPL.

    Headless mode (``_configure_logging`` path) is unchanged.

    Args:
      level: Optional CLI log level; overrides ``SAGENT_LOG_LEVEL``.
      session_dir: Session directory containing ``session.jsonl``.

    """
    logging.lastResort = logging.NullHandler()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and getattr(
            cast(object, handler), "stream", None
        ) in (sys.stderr, sys.stdout):
            root.removeHandler(cast(logging.Handler, handler))

    raw = level or os.environ.get("SAGENT_LOG_LEVEL") or "DEBUG"
    name = raw.upper()
    value = getattr(logging, name, None)
    if not isinstance(value, int):
        # ``_configure_logging`` runs first and would have already
        # rejected an invalid level; reaching here means a bad value
        # was set after that point. Be quiet (REPL has no good
        # place to surface this) and fall back to NullHandler-only.
        return
    log_file = os.environ.get("SAGENT_LOG_FILE")
    if log_file is None:
        base = Path(session_dir) if session_dir is not None else Path.home() / ".sagent"
        log_path = base / "repl.log"
    else:
        log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    root.addHandler(file_handler)
    logging.getLogger("sagent").setLevel(value)
    logging.getLogger("sagent").setLevel(value)


async def _with_signals(
    agent: Agent,
    coro: Coroutine[object, object, None],
) -> None:
    """Install SIGINT/SIGTERM handlers around ``coro`` for graceful + escape exit.

    First signal: push ``Quit()`` to ``agent.inbox`` so the runtime
    drains cleanly. Second signal: ``os._exit(1)``.
    """
    loop = asyncio.get_running_loop()
    handler = _quit_handler(agent)
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handler)
    try:
        await coro
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                _ = loop.remove_signal_handler(sig)


def _parse_stream_json(raw: str) -> str:
    """Parse NDJSON stdin: each non-empty line is ``{"prompt": "..."}``."""
    prompts: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj: object = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON line in stream-json input: {e}") from e
        if not isinstance(obj, dict):
            raise TypeError("stream-json input requires JSON objects per line.")
        parsed = json_unfreeze(cast(Mapping[str, object], obj))
        p = parsed.get("prompt")
        if isinstance(p, str) and p:
            prompts.append(p)
    return "\n\n".join(prompts)


def _event_to_json_record(event: types.runtime.RuntimeEvent) -> MutableJSON | None:
    """Serialize a ``RuntimeEvent`` for stream-json output, or skip."""
    if isinstance(event, types.runtime.ModelResponsePartial):
        return {"descriptor": "text/plain", "content": event.text}
    if isinstance(event, types.runtime.ModelResponseThinking):
        return {"descriptor": "text/x-thinking", "content": event.text}
    if isinstance(event, types.runtime.ToolLabel):
        return {
            "descriptor": "text/x-tool-label",
            "content": event.text,
            "call_id": event.call_id,
        }
    if isinstance(event, types.history.ToolResult):
        return {
            "descriptor": "application/x-tool-result",
            "call_id": event.call_id,
            "content": event.content,
            "is_error": event.is_error,
        }
    if isinstance(event, types.runtime.ModelResponseError):
        exc = event.exception
        return {
            "descriptor": "application/x-error",
            "content": f"{type(exc).__name__}: {exc}",
        }
    return None


async def _run_headless(
    agent: Agent,
    *,
    input_format: str,
    output_format: str,
) -> None:
    """Non-interactive execution for piped/scripted usage.

    Reads stdin via ``asyncio.to_thread`` so the asyncio event loop
    keeps iterating while the read is blocked. The asyncio signal
    handler from :func:`_with_signals` is suspended for the duration
    of the read because ``to_thread`` cannot be cancelled mid-read --
    Python's default SIGINT handler (KeyboardInterrupt) is what gets
    the user out.
    """
    loop = asyncio.get_running_loop()
    suspended: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(sig)
            suspended.append(sig)
    try:
        raw = await asyncio.to_thread(sys.stdin.read)
    except KeyboardInterrupt:
        # Unix convention: signal N → exit code 128 + N. SIGINT = 130.
        sys.stderr.write("Interrupted before input was provided.\n")
        sys.exit(130)
    finally:
        for sig in suspended:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _quit_handler(agent))
    if input_format == "stream-json":
        try:
            prompt = _parse_stream_json(raw)
        except (ValueError, TypeError) as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.exit(1)
    else:
        prompt = raw.strip()
    if not prompt:
        sys.stderr.write("Error: no input on stdin.\n")
        sys.exit(1)

    user_msg = types.history.UserMessage(text=prompt)
    if output_format == "stream-json":
        async for event in agent.run(user_msg):
            record = _event_to_json_record(event)
            if record is None:
                continue
            json.dump(record, sys.stdout)
            sys.stdout.write("\n")
    else:
        async for _event in agent.run(user_msg):
            pass
    result_text = _last_assistant_text(agent.history)
    if output_format == "stream-json":
        json.dump({"content": result_text, "descriptor": "result"}, sys.stdout)
        sys.stdout.write("\n")
    elif output_format == "json":
        json.dump({"content": result_text}, sys.stdout)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(result_text)
        sys.stdout.write("\n")


def _last_assistant_text(history: list[types.history.HistoryEntry]) -> str:
    """Return the text from the most recent ``AssistantMessage`` in ``history``."""
    for entry in reversed(history):
        if isinstance(entry, types.history.AssistantMessage):
            return entry.text
    return ""


def _quit_handler(agent: Agent) -> Callable[[], None]:
    """Return a signal handler that pushes ``Quit`` once, then exits."""
    triggered = False

    def _on_signal() -> None:
        nonlocal triggered
        if triggered:
            os._exit(1)
        triggered = True
        agent.runtime.inbox.push_back(types.runtime.Quit())

    return _on_signal


def _install_session_persistence(agent: Agent, session_dir: Path) -> None:
    """Attach a ``SaveSession`` observer that appends history deltas to disk.

    Re-writes ``meta`` whenever ``agent.status`` changes (via the
    ``StatusChanged`` event), even when there's no history delta, so a
    status update survives a crash before the next history append.

    Tracks in-place splice updates from ``HistoryEntryUpdated`` events
    (emitted when ``DetachedResult`` splices real tool output into a
    ``[detached]`` placeholder). Splice updates are re-emitted as
    ``kind=history`` records sharing the same ``id``; the loader dedupes
    by ``id``, last write wins, so the spliced content survives session
    resume.
    """
    persisted_len = len(agent.history)
    meta_written = False
    last_status = agent.status
    pending_updates: dict[int, types.history.ToolResult] = {}

    def _on_event(event: types.runtime.RuntimeEvent) -> None:
        nonlocal persisted_len, meta_written, last_status
        if isinstance(event, types.runtime.HistoryEntryUpdated):
            if isinstance(event.entry, types.history.ToolResult):
                pending_updates[event.entry.id] = event.entry
            return
        if not isinstance(
            event, (types.runtime.SaveSession, types.runtime.StatusChanged)
        ):
            return
        delta = agent.history[persisted_len:]
        # Drop pending updates whose id is in the delta -- the delta
        # already carries the latest content. Keep the rest as separate
        # ``kind=update`` records (patches against entries that were
        # already persisted).
        delta_ids = {e.id for e in delta}
        updates = [upd for uid, upd in pending_updates.items() if uid not in delta_ids]
        pending_updates.clear()
        status_changed = agent.status != last_status
        write_meta = delta or updates or status_changed or not meta_written
        spec = agent.model_spec
        meta = SessionMeta(
            session_id=agent.session_id,
            model_id=agent.model.model_id,
            provider=spec.provider if spec else "",
            auth=spec.auth if spec else "",
            account=(spec.account or "") if spec else "",
            name=agent.name,
            status=agent.status,
            tokens=agent.total_tokens,
            total_cost_usd=agent.total_cost_usd,
            num_tool_call_rounds=agent.num_tool_call_rounds,
            compact_count=agent.compaction_state.compact_count,
            bash_cwd=agent.tool_state.bash_cwd,
            total_active_elapsed_seconds=agent.activity.elapsed_seconds,
        )
        append_session(
            session_dir / "session.jsonl",
            meta=meta.serialize() if write_meta else None,
            history_delta=delta or None,
            history_updates=updates or None,
            tool_state_snapshot=serialize_tool_state(agent.tool_state),
        )
        persisted_len = len(agent.history)
        meta_written = True
        last_status = agent.status

    agent.runtime.observers.append(_on_event)


def main() -> None:
    """Parse args and launch the REPL or headless runner."""
    parser = argparse.ArgumentParser(
        description="CLI agent (REPL or headless).",
        epilog=(
            "modes:\n"
            "  tty stdin       interactive REPL\n"
            "  non-tty stdin   headless one-shot (read prompt from stdin to EOF)\n"
            "\n"
            "examples:\n"
            "  sagent                                  # REPL\n"
            "  echo 'fix the bug' | sagent             # headless, text in, text out\n"
            "  sagent < prompt.txt                     # headless, text from file\n"
            "  sagent --output-format json < p.txt     # text in, JSON result out\n"
            "  sagent --input-format stream-json \\\n"
            "         --output-format stream-json \\\n"
            "         < prompts.ndjson                 # NDJSON in, NDJSON events out\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args, remaining = _parse_cli_args(parser)
    if remaining == ["login"]:
        _do_login(args)
        return
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    _configure_logging(args.log_level)
    if args.recipe is not None:
        set_recipe(args.recipe)
    session_dir = None if args.no_session else _resolve_session_dir(args)
    loaded_session = None
    if session_dir is not None:
        loaded_session = load_session(Path(session_dir), {})
        if loaded_session is not None:
            _apply_resume_model_defaults(args, loaded_session[0])
    try:
        provider, model, resolved_auth = _build_provider_model(args)
    except (AttributeError, RuntimeError, ValueError) as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    model_spec = types.model.ModelSpec(
        provider=args.provider,
        auth=resolved_auth,
        model_id=model.model_id,
        account=args.account,
    )
    if loaded_session is not None:
        meta, history, tool_state = loaded_session
        loaded_session = (
            dataclasses.replace(
                meta,
                provider=model_spec.provider,
                auth=model_spec.auth,
                model_id=model_spec.model_id,
                account=model_spec.account or "",
            ),
            history,
            tool_state,
        )
    compactor = SummaryCompactor() if args.compact else None

    headless = not sys.stdin.isatty()
    if not headless:
        sys.stderr.write(f"[{args.provider}] {model.model_id}\n")

    tool_names = args.tools or DEFAULT_TOOLS
    agent_tools = resolve_tools(tool_names)
    if args.advisor:
        advisor_model = provider.model(args.advisor)
        agent_tools.append(
            Advisor(model=advisor_model, max_uses=args.advisor_max_uses),
        )
        if not headless:
            sys.stderr.write(f"[advisor] {advisor_model.model_id}\n")

    custom_system = args.system

    def _system() -> str:
        return build_system(
            model.model_id,
            custom=custom_system,
            include_memory=not args.no_session,
        )

    agent = Agent(
        name=args.name,
        description="Interactive CLI agent.",
        model=model,
        model_spec=model_spec,
        system=_system,
        tools=agent_tools,
        compactor=compactor,
        session_dir=session_dir,
        effort=args.effort,
        max_tool_call_rounds=args.max_tool_call_rounds,
        max_budget_usd=args.max_budget_usd,
    )
    if args.max_request_tokens is not None:
        agent.max_request_tokens = args.max_request_tokens
    if args.max_response_tokens is not None:
        agent.max_response_tokens = args.max_response_tokens

    if loaded_session is not None:
        agent.resume(*loaded_session)
    if session_dir is not None:
        _install_session_persistence(agent, Path(session_dir))

    agent.tool_state.additional_dirs = list(args.add_dir)

    if not headless:
        if args.output_format != "text":
            sys.stderr.write(
                "Note: --output-format is ignored in interactive REPL mode.\n"
            )
        _install_repl_logging(args.log_level, session_dir=session_dir)
        asyncio.run(
            _with_signals(
                agent,
                run_repl(agent, history=args.history),
            ),
        )
    else:
        asyncio.run(
            _with_signals(
                agent,
                _run_headless(
                    agent,
                    input_format=args.input_format,
                    output_format=args.output_format,
                ),
            )
        )


def _do_login(args: argparse.Namespace) -> None:
    """Run the OAuth flow for ``args.provider`` and save under ``args.account``."""
    cls = getattr(providers, args.provider, None)
    if cls is None:
        sys.stderr.write(f"Error: unknown provider {args.provider!r}\n")
        sys.exit(1)
    login_fn = getattr(cls, "login", None)
    save_fn = getattr(cls, "save", None)
    if login_fn is None or save_fn is None:
        sys.stderr.write(
            f"Error: {args.provider} does not support interactive login.\n"
        )
        sys.exit(1)
    account = args.account or "default"
    sys.stderr.write(f"[login] provider={args.provider} account={account!r}\n")
    creds = login_fn(output=sys.stderr, account=args.account, manual=args.headless)
    save_fn(creds, account=args.account)
    sys.stderr.write(f"[login] saved credentials for account '{account}'.\n")


if __name__ == "__main__":
    main()
# vim: ft=python
