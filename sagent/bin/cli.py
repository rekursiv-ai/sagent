#!/bin/sh
# ruff: noqa: EXE003, D300  -- Polyglot: #!/bin/sh + triple-single-quotes are intentional.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen python3 "$0" "$@"
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

from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import cast

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import sys

from sagent import providers, sessions, tools
from sagent.agent import Agent
from sagent.compactor import SummaryCompactor
from sagent.custom_types import (
    ErrorEvent,
    InterruptedEvent,
    Message,
    Model,
    ModelSpec,
    Provider,
    StatusUpdateEvent,
    TextChunkEvent,
    TextMessage,
    ThinkingEvent,
    Tool,
    ToolLabelEvent,
    ToolResultEvent,
)
from sagent.lib.descriptors import QUIT_SENTINEL
from sagent.lib.json import json_unfreeze
from sagent.lib.message import response_text
from sagent.prompt import build_system_dict
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


def resolve_tools(names: list[str]) -> list[Tool]:
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
    # First pass: instantiate every non-Bash tool.
    non_bash: dict[str, Tool] = {}
    for name in names:
        if name == "Bash":
            continue
        cls = getattr(tools, name, None)
        if cls is None:
            raise SystemExit(f"unknown tool: {name!r}")
        non_bash[name] = cls()
    # Second pass: assemble in the order requested, wiring Bash peers.
    resolved: list[Tool] = []
    peers = tuple(non_bash.values())
    for name in names:
        if name == "Bash":
            resolved.append(tools.Bash(peers=peers))
        else:
            resolved.append(non_bash[name])
    return resolved


def _resolve_session_dir(args: argparse.Namespace) -> str | None:
    """Pick the session directory per --session / --resume / --continue.

    Precedence:
    - ``--session <path>`` wins, always (explicit user intent).
    - ``--continue`` → most recent session for cwd, or a new one if none.
    - ``--resume``   → interactive picker, or a new session on abort.
    - Otherwise: a new session dir under the default project root.
    """
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
    all_sessions = sessions.list_all_sessions()
    if all_sessions:
        sys.stderr.write(f"[resume] {all_sessions[0].path}\n")
        return str(all_sessions[0].path)
    sys.stderr.write("[resume] no prior sessions; starting fresh.\n")
    return str(sessions.new_session_dir(Path.cwd()))


def _resolve_resume_all() -> str:
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
    """Add shared agent flags and parse.

    Leaf of the parse cascade: adds LLM/tool/compaction/effort flags
    and calls ``parse_known_args``.  Callers add their own flags
    first, then delegate here.
    """
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
    return parser.parse_known_args(argv)


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
        help="Enable stderr logging at this level. Overrides SAGENT_LOG_LEVEL.",
    )
    return parse_agent_args(parser, argv)


def _build_provider_model(
    args: argparse.Namespace,
) -> tuple[Provider, Model, str]:
    """Build the provider/model pair requested by CLI flags.

    ``SelfHosted`` is path-backed: ``--model /path/to/hf-snapshot`` is the
    model load path, so the provider must be built from that path before
    ``provider.model(...)`` can validate the bound model id.
    """
    auth = str(args.auth)
    model_id = cast(str | None, args.model)
    model_lookup = model_id
    if args.provider == "SelfHosted":
        auth = model_id or "env"
        model_lookup = None
    provider = build_provider(str(args.provider), auth, account=args.account)
    model = provider.model(model_lookup)
    return provider, model, auth


def _configure_logging(level: str | None) -> None:
    """Configure CLI logging from flag or environment."""
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


async def _with_signals(
    agent: Agent,
    coro: Coroutine[object, object, None],
) -> None:
    """Install SIGINT/SIGTERM handlers around ``coro`` for graceful + escape exit.

    First signal: post ``text/x-quit`` to ``agent.inbox`` and let the
    dispatch loop drain on its own. Mirrors typing ``/quit`` from the
    REPL but works when prompt-toolkit isn't capturing input (between
    turns, during shutdown drain).

    Second signal: ``os._exit(1)``. Skips ``atexit`` cleanup so a sync
    tool wedged in ``asyncio.to_thread`` can't block process exit on a
    ``ThreadPoolExecutor.join``. The user already asked twice; honor it.

    Headless note: the asyncio handler is suspended inside
    :func:`_run_headless` while the initial stdin read is blocked --
    ``to_thread`` cannot be cancelled mid-read, so during that window
    the default Python KeyboardInterrupt is what gets the user out.

    Modeled on ``loop/experimental/switchboard/hub.py:828``.
    """
    loop = asyncio.get_running_loop()
    handler = _quit_handler(agent)
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows / non-mainthread runners don't support add_signal_handler.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handler)
    try:
        await coro
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                _ = loop.remove_signal_handler(sig)


def _parse_stream_json(raw: str) -> str:
    """Parse NDJSON stdin: each non-empty line is ``{"prompt": "..."}``.

    Joins prompt fields with blank lines. Raises ``ValueError`` on
    unparseable JSON or ``TypeError`` on the wrong shape, so programmatic
    clients see a hard failure instead of silently wrong behavior; the
    caller maps that to an exit code.
    """
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
        # cast: isinstance narrows to dict but can't parameterize key type.
        # JSON deserialization guarantees str keys, so this assumption holds.
        parsed = json_unfreeze(cast(Mapping[str, object], obj))
        p = parsed.get("prompt")
        if isinstance(p, str) and p:
            prompts.append(p)
    return "\n\n".join(prompts)


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
    the user out. Once the prompt is read and the agent run starts,
    the asyncio handlers take over and the first signal posts a
    graceful ``text/x-quit``.
    """
    loop = asyncio.get_running_loop()
    # Suspend the asyncio signal handler so SIGINT during the blocked
    # stdin read raises KeyboardInterrupt instead of being swallowed
    # by ``add_signal_handler`` (which would post QUIT to an inbox no
    # one reads pre-prompt). Restored before agent.run() so the
    # graceful path is active during the actual work.
    suspended: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.remove_signal_handler(sig)
            suspended.append(sig)
    try:
        raw = await asyncio.to_thread(sys.stdin.read)
    except KeyboardInterrupt:
        # Unix convention: a process killed by signal N exits with
        # ``128 + N``. SIGINT is signal 2, so 130. Shells interpret
        # this as "interrupted by Ctrl+C" -- scripts wrapping the
        # headless mode see the same exit code as any other
        # interrupted command.
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

    user_msg = TextMessage(prompt, "text/x-user-message")
    if output_format == "stream-json":
        async for event in agent.run(user_msg):
            descriptor, content = _event_to_jsonable(event)
            if descriptor is None:
                continue
            json.dump({"descriptor": descriptor, "content": content}, sys.stdout)
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


def _event_to_jsonable(event: object) -> tuple[str | None, str]:
    """Translate one ``Event`` into a stream-json (descriptor, content) pair."""
    if isinstance(event, TextChunkEvent):
        return "text/plain", event.text
    if isinstance(event, ThinkingEvent):
        return "text/x-thinking", event.text
    if isinstance(event, ToolLabelEvent):
        return "text/x-tool-label", event.text
    if isinstance(event, ToolResultEvent):
        return "multipart/x-tool-result", str(event.msg.content)
    if isinstance(event, ErrorEvent):
        return "text/x-error", event.text
    if isinstance(event, InterruptedEvent):
        return "text/x-interrupted", ""
    if isinstance(event, StatusUpdateEvent):
        return "text/x-status-update", event.text
    return None, ""


def _last_assistant_text(history: Sequence[Message]) -> str:
    """Return the final text from the last assistant message in ``history``."""
    for msg in reversed(history):
        if msg.descriptor == "multipart/x-model-message":
            return response_text(msg)
    return ""


def _quit_handler(agent: Agent) -> Callable[[], None]:
    """Return a signal handler that posts ``text/x-quit`` once, then exits.

    Mirrors :func:`_with_signals`' two-strikes contract for
    re-installation after a signal-handler suspension.
    """
    triggered = False

    def _on_signal() -> None:
        nonlocal triggered
        if triggered:
            os._exit(1)
        triggered = True
        _ = agent.inbox.put_left(TextMessage("", QUIT_SENTINEL))

    return _on_signal


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
    try:
        provider, model, resolved_auth = _build_provider_model(args)
    except (AttributeError, RuntimeError, ValueError) as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    model_spec = ModelSpec(
        provider=args.provider,
        auth=resolved_auth,
        model_id=model.model_id,
        account=args.account,
    )
    compactor = SummaryCompactor() if args.compact else None

    headless = not sys.stdin.isatty()
    if not headless:
        sys.stderr.write(f"[{args.provider}] {model.model_id}\n")

    session_dir = None if args.no_session else _resolve_session_dir(args)

    tool_names = args.tools or DEFAULT_TOOLS
    agent_tools = resolve_tools(tool_names)
    if args.advisor:
        advisor_model = provider.model(args.advisor)
        agent_tools.append(
            Advisor(model=advisor_model, max_uses=args.advisor_max_uses),
        )
        if not headless:
            sys.stderr.write(f"[advisor] {advisor_model.model_id}\n")

    agent = Agent(
        name=args.name,
        description="Interactive CLI agent.",
        model=model,
        model_spec=model_spec,
        system=build_system_dict(
            model.model_id,
            custom=args.system,
            include_memory=not args.no_session,
        ),
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
    agent.tool_state.additional_dirs = list(args.add_dir)
    if not headless:
        if args.output_format != "text":
            sys.stderr.write(
                "Note: --output-format is ignored in interactive REPL mode.\n"
            )
        asyncio.run(_with_signals(agent, run_repl(agent, history=args.history)))
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
    """Run the OAuth flow for ``args.provider`` and save under ``args.account``.

    Fails fast if the provider class doesn't expose both ``login``
    and ``save`` classmethods. Output of the login flow goes to
    stderr so pipes and scripts can capture them cleanly.
    """
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
