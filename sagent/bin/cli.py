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

    # Claude subscription (reuses `claude auth login --claudeai`)
    ./cli.py --provider AnthropicCLI

    # OpenAI
    ./cli.py --provider OpenAI --model gpt-5.6-sol

    # ChatGPT subscription (reuses `codex login`)
    ./cli.py --provider OpenAISubscription --model gpt-5.6-sol

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
from typing import Final, cast

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shlex
import signal
import sys
import time

from sagent import (
    providers,
    sessions,
    tools,
    types,
)
from sagent.agent import Agent
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.session_io import (
    PersistentAgentRecord,
    SessionMeta,
    load_persistent_agents,
    load_session,
)
from sagent.agent.state import agent_registry, unique_registry_label
from sagent.compaction.summary import SummaryCompactor
from sagent.lib.custom_json import MutableJSON, json_unfreeze
from sagent.lib.userdirs import data_dir
from sagent.prompt import build_system
from sagent.providers import (
    PROVIDER_NAMES,
    build_provider,
    default_auth_for_provider,
    supported_provider_options,
)
from sagent.repl import run_repl
from sagent.thinking import (
    THINKING_COMMANDS,
    THINKING_STATES,
    ThinkingState,
    resolve_thinking_command,
    should_redact_thinking,
)
from sagent.tools.advisor import Advisor
from sagent.tools.agent_spawn import (
    _augment_system_for_persistent,
    _build_forwarder,
)
from sagent.tools.core import set_recipe


_DEFAULT_PROVIDER = "Anthropic"
_DEFAULT_AUTH = "env"
_PROVIDER_STARTUP_ERRORS = (FileNotFoundError, RuntimeError, ValueError)


def _default_allow_providers() -> tuple[str, ...]:
    """Default allow-list, led by ``_DEFAULT_PROVIDER``.

    ``--provider`` defaults to the first allowed provider, so the lead
    entry is the zero-flag default; the rest follow in declaration order.
    """
    rest = tuple(p for p in PROVIDER_NAMES if p != _DEFAULT_PROVIDER)
    return (_DEFAULT_PROVIDER, *rest)


DEFAULT_TOOLS: Final = [
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


def resolve_tools(
    names: list[str],
    *,
    allow_providers: tuple[str, ...] | None = None,
) -> list[types.tools.Tool]:
    """Instantiate tools by class name from the ``tools`` module.

    Bash is constructed last with ``peers=`` set to its sibling tools
    so its ``[bash-lint]`` feature can suggest dedicated replacements
    for invocations like ``grep``/``cat``/``find``/``sed -i``.

    Args:
      names: Tool class names to look up in the tools submodule.
      allow_providers: Optional provider allow-list forwarded to
        ``AgentSpawn`` and ``AgentSelf`` so their schemas and catalogs
        only enumerate providers the host can actually use. ``None``
        means "every provider in ``sagent.providers``".

    Returns:
      tools: Instantiated tool objects in the requested order.

    Raises:
      SystemExit: If a tool name is not found in the tools module.

    """
    if names == ["none"]:
        return []
    provider_aware = {"AgentSpawn", "AgentSelf"}
    non_bash: dict[str, types.tools.Tool] = {}
    for name in names:
        if name == "Bash":
            continue
        cls = getattr(tools, name, None)
        if cls is None:
            raise SystemExit(f"unknown tool: {name!r}")
        if name in provider_aware:
            non_bash[name] = cls(allow_providers=allow_providers)
        else:
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
            "Provider class name from ``sagent.providers``. Default: the"
            " first entry of ``--allow-providers``"
            f" (``{_DEFAULT_PROVIDER}`` unless narrowed)."
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
        "--allow-providers",
        default=os.environ.get(
            "SAGENT_ALLOW_PROVIDERS", ",".join(_default_allow_providers())
        ),
        metavar="LIST",
        help=(
            "Comma-separated provider names this agent and its spawned"
            " children may use. Reads ``SAGENT_ALLOW_PROVIDERS`` env var"
            " as fallback; default if neither is set is every provider"
            " in ``sagent.providers``. Default: %(default)s."
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
        help=(
            "Model ID, or a symbolic tier resolved per provider: 'default'"
            " (the provider's default model) or 'utility' (its cheapest/fastest"
            " model for summarizers and other internal tasks). Append +1m / +200k"
            " to set window."
        ),
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
        "--server-side-context-management",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Opt in to the provider's server-side context management"
            " (Anthropic's clear_tool_uses beta). Default: the provider's"
            " own default (off). Rejected at startup by providers that do"
            " not support it."
        ),
    )
    parser.add_argument(
        "--thinking",
        default="default",
        choices=("default", *THINKING_COMMANDS),
        help=(
            "Thinking state or partial command. Full states: adaptive-show,"
            " adaptive-hide, on-show, on-hide, off-hide, redact-hide."
            " Partials: adaptive, on, off, redact, show, hide."
            " Default: no Agent override."
        ),
    )
    parser.add_argument(
        "--effort",
        default=None,
        help=(
            "Provider-specific reasoning effort"
            " (none|minimal|low|medium|high|xhigh|max)."
            " Unset = provider default; unsupported values raise."
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
        "--ephemeral",
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
        "--resume-persistent",
        dest="resume_persistent",
        action="store_true",
        default=True,
        help="Resume live persistent subagents recorded in the session (default).",
    )
    parser.add_argument(
        "--no-resume-persistent",
        dest="resume_persistent",
        action="store_false",
        help="Do not restart persistent subagents when resuming a session.",
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


def _parse_allow_providers(spec: str) -> tuple[str, ...]:
    """Parse ``--allow-providers`` / ``SAGENT_ALLOW_PROVIDERS`` CSV.

    Exits with a clear error on empty input or unknown provider names.
    """
    parsed = tuple(p.strip() for p in spec.split(",") if p.strip())
    if not parsed:
        sys.stderr.write(
            "Error: --allow-providers requires at least one provider name;"
            f" valid: {list(PROVIDER_NAMES)}\n"
        )
        sys.exit(1)
    unknown = [p for p in parsed if p not in PROVIDER_NAMES]
    if unknown:
        sys.stderr.write(
            f"Error: --allow-providers contains unknown: {unknown};"
            f" valid: {list(PROVIDER_NAMES)}\n"
        )
        sys.exit(1)
    return parsed


def _resolve_provider_and_allow(
    spec: str,
    *,
    primary: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve the provider and its allow-list together from ``spec``.

    ``primary`` is the provider when the user passed ``--provider`` or a
    resumed session pinned one; ``None`` means "use the default", which
    is the first allowed provider. An explicit provider is unioned into
    the allow-set so the caller need not name it twice. Unknown or empty
    ``spec`` exits via :func:`_parse_allow_providers`.
    """
    parsed = _parse_allow_providers(spec)
    if primary is None:
        return parsed[0], parsed
    if primary in parsed:
        return primary, parsed
    # Route the union back through ``_parse_allow_providers`` so an
    # unknown ``primary`` surfaces the same error/exit as any other
    # unknown CSV entry.
    return primary, _parse_allow_providers(f"{spec},{primary}")


def _build_provider_model(
    args: argparse.Namespace,
    thinking_state: ThinkingState | None,
    *,
    allow_providers: tuple[str, ...] = PROVIDER_NAMES,
) -> tuple[types.providers.Provider, types.model.Model, str]:
    """Build the provider/model pair requested by CLI flags."""
    try:
        return _build_provider_model_once(args, thinking_state)
    except _PROVIDER_STARTUP_ERRORS as error:
        return _build_provider_model_fallback(
            args,
            thinking_state,
            error,
            allow_providers=allow_providers,
        )


def _build_provider_model_once(
    args: argparse.Namespace,
    thinking_state: ThinkingState | None,
) -> tuple[types.providers.Provider, types.model.Model, str]:
    """Build one provider/model pair without fallback."""
    provider_name = str(args.provider)
    auth = str(args.auth)
    if not bool(getattr(args, "auth_explicit", False)):
        auth = default_auth_for_provider(provider_name)
    model_id = cast(str | None, args.model)
    # SelfHosted encodes the auth (a local snapshot path) in ``--model`` and has
    # no symbolic tiers, so it always resolves to the provider's default model.
    if args.provider == "SelfHosted":
        auth = model_id or "env"
        model_id = None
    options = _cli_provider_options(args)
    if thinking_state is not None and "redact_thinking" in (
        supported_provider_options(provider_name)
    ):
        options = dataclasses.replace(
            options,
            redact_thinking=should_redact_thinking(thinking_state),
        )
    provider = build_provider(
        provider_name,
        auth,
        account=args.account,
        options=options,
    )
    if model_id == "utility":
        model = provider.utility_model()
    else:
        model = provider.model(None if model_id == "default" else model_id)
    return provider, model, auth


def _build_provider_model_fallback(
    args: argparse.Namespace,
    thinking_state: ThinkingState | None,
    error: Exception,
    *,
    allow_providers: tuple[str, ...],
) -> tuple[types.providers.Provider, types.model.Model, str]:
    """Try another subscription provider for implicit startup auth failures."""
    if not _allow_implicit_provider_fallback(args):
        raise RuntimeError(
            _credential_error_message(
                str(args.provider),
                error,
                allow_providers=allow_providers,
                account=args.account,
            )
        ) from error
    original_provider = str(args.provider)
    for fallback_provider in _credential_fallback_providers(
        original_provider,
        allow_providers=allow_providers,
    ):
        args.provider = fallback_provider
        args.auth = "credentials"
        args.model = None
        try:
            provider, model, auth = _build_provider_model_once(args, thinking_state)
        except _PROVIDER_STARTUP_ERRORS:
            continue
        sys.stderr.write(
            f"[provider] {original_provider} unavailable: {error}\n"
            f"[provider] falling back to {fallback_provider} ({model.model_id}).\n"
            f"[provider] To use {original_provider}, run: "
            f"sagent --provider {original_provider} login\n"
        )
        return provider, model, auth
    args.provider = original_provider
    raise RuntimeError(
        _credential_error_message(
            original_provider,
            error,
            allow_providers=allow_providers,
            account=args.account,
        )
    ) from error


def _allow_implicit_provider_fallback(args: argparse.Namespace) -> bool:
    """Return whether startup may change provider instead of failing."""
    return not any(
        bool(getattr(args, name, False))
        for name in (
            "provider_explicit",
            "provider_from_resume",
            "auth_explicit",
            "account_explicit",
            "model_explicit",
        )
    )


def _credential_fallback_providers(
    provider_name: str,
    *,
    allow_providers: tuple[str, ...] = PROVIDER_NAMES,
) -> tuple[str, ...]:
    """Return implicit subscription providers to try after ``provider_name``."""
    candidates = (
        "AnthropicCLI",
        "OpenAISubscription",
    )
    return tuple(
        candidate
        for candidate in candidates
        if candidate != provider_name and candidate in allow_providers
    )


def _credential_setup_commands(
    provider_name: str,
    *,
    account: str | None = None,
) -> tuple[str, ...]:
    """Return valid setup commands for one provider without inventing login APIs."""
    account_name = account or "default"
    account_args = (
        "" if account_name == "default" else f" --account {shlex.quote(account_name)}"
    )
    if provider_name == "AnthropicCLI":
        if account_args:
            # Native Claude login writes only its default credential store;
            # named AnthropicCLI accounts are legacy file slots and the
            # provider error names their required path.
            return ()
        return ("claude auth login --claudeai", "sagent --provider AnthropicCLI")
    if provider_name == "OpenAISubscription":
        login = f"sagent --provider OpenAISubscription{account_args} login"
        run = f"sagent --provider OpenAISubscription{account_args}"
        if account_args:
            return (login, run)
        return ("codex login", login, run)

    cls = getattr(providers, provider_name, None)
    env_var = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
    }.get(
        provider_name,
        getattr(cls, "ENV_VAR", None),
    )
    if isinstance(env_var, str) and env_var:
        return (f"export {env_var}=...",)
    return ()


def _credential_error_message(
    provider_name: str,
    error: Exception,
    *,
    allow_providers: tuple[str, ...] = PROVIDER_NAMES,
    account: str | None = None,
) -> str:
    """Return an actionable credential-startup error."""
    commands = list(_credential_setup_commands(provider_name, account=account))
    for fallback_provider in _credential_fallback_providers(
        provider_name,
        allow_providers=allow_providers,
    ):
        commands.extend(
            _credential_setup_commands(
                fallback_provider,
                account=account,
            )
        )
    commands = list(dict.fromkeys(commands))
    lines = [f"{provider_name} credentials are unavailable: {error}"]
    if commands:
        lines.append("Try one of:")
        lines.extend(f"  {command}" for command in commands)
    return "\n".join(lines)


def _cli_provider_options(args: argparse.Namespace) -> types.providers.ProviderOptions:
    """Return construction-time provider options from explicit CLI flags."""
    return types.providers.ProviderOptions(
        server_side_context_management=cast(
            "bool | None", args.server_side_context_management
        ),
    )


def _resolve_cli_thinking_state(args: argparse.Namespace) -> ThinkingState | None:
    """Resolve the ``--thinking`` flag to an Agent-level state override."""
    raw = str(args.thinking)
    if raw == "default":
        return None
    return resolve_thinking_command(raw)


def _validate_cli_thinking_state(
    model: types.model.Model,
    state: ThinkingState | None,
) -> None:
    """Validate a resolved CLI thinking state against model/provider support."""
    if state is None:
        return
    valid = model.valid_thinking_states
    if state not in valid:
        options = ", ".join(valid)
        raise ValueError(
            f"thinking state {state!r} not supported by {model.model_id!r};"
            f" options: {options}"
        )


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
    """Return True when the named provider's catalog includes ``model_id``.

    Mirrors the providers' profile-lookup rule: latency tags (``+fast``)
    ride on catalog ids and are stripped before the membership check,
    while context tags stay -- ``+1m`` variants are catalog keys where
    supported and must keep failing the check elsewhere.
    """
    cls = getattr(providers, provider_name, None)
    if cls is None:
        return False
    known = getattr(cls, "KNOWN_MODELS", None)
    if not isinstance(known, dict):
        return False
    return model_id in known or types.model.strip_latency_tags(model_id) in known


async def _resume_persistent_agents(
    parent: Agent,
    session_dir: Path,
    *,
    allow_providers: tuple[str, ...],
    parent_label: str,
) -> None:
    """Restart persistent subagents recorded as running in ``session_dir``."""
    records = load_persistent_agents(session_dir)
    for record in records:
        if record.provider not in allow_providers:
            sys.stderr.write(
                f"[resume-persistent] skipping {record.label!r}:"
                f" provider {record.provider!r} is not allowed.\n"
            )
            continue
        try:
            child = _build_persistent_child(
                record,
                allow_providers=allow_providers,
                parent_label=parent_label,
            )
        except (RuntimeError, ValueError) as e:
            sys.stderr.write(f"[resume-persistent] skipping {record.label!r}: {e}\n")
            continue
        if not record.session_dir:
            sys.stderr.write(
                f"[resume-persistent] skipping {record.label!r}: missing session_dir.\n"
            )
            continue
        loaded_child = load_session(Path(record.session_dir), {})
        if loaded_child is not None:
            child.resume(*loaded_child)
        label = _resume_label(record.label)
        if label != record.label:
            sys.stderr.write(
                f"[resume-persistent] label {record.label!r} already active;"
                f" restored as {label!r}.\n"
            )
        _start_resumed_persistent(parent, child, record, label)


def _build_persistent_child(
    record: PersistentAgentRecord,
    *,
    allow_providers: tuple[str, ...],
    parent_label: str,
) -> Agent:
    """Construct a persistent child from its lifecycle record."""
    thinking_state = _persistent_thinking_state(record.thinking_state)
    options = record.provider_options
    if thinking_state is not None and "redact_thinking" in (
        supported_provider_options(record.provider)
    ):
        options = dataclasses.replace(
            options,
            redact_thinking=should_redact_thinking(thinking_state),
        )
    provider = build_provider(
        record.provider,
        record.auth,
        account=record.account or None,
        options=options,
    )
    model = provider.model(record.model_id)
    agent = Agent(
        name=record.label,
        model=model,
        model_spec=types.model.ModelSpec(
            provider=record.provider,
            auth=record.auth,
            model_id=model.model_id,
            account=record.account,
        ),
        system=_augment_system_for_persistent(record.system, parent_label=parent_label),
        tools=resolve_tools(list(record.tools), allow_providers=allow_providers),
        session_dir=record.session_dir,
        max_tool_call_rounds=record.max_tool_call_rounds,
        thinking=record.thinking,
        thinking_state=thinking_state,
        effort=record.effort,
        max_budget_usd=record.max_budget_usd,
        persistent_retry=record.persistent_retry,
        provider_options=record.provider_options,
    )
    if record.max_request_tokens is not None:
        agent.max_request_tokens = record.max_request_tokens
    if record.max_response_tokens is not None:
        agent.max_response_tokens = record.max_response_tokens
    if record.service_tier is not None:
        agent.service_tier = record.service_tier
    agent.cache_ttl = record.cache_ttl
    return agent


def _persistent_thinking_state(raw: str | None) -> ThinkingState | None:
    """Narrow a persisted thinking-state string to ``ThinkingState``."""
    for state in THINKING_STATES:
        if raw == state:
            return state
    return None


def _resume_label(label: str) -> str:
    """Return a live registry label for a resumed persistent subagent."""
    return label if label not in agent_registry else unique_registry_label(label)


def _start_resumed_persistent(
    parent: Agent,
    child: Agent,
    record: PersistentAgentRecord,
    label: str,
) -> None:
    """Register and launch a resumed persistent subagent."""
    child._lifecycle = "serviced"  # noqa: SLF001 -- resume restores serviced runtime mode
    child._is_subagent = True  # noqa: SLF001 -- resumed child is a subagent
    child.name = label
    agent_registry[label] = child
    forwarder = _build_forwarder(
        label,
        1,
        parent,
        child=child,
        notify_on_asleep=record.notify_on_asleep,
    )
    if forwarder is not None:
        child.runtime.observers.append(forwarder)
    task = asyncio.create_task(
        _serve_resumed_persistent(parent, child, label, forwarder)
    )
    parent.register_background(
        f"persistent:{label}",
        BackgroundTaskEntry(
            task=task,
            tool_name="persistent-agent",
            queue_id=label,
            started=time.time(),
            hidden=False,
            kind="subagent",
            lifecycle="serviced",
            persistent_run_id=record.run_id,
            notify_on_asleep=record.notify_on_asleep,
        ),
    )


async def _serve_resumed_persistent(
    parent: Agent,
    child: Agent,
    label: str,
    forwarder: Callable[[types.runtime.RuntimeEvent], None] | None,
) -> None:
    """Run a resumed persistent child and clean parent indexes on exit."""
    bg_key = f"persistent:{label}"
    try:
        await child.serve_forever()
    except Exception:
        logging.getLogger(__name__).exception(
            "persistent agent %r crashed after resume",
            label,
        )
    finally:
        if forwarder is not None and forwarder in child.runtime.observers:
            child.runtime.observers.remove(forwarder)
        agent_registry.pop(label, None)
        parent.forget_background(bg_key)


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
        base = Path(session_dir) if session_dir is not None else data_dir("sagent")
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


async def _with_resumed_persistent(
    agent: Agent,
    coro: Coroutine[object, object, None],
    *,
    session_dir: str | Path | None,
    resume_persistent: bool,
    allow_providers: tuple[str, ...],
) -> None:
    """Resume persistent children before running ``coro``."""
    if resume_persistent and session_dir is not None:
        await _resume_persistent_agents(
            agent,
            Path(session_dir),
            allow_providers=allow_providers,
            parent_label=agent.name,
        )
    await coro


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
    if isinstance(event, types.runtime.ToolResult):
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
    if isinstance(event, types.runtime.ModelServiceSuspended):
        return {
            "descriptor": "application/x-model-service-suspended",
            "provider": event.provider,
            "auth": event.auth,
            "account": event.account,
            "model_id": event.model_id,
            "retry_at": event.retry_at,
            "delay_sec": event.delay_sec,
            "server_supplied": event.server_supplied,
            "error": {
                "type_name": event.error.type_name,
                "message": event.error.message,
                "status": event.error.status,
            },
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

    user_msg = types.runtime.UserMessage(text=prompt)
    model_error: BaseException | None = None
    if output_format == "stream-json":
        async for event in agent.run(user_msg):
            if isinstance(event, types.runtime.ModelResponseError):
                model_error = event.exception
            record = _event_to_json_record(event)
            if record is None:
                continue
            json.dump(record, sys.stdout)
            sys.stdout.write("\n")
    else:
        async for event in agent.run(user_msg):
            if isinstance(event, types.runtime.ModelResponseError):
                model_error = event.exception
    if model_error is not None:
        message = f"{type(model_error).__name__}: {model_error}"
        if output_format == "json":
            json.dump({"error": message}, sys.stdout)
            sys.stdout.write("\n")
        elif output_format == "text":
            sys.stderr.write(f"Error: {message}\n")
        raise SystemExit(1)
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


def _last_assistant_text(history: list[types.runtime.ModelContextEvent]) -> str:
    """Return the text from the most recent ``AssistantMessage`` in ``history``."""
    for entry in reversed(history):
        if isinstance(entry, types.runtime.AssistantMessage):
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


def main() -> int:
    """Parse args, launch the agent, and return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
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
        return 0
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    _configure_logging(args.log_level)
    # Copy a pre-convention sagent home (a real ``~/.sagent`` or, for users who
    # symlinked it, the ``~/.claude`` squat) into the XDG home, before any
    # sagent path (sessions, caches) is read below.
    sessions.migrate_legacy_home()
    if args.recipe is not None:
        set_recipe(args.recipe)
    session_dir = None if args.ephemeral else _resolve_session_dir(args)
    loaded_session = None
    if session_dir is not None:
        loaded_session = load_session(Path(session_dir), {})
        if loaded_session is not None:
            _apply_resume_model_defaults(args, loaded_session[0])
    # The provider is "explicit" when the user passed ``--provider`` or a
    # resumed session pinned one; otherwise it defaults to the first
    # allowed provider (``primary=None``).
    resumed_provider = loaded_session is not None and bool(loaded_session[0].provider)
    args.provider_from_resume = resumed_provider
    explicit = bool(getattr(args, "provider_explicit", False)) or resumed_provider
    args.provider, allow_providers = _resolve_provider_and_allow(
        args.allow_providers,
        primary=args.provider if explicit else None,
    )
    try:
        thinking_state = _resolve_cli_thinking_state(args)
        provider, model, resolved_auth = _build_provider_model(
            args,
            thinking_state,
            allow_providers=allow_providers,
        )
        _validate_cli_thinking_state(model, thinking_state)
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
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
    agent_tools = resolve_tools(tool_names, allow_providers=allow_providers)
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
            include_memory=not args.ephemeral,
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
        thinking_state=thinking_state,
        provider_options=_cli_provider_options(args),
    )
    if args.max_request_tokens is not None:
        agent.max_request_tokens = args.max_request_tokens
    if args.max_response_tokens is not None:
        agent.max_response_tokens = args.max_response_tokens

    if loaded_session is not None:
        agent.resume(*loaded_session)

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
                _with_resumed_persistent(
                    agent,
                    run_repl(agent, history=args.history),
                    session_dir=session_dir,
                    resume_persistent=args.resume_persistent,
                    allow_providers=allow_providers,
                ),
            ),
        )
    else:
        asyncio.run(
            _with_signals(
                agent,
                _with_resumed_persistent(
                    agent,
                    _run_headless(
                        agent,
                        input_format=args.input_format,
                        output_format=args.output_format,
                    ),
                    session_dir=session_dir,
                    resume_persistent=args.resume_persistent,
                    allow_providers=allow_providers,
                ),
            )
        )
    return 0


def _do_login(args: argparse.Namespace) -> None:
    """Run the OAuth flow for ``args.provider`` and save under ``args.account``."""
    cls = getattr(providers, args.provider, None)
    if cls is None:
        sys.stderr.write(f"Error: unknown provider {args.provider!r}\n")
        sys.exit(1)
    login_fn = getattr(cls, "login", None)
    save_fn = getattr(cls, "save", None)
    if login_fn is None or save_fn is None:
        if args.provider == "AnthropicCLI":
            if args.account not in (None, "default"):
                sys.stderr.write(
                    "Error: AnthropicCLI named accounts use legacy credential "
                    "files and do not support interactive login.\n"
                )
                sys.exit(1)
            sys.stderr.write(
                "Error: AnthropicCLI uses the Claude CLI login. Run:\n"
                "  claude auth login --claudeai\n"
            )
            sys.exit(1)
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
    raise SystemExit(main())
# vim: ft=python
