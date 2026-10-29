"""Run a coding agent: interactively on a terminal, or once on piped input.

A terminal on stdin opens the interactive screen; anything else is read to
its end as one prompt, answered, and printed. Sessions are journaled under
the per-project directory and resumable by id, from this directory or any.

``run`` builds its console while stdout is patched: Rich keeps the stream it
was built with, and only the patched one prints above the prompt.

Examples:
  python -m sagent                                   # interactive
  python -m sagent --provider anthropic --model sonnet
  echo 'fix the bug' | python -m sagent              # text in, text out
  python -m sagent --output-format json < prompt.txt
  python -m sagent --input-format stream-json --output-format stream-json < p.ndjson
  python -m sagent --resume 3f2a                     # resume by id prefix
  python -m sagent --model work@oaisub/gpt+highthink+fast
  python -m sagent --effort max --tool.Bash.display.output.lines 3,1
  python -m sagent login --provider oaisub --account work

Every setting ``/help`` lists is a flag too, matched as the prompt matches it:
``--NAME VALUE``. They apply once the agent is built, in order, ``--model``
last, so what the model's name says wins.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import asyncio
import contextlib
import functools
import logging
import math
import os
import signal
import sys
import time

from treekle import from_plain, json_dumps, json_loads
from wesearch.fetch import FetchSession

from sagent.agent.agent import Agent, NoShrinker, default_system_prompt
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.pruner import Pruner
from sagent.agent.team import Team
from sagent.lib.userdirs import data_dir, state_dir
from sagent.model.exceptions import AuthError
from sagent.providers.catalog import ModelCatalog, default_provider
from sagent.providers.table import ModelTags
from sagent.providers.types import Login
from sagent.runtime import types
from sagent.session.jsonl import JournalInUseError, JsonlJournal
from sagent.session.legacy_journal import (
    find_old,
    is_legacy,
    upgrade,
)
from sagent.session.projects import (
    find_sessions,
    latest_session,
    list_sessions,
    new_session_path,
    project_dir,
)
from sagent.session.types import ToolCall, ToolResult
from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.background_task import BackgroundTask
from sagent.tools.bash import Bash
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.memory import Memory
from sagent.tools.paper_author import PaperAuthor
from sagent.tools.paper_details import PaperDetails
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.paper_search import PaperSearch
from sagent.tools.play_audio import PlayAudio
from sagent.tools.read import Read
from sagent.tools.session import Session
from sagent.tools.skill import Skill
from sagent.tools.web_fetch import WebFetch
from sagent.tools.web_search import WebSearch
from sagent.tools.write import Write
from sagent.tui.console import ConsolePrinter
from sagent.tui.prompt import (
    OPEN_TREE,
    Input,
    Panes,
    PromptSource,
    prompt_text,
    repaint_on_change,
)
from sagent.tui.render import error_text, suspended_text
from sagent.tui.settings.flags import flags
from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.match import match
from sagent.tui.settings.types import SettingError
from sagent.tui.slash import Commands, headless_wanted
from sagent.tui.tree import pick
from sagent.tui.types import Frame
from sagent.tui.views import Views


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import DynamicKeyBindings
    from prompt_toolkit.patch_stdout import StdoutProxy
    from prompt_toolkit.styles import Style
    from rich.console import Console
    from treekle import PlainTree

    from sagent.agent.types import (
        Agent as AgentProtocol,
        Factory,
        Shrinker,
    )
    from sagent.model.types import Response
    from sagent.providers.types import (
        Model,
        ModelCatalog as Catalog,
        Provider,
    )
    from sagent.session.projects import SessionInfo
    from sagent.tools.types import Tool
    from sagent.tui.settings.types import Setting
    from sagent.tui.types import CommandOutput
else:
    from wrapt import lazy_import

    PromptSession = lazy_import("prompt_toolkit", "PromptSession")  # 80 ms cold.
    AutoSuggestFromHistory = lazy_import(
        "prompt_toolkit.auto_suggest",
        "AutoSuggestFromHistory",
    )
    FileHistory = lazy_import("prompt_toolkit.history", "FileHistory")
    DynamicKeyBindings = lazy_import("prompt_toolkit.key_binding", "DynamicKeyBindings")
    StdoutProxy = lazy_import("prompt_toolkit.patch_stdout", "StdoutProxy")
    Style = lazy_import("prompt_toolkit.styles", "Style")
    Console = lazy_import("rich.console", "Console")  # 60 ms cold.


__all__ = [
    "main",
    "run",
]


logger = logging.getLogger(__name__)


def main() -> int:
    """Run the agent.

    Returns:
      code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    static = _add_arguments(parser)
    try:
        rest, given = flags(sys.argv[1:], static=static)
    except ValueError as e:
        parser.error(str(e))
    parsed = cast("Flags", parser.parse_args(rest))
    if parsed.command not in (None, "login"):
        parser.error(f"unknown command {parsed.command!r}; the only one is login")
    try:
        return asyncio.run(_main(parsed, given=given))
    # ``_signals`` answers the first SIGINT by cancelling the run, which
    # ``asyncio.run`` re-raises as ``CancelledError``: uncaught, a Ctrl+C
    # printed a traceback and exited 1.
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130


class Flags(Protocol):
    """Parsed command-line flags."""

    command: str | None
    headless: bool | None
    allow_providers: str
    name: str
    system_prompt: str | None
    append_system_prompt: str
    compact: bool
    compactor: str
    prune: bool
    tools: list[str] | None
    max_budget_usd: float | None
    max_tool_call_rounds: int | None
    session: Path | None
    ephemeral: bool
    resume: str | None
    continue_: bool
    resume_all: bool
    continue_all: bool
    resume_limit: int
    resume_persistent: bool
    add_dir: list[Path]
    history: Path | None
    output_format: str
    input_format: str
    log_level: str | None


async def run(
    agent: AgentProtocol,
    *,
    catalog: Catalog,
    team: Team | None = None,
    history: Path | None = None,
    show_thinking: bool = True,
    print_batch_sec: float = 0.2,
) -> None:
    """Show ``agent``'s session and take what the user types, until they quit.

    Args:
      agent: An open agent.
      catalog: Where the model's settings find models.
      team: The agents ``/send``, ``/halt`` and ``/kill`` may reach by label.
      history: The input history; the per-user one when omitted.
      show_thinking: Whether reasoning shows; ``/screen.thinking`` flips it.
      print_batch_sec: How long prints above the prompt gather into one repaint;
        quitting waits out the last one.

    """
    path = (
        history
        if history is not None
        else state_dir() / "rekursiv-ai" / "sagent" / "repl-history"
    )
    # Enter writes history before it clears the line, so a missing directory
    # would leave the sent text in the prompt.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not ``patch_stdout``: it builds its proxy with no way to set the batch time.
    with (
        StdoutProxy(sleep_between_writes=print_batch_sec, raw=True) as proxy,
        contextlib.redirect_stdout(proxy),
        contextlib.redirect_stderr(proxy),
    ):
        printer = ConsolePrinter(Console(stderr=True), show_thinking=show_thinking)
        entry = Input()
        views = Views(
            agent,
            team=team,
            printer=printer,
            catalog=catalog,
            commands=Commands,
            switched=lambda: setattr(source, "panes", views.current.panes),
        )
        session = _prompt_session(views, entry=entry, history=path)
        source = PromptSource(session, panes=views.current.panes, notice=printer.note)
        background = [
            asyncio.create_task(
                repaint_on_change(
                    session.app.invalidate,
                    lambda: Frame(
                        deferred=views.current.deferred.frame(),
                        input=entry.frame(),
                        status=views.current.status.frame(),
                    ),
                ),
            ),
            asyncio.create_task(views.watch()),
        ]
        try:
            while (line := await source.line()) is not None:
                if line == OPEN_TREE:
                    await _tree(views)
                    continue
                try:
                    if await views.current.commands(line):
                        break
                except Exception as e:
                    logger.warning("command %r failed", line, exc_info=True)
                    printer.error(error_text(e))
        finally:
            for task in background:
                task.cancel()
            await views.aclose()
    journal = agent.session.journal.path
    if journal is not None:
        printer.line(
            "Resume this session with:\n"
            f"sagent --resume {journal.parent.name}   # this session\n"
            "sagent --continue         # the latest in this directory\n"
            "sagent --resume           # pick one in this directory\n"
            "sagent --continue-all     # the latest anywhere\n"
            "sagent --resume-all       # pick one anywhere",
        )


def _prompt_session(views: Views, *, entry: Input, history: Path) -> PromptSession[str]:
    """Return the prompt: what waits over ``> ``, status under, for the agent shown."""
    return PromptSession(
        lambda: prompt_text(views.current.deferred, entry),
        multiline=True,
        erase_when_done=True,
        history=FileHistory(str(history)),
        auto_suggest=AutoSuggestFromHistory(),
        bottom_toolbar=views.toolbar,
        key_bindings=DynamicKeyBindings(lambda: views.current.keys),
        style=Style.from_dict(
            {
                "bottom-toolbar": "fg:ansibrightblack noreverse bg:default",
                "queued_input_pane": "fg:ansibrightblack",
                "input_pane": "bold",
            },
        ),
    )


# Nothing may print while the tree is up: a print suspends a full-screen app.
async def _tree(views: Views) -> None:
    """Show the tree; show the agent picked, or the one before with what it did."""
    views.hold()
    picked: str | None = None
    try:
        picked = await pick(
            views.lineage,
            focused=views.current.label,
            stop=views.stop,
            cancel=views.cancel,
        )
    finally:
        views.release(show=picked is None or picked == views.current.label)
    if picked is not None and picked in views.lineage:
        views.focus(picked)


# A setting is no flag here: ``flags`` hands argparse only what this knows.
def _add_arguments(parser: argparse.ArgumentParser) -> frozenset[str]:
    """Register flags on ``parser``; return every option string it now knows."""
    actions = [
        parser.add_argument(
            "command",
            nargs="?",
            help="'login' signs in to the model's provider, as --provider and"
            " --account say, and exits.",
        ),
        *_session_arguments(parser),
        *_run_arguments(parser),
    ]
    return frozenset(
        ("-h", "--help", *(o for action in actions for o in action.option_strings)),
    )


def _session_arguments(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Register what picks the session and its journal."""
    source = parser.add_mutually_exclusive_group()
    return [
        source.add_argument(
            "--session",
            type=Path,
            help="Journal file to write and resume.",
        ),
        source.add_argument(
            "--ephemeral",
            action="store_true",
            help="Keep no journal and no memories.",
        ),
        source.add_argument(
            "--resume",
            nargs="?",
            const="",
            metavar="ID",
            help="Resume a session: by id prefix, or pick one for this directory.",
        ),
        source.add_argument(
            "--continue",
            dest="continue_",
            action="store_true",
            help="Resume this directory's latest session.",
        ),
        source.add_argument(
            "--resume-all",
            action="store_true",
            help="Pick a session from every directory.",
        ),
        source.add_argument(
            "--continue-all",
            action="store_true",
            help="Resume the latest session of any directory.",
        ),
        parser.add_argument(
            "--resume-limit",
            type=_positive_int,
            default=20,
            metavar="N",
            help="Sessions the picker lists (default: %(default)s).",
        ),
        parser.add_argument(
            "--resume-persistent",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Restart the persistent agents a resumed session left running"
            " (default: on).",
        ),
        parser.add_argument(
            "--history",
            type=Path,
            metavar="PATH",
            help="Input history file (default: the per-user one).",
        ),
        parser.add_argument(
            "--output-format",
            choices=("text", "json", "stream-json"),
            default="text",
            help="Piped mode: the reply, the reply as JSON, or every event as NDJSON.",
        ),
        parser.add_argument(
            "--input-format",
            choices=("text", "stream-json"),
            default="text",
            help="Piped mode: text, or NDJSON prompt objects joined by blank lines.",
        ),
        parser.add_argument(
            "--log-level",
            default=os.environ.get("SAGENT_LOG_LEVEL"),
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            help="Log level; interactive logs go to the session's .log file.",
        ),
    ]


def _run_arguments(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Register what builds the agent: its tools, prompt, caps, and helpers."""
    return [
        parser.add_argument(
            "--headless",
            action=argparse.BooleanOptionalAction,
            help="For login: sign in without a browser; print the URL, paste the"
            " code back. Default: headless over ssh, or on Linux with no display.",
        ),
        parser.add_argument(
            "--allow-providers",
            default=os.environ.get("SAGENT_ALLOW_PROVIDERS", ""),
            metavar="LIST",
            help="Comma-separated providers this agent and the agents it spawns"
            " may use; $SAGENT_PROVIDER joins them. Default:"
            " $SAGENT_ALLOW_PROVIDERS, else every provider.",
        ),
        parser.add_argument(
            "--name",
            default="Agent",
            help="The agent's label in its team (default: %(default)s).",
        ),
        parser.add_argument(
            "--system-prompt",
            help="System prompt replacing the default one.",
        ),
        parser.add_argument(
            "--append-system-prompt",
            default="",
            help="Instructions added after the system prompt.",
        ),
        parser.add_argument(
            "--compact",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Compact the session when the window fills (default: on).",
        ),
        parser.add_argument(
            "--compactor",
            choices=["summary", "session-audit"],
            default="summary",
            help="How to compact: condense what the model sees, or audit the whole"
            " session from its journal (default: summary).",
        ),
        parser.add_argument(
            "--prune",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Drop old tool output once that costs less than keeping it cached"
            " (default: on).",
        ),
        parser.add_argument(
            "--tools",
            nargs="+",
            metavar="NAME",
            help="Tools to load, or 'none'. Default: every tool, without Memory"
            " when --ephemeral.",
        ),
        parser.add_argument(
            "--max-budget-usd",
            type=_positive_float,
            metavar="USD",
            help="Most the session may spend on model calls.",
        ),
        parser.add_argument(
            "--max-tool-call-rounds",
            type=_positive_int,
            metavar="N",
            help="Most tool-call rounds one prompt may run. Default: no cap.",
        ),
        parser.add_argument(
            "--add-dir",
            type=Path,
            nargs="+",
            default=[],
            metavar="DIR",
            help="More directories whose AGENTS.md files extend the prompt.",
        ),
    ]


async def _main(flags: Flags, *, given: Sequence[tuple[str, str]]) -> int:
    """Build the agent from ``flags`` and the settings ``given``; run it until done."""
    cwd = Path.cwd()
    interactive = sys.stdin.isatty()
    screen = _Screen()
    async with (
        ModelCatalog() as catalog,
        Team(catalog=catalog) as team,
        FetchSession() as fetch,
    ):
        try:
            name = _provider_name(catalog, flags)
            model = catalog.model("default", current=name)
            if flags.command == "login":
                agent = Agent(model=model)
                await _configure(agent, catalog=catalog, screen=screen, given=given)
                return await _login(
                    agent.model.provider,
                    headless=headless_wanted(os.environ)
                    if flags.headless is None
                    else flags.headless,
                )
            tools = _tools(flags, cwd=cwd, team=team, session=fetch)
        except (KeyError, ValueError) as e:
            sys.stderr.write(f"Error: {error_text(e)}\n")
            return 1
        path = _journal_path(flags, cwd=cwd, interactive=interactive)
        if path is not None and is_legacy(path) and not upgrade(path):
            sys.stderr.write("[resume] nothing to resume; starting fresh.\n")
            path = new_session_path(project_dir(cwd))
        journal = JsonlJournal(path) if path is not None else None
        _logging(flags.log_level, path=path, interactive=interactive)
        agent = _agent(
            flags,
            model=model,
            tools=tools,
            journal=journal,
            cwd=cwd,
            interactive=interactive,
        )
        async with contextlib.AsyncExitStack() as stack:
            if journal is not None:
                try:
                    await stack.enter_async_context(journal)
                except JournalInUseError as e:
                    sys.stderr.write(f"Error: {e}\n")
                    return 1
            # What the session stood comes back first; the flags apply on top.
            try:
                await agent.recall(catalog)
                # The team signs in as the session did, until a flag says otherwise.
                catalog.use_account(agent.account)
                system = _system(flags)
                if system is not None:
                    agent.system_prompt = system
                await _configure(agent, catalog=catalog, screen=screen, given=given)
            except ValueError as e:
                sys.stderr.write(f"Error: {error_text(e)}\n")
                return 1
            if interactive:
                described = ModelTags.from_settings(
                    agent.model.capability,
                    agent.model.settings,
                    provider=agent.model.provider.name,
                    account=catalog.account,
                )
                sys.stderr.write(f"[/model] {described}\n")
            await stack.enter_async_context(agent)
            label = team.join(agent, label=flags.name)
            spawn = agent.tools.get("AgentSpawn")
            if flags.resume_persistent and isinstance(spawn, AgentSpawn):
                await spawn.resume(label)
            with _signals():
                if interactive:
                    await run(
                        agent,
                        catalog=catalog,
                        team=team,
                        history=flags.history,
                        show_thinking=screen.show_thinking,
                    )
                    return 0
                return await _headless(agent, flags)


# The allow list bounds the team too: a spawned agent finds its models in the same
# catalog. ``$SAGENT_PROVIDER`` joins it, never fails it.
def _provider_name(catalog: ModelCatalog, flags: Flags) -> str:
    """Narrow ``catalog`` to ``--allow-providers``; return the provider to start on."""
    allowed = [p.strip() for p in flags.allow_providers.split(",") if p.strip()]
    provider = os.environ.get("SAGENT_PROVIDER")
    if not allowed:
        return catalog.name(provider if provider is not None else default_provider())
    if provider is None:
        return catalog.allow(allowed)[0]
    _ = catalog.allow([*allowed, provider])
    return catalog.name(provider)


async def _login(provider: Provider, *, headless: bool) -> int:
    """Sign in to ``provider``; return the exit code."""
    if not isinstance(provider, Login):
        sys.stderr.write(f"Error: {provider.name} has no interactive sign-in.\n")
        return 1
    # A pasted code from another sign-in, or no answer at the prompt at all.
    try:
        await provider.login(headless=headless)
    except (AuthError, EOFError, OSError, RuntimeError, ValueError) as e:
        sys.stderr.write(f"Error: {error_text(e)}\n")
        return 1
    return 0


def _system(flags: Flags) -> str | None:
    """Return the system prompt the flags give; ``None`` when they give none."""
    system = flags.system_prompt
    if flags.append_system_prompt:
        base = system if system is not None else default_system_prompt()
        system = f"{base}\n\n{flags.append_system_prompt}"
    return system


# At a terminal a long rate limit is waited out, the user watching; piped, it ends the
# run, whose caller decides. ``--runtime.retry`` says otherwise.
def _agent(
    flags: Flags,
    *,
    model: Model,
    tools: Sequence[Tool],
    journal: JsonlJournal | None,
    cwd: Path,
    interactive: bool,
) -> Agent:
    """Return the root agent ``flags`` describe, not yet opened."""
    audits = flags.compactor == "session-audit"
    compactor: Factory[Shrinker] = (
        (
            functools.partial(SummaryCompactor, style="session-audit")
            if audits
            else SummaryCompactor
        )
        if flags.compact
        else NoShrinker
    )
    pruner: Factory[Shrinker] = Pruner if flags.prune else NoShrinker
    return Agent(
        model=model,
        tools=tools,
        system_prompt=_system(flags),
        journal=journal,
        compactor=compactor,
        pruner=pruner,
        cwd=cwd,
        max_rounds=flags.max_tool_call_rounds,
        max_spend_usd=flags.max_budget_usd,
        persistent_retry=interactive,
        agents_md=AgentsMdConfig(additional_dirs=tuple(flags.add_dir)),
    )


# The account applies first, so every model a later setting names signs in as it; then
# the model, so each other flag is checked against it; then the model again, so what
# its name says wins over a flag for the same choice.
async def _configure(
    agent: AgentProtocol,
    *,
    catalog: Catalog,
    screen: _Screen,
    given: Sequence[tuple[str, str]],
) -> None:
    """Apply each setting ``given``, as the prompt would."""
    commands = Commands(
        agent,
        catalog=catalog,
        say=_dropped,
        screen=screen,
        panes=Panes(),
    )
    settings = [s for s in harvest(commands.owners()) if s.cli]
    chosen: list[tuple[Setting, str]] = []
    for name, value in given:
        setting = match(settings, name)
        if setting is None:
            raise SettingError(f"--{name} names no setting; see /help")
        chosen.append((setting, value))
    account, model, provider = (
        ("model", "provider", "account"),
        ("model",),
        ("model", "provider"),
    )
    named = {s.name: v for s, v in chosen if s.name in (model, provider)}
    rest = [(s, v) for s, v in chosen if s.name not in (model, provider)]
    if named:
        spec = named.get(model, "default")
        if provider in named and not _owned(catalog, spec):
            spec = f"{_named(catalog, named[provider])}/{spec}"
        whole = (next(s for s in settings if s.name == model), spec)
        rest = [whole, *rest, whole]
    for setting, value in sorted(rest, key=lambda c: c[0].name != account):
        try:
            await setting.apply(value)
        except ValueError as e:
            name = ".".join(setting.name)
            raise SettingError(f"[--{name}] {error_text(e)}") from e


def _owned(catalog: Catalog, model: str) -> bool:
    """Return whether ``model`` names its own provider, as ``[ACCOUNT@]P/M`` does."""
    prefix, slash, _ = model.partition("/")
    try:
        catalog.name(prefix.rpartition("@")[2])
    except ValueError:
        return False
    return bool(slash)


def _named(catalog: Catalog, provider: str) -> str:
    """Return the provider ``--provider`` names, as the setting would refuse it."""
    try:
        return catalog.name(provider)
    except ValueError as e:
        raise SettingError(f"[--model.provider] {error_text(e)}") from e


def _dropped(output: CommandOutput) -> None:
    """Drop what a command run from a flag says; a failure raises instead."""
    del output


def _tools(
    flags: Flags,
    *,
    cwd: Path,
    team: Team,
    session: FetchSession,
) -> list[Tool]:
    """Return the tools ``--tools`` names."""
    read, edit, grep, glob, ls = Read(), Edit(), Grep(), Glob(), List()
    fetch = WebFetch(session=session)
    builders: dict[str, Callable[[], Tool]] = {
        "Bash": lambda: Bash(peers=(read, edit, grep, glob, ls, fetch)),
        "Read": lambda: read,
        "Write": Write,
        "Edit": lambda: edit,
        "Grep": lambda: grep,
        "Glob": lambda: glob,
        "List": lambda: ls,
        "WebSearch": lambda: WebSearch(session=session),
        "WebFetch": lambda: fetch,
        "PaperSearch": lambda: PaperSearch(session=session),
        "PaperDetails": lambda: PaperDetails(session=session),
        "PaperAuthor": lambda: PaperAuthor(session=session),
        "PaperFetch": lambda: PaperFetch(session=session),
        "Memory": lambda: Memory(directory=project_dir(cwd) / "memory"),
        "PlayAudio": PlayAudio,
        "AgentSpawn": lambda: AgentSpawn(team=team),
        "AgentSend": lambda: AgentSend(team=team),
        "AgentSelf": lambda: AgentSelf(team=team),
        "Session": lambda: Session(team=team),
        "Skill": Skill,
        "BackgroundTask": BackgroundTask,
    }
    named = flags.tools
    names = list(builders) if named is None else [] if named == ["none"] else named
    if flags.ephemeral and "Memory" in names:
        if named is not None:
            raise ValueError("--ephemeral saves no memories; drop Memory from --tools.")
        names.remove("Memory")
    unknown = sorted(set(names) - set(builders))
    if unknown:
        raise ValueError(
            f"unknown tools {', '.join(unknown)}; known: {', '.join(builders)}",
        )
    tools = {name: builders[name]() for name in names}
    return list(tools.values())


def _journal_path(flags: Flags, *, cwd: Path, interactive: bool) -> Path | None:
    """Return the journal ``--session``, ``--resume`` and the rest pick; ``None`` for none."""
    here = project_dir(cwd)
    if flags.ephemeral:
        return None
    if flags.session is not None:
        return flags.session
    found: SessionInfo | None = None
    if flags.continue_:
        found = latest_session(here)
    elif flags.continue_all:
        found = next(iter(find_sessions("", limit=1)), None)
    elif flags.resume is not None and flags.resume:
        path = _by_prefix(flags.resume, here=here)
        sys.stderr.write(f"[resume] {path}\n")
        return path
    elif flags.resume is not None or flags.resume_all:
        listed = (
            find_sessions("", limit=flags.resume_limit + 1)
            if flags.resume_all
            else list_sessions(here, limit=flags.resume_limit + 1)
        )
        found = _pick(listed, cap=flags.resume_limit) if interactive else None
    if found is not None:
        sys.stderr.write(f"[resume] {found.path}\n")
        return found.path
    if (
        flags.continue_
        or flags.continue_all
        or flags.resume is not None
        or flags.resume_all
    ):
        sys.stderr.write("[resume] nothing to resume; starting fresh.\n")
    return new_session_path(here)


# This directory first, then every one: a prefix unique here wins even when another
# project shares it. Ambiguity in the scope that matched is an error, never a guess.
def _by_prefix(prefix: str, *, here: Path) -> Path:
    """Return the journal or old log of the one session ``prefix`` names; else exit."""
    for scope, root in (("this directory", here), ("all projects", None)):
        matches = [
            *(s.path for s in find_sessions(prefix, root=root)),
            *find_old(prefix, root=root),
        ]
        if len(matches) > 1:
            ids = ", ".join(sorted(path.parent.name for path in matches))
            sys.stderr.write(f"[resume] {prefix!r} is ambiguous in {scope}: {ids}.\n")
            raise SystemExit(1)
        if matches:
            return matches[0]
    sys.stderr.write(f"[resume] no session matching {prefix!r}.\n")
    raise SystemExit(1)


def _pick(sessions: Sequence[SessionInfo], *, cap: int) -> SessionInfo | None:
    """Ask which of ``sessions`` to resume; ``None`` when none is chosen."""
    if not sessions:
        return None
    shown = sessions[:cap]
    if len(sessions) > cap:
        sys.stderr.write(f"  (showing {cap} sessions; older ones hidden)\n")
    now = time.time()
    for i, s in enumerate(shown, start=1):
        title = s.title if len(s.title) <= 60 else s.title[:59] + "…"
        sys.stderr.write(
            f"  [{i:>2}] {_ago(now - s.mtime):>7} · {s.records:>4} records"
            f" · {title or '(no user messages)'}\n",
        )
    while True:
        sys.stderr.write("Resume which? [1] ")
        sys.stderr.flush()
        line = sys.stdin.readline()
        if not line:
            return None
        raw = line.strip()
        if not raw:
            return shown[0]
        if not raw.isdigit():
            sys.stderr.write(f"  not a number: {raw!r}\n")
            continue
        return shown[int(raw) - 1] if 0 < int(raw) <= len(shown) else None


def _ago(seconds: float) -> str:
    """Return ``seconds`` as ``12s ago``, ``5m ago``, ``3h ago``, or ``2d ago``."""
    s = max(0, int(seconds))
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if s >= size:
            return f"{s // size}{unit} ago"
    return f"{s}s ago"


# Interactive logs never touch stderr: any write there overlays the prompt.
def _logging(level: str | None, *, path: Path | None, interactive: bool) -> None:
    """Send logs to stderr when piped, to the session's ``.log`` when interactive."""
    if not interactive:
        if level is not None:
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            logging.getLogger("sagent").setLevel(level)
        return
    logging.lastResort = logging.NullHandler()
    root = logging.getLogger()
    root.handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler)
        or not isinstance(h, logging.StreamHandler)
    ]
    log = (
        path.with_suffix(".log")
        if path is not None
        else data_dir() / "rekursiv-ai" / "sagent" / "repl.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    root.addHandler(handler)
    logging.getLogger("sagent").setLevel(level or "DEBUG")


@contextlib.contextmanager
def _signals() -> Generator[None]:
    """Cancel the running task on the first SIGINT or SIGTERM; exit on the second."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert isinstance(task, asyncio.Task)
    hits = 0

    def handle() -> None:
        nonlocal hits
        hits += 1
        if hits > 1:
            os._exit(1)
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, handle)
    try:
        yield
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(sig)


async def _headless(agent: AgentProtocol, flags: Flags) -> int:
    """Answer stdin as one prompt; print the reply as ``--output-format`` says."""
    raw = await asyncio.to_thread(sys.stdin.read)
    if flags.input_format == "stream-json":
        try:
            prompts = [
                p
                for line in raw.splitlines()
                if line.strip()
                if (
                    p := from_plain(
                        json_loads(line, dict[str, object]).get("prompt"),
                        str,
                        default="",
                    )
                )
            ]
        except ValueError as e:
            sys.stderr.write(f"Error: invalid stream-json input: {e}\n")
            return 1
        prompt = "\n\n".join(prompts)
    else:
        prompt = raw.strip()
    if not prompt:
        sys.stderr.write("Error: no input on stdin.\n")
        return 1
    stream = flags.output_format == "stream-json"

    def emit(record: Mapping[str, PlainTree]) -> None:
        sys.stdout.write(json_dumps(record) + "\n")
        sys.stdout.flush()

    def observe(event: types.RuntimeEvent) -> None:
        for record in _records(event, tools=agent.tools):
            emit(record)

    if stream:
        agent.runtime.observers.append(observe)
    try:
        reply = await agent.send(prompt)
    except Exception as e:  # noqa: BLE001 -- Any failure of the turn is the run's result.
        message = error_text(e)
        if flags.output_format == "text":
            sys.stderr.write(f"Error: {message}\n")
        else:
            emit({"error": message})
        return 1
    finally:
        if stream:
            agent.runtime.observers.remove(observe)
    if flags.output_format == "text":
        sys.stdout.write(reply + "\n")
    else:
        emit({"content": reply, **({"descriptor": "result"} if stream else {})})
    return 0


def _records(
    event: types.RuntimeEvent,
    *,
    tools: Mapping[str, Tool],
) -> list[dict[str, PlainTree]]:
    """Return ``event`` as stream-json records; none for an event not streamed."""
    if isinstance(event, types.ModelResponsePartial):
        return [{"descriptor": "text/plain", "content": event.text}]
    if isinstance(event, types.ModelResponseThinking):
        return [{"descriptor": "text/x-thinking", "content": event.text}]
    if isinstance(event, types.ModelResponseComplete):
        return [
            {
                "descriptor": "text/x-tool-label",
                "call_id": call.call_id,
                "content": tool.summary(call.arguments)
                if (tool := tools.get(call.name))
                else call.name,
            }
            for call in _calls(event.response)
        ]
    # A detached call streams its stand-in, marked, then its result when it lands.
    result = event.result if isinstance(event, types.DetachedResult) else event
    if isinstance(result, ToolResult):
        return [
            {
                "descriptor": "application/x-tool-result",
                "call_id": result.call_id,
                "content": result.text(),
                "is_error": result.is_error,
                **({} if result.stub is None else {"stub": result.stub}),
            },
        ]
    if isinstance(event, types.ModelResponseError):
        return [
            {
                "descriptor": "application/x-error",
                "content": error_text(event.exception),
            },
        ]
    if isinstance(event, types.ModelCallSuspended):
        return [{"descriptor": "text/x-notice", "content": suspended_text(event)}]
    return []


def _calls(response: Response) -> list[ToolCall]:
    """Return the tool calls of a model turn."""
    return [r for r in response.records if isinstance(r, ToolCall)]


def _positive_int(raw: str) -> int:
    """Parse a count of at least one."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _positive_float(raw: str) -> float:
    """Parse a finite amount above zero."""
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from None
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be finite and > 0, got {raw}")
    return value


@dataclass(slots=True, kw_only=True)
class _Screen:
    """Whether reasoning shows, settled by ``--screen.thinking`` before a screen opens."""

    show_thinking: bool = True
