"""Tests for the command line: flags, journals, tools, and both front ends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import argparse
import asyncio
import fcntl
import functools
import io
import logging
import os
import re
import signal
import sys
import time

from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.plain_text import PlainTextOutput
from prompt_toolkit.patch_stdout import StdoutProxy
from rich.console import Console
from treekle import PlainTree, json_dumps, json_loads
from wesearch.fetch import FetchSession

import pytest

from sagent.agent.agent import Agent, NoShrinker, default_system_prompt
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.team import Team
from sagent.lib.userdirs import data_dir, state_dir
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog
from sagent.model.exceptions import (
    AuthError,
    RateLimitedError,
    UnknownModelError,
)
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.providers.catalog import ModelCatalog, default_provider
from sagent.providers.table import ModelTable
from sagent.runtime import types
from sagent.runtime.exceptions import BudgetExhaustedError
from sagent.session import projects
from sagent.session.jsonl import JsonlJournal, read_record
from sagent.session.projects import list_sessions, project_dir
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AnyToolResult,
    AssistantMessage,
    SessionRecord,
    ShellCommandResult,
    Thinking,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
    record_classes,
)
from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_send import AgentSend
from sagent.tools.agent_spawn import AgentSpawn
from sagent.tools.bash import Bash
from sagent.tools.display import ToolDisplay
from sagent.tools.memory import Memory
from sagent.tools.paper_author import PaperAuthor
from sagent.tools.paper_details import PaperDetails
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.paper_search import PaperSearch
from sagent.tools.read import Read
from sagent.tools.session import Session
from sagent.tools.web_fetch import WebFetch
from sagent.tools.web_search import WebSearch
from sagent.tui import main
from sagent.tui.console import ConsolePrinter
from sagent.tui.prompt import Input, prompt_text
from sagent.tui.settings.flags import flags
from sagent.tui.settings.types import SettingError
from sagent.tui.slash import Commands
from sagent.tui.types import CommandOutput
from sagent.tui.views import Views


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from sagent.model.types import Request, Response
    from sagent.providers.types import (
        Provider,
        ProviderFactory,
        ProviderSharedResources,
    )
    from sagent.tools.types import ToolContext


# ---- fixtures --------------------------------------------------------------


def _flags(*argv: str) -> main.Flags:
    parser = argparse.ArgumentParser()
    main._add_arguments(parser)
    return cast("main.Flags", parser.parse_args(argv))


_DEEP = ModelCapability(
    model_id="deep",
    thinking_effort=frozenset({"none", "high"}),
    thinking_budget=frozenset({"none", "auto"}),
    thinking_output=frozenset({"none", "text"}),
    default=ModelSettings(context_size=PriceCatalog()["auto", "fullcontext"]),
)


class _Picky(ScriptedProvider):
    """Answers only to ``scripted`` and ``deep``, the way a real catalog row does."""

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        # Its one size is its whole window, which a saved session names.
        if model.removesuffix("+fullcontext") not in ("default", "scripted", "deep"):
            raise UnknownModelError(f"No model {model!r}.")
        return super().__getitem__(model)


class _Signing(_Picky):
    """Signs in, and serves only its own default: none of the starting provider's."""

    def __init__(self, *, name: str, resources: ProviderSharedResources) -> None:
        super().__init__(name=name, resources=resources)
        self.signed_in = 0
        self.headless: list[bool] = []

    @override
    def __getitem__(self, model: str) -> ScriptedModel:
        if model != "default":
            raise UnknownModelError(f"Unknown model {model!r}.")
        return super().__getitem__(model)

    async def login(self, *, headless: bool = False) -> None:
        self.signed_in += 1
        self.headless.append(headless)


@dataclass(slots=True, kw_only=True)
class _Script:
    """The replies the catalog's providers answer with, and what was built."""

    replies: list[Reply] = field(default_factory=list[Reply])
    capability: ModelCapability | None = None
    catalogs: list[ModelCatalog] = field(default_factory=list[ModelCatalog])
    accounts: list[str | None] = field(default_factory=list[str | None])
    providers: dict[str, ScriptedProvider] = field(
        default_factory=dict[str, ScriptedProvider],
    )

    def catalog(self, *, account: str | None = None) -> ModelCatalog:
        self.accounts.append(account)

        def scripted(*, name: str, resources: ProviderSharedResources) -> Provider:
            return self._keep(
                _Picky(
                    name=name,
                    resources=resources,
                    replies=self.replies,
                    capability=self.capability,
                ),
            )

        def deep(*, name: str, resources: ProviderSharedResources) -> Provider:
            return self._keep(_Picky(name=name, resources=resources, capability=_DEEP))

        def signing(*, name: str, resources: ProviderSharedResources) -> Provider:
            return self._keep(_Signing(name=name, resources=resources))

        def per_account(account: str | None) -> dict[str, ProviderFactory]:
            del account
            return {"scripted": scripted, "deep": deep, "signing": signing}

        made = ModelCatalog(factories=per_account, account=account)
        self.catalogs.append(made)
        return made

    def _keep(self, provider: ScriptedProvider) -> ScriptedProvider:
        self.providers[provider.name] = provider
        return provider


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> _Script:
    """Make ``main`` build its catalog from a script, never a real provider."""
    made = _Script()
    monkeypatch.setattr(main, "ModelCatalog", made.catalog)
    monkeypatch.setenv("SAGENT_PROVIDER", "scripted")
    monkeypatch.delenv("SAGENT_ALLOW_PROVIDERS", raising=False)
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    return made


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[logging.Logger]:
    """Let a test change the logging setup, and put it back after."""
    root = logging.getLogger()
    ours = logging.getLogger("sagent")
    before = list(root.handlers)
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(logging, "lastResort", logging.lastResort)
    level, root_level = ours.level, root.level
    yield root
    for handler in root.handlers:
        if handler not in before:
            handler.close()
    ours.setLevel(level)
    root.setLevel(root_level)


class _Tty(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


def _stdin(
    monkeypatch: pytest.MonkeyPatch,
    text: str = "",
    *,
    tty: bool = False,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Tty(text) if tty else io.StringIO(text))


def _argv(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["sagent", *argv])


def _quick_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``main``'s screen batch no prints; quitting waits out a batch."""
    monkeypatch.setattr(main, "run", functools.partial(main.run, print_batch_sec=0))


def _journal(
    directory: Path,
    *records: SessionRecord,
    mtime: float = 1_000.0,
) -> Path:
    """Write ``records`` as the journal of the session ``directory``."""
    path = directory / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = record_classes()
    path.write_text(
        "".join(json_dumps(r, classes=classes, sort=False) + "\n" for r in records),
    )
    os.utime(path, (mtime, mtime))
    return path


# ---- flags -----------------------------------------------------------------


def test_the_flags_default_to_a_fresh_interactive_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SAGENT_PROVIDER", "SAGENT_ALLOW_PROVIDERS", "SAGENT_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    flags = _flags()
    assert (flags.command, flags.allow_providers) == (None, "")
    assert (flags.system_prompt, flags.append_system_prompt) == (None, "")
    assert flags.compact is True
    assert (flags.tools, flags.name) == (None, "Agent")
    assert (flags.max_budget_usd, flags.max_tool_call_rounds) == (None, None)
    assert (flags.session, flags.ephemeral, flags.resume, flags.continue_) == (
        None,
        False,
        None,
        False,
    )
    assert (flags.resume_all, flags.continue_all, flags.resume_limit) == (
        False,
        False,
        20,
    )
    assert (
        flags.history,
        flags.output_format,
        flags.input_format,
        flags.log_level,
    ) == (None, "text", "text", None)
    assert flags.resume_persistent is True
    assert flags.add_dir == []


def test_the_environment_supplies_provider_allow_list_and_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGENT_PROVIDER", "deep")
    monkeypatch.setenv("SAGENT_ALLOW_PROVIDERS", "deep,scripted")
    monkeypatch.setenv("SAGENT_LOG_LEVEL", "INFO")
    flags = _flags()
    assert (flags.allow_providers, flags.log_level) == ("deep,scripted", "INFO")
    assert main._provider_name(_Script().catalog(), _flags()) == "deep"
    monkeypatch.setenv("SAGENT_PROVIDER", "sig")
    catalog = _Script().catalog()
    assert main._provider_name(catalog, _flags()) == "signing"
    assert list(catalog.factories) == ["deep", "scripted", "signing"]


def test_the_parser_names_every_option_it_knows_so_the_rest_are_settings() -> None:
    static = main._add_arguments(argparse.ArgumentParser())
    assert {"-h", "--help", "--resume", "--no-compact", "--allow-providers"} <= static
    assert {"--model", "--provider", "--effort", "--tool"}.isdisjoint(static)


def test_the_flags_parse_into_their_types() -> None:
    flags = _flags(
        "login",
        "--no-compact",
        "--tools",
        "Read",
        "Bash",
        "--max-budget-usd",
        "2.5",
        "--max-tool-call-rounds",
        "3",
        "--session",
        "s.jsonl",
        "--history",
        "h",
        "--resume-limit",
        "5",
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--name",
        "lead",
        "--no-resume-persistent",
        "--add-dir",
        "a",
        "b",
    )
    assert (flags.command, flags.compact) == ("login", False)
    assert flags.tools == ["Read", "Bash"]
    assert (flags.max_budget_usd, flags.max_tool_call_rounds) == (2.5, 3)
    assert (flags.session, flags.history, flags.resume_limit) == (
        Path("s.jsonl"),
        Path("h"),
        5,
    )
    assert (flags.output_format, flags.input_format, flags.name) == (
        "stream-json",
        "stream-json",
        "lead",
    )
    assert flags.resume_persistent is False
    assert flags.add_dir == [Path("a"), Path("b")]


def test_resume_takes_an_optional_id() -> None:
    assert _flags("--resume").resume == ""
    assert _flags("--resume", "3f2a").resume == "3f2a"
    assert _flags("--continue").continue_ is True


@pytest.mark.parametrize(
    "pair",
    [
        ["--session", "x", "--ephemeral"],
        ["--resume", "--continue"],
        ["--continue", "--resume-all"],
        ["--resume-all", "--continue-all"],
        ["--ephemeral", "--continue-all"],
    ],
)
def test_the_session_selectors_exclude_each_other(pair: list[str]) -> None:
    with pytest.raises(SystemExit):
        _flags(*pair)


@pytest.mark.parametrize(
    "argv",
    [
        ["--output-format", "yaml"],
        ["--input-format", "json"],
        ["--log-level", "LOUD"],
        ["--resume-limit", "0"],
        ["--resume-limit", "-2"],
        ["--resume-limit", "x"],
        ["--max-tool-call-rounds", "0"],
        ["--max-budget-usd", "0"],
        ["--max-budget-usd", "-1"],
        ["--max-budget-usd", "nan"],
        ["--max-budget-usd", "inf"],
        ["--max-budget-usd", "x"],
    ],
)
def test_a_bad_value_is_a_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _flags(*argv)


@pytest.mark.parametrize(
    ("parse", "raw", "error"),
    [
        (main._positive_int, "x", "expected an integer, got 'x'"),
        (main._positive_int, "0", "must be >= 1, got 0"),
        (main._positive_float, "x", "expected a number, got 'x'"),
        (main._positive_float, "-1", "must be finite and > 0, got -1"),
        (main._positive_float, "nan", "must be finite and > 0, got nan"),
        (main._positive_float, "inf", "must be finite and > 0, got inf"),
    ],
)
def test_a_count_or_amount_says_why_it_was_refused(
    parse: Callable[[str], object],
    raw: str,
    error: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=f"^{error}$"):
        parse(raw)


def test_a_count_or_amount_that_fits_is_taken() -> None:
    assert main._positive_int("1") == 1
    assert main._positive_float("0.01") == 0.01


# ---- main ------------------------------------------------------------------


def test_an_unknown_command_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "logout")
    with pytest.raises(SystemExit) as exit_info:
        main.main()
    assert exit_info.value.code == 2
    assert "unknown command 'logout'; the only one is login" in capsys.readouterr().err


def test_what_a_setting_flag_says_is_dropped() -> None:
    """A flag applies before any screen exists; only a failure, which raises, shows."""
    assert main._dropped(CommandOutput(text="[/effort] high", tone="note")) is None


def test_a_setting_flag_with_no_value_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "--effort")
    with pytest.raises(SystemExit) as exit_info:
        main.main()
    assert exit_info.value.code == 2
    assert "error: --effort needs a value" in capsys.readouterr().err


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, asyncio.CancelledError])
def test_an_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
) -> None:
    async def interrupted(
        flags: main.Flags,
        *,
        given: Sequence[tuple[str, str]],
    ) -> int:
        del flags, given
        raise interrupt

    _argv(monkeypatch)
    monkeypatch.setattr(main, "_main", interrupted)
    assert main.main() == 130


def test_piped_input_is_answered_once_and_printed(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script.replies.append(reply("the answer"))
    _argv(monkeypatch, "--provider", "scripted", "--model", "scripted", "--ephemeral")
    _stdin(monkeypatch, "  the question \n")

    def welcome(**_kwargs: object) -> None:
        pytest.fail("Piped input must not show the welcome")

    monkeypatch.setattr(main, "print_welcome", welcome)
    assert main.main() == 0
    out, err = capsys.readouterr()
    assert (out, err) == ("the answer\n", "")
    provider = script.providers["scripted"]
    (request,) = provider.requests
    assert [r.content for r in request.records if isinstance(r, UserMessage)] == [
        "the question",
    ]


def test_a_signal_mid_answer_exits_130_without_a_traceback(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupted(request: Request) -> Response:
        del request
        signal.raise_signal(signal.SIGINT)
        return reply("too late")

    script.replies.append(interrupted)
    _argv(monkeypatch, "--provider", "scripted", "--model", "scripted", "--ephemeral")
    _stdin(monkeypatch, "go")
    assert main.main() == 130
    out, err = capsys.readouterr()
    assert "too late" not in out
    assert "Traceback" not in err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_session_open_elsewhere_is_refused_without_a_traceback(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "000000000001" / "journal.jsonl"
    path.parent.mkdir()
    held = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        _argv(
            monkeypatch,
            "--provider",
            "scripted",
            "--model",
            "scripted",
            "--session",
            str(path),
        )
        _stdin(monkeypatch, "hi")
        assert main.main() == 1
    finally:
        os.close(held)
    assert capsys.readouterr().err == (
        f"Error: Journal {path} is open in another process.\n"
    )
    assert script.providers["scripted"].requests == []


def test_the_journal_is_written_where_resume_finds_it(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script.replies.append(reply("noted"))
    monkeypatch.chdir(tmp_path)
    _argv(monkeypatch, "--provider", "scripted", "--model", "scripted")
    _stdin(monkeypatch, "remember this")
    assert main.main() == 0
    (journal,) = project_dir(tmp_path).glob("*/session.jsonl")
    assert "remember this" in journal.read_text()
    assert "noted" in journal.read_text()


def test_a_resumed_session_comes_back_on_its_model_and_prompt_and_a_flag_wins(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script.capability = ModelCapability(
        model_id="scripted",
        thinking_effort=frozenset({"none", "low", "high"}),
        default=ModelSettings(context_size=PriceCatalog()["auto", "fullcontext"]),
    )
    monkeypatch.chdir(tmp_path)
    runs = (
        ("--effort", "high", "--system-prompt", "mine"),
        ("--continue",),
        ("--continue", "--effort", "low"),
    )
    stood: list[str] = []
    for given in runs:
        script.replies.append(reply("ok"))
        _argv(monkeypatch, "--provider", "scripted", "--model", "scripted", *given)
        _stdin(monkeypatch, "hi")
        assert main.main() == 0
        (path,) = project_dir(tmp_path).glob("*/session.jsonl")
        stood.append(_stood_model(path))

    assert stood == [
        "scripted/scripted+fullcontext+highthink",
        "scripted/scripted+fullcontext+highthink",
        "scripted/scripted+fullcontext+lowthink",
    ]
    (last,) = script.providers["scripted"].requests
    assert last.system[0].endswith("mine")


def test_a_resumed_session_comes_back_on_its_account(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The catalog takes it too, so the next ``/model`` signs in as it."""
    monkeypatch.chdir(tmp_path)
    for given in (("--account", "work"), ("--continue",)):
        script.replies.append(reply("ok"))
        _argv(monkeypatch, "--provider", "scripted", *given)
        _stdin(monkeypatch, "hi")
        assert main.main() == 0
    (path,) = project_dir(tmp_path).glob("*/session.jsonl")

    assert (
        _stood_model(path)
        == "work@scripted/scripted+fullcontext+nothink"  # codespell:ignore nothink
    )
    assert script.catalogs[-1].account == "work"


def _stood_model(path: Path) -> str:
    """Return the model line ``path``'s journal stands last."""
    lines = [
        record.payload["model"]
        for record in (read_record(line) for line in path.read_bytes().splitlines())
        if isinstance(record, UncategorizedRecord) and record.kind == "model"
    ]
    model = lines[-1]
    assert isinstance(model, str)
    return model


def _old_session(directory: Path, text: str) -> Path:
    """Write an old sagent's ``session.jsonl`` in which the user said ``text``."""
    log = directory / "session.jsonl"
    log.parent.mkdir(parents=True)
    _ = log.write_text(
        json_dumps({"kind": "meta", "session_id": directory.name})
        + "\n"
        + json_dumps({"kind": "history", "type": "user", "text": text})
        + "\n",
    )
    return log


def test_an_old_session_resumes_by_id_converted_in_place(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = _old_session(project_dir(tmp_path) / "ab1200000000", "remember 42")
    script.replies.append(reply("42"))
    monkeypatch.chdir(tmp_path)
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--resume",
        "ab",
    )
    _stdin(monkeypatch, "what was it?")
    assert main.main() == 0
    assert capsys.readouterr().err == f"[resume] {log}\n"
    (request,) = script.providers["scripted"].requests
    assert [r.content for r in request.records if isinstance(r, UserMessage)] == [
        "remember 42\n\nwhat was it?",
    ]
    assert "what was it?" in log.read_text()
    assert not log.with_name("journal.jsonl").exists()


def test_an_old_session_with_nothing_said_resumes_fresh(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log = project_dir(tmp_path) / "ab1200000000" / "session.jsonl"
    log.parent.mkdir(parents=True)
    _ = log.write_text(
        json_dumps({"kind": "meta", "session_id": "ab1200000000"}) + "\n",
    )
    script.replies.append(reply("hi"))
    monkeypatch.chdir(tmp_path)
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--resume",
        "ab",
    )
    _stdin(monkeypatch, "hello")
    assert main.main() == 0
    assert capsys.readouterr().err == (
        f"[resume] {log}\n[resume] nothing to resume; starting fresh.\n"
    )
    assert "hello" not in log.read_text()
    (journal,) = [
        j
        for j in project_dir(tmp_path).glob("*/session.jsonl")
        if j.parent != log.parent
    ]
    assert "hello" in journal.read_text()


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (
            ["--model", "nope"],
            "Error: [--model] UnknownModelError: No model 'nope'.\n",
        ),
        (
            ["--model", "deep", "--provider", "deep", "--effort", "max"],
            "Error: [--model.thinking.effort] 'max' is not one of: none, high\n",
        ),
        (
            ["--model", "deep/deep+xhigh"],
            (
                "Error: [--model] ValueError: thinking_effort='xhigh' is not offered"
                " by deep; allowed: 'high', 'none'\n"
            ),
        ),
        (
            ["--screen.thinking", "louder"],
            "Error: [--screen.thinking] 'louder' is not one of: default, show, hide\n",
        ),
        (
            ["--context", "1m"],
            "Error: [--model.context] UnknownModelError: No model 'scripted+1m'.\n",
        ),
        (["--nothing-at-all", "x"], "Error: --nothing-at-all names no setting"),
        (["--help.x", "x"], "Error: --help.x names no setting"),
        (["--tools", "Nope"], "Error: ValueError: unknown tools Nope; known: "),
        (
            ["--provider", "nowhere"],
            (
                "Error: [--model.provider] UnknownProviderError: unknown provider"
                " 'nowhere'; known: scripted, deep, signing\n"
            ),
        ),
        (
            ["--allow-providers", "deep,ghost"],
            "Error: UnknownProviderError: unknown providers ghost; known: scripted, deep, signing\n",
        ),
    ],
)
def test_a_flag_the_catalog_cannot_honor_is_an_error(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    error: str,
) -> None:
    _argv(monkeypatch, "--provider", "scripted", "--ephemeral", *argv)
    _stdin(monkeypatch, "unused")
    assert main.main() == 1
    assert capsys.readouterr().err.startswith(error)
    del script


@pytest.mark.parametrize(
    ("argv", "remote", "headless"),
    [
        ((), False, False),
        ((), True, True),
        (("--no-headless",), True, False),
        (("--headless",), False, True),
    ],
    ids=["local", "remote", "told-browser", "told-headless"],
)
def test_login_signs_in_headless_where_it_must_unless_told(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
    remote: bool,
    headless: bool,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    if remote:
        monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    _argv(monkeypatch, "login", "--provider", "signing", *argv)
    assert main.main() == 0
    provider = script.providers["signing"]
    assert isinstance(provider, _Signing)
    assert provider.signed_in == 1
    assert provider.headless == [headless]
    assert script.accounts == [None]


def test_login_headless_as_an_account_reaches_the_provider(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(
        monkeypatch,
        "login",
        "--provider",
        "signing",
        "--headless",
        "--account",
        "work",
    )
    assert main.main() == 0
    provider = script.providers["signing"]
    assert isinstance(provider, _Signing)
    assert provider.headless == [True]
    assert script.catalogs[0].account == "work"


def test_an_account_that_is_no_name_is_an_error(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "login", "--provider", "signing", "--account", "../x")

    assert main.main() == 1
    assert capsys.readouterr().err == (
        "Error: [--model.provider.account] ValueError: account '../x' is not a name:"
        " use letters, digits, '_', '-'\n"
    )
    assert "signing" not in script.providers


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (
            AuthError("the code was for another sign-in"),
            "Error: AuthError: the code was for another sign-in\n",
        ),
        (EOFError(), "Error: EOFError\n"),
    ],
    ids=["refused", "no-input"],
)
def test_a_failed_login_is_an_error_not_a_traceback(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    said: str,
) -> None:
    async def refuse(self: _Signing, *, headless: bool = False) -> None:
        del self, headless
        raise error

    monkeypatch.setattr(_Signing, "login", refuse)
    _argv(monkeypatch, "login", "--provider", "signing")

    assert main.main() == 1
    assert capsys.readouterr().err == said
    del script


def test_login_to_a_provider_without_one_is_an_error(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "login", "--provider", "scripted")
    assert main.main() == 1
    assert capsys.readouterr().err == "Error: scripted has no interactive sign-in.\n"
    del script


def test_the_allow_list_picks_the_provider_and_bounds_the_team(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGENT_PROVIDER")
    _argv(
        monkeypatch,
        "--allow-providers",
        " deep , scripted",
        "--model",
        "deep",
        "--ephemeral",
    )
    _stdin(monkeypatch, "hi")
    seen: list[str] = []
    monkeypatch.setattr(main, "_headless", _note_model(seen))
    assert main.main() == 0
    assert seen == ["deep/deep", "label=Agent"]
    (catalog,) = script.catalogs
    assert list(catalog.factories) == ["deep", "scripted"]


def test_the_provider_to_start_on_joins_the_allow_list(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(
        monkeypatch,
        "--allow-providers",
        "deep",
        "--model",
        "scripted",
        "--ephemeral",
    )
    _stdin(monkeypatch, "hi")
    seen: list[str] = []
    monkeypatch.setattr(main, "_headless", _note_model(seen))
    assert main.main() == 0
    assert seen == ["scripted/scripted", "label=Agent"]
    assert list(script.catalogs[0].factories) == ["deep", "scripted"]


def test_without_an_allow_list_every_provider_is_offered(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(monkeypatch, "--provider", "deep", "--model", "deep", "--ephemeral")
    _stdin(monkeypatch, "hi")
    seen: list[str] = []
    monkeypatch.setattr(main, "_headless", _note_model(seen))
    assert main.main() == 0
    assert seen == ["deep/deep", "label=Agent"]
    assert list(script.catalogs[0].factories) == ["scripted", "deep", "signing"]


def _note_model(seen: list[str]) -> Callable[[Agent, main.Flags], object]:
    async def headless(agent: Agent, flags: main.Flags) -> int:
        del flags
        seen.append(f"{agent.model.provider.name}/{agent.model.capability.model_id}")
        seen.append(f"label={agent.runtime.tool_context.label}")
        return 0

    return headless


def test_the_name_labels_the_root_agent(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--ephemeral",
        "--name",
        "lead",
    )
    _stdin(monkeypatch, "hi")
    seen: list[str] = []
    monkeypatch.setattr(main, "_headless", _note_model(seen))
    assert main.main() == 0
    assert seen == ["scripted/scripted", "label=lead"]
    del script


@pytest.mark.parametrize(
    ("argv", "resumed"),
    [
        ((), ["lead"]),
        (("--no-resume-persistent",), []),
        (("--tools", "Read"), []),
    ],
)
def test_resume_persistent_restarts_the_children_the_session_left_running(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
    resumed: list[str],
) -> None:
    labels: list[str] = []

    async def resume(self: AgentSpawn, parent_label: str) -> list[str]:
        del self
        labels.append(parent_label)
        return []

    monkeypatch.setattr(AgentSpawn, "resume", resume)
    monkeypatch.setattr(main, "_headless", _note_model([]))
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--ephemeral",
        "--name",
        "lead",
        *argv,
    )
    _stdin(monkeypatch, "hi")
    assert main.main() == 0
    assert labels == resumed
    del script


def test_add_dir_extends_where_agents_md_is_read(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--ephemeral",
        "--add-dir",
        str(tmp_path),
    )
    _stdin(monkeypatch, "hi")
    dirs: list[tuple[Path, ...]] = []

    async def headless(agent: Agent, flags: main.Flags) -> int:
        del flags
        dirs.append(agent.agents_md.additional_dirs)
        return 0

    monkeypatch.setattr(main, "_headless", headless)
    assert main.main() == 0
    assert dirs == [(tmp_path,)]
    del script


def test_the_fetch_session_is_closed_when_the_run_ends(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[_Closing] = []

    class _Closing(FetchSession):
        closed: bool = False

        @override
        def close(self) -> None:
            self.closed = True
            super().close()

    def make() -> FetchSession:
        session = _Closing()
        sessions.append(session)
        return session

    monkeypatch.setattr(main, "FetchSession", make)
    script.replies.append(reply("x"))
    _argv(monkeypatch, "--provider", "scripted", "--model", "scripted", "--ephemeral")
    _stdin(monkeypatch, "hi")
    assert main.main() == 0
    assert [s.closed for s in sessions] == [True]


def test_an_interactive_run_opens_the_prompt_and_names_the_model(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    logs: logging.Logger,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--history",
        str(tmp_path / "h"),
    )
    _stdin(monkeypatch, tty=True)
    _quick_prints(monkeypatch)
    welcomed: list[tuple[Path, bool]] = []

    def welcome(*, folder: Path, resumed: bool) -> None:
        welcomed.append((folder, resumed))

    monkeypatch.setattr(main, "print_welcome", welcome)
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        pipe.send_text("/quit\r")
        assert main.main() == 0
    assert welcomed == [(tmp_path, False)]
    err = capsys.readouterr().err
    assert err.startswith(
        "[/model] scripted/scripted+fullcontext+nothink\n",  # codespell:ignore nothink
    )
    (journal,) = project_dir(tmp_path).glob("*/session.jsonl")
    assert any(
        isinstance(h, logging.FileHandler)
        and h.baseFilename == str(journal.with_suffix(".log"))
        for h in logs.handlers
    )
    del script


# ---- settings --------------------------------------------------------------


_WIDE_PRICES = PriceCatalog(window=1_000_000)
_WIDE = ModelCapability(
    model_id="wide-1.0",
    limits=ModelLimits(max_request_tokens=1_000_000),
    prices=_WIDE_PRICES,
    thinking_effort=frozenset({"none", "high"}),
    thinking_budget=frozenset({"none", "auto"}),
    thinking_output=frozenset({"none", "text"}),
    default=ModelSettings(context_size=_WIDE_PRICES["auto", "fullcontext"]),
)
_SLIM_PRICES = PriceCatalog(window=100_000)
_SLIM = ModelCapability(
    model_id="slim-1.0",
    limits=ModelLimits(max_request_tokens=100_000),
    prices=_SLIM_PRICES,
    thinking_effort=frozenset({"none", "low"}),
    default=ModelSettings(context_size=_SLIM_PRICES["auto", "fullcontext"]),
)


async def _set(*argv: str) -> tuple[main._Screen, Agent]:
    """Apply the setting flags ``argv`` to an agent of ``wide``; ``slim`` serves others."""
    catalog = ModelCatalog(
        factories={
            "wide": partial(ScriptedProvider, models=ModelTable(rows=(_WIDE,))),
            "slim": partial(
                ScriptedProvider,
                models=ModelTable(rows=(_SLIM,), roles={"default": "slim-1.0"}),
            ),
        },
    )
    agent = Agent(model=catalog["wide"]["wide"], tools=[Read()])
    screen = main._Screen()
    _, given = flags(argv, static=frozenset())
    await main._configure(agent, catalog=catalog, screen=screen, given=given)
    return screen, agent


@pytest.mark.asyncio
async def test_each_setting_flag_applies_as_its_slash_command_would() -> None:
    screen, agent = await _set(
        "--context",
        "200k",
        "--effort",
        "high",
        "--thinking",
        "budget=auto",
        "--screen.thinking",
        "hide",
        "--Read.output.lines",
        "1,1",
    )
    settings = agent.model.settings
    assert screen.show_thinking is False
    assert (
        settings.context_size.tokens,
        settings.thinking_effort,
        settings.thinking_budget,
    ) == (200_000, "high", "auto")
    assert agent.tools["Read"].display.output.lines == (1, 1)


@pytest.mark.asyncio
async def test_what_the_model_s_name_says_wins_over_a_flag_given_after_it() -> None:
    _, agent = await _set(
        "--model",
        "wide+258k+high",
        "--context",
        "200k",
        "--effort",
        "none",
    )
    settings = agent.model.settings
    assert (settings.context_size.tokens, settings.thinking_effort) == (258_000, "high")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("--model", "slim/slim-1.0", "--effort", "low"),
        ("--effort", "low", "--model", "slim/slim-1.0"),
        ("--provider", "slim", "--effort", "low"),
        ("--provider", "slim", "--model", "slim-1.0", "--effort", "low"),
        ("--model", "slim-1.0", "--effort", "low", "--provider", "slim"),
    ],
)
async def test_a_flag_is_checked_against_the_model_named_not_the_one_started_on(
    argv: tuple[str, ...],
) -> None:
    _, agent = await _set(*argv)
    assert (agent.model.capability.model_id, agent.model.settings.thinking_effort) == (
        "slim-1.0",
        "low",
    )


@pytest.mark.asyncio
async def test_the_named_model_s_tags_win_over_a_flag_it_alone_offers() -> None:
    _, agent = await _set("--model", "slim/slim-1.0+none", "--effort", "low")
    assert agent.model.settings.thinking_effort == "none"


@pytest.mark.asyncio
async def test_no_setting_flag_leaves_the_model_s_own_and_shows_reasoning() -> None:
    screen, agent = await _set()
    assert screen.show_thinking is True
    settings = agent.model.settings
    assert (settings.context_size.tokens, settings.thinking_budget) == (
        1_000_000,
        "none",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ("--thinking", "budget=fixed"),
            "[--model.thinking] budget: 'fixed' is not one of: none, auto",
        ),
        (("--compact", "x"), "--compact names no setting; see /help"),
    ],
)
async def test_a_refused_setting_flag_says_why(
    argv: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(SettingError, match=f"^{re.escape(message)}$"):
        await _set(*argv)


# ---- tools -----------------------------------------------------------------


def _tools(*argv: str, cwd: Path = Path("/w")) -> dict[str, object]:
    catalog = _Script().catalog()
    team = Team(catalog=catalog)
    tools = main._tools(
        _flags(*argv),
        cwd=cwd,
        team=team,
        session=FetchSession(),
    )
    return {tool.name: tool for tool in tools}


_DEFAULT_TOOLS = [
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
    "Memory",
    "PlayAudio",
    "AgentSpawn",
    "AgentSend",
    "AgentSelf",
    "Session",
    "Skill",
    "BackgroundTask",
]


def test_every_tool_loads_by_default() -> None:
    assert list(_tools()) == _DEFAULT_TOOLS


def test_an_ephemeral_session_keeps_no_memories() -> None:
    assert "Memory" not in _tools("--ephemeral")
    with pytest.raises(ValueError, match="--ephemeral saves no memories"):
        _tools("--ephemeral", "--tools", "Memory")


def test_tools_loads_only_the_ones_named_or_none() -> None:
    assert list(_tools("--tools", "Read", "Bash")) == ["Read", "Bash"]
    assert _tools("--tools", "none") == {}


def test_an_unknown_tool_lists_the_known_ones() -> None:
    with pytest.raises(
        ValueError,
        match=r"^unknown tools Nope, Zed; known: Bash, Read, ",
    ) as error:
        _tools("--tools", "Zed", "Read", "Nope")
    assert str(error.value).endswith(", Skill, BackgroundTask")


def test_the_web_tools_share_one_session_and_bash_knows_its_peers() -> None:
    catalog = _Script().catalog()
    session = FetchSession()
    team = Team(catalog=catalog)
    tools = {
        t.name: t
        for t in main._tools(
            _flags(),
            cwd=Path("/w"),
            team=team,
            session=session,
        )
    }
    web = [WebFetch, WebSearch, PaperSearch, PaperDetails, PaperAuthor, PaperFetch]
    for kind in web:
        tool = tools[kind.__name__]
        assert isinstance(tool, kind)
        assert tool.session is session
    for kind in (AgentSpawn, AgentSend, AgentSelf, Session):
        tool = tools[kind.__name__]
        assert isinstance(tool, kind)
        assert tool.team is team
    bash = tools["Bash"]
    assert isinstance(bash, Bash)
    assert [p.name for p in bash.peers] == [
        "Read",
        "Edit",
        "Grep",
        "Glob",
        "List",
        "WebFetch",
    ]
    assert all(peer is tools[peer.name] for peer in bash.peers)


def test_memories_live_in_the_project_directory(tmp_path: Path) -> None:
    memory = _tools("--tools", "Memory", cwd=tmp_path)["Memory"]
    assert isinstance(memory, Memory)
    assert memory.directory == project_dir(tmp_path) / "memory"


# ---- journal paths ---------------------------------------------------------


def test_an_ephemeral_session_has_no_journal() -> None:
    assert (
        main._journal_path(_flags("--ephemeral"), cwd=Path("/w"), interactive=True)
        is None
    )


def test_session_names_the_journal(tmp_path: Path) -> None:
    path = tmp_path / "mine.jsonl"
    assert (
        main._journal_path(
            _flags("--session", str(path)),
            cwd=Path("/w"),
            interactive=True,
        )
        == path
    )


def test_a_fresh_session_gets_a_new_journal_in_the_project(tmp_path: Path) -> None:
    path = main._journal_path(_flags(), cwd=tmp_path, interactive=False)
    assert path is not None
    assert path.parent.parent == project_dir(tmp_path)
    assert path.read_bytes() == b""


def _said(text: str) -> tuple[SessionRecord, ...]:
    return (UserMessage(content=text), AssistantMessage(content="ok"))


def test_continue_resumes_this_directory_s_latest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    here = project_dir(tmp_path)
    _journal(here / "000000000001", *_said("a"), mtime=1)
    latest = _journal(here / "000000000002", *_said("b"), mtime=2)
    _journal(project_dir(tmp_path / "other") / "000000000003", *_said("c"), mtime=3)
    assert (
        main._journal_path(_flags("--continue"), cwd=tmp_path, interactive=False)
        == latest
    )
    assert capsys.readouterr().err == f"[resume] {latest}\n"


def test_continue_all_resumes_the_latest_anywhere_reading_one_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(30):
        _journal(
            project_dir(tmp_path / f"p{i % 3}") / f"{i:012}",
            *_said(str(i)),
            mtime=1_000 + i,
        )
    peeked: list[Path] = []

    peek = projects._peek

    def counting(path: Path, *, mtime: float) -> projects.SessionInfo | None:
        peeked.append(path)
        return peek(path, mtime=mtime)

    monkeypatch.setattr(projects, "_peek", counting)
    found = main._journal_path(
        _flags("--continue-all"),
        cwd=tmp_path,
        interactive=False,
    )
    assert found is not None
    assert found.parent.name == "000000000029"
    assert len(peeked) == 1


@pytest.mark.parametrize(
    "argv",
    [["--continue"], ["--continue-all"], ["--resume"], ["--resume-all"]],
)
def test_nothing_to_resume_starts_fresh_and_says_so(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    path = main._journal_path(_flags(*argv), cwd=tmp_path, interactive=True)
    assert path is not None
    assert path.parent.parent == project_dir(tmp_path)
    assert capsys.readouterr().err == "[resume] nothing to resume; starting fresh.\n"


def test_resume_by_prefix_prefers_this_directory(tmp_path: Path) -> None:
    here = _journal(project_dir(tmp_path) / "ab1200000000", *_said("here"))
    _journal(project_dir(tmp_path / "other") / "ab3400000000", *_said("there"))
    assert (
        main._journal_path(_flags("--resume", "ab"), cwd=tmp_path, interactive=False)
        == here
    )


def test_resume_by_prefix_looks_in_every_project_next(tmp_path: Path) -> None:
    there = _journal(
        project_dir(tmp_path / "other") / "cd3400000000",
        *_said("there"),
    )
    assert (
        main._journal_path(_flags("--resume", "cd"), cwd=tmp_path, interactive=False)
        == there
    )


@pytest.mark.parametrize(
    ("layout", "error"),
    [
        (
            ("here/ab1000000000", "here/ab2000000000"),
            (
                "[resume] 'ab' is ambiguous in this directory:"
                " ab1000000000, ab2000000000.\n"
            ),
        ),
        (
            ("x/ab1000000000", "y/ab2000000000"),
            "[resume] 'ab' is ambiguous in all projects: ab1000000000, ab2000000000.\n",
        ),
        (
            ("here/ab1000000000", "old:here/ab2000000000"),
            (
                "[resume] 'ab' is ambiguous in this directory:"
                " ab1000000000, ab2000000000.\n"
            ),
        ),
        ((), "[resume] no session matching 'ab'.\n"),
    ],
)
def test_resume_by_prefix_refuses_to_guess(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    layout: tuple[str, ...],
    error: str,
) -> None:
    for entry in layout:
        kind, _, place = entry.rpartition(":")
        where, _, stem = place.partition("/")
        cwd = tmp_path if where == "here" else tmp_path / where
        if kind == "old":
            _old_session(project_dir(cwd) / stem, stem)
        else:
            _journal(project_dir(cwd) / f"{stem}", *_said(stem))
    with pytest.raises(SystemExit) as exit_info:
        main._journal_path(_flags("--resume", "ab"), cwd=tmp_path, interactive=False)
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == error


def test_resume_picks_from_this_directory_when_interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    here = project_dir(tmp_path)
    _journal(here / "00000000000a", *_said("first"), mtime=1)
    chosen = _journal(here / "00000000000b", *_said("second"), mtime=2)
    _journal(project_dir(tmp_path / "x") / "00000000000c", *_said("elsewhere"), mtime=3)
    _stdin(monkeypatch, "\n")
    assert (
        main._journal_path(_flags("--resume"), cwd=tmp_path, interactive=True) == chosen
    )


def test_resume_does_not_pick_when_piped(tmp_path: Path) -> None:
    _journal(project_dir(tmp_path) / "00000000000a", *_said("first"))
    path = main._journal_path(_flags("--resume"), cwd=tmp_path, interactive=False)
    assert path is not None
    assert path.parent.name != "00000000000a"


def test_resume_all_picks_from_every_project_reading_only_the_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for i in range(6):
        _journal(
            project_dir(tmp_path / f"p{i}") / f"{i:012}",
            *_said(f"title {i}"),
            mtime=1_000 + i,
        )
    _stdin(monkeypatch, "2\n")
    path = main._journal_path(
        _flags("--resume-all", "--resume-limit", "3"),
        cwd=tmp_path,
        interactive=True,
    )
    assert path is not None
    assert path.parent.name == "000000000004"
    err = capsys.readouterr().err
    assert "(showing 3 sessions; older ones hidden)" in err
    assert "title 5" in err
    assert "title 2" not in err


# ---- the picker ------------------------------------------------------------


def _infos(tmp_path: Path, *titles: str) -> list[projects.SessionInfo]:

    for i, title in enumerate(titles):
        records: tuple[SessionRecord, ...] = (UserMessage(content=title or None),)
        _journal(tmp_path / f"{i:012}", *records, mtime=1_800_000_000 - i)
    return list_sessions(tmp_path)


def test_the_picker_lists_age_size_and_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = _infos(tmp_path, "short", "x" * 61, "", "y" * 60)
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0 + 3 * 86_400)
    _stdin(monkeypatch, "1\n")
    assert main._pick(sessions, cap=5) is sessions[0]
    rows = capsys.readouterr().err.split("\n")
    assert rows[0] == "  [ 1]  3d ago ·    1 records · short"
    assert rows[1] == f"  [ 2]  3d ago ·    1 records · {'x' * 59}…"
    assert rows[2] == "  [ 3]  3d ago ·    1 records · (no user messages)"
    assert rows[3] == f"  [ 4]  3d ago ·    1 records · {'y' * 60}"
    assert rows[4] == "Resume which? [1] "


@pytest.mark.parametrize(
    ("typed", "picked"),
    [
        ("\n", 0),
        ("2\n", 1),
        ("  2 \n", 1),
        ("0\n", None),
        ("3\n", None),
        ("", None),
        ("abc\n", None),
        ("abc\n2\n", 1),
    ],
)
def test_the_picker_reads_a_number_or_takes_the_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    typed: str,
    picked: int | None,
) -> None:
    sessions = _infos(tmp_path, "a", "b")
    _stdin(monkeypatch, typed)
    chosen = main._pick(sessions, cap=5)
    assert chosen is (None if picked is None else sessions[picked])
    if "abc" in typed:
        assert "  not a number: 'abc'\n" in capsys.readouterr().err


def test_the_picker_with_nothing_asks_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main._pick([], cap=5) is None
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (-5, "0s ago"),
        (12, "12s ago"),
        (60, "1m ago"),
        (3_599, "59m ago"),
        (3_600, "1h ago"),
        (86_399, "23h ago"),
        (86_400, "1d ago"),
    ],
)
def test_ago_rounds_down_to_its_largest_unit(seconds: float, text: str) -> None:
    assert main._ago(seconds) == text


# ---- logging ---------------------------------------------------------------


def test_piped_logging_goes_to_stderr_only_when_asked(logs: logging.Logger) -> None:
    ours = logging.getLogger("sagent")
    before = ours.level
    main._logging(None, path=None, interactive=False)
    assert ours.level == before
    main._logging("ERROR", path=None, interactive=False)
    assert ours.level == logging.ERROR
    del logs


def test_interactive_logging_goes_to_the_session_s_file_never_stderr(
    logs: logging.Logger,
    tmp_path: Path,
) -> None:
    logs.addHandler(logging.StreamHandler(io.StringIO()))
    kept = logging.FileHandler(tmp_path / "kept.log")
    logs.addHandler(kept)
    main._logging(None, path=tmp_path / "s.jsonl", interactive=True)
    streams: list[logging.Handler] = [
        h for h in logs.handlers if type(h) is logging.StreamHandler
    ]
    files = [h for h in logs.handlers if isinstance(h, logging.FileHandler)]
    assert streams == []
    assert kept in files
    assert any(h.baseFilename == str(tmp_path / "s.log") for h in files)
    assert isinstance(logging.lastResort, logging.NullHandler)
    assert logging.getLogger("sagent").level == logging.DEBUG
    logging.getLogger("sagent.test").warning("recorded")
    for handler in files:
        handler.flush()
    assert "WARNING sagent.test: recorded" in (tmp_path / "s.log").read_text()


def test_interactive_logging_without_a_journal_uses_the_shared_file(
    logs: logging.Logger,
) -> None:
    main._logging("WARNING", path=None, interactive=True)
    log = data_dir() / "rekursiv-ai" / "sagent" / "repl.log"
    assert any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log)
        for h in logs.handlers
    )
    assert logging.getLogger("sagent").level == logging.WARNING


# ---- signals ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_first_signal_cancels_the_run_and_the_second_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)

    async def guarded() -> None:
        with main._signals():
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(10)

    task = asyncio.create_task(guarded())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert exits == [1]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ---- headless --------------------------------------------------------------


async def _headless(
    stdin: str,
    *argv: str,
    replies: Sequence[Reply] = (),
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    _stdin(monkeypatch, stdin)
    provider = ScriptedProvider(replies=replies)
    async with Agent(model=provider["scripted"], tools=[Read()]) as agent:
        before = list(agent.runtime.observers)
        code = await main._headless(agent, _flags(*argv))
        assert agent.runtime.observers == before
    return code


@pytest.mark.asyncio
async def test_headless_text_prints_the_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        await _headless("question", replies=[reply("answer")], monkeypatch=monkeypatch)
        == 0
    )
    assert capsys.readouterr() == ("answer\n", "")


@pytest.mark.asyncio
async def test_headless_json_prints_the_reply_as_one_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await _headless(
        "q",
        "--output-format",
        "json",
        replies=[reply("answer")],
        monkeypatch=monkeypatch,
    )
    assert code == 0
    assert json_loads(capsys.readouterr().out) == {"content": "answer"}


@pytest.mark.asyncio
async def test_headless_stream_json_prints_every_event_then_the_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha")
    monkeypatch.chdir(tmp_path)
    replies = [
        reply(
            "looking",
            calls=[call("Read", "c1", file_path="a.txt"), call("Gone", "c2")],
        ),
        reply("done"),
    ]
    code = await _headless(
        "q",
        "--output-format",
        "stream-json",
        replies=replies,
        monkeypatch=monkeypatch,
    )
    assert code == 0
    records = [
        json_loads(line, dict[str, object])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert records[0] == {"descriptor": "text/plain", "content": "looking"}
    assert records[1]["descriptor"] == "text/x-tool-label"
    assert records[1]["call_id"] == "c1"
    assert records[2] == {
        "descriptor": "text/x-tool-label",
        "call_id": "c2",
        "content": "Gone",
    }
    results = {
        r["call_id"]: r
        for r in records
        if r.get("descriptor") == "application/x-tool-result"
    }
    assert results["c1"]["is_error"] in (None, False)
    assert "alpha" in str(results["c1"]["content"])
    assert results["c2"]["is_error"] is True
    assert records[-2] == {"descriptor": "text/plain", "content": "done"}
    assert records[-1] == {"content": "done", "descriptor": "result"}


@pytest.mark.asyncio
async def test_headless_stream_json_input_joins_the_prompts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[list[str | None]] = []

    def answer(request: Request) -> Response:
        seen.append([r.content for r in request.records if isinstance(r, UserMessage)])
        return reply("ok")

    stdin = '{"prompt": "one"}\n\n{}\n{"prompt": ""}\n{"prompt": "two", "x": 1}\n'
    code = await _headless(
        stdin,
        "--input-format",
        "stream-json",
        replies=[answer],
        monkeypatch=monkeypatch,
    )
    assert code == 0
    assert seen == [["one\n\ntwo"]]
    assert capsys.readouterr().out == "ok\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("line", ["not json", "[1, 2]", '"text"', '{"prompt": 3}'])
async def test_headless_stream_json_rejects_a_line_that_is_no_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    line: str,
) -> None:
    assert (
        await _headless(line, "--input-format", "stream-json", monkeypatch=monkeypatch)
        == 1
    )
    assert capsys.readouterr().err.startswith("Error: invalid stream-json input: ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdin", "argv"),
    [("", ()), ("  \n", ()), ("{}\n", ("--input-format", "stream-json"))],
)
async def test_headless_with_nothing_to_answer_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stdin: str,
    argv: tuple[str, ...],
) -> None:
    assert await _headless(stdin, *argv, monkeypatch=monkeypatch) == 1
    assert capsys.readouterr() == ("", "Error: no input on stdin.\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argv", "out", "err"),
    [
        ((), "", "Error: AuthError: expired\n"),
        (("--output-format", "json"), '{"error": "AuthError: expired"}\n', ""),
    ],
)
async def test_a_failed_turn_is_reported_and_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    out: str,
    err: str,
) -> None:
    assert (
        await _headless(
            "q",
            *argv,
            replies=[AuthError("expired")],
            monkeypatch=monkeypatch,
        )
        == 1
    )
    assert capsys.readouterr() == (out, err)


@pytest.mark.asyncio
async def test_a_polished_failure_has_no_class_prefix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    budget = BudgetExhaustedError(total_cost_usd=2.0, max_budget_usd=1.0)
    assert await _headless("q", replies=[budget], monkeypatch=monkeypatch) == 1
    assert capsys.readouterr().err == "Error: Budget exhausted: $2.00 >= $1.00\n"


@pytest.mark.asyncio
async def test_a_failure_while_streaming_is_its_own_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await _headless(
        "q",
        "--output-format",
        "stream-json",
        replies=[AuthError("expired")],
        monkeypatch=monkeypatch,
    )
    assert code == 1
    records = [
        json_loads(line, dict[str, object])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert {
        "descriptor": "application/x-error",
        "content": "AuthError: expired",
    } in records
    assert records[-1] == {"error": "AuthError: expired"}


def test_records_streams_each_kind_of_event_and_drops_the_rest() -> None:
    tools = {"Read": Read()}
    assert main._records(types.ModelResponseThinking(text="hm"), tools=tools) == [
        {"descriptor": "text/x-thinking", "content": "hm"},
    ]
    suspended = types.ModelCallSuspended(
        exception=RateLimitedError("spent", limit="weekly"),
        until=time.time() + 30.0,
        attempt=1,
    )
    ((record,),) = [main._records(suspended, tools=tools)]
    assert record["descriptor"] == "text/x-notice"
    assert str(record["content"]).startswith("Weekly limit reached. Resumes ")
    result = ShellCommandResult(call_id="1", stdout="out", exit_code=2, is_error=True)
    assert main._records(result, tools=tools) == [
        {
            "descriptor": "application/x-tool-result",
            "call_id": "1",
            "content": "out\n[exit 2]",
            "is_error": True,
        },
    ]
    assert main._records(types.AgentIdle(), tools=tools) == []
    assert main._records(UserMessage(content="x"), tools=tools) == []


def test_a_detached_call_streams_its_stub_marked_then_its_result() -> None:
    """A reader can tell the stand-in from the result that replaces it."""
    tools = {"Read": Read()}
    stub = UncategorizedToolResult(call_id="1", content="[detached]", stub="pending")
    real = UncategorizedToolResult(call_id="1", content="done")
    assert main._records(stub, tools=tools) == [
        {
            "descriptor": "application/x-tool-result",
            "call_id": "1",
            "content": "[detached]",
            "is_error": None,
            "stub": "pending",
        },
    ]
    assert main._records(types.DetachedResult(result=real), tools=tools) == [
        {
            "descriptor": "application/x-tool-result",
            "call_id": "1",
            "content": "done",
            "is_error": None,
        },
    ]
    assert (
        main._records(
            types.ModelResponseComplete(response=reply("no calls")),
            tools=tools,
        )
        == []
    )


# ---- the interactive screen ------------------------------------------------


@pytest.mark.asyncio
async def test_run_redraws_the_session_then_answers_what_is_typed(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content="earlier"), AssistantMessage(content="before"))
    provider = ScriptedProvider(replies=[reply("hi there")])
    screen = io.StringIO()
    async with Agent(model=provider["scripted"], journal=journal) as agent:
        before = list(agent.runtime.observers)
        hook = agent.runtime.before_tool_spawn
        answered = asyncio.Event()

        def watch(event: types.RuntimeEvent) -> None:
            if isinstance(event, types.AgentIdle) and provider.requests:
                answered.set()

        agent.runtime.observers.append(watch)
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            running = asyncio.create_task(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "new" / "history",
                    print_batch_sec=0,
                ),
            )
            await asyncio.sleep(0.05)
            pipe.send_text("hello\r")
            await asyncio.wait_for(answered.wait(), timeout=5)
            pipe.send_text("/quit\r")
            await asyncio.wait_for(running, timeout=5)
        agent.runtime.observers.remove(watch)
        assert agent.runtime.observers == before
        assert agent.runtime.before_tool_spawn is hook
    shown = screen.getvalue()
    for text in ("earlier", "before", "resumed · 2 messages", "> hello", "hi there"):
        assert text in shown
    assert shown.index("earlier") < shown.index("> hello") < shown.index("hi there")
    assert "Resume this session with" not in shown
    assert "hello" in (tmp_path / "new" / "history").read_text()


@pytest.mark.asyncio
async def test_run_says_how_to_resume_a_journaled_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "abc123" / "journal.jsonl"
    screen = io.StringIO()
    async with (
        JsonlJournal(path) as journal,
        Agent(model=ScriptedProvider()["scripted"], journal=journal) as agent,
    ):
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text("\x04")
            await asyncio.wait_for(
                main.run(agent, catalog=_Script().catalog(), print_batch_sec=0),
                timeout=5,
            )
    assert capsys.readouterr().err.split("\n")[:2] == [
        "Resume this session with:",
        "sagent --resume abc123   # this session",
    ]
    assert (state_dir() / "rekursiv-ai" / "sagent").is_dir()


@pytest.mark.asyncio
async def test_run_reports_a_failing_command_and_goes_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken(Commands):
        @override
        async def __call__(self, line: str) -> bool:
            if line == "/boom":
                # Leaves run's observers gone too: its cleanup must not trip.
                self.agent.runtime.observers.clear()
                raise RuntimeError("kaboom")
            return await super().__call__(line)

    monkeypatch.setattr(main, "Commands", _Broken)
    screen = io.StringIO()
    async with Agent(model=ScriptedProvider()["scripted"]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text("/boom\r/exit\r")
            await asyncio.wait_for(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
                timeout=5,
            )
    assert "✗ RuntimeError: kaboom" in screen.getvalue()


@pytest.mark.asyncio
async def test_a_quit_with_words_after_it_still_quits(tmp_path: Path) -> None:
    screen = io.StringIO()
    async with Agent(model=ScriptedProvider()["scripted"]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text("/help\r/quit now\r")
            await asyncio.wait_for(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
                timeout=5,
            )


@pytest.mark.parametrize(
    ("argv", "system"),
    [
        ((), None),
        (("--system-prompt", "Be terse."), "Be terse."),
        (
            ("--system-prompt", "Be terse.", "--append-system-prompt", "Use tabs."),
            "Be terse.\n\nUse tabs.",
        ),
    ],
)
def test_the_system_prompt_is_replaced_or_extended(
    argv: tuple[str, ...],
    system: str | None,
) -> None:
    agent = main._agent(
        _flags(*argv),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    assert agent.system_prompt == system


def test_an_appended_prompt_extends_the_default_one() -> None:
    agent = main._agent(
        _flags("--append-system-prompt", "Use tabs."),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    assert agent.system_prompt == f"{default_system_prompt()}\n\nUse tabs."


@pytest.mark.parametrize(
    ("argv", "compactor"),
    [((), SummaryCompactor), (("--no-compact",), NoShrinker)],
)
def test_compact_picks_the_compactor(argv: tuple[str, ...], compactor: type) -> None:
    agent = main._agent(
        _flags(*argv),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    assert type(agent.session.compactor) is compactor


@pytest.mark.parametrize(
    ("argv", "style"),
    [((), "summary"), (("--compactor", "session-audit"), "session-audit")],
)
def test_compactor_picks_how_it_compacts(argv: tuple[str, ...], style: str) -> None:
    agent = main._agent(
        _flags(*argv),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    assert compactor.style == style


def test_the_caps_reach_the_agent() -> None:
    agent = main._agent(
        _flags("--max-tool-call-rounds", "3", "--max-budget-usd", "1.5"),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    assert (agent.max_rounds, agent.max_spend_usd) == (3, 1.5)


def test_the_usage_names_each_flag_s_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "2000")
    parser = argparse.ArgumentParser()
    main._add_arguments(parser)
    usage = parser.format_usage()
    for shown in (
        "--allow-providers LIST",
        "--tools NAME [NAME ...]",
        "--max-budget-usd USD",
        "--max-tool-call-rounds N",
        "--resume [ID]",
        "--resume-limit N",
        "--add-dir DIR [DIR ...]",
        "--history PATH",
        "--output-format {text,json,stream-json}",
        "--input-format {text,stream-json}",
        "--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}",
    ):
        assert shown in usage


def test_help_opens_with_the_module_s_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "--help")
    with pytest.raises(SystemExit) as exit_info:
        main.main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert (
        "Run a coding agent: interactively on a terminal, or once on piped input."
        in out
    )
    assert "python -m sagent login" in out
    assert "Every setting ``/help`` lists is a flag too" in out


@pytest.mark.parametrize("tool", _DEFAULT_TOOLS)
def test_each_tool_loads_by_its_own_name(tool: str) -> None:
    assert list(_tools("--tools", tool)) == [tool]


def test_the_ephemeral_refusal_says_what_to_drop() -> None:
    with pytest.raises(
        ValueError,
        match=r"^--ephemeral saves no memories; drop Memory from --tools\.$",
    ):
        _tools("--ephemeral", "--tools", "Memory")


def test_no_provider_named_anywhere_means_the_default_provider(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("SAGENT_PROVIDER")
    _argv(monkeypatch, "--ephemeral")
    _stdin(monkeypatch, "hi")
    assert main.main() == 1
    assert capsys.readouterr().err == (
        f"Error: UnknownProviderError: unknown provider {default_provider()!r};"
        " known: scripted, deep, signing\n"
    )
    del script


def test_every_unknown_allowed_provider_is_named(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _argv(monkeypatch, "--allow-providers", "zed,ghost,deep", "--ephemeral")
    _stdin(monkeypatch, "hi")
    assert main.main() == 1
    assert capsys.readouterr().err.startswith(
        "Error: UnknownProviderError: unknown providers ghost, zed; known:",
    )
    del script


def test_the_run_s_tools_and_team_share_its_session_and_catalog(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[FetchSession] = []

    def make() -> FetchSession:
        made.append(FetchSession())
        return made[-1]

    seen: list[bool] = []

    async def headless(agent: Agent, flags: main.Flags) -> int:
        del flags
        fetch, spawn = agent.tools["WebFetch"], agent.tools["AgentSpawn"]
        assert isinstance(fetch, WebFetch)
        assert isinstance(spawn, AgentSpawn)
        seen.append(fetch.session is made[0])
        seen.append(spawn.team.catalog is script.catalogs[0])
        seen.append(agent.runtime.tool_context.cwd == Path.cwd())
        return 0

    monkeypatch.setattr(main, "FetchSession", make)
    monkeypatch.setattr(main, "_headless", headless)
    _argv(monkeypatch, "--provider", "scripted", "--model", "scripted", "--ephemeral")
    _stdin(monkeypatch, "hi")
    assert main.main() == 0
    assert seen == [True, True, True]


def test_an_interactive_resume_picks_and_logs_at_the_level_asked(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    logs: logging.Logger,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    earlier = _journal(
        project_dir(tmp_path) / "000000000001",
        UserMessage(content="before"),
        Thinking(content="deliberating"),
        AssistantMessage(content="ok"),
    )
    screen = io.StringIO()
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--resume",
        "--log-level",
        "INFO",
        "--history",
        str(tmp_path / "h"),
    )
    _stdin(monkeypatch, "1\n", tty=True)
    _quick_prints(monkeypatch)
    welcomed: list[tuple[Path, bool]] = []

    def welcome(*, folder: Path, resumed: bool) -> None:
        welcomed.append((folder, resumed))

    monkeypatch.setattr(main, "print_welcome", welcome)
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=PlainTextOutput(screen)),
    ):
        pipe.send_text("/model deep/deep\r/quit\r")
        assert main.main() == 0
    assert welcomed == [(tmp_path, True)]
    assert f"[resume] {earlier}\n" in capsys.readouterr().err
    assert logging.getLogger("sagent").level == logging.INFO
    shown = screen.getvalue()
    assert "deliberating" in shown
    assert (
        "[/model] deep/deep+fullcontext+nothink\n" in shown  # codespell:ignore nothink
    )
    assert "/model deep/deep" in (tmp_path / "h").read_text()
    del script, logs


def test_piped_logging_sets_a_warning_root_with_a_timestamped_format(
    logs: logging.Logger,
) -> None:
    logs.handlers = []
    logs.setLevel(logging.DEBUG)
    main._logging("DEBUG", path=None, interactive=False)
    (handler,) = logs.handlers
    assert logs.level == logging.WARNING
    assert handler.formatter is not None
    assert handler.formatter._fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    assert logging.getLogger("sagent").level == logging.DEBUG


def test_a_session_log_line_has_time_level_and_logger(
    logs: logging.Logger,
    tmp_path: Path,
) -> None:
    main._logging(None, path=tmp_path / "s.jsonl", interactive=True)
    name = "sagent.test"
    logging.getLogger(name).info("hello")
    for handler in logs.handlers:
        handler.flush()
    (row,) = (tmp_path / "s.log").read_text().splitlines()
    assert re.fullmatch(
        rf"\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d,\d{{3}} INFO {re.escape(name)}: hello",
        row,
    )


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_every_log_level_is_accepted(level: str) -> None:
    assert _flags("--log-level", level).log_level == level


def test_text_is_a_format_on_both_ends() -> None:
    flags = _flags("--output-format", "text", "--input-format", "text")
    assert (flags.output_format, flags.input_format) == ("text", "text")
    assert _flags("--output-format", "json").output_format == "json"


@pytest.mark.asyncio
async def test_the_prompt_session_wears_the_screen_s_settings_for_the_agent_shown(
    tmp_path: Path,
) -> None:
    team = Team(catalog=_Script().catalog())
    root = Agent(model=ScriptedProvider()["scripted"])
    child = Agent(model=ScriptedProvider()["scripted"])
    _ = team.join(root)
    _ = team.join(child, label="kid", parent="Agent")
    printer = ConsolePrinter(Console(file=io.StringIO()))
    views = Views(root, team=team, printer=printer, catalog=_Script().catalog())
    entry = Input()
    session = main._prompt_session(views, entry=entry, history=tmp_path / "h")
    assert (
        session.multiline,
        session.app.erase_when_done,
        session.enable_open_in_editor,
    ) == (True, True, False)
    assert session.refresh_interval == 0
    assert isinstance(session.auto_suggest, AutoSuggestFromHistory)
    assert isinstance(session.history, FileHistory)
    assert session.history.filename == str(tmp_path / "h")
    assert session.style is not None
    assert session.style.style_rules == [
        ("bottom-toolbar", "fg:ansibrightblack noreverse bg:default"),
        ("queued_input_pane", "fg:ansibrightblack"),
        ("input_pane", "bold"),
    ]
    keys, message = session.key_bindings, session.message
    assert keys is not None
    assert callable(message)
    for label in ("Agent", "kid"):
        views.focus(label)
        shown = views.current
        shown.panes.stage(f"waiting on {label}", deferred=False)
        (enter,) = [b for b in keys.bindings if tuple(b.keys) == (Keys.ControlM,)]
        handler = enter.handler
        assert isinstance(handler, functools.partial)
        assert handler.args[:2] == (shown.agent.runtime, shown.panes)
        assert session.bottom_toolbar == views.toolbar
        assert message() == prompt_text(shown.deferred, entry)
    await views.aclose()


@pytest.mark.asyncio
async def test_run_wires_the_catalog_team_and_printer_into_its_commands(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = _Script().catalog()
    screen = io.StringIO()
    async with (
        Team(catalog=catalog) as team,
        Agent(model=ScriptedProvider()["scripted"]) as agent,
    ):
        team.join(agent)
        child = Agent(model=ScriptedProvider()["scripted"])
        team.join(child, label="kid", parent="Agent")
        agent.spend_usd = 1.25
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text("/model deep/deep\r/halt kid\r/help\r/quit\r")
            await asyncio.wait_for(
                main.run(
                    agent,
                    catalog=catalog,
                    team=team,
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
                timeout=5,
            )
        assert agent.model.capability.model_id == "deep"
        assert child.runtime.inbox.drain_nowait() == [types.Halt()]
    shown = screen.getvalue()
    assert (
        "[/model] deep/deep+fullcontext+nothink\n" in shown  # codespell:ignore nothink
    )
    assert "[/halt] halted kid" in shown
    assert "sagent commands" in shown
    assert "$1.25" in shown
    del caplog


_LEFT, _DOWN, _RIGHT = "\x1b[D", "\x1b[B", "\x1b[C"


@pytest.mark.asyncio
async def test_left_opens_the_tree_and_enter_shows_the_agent_picked(
    tmp_path: Path,
) -> None:
    """Keys the prompt leaves unread reach the tree, and the prompt after it."""
    catalog = _Script().catalog()
    screen = io.StringIO()
    kid_provider = ScriptedProvider(replies=[reply("kid answers")])
    async with (
        Team(catalog=catalog) as team,
        Agent(model=ScriptedProvider()["scripted"]) as agent,
        Agent(model=kid_provider["scripted"]) as kid,
    ):
        team.join(agent)
        team.join(kid, label="kid", parent="Agent")
        kid.session.append(AssistantMessage(content="kid earlier"))
        answered = asyncio.Event()
        kid.runtime.observers.append(
            lambda e: answered.set() if isinstance(e, types.AgentIdle) else None,
        )
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text(f"{_LEFT}{_RIGHT}{_LEFT}{_DOWN}\rhi kid\r/quit\r")
            await asyncio.wait_for(
                main.run(
                    agent,
                    catalog=catalog,
                    team=team,
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
                timeout=5,
            )
        await asyncio.wait_for(answered.wait(), timeout=5)
        assert UserMessage(content="hi kid") in kid_provider.requests[0].records
        # What was typed went to the kid alone; its parent hears only that it stopped.
        assert agent.runtime.inbox.drain_nowait() == [
            AgentToAgentMessage(
                sender="kid",
                recipient="Agent",
                content="[kid is idle] kid answers",
            ),
        ]
    shown = screen.getvalue()
    assert "Enter focus" in shown
    assert shown.index("── kid ──") < shown.index("kid earlier")


@pytest.mark.asyncio
async def test_what_the_shown_agent_did_under_the_tree_prints_once_it_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedProvider(replies=[reply("said under the tree")])
    screen = io.StringIO()
    printer = ConsolePrinter(Console(file=screen, width=80))
    async with Agent(model=provider["scripted"]) as agent:
        views = Views(agent, team=None, printer=printer, catalog=_Script().catalog())
        idle = asyncio.Event()
        agent.runtime.observers.append(
            lambda e: idle.set() if isinstance(e, types.AgentIdle) else None,
        )

        async def busy(*args: object, **kwargs: object) -> None:
            del args, kwargs
            agent.runtime.push(UserMessage(content="go"))
            await asyncio.wait_for(idle.wait(), timeout=5)
            assert "said under the tree" not in screen.getvalue()

        monkeypatch.setattr(main, "pick", busy)
        await main._tree(views)
        await views.aclose()
    assert "said under the tree" in screen.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("show", "shown"),
    [(None, True), (True, True), (False, False)],
)
async def test_run_shows_reasoning_unless_told_not_to(
    tmp_path: Path,
    show: bool | None,
    shown: bool,
) -> None:
    journal = RecordingJournal()
    journal.append(
        UserMessage(content="q"),
        Thinking(content="secret reasoning"),
        AssistantMessage(content="a"),
    )
    screen = io.StringIO()
    async with Agent(model=ScriptedProvider()["scripted"], journal=journal) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            pipe.send_text("\x04")
            running = (
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                )
                if show is None
                else main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    show_thinking=show,
                    print_batch_sec=0,
                )
            )
            await asyncio.wait_for(running, timeout=5)
    assert ("secret reasoning" in screen.getvalue()) is shown


@pytest.mark.asyncio
async def test_run_labels_calls_by_their_tool_and_quits_by_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "a.txt").write_text("alpha")
    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("Read", "c1", file_path="a.txt")]),
            reply("read it"),
        ],
    )
    screen = io.StringIO()
    async with (
        JsonlJournal(tmp_path / "abc" / "journal.jsonl") as journal,
        Agent(
            model=provider["scripted"],
            tools=[Read()],
            journal=journal,
            cwd=tmp_path,
        ) as agent,
    ):
        answered = asyncio.Event()

        def watch(event: types.RuntimeEvent) -> None:
            if isinstance(event, types.AgentIdle) and len(provider.requests) == 2:
                answered.set()

        agent.runtime.observers.append(watch)
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            running = asyncio.create_task(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
            )
            await asyncio.sleep(0.05)
            pipe.send_text("read it\r")
            await asyncio.wait_for(answered.wait(), timeout=5)
            pipe.send_text("/kill all\r/quit now\r")
            await asyncio.wait_for(running, timeout=5)
    shown = screen.getvalue()
    assert Read().summary(call("Read", file_path="a.txt").arguments) in shown
    assert "[/kill] cancelled all tool calls" in shown
    assert "sagent --resume abc   # this session" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_says_every_way_to_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with (
        JsonlJournal(tmp_path / "abc" / "journal.jsonl") as journal,
        Agent(model=ScriptedProvider()["scripted"], journal=journal) as agent,
    ):
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            pipe.send_text("\x04")
            await asyncio.wait_for(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
                timeout=5,
            )
    assert capsys.readouterr().err.split("\n")[:6] == [
        "Resume this session with:",
        "sagent --resume abc   # this session",
        "sagent --continue         # the latest in this directory",
        "sagent --resume           # pick one in this directory",
        "sagent --continue-all     # the latest anywhere",
        "sagent --resume-all       # pick one anywhere",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("given", "batch_sec"), [(None, 0.2), (0.01, 0.01)])
async def test_run_prints_above_the_prompt_in_batches_as_long_as_told(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    given: float | None,
    batch_sec: float,
) -> None:
    proxies: list[StdoutProxy] = []

    class _Kept(StdoutProxy):
        @override
        def __enter__(self) -> StdoutProxy:
            proxies.append(self)
            return super().__enter__()

    monkeypatch.setattr(main, "StdoutProxy", _Kept)
    stdout, stderr = sys.stdout, sys.stderr
    async with Agent(model=ScriptedProvider()["scripted"]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            pipe.send_text("\x04")
            running = (
                main.run(agent, catalog=_Script().catalog(), history=tmp_path / "h")
                if given is None
                else main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=given,
                )
            )
            await asyncio.wait_for(running, timeout=5)
    (proxy,) = proxies
    assert (proxy.sleep_between_writes, proxy.raw, proxy.closed) == (
        batch_sec,
        True,
        True,
    )
    assert (sys.stdout, sys.stderr) == (stdout, stderr)


@pytest.mark.asyncio
async def test_run_logs_a_failing_command_with_its_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Broken(Commands):
        @override
        async def __call__(self, line: str) -> bool:
            if line == "/boom":
                raise RuntimeError("kaboom")
            return await super().__call__(line)

    monkeypatch.setattr(main, "Commands", _Broken)
    async with Agent(model=ScriptedProvider()["scripted"]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            pipe.send_text("/boom\r/exit\r")
            with caplog.at_level(logging.WARNING, logger=main.__name__):
                await asyncio.wait_for(
                    main.run(
                        agent,
                        catalog=_Script().catalog(),
                        history=tmp_path / "h",
                        print_batch_sec=0,
                    ),
                    timeout=5,
                )
    (record,) = [r for r in caplog.records if r.name == main.__name__]
    assert record.getMessage() == "command '/boom' failed"
    assert isinstance(record.exc_info, tuple)
    assert record.exc_info[0] is RuntimeError


@pytest.mark.asyncio
async def test_run_keeps_history_in_the_per_user_file_by_default() -> None:
    async with Agent(model=ScriptedProvider()["scripted"]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=DummyOutput()),
        ):
            pipe.send_text("/help\r\x04")
            await asyncio.wait_for(
                main.run(agent, catalog=_Script().catalog(), print_batch_sec=0),
                timeout=5,
            )
    history = state_dir() / "rekursiv-ai" / "sagent" / "repl-history"
    assert "/help" in history.read_text()


def test_a_listing_exactly_at_the_cap_hides_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = _infos(tmp_path, "a", "b")
    _stdin(monkeypatch, "\n")
    main._pick(sessions, cap=2)
    assert "older ones hidden" not in capsys.readouterr().err
    main._pick(sessions, cap=1)
    assert capsys.readouterr().err.startswith(
        "  (showing 1 sessions; older ones hidden)\n",
    )


def test_a_tool_label_record_is_the_tool_s_summary_in_field_order() -> None:
    read = call("Read", "c1", file_path="a.txt")
    (record,) = main._records(
        types.ModelResponseComplete(response=reply(calls=[read])),
        tools={"Read": Read()},
    )
    assert record == {
        "descriptor": "text/x-tool-label",
        "call_id": "c1",
        "content": Read().summary(read.arguments),
    }
    assert record["content"] != "Read"
    assert list(record) == ["descriptor", "call_id", "content"]


@pytest.mark.asyncio
async def test_stream_json_keeps_each_record_s_fields_in_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await _headless(
        "q",
        "--output-format",
        "stream-json",
        replies=[reply(calls=[call("Gone", "c9")]), reply("done")],
        monkeypatch=monkeypatch,
    )
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert (
        lines[0]
        == '{"descriptor": "text/x-tool-label", "call_id": "c9", "content": "Gone"}'
    )


@pytest.mark.parametrize("argv", [("--resume",), ("--resume-all",)])
def test_a_picker_reads_one_journal_past_its_limit_and_no_more(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
) -> None:

    for i in range(8):
        _journal(
            project_dir(tmp_path) / f"{i:012}",
            *_said(f"title {i}"),
            mtime=1_000 + i,
        )
    peeked: list[str] = []
    peek = projects._peek

    def counting(path: Path, *, mtime: float) -> projects.SessionInfo | None:
        peeked.append(path.parent.name)
        return peek(path, mtime=mtime)

    monkeypatch.setattr(projects, "_peek", counting)
    _stdin(monkeypatch, "3\n")
    path = main._journal_path(
        _flags(*argv, "--resume-limit", "3"),
        cwd=tmp_path,
        interactive=True,
    )
    assert path is not None
    assert path.parent.name == f"{5:012}"
    assert peeked == [f"{i:012}" for i in (7, 6, 5, 4)]
    assert capsys.readouterr().err.count(" records · title ") == 3


def test_the_agent_works_where_it_was_told() -> None:
    agent = main._agent(
        _flags(),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=False,
    )
    assert agent.runtime.tool_context.cwd == Path("/w")


@pytest.mark.parametrize("interactive", [True, False])
def test_a_terminal_waits_out_a_rate_limit_and_a_pipe_does_not(
    interactive: bool,
) -> None:
    agent = main._agent(
        _flags(),
        model=ScriptedProvider()["scripted"],
        tools=(),
        journal=None,
        cwd=Path("/w"),
        interactive=interactive,
    )
    assert agent.runtime.persistent_retry is interactive


def test_an_interactive_run_hides_reasoning_when_the_flag_says_so(
    script: _Script,
    monkeypatch: pytest.MonkeyPatch,
    logs: logging.Logger,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    journal = _journal(
        project_dir(tmp_path) / "s",
        UserMessage(content="q"),
        Thinking(content="deliberating"),
        AssistantMessage(content="ok"),
    )
    _argv(
        monkeypatch,
        "--provider",
        "scripted",
        "--model",
        "scripted",
        "--session",
        str(journal),
        "--screen.thinking",
        "hide",
        "--history",
        str(tmp_path / "h"),
    )
    _stdin(monkeypatch, tty=True)
    _quick_prints(monkeypatch)
    screen = io.StringIO()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=PlainTextOutput(screen)),
    ):
        pipe.send_text("\x04")
        assert main.main() == 0
    assert "deliberating" not in screen.getvalue()
    del script, logs


@dataclass(slots=True, kw_only=True)
class _Slow:
    """A tool that runs until ``release`` is set."""

    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    name: str = "Slow"
    description: str = "Take a while."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, PlainTree])
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Slow"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        del context, arguments
        self.started.set()
        await self.release.wait()
        return UncategorizedToolResult(call_id="", content="done")


@pytest.mark.asyncio
async def test_quitting_with_a_message_waiting_says_it_is_discarded(
    tmp_path: Path,
) -> None:
    slow = _Slow()
    provider = ScriptedProvider(
        replies=[reply(calls=[call("Slow", "c1")]), reply("finished")],
    )
    screen = io.StringIO()
    async with Agent(model=provider["scripted"], tools=[slow]) as agent:
        with (
            create_pipe_input() as pipe,
            create_app_session(input=pipe, output=PlainTextOutput(screen)),
        ):
            running = asyncio.create_task(
                main.run(
                    agent,
                    catalog=_Script().catalog(),
                    history=tmp_path / "h",
                    print_batch_sec=0,
                ),
            )
            await asyncio.sleep(0.05)
            pipe.send_text("go\r")
            await asyncio.wait_for(slow.started.wait(), timeout=5)
            pipe.send_text("after the tools\t\x04")
            await asyncio.wait_for(running, timeout=5)
        slow.release.set()
    assert "[discarding 1 queued message: after the tools]" in screen.getvalue()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
