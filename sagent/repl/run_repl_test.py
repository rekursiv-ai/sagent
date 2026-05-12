"""Tests for ``repl.run_repl``: command helpers (no REPL loop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock, patch

import inspect

from sagent.agent.agent import Agent
from sagent.agent.background import BackgroundTaskEntry
from sagent.custom_types import ModelSpec
from sagent.repl.render import RecordingPrinter
from sagent.repl.run_repl import (
    _parse_model_args,
    do_login,
    do_switch_model,
    format_tasks,
    run_repl,
)


_DEFAULT_PROV = "Anthropic"
_DEFAULT_AUTH = "api"
_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_ACCOUNT: str | None = None


def _parse(*tokens: str) -> tuple[str, str, str | None, str] | str:
    return _parse_model_args(
        list(tokens),
        _DEFAULT_PROV,
        _DEFAULT_AUTH,
        _DEFAULT_MODEL,
        _DEFAULT_ACCOUNT,
    )


def test_parse_model_args_no_tokens_returns_usage_string() -> None:
    out = _parse_model_args([], _DEFAULT_PROV, _DEFAULT_AUTH, _DEFAULT_MODEL, None)
    assert isinstance(out, str)
    assert "usage" in out


def test_parse_model_args_bare_model_id() -> None:
    out = _parse("claude-sonnet-4-6")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-sonnet-4-6")


def test_parse_model_args_flag_provider() -> None:
    out = _parse("--provider", "Google", "gemini-3-pro")
    assert out == ("Google", _DEFAULT_AUTH, None, "gemini-3-pro")


def test_parse_model_args_short_flag_provider() -> None:
    out = _parse("-p", "Google")
    assert out == ("Google", _DEFAULT_AUTH, None, _DEFAULT_MODEL)


def test_parse_model_args_flag_auth() -> None:
    out = _parse("--auth", "sub")
    assert out == (_DEFAULT_PROV, "sub", None, _DEFAULT_MODEL)


def test_parse_model_args_flag_account() -> None:
    out = _parse("--account", "work")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, "work", _DEFAULT_MODEL)


def test_parse_model_args_kv_provider() -> None:
    out = _parse("provider=Google")
    assert out == ("Google", _DEFAULT_AUTH, None, _DEFAULT_MODEL)


def test_parse_model_args_kv_auth() -> None:
    out = _parse("auth=sub")
    assert out == (_DEFAULT_PROV, "sub", None, _DEFAULT_MODEL)


def test_parse_model_args_kv_model() -> None:
    out = _parse("model=claude-haiku-4")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-haiku-4")


def test_parse_model_args_kv_model_id_alias() -> None:
    out = _parse("model_id=claude-haiku-4")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-haiku-4")


def test_parse_model_args_kv_account_default_normalized_to_none() -> None:
    out = _parse("account=default")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, _DEFAULT_MODEL)


def test_parse_model_args_kv_account_empty_normalized_to_none() -> None:
    out = _parse("account=")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, _DEFAULT_MODEL)


def test_parse_model_args_unknown_kv_returns_error_string() -> None:
    out = _parse("bogus=x")
    assert isinstance(out, str)
    assert "unknown key" in out


def test_parse_model_args_unknown_flag_returns_error_string() -> None:
    out = _parse("--bogus", "x")
    assert isinstance(out, str)
    assert "unknown flag" in out


def test_parse_model_args_mixed_flags_and_bare_model() -> None:
    out = _parse("--provider", "Google", "--auth", "sub", "gemini-3-pro")
    assert out == ("Google", "sub", None, "gemini-3-pro")


@dataclass(slots=True, kw_only=True)
class _FakeModel:
    model_id: str = "claude-opus-4-7"


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    model: _FakeModel = field(default_factory=_FakeModel)
    model_spec: ModelSpec | None = field(
        default_factory=lambda: ModelSpec(
            provider="Anthropic", auth="api", model_id="claude-opus-4-7"
        ),
    )
    swap_calls: list[tuple[_FakeModel, ModelSpec | None]] = field(default_factory=list)

    def swap_model(self, model: _FakeModel, *, spec: ModelSpec | None = None) -> None:
        self.swap_calls.append((model, spec))
        self.model = model
        self.model_spec = spec


def _as_agent(a: _FakeAgent) -> Agent:
    return cast(Agent, a)


def test_do_switch_model_no_spec_writes_error() -> None:
    agent = _FakeAgent(model_spec=None)
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "", printer)
    assert any("no model spec" in line for line in printer.lines)
    assert agent.swap_calls == []


def test_do_switch_model_empty_args_shows_status() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "", printer)
    assert any(
        "provider=Anthropic" in line and "model=claude-opus-4-7" in line
        for line in printer.lines
    )
    assert agent.swap_calls == []


def test_do_switch_model_shlex_parse_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), 'unclosed "quote', printer)
    assert any("parse error" in line for line in printer.lines)
    assert agent.swap_calls == []


def test_do_switch_model_unknown_flag_writes_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "--bogus x", printer)
    assert any("unknown flag" in line for line in printer.lines)
    assert agent.swap_calls == []


def test_do_switch_model_success_swaps_and_prints_label() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    new_model = _FakeModel(model_id="claude-sonnet-4-6")
    provider = MagicMock()
    provider.model.return_value = new_model
    with (
        patch(
            "sagent.repl.run_repl.build_provider",
            return_value=provider,
        ),
        patch(
            "sagent.repl.run_repl.infer_provider",
            return_value=None,
        ),
    ):
        do_switch_model(_as_agent(agent), "claude-sonnet-4-6", printer)
    assert len(agent.swap_calls) == 1
    swapped_model, spec = agent.swap_calls[0]
    assert swapped_model is new_model
    assert spec is not None
    assert spec.model_id == "claude-sonnet-4-6"
    assert any("claude-sonnet-4-6" in line for line in printer.lines)


def test_do_switch_model_infer_provider_overrides_provider_and_auth() -> None:
    # When inference returns a (provider, auth) pair and the user passed
    # a bare model id whose declared provider matches the current spec,
    # the inferred pair takes over.
    agent = _FakeAgent()
    printer = RecordingPrinter()
    new_model = _FakeModel(model_id="gemini-3-pro")
    provider = MagicMock()
    provider.model.return_value = new_model
    with (
        patch(
            "sagent.repl.run_repl.build_provider",
            return_value=provider,
        ),
        patch(
            "sagent.repl.run_repl.infer_provider",
            return_value=("Google", "sub"),
        ),
    ):
        do_switch_model(_as_agent(agent), "gemini-3-pro", printer)
    assert len(agent.swap_calls) == 1
    _, spec = agent.swap_calls[0]
    assert spec is not None
    assert spec.provider == "Google"
    assert spec.auth == "sub"
    # Cross-provider label: "Anthropic/old -> Google/new".
    assert any(
        "Anthropic/claude-opus-4-7 -> Google/gemini-3-pro" in line
        for line in printer.lines
    )


def test_do_switch_model_provider_build_failure_writes_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    with patch(
        "sagent.repl.run_repl.build_provider",
        side_effect=RuntimeError("no credentials"),
    ):
        do_switch_model(_as_agent(agent), "claude-sonnet-4-6", printer)
    assert any("no credentials" in line for line in printer.lines)
    assert agent.swap_calls == []


def test_do_login_no_spec_writes_error() -> None:
    agent = _FakeAgent(model_spec=None)
    printer = RecordingPrinter()
    do_login(_as_agent(agent), printer)
    assert any("no model spec" in line for line in printer.lines)


def test_do_login_unknown_provider_writes_error() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(provider="NotAProvider", auth="api", model_id="x"),
    )
    printer = RecordingPrinter()
    do_login(_as_agent(agent), printer)
    assert any("unknown provider" in line for line in printer.lines)


def test_do_login_provider_without_login_writes_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    fake_cls = type("FakeProvider", (), {})  # no login attr
    with patch(
        "sagent.repl.run_repl.providers",
        MagicMock(Anthropic=fake_cls),
    ):
        do_login(_as_agent(agent), printer)
    assert any("no login method" in line for line in printer.lines)


def test_do_login_success_writes_confirmation() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    fake_cls = MagicMock()
    with patch(
        "sagent.repl.run_repl.providers",
        MagicMock(Anthropic=fake_cls),
    ):
        do_login(_as_agent(agent), printer)
    fake_cls.login.assert_called_once()
    assert any("re-authenticated" in line for line in printer.lines)


def test_do_login_failure_writes_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    fake_cls = MagicMock()
    fake_cls.login.side_effect = RuntimeError("oauth failed")
    with patch(
        "sagent.repl.run_repl.providers",
        MagicMock(Anthropic=fake_cls),
    ):
        do_login(_as_agent(agent), printer)
    assert any("oauth failed" in line for line in printer.lines)


def test_format_tasks_no_registry_header_only() -> None:
    agent = _FakeAgent()
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {},
    ):
        out = format_tasks(_as_agent(agent))
    assert out.startswith("sagent: 0 agent(s)")
    assert "foreground" in out
    assert "background" in out


def test_format_tasks_lists_registered_agent_fg_idle() -> None:
    agent = _FakeAgent()
    other = MagicMock()
    other.work = None
    other.background = {}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"Agent_0": other},
    ):
        out = format_tasks(_as_agent(agent))
    assert "Agent_0" in out
    assert "fg=0" in out
    assert "bg=0" in out


def test_format_tasks_marks_self() -> None:
    agent = _FakeAgent()
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"Agent_0": agent},
    ):
        out = format_tasks(_as_agent(agent))
    assert "(self)" in out


def test_format_tasks_lists_bg_jobs() -> None:
    agent = _FakeAgent()
    task = MagicMock()
    task.done.return_value = False
    task.cancelled.return_value = False
    job = BackgroundTaskEntry(
        task=task,
        tool_name="Bash",
        queue_id="bg-1",
        started=0.0,
        kind="tool",
        hidden=False,
        delay_sec=0.0,
    )
    other = MagicMock()
    other.work = task
    other.background = {"bg-1": job}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"Agent_0": other},
    ):
        out = format_tasks(_as_agent(agent))
    assert "bg-1" in out
    assert "Bash" in out
    assert "running" in out


def test_run_repl_invokes_replay_messages() -> None:
    # --resume / --continue rely on replay_messages to render persisted
    # history into scrollback. The unit test on replay_messages itself
    # stays green even when the call site is dropped (see eb4700ef).
    src = inspect.getsource(run_repl)
    assert "replay_messages(" in src


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
