"""Tests for ``repl.run_repl``: command helpers (no REPL loop)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock, patch

import asyncio
import dataclasses
import inspect

import pytest

from sagent.agent import runtime as agent_runtime
from sagent.agent.agent import Agent, _resolve_target_spec
from sagent.agent.background import BackgroundTaskEntry
from sagent.lib import last_models
from sagent.providers import Google
from sagent.repl.render import (
    RecordingPrinter,
    make_render_observer,
)
from sagent.repl.run_repl import (
    _parse_model_args,
    do_login,
    do_switch_model,
    format_tasks,
    make_queued_input_committer,
    run_repl,
)
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ModelSpec
from sagent.types.runtime import (
    ModelIdle,
    ModelResponseError,
    Quit,
    RuntimeEvent,
    UserQueuedMessage,
)


_DEFAULT_PROV = "Anthropic"
_DEFAULT_AUTH = "api"
_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_ACCOUNT: str | None = None
_DEFAULT_SPEC = ModelSpec(
    provider=_DEFAULT_PROV,
    auth=_DEFAULT_AUTH,
    model_id=_DEFAULT_MODEL,
    account=_DEFAULT_ACCOUNT,
)


def _parse(*tokens: str) -> tuple[str, str, str | None, str] | str:
    """Parse + resolve helper: composes the slash-syntax parser with
    the Agent's resolver and projects the resolved ``ModelSpec`` back
    into the legacy ``(provider, auth, account, model_id)`` tuple.
    """
    parsed = _parse_model_args(list(tokens))
    if isinstance(parsed, str):
        return parsed
    target = _resolve_target_spec(
        _DEFAULT_SPEC,
        provider=parsed.provider,
        auth=parsed.auth,
        model_id=parsed.model_id,
        account=parsed.account if parsed.account_set else None,
    )
    return target.provider, target.auth, target.account, target.model_id


def test_parse_model_args_no_tokens_returns_usage_string() -> None:
    out = _parse_model_args([])
    assert isinstance(out, str)
    assert "usage" in out


def test_parse_model_args_bare_model_id() -> None:
    out = _parse("claude-sonnet-4-6")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-sonnet-4-6")


def test_parse_model_args_flag_provider() -> None:
    out = _parse("--provider", "Google", "gemini-3-pro")
    assert out == ("Google", _DEFAULT_AUTH, None, "gemini-3-pro")


def test_parse_model_args_short_flag_provider_falls_back_to_default_model() -> None:
    """Switching provider with no model → provider's DEFAULT_MODEL.

    With no entry in ``~/.sagent/last-models.json`` for Google, the
    resolver falls back to ``Google.DEFAULT_MODEL``.
    """
    out = _parse("-p", "Google")
    assert out == ("Google", _DEFAULT_AUTH, None, Google.DEFAULT_MODEL)


def test_parse_model_args_flag_auth() -> None:
    out = _parse("--auth", "sub")
    assert out == (_DEFAULT_PROV, "sub", None, _DEFAULT_MODEL)


def test_parse_model_args_flag_account() -> None:
    out = _parse("--account", "work")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, "work", _DEFAULT_MODEL)


def test_parse_model_args_kv_provider_falls_back_to_default_model() -> None:
    """``/model provider=Google`` with no model picks Google.DEFAULT_MODEL."""
    out = _parse("provider=Google")
    assert out == ("Google", _DEFAULT_AUTH, None, Google.DEFAULT_MODEL)


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


def test_provider_switch_uses_last_used_when_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/model provider=Google`` prefers the last-used Google model over the default.

    The current spec's ``claude-opus-4-7`` is not in ``Google.KNOWN_MODELS``
    so cross-provider preservation falls through. With a recorded last-used
    Google model in ``~/.sagent/last-models.json``, the resolver picks
    that up. ``Google.DEFAULT_MODEL`` is the cold-start fallback.
    """
    monkeypatch.setattr(last_models, "load", lambda: {"Google": "remembered-model"})
    out = _parse("provider=Google")
    assert out == ("Google", _DEFAULT_AUTH, None, "remembered-model")


def test_provider_switch_preserves_current_model_when_new_provider_knows_it() -> None:
    """``/model provider=AnthropicCLI`` from ``Anthropic/claude-opus-4-7`` keeps the model.

    AnthropicCLI inherits ``KNOWN_MODELS`` from ``Anthropic`` so the
    current model id is valid on the new provider. The resolver must
    preserve it across the swap rather than demoting to last_models
    or ``DEFAULT_MODEL``.
    """
    out = _parse("provider=AnthropicCLI")
    assert out == ("AnthropicCLI", _DEFAULT_AUTH, None, _DEFAULT_MODEL)


@dataclass(slots=True, kw_only=True)
class _FakeModel:
    model_id: str = "claude-opus-4-7"
    _provider: object | None = None


@dataclass(slots=True, kw_only=True)
class _FakeInbox:
    pushed: list[object] = field(default_factory=list)

    def push_back(self, item: object) -> None:
        self.pushed.append(item)


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    inbox: _FakeInbox = field(default_factory=_FakeInbox)


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    model: _FakeModel = field(default_factory=_FakeModel)
    model_spec: ModelSpec | None = field(
        default_factory=lambda: ModelSpec(
            provider="Anthropic", auth="api", model_id="claude-opus-4-7"
        ),
    )
    runtime: _FakeRuntime = field(default_factory=_FakeRuntime)
    work: object = None
    swap_calls: list[tuple[_FakeModel, ModelSpec | None]] = field(default_factory=list)
    change_model_calls: list[dict[str, object]] = field(default_factory=list)
    change_model_result: ModelSpec | None = None
    change_model_side_effect: BaseException | None = None
    relogin_calls: int = 0
    relogin_side_effect: BaseException | None = None

    def swap_model(self, model: _FakeModel, *, spec: ModelSpec | None = None) -> None:
        self.swap_calls.append((model, spec))
        self.model = model
        self.model_spec = spec

    def change_model(
        self,
        *,
        provider: str | None = None,
        auth: str | None = None,
        model_id: str | None = None,
        account: str | None = None,
    ) -> ModelSpec:
        """Fake delegate matching ``Agent.change_model``."""
        self.change_model_calls.append(
            {
                "provider": provider,
                "auth": auth,
                "model_id": model_id,
                "account": account,
            }
        )
        if self.change_model_side_effect is not None:
            raise self.change_model_side_effect
        if self.change_model_result is not None:
            return self.change_model_result
        spec = self.model_spec
        assert spec is not None
        return dataclasses.replace(
            spec,
            provider=provider or spec.provider,
            auth=auth or spec.auth,
            model_id=model_id or spec.model_id,
            account=account if account is not None else spec.account,
        )

    async def relogin(self) -> None:
        """Fake delegate matching ``Agent.relogin``."""
        self.relogin_calls += 1
        if self.relogin_side_effect is not None:
            raise self.relogin_side_effect


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


def test_do_switch_model_success_delegates_to_change_model() -> None:
    """Slash handler parses, delegates to ``Agent.change_model``, prints label."""
    agent = _FakeAgent()
    printer = RecordingPrinter()
    with patch(
        "sagent.repl.run_repl.infer_provider",
        return_value=None,
    ):
        do_switch_model(_as_agent(agent), "claude-sonnet-4-6", printer)
    assert agent.change_model_calls == [
        {
            "provider": None,
            "auth": None,
            "model_id": "claude-sonnet-4-6",
            "account": None,
        }
    ]
    assert any("claude-sonnet-4-6" in line for line in printer.lines)


def test_do_switch_model_infer_provider_overrides_provider_and_auth() -> None:
    """A bare model id whose ``infer_provider`` matches a foreign provider
    rewrites the provider/auth before calling ``change_model``.
    """
    agent = _FakeAgent()
    printer = RecordingPrinter()
    agent.change_model_result = ModelSpec(
        provider="Google", auth="sub", model_id="gemini-3-pro"
    )
    with patch(
        "sagent.repl.run_repl.infer_provider",
        return_value=("Google", "sub"),
    ):
        do_switch_model(_as_agent(agent), "gemini-3-pro", printer)
    assert agent.change_model_calls == [
        {
            "provider": "Google",
            "auth": "sub",
            "model_id": "gemini-3-pro",
            "account": None,
        }
    ]
    assert any(
        "Anthropic/claude-opus-4-7 -> Google/gemini-3-pro" in line
        for line in printer.lines
    )


def test_do_switch_model_change_model_error_writes_to_printer() -> None:
    """``Agent.change_model`` errors surface verbatim to the printer."""
    agent = _FakeAgent()
    agent.change_model_side_effect = ValueError("no credentials")
    printer = RecordingPrinter()
    with patch(
        "sagent.repl.run_repl.infer_provider",
        return_value=None,
    ):
        do_switch_model(_as_agent(agent), "claude-sonnet-4-6", printer)
    assert any("no credentials" in line for line in printer.lines)


@pytest.mark.asyncio
async def test_do_login_no_spec_writes_error() -> None:
    agent = _FakeAgent(model_spec=None)
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
    assert any("no model spec" in line for line in printer.lines)


@pytest.mark.asyncio
async def test_do_login_success_delegates_to_relogin() -> None:
    """Slash handler delegates to ``Agent.relogin`` and prints confirmation."""
    agent = _FakeAgent()
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
    assert agent.relogin_calls == 1
    assert any("re-authenticated" in line for line in printer.lines)


@pytest.mark.asyncio
async def test_do_login_relogin_error_writes_to_printer() -> None:
    """``Agent.relogin`` errors surface verbatim to the printer."""
    agent = _FakeAgent()
    agent.relogin_side_effect = RuntimeError("oauth failed")
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
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


@pytest.mark.asyncio
async def test_make_queued_input_committer_pushes_user_queued_on_model_idle() -> None:
    """``ModelIdle`` with non-empty queue → coalesced ``UserQueuedMessage`` pushed; queue cleared."""
    queued_input = ["elephant", "banana", "chair"]
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    observer = make_queued_input_committer(runtime, queued_input)
    observer(ModelIdle())
    assert queued_input == []
    pushed = await runtime.inbox.drain()
    assert len(pushed) == 1
    item = pushed[0]
    assert isinstance(item, UserQueuedMessage)
    assert item.text == "elephant\n\nbanana\n\nchair"


def test_make_queued_input_committer_ignores_non_idle_events() -> None:
    """Non-``ModelIdle`` events leave ``queued_input`` and the inbox untouched."""
    queued_input = ["elephant"]
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    observer = make_queued_input_committer(runtime, queued_input)
    observer(UserMessage(text="real submission"))
    observer(ModelResponseError(RuntimeError("x")))
    assert queued_input == ["elephant"]
    assert runtime.inbox._queue.empty()


def test_make_queued_input_committer_ignores_idle_when_queue_empty() -> None:
    """``ModelIdle`` with an empty queue is a no-op (no spurious push)."""
    queued_input: list[str] = []
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    observer = make_queued_input_committer(runtime, queued_input)
    observer(ModelIdle())
    assert runtime.inbox._queue.empty()


@dataclass(kw_only=True, slots=True)
class _TextOnlyModel:
    """One-shot scripted model: returns text, no tool calls."""

    text: str = "ok"

    async def stream(
        self,
        history: list[HistoryEntry],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_thinking
        for ch in self.text:
            on_text(ch)
        return AssistantMessage(text=self.text)


@pytest.mark.asyncio
async def test_queued_input_committed_and_cleared_on_model_idle() -> None:
    """Integration: Tab-staged ``queued_input`` commits as ``UserQueuedMessage``
    on ``ModelIdle`` and the local list clears.

    Wires the option-1 contract end-to-end: REPL stages locally via
    Tab (here we pre-populate the list to simulate that), the
    committer observer sees ``ModelIdle`` after the round answering
    the initial ``UserMessage`` settles, and pushes the coalesced
    queue back as ``UserQueuedMessage``. The runtime then drains it
    at the next gate-section pass and fires a fresh round.
    """
    queued_input = ["elephant", "banana", "chair"]
    agent = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="committed"))
    agent.observers.append(make_queued_input_committer(agent, queued_input))

    second_round = asyncio.Event()
    rounds = 0

    def _watch_idle(event: RuntimeEvent) -> None:
        nonlocal rounds
        if isinstance(event, ModelIdle):
            rounds += 1
            if rounds >= 2:
                second_round.set()

    agent.observers.append(_watch_idle)

    agent.inbox.push_back(UserMessage(text="real submission"))
    task = asyncio.create_task(agent.run_forever())
    await asyncio.wait_for(second_round.wait(), timeout=2.0)
    agent.inbox.push_back(Quit())
    await task

    assert queued_input == [], f"expected queued_input cleared, got {queued_input}"
    queued_texts = [m.text for m in agent.history if isinstance(m, UserMessage)]
    assert "elephant\n\nbanana\n\nchair" in queued_texts


@pytest.mark.asyncio
async def test_staging_path_end_to_end_renders_user_bar() -> None:
    """End-to-end: committed ``UserMessage`` reaches ``console_pane``.

    Wires the full chain: REPL pushes ``UserMessage`` (the staging
    model's commit type) → runtime matches it, publishes the event →
    render observer writes ``user_bar`` to printer. Validates the
    end-to-end path the user observes after pressing Enter on an
    empty buffer with a non-empty queue.
    """
    printer = RecordingPrinter()
    agent = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ack"))
    agent.observers.append(make_render_observer(printer))

    agent.inbox.push_back(UserMessage(text="hello world"))

    done = asyncio.Event()

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            done.set()

    agent.observers.append(_watch)

    task = asyncio.create_task(agent.run_forever())
    await asyncio.wait_for(done.wait(), timeout=2.0)
    agent.inbox.push_back(Quit())
    await task

    assert "hello world" in printer.user_bars, (
        f"expected the staged user content to land as a user bar in the"
        f" console pane; got user_bars={printer.user_bars}"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_repl_commit_during_cohort_preempts_tools_to_background() -> None:
    """Regression: REPL commit while a cohort is running must preempt.

    The user's "type to redirect" intent: when the agent is in the
    middle of running tools, committing a staged message should detach
    those tools to background (``[detached]`` placeholders) and fire
    a fresh model round for the committed content immediately. T2
    semantics.

    Earlier we accidentally pushed ``UserQueuedMessage`` (T4 / drain
    at ``ModelIdle``) at commit time. The keybinding unit test
    ``test_enter_on_empty_buf_with_queue_commits_*`` asserted whatever
    type the implementation pushed, so the regression slipped. This
    test asserts the behavior end-to-end against a real ``AgentRuntime``:
    if commit pushes anything that fails to preempt the cohort, the
    fresh round never fires while tools are still running and this
    test fails.
    """
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    @dataclass(kw_only=True, slots=True)
    class _BlockingTool:
        _name: str = "echo"

        @property
        def name(self) -> str:
            return self._name

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await release_tool.wait()
            return ToolResult(call_id="", content="completed late")

    @dataclass(kw_only=True, slots=True)
    class _TwoRoundModel:
        call_histories: list[list[HistoryEntry]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[HistoryEntry],
            system: str,
            tools: list[agent_runtime.Tool],
            on_text: Callable[[str], None],
            on_thinking: Callable[[str], None],
        ) -> AssistantMessage:
            del system, tools, on_thinking
            self.call_histories.append(list(history))
            idx = self._i
            self._i += 1
            if idx == 0:
                # Round 1: request a slow tool. Cohort spawns.
                return AssistantMessage(
                    tool_calls=(ToolCall(id="t1", name="echo", args={}),),
                )
            # Round 2 (commit fire): respond to the staged message.
            msg = AssistantMessage(text="redirected")
            for ch in msg.text:
                on_text(ch)
            return msg

    tool = _BlockingTool()
    model = _TwoRoundModel()
    runtime = agent_runtime.AgentRuntime(model=model, tools=[tool])

    # Round 1 trigger.
    runtime.inbox.push_back(UserMessage(text="initial"))

    # Simulate the REPL's text-Enter path: text input + Enter pushes
    # ``UserMessage`` immediately. The runtime sees a mid-cohort
    # ``UserMessage`` and must preempt -- stub the slow tool to
    # ``detached`` and fire a fresh round.
    async def commit_during_cohort() -> None:
        await tool_started.wait()
        runtime.inbox.push_back(UserMessage(text="redirect please"))
        # Sleep gives the runtime time to preempt + fire Round 2.
        await asyncio.sleep(0.2)
        release_tool.set()
        await asyncio.sleep(0.1)
        runtime.inbox.push_back(Quit())

    async def drive() -> None:
        try:
            await asyncio.wait_for(runtime.run_forever(), timeout=3.0)
        except TimeoutError:
            pytest.fail("runtime did not quit within timeout")

    await asyncio.gather(drive(), commit_during_cohort())

    assert len(model.call_histories) >= 2, (
        "Round 2 (model call for the committed staged content) never"
        " fired. The commit must preempt the in-flight cohort."
    )
    round2_users = [
        m.text for m in model.call_histories[1] if isinstance(m, UserMessage)
    ]
    assert "redirect please" in round2_users, (
        f"Round 2 did not see the committed message; got user messages {round2_users}"
    )
    # Tool's [detached] placeholder must be present in history (proves
    # the cohort got stubbed to background rather than blocking the gate).
    placeholders = [
        m
        for m in runtime.history
        if isinstance(m, ToolResult) and m.content in {"[detached]", "completed late"}
    ]
    assert placeholders, (
        f"Expected the slow tool to be detached or eventually splice"
        f" 'completed late' into history; saw history:"
        f" {[type(m).__name__ for m in runtime.history]}"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
