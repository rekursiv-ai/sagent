"""Tests for ``repl.run_repl``: command helpers (no REPL loop)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import asyncio
import contextlib
import dataclasses
import inspect
import sys

import pytest

from sagent import thinking
from sagent.agent import runtime as agent_runtime
from sagent.agent.agent import Agent, _resolve_target_spec
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import AgentLike
from sagent.lib import last_models
from sagent.providers import Google
from sagent.repl.input_queues import InputQueues, QueuedInputBlock
from sagent.repl.keybindings import (
    NavState,
    _kb_defer,
    _kb_submit,
    _kb_up,
)
from sagent.repl.render import (
    RecordingPrinter,
    make_render_observer,
)
from sagent.repl.run_repl import (
    _background_tasks_for_repl_cancel,
    _input_queue_committer_observer,
    _parse_model_args,
    _subagent_phase,
    do_login,
    do_switch_effort,
    do_switch_model,
    do_switch_thinking,
    format_tasks,
    install_input_queue_committer,
    run_repl,
)
from sagent.types.model import ModelSpec
from sagent.types.runtime import (
    DETACHED_PLACEHOLDER,
    AgentIdle,
    AssistantMessage,
    ClearComplete,
    Compact,
    Halt,
    ModelContextEvent,
    ModelIdle,
    ModelResponseComplete,
    ModelResponseError,
    ModelResponsePartial,
    Quit,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserDeferredMessage,
    UserMessage,
)
from sagent.types.tape import ContextSplice, TapeRef


async def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_sec: float = 1.0,
) -> None:
    """Wait until a predicate is true without adding fixed sleeps."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while not predicate():
        if loop.time() >= deadline:
            pytest.fail("condition did not become true within timeout")
        await asyncio.sleep(0)


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


def test_install_input_queue_committer_installs_before_tool_spawn_hook() -> None:
    agent = _QueueAgent()
    queues = InputQueues(
        queue=QueuedInputBlock(text="interrupt now"),
        deferred=QueuedInputBlock(text="later"),
    )
    _ = install_input_queue_committer(_as_queue_agent(agent), queues)
    hook = agent.runtime.before_tool_spawn
    assert callable(hook)
    pushed = hook(AssistantMessage(text="tool next"))
    assert queues.queue is None
    assert queues.deferred is not None
    assert queues.deferred.text == "later"
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "interrupt now"


def test_install_input_queue_committer_preserves_existing_before_tool_spawn_hook() -> (
    None
):
    original_error = ModelResponseError(RuntimeError("too many tool rounds"))

    def _original(_message: AssistantMessage) -> RuntimeEvent | None:
        return original_error

    agent = _QueueAgent()
    agent.runtime.before_tool_spawn = _original
    queues = InputQueues(queue=QueuedInputBlock(text="interrupt now"))

    _ = install_input_queue_committer(_as_queue_agent(agent), queues)

    hook = agent.runtime.before_tool_spawn
    assert callable(hook)
    assert hook(AssistantMessage(text="tool next")) is original_error
    assert queues.queue is not None
    assert queues.queue.text == "interrupt now"


def test_install_input_queue_committer_composes_existing_empty_hook() -> None:
    seen: list[AssistantMessage] = []

    def _original(message: AssistantMessage) -> RuntimeEvent | None:
        seen.append(message)
        return None

    agent = _QueueAgent()
    agent.runtime.before_tool_spawn = _original
    queues = InputQueues(queue=QueuedInputBlock(text="interrupt now"))

    _ = install_input_queue_committer(_as_queue_agent(agent), queues)

    hook = agent.runtime.before_tool_spawn
    assert callable(hook)
    message = AssistantMessage(text="tool next")
    pushed = hook(message)
    assert seen == [message]
    assert isinstance(pushed, UserMessage)
    assert pushed.text == "interrupt now"
    assert queues.queue is None


def test_install_input_queue_committer_composes_later_before_tool_spawn_hook() -> None:
    agent = _QueueAgent()
    queues = InputQueues()
    _ = install_input_queue_committer(_as_queue_agent(agent), queues)
    later_error = ModelResponseError(RuntimeError("later hook"))

    def _later(_message: AssistantMessage) -> RuntimeEvent | None:
        return later_error

    agent.runtime.before_tool_spawn = _later
    observer = _input_queue_committer_observer(_as_queue_agent(agent), queues)
    observer(AgentIdle())

    hook = agent.runtime.before_tool_spawn
    assert callable(hook)
    assert hook(AssistantMessage(text="tool next")) is later_error


def test_install_input_queue_committer_deferred_skips_model_response_complete() -> None:
    agent = _QueueAgent()
    queues = InputQueues(deferred=QueuedInputBlock(text="later"))
    observer = _input_queue_committer_observer(_as_queue_agent(agent), queues)
    observer(ModelResponseComplete(message=AssistantMessage(text="done")))
    assert queues.deferred is not None
    assert queues.deferred.text == "later"
    assert agent.runtime.inbox.pushed == []


def test_install_input_queue_committer_uninstall_restores_before_tool_spawn() -> None:
    """REPL-043: uninstall closure restores the prior hook and detaches observer.

    Without this the caller would have to capture / restore the
    ``before_tool_spawn`` slot by hand -- exactly the asymmetric
    capture pattern this refactor removes from ``run_repl``.
    """
    agent = _QueueAgent()
    original = agent.runtime.before_tool_spawn
    queues = InputQueues(queue=QueuedInputBlock(text="interrupt now"))

    uninstall = install_input_queue_committer(_as_queue_agent(agent), queues)
    # Install: hook wrapped, observer attached.
    assert agent.runtime.before_tool_spawn is not original
    initial_observer_count = len(agent.runtime.observers)
    assert initial_observer_count >= 1

    uninstall()
    assert agent.runtime.before_tool_spawn is original, (
        f"uninstall must restore prior before_tool_spawn;"
        f" got {agent.runtime.before_tool_spawn}"
    )
    assert len(agent.runtime.observers) == initial_observer_count - 1, (
        f"uninstall must detach the observer it appended;"
        f" observers={agent.runtime.observers!r}"
    )
    # Idempotent: a second call is safe.
    uninstall()
    assert agent.runtime.before_tool_spawn is original


def test_install_input_queue_committer_uninstall_preserves_later_hook() -> None:
    """Uninstall must not clobber a later install that took ownership.

    A subsequent owner replaces ``before_tool_spawn``; uninstall sees
    the slot no longer holds our wrapper and leaves it alone.
    """
    agent = _QueueAgent()
    queues = InputQueues()
    uninstall = install_input_queue_committer(_as_queue_agent(agent), queues)
    later_error = ModelResponseError(RuntimeError("later owner"))

    def _later_hook(_message: AssistantMessage) -> RuntimeEvent | None:
        return later_error

    agent.runtime.before_tool_spawn = _later_hook
    uninstall()
    assert agent.runtime.before_tool_spawn is _later_hook, (
        "uninstall must not clobber a downstream hook that took over the slot"
    )


def test_parse_model_args_no_tokens_returns_usage_string() -> None:
    out = _parse_model_args([])
    assert isinstance(out, str)
    assert "usage" in out


def test_parse_model_args_bare_model_id() -> None:
    out = _parse("claude-sonnet-4-6")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-sonnet-4-6")


def test_parse_model_args_flag_provider() -> None:
    out = _parse("--provider", "Google", "gemini-3-pro")
    assert out == ("Google", "env", None, "gemini-3-pro")


def test_parse_model_args_short_flag_provider_falls_back_to_default_model() -> None:
    """Switching provider with no model → provider's DEFAULT_MODEL.

    With no entry in ``~/.sagent/last-models.json`` for Google, the
    resolver falls back to ``Google.DEFAULT_MODEL``.
    """
    with patch.object(last_models, "load", return_value={}):
        out = _parse("-p", "Google")
    assert out == ("Google", "env", None, Google.DEFAULT_MODEL)


def test_parse_model_args_flag_auth() -> None:
    out = _parse("--auth", "sub")
    assert out == (_DEFAULT_PROV, "sub", None, _DEFAULT_MODEL)


def test_parse_model_args_flag_account() -> None:
    out = _parse("--account", "work")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, "work", _DEFAULT_MODEL)


def test_parse_model_args_kv_provider_falls_back_to_default_model() -> None:
    """``/model provider=Google`` with no model picks Google.DEFAULT_MODEL."""
    with patch.object(last_models, "load", return_value={}):
        out = _parse("provider=Google")
    assert out == ("Google", "env", None, Google.DEFAULT_MODEL)


def test_parse_model_args_kv_auth() -> None:
    out = _parse("auth=sub")
    assert out == (_DEFAULT_PROV, "sub", None, _DEFAULT_MODEL)


def test_parse_model_args_kv_model() -> None:
    out = _parse("model=claude-haiku-4")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-haiku-4")


def test_parse_model_args_kv_model_id_alias() -> None:
    out = _parse("model_id=claude-haiku-4")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, None, "claude-haiku-4")


def test_parse_model_args_kv_account_default_is_preserved() -> None:
    out = _parse("account=default")
    assert out == (_DEFAULT_PROV, _DEFAULT_AUTH, "default", _DEFAULT_MODEL)


def test_parse_model_args_kv_account_empty_is_preserved_for_repl_syntax() -> None:
    parsed = _parse_model_args(["account="])
    assert not isinstance(parsed, str)
    assert parsed.account == ""
    assert parsed.account_set is True


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
    assert out == ("Google", "env", None, "remembered-model")


def test_provider_switch_preserves_current_model_when_new_provider_knows_it() -> None:
    """``/model provider=AnthropicCLI`` from ``Anthropic/claude-opus-4-7`` keeps the model.

    AnthropicCLI inherits ``KNOWN_MODELS`` from ``Anthropic`` so the
    current model id is valid on the new provider. The resolver must
    preserve it across the swap rather than demoting to last_models
    or ``DEFAULT_MODEL``.
    """
    out = _parse("provider=AnthropicCLI")
    assert out == ("AnthropicCLI", "credentials", None, _DEFAULT_MODEL)


@dataclass(slots=True, kw_only=True)
class _FakeModel:
    model_id: str = "claude-opus-4-7"
    supports_thinking: bool = True
    supports_redaction: bool = True
    valid_efforts: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
    _provider: object | None = None

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        return thinking.valid_thinking_states(
            thinking.ThinkingCapability(
                supports_thinking=self.supports_thinking,
                supports_redaction=self.supports_redaction,
            ),
        )


@dataclass(slots=True, kw_only=True)
class _FakeInbox:
    pushed: list[object] = field(default_factory=list)

    def push_back(self, item: object) -> None:
        self.pushed.append(item)


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    inbox: _FakeInbox = field(default_factory=_FakeInbox)
    model_call: object = None
    compact_task: object = None
    cohort: set[str] = field(default_factory=set)
    running_tools: tuple[object, ...] = ()
    service_suspended_until: float | None = None


@dataclass(slots=True, kw_only=True)
class _RuntimeHolder:
    runtime: agent_runtime.AgentRuntime


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
    thinking_state: str | None = None
    thinking: str | None = "adaptive"
    show_thinking: bool = True
    background: dict[str, BackgroundTaskEntry] = field(default_factory=dict)
    _effort: str | None = None

    @property
    def effort(self) -> str | None:
        return self._effort

    @effort.setter
    def effort(self, value: str | None) -> None:
        # Mirror ``Agent.effort`` setter EXACTLY (agent.py), including the
        # no-support branch the earlier copy omitted, so this fake cannot
        # diverge from production. The real setter's own branches are
        # tested directly in agent_test.py (test_effort_setter_*); this
        # mirror only lets the REPL-adapter tests drive a faithful agent.
        if value is not None:
            valid = self.model.valid_efforts
            if not valid:
                raise ValueError(
                    f"Model {self.model.model_id!r} does not support effort."
                )
            if value not in valid:
                quoted = ", ".join(repr(e) for e in valid)
                raise ValueError(f"effort must be one of {quoted}, got {value!r}")
        self._effort = value

    def set_thinking_state(self, state: str) -> None:
        # Delegate to the REAL derivation so these tests exercise
        # production logic, not a divergent copy. ``thinking`` /
        # ``show_thinking`` mirror ``Agent.set_thinking_state``.
        # (``redact_thinking`` is derived at provider-build time inside
        # ``Agent._provider_build_options`` -- no bag to mirror here.)
        canonical = cast(thinking.ThinkingState, state)
        self.thinking_state = state
        self.thinking = thinking.request_thinking(canonical)
        self.show_thinking = thinking.should_show_thinking(canonical)

    def restore_thinking_state(
        self, state: str | None, thinking: str | None, show_thinking: bool
    ) -> None:
        self.thinking_state = state
        self.thinking = thinking
        self.show_thinking = show_thinking

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


@dataclass(slots=True, kw_only=True)
class _QueueRuntime:
    inbox: _FakeInbox = field(default_factory=_FakeInbox)
    before_tool_spawn: Callable[[AssistantMessage], RuntimeEvent | None] | None = None
    observers: list[Callable[[RuntimeEvent], None]] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class _QueueAgent:
    runtime: _QueueRuntime = field(default_factory=_QueueRuntime)


def _as_queue_agent(a: _QueueAgent) -> Agent:
    return cast(Agent, a)


def test_do_switch_model_no_spec_writes_error() -> None:
    agent = _FakeAgent(model_spec=None)
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "", printer)
    assert any("no model spec" in line for line in printer.slash_blocks)
    assert agent.swap_calls == []


def test_do_switch_model_empty_args_shows_status() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "", printer)
    assert any(
        "provider=Anthropic" in line and "model=claude-opus-4-7" in line
        for line in printer.slash_blocks
    )
    assert agent.swap_calls == []


def test_do_switch_model_shlex_parse_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), 'unclosed "quote', printer)
    assert any("parse error" in line for line in printer.slash_blocks)
    assert agent.swap_calls == []


def test_do_switch_model_unknown_flag_writes_error() -> None:
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_model(_as_agent(agent), "--bogus x", printer)
    assert any("unknown flag" in line for line in printer.slash_blocks)
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
    assert any("claude-sonnet-4-6" in line for line in printer.slash_blocks)


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
        for line in printer.slash_blocks
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
    assert any("no credentials" in line for line in printer.slash_blocks)


def test_do_switch_thinking_full_state_sets_adaptive_show() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        )
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "adaptive-show", printer)
    assert agent.thinking_state == "adaptive-show"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is True
    assert len(agent.change_model_calls) == 1


def test_do_switch_thinking_same_state_skips_model_change() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        ),
        thinking_state="adaptive-show",
        thinking="adaptive",
        show_thinking=True,
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "adaptive-show", printer)
    assert agent.thinking_state == "adaptive-show"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is True
    assert agent.change_model_calls == []


def test_do_switch_thinking_hide_preserves_adaptive_mode() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        )
    )
    printer = RecordingPrinter()
    agent.thinking_state = "adaptive-show"
    do_switch_thinking(_as_agent(agent), "hide", printer)
    assert agent.thinking_state == "adaptive-hide"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is False
    assert len(agent.change_model_calls) == 1


def test_do_switch_thinking_redact_enables_redaction_and_hides() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        )
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "redact", printer)
    assert agent.thinking_state == "redact-hide"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is False
    # The redact-supporting provider triggers a rebuild; the redaction
    # value itself is derived inside ``Agent._provider_build_options``
    # (covered by agent_test.py).
    assert len(agent.change_model_calls) == 1


def test_do_switch_thinking_rejects_models_without_thinking() -> None:
    agent = _FakeAgent(model=_FakeModel(supports_thinking=False))
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "adaptive-show", printer)
    assert "not supported" in printer.slash_blocks[0]
    assert "off-hide" in printer.slash_blocks[0]
    assert agent.change_model_calls == []


def test_do_switch_thinking_allows_off_without_model_thinking() -> None:
    agent = _FakeAgent(model=_FakeModel(supports_thinking=False))
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "off", printer)
    assert agent.thinking_state == "off-hide"
    assert agent.thinking is None


def test_do_switch_thinking_show_errors_from_off() -> None:
    agent = _FakeAgent(thinking_state="off-hide", thinking=None, show_thinking=False)
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "show", printer)
    assert "cannot show" in printer.slash_blocks[0]
    assert agent.change_model_calls == []


def test_do_switch_thinking_no_rebuild_without_provider_support() -> None:
    """A provider without the redact option changes state locally, no rebuild."""
    agent = _FakeAgent(
        model_spec=ModelSpec(provider="Google", auth="env", model_id="gemini-3-pro"),
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "adaptive-show", printer)
    assert agent.thinking_state == "adaptive-show"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is True
    assert agent.change_model_calls == []


def test_do_switch_thinking_rejects_redact_without_provider_support() -> None:
    agent = _FakeAgent(
        model=_FakeModel(model_id="gemini-3-pro", supports_redaction=False),
        model_spec=ModelSpec(provider="Google", auth="env", model_id="gemini-3-pro"),
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "redact", printer)
    assert "not supported" in printer.slash_blocks[0]
    assert "redact-hide" not in printer.slash_blocks[0].split("options:")[1]
    assert agent.change_model_calls == []


def test_do_switch_thinking_rebuild_failure_preserves_state() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        ),
        thinking_state="adaptive-show",
        thinking="adaptive",
        show_thinking=True,
    )
    agent.change_model_side_effect = RuntimeError("rebuild failed")
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "redact", printer)
    assert agent.thinking_state == "adaptive-show"
    assert agent.thinking == "adaptive"
    assert agent.show_thinking is True
    assert "rebuild failed" in printer.slash_blocks[0]


def test_do_switch_effort_bare_lists_current_and_options() -> None:
    """Bare ``/effort`` prints current effort plus the model's valid options."""
    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_effort(_as_agent(agent), "", printer)
    block = printer.slash_blocks[0]
    assert "unset" in block
    assert "options:" in block
    assert "high" in block


def test_do_switch_effort_sets_value() -> None:

    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_effort(_as_agent(agent), "high", printer)
    assert agent.effort == "high"
    assert "high" in printer.slash_blocks[0]


def test_do_switch_effort_rejects_invalid_value() -> None:

    agent = _FakeAgent()
    printer = RecordingPrinter()
    do_switch_effort(_as_agent(agent), "bogus", printer)
    assert agent.effort is None
    assert "must be one of" in printer.slash_blocks[0]


def test_do_switch_effort_clears_with_alias() -> None:
    agent = _FakeAgent(_effort="high")
    printer = RecordingPrinter()
    do_switch_effort(_as_agent(agent), "off", printer)
    assert agent.effort is None
    assert "unset" in printer.slash_blocks[0]


def test_do_switch_effort_none_sets_literal_not_clears() -> None:
    """``none`` is a real effort value on some providers, not a clear alias."""
    agent = _FakeAgent(
        model=_FakeModel(valid_efforts=("none", "low", "high")), _effort="high"
    )
    printer = RecordingPrinter()
    do_switch_effort(_as_agent(agent), "none", printer)
    assert agent.effort == "none"
    assert "none" in printer.slash_blocks[0]


def test_do_switch_thinking_bare_lists_state_and_options() -> None:
    """Bare ``/thinking`` prints current state plus provider's valid options."""
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic", auth="env", model_id="claude-opus-4-7"
        ),
        thinking_state="adaptive-hide",
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "", printer)
    block = printer.slash_blocks[0]
    assert "adaptive-hide" in block
    assert "options:" in block
    assert "redact-hide" in block
    assert agent.change_model_calls == []


def test_do_switch_thinking_bare_options_provider_specific() -> None:
    """A no-redaction provider omits ``redact-hide`` from the listed options."""
    agent = _FakeAgent(
        model=_FakeModel(model_id="gemini-3-pro", supports_redaction=False),
        model_spec=ModelSpec(provider="Google", auth="env", model_id="gemini-3-pro"),
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "", printer)
    options = printer.slash_blocks[0].split("options:")[1]
    assert "adaptive-show" in options
    assert "redact-hide" not in options


def test_do_switch_thinking_show_errors_from_redact() -> None:
    agent = _FakeAgent(
        model_spec=ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="claude-opus-4-7",
        ),
        thinking_state="redact-hide",
        show_thinking=False,
    )
    printer = RecordingPrinter()
    do_switch_thinking(_as_agent(agent), "show", printer)
    assert "cannot show" in printer.slash_blocks[0]
    assert agent.change_model_calls == []


@pytest.mark.asyncio
async def test_do_login_no_spec_writes_error() -> None:
    agent = _FakeAgent(model_spec=None)
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
    assert any("no model spec" in line for line in printer.slash_blocks)


@pytest.mark.asyncio
async def test_do_login_success_delegates_to_relogin() -> None:
    """Slash handler delegates to ``Agent.relogin`` and prints confirmation."""
    agent = _FakeAgent()
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
    assert agent.relogin_calls == 1
    assert any("re-authenticated" in line for line in printer.slash_blocks)


@pytest.mark.asyncio
async def test_do_login_relogin_error_writes_to_printer() -> None:
    """``Agent.relogin`` errors surface verbatim to the printer."""
    agent = _FakeAgent()
    agent.relogin_side_effect = RuntimeError("oauth failed")
    printer = RecordingPrinter()
    await do_login(_as_agent(agent), printer)
    assert any("oauth failed" in line for line in printer.slash_blocks)


def test_format_tasks_no_registry_header_only() -> None:
    agent = _FakeAgent()
    empty: dict[str, AgentLike] = {}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        empty,
    ):
        out = format_tasks(_as_agent(agent))
    assert out.startswith("sagent: 0 agent(s)")
    assert "foreground" in out
    assert "background" in out


def test_format_tasks_lists_registered_agent_fg_idle() -> None:
    agent = _FakeAgent()
    other = MagicMock()
    other.runtime.model_call = None
    other.runtime.compact_task = None
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
    assert "Agent_0/bg-1" in out
    assert "Bash" in out
    assert "running" in out


def test_format_tasks_namespaces_same_job_id_by_agent_label() -> None:
    agent = _FakeAgent()
    task = MagicMock()
    task.done.return_value = False
    task.cancelled.return_value = False
    job = BackgroundTaskEntry(
        task=task,
        tool_name="Bash",
        queue_id="job-1",
        started=0.0,
        kind="tool",
        hidden=False,
    )
    first = MagicMock()
    first.work = None
    first.background = {"job-1": job}
    second = MagicMock()
    second.work = None
    second.background = {"job-1": job}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"Agent": first, "fix-tools": second},
    ):
        out = format_tasks(_as_agent(agent))
    assert "Agent/job-1" in out
    assert "fix-tools/job-1" in out


def test_run_repl_invokes_replay_messages() -> None:
    # --resume / --continue rely on replay_messages to render persisted
    # history into scrollback. The unit test on replay_messages itself
    # stays green even when the call site is dropped (see eb4700ef).
    src = inspect.getsource(run_repl)
    assert "replay_messages(" in src


@pytest.mark.asyncio
async def test_run_repl_unwinds_observers_and_before_tool_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-entering ``run_repl`` on the same agent must not stack state.

    F134: the body appends 2 observers and patches ``before_tool_spawn``
    via ``install_input_queue_committer``. Without a ``finally`` that
    detaches them, repeated entries pile up across re-entries.
    """
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))

    def _sentinel_observer(event: RuntimeEvent) -> None:
        del event

    runtime.observers.append(_sentinel_observer)
    original_before_tool_spawn = runtime.before_tool_spawn
    observers_before = list(runtime.observers)

    @dataclass(slots=True, kw_only=True)
    class _Holder:
        runtime: agent_runtime.AgentRuntime
        show_thinking: bool = False
        name: str = "test"
        status: str | None = None
        session_dir: object | None = None
        background: dict[str, BackgroundTaskEntry] = field(default_factory=dict)

        async def serve_forever(self) -> None:
            return None

        def shutdown(self, *, force: bool = False) -> None:
            del force

        def cancel_background(self, key: str) -> None:
            del key

    holder = _Holder(runtime=runtime)

    @contextlib.contextmanager
    def _stub_patch_stdout(**_kwargs: object):
        yield

    def _stub_console(**_kwargs: object) -> MagicMock:
        return MagicMock()

    def _stub_session(*_args: object, **_kwargs: object) -> MagicMock:
        return MagicMock()

    def _stub_history(_path: object) -> MagicMock:
        return MagicMock()

    def _stub_input_source(*_args: object, **_kwargs: object) -> MagicMock:
        return MagicMock()

    def _stub_replay(_agent: object, _printer: object) -> None:
        return None

    fake_pump: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))

    def _stub_spawn(
        _agent: object, _source: object, **_kwargs: object
    ) -> asyncio.Task[None]:
        return fake_pump

    run_repl_mod = sys.modules["sagent.repl.run_repl"]

    monkeypatch.setattr(run_repl_mod, "patch_stdout", _stub_patch_stdout)
    monkeypatch.setattr(run_repl_mod, "Console", _stub_console)
    monkeypatch.setattr(run_repl_mod, "PromptSession", _stub_session)
    monkeypatch.setattr(run_repl_mod, "FileHistory", _stub_history)
    monkeypatch.setattr(run_repl_mod, "PromptToolkitInputSource", _stub_input_source)
    monkeypatch.setattr(run_repl_mod, "replay_messages", _stub_replay)
    monkeypatch.setattr(run_repl_mod, "spawn_repl_pump", _stub_spawn)

    await run_repl(cast(Agent, holder), history=tmp_path / "history")

    assert runtime.observers == observers_before, (
        f"run_repl must detach the observers it installed; got {runtime.observers}"
    )
    assert runtime.before_tool_spawn is original_before_tool_spawn, (
        f"run_repl must restore before_tool_spawn; got {runtime.before_tool_spawn}"
    )


@pytest.mark.asyncio
async def test_repl_teardown_skips_persistent_subagent_tasks_after_shutdown() -> None:
    agent = _FakeAgent()
    tool_task = asyncio.create_task(asyncio.sleep(10.0))
    child_task = asyncio.create_task(asyncio.sleep(10.0))
    agent.background = {
        "tool": BackgroundTaskEntry(
            task=tool_task,
            tool_name="Bash",
            queue_id="tool",
            started=0.0,
            kind="tool",
            hidden=False,
        ),
        "child": BackgroundTaskEntry(
            task=child_task,
            tool_name="Agent",
            queue_id="child",
            started=0.0,
            kind="persistent_subagent",
            persistent_run_id="run-child",
            hidden=False,
        ),
    }
    try:
        assert _background_tasks_for_repl_cancel(_as_agent(agent)) == [tool_task]
    finally:
        _ = tool_task.cancel()
        _ = child_task.cancel()
        await asyncio.gather(tool_task, child_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_input_queue_committer_observer_pushes_deferred_on_agent_idle() -> None:
    """``AgentIdle`` with a deferred message pushes ``UserDeferredMessage``; pane cleared."""
    queues = InputQueues(deferred=QueuedInputBlock(text="elephant\n\nbanana\n\nchair"))
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    observer = _input_queue_committer_observer(cast(Agent, holder), queues)
    observer(AgentIdle())
    assert not queues.has_any()
    pushed = await runtime.inbox.drain()
    assert len(pushed) == 1
    item = pushed[0]
    assert isinstance(item, UserDeferredMessage)
    assert item.text == "elephant\n\nbanana\n\nchair"


def test_input_queue_committer_observer_ignores_non_agent_idle_events() -> None:
    """Non-flush events leave ``queued_input`` and the inbox untouched."""
    queues = InputQueues(deferred=QueuedInputBlock(text="elephant"))
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    observer = _input_queue_committer_observer(cast(Agent, holder), queues)
    observer(UserMessage(text="real submission"))
    observer(ModelResponseError(RuntimeError("x")))
    observer(ModelIdle())
    assert queues.deferred is not None
    assert queues.deferred.text == "elephant"
    assert runtime.inbox._queue.empty()


@pytest.mark.asyncio
async def test_input_queue_committer_observer_flushes_deferred_on_clear_complete() -> (
    None
):
    """``ClearComplete`` flushes deferred input the same as ``AgentIdle``.

    A self-issued ``Clear`` (``AgentSelf context=clear``) arms ``AWAIT_USER``,
    so ``_fully_drained`` stays False and ``AgentIdle`` never publishes -- the
    deferred queue would wedge until Ctrl+D. ``ClearComplete`` is the signal
    Halt / error do not emit, so keying the deferred flush on it releases the
    wedge without claiming ``AWAIT_USER`` is idle.
    """
    queues = InputQueues(deferred=QueuedInputBlock(text="resume me"))
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    observer = _input_queue_committer_observer(cast(Agent, holder), queues)
    observer(ClearComplete())
    assert not queues.has_any()
    pushed = await runtime.inbox.drain()
    assert len(pushed) == 1
    assert isinstance(pushed[0], UserDeferredMessage)
    assert pushed[0].text == "resume me"


def test_input_queue_committer_observer_ignores_agent_idle_when_empty() -> None:
    """``AgentIdle`` with an empty queue is a no-op (no spurious push)."""
    queues = InputQueues()
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    observer = _input_queue_committer_observer(cast(Agent, holder), queues)
    observer(AgentIdle())
    assert runtime.inbox._queue.empty()


@pytest.mark.asyncio
async def test_startup_idle_flushes_staged_queue_on_first_agent_idle() -> None:
    """At REPL startup the runtime naturally emits ``AgentIdle`` before its first
    blocking drain, so pre-staged deferred blocks are flushed without any
    synthetic startup pulse.
    """
    queues = InputQueues(
        deferred=QueuedInputBlock(text="were we implementing issue 25?")
    )
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    _ = install_input_queue_committer(cast(Agent, holder), queues)

    flushed = asyncio.Event()

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, AgentIdle) and not queues.has_any():
            flushed.set()
            runtime.inbox.push_back(Quit())

    runtime.observers.append(_watch)
    await asyncio.wait_for(runtime.run_forever(), timeout=2.0)

    assert flushed.is_set()
    assert not queues.has_any()


@pytest.mark.asyncio
async def test_startup_idle_not_fired_when_history_needs_model() -> None:
    """An incoming user message fires the model BEFORE ``AgentIdle``.

    Setup: inbox already holds a ``UserMessage`` (the user typed
    something). The runtime drains it, fires the model, completes the
    round. Only after that round (with no follow-up work) does the
    runtime idle and publish ``AgentIdle``. The committer must not see
    ``AgentIdle`` before ``ModelIdle`` -- otherwise the deferred queue
    would flush prematurely.
    """
    queues = InputQueues(deferred=QueuedInputBlock(text="for later"))
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    _ = install_input_queue_committer(cast(Agent, holder), queues)
    runtime.inbox.push_back(UserMessage(text="answer this first"))

    order: list[type] = []

    def _watch(event: RuntimeEvent) -> None:
        if isinstance(event, (ModelIdle, AgentIdle)):
            order.append(type(event))
            if isinstance(event, AgentIdle):
                runtime.inbox.push_back(Quit())

    runtime.observers.append(_watch)
    await asyncio.wait_for(runtime.run_forever(), timeout=2.0)

    # ``ModelIdle`` must precede ``AgentIdle`` -- the deferred queue
    # was not flushed before the model handled the active user message.
    assert order[:2] == [ModelIdle, AgentIdle]


@dataclass(kw_only=True, slots=True)
class _TextOnlyModel:
    """One-shot scripted model: returns text, no tool calls."""

    text: str = "ok"

    async def stream(
        self,
        history: list[ModelContextEvent],
        publish: Callable[[RuntimeEvent], None],
    ) -> AssistantMessage:
        del history
        for ch in self.text:
            publish(ModelResponsePartial(ch))
        return AssistantMessage(text=self.text)


@dataclass(kw_only=True, slots=True)
class _ScriptedModel:
    """Returns successive scripted assistant messages, one per call."""

    messages: list[AssistantMessage]
    _index: int = 0

    async def stream(
        self,
        history: list[ModelContextEvent],
        publish: Callable[[RuntimeEvent], None],
    ) -> AssistantMessage:
        del history
        message = self.messages[min(self._index, len(self.messages) - 1)]
        self._index += 1
        publish(ModelResponsePartial(message.text))
        return message


@dataclass(kw_only=True, slots=True)
class _SlowTool:
    """A tool that blocks long enough to still be running at detach time."""

    name: str = "echo"
    call_count: int = 0

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        self.call_count += 1
        await asyncio.sleep(10.0)
        return ToolResult(call_id="", content="tool output")

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None


@dataclass(kw_only=True, slots=True)
class _GatedTool:
    """A tool that runs until ``release`` is set; lets a test hold a cohort.

    Unlike ``_SlowTool`` (a fixed long sleep), this completes the moment
    the test releases it, so a deferred message staged mid-cohort can
    actually fire once the cohort drains.
    """

    release: asyncio.Event
    name: str = "echo"
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        self.started.set()
        await self.release.wait()
        return ToolResult(call_id="", content="tool output")

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None


@dataclass(kw_only=True, slots=True)
class _GatedModel:
    """Streams once it is released; lets a test hold the runtime mid-stream."""

    release: asyncio.Event
    text: str = "answered"
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def stream(
        self,
        history: list[ModelContextEvent],
        publish: Callable[[RuntimeEvent], None],
    ) -> AssistantMessage:
        del history
        self.started.set()
        await self.release.wait()
        publish(ModelResponsePartial(self.text))
        return AssistantMessage(text=self.text)


@dataclass(kw_only=True, slots=True)
class _GatedCompactor:
    """Blocks compaction until released; holds the runtime mid-compaction."""

    release: asyncio.Event
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def compact(
        self,
        tape: object,
        context: object,
        model: object,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        del tape, context, model, custom_instructions
        self.started.set()
        await self.release.wait()
        return ContextSplice(
            ref=mint_ref(),
            mask=(),
            insert_after=None,
            payload=(),
            strategy="test-noop",
        )


def _make_kb_event(text: str) -> MagicMock:
    """Build a prompt-toolkit key event whose buffer holds ``text``.

    The harness drives ``_kb_submit`` / ``_kb_defer`` directly rather
    than through the prompt-toolkit dispatcher; the handlers only
    touch ``event.current_buffer.text`` / ``.cursor_position`` /
    ``.reset()`` / ``.append_to_history()``.
    """
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = len(text)
    buf.document.text_before_cursor = text
    buf.document.text = text
    buf.history.get_strings.return_value = []
    event = MagicMock()
    event.current_buffer = buf
    return event


async def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout_sec: float = 2.0,
) -> None:
    """Yield to the event loop until ``predicate`` is True or timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while not predicate():
        if loop.time() >= deadline:
            return
        await asyncio.sleep(0)


@contextlib.asynccontextmanager
async def _running_runtime(
    runtime: agent_runtime.AgentRuntime,
):
    """Start ``run_forever`` and tear it down on exit.

    Lets the test push to the inbox at any point and wait on predicates
    against history; ``Quit`` is pushed at exit to drain the engine.
    """
    task = asyncio.create_task(runtime.run_forever())
    try:
        yield
    finally:
        runtime.inbox.push_back(Quit())
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)


def _harness_runtime() -> tuple[
    agent_runtime.AgentRuntime, _RuntimeHolder, InputQueues
]:
    """Build a runtime + queues + committer wired together."""
    queues = InputQueues()
    runtime = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="ok"))
    holder = _RuntimeHolder(runtime=runtime)
    _ = install_input_queue_committer(cast(Agent, holder), queues)
    return runtime, holder, queues


def _history_user_texts(runtime: agent_runtime.AgentRuntime) -> list[str]:
    return [m.text for m in runtime.context().messages if isinstance(m, UserMessage)]


def _history_has(runtime: agent_runtime.AgentRuntime, needle: str) -> bool:
    """True when ``needle`` appears in any user message (coalesced or not)."""
    return any(needle in text for text in _history_user_texts(runtime))


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_harness_enter_at_cursor_zero_fresh_session_reaches_history() -> None:
    """Enter on a fresh session: ``UserMessage`` reaches history."""
    runtime, holder, queues = _harness_runtime()
    nav = NavState()
    async with _running_runtime(runtime):
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("hi"))
        await _wait_for(lambda: "hi" in _history_user_texts(runtime))
    assert "hi" in _history_user_texts(runtime)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_harness_tab_at_cursor_zero_fresh_session_drains_to_history() -> None:
    """Tab on a fresh session: deferred drains via the initial ``AgentIdle``."""
    runtime, holder, queues = _harness_runtime()
    nav = NavState()
    async with _running_runtime(runtime):
        _kb_defer(cast(Agent, holder), queues, nav, _make_kb_event("for later"))
        await _wait_for(lambda: "for later" in _history_user_texts(runtime))
    assert "for later" in _history_user_texts(runtime), (
        "deferred queue must drain on the first AgentIdle of a fresh session"
    )


async def _wait_until_idle(idle_events: list[type]) -> None:
    """Yield until at least one ``AgentIdle`` has been observed."""
    await _wait_for(lambda: AgentIdle in idle_events)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_harness_tab_after_idle_turn_drains_to_history() -> None:
    """Tab AFTER a completed model turn: deferred must still drain.

    The bug class: ``_was_idle`` flips True after the first AgentIdle,
    then stays True. Tab without ``gate_armed`` stages to the deferred
    queue but no further ``AgentIdle`` fires, so the text is stuck.
    This is the wider variant of bug #2 the cold-start fix missed.
    """
    runtime, holder, queues = _harness_runtime()
    nav = NavState()
    idle_events: list[type] = []
    runtime.observers.append(
        lambda ev: idle_events.append(type(ev)) if isinstance(ev, AgentIdle) else None
    )
    async with _running_runtime(runtime):
        # First turn: user types, model answers, agent idles. Wait for
        # the AgentIdle event so we know the runtime is truly settled
        # (not just mid-stream with "first" already appended).
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("first"))
        await _wait_until_idle(idle_events)
        idle_events.clear()
        # Now agent is fully idle and ``_was_idle=True``. Tab a deferred
        # block. Without the fix, nothing wakes the drain.
        _kb_defer(cast(Agent, holder), queues, nav, _make_kb_event("after idle tab"))
        await _wait_for(lambda: "after idle tab" in _history_user_texts(runtime))
    assert "after idle tab" in _history_user_texts(runtime), (
        "Tab after a completed turn must still drain; the deferred queue"
        " is stuck because no AgentIdle re-fires on an already-idle runtime"
    )


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_queued_pane_message_detaches_running_tool() -> None:
    """REGRESSION: a queue-pane message must still trigger a tool detach.

    Enter-while-busy stages into the queue pane; the committer's
    ``before_tool_spawn`` hook pops it at ``ModelResponseComplete`` and
    the runtime relegates the model's tool calls to the background so the
    queued message cuts in. Without the pop hook (or with a queue the
    runtime never consults) the tool would run inline and the queued
    message would wait a full round -- the "type to redirect" path lost.
    """
    queues = InputQueues()
    runtime = agent_runtime.AgentRuntime(
        model=_ScriptedModel(
            messages=[
                AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
                AssistantMessage(text="queued answered"),
            ]
        ),
        tools=[_SlowTool()],
    )
    holder = _RuntimeHolder(runtime=runtime)
    _ = install_input_queue_committer(cast(Agent, holder), queues)

    idle = asyncio.Event()

    def _quit_on_idle(event: RuntimeEvent) -> None:
        if isinstance(event, ModelIdle):
            idle.set()
            runtime.inbox.push_back(Quit())

    runtime.observers.append(_quit_on_idle)

    # Stage a queue-pane message (as Enter-while-busy would) BEFORE the
    # round completes, so ``before_tool_spawn`` pops it at MRC.
    queues.stage_queue("cut in line")
    task = asyncio.create_task(runtime.run_forever())
    try:
        runtime.inbox.push_back(UserMessage(text="start"))
        await asyncio.wait_for(idle.wait(), timeout=2.0)
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    tool_results = [m for m in runtime.context().messages if isinstance(m, ToolResult)]
    assert any(
        r.call_id == "t1" and r.content == DETACHED_PLACEHOLDER for r in tool_results
    ), f"queued message must detach the tool; results={tool_results!r}"
    assert "cut in line" in _history_user_texts(runtime)
    assert not queues.has_any()


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_typed_input_reaches_model_in_every_runtime_state() -> None:
    """EXHAUSTIVE: typed Enter AND Tab must reach the model from every state.

    The dispatch-vs-stage decision plus the commit machinery is exercised
    end-to-end against the REAL runtime for each reachable state, rather
    than cherry-picked cases. In every state the typed message must land
    in history (reach the model) and nothing must be left stranded in a
    pane. This is the regression net for "I typed and it never got
    through" -- the bug class that keeps recurring.

    States covered: cold idle, idle-after-a-turn, mid-stream, mid-cohort,
    post-Halt (gate armed), mid-compaction. Both keys are exercised:
    Enter (queue intent) and Tab (defer intent). They differ in WHEN the
    message reaches the model (Enter preempts mid-cohort; Tab waits for
    idle), but the end invariant is identical: it reaches the model and
    nothing is stranded once the runtime settles.
    """

    async def _drive(setup: str, key: str) -> None:
        nav = NavState()
        release = asyncio.Event()
        idle_events: list[type] = []

        if setup == "mid_stream":
            model: object = _GatedModel(release=release)
        elif setup == "mid_cohort":
            model = _ScriptedModel(
                messages=[
                    AssistantMessage(
                        tool_calls=(ToolCall(id="t1", name="echo", args={}),)
                    ),
                    AssistantMessage(text="done"),
                ]
            )
        else:
            model = _TextOnlyModel(text="ok")

        compactor = _GatedCompactor(release=release) if setup == "mid_compact" else None
        runtime = agent_runtime.AgentRuntime(
            model=cast(agent_runtime.Model, model),
            tools=[_GatedTool(release=release)] if setup == "mid_cohort" else [],
            compactor=cast(agent_runtime.Compactor, compactor)
            if compactor is not None
            else None,
        )
        queues = InputQueues()
        holder = _RuntimeHolder(runtime=runtime)
        _ = install_input_queue_committer(cast(Agent, holder), queues)
        runtime.observers.append(
            lambda ev: (
                idle_events.append(type(ev)) if isinstance(ev, AgentIdle) else None
            )
        )

        async with _running_runtime(runtime):
            if setup == "cold_idle":
                pass
            elif setup == "idle_after_turn":
                runtime.inbox.push_back(UserMessage(text="warmup"))
                await _wait_until_idle(idle_events)
            elif setup == "mid_stream":
                runtime.inbox.push_back(UserMessage(text="warmup"))
                await _wait_for(cast(_GatedModel, model).started.is_set)
            elif setup == "mid_cohort":
                runtime.inbox.push_back(UserMessage(text="warmup"))
                await _wait_for(
                    lambda: bool(runtime.cohort) and runtime.model_call is None
                )
            elif setup == "post_halt":
                runtime.inbox.push_back(UserMessage(text="warmup"))
                await _wait_until_idle(idle_events)
                runtime.inbox.push_back(Halt())
                await _wait_for(lambda: runtime.inbox.gate_armed)
            elif setup == "mid_compact":
                runtime.inbox.push_back(UserMessage(text="warmup"))
                await _wait_until_idle(idle_events)
                runtime.inbox.push_back(Compact(args=""))
                await _wait_for(cast(_GatedCompactor, compactor).started.is_set)

            # The universal action: the user types a message and submits
            # it with the key under test (Enter -> queue, Tab -> defer).
            handler = _kb_submit if key == "enter" else _kb_defer
            handler(cast(Agent, holder), queues, nav, _make_kb_event("REACHME"))

            # Mechanism check, captured BEFORE release: in a busy state the
            # message must STAGE into the matching pane rather than dispatch
            # -- except Enter mid-cohort, which preempts. This pins the
            # stage-vs-dispatch decision the "reaches model" invariant
            # cannot see (a wrongly-dispatched deferred message still
            # eventually reaches the model).
            staged_now = setup in ("mid_stream", "mid_compact") or (
                setup == "mid_cohort" and key == "tab"
            )
            if staged_now:
                pane = queues.deferred if key == "tab" else queues.queue
                assert pane is not None, (
                    f"[{setup}/{key}] busy input must stage into its pane,"
                    f" not dispatch; queue={queues.queue!r}"
                    f" deferred={queues.deferred!r}"
                )
                assert pane.text == "REACHME"

            # Release any held model/compaction so the runtime can settle.
            release.set()
            await _wait_for(lambda: _history_has(runtime, "REACHME"), timeout_sec=3.0)

        assert _history_has(runtime, "REACHME"), (
            f"[{setup}/{key}] typed input never reached the model;"
            f" history={_history_user_texts(runtime)}"
            f" queue={queues.queue!r} deferred={queues.deferred!r}"
        )
        assert not queues.has_any(), (
            f"[{setup}/{key}] message stranded in a pane:"
            f" {queues.queue!r} / {queues.deferred!r}"
        )
        if setup == "mid_cohort":
            # Defer semantics: Enter preempts (the tool detaches); Tab waits
            # for the round chain (the tool completes normally, no detach).
            results = [
                m for m in runtime.context().messages if isinstance(m, ToolResult)
            ]
            detached = any(
                r.call_id == "t1" and r.content == DETACHED_PLACEHOLDER for r in results
            )
            if key == "enter":
                assert detached, (
                    "[mid_cohort/enter] Enter must preempt and detach the tool;"
                    f" results={results!r}"
                )
            else:
                assert not detached, (
                    "[mid_cohort/tab] Tab must DEFER, not preempt; the tool must"
                    f" complete normally (no detach). results={results!r}"
                )

    for state in (
        "cold_idle",
        "idle_after_turn",
        "mid_stream",
        "mid_cohort",
        "post_halt",
        "mid_compact",
    ):
        for key in ("enter", "tab"):
            await _drive(state, key)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_enter_mid_cohort_detaches_running_tool() -> None:
    """REGRESSION: Enter while TOOLS are running must detach them.

    Distinct from the mid-stream case: here ``model_call is None`` and a
    cohort is executing. The runtime's ``UserMessage`` handler preempts
    and detaches the cohort -- but only if the REPL DISPATCHES the message
    to the inbox. Staging into the queue pane (the busy default) never
    reaches that handler, so the tool runs to completion: no detach. The
    user typed to redirect and nothing happened.
    """
    queues = InputQueues()
    nav = NavState()
    runtime = agent_runtime.AgentRuntime(
        model=_ScriptedModel(
            messages=[
                AssistantMessage(tool_calls=(ToolCall(id="t1", name="echo", args={}),)),
                AssistantMessage(text="redirected"),
            ]
        ),
        tools=[_SlowTool()],
    )
    holder = _RuntimeHolder(runtime=runtime)
    _ = install_input_queue_committer(cast(Agent, holder), queues)

    async with _running_runtime(runtime):
        runtime.inbox.push_back(UserMessage(text="start"))
        # Wait until the cohort is actually running (model_call cleared).
        await _wait_for(lambda: bool(runtime.cohort) and runtime.model_call is None)
        # Type to redirect while tools run.
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("redirect"))
        await _wait_for(lambda: "redirect" in _history_user_texts(runtime))

    tool_results = [m for m in runtime.context().messages if isinstance(m, ToolResult)]
    assert any(
        r.call_id == "t1" and r.content == DETACHED_PLACEHOLDER for r in tool_results
    ), (
        "Enter while a tool cohort runs must detach the tool; instead it"
        f" ran inline. results={tool_results!r}"
    )
    assert "redirect" in _history_user_texts(runtime)


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_harness_enter_after_halt_dispatches_not_wedged() -> None:
    """REGRESSION (Bug 2): Enter after a Halt must reach the model.

    Halt arms ``AWAIT_USER`` and publishes neither ``AgentIdle`` nor
    ``ClearComplete``, so a queue-staged message would never commit -- the
    user's Enter would wedge in the queue pane. The user's post-Halt Enter
    is precisely what releases the gate, so it must dispatch directly.
    """
    runtime, holder, queues = _harness_runtime()
    nav = NavState()
    idle_events: list[type] = []
    runtime.observers.append(
        lambda ev: idle_events.append(type(ev)) if isinstance(ev, AgentIdle) else None
    )
    async with _running_runtime(runtime):
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("first"))
        await _wait_until_idle(idle_events)
        idle_events.clear()
        # Simulate Ctrl+C: Halt arms AWAIT_USER. Wait until the gate is armed.
        runtime.inbox.push_back(Halt())
        await _wait_for(lambda: runtime.inbox.gate_armed)
        # Now type + Enter. This must dispatch (release the gate), not stage.
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("resume me"))
        await _wait_for(lambda: "resume me" in _history_user_texts(runtime))
    assert "resume me" in _history_user_texts(runtime), (
        "Enter after Halt must dispatch to release AWAIT_USER, not wedge in"
        " the queue pane"
    )
    assert not queues.has_any()


@pytest.mark.asyncio
@pytest.mark.real_sleep
async def test_harness_enter_at_nav_stop_on_idle_dispatches() -> None:
    """Enter at a nav stop on a fully-idle agent: text reaches history.

    Setup: a message sits in the queue pane (staged while busy earlier);
    the agent is now idle. Up unlifts it into the buffer; Enter at the
    nav stop dispatches it immediately (idle -> dispatch), so "draft"
    reaches history with nothing left staged.
    """
    runtime, holder, queues = _harness_runtime()
    nav = NavState()
    idle_events: list[type] = []
    runtime.observers.append(
        lambda ev: idle_events.append(type(ev)) if isinstance(ev, AgentIdle) else None
    )
    async with _running_runtime(runtime):
        _kb_submit(cast(Agent, holder), queues, nav, _make_kb_event("warm up"))
        await _wait_until_idle(idle_events)
        idle_events.clear()

        # A queued message + idle agent. Up unlifts it, Enter dispatches.
        queues.stage_queue("draft")
        buf = _make_kb_event("")
        _kb_up(queues, nav, buf)
        assert buf.current_buffer.text == "draft"
        _kb_submit(cast(Agent, holder), queues, nav, buf)

        await _wait_for(lambda: "draft" in _history_user_texts(runtime))
    assert "draft" in _history_user_texts(runtime)
    assert not queues.has_any()


@pytest.mark.asyncio
async def test_queued_input_committed_and_cleared_on_model_idle() -> None:
    """Integration: Tab-staged ``queued_input`` commits as ``UserDeferredMessage``
    on ``ModelIdle`` and the local list clears.

    Wires the option-1 contract end-to-end: REPL stages locally via
    Tab (here we pre-populate the list to simulate that), the
    committer observer sees ``ModelIdle`` after the round answering
    the initial ``UserMessage`` settles, and pushes the coalesced
    queue back as ``UserDeferredMessage``. The runtime then drains it
    at the next gate-section pass and fires a fresh round.
    """
    queues = InputQueues(deferred=QueuedInputBlock(text="elephant\n\nbanana\n\nchair"))
    agent = agent_runtime.AgentRuntime(model=_TextOnlyModel(text="committed"))

    class _Holder:
        runtime = agent

    _ = install_input_queue_committer(cast(Agent, _Holder()), queues)

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

    assert not queues.has_any()
    queued_texts = [
        m.text for m in agent.context().messages if isinstance(m, UserMessage)
    ]
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

        def serialize_key(self, args: Mapping[str, object]) -> str | None:
            del args
            return None

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            del args
            tool_started.set()
            await release_tool.wait()
            return ToolResult(call_id="", content="completed late")

    @dataclass(kw_only=True, slots=True)
    class _TwoRoundModel:
        call_histories: list[list[ModelContextEvent]] = field(default_factory=list)
        _i: int = field(default=0, init=False)

        async def stream(
            self,
            history: list[ModelContextEvent],
            publish: Callable[[RuntimeEvent], None],
        ) -> AssistantMessage:
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
                publish(ModelResponsePartial(ch))
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
        await wait_until(lambda: len(model.call_histories) >= 2)
        release_tool.set()
        await wait_until(
            lambda: any(isinstance(m, ToolResult) for m in runtime.context().messages)
        )
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
        for m in runtime.context().messages
        if isinstance(m, ToolResult)
        and m.content in {DETACHED_PLACEHOLDER, "completed late"}
    ]
    assert placeholders, (
        f"Expected the slow tool to be detached or eventually splice"
        f" 'completed late' into history; saw history:"
        f" {[type(m).__name__ for m in runtime.context().messages]}"
    )


def _make_subagent_job(queue_id: str = "child-1") -> BackgroundTaskEntry:
    """Return a BackgroundTaskEntry with kind='persistent_subagent'."""
    task = MagicMock()
    task.done.return_value = False
    task.cancelled.return_value = False
    return BackgroundTaskEntry(
        task=task,
        tool_name="AgentSpawn",
        queue_id=queue_id,
        started=0.0,
        kind="persistent_subagent",
        persistent_run_id=f"run-{queue_id}",
        hidden=False,
        delay_sec=0.0,
    )


def _make_runtime(
    *,
    model_call: object = None,
    compact_task: object = None,
    cohort: set[str] | None = None,
    gate_armed: bool = False,
) -> MagicMock:
    rt = MagicMock()
    rt.model_call = model_call
    rt.compact_task = compact_task
    rt.cohort = cohort if cohort is not None else set()
    rt.inbox = MagicMock()
    rt.inbox.gate_armed = gate_armed
    return rt


def test_subagent_phase_stopped_when_task_done() -> None:
    task = MagicMock()
    task.done.return_value = True
    task.cancelled.return_value = False
    task.exception.return_value = None
    job = BackgroundTaskEntry(
        task=cast("asyncio.Task[object]", task),
        tool_name="AgentSpawn",
        queue_id="child-1",
        started=0.0,
        kind="persistent_subagent",
        persistent_run_id="run-child-1",
        hidden=False,
        delay_sec=0.0,
    )
    assert _subagent_phase(job) == "stopped"


def test_format_tasks_non_persistent_job_crashed_shows_errored() -> None:
    """A non-persistent background job that crashed must NOT show
    "completed" -- the operator needs to tell a crash from a graceful
    finish, same as the persistent-subagent ``_subagent_phase`` path.
    """
    agent = _FakeAgent()
    task = MagicMock()
    task.done.return_value = True
    task.cancelled.return_value = False
    task.exception.return_value = RuntimeError("boom")
    crashed_job = BackgroundTaskEntry(
        task=cast("asyncio.Task[object]", task),
        tool_name="Bash",
        queue_id="job-x",
        started=0.0,
        kind="tool",
        hidden=False,
        delay_sec=0.0,
    )
    other = MagicMock()
    other.work = None
    other.background = {"job-x": crashed_job}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"agent-0": other},
    ):
        out = format_tasks(_as_agent(agent))
    assert "errored" in out, (
        f"crashed background job must surface as 'errored', not"
        f" 'completed'; got {out!r}"
    )


def test_subagent_phase_errored_when_task_crashed() -> None:
    """A persistent subagent that crashed must NOT show as plain 'stopped'.

    Operators need to tell a graceful exit from a crash. Conflating
    both under "stopped" hides failures behind UI parity with normal
    completion.
    """
    task = MagicMock()
    task.done.return_value = True
    task.cancelled.return_value = False
    task.exception.return_value = RuntimeError("boom")
    job = BackgroundTaskEntry(
        task=cast("asyncio.Task[object]", task),
        tool_name="AgentSpawn",
        queue_id="child-crashed",
        started=0.0,
        kind="persistent_subagent",
        persistent_run_id="run-child-crashed",
        hidden=False,
        delay_sec=0.0,
    )
    assert _subagent_phase(job) == "errored", (
        f"crashed persistent subagent must be distinguishable from a"
        f" graceful exit; got {_subagent_phase(job)!r}"
    )


def test_subagent_phase_running_when_child_not_in_registry() -> None:
    job = _make_subagent_job("missing-child")
    empty_registry: dict[str, object] = {}
    with patch(
        "sagent.repl.run_repl.agent_registry",
        empty_registry,
    ):
        assert _subagent_phase(job) == "running"


def test_subagent_phase_running_when_model_call_active() -> None:
    job = _make_subagent_job("child-1")
    child = MagicMock()
    child.runtime = _make_runtime(model_call=MagicMock())
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"child-1": child},
    ):
        assert _subagent_phase(job) == "running"


def test_subagent_phase_compacting_when_compact_task_active() -> None:
    job = _make_subagent_job("child-1")
    child = MagicMock()
    child.runtime = _make_runtime(compact_task=MagicMock())
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"child-1": child},
    ):
        assert _subagent_phase(job) == "compacting"


def test_subagent_phase_tool_wait_when_cohort_nonempty() -> None:
    job = _make_subagent_job("child-1")
    child = MagicMock()
    child.runtime = _make_runtime(cohort={"call-1"})
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"child-1": child},
    ):
        assert _subagent_phase(job) == "tool-wait"


def test_subagent_phase_gate_armed() -> None:
    job = _make_subagent_job("child-1")
    child = MagicMock()
    child.runtime = _make_runtime(gate_armed=True)
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"child-1": child},
    ):
        assert _subagent_phase(job) == "gate-armed"


def test_subagent_phase_idle_when_all_fields_quiet() -> None:
    """All-quiet child runtime must show 'idle', not 'running' (the bug)."""
    job = _make_subagent_job("child-1")
    child = MagicMock()
    child.runtime = _make_runtime()
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"child-1": child},
    ):
        assert _subagent_phase(job) == "idle"


def test_format_tasks_persistent_subagent_shows_idle_phase() -> None:
    """``format_tasks`` must display 'idle' for a quiet persistent subagent."""
    agent = _FakeAgent()
    job = _make_subagent_job("child-1")
    parent = MagicMock()
    parent.work = None
    parent.background = {"child-1": job}
    child = MagicMock()
    child.work = None
    child.background = {}
    child.runtime = _make_runtime()
    with patch(
        "sagent.repl.run_repl.agent_registry",
        {"parent": parent, "child-1": child},
    ):
        out = format_tasks(_as_agent(agent))
    assert "idle" in out
    assert "parent/child-1" in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
