"""Tests for ``tools.AgentSpawn`` - the factory tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import dataclasses

import pytest

from sagent.agent import Agent as _Agent
from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    Compactor,
    JsonMessage,
    Message,
    MessageBase,
    MessageContent,
    Model,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    MultipartDescriptor,
    MultipartMessage,
    TextDescriptor,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import tool_call_message
from sagent.testing import MockModelCaps
from sagent.tools.agent_spawn import (
    AgentSpawn as AgentTool,
    ChildStats,
    _make_parent_forwarder,
)
from sagent.tools.core import (
    CostLedger,
    cost_ledger_var,
    current_agent_var,
    max_depth_var,
    tool_state_context,
)
from sagent.tools.write import Write


class _ParentEvents:
    """Stub parent agent for ChildForwarder tests.

    Mirrors v2's ``Agent.inbox`` (the new spine) plus
    ``active_children`` (used by ``_ChildForwarder`` for live token
    accounting). The forwarder posts ``multipart/x-child-event`` here
    instead of v1's ``_events`` queue.
    """

    def __init__(self) -> None:

        self.inbox: Deque[Message] = Deque()
        self.active_children: dict[str, ChildStats] = {}


def Media(content: MessageContent, descriptor: str) -> Message:  # noqa: N802 -- PascalCase factory mimics Message constructor
    if isinstance(content, str):
        return TextMessage(content, cast(TextDescriptor, descriptor))
    if isinstance(content, tuple):
        return MultipartMessage(
            cast(tuple[Message, ...], content),  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright considers it redundant after isinstance
            cast(MultipartDescriptor, descriptor),
        )
    if isinstance(content, bytes):
        return BytesMessage(content, cast(BytesDescriptor, descriptor))
    return JsonMessage(content, descriptor)


def _msg(directive: JSON) -> Message:
    """Wrap a directive in a minimal tool-call Message."""
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


# -- Mock model --------------------------------------------------------


class _MockCaps(MockModelCaps):
    max_image_dim: int = 2000


class _MockModel(_MockCaps):
    """Returns a fixed sequence of canned responses."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        model_id: str = "mock",
    ) -> None:
        self._responses = responses or [_response("done")]
        self._call_idx = 0
        self._model_id = model_id
        self.requests: list[ModelRequest] = []

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return self._model_id

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_text, on_thinking
        return await self.buffer(request=request)


# -- Mock tool used to verify tool-inheritance semantics --------------


class _MarkerTool:
    """Minimal tool; used to check subset/inheritance routing."""

    name = "Marker"
    tool_id = "application/x-tool-marker"
    description = "Marker tool."
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )
    supports_microcompaction = False

    def __init__(self) -> None:
        self.calls = 0

    def summary(self, msg: Message) -> str:
        del msg
        return self.name

    def prompt(self) -> str:
        return ""

    async def run(self, msg: Message) -> Message:
        del msg
        self.calls += 1
        return TextMessage("marker-result", "text/plain")


# -- Helpers ----------------------------------------------------------


def _response(
    text: str = "done",
    *,
    tool_calls: list[Message] | None = None,
    stop_reason: str = "model_finished",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> ModelResponse:
    parts: list[Message] = []
    if text:
        parts.append(TextMessage(text, "text/plain"))
    parts.extend(tool_calls or [])
    return ModelResponse(
        content=MultipartMessage(
            tuple(parts),
            "multipart/x-model-message",
        ),
        stop_reason=stop_reason,
        tokens=TokenCount(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _parent(
    model: _MockModel | None = None,
    tools: list[Tool] | None = None,
    model_spec: ModelSpec | None = None,
) -> _Agent:
    """Build a parent Agent suitable for hosting an AgentSpawn call."""
    return _Agent(
        model=model or _MockModel(),
        model_spec=model_spec,
        system="parent system",
        tools=tools or [],
    )


def _default_spec(model_id: str = "parent") -> ModelSpec:
    """Canonical spec for tests that need one."""
    return ModelSpec(
        provider="Anthropic",
        auth="env",
        model_id=model_id,
        account=None,
    )


class _FakeProvider:
    """Stand-in for ``Provider``; returns a ``_MockModel``."""

    def __init__(self, factory: Callable[[str], _MockModel]) -> None:
        self._factory = factory

    def model(self, model_id: str | None = None, /) -> Model:
        return self._factory(model_id or "default")


# -- Tests -------------------------------------------------------------


class TestAgentFactoryBasics:
    def test_name_and_schema(self) -> None:
        t = AgentTool()
        assert t.name == "AgentSpawn"
        # ``directive_schema`` is typed as recursive JSON (str | int | dict | ...);
        # narrow to Mapping[str, object] for keyed access + ``in``.
        schema = cast("dict[str, object]", t.directive_schema)
        assert schema["required"] == ("prompt",)
        props = cast("dict[str, object]", schema["properties"])
        for key in (
            "system",
            "provider",
            "auth",
            "model_id",
            "account",
            "tools",
            "max_tool_call_rounds",
            "max_depth",
        ):
            assert key in props

    def test_describe_shows_prompt(self) -> None:
        t = AgentTool()
        label = t.summary(_msg(json_freeze({"prompt": "analyze the bug"})))
        assert label == "AgentSpawn analyze the bug"

    def test_describe_truncates_long_prompt(self) -> None:
        t = AgentTool()
        long_prompt = "x" * 200
        label = t.summary(_msg(json_freeze({"prompt": long_prompt})))
        assert label.startswith("AgentSpawn ")
        assert label.endswith("...")
        # Body after "AgentSpawn " is 60 chars (57 + "...").
        assert len(label) == len("AgentSpawn ") + 60

    def test_describe_collapses_newlines(self) -> None:
        t = AgentTool()
        label = t.summary(_msg(json_freeze({"prompt": "line one\nline two"})))
        assert label == "AgentSpawn line one line two"

    def test_describe_includes_model_id(self) -> None:
        t = AgentTool()
        label = t.summary(_msg(json_freeze({"prompt": "hi", "model_id": "opus"})))
        assert label == "AgentSpawn hi [opus]"

    def test_describe_bare_when_prompt_missing(self) -> None:
        t = AgentTool()
        label = t.summary(_msg(json_freeze({})))
        assert label == "AgentSpawn"


class TestHappyPath:
    @pytest.mark.anyio
    async def test_spawn_child_returns_text(self) -> None:
        # Child inherits parent.model by default; make parent's model
        # emit the text we want to observe back at the tool layer.
        parent = _parent(model=_MockModel([_response("child-output")]))
        tool = AgentTool()
        response = await _run_under_parent(parent, tool, prompt="hi")
        assert "child-output" in str(response.content)


class TestInheritance:
    @pytest.mark.anyio
    async def test_system_inherits_from_parent_by_default(self) -> None:
        parent = _parent()
        tool = AgentTool()
        child_system = tool._resolve_system(None, parent)
        assert child_system == "parent system"

    @pytest.mark.anyio
    async def test_system_llm_override_wins(self) -> None:
        tool = AgentTool()
        parent = _parent()
        assert tool._resolve_system("from llm", parent) == "from llm"

    @pytest.mark.anyio
    async def test_system_factory_beats_parent(self) -> None:
        tool = AgentTool(system="factory system")
        parent = _parent()
        assert tool._resolve_system(None, parent) == "factory system"

    def test_tools_inherit_parent_by_default(self) -> None:
        marker = _MarkerTool()
        parent = _parent(tools=[marker])
        tool = AgentTool()
        resolved = tool._resolve_tools(None, parent)
        assert resolved == [marker]

    def test_tools_explicit_empty_honored(self) -> None:
        marker = _MarkerTool()
        parent = _parent(tools=[marker])
        tool = AgentTool()
        resolved = tool._resolve_tools([], parent)
        assert resolved == []

    def test_tools_subset_by_name(self) -> None:
        class _Other:
            name = "Other"
            tool_id = "application/x-tool-other"
            description = "."
            directive_schema: JSON = json_freeze(
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            )
            supports_microcompaction = False

            def summary(self, msg: Message) -> str:
                del msg
                return self.name

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("", "text/plain")

        marker = _MarkerTool()
        other = _Other()
        parent = _parent(tools=[marker, other])
        tool = AgentTool()
        resolved = tool._resolve_tools(["Marker"], parent)
        assert isinstance(resolved, list)
        assert [t.name for t in resolved] == ["Marker"]

    def test_tools_unknown_name_returns_error(self) -> None:
        parent = _parent(tools=[_MarkerTool()])
        tool = AgentTool()
        result = tool._resolve_tools(["Nonexistent"], parent)
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Unknown tools" in str(result.content)


class TestModelResolution:
    """Per-field resolution: LLM arg → factory arg → parent.model_spec.<field>.

    When every resolved field matches the parent's spec the child
    reuses ``parent.model``; otherwise a fresh ``Model`` is built
    via ``providers.build_provider`` (monkey-patched in these tests).
    """

    def _resolve(
        self,
        tool: AgentTool,
        parent: _Agent,
        **overrides: str | None,
    ) -> tuple[Model, ModelSpec | None]:
        kwargs: dict[str, str | None] = {
            "provider": None,
            "auth": None,
            "model_id": None,
            "account": None,
        }
        kwargs.update(overrides)
        result = tool._resolve_model(parent_agent=parent, **kwargs)
        assert isinstance(result, tuple), f"Expected tuple, got {result!r}"
        return result

    def test_silent_llm_no_spec_inherits_parent_model(self) -> None:
        """No spec on parent, no overrides anywhere → reuse parent.model."""
        parent_model = _MockModel(model_id="parent")
        parent = _parent(model=parent_model)
        tool = AgentTool()
        model, spec = self._resolve(tool, parent)
        assert model is parent_model
        assert spec is None

    def test_silent_llm_with_spec_reuses_parent_model(self) -> None:
        """Spec matches on all fields → reuse parent.model, preserve spec."""
        parent_spec = _default_spec(model_id="sonnet")
        parent_model = _MockModel(model_id="sonnet")
        parent = _parent(model=parent_model, model_spec=parent_spec)
        tool = AgentTool()
        result = self._resolve(tool, parent)
        assert isinstance(result, tuple)
        model, spec = result
        assert model is parent_model
        assert spec is parent_spec

    def test_llm_override_without_parent_spec_errors(self) -> None:
        """LLM asked to switch models but parent has no spec to fill gaps."""
        parent = _parent()  # no spec
        tool = AgentTool()
        result = tool._resolve_model(
            parent_agent=parent,
            provider=None,
            auth=None,
            model_id="opus",
            account=None,
        )
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Cannot build a model" in str(result.content)

    def test_llm_full_triplet_builds_fresh_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM picks provider/auth/model_id → build_provider called,
        new model built, returned spec reflects the resolved fields.
        """
        built: dict[str, object] = {}

        def fake_build_provider(
            provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            built["provider"] = provider
            built["auth"] = auth
            built["account"] = account
            return _FakeProvider(lambda mid: _MockModel(model_id=mid))

        monkeypatch.setattr(
            "sagent.tools.agent_spawn.build_provider",
            fake_build_provider,
        )
        parent = _parent()
        tool = AgentTool()
        model, spec = self._resolve(
            tool,
            parent,
            provider="Google",
            auth="env",
            model_id="gemini-2.5-flash",
        )
        assert built == {"provider": "Google", "auth": "env", "account": None}
        assert model.model_id == "gemini-2.5-flash"
        assert spec == ModelSpec(
            provider="Google",
            auth="env",
            model_id="gemini-2.5-flash",
            account=None,
        )

    def test_partial_override_fills_from_parent_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM picks only model_id; provider/auth/account come from spec."""
        captured: dict[str, object] = {}

        def fake_build_provider(
            provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            captured["provider"] = provider
            captured["auth"] = auth
            captured["account"] = account
            return _FakeProvider(lambda mid: _MockModel(model_id=mid))

        monkeypatch.setattr(
            "sagent.tools.agent_spawn.build_provider",
            fake_build_provider,
        )
        parent_spec = _default_spec(model_id="sonnet")
        parent = _parent(model=_MockModel(model_id="sonnet"), model_spec=parent_spec)
        tool = AgentTool()
        model, spec = self._resolve(tool, parent, model_id="opus")
        assert captured == {
            "provider": "Anthropic",
            "auth": "env",
            "account": None,
        }
        assert model.model_id == "opus"
        assert spec is not None
        assert spec.model_id == "opus"
        assert spec.provider == "Anthropic"

    def test_llm_arg_beats_factory_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory-pinned model_id is overridden when LLM supplies one."""
        seen_ids: list[str] = []

        def fake_build_provider(
            provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            del provider, auth, account

            def factory(mid: str) -> _MockModel:
                seen_ids.append(mid)
                return _MockModel(model_id=mid)

            return _FakeProvider(factory)

        monkeypatch.setattr(
            "sagent.tools.agent_spawn.build_provider",
            fake_build_provider,
        )
        parent_spec = _default_spec(model_id="sonnet")
        parent = _parent(model=_MockModel(model_id="sonnet"), model_spec=parent_spec)
        tool = AgentTool(model_id="haiku")
        # LLM asks for opus; factory asks for haiku; LLM wins.
        model, _spec = self._resolve(tool, parent, model_id="opus")
        assert seen_ids == ["opus"]
        assert model.model_id == "opus"

    def test_no_build_when_spec_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spec inheritance with zero overrides must not hit build_provider."""

        def boom(*args: object, **kwargs: object) -> None:
            del args, kwargs
            pytest.fail("build_provider should not be called")

        monkeypatch.setattr("sagent.tools.agent_spawn.build_provider", boom)
        parent_spec = _default_spec(model_id="sonnet")
        parent_model = _MockModel(model_id="sonnet")
        parent = _parent(model=parent_model, model_spec=parent_spec)
        tool = AgentTool()
        model, spec = self._resolve(tool, parent)
        assert model is parent_model
        assert spec is parent_spec

    def test_factory_only_model_id_without_parent_spec_errors(self) -> None:
        """Factory asks for a model switch but no provider/auth available."""
        parent = _parent()  # no spec
        tool = AgentTool(model_id="opus")
        result = tool._resolve_model(
            parent_agent=parent,
            provider=None,
            auth=None,
            model_id=None,
            account=None,
        )
        assert isinstance(result, MessageBase)
        assert result.descriptor == "text/x-error"
        assert "Cannot build a model" in str(result.content)

    def test_llm_can_switch_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Account is just another field; LLM can override it too."""
        captured: dict[str, object] = {}

        def fake_build_provider(
            provider: str, auth: str, *, account: str | None = None
        ) -> _FakeProvider:
            captured["provider"] = provider
            captured["auth"] = auth
            captured["account"] = account
            return _FakeProvider(lambda mid: _MockModel(model_id=mid))

        monkeypatch.setattr(
            "sagent.tools.agent_spawn.build_provider",
            fake_build_provider,
        )
        parent_spec = ModelSpec(
            provider="Anthropic",
            auth="env",
            model_id="sonnet",
            account="personal",
        )
        parent = _parent(model=_MockModel(model_id="sonnet"), model_spec=parent_spec)
        tool = AgentTool()
        _model, spec = self._resolve(tool, parent, account="work")
        assert captured["account"] == "work"
        assert spec is not None
        assert spec.account == "work"


class TestMaxDepth:
    @pytest.mark.anyio
    async def test_max_depth_zero_allows_leaf_spawn(self) -> None:
        parent = _parent(model=_MockModel([_response("leaf-output")]))
        tool = AgentTool(max_depth=0)
        result = await _run_under_parent(parent, tool, prompt="x")
        assert "leaf-output" in str(result.content)

    @pytest.mark.anyio
    async def test_max_depth_zero_blocks_grandchild_spawn(self) -> None:
        model = await self._run_child_that_spawns_grandchild(AgentTool(max_depth=0))
        self._assert_grandchild_blocked(model)

    @pytest.mark.anyio
    async def test_max_depth_one_blocks_great_grandchild_spawn(self) -> None:
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "grandchild"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a2",
                            "AgentSpawn",
                            json_freeze({"prompt": "great-grandchild"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("grandchild-done"),
                _response("child-done"),
            ]
        )
        factory = AgentTool(max_depth=1)
        parent = _parent(model=model, tools=[factory])
        result = await _run_under_parent(parent, factory, prompt="child")
        assert "child-done" in str(result.content)
        assert any(
            "max_depth 1 exceeded" in str(request.messages)
            for request in model.requests
        )

    @pytest.mark.anyio
    async def test_max_depth_propagates_via_context_var(self) -> None:
        token = max_depth_var.set(0)
        try:
            model = await self._run_child_that_spawns_grandchild(AgentTool())
            self._assert_grandchild_blocked(model)
        finally:
            max_depth_var.reset(token)

    @pytest.mark.anyio
    async def test_max_depth_tightens_not_raises(self) -> None:
        token = max_depth_var.set(0)
        try:
            model = await self._run_child_that_spawns_grandchild(
                AgentTool(max_depth=99)
            )
            self._assert_grandchild_blocked(model)
        finally:
            max_depth_var.reset(token)

    async def _run_child_that_spawns_grandchild(self, factory: AgentTool) -> _MockModel:
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "grandchild"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("child-done"),
            ]
        )
        parent = _parent(model=model, tools=[factory])
        result = await _run_under_parent(parent, factory, prompt="child")
        assert "child-done" in str(result.content)
        return model

    def _assert_grandchild_blocked(self, model: _MockModel) -> None:
        assert any(
            "max_depth 0 exceeded" in str(request.messages)
            for request in model.requests
        )


class _StubCompactor:
    """Minimal ``Compactor`` stub; methods never run in tests."""

    def maintain(
        self, messages: list[Any], tools: dict[str, Any], **kwargs: object
    ) -> None:
        del messages, tools, kwargs

    async def should_compact(
        self,
        input_tokens: int,
        max_request_tokens: int,
        max_response_tokens: int = 0,
    ) -> bool:
        del input_tokens, max_request_tokens, max_response_tokens
        return False

    async def compact(
        self,
        messages: list[Any],
        model: Any,
        transcript_path: Path | None = None,
        direction: str = "from",
        keep_recent: int | None = None,
        custom_instructions: str | None = None,
        summary_pointers: list[tuple[str, str]] | None = None,
    ) -> list[Any]:
        del (
            messages,
            model,
            transcript_path,
            direction,
            keep_recent,
            custom_instructions,
            summary_pointers,
        )
        return []


class TestCompactorInheritance:
    def test_compactor_inherits_from_parent(self) -> None:
        parent_comp = cast(Compactor, _StubCompactor())
        parent = _parent()
        parent.compactor = parent_comp
        tool = AgentTool()  # no factory compactor
        # _inherit reads parent.compactor (property) when factory is None.
        assert tool._inherit("compactor", parent) is parent_comp

    def test_compactor_factory_wins_over_parent(self) -> None:
        parent_comp = cast(Compactor, _StubCompactor())
        factory_comp = cast(Compactor, _StubCompactor())
        parent = _parent()
        parent.compactor = parent_comp
        tool = AgentTool(compactor=factory_comp)
        assert tool._inherit("compactor", parent) is factory_comp


class TestSessionDir:
    def test_none_is_ephemeral(self) -> None:
        tool = AgentTool()
        assert tool._child_session_dir(None) is None

    def test_configured_root_produces_subdir(self, tmp_path: Path) -> None:
        tool = AgentTool(session_root_dir=tmp_path)
        parent = _parent()
        sub = tool._child_session_dir(parent)
        assert sub is not None
        assert sub.parent.name == parent.session_id
        assert sub.parent.parent == tmp_path


class TestCostLedgerAggregation:
    @pytest.mark.anyio
    async def test_parent_ledger_captures_child_tokens(self) -> None:
        parent = _parent(
            model=_MockModel(
                [
                    dataclasses.replace(
                        _response("c"),
                        tokens=TokenCount(input_tokens=7, output_tokens=3),
                    )
                ],
                model_id="child-model",
            )
        )
        tool = AgentTool()
        await _run_under_parent(parent, tool, prompt="go")
        ledger = parent.cost_ledger
        # Parent ledger exists only during __call__ - ``_run_under_parent``
        # captures it mid-call via a stashed reference.
        assert ledger is None


async def _run_under_parent(
    parent: _Agent,
    tool: AgentTool,
    *,
    prompt: str,
) -> Message:
    """Simulate ``tool.run(prompt=...)`` being invoked mid-request.

    The factory looks up the calling agent via ``current_agent_var``
    and the current depth via ``get_tool_state().depth``. Real
    invocation happens inside ``Agent.run`` (which installs both);
    here we mimic just enough of that wiring to call the factory
    directly without a full model round-trip.
    """
    # Mirror ``_Agent.run``'s entry: install parent as current agent,
    # install a fresh cost ledger for this scope.
    parent.tool_state.depth = 0
    agent_tok = current_agent_var.set(parent)
    ledger_tok = cost_ledger_var.set(CostLedger())
    try:
        with tool_state_context(parent.tool_state):
            return await tool.run(_msg(json_freeze({"prompt": prompt})))
    finally:
        current_agent_var.reset(agent_tok)
        cost_ledger_var.reset(ledger_tok)


class TestChildStats:
    def test_streamed_text_updates_child_live_tokens(self) -> None:
        parent = _ParentEvents()
        forwarder = _make_parent_forwarder("Agent_0", 1, cast(Any, parent))
        assert forwarder is not None

        forwarder(TextMessage("abcdefgh", "text/plain"))

        assert parent.active_children["Agent_0"].model_response_tokens == 2

    def test_active_children_are_scoped_by_parent(self) -> None:
        parent1 = _ParentEvents()
        parent2 = _ParentEvents()
        forwarder1 = _make_parent_forwarder("Agent_0", 1, cast(Any, parent1))
        forwarder2 = _make_parent_forwarder("Agent_0", 1, cast(Any, parent2))
        assert forwarder1 is not None
        assert forwarder2 is not None

        forwarder1(TextMessage("abcdefgh", "text/plain"))
        forwarder2(TextMessage("abcd", "text/plain"))

        children1 = parent1.active_children
        children2 = parent2.active_children
        assert children1["Agent_0"].model_response_tokens == 2
        assert children2["Agent_0"].model_response_tokens == 1


# -- End-to-end: parent Agent dispatches the factory tool -------------


class TestEndToEndDispatch:
    """Exercise the real agent loop: a parent Agent with the factory
    tool in its toolset dispatches it via the normal tool-dispatch
    path (no helper shortcut). Validates that depth, cost-ledger
    inheritance, and parent-context discovery all wire up correctly.
    """

    @pytest.mark.anyio
    async def test_parent_dispatches_factory_gets_child_text(self) -> None:
        # Child inherits parent.model. Give parent responses that
        # serve both its own model requests and the child's single request.
        factory = AgentTool()
        parent_model = _MockModel(
            [
                _response(
                    "dispatching",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "do it"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("final-from-parent", input_tokens=20, output_tokens=10),
            ],
            model_id="parent",
        )
        parent = _Agent(model=parent_model, system="p", tools=[factory])

        response = await parent.run(json_freeze({"prompt": "start"}))
        assert "final-from-parent" in str(response.content)
        assert parent.last_run_tokens.input_tokens == 10 + 20 + 20
        assert parent.last_run_tokens.output_tokens == 5 + 10 + 10

    @pytest.mark.anyio
    async def test_cost_ledger_aggregates_parent_and_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The root ledger picks up both parent and child request costs."""

        # Factory pins model_id="child" over the parent's spec so the
        # child gets its own distinct model instance (built by our fake
        # build_provider), enabling the per-model breakdown assertion.
        def _fake_bp(*_a: object, **_k: object) -> _FakeProvider:
            return _FakeProvider(
                lambda mid: _MockModel(
                    [
                        dataclasses.replace(
                            _response("c"),
                            tokens=TokenCount(input_tokens=7, output_tokens=3),
                        )
                    ],
                    model_id=mid,
                )
            )

        monkeypatch.setattr("sagent.tools.agent_spawn.build_provider", _fake_bp)
        factory = AgentTool(model_id="child")
        parent_model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=11,
                    output_tokens=1,
                ),
                _response("done", input_tokens=13, output_tokens=2),
            ],
            model_id="parent",
        )
        # Capture the ledger mid-call: stash a reference from inside
        # the mock's request handler (runs while the parent holds the
        # ContextVar).
        captured: dict[str, CostLedger] = {}

        orig_buffer = parent_model.buffer

        async def capturing_buffer(request: ModelRequest) -> ModelResponse:
            ledger = cost_ledger_var.get()
            if ledger is not None:
                captured["ledger"] = ledger
            return await orig_buffer(request)

        async def capturing_stream(
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            del on_text, on_thinking
            return await capturing_buffer(request)

        parent_model.buffer = capturing_buffer  # ty: ignore[invalid-assignment] -- test mock
        parent_model.stream = capturing_stream  # ty: ignore[invalid-assignment] -- test mock

        parent = _Agent(
            model=parent_model,
            model_spec=_default_spec(model_id="parent"),
            system="p",
            tools=[factory],
        )
        await parent.run(json_freeze({"prompt": "go"}))
        ledger = captured["ledger"]
        # Parent: 2 calls (11+13 in, 1+2 out). Child: 1 call (7 in, 3 out).
        assert ledger.tokens.input_tokens == 11 + 13 + 7
        assert ledger.tokens.output_tokens == 1 + 2 + 3
        # Both models visible in the per-model breakdown.
        assert ledger.calls_by_model == {"parent": 2, "child": 1}
        # After run completion, root's session-cumulative tracker should
        # reflect the full subtree, not just parent-only spend.
        assert parent.total_tokens.input_tokens == 11 + 13 + 7
        assert parent.total_tokens.output_tokens == 1 + 2 + 3
        assert parent.total_cost_usd == pytest.approx(ledger.total_cost_usd)

    @pytest.mark.anyio
    async def test_cumulative_subtree_across_two_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the fold-back math must compose across multiple
        completed runs. Run 1's subtree + Run 2's subtree should equal
        ``parent.total_tokens`` after both finish.
        """

        def _fake_bp(*_a: object, **_k: object) -> _FakeProvider:
            return _FakeProvider(
                lambda mid: _MockModel(
                    [
                        dataclasses.replace(
                            _response("c"),
                            tokens=TokenCount(input_tokens=7, output_tokens=3),
                        )
                    ],
                    model_id=mid,
                )
            )

        monkeypatch.setattr("sagent.tools.agent_spawn.build_provider", _fake_bp)
        factory = AgentTool(model_id="child")
        parent_model = _MockModel(
            [
                # Run 1: parent spawns child, then finishes.
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=11,
                    output_tokens=1,
                ),
                _response("done", input_tokens=13, output_tokens=2),
                # Run 2: parent spawns child, then finishes.
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a2",
                            "AgentSpawn",
                            json_freeze({"prompt": "again"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                    input_tokens=4,
                    output_tokens=2,
                ),
                _response("done2", input_tokens=6, output_tokens=4),
            ],
            model_id="parent",
        )
        parent = _Agent(
            model=parent_model,
            model_spec=_default_spec(model_id="parent"),
            system="p",
            tools=[factory],
        )
        await parent.run(json_freeze({"prompt": "go"}))
        await parent.run(json_freeze({"prompt": "again"}))
        # Run 1 subtree: parent (11+13 / 1+2) + child (7/3).
        # Run 2 subtree: parent (4+6 / 2+4) + child (7/3).
        assert parent.total_tokens.input_tokens == (11 + 13 + 7) + (4 + 6 + 7)
        assert parent.total_tokens.output_tokens == (1 + 2 + 3) + (2 + 4 + 3)

    @pytest.mark.anyio
    async def test_parallel_siblings_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three concurrent AgentSpawn calls from one parent request all return.

        AgentSpawn is in ``_READ_ONLY_TOOLS``, so the dispatcher batches
        sibling calls into ``asyncio.gather``. Each sibling builds its own
        child Agent; they don't share ToolState.
        """

        def _fake_bp(*_a: object, **_k: object) -> _FakeProvider:
            return _FakeProvider(
                lambda mid: _MockModel(
                    [
                        dataclasses.replace(
                            _response("child-out"),
                            tokens=TokenCount(input_tokens=5, output_tokens=2),
                        )
                    ],
                    model_id=mid,
                )
            )

        monkeypatch.setattr("sagent.tools.agent_spawn.build_provider", _fake_bp)
        factory = AgentTool(model_id="child")
        parent_model = _MockModel(
            [
                _response(
                    "fanning out",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "task 1"}),
                        ),
                        tool_call_message(
                            "a2",
                            "AgentSpawn",
                            json_freeze({"prompt": "task 2"}),
                        ),
                        tool_call_message(
                            "a3",
                            "AgentSpawn",
                            json_freeze({"prompt": "task 3"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                dataclasses.replace(
                    _response("joined"),
                    tokens=TokenCount(input_tokens=20, output_tokens=10),
                ),
            ],
            model_id="parent",
        )
        parent = _Agent(
            model=parent_model,
            model_spec=_default_spec(model_id="parent"),
            system="p",
            tools=[factory],
        )
        response = await parent.run(json_freeze({"prompt": "fan-out"}))
        assert "joined" in str(response.content)


class TestStreaming:
    """``on_message`` callback delivers the child's event stream."""

    @pytest.mark.anyio
    async def test_on_message_receives_done_event(self) -> None:
        """Child emits a DoneEvent at close; the factory pumps it."""
        events: list[Message] = []
        factory = AgentTool(on_message=events.append)
        parent = _parent(
            model=_MockModel(
                [
                    dataclasses.replace(
                        _response("c"),
                        tokens=TokenCount(input_tokens=4, output_tokens=2),
                    )
                ],
                model_id="child",
            )
        )
        await _run_under_parent(parent, factory, prompt="go")
        assert any(e.descriptor == "application/x-done" for e in events)

    @pytest.mark.anyio
    async def test_buffered_mode_no_pump(self) -> None:
        """Without ``on_message`` or parent queue, the factory runs buffered."""
        factory = AgentTool()
        parent = _parent(
            model=_MockModel(
                [
                    dataclasses.replace(
                        _response("c"),
                        tokens=TokenCount(input_tokens=4, output_tokens=2),
                    )
                ],
            )
        )
        response = await _run_under_parent(parent, factory, prompt="go")
        assert response is not None

    @pytest.mark.anyio
    async def test_parent_run_yields_child_tool_events(self) -> None:
        """Child tool events surface as ``multipart/x-child-event`` in parent generator."""
        factory = AgentTool()
        marker = _MarkerTool()
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response(
                    "",
                    tool_calls=[
                        tool_call_message("t1", "Marker", json_freeze({})),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("child-done"),
                _response("parent-done"),
            ]
        )
        parent = _Agent(model=model, system="p", tools=[factory, marker])
        forwarded = [
            msg
            async for msg in parent.run(json_freeze({"prompt": "go"}))
            if msg.descriptor == "multipart/x-child-event"
        ]
        assert forwarded
        assert all(e.descriptor == "multipart/x-child-event" for e in forwarded)
        inner_descriptors = [
            e.content[1].descriptor
            for e in forwarded
            if isinstance(e, MultipartMessage) and len(e.content) >= 2
        ]
        assert "text/x-tool-label" in inner_descriptors
        assert "multipart/x-tool-result" in inner_descriptors

    @pytest.mark.anyio
    async def test_child_tool_error_forwards_as_error(self) -> None:
        """Child validation error surfaces in parent generator with text/x-error."""
        factory = AgentTool()
        write_tool = Write()
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "w1",
                            "Write",
                            json_freeze({"file_path": "/nowhere/x"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("recovered"),
                _response("parent-done"),
            ]
        )
        parent = _Agent(model=model, system="p", tools=[factory, write_tool])
        forwarded = [
            msg
            async for msg in parent.run(json_freeze({"prompt": "go"}))
            if msg.descriptor == "multipart/x-child-event"
        ]
        results: list[Message] = []
        for e in forwarded:
            if not isinstance(e, MultipartMessage) or len(e.content) < 2:
                continue
            inner = e.content[1]
            if inner.descriptor == "multipart/x-tool-result":
                results.append(inner)
        assert results
        assert isinstance(results[0], MultipartMessage)
        parts = results[0].content
        error_parts = [p for p in parts if p.descriptor == "text/x-error"]
        assert error_parts
        assert "content" in str(error_parts[0].content).lower()

    @pytest.mark.anyio
    async def test_child_events_labeled_with_agent_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parallel children get distinct Agent_N labels from the per-run counter."""

        def _fake_bp(*_a: object, **_k: object) -> _FakeProvider:
            return _FakeProvider(
                lambda mid: _MockModel(
                    [
                        _response(
                            "",
                            tool_calls=[
                                tool_call_message(
                                    "t1",
                                    "Marker",
                                    json_freeze({}),
                                ),
                            ],
                            stop_reason="model_tool_use",
                        ),
                        _response("child-done"),
                    ],
                    model_id=mid,
                )
            )

        monkeypatch.setattr(
            "sagent.tools.agent_spawn.build_provider",
            _fake_bp,
        )
        factory = AgentTool(model_id="child")
        marker = _MarkerTool()
        parent_model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "first"}),
                        ),
                        tool_call_message(
                            "a2",
                            "AgentSpawn",
                            json_freeze({"prompt": "second"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("parent-done"),
            ],
            model_id="parent",
        )
        parent = _Agent(
            model=parent_model,
            model_spec=_default_spec(model_id="parent"),
            system="p",
            tools=[factory, marker],
        )
        labels: set[str] = set()
        async for msg in parent.run(json_freeze({"prompt": "go"})):
            if (
                isinstance(msg, MultipartMessage)
                and msg.descriptor == "multipart/x-child-event"
                and len(msg.content) >= 2
                and msg.content[0].descriptor == "text/x-agent-label"
            ):
                labels.add(str(msg.content[0].content))
        assert "Agent_0" in labels
        assert "Agent_1" in labels

    @pytest.mark.anyio
    async def test_verbosity_0_only_forwards_errors(self) -> None:
        """verbosity=0 suppresses tool calls and text; errors still forward."""
        factory = AgentTool(verbosity=0)
        marker = _MarkerTool()
        write_tool = Write()
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response(
                    "",
                    tool_calls=[
                        tool_call_message("t1", "Marker", json_freeze({})),
                        tool_call_message(
                            "w1",
                            "Write",
                            json_freeze({"file_path": "/nowhere/x"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("recovered"),
                _response("parent-done"),
            ]
        )
        parent = _Agent(
            model=model,
            system="p",
            tools=[factory, marker, write_tool],
        )
        forwarded = [
            msg
            async for msg in parent.run(json_freeze({"prompt": "go"}))
            if msg.descriptor == "multipart/x-child-event"
        ]
        assert all(e.descriptor == "multipart/x-child-event" for e in forwarded)
        inner_descriptors = [
            e.content[1].descriptor
            for e in forwarded
            if isinstance(e, MultipartMessage) and len(e.content) >= 2
        ]
        assert "multipart/x-tool-call" not in inner_descriptors
        assert "text/plain" not in inner_descriptors
        assert "multipart/x-tool-result" in inner_descriptors
        assert "application/x-child-done" in inner_descriptors

    @pytest.mark.anyio
    async def test_child_done_event_forwarded(self) -> None:
        """Child-done event carries elapsed, model_response_tokens, cost_usd."""
        factory = AgentTool()
        model = _MockModel(
            [
                _response(
                    "",
                    tool_calls=[
                        tool_call_message(
                            "a1",
                            "AgentSpawn",
                            json_freeze({"prompt": "go"}),
                        ),
                    ],
                    stop_reason="model_tool_use",
                ),
                _response("child-out", input_tokens=10, output_tokens=5),
                _response("parent-done"),
            ]
        )
        parent = _Agent(model=model, system="p", tools=[factory])
        forwarded = [
            msg
            async for msg in parent.run(json_freeze({"prompt": "go"}))
            if msg.descriptor == "multipart/x-child-event"
        ]
        done_events = [
            e
            for e in forwarded
            if isinstance(e, MultipartMessage)
            and len(e.content) >= 2
            and e.content[1].descriptor == "application/x-child-done"
        ]
        assert len(done_events) == 1
        assert isinstance(done_events[0], MultipartMessage)
        wrapper = done_events[0].content
        done_data = cast("dict[str, object]", wrapper[1].content)
        assert "elapsed" in done_data
        assert done_data["model_response_tokens"] == 5
        assert done_data["cost_usd"] == 0.0

    @pytest.mark.anyio
    async def test_on_message_exception_does_not_cancel_child(self) -> None:
        """A broken ``on_message`` must not cancel the child."""

        def broken(event: object) -> None:
            del event
            raise RuntimeError("callback bug")

        factory = AgentTool(on_message=broken)
        parent = _parent(
            model=_MockModel(
                [
                    dataclasses.replace(
                        _response("from-child"),
                        tokens=TokenCount(input_tokens=4, output_tokens=2),
                    )
                ],
                model_id="child",
            )
        )
        response = await _run_under_parent(parent, factory, prompt="go")
        assert "from-child" in str(response.content)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
