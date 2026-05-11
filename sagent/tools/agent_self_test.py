"""Tests for AgentSelf tool."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from sagent.agent import PendingOp
from sagent.custom_types import (
    ContextBudget,
    JsonMessage,
    Message,
    ModelSpec,
    MultipartMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.agent_self import AgentSelf
from sagent.tools.core import (
    ToolState,
    current_agent_var,
    tool_state_context,
)


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-agentself"),),
        "multipart/x-tool-call",
    )


@pytest.fixture
def tool_state() -> Iterator[ToolState]:
    ts = ToolState()
    with tool_state_context(ts):
        yield ts


@pytest.mark.usefixtures("tool_state")
class TestPatch:
    @pytest.mark.anyio
    async def test_empty_patch_is_noop(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg(json_freeze({})))
            assert result.descriptor == "text/plain"
            assert result.content == "No changes."
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_sets_status_and_limits_together(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(
                    json_freeze(
                        {
                            "status": "Debugging tests",
                            "max_request_tokens": 180_000,
                            "max_response_tokens": 7_000,
                        }
                    )
                )
            )
            assert result.descriptor == "text/plain"
            assert agent.status == "Debugging tests"
            assert agent.max_request_tokens == 180_000
            assert agent.max_response_tokens == 7_000
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_empty_status_is_error_when_provided(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg(json_freeze({"status": ""})))
            assert result.descriptor == "text/x-error"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_context_prompt_requires_context(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(json_freeze({"context_prompt": "keep details"}))
            )
            assert result.descriptor == "text/x-error"
            assert "context_prompt" in str(result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_queues_context_actions(self, tool_state: ToolState) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(
                    json_freeze(
                        {"context": "compact", "context_prompt": "keep the API spec"}
                    )
                )
            )
            assert result.descriptor == "text/plain"
            del tool_state  # not used in v3 -- ops live on agent._next_op
            assert agent._next_op is not None
            assert agent._next_op.kind == "compact"
            assert agent._next_op.args == "keep the API spec"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_diagnostics_reports_model_options(
        self, tool_state: ToolState
    ) -> None:
        tool_state.stats = {"max_request_tokens": 0, "input_tokens": 100}
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg(json_freeze({"diagnostics": True})))
            text = str(result.content)
            assert "0.0%" in text
            assert "Thinking:           on" in text
            assert "Effort:             unset" in text
            assert (
                "Supported model_options: thinking: boolean, effort: string,"
                " cache_ttl: '5m' | '1h'"
            ) in text
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_diagnostics_lists_provider_catalog(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(json_freeze({"diagnostics": True, "catalog": "providers"}))
            )
            text = str(result.content)
            assert result.descriptor == "text/plain"
            assert "Known providers:" in text
            assert "Anthropic" in text
            assert "OpenAI" in text
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_diagnostics_lists_model_catalog(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(
                    json_freeze(
                        {
                            "diagnostics": True,
                            "catalog": "models",
                            "catalog_provider": "Anthropic",
                        }
                    )
                )
            )
            text = str(result.content)
            assert result.descriptor == "text/plain"
            assert "Provider catalog: Anthropic" in text
            assert "Default model:" in text
            assert "claude" in text
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_options_reject_unsupported_keys(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(json_freeze({"model_options": {"temperature": 0.2}}))
            )
            assert result.descriptor == "text/x-error"
            assert "temperature" in str(result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_options_update_thinking_and_effort(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(
                    json_freeze(
                        {"model_options": {"thinking": False, "effort": "high"}}
                    )
                )
            )
            assert result.descriptor == "text/plain"
            assert agent.thinking is None
            assert agent.effort == "high"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_rejects_model_options_when_unsupported(self) -> None:
        agent = _LimitAgent(supports_thinking=False, supports_effort=False)
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(json_freeze({"model_options": {"thinking": True}}))
            )
            assert result.descriptor == "text/x-error"
            assert "thinking" in str(result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_rejects_zero_max_request_tokens(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg(json_freeze({"max_request_tokens": 0})))
            assert result.descriptor == "text/x-error"
            assert agent.max_request_tokens == 200_000
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_invalid_patch_does_not_apply_earlier_fields(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(
                    json_freeze(
                        {
                            "status": "Mutated",
                            "model_options": {"cache_ttl": "1h"},
                            "max_request_tokens": 1,
                            "context": "compact",
                        }
                    )
                )
            )
            assert result.descriptor == "text/x-error"
            assert agent.status == ""
            assert agent.cache_ttl == "5m"
            assert agent.max_request_tokens == 200_000
            assert agent._next_op is None
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_options_sets_cache_ttl(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg(json_freeze({"model_options": {"cache_ttl": "1h"}}))
            )
            assert result.descriptor == "text/plain"
            assert agent.cache_ttl == "1h"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_selfhosted_local_path_updates_auth(self) -> None:
        agent = _LimitAgent()
        agent.model_spec = ModelSpec(
            provider="SelfHosted",
            auth="/old/model",
            model_id="/old/model",
        )
        agent.model.model_id = "/old/model"
        token = current_agent_var.set(cast(Any, agent))
        try:
            with patch("sagent.tools.agent_self.build_provider") as mock_bp:
                mock_bp.return_value.model.return_value = MagicMock(
                    model_id="/new/model",
                    max_request_tokens=200_000,
                    max_response_tokens=8_000,
                    supports_thinking=True,
                    supports_effort=True,
                )
                result = await AgentSelf().run(
                    _msg(json_freeze({"model_id": "/new/model"}))
                )
            assert result.descriptor == "text/plain"
            mock_bp.assert_called_once_with("SelfHosted", "/new/model", account=None)
            assert agent.swapped_spec is not None
            spec = agent.swapped_spec
            assert spec.provider == "SelfHosted"
            assert spec.auth == "/new/model"
            assert spec.model_id == "/new/model"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_swap_resets_budget_on_smaller_model(self) -> None:
        agent = _LimitAgent()  # budget: max_request=200K, max_response=8K
        token = current_agent_var.set(cast(Any, agent))
        try:
            with patch("sagent.tools.agent_self.build_provider") as mock_bp:
                mock_bp.return_value.model.return_value = _ModelStub(
                    model_id="small-model",
                    max_request_tokens=100_000,
                    max_response_tokens=8_000,
                )
                result = await AgentSelf().run(
                    _msg(json_freeze({"model_id": "small-model"}))
                )
            assert result.descriptor == "text/plain", result.content
            assert agent.swapped_spec is not None
            assert agent.model.model_id == "small-model"
            assert agent._budget is None
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_swap_clears_effort_when_unsupported(self) -> None:
        agent = _LimitAgent()
        agent._effort = "medium"
        token = current_agent_var.set(cast(Any, agent))
        try:
            with patch("sagent.tools.agent_self.build_provider") as mock_bp:
                mock_bp.return_value.model.return_value = _ModelStub(
                    model_id="no-effort-model",
                    supports_effort=False,
                )
                result = await AgentSelf().run(
                    _msg(json_freeze({"model_id": "no-effort-model"}))
                )
            assert result.descriptor == "text/plain", result.content
            assert agent.effort is None
            assert "effort=unset" in str(result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_model_swap_clears_thinking_when_unsupported(self) -> None:
        agent = _LimitAgent()  # thinking is "adaptive" by default
        token = current_agent_var.set(cast(Any, agent))
        try:
            with patch("sagent.tools.agent_self.build_provider") as mock_bp:
                mock_bp.return_value.model.return_value = _ModelStub(
                    model_id="no-thinking-model",
                    supports_thinking=False,
                )
                result = await AgentSelf().run(
                    _msg(json_freeze({"model_id": "no-thinking-model"}))
                )
            assert result.descriptor == "text/plain", result.content
            assert agent.thinking is None
            assert "thinking=off" in str(result.content)
        finally:
            current_agent_var.reset(token)


class TestPrompt:
    def test_static_guidance_lives_in_description(self) -> None:
        assert AgentSelf().prompt() == ""
        desc = AgentSelf.description
        assert "model_options" in desc
        assert "context" in desc
        assert "max_request_tokens" in desc


class TestHelp:
    def test_summary(self) -> None:
        h = AgentSelf().summary(
            _msg(
                json_freeze(
                    {
                        "status": "Debugging",
                        "context": "compact",
                        "max_request_tokens": 180_000,
                    }
                )
            )
        )
        assert "status=Debugging" in h
        assert "context=compact" in h
        assert "max_request_tokens=180000" in h


class TestSchema:
    def test_patch_schema_has_no_required_operation(self) -> None:
        schema = cast(dict[str, Any], AgentSelf.directive_schema)
        props = cast(dict[str, dict[str, Any]], schema["properties"])
        assert schema["additionalProperties"] is False
        assert schema["required"] == ()
        assert "operation" not in props
        assert tuple(props["context"]["enum"]) == ("clear", "compact", "recompact")
        assert "cache_ttl" not in props
        assert props["max_request_tokens"]["minimum"] == 1
        assert props["model_options"]["type"] == "object"
        assert tuple(props["catalog"]["enum"]) == ("providers", "models")
        assert props["catalog_provider"]["type"] == "string"

    def test_patch_schema_preserves_contract_guidance(self) -> None:
        schema = cast(dict[str, Any], AgentSelf.directive_schema)
        props = cast(dict[str, dict[str, Any]], schema["properties"])
        context_prompt = str(props["context_prompt"]["description"])
        model_options = str(props["model_options"]["description"])
        diagnostics = str(props["diagnostics"]["description"])
        assert "Only valid when context is set" in context_prompt
        assert "Supported keys" in model_options
        assert "diagnostics" in model_options
        assert "current diagnostics" in diagnostics
        assert "catalog" in props
        assert "catalog_provider" in props


class _ModelStub:
    def __init__(
        self,
        *,
        model_id: str = "model",
        max_request_tokens: int = 200_000,
        max_response_tokens: int = 8_000,
        supports_thinking: bool = True,
        supports_effort: bool = True,
        supports_cache_control: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_request_tokens = max_request_tokens
        self.max_response_tokens = max_response_tokens
        self.supports_thinking = supports_thinking
        self.supports_effort = supports_effort
        self.supports_cache_control = supports_cache_control


class _LimitAgent:
    def __init__(
        self,
        *,
        supports_thinking: bool = True,
        supports_effort: bool = True,
        supports_cache_control: bool = True,
    ) -> None:
        self._max_request_tokens = 200_000
        self._max_response_tokens = 8_000
        self._cache_ttl = "5m"
        self._thinking = "adaptive"
        self._effort: str | None = None
        self._status = ""
        self._budget: ContextBudget | None = None
        self.model = _ModelStub(
            supports_thinking=supports_thinking,
            supports_effort=supports_effort,
            supports_cache_control=supports_cache_control,
        )
        self.model_spec = ModelSpec(provider="Anthropic", auth="env", model_id="model")
        self.swapped_spec: ModelSpec | None = None
        self._next_op: PendingOp | None = None
        self.session_id: str = "test1234"
        self.session_dir: Path | None = None
        self.tool_state = ToolState()

    @property
    def budget(self) -> ContextBudget:
        if self._budget is not None:
            return self._budget
        return ContextBudget(
            max_request_tokens=self.max_request_tokens,
            max_response_tokens=self.max_response_tokens,
            buffer_tokens=128_000,
        )

    @property
    def max_request_tokens(self) -> int:
        return self._max_request_tokens

    @max_request_tokens.setter
    def max_request_tokens(self, value: int) -> None:
        if value < 128_000:
            raise ValueError(
                f"buffer_tokens (128000) must be < max_request_tokens ({value})"
            )
        self._max_request_tokens = value

    @property
    def max_response_tokens(self) -> int:
        return self._max_response_tokens

    @max_response_tokens.setter
    def max_response_tokens(self, value: int) -> None:
        self._max_response_tokens = value

    @property
    def cache_ttl(self) -> str:
        return self._cache_ttl

    @cache_ttl.setter
    def cache_ttl(self, value: str) -> None:
        if value not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {value!r}")
        self._cache_ttl = value

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value

    @property
    def thinking(self) -> str | None:
        return self._thinking

    @thinking.setter
    def thinking(self, value: str | None) -> None:
        self._thinking = value

    @property
    def effort(self) -> str | None:
        return self._effort

    @effort.setter
    def effort(self, value: str | None) -> None:
        if value is not None and not self.model.supports_effort:
            raise ValueError(f"Model {self.model.model_id!r} does not support effort.")
        self._effort = value

    def swap_model(self, model: _ModelStub, *, spec: ModelSpec | None = None) -> None:
        if self._budget is not None:
            budget = self._budget
            if (
                budget.max_request_tokens > model.max_request_tokens
                or budget.max_response_tokens > model.max_response_tokens
            ):
                raise ValueError(
                    f"budget.max_request_tokens={budget.max_request_tokens} "
                    f"exceeds model's {model.max_request_tokens}"
                )
        self.model = model
        self.model_spec = spec
        self.swapped_spec = spec

    def reset_budget(self) -> None:
        self._budget = None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
