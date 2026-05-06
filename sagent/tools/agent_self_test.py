"""Tests for AgentSelf tool."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    ModelSpec,
    MultipartMessage,
)
from sagent.lib.json import MutableJSON, json_freeze
from sagent.tools.agent_self import AgentSelf
from sagent.tools.core import (
    ToolState,
    current_agent_var,
    tool_state_context,
)


def _msg(op: str, **kwargs: object) -> Message:
    d = cast(MutableJSON, {"operation": op, **kwargs})
    return MultipartMessage(
        (
            JsonMessage(
                json_freeze(d),
                "application/x-tool-agentself",
            ),
        ),
        "multipart/x-tool-call",
    )


@pytest.fixture
def tool_state() -> Iterator[ToolState]:
    ts = ToolState()
    with tool_state_context(ts):
        yield ts


@pytest.mark.usefixtures("tool_state")
class TestStatus:
    @pytest.mark.anyio
    async def test_sets_status(self) -> None:
        agent = MagicMock()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("status", status="Debugging tests"))
            assert result.descriptor == "text/x-status"
            assert result.content == "Debugging tests"
            agent.set_status.assert_called_once_with("Debugging tests")
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_empty_status_returns_error(self) -> None:
        result = await AgentSelf().run(_msg("status", status=""))
        assert result.descriptor == "text/x-error"

    @pytest.mark.anyio
    async def test_rejects_limit_fields_before_setting_status(self) -> None:
        agent = MagicMock()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg("status", status="Debugging tests", max_request_tokens=0)
            )
            assert result.descriptor == "text/x-error"
            assert 'operation="limits"' in cast(str, result.content)
            agent.set_status.assert_not_called()
        finally:
            current_agent_var.reset(token)


class TestClear:
    @pytest.mark.anyio
    async def test_sets_deferred_flag(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(_msg("clear"))
        assert tool_state.clear_requested is not None
        assert "queued" in cast(str, result.content).lower()

    @pytest.mark.anyio
    async def test_with_reason(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(_msg("clear", reason="fresh start"))
        assert tool_state.clear_requested == "fresh start"
        assert "fresh start" in cast(str, result.content)


class TestCompact:
    @pytest.mark.anyio
    async def test_sets_deferred_flag(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(_msg("compact"))
        assert tool_state.compact_requested is not None
        assert "queued" in cast(str, result.content).lower()

    @pytest.mark.anyio
    async def test_with_instructions(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(
            _msg("compact", custom_instructions="keep the API spec")
        )
        assert tool_state.compact_requested == "keep the API spec"
        assert "keep the API spec" in cast(str, result.content)


class TestRecompact:
    @pytest.mark.anyio
    async def test_sets_deferred_flag(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(_msg("recompact"))
        assert tool_state.recompact_requested is not None
        assert "queued" in cast(str, result.content).lower()

    @pytest.mark.anyio
    async def test_with_instructions(self, tool_state: ToolState) -> None:
        result = await AgentSelf().run(
            _msg("recompact", custom_instructions="preserve numbers")
        )
        assert tool_state.recompact_requested == "preserve numbers"
        assert "preserve numbers" in cast(str, result.content)


class TestDiagnostics:
    @pytest.mark.anyio
    @pytest.mark.usefixtures("tool_state")
    async def test_empty_stats(self) -> None:
        result = await AgentSelf().run(_msg("diagnostics"))
        assert "No stats yet" in cast(str, result.content)

    @pytest.mark.anyio
    async def test_formats_populated_stats(self, tool_state: ToolState) -> None:
        tool_state.stats = {
            "max_request_tokens": 200_000,
            "input_tokens": 50_000,
            "output_tokens": 10_000,
            "cache_creation_tokens": 1_000,
            "cache_read_tokens": 500,
            "total_cost_usd": 1.23,
            "num_tool_call_rounds": 5,
        }
        result = await AgentSelf().run(_msg("diagnostics"))
        text = cast(str, result.content)
        assert "50,000" in text
        assert "25.0%" in text
        assert "$1.23" in text

    @pytest.mark.anyio
    async def test_zero_max_request_tokens(self, tool_state: ToolState) -> None:
        tool_state.stats = {"max_request_tokens": 0, "input_tokens": 100}
        result = await AgentSelf().run(_msg("diagnostics"))
        assert "0.0%" in cast(str, result.content)


@pytest.mark.usefixtures("tool_state")
class TestModel:
    @pytest.mark.anyio
    async def test_empty_model_id_returns_error(self) -> None:
        agent = MagicMock()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("model", model_id=""))
            assert result.descriptor == "text/x-error"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_no_model_spec_returns_error(self) -> None:
        agent = MagicMock()
        agent.model_spec = None
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("model", model_id="claude-sonnet-4-6"))
            assert result.descriptor == "text/x-error"
            assert "no model spec" in cast(str, result.content).lower()
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_rejects_limit_fields_on_model(self) -> None:
        agent = MagicMock()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg("model", model_id="gpt-5.5", max_request_tokens=1)
            )
            assert result.descriptor == "text/x-error"
            assert 'operation="limits"' in cast(str, result.content)
            agent.swap_model.assert_not_called()
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_selfhosted_local_path_updates_auth(self) -> None:
        agent = MagicMock()
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
                    model_id="/new/model"
                )
                result = await AgentSelf().run(_msg("model", model_id="/new/model"))
            assert result.descriptor == "text/plain"
            mock_bp.assert_called_once_with("SelfHosted", "/new/model", account=None)
            spec = agent.swap_model.call_args.kwargs["spec"]
            assert spec.provider == "SelfHosted"
            assert spec.auth == "/new/model"
            assert spec.model_id == "/new/model"
        finally:
            current_agent_var.reset(token)


class _LimitAgent:
    def __init__(self) -> None:
        self._max_request_tokens = 200_000
        self._max_response_tokens = 8_000

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


@pytest.mark.usefixtures("tool_state")
class TestLimits:
    @pytest.mark.anyio
    async def test_requires_a_limit_field(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("limits"))
            assert result.descriptor == "text/x-error"
            assert "requires" in cast(str, result.content)
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_rejects_zero_max_request_tokens(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("limits", max_request_tokens=0))
            assert result.descriptor == "text/x-error"
            text = cast(str, result.content)
            assert "max_request_tokens=0" in text
            assert "at least 1" in text
            assert agent.max_request_tokens == 200_000
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_surfaces_buffer_invariant_for_too_small_request_tokens(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("limits", max_request_tokens=1))
            assert result.descriptor == "text/x-error"
            assert "buffer_tokens" in cast(str, result.content)
            assert agent.max_request_tokens == 200_000
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_updates_limits(self) -> None:
        agent = _LimitAgent()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(
                _msg("limits", max_request_tokens=160_000, max_response_tokens=12_000)
            )
            assert result.descriptor == "text/plain"
            assert agent.max_request_tokens == 160_000
            assert agent.max_response_tokens == 12_000
        finally:
            current_agent_var.reset(token)


class TestCacheTtl:
    @pytest.mark.anyio
    async def test_sets_5m(self) -> None:
        agent = MagicMock(cache_ttl="5m")
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("cache_ttl", ttl="5m"))
            assert result.descriptor == "text/plain"
            assert agent.cache_ttl == "5m"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_sets_1h(self) -> None:
        agent = MagicMock(cache_ttl="5m")
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("cache_ttl", ttl="1h"))
            assert result.descriptor == "text/plain"
            assert agent.cache_ttl == "1h"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_rejects_invalid_ttl(self) -> None:
        """Setter raises on bad values; handler surfaces the error.

        Models the real ``Agent`` setter; without a validating setter,
        a plain ``MagicMock`` would silently accept any string and the
        tool would report success.
        """

        class _AgentStub:
            def __init__(self) -> None:
                self._cache_ttl = "5m"

            @property
            def cache_ttl(self) -> str:
                return self._cache_ttl

            @cache_ttl.setter
            def cache_ttl(self, value: str) -> None:
                if value not in ("5m", "1h"):
                    raise ValueError(
                        f"cache_ttl must be '5m' or '1h', got {value!r}",
                    )
                self._cache_ttl = value

        agent = _AgentStub()
        token = current_agent_var.set(cast(Any, agent))
        try:
            result = await AgentSelf().run(_msg("cache_ttl", ttl="2h"))
            assert result.descriptor == "text/x-error"
        finally:
            current_agent_var.reset(token)

    @pytest.mark.anyio
    async def test_no_active_agent(self) -> None:
        result = await AgentSelf().run(_msg("cache_ttl", ttl="1h"))
        assert result.descriptor == "text/x-error"


@pytest.mark.usefixtures("tool_state")
class TestUnknownOperation:
    @pytest.mark.anyio
    async def test_returns_error(self) -> None:
        result = await AgentSelf().run(_msg("bogus"))
        assert result.descriptor == "text/x-error"
        assert "bogus" in cast(str, result.content)


class TestPrompt:
    def test_static_guidance_lives_in_description(self) -> None:
        assert AgentSelf().prompt() == ""
        desc = AgentSelf.description
        assert "model_id" in desc
        assert "provider" in desc
        assert "auth" in desc
        assert "max_request_tokens" in desc
        assert 'operation="limits"' in desc


class TestHelp:
    def test_status(self) -> None:
        h = AgentSelf().summary(_msg("status", status="Debugging"))
        assert "AgentSelf" in h
        assert "Debugging" in h

    def test_compact(self) -> None:
        assert "AgentSelf compact" in AgentSelf().summary(_msg("compact"))

    def test_clear(self) -> None:
        assert "AgentSelf clear" in AgentSelf().summary(_msg("clear"))

    def test_diagnostics(self) -> None:
        assert "AgentSelf diagnostics" in AgentSelf().summary(_msg("diagnostics"))

    def test_model(self) -> None:
        h = AgentSelf().summary(_msg("model", model_id="claude-opus-4-6"))
        assert "AgentSelf" in h
        assert "opus" in h

    def test_limits(self) -> None:
        h = AgentSelf().summary(
            _msg("limits", max_request_tokens=160_000, max_response_tokens=12_000)
        )
        assert "AgentSelf limits" in h
        assert "max_request_tokens=160000" in h
        assert "max_response_tokens=12000" in h

    def test_status_hides_unexpected_limits(self) -> None:
        h = AgentSelf().summary(
            _msg("status", status="Debugging", max_request_tokens=0)
        )
        assert "status=Debugging" in h
        assert "max_request_tokens=0" not in h


class TestSchema:
    def test_limit_fields_have_minimums_and_guidance(self) -> None:
        schema = cast(dict[str, Any], AgentSelf.directive_schema)
        props = cast(dict[str, dict[str, Any]], schema["properties"])
        operation = props["operation"]
        assert schema["additionalProperties"] is False
        assert "limits" in cast(list[str], operation["enum"])
        req = props["max_request_tokens"]
        resp = props["max_response_tokens"]
        assert req["minimum"] == 1
        assert resp["minimum"] == 1
        assert "operation='limits'" in cast(str, req["description"])
        assert "current max_request_tokens" in cast(str, req["description"])


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
