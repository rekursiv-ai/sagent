"""Regression for cached OpenAI request prefixes across tool-result growth."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import json

import httpx2
import pytest

from sagent.agent.agent import Agent
from sagent.compaction.summary import SummaryCompactor
from sagent.lib.custom_json import DictCodec, ListCodec
from sagent.providers.openai.api import OpenAI
from sagent.tools.bash import Bash
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.settings import AgentSettings


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    captured = _Capture()

    async def send(
        client: httpx2.AsyncClient,
        request: httpx2.Request,
        **kwargs: object,
    ) -> httpx2.Response:
        del client, kwargs
        return await captured.send(request)

    monkeypatch.setattr(httpx2.AsyncClient, "send", send)
    return captured


@pytest.mark.anyio
async def test_compacting_agent_preserves_prefix_past_half_window(
    capture: _Capture,
) -> None:
    provider = OpenAI.from_key("offline")
    model = provider.model("sol-6")
    agent = Agent(
        model=model,
        system="Stable instructions",
        tools=[Bash()],
        compactor=SummaryCompactor(),
        budget=AgentSettings(
            max_request_tokens=20_000,
            max_response_tokens=128,
            buffer_tokens=1_000,
        ),
    )
    content = "cache-result " * 1_750
    result_tokens = model.approx_text_tokens(content)
    assert result_tokens < agent.tool_results.persist_tokens
    assert result_tokens * 3 > agent.max_request_tokens // 2
    agent.runtime.append_history(UserMessage(text="Start."))
    try:
        with agent._install_contextvars():
            for index in range(3):
                call_id = f"call_{index}"
                agent.runtime.append_history(
                    AssistantMessage(
                        tool_calls=(
                            ToolCall(
                                id=call_id,
                                name="Bash",
                                args={"command": "cat sample.txt"},
                            ),
                        ),
                    ),
                )
                agent.runtime.append_history(
                    ToolResult(call_id=call_id, content=content),
                )
                await agent._agent_model.stream(
                    agent.history,
                    publish=lambda _event: None,
                )
    finally:
        await provider.close_sdk()
    bodies = [DictCodec.coerce(json.loads(raw)) for raw in capture.payloads]
    for before, after in pairwise(bodies):
        assert _prefix_items(before, after=after) == len(
            ListCodec.mappings(before["input"]),
        )
        assert (
            json.dumps(before["tools"]).encode() == json.dumps(after["tools"]).encode()
        )
        assert before["instructions"] == after["instructions"]


@dataclass(slots=True, kw_only=True)
class _Capture:
    payloads: list[bytes] = field(default_factory=list)

    async def send(
        self,
        request: httpx2.Request,
    ) -> httpx2.Response:
        self.payloads.append(request.content)
        return httpx2.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"type":"response.completed","sequence_number":0,'
                b'"response":{"id":"resp_replay","object":"response",'
                b'"created_at":0,"model":"gpt-6-sol","status":"completed",'
                b'"output":[]}}\n\n'
            ),
        )


def _prefix_items(before: Mapping[str, object], *, after: Mapping[str, object]) -> int:
    old_items = ListCodec.mappings(before["input"])
    new_items = ListCodec.mappings(after["input"])
    for index, (old, new) in enumerate(zip(old_items, new_items, strict=False)):
        if json.dumps(old).encode() != json.dumps(new).encode():
            return index
    return min(len(old_items), len(new_items))


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
