"""Offline tests for public examples."""

from __future__ import annotations

import asyncio

from examples.custom_tool import CharacterCount
from examples.decorator_tool import word_count
from examples.offline_custom_tool import run_example
from examples.openai_compatible_provider import LocalOpenAI
from sagent.custom_types import JsonMessage, MultipartMessage
from sagent.lib.json import json_freeze
from sagent.tools.agent_spawn import AgentSpawn


def _tool_call(**directive: object) -> MultipartMessage:
    return MultipartMessage(
        (
            JsonMessage(
                json_freeze(directive),
                "application/x-tool-character-count",
            ),
        ),
        "multipart/x-tool-call",
    )


def test_custom_tool_counts_text() -> None:
    tool = CharacterCount()

    result = asyncio.run(tool.run(_tool_call(text="agentic systems")))

    assert result.content == "15"
    assert result.descriptor == "text/plain"


def test_decorator_tool_counts_words() -> None:
    result = asyncio.run(word_count.run(_tool_call(text="typed agents compose")))

    assert result.content == "3"
    assert result.descriptor == "text/plain"


def test_offline_custom_tool_example_runs_without_api_keys() -> None:
    assert asyncio.run(run_example()) == "Echo said: hello"


def test_local_openai_provider_uses_env_overrides() -> None:
    assert LocalOpenAI.DEFAULT_MODEL
    assert LocalOpenAI.DEFAULT_MODEL in LocalOpenAI.KNOWN_MODELS
    assert LocalOpenAI.BASE_URL


def test_multi_agent_example_uses_agent_spawn() -> None:
    assert AgentSpawn.name == "AgentSpawn"
    assert AgentSpawn.directive_schema.get("required") == ("prompt",)
