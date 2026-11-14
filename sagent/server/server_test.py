"""Tests for the Sagent headless daemon RPC protocol."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import asyncio
import os
import subprocess
import sys
from functools import partial

from treekle import json_dumps, json_loads

import pytest

from sagent.model.capability import ModelCapability
from sagent.model.scripted import ScriptedProvider, reply
from sagent.providers.catalog import ModelCatalog
from sagent.providers.usage import ProviderUsage
from sagent.runtime import types
from sagent.server.protocol import (
    CATALOG_METHODS,
    COMMAND_METHODS,
    NATIVE_EVENT_METHOD,
    NATIVE_NOTIFICATION_METHOD,
    NATIVE_PROTOCOL,
    SESSION_METHODS,
    SYSTEM_METHODS,
)
from sagent.server.model_settings import (
    _model_settings_arguments,
    _setting_values,
    _usage_payload,
)
from sagent.server.server import (
    SagentServer,
    _resolve_extra_prompt,
    _resolve_prompt_text,
    serve,
)
from sagent.session.controller import AgentSessionController
from sagent.session.host import AgentSession
from sagent.session.snapshot import _record_metadata
from sagent.session.types import (
    AssistantMessage,
    ShellCommandResult,
    ToolCall,
    UserMessage,
)


def _wire(message: object) -> dict[str, Any]:
    """Read a queued daemon message exactly as the process client does."""
    parsed = json_loads(json_dumps(message))
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.asyncio
async def test_protocol_serves_requests_until_shutdown(tmp_path: Path) -> None:
    incoming = StringIO(
        '{"id":1,"method":"system.initialize","params":{}}\n'
        '{"id":2,"method":"session.team.state","params":{}}\n'
        '{"id":3,"method":"session.list","params":{}}\n'
        '{"id":4,"method":"system.shutdown","params":{}}\n',
    )
    outgoing = StringIO()

    await serve(
        input_stream=incoming,
        output_stream=outgoing,
        cwd=tmp_path,
        model_name="default",
    )

    messages = [json_loads(line) for line in outgoing.getvalue().splitlines()]
    initialize_result = messages[0]["result"]
    assert isinstance(initialize_result, dict)
    assert initialize_result["protocol"] == NATIVE_PROTOCOL
    assert initialize_result["implementation"]["name"] == "sagent"
    assert initialize_result["implementation"]["version"]
    assert str(initialize_result["implementation"]["serverPath"]).endswith(
        "/sagent/server/server.py",
    )
    assert initialize_result["capabilities"] == {
        "streaming": True,
        "sessions": True,
        "team": True,
        "agentSessions": True,
        "modelSettings": True,
        "commands": True,
        "nativeWorkspace": True,
        "skills": True,
        "nativeEvents": True,
        "nativeRecords": True,
    }
    assert messages[1] == {"id": 2, "result": {"open": False, "agents": []}}
    assert messages[2] == {
        "id": 3,
        "result": {"activePath": None, "sessions": []},
    }
    assert messages[3] == {"id": 4, "result": {"stopped": True}}
    assert messages[4] == {"method": "sagent/stop", "params": {}}


@pytest.mark.cli_python_subprocess
def test_module_entrypoint_supports_native_handshake(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repository), environment.get("PYTHONPATH")) if part
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "sagent.server"],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            '{"id":1,"method":"system.initialize",'
            '"params":{}}\n',
        )
        process.stdin.flush()
        initialize = json_loads(process.stdout.readline(), dict[str, Any])
        result = initialize.get("result")
        assert isinstance(result, dict)
        assert result["protocol"] == NATIVE_PROTOCOL
        assert result["capabilities"]["nativeEvents"] is True
        assert result["capabilities"]["nativeRecords"] is True
        implementation = result["implementation"]
        assert implementation["name"] == "sagent"
        assert implementation["version"]
        assert str(implementation["serverPath"]).endswith("/sagent/server/server.py")

        process.stdin.write(
            '{"id":2,"method":"session.state","params":{}}\n'
            '{"id":3,"method":"system.shutdown","params":{}}\n',
        )
        process.stdin.flush()
        state = json_loads(process.stdout.readline(), dict[str, Any])
        assert state == {
            "id": 2,
            "result": {
                "open": False,
                "sessionId": None,
                "snapshot": {"version": 1, "items": []},
                "records": [],
                "team": {"open": False, "agents": []},
            },
        }
        shutdown = json_loads(process.stdout.readline(), dict[str, Any])
        assert shutdown == {"id": 3, "result": {"stopped": True}}
        stopped = json_loads(process.stdout.readline(), dict[str, Any])
        assert stopped["method"] == "sagent/stop"
        process.stdin.close()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.asyncio
async def test_get_skills_discovers_configured_skill_roots(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skill = skills / "trax"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: trax\ndescription: Track durable work.\n---\n\nUse trax.\n",
        encoding="utf-8",
    )
    incoming = StringIO(
        '{"id":1,"method":"catalog.skills","params":{}}\n'
        '{"id":2,"method":"system.shutdown","params":{}}\n',
    )
    outgoing = StringIO()

    await serve(
        input_stream=incoming,
        output_stream=outgoing,
        cwd=tmp_path,
        model_name="default",
        skill_directories=(skills,),
    )

    messages = [
        json_loads(line, dict[str, Any]) for line in outgoing.getvalue().splitlines()
    ]
    result = messages[0].get("result")
    assert isinstance(result, dict)
    assert "skills" in result
    skills_list = result["skills"]
    assert isinstance(skills_list, list)
    skill_item = skills_list[0]
    assert isinstance(skill_item, dict)
    assert skill_item["name"] == "trax"
    assert skill_item["description"] == "Track durable work."
    assert skill_item["source"] == "user"


def test_model_capability_values_keep_native_json_types() -> None:
    capability = ModelCapability(
        cache_ttl_sec=frozenset({0.0, 300.0}),
        manage_context_server_side=frozenset({False, True}),
    )

    assert _setting_values(capability, "cache_ttl_sec") == [0.0, 300.0]
    assert _setting_values(capability, "manage_context_server_side") == [False, True]


def test_record_metadata_preserves_native_timestamp_before_fallback() -> None:
    record = AssistantMessage(timestamp="2026-01-02T03:04:05+00:00")

    assert _record_metadata(record, fallback_timestamp="fallback") == {
        "timestamp": "2026-01-02T03:04:05+00:00",
    }


def test_record_metadata_can_supply_completion_timestamp() -> None:
    assert _record_metadata(AssistantMessage(), fallback_timestamp="fallback") == {
        "timestamp": "fallback",
    }


def test_provider_selection_uses_sagents_native_default_model_role() -> None:
    assert _model_settings_arguments(
        {"type": "select-provider", "provider": "google-cli"},
    ) == {"provider": "google-cli", "model": "default"}


def test_custom_provider_configuration_requires_an_exact_model() -> None:
    assert _model_settings_arguments(
        {
            "type": "configure-custom-provider",
            "baseUrl": "https://api.experientiallabs.ai/v1",
            "model": "gpt-6-luna",
        },
    ) == {"provider": "custom", "model": "gpt-6-luna"}

    with pytest.raises(ValueError, match="Model is required"):
        _model_settings_arguments(
            {
                "type": "configure-custom-provider",
                "baseUrl": "https://api.experientiallabs.ai/v1",
            },
        )


@pytest.mark.asyncio
async def test_unknown_method_is_returned_as_a_protocol_error(tmp_path: Path) -> None:
    incoming = StringIO('{"id":"bad","method":"not-a-method"}\n')
    outgoing = StringIO()

    await serve(
        input_stream=incoming,
        output_stream=outgoing,
        cwd=tmp_path,
        model_name="default",
    )

    messages = [json_loads(line) for line in outgoing.getvalue().splitlines()]
    assert messages[0] == {
        "id": "bad",
        "error": {"code": "sagent_error", "message": "unknown method 'not-a-method'"},
    }
    assert messages[1] == {"method": "sagent/stop", "params": {}}


def test_resolve_prompt_text_handles_none(tmp_path: Path) -> None:
    assert _resolve_prompt_text(None, tmp_path) is None


def test_resolve_extra_prompt_reads_file_when_path_exists(tmp_path: Path) -> None:
    prompt_file = tmp_path / "custom-prompt.md"
    prompt_file.write_text("Custom instructions.", encoding="utf-8")
    assert _resolve_extra_prompt(str(prompt_file), tmp_path) == "Custom instructions."


def test_resolve_extra_prompt_preserves_inline_string(tmp_path: Path) -> None:
    assert (
        _resolve_extra_prompt("Inline instructions.", tmp_path)
        == "Inline instructions."
    )


def test_resolve_extra_prompt_handles_empty_string(tmp_path: Path) -> None:
    assert _resolve_extra_prompt("", tmp_path) == ""


def test_server_initializes_system_and_append_prompts(tmp_path: Path) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        system_prompt="Base instructions.",
        append_system_prompt="Appended instructions.",
        output=queue,
    )
    assert server.system_prompt == "Base instructions."
    assert server.append_system_prompt == "Appended instructions."


def test_server_supports_extra_prompt_backwards_compatibility(tmp_path: Path) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        extra_prompt="Legacy instructions.",
        output=queue,
    )
    assert server.system_prompt is None
    assert server.append_system_prompt == "Legacy instructions."


@pytest.mark.asyncio
async def test_native_protocol_serializes_runtime_events_without_ui_projection(
    tmp_path: Path,
) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=queue,
    )
    initialized = await server._dispatch("system.initialize", {})

    assert initialized["protocol"] == "sagent/v2"
    assert initialized["capabilities"]["nativeEvents"] is True

    server._observe(types.ModelResponsePartial(text="hello", record=2), "Agent")
    message = _wire(queue.get_nowait())
    assert message["method"] == "session.event"
    assert message["params"]["protocol"] == NATIVE_PROTOCOL
    assert message["params"]["eventType"] == "ModelResponsePartial"
    assert message["params"]["agentLabel"] == "Agent"
    assert str(message["params"]["value"]["py/object"]).endswith(
        ".ModelResponsePartial",
    )
    assert message["params"]["value"]["text"] == "hello"
    assert message["params"]["value"]["record"] == 2
    assert message["params"]["sequence"] == 1


@pytest.mark.asyncio
async def test_native_protocol_serializes_daemon_notifications_with_native_envelope(
    tmp_path: Path,
) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=queue,
    )
    await server._dispatch("system.initialize", {})

    server._notify(
        "command.output",
        {"text": "done", "tone": "info", "command": "/help"},
    )

    message = _wire(queue.get_nowait())
    assert message["method"] == NATIVE_NOTIFICATION_METHOD
    assert message["params"]["eventType"] == "command.output"
    assert message["params"]["value"] == {
        "type": "command.output",
        "text": "done",
        "tone": "info",
        "command": "/help",
    }


@pytest.mark.asyncio
async def test_native_protocol_vertical_slice_uses_a_real_scripted_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the daemon request boundary with Sagent's real runtime.

    The subprocess handshake test above proves the real process entrypoint. This
    test supplies a deterministic native Sagent provider while exercising the
    real session construction, daemon dispatch, streamed events, state, close,
    and shutdown without credentials or a network provider.
    """
    monkeypatch.setattr(
        "sagent.session.host.ModelCatalog",
        lambda: ModelCatalog(
            factories={
                "scripted": partial(
                    ScriptedProvider,
                    replies=[reply("PONG")],
                ),
            },
        ),
    )
    output = asyncio.Queue[dict[str, object]]()

    server = SagentServer(
        cwd=tmp_path,
        model_name="scripted/scripted",
        output=output,
    )
    notifications: list[dict[str, object]] = []

    async def request(
        request_id: int,
        method: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        await server.handle({"id": request_id, "method": method, "params": params})
        while True:
            message = await asyncio.wait_for(output.get(), timeout=2)
            if message.get("id") == request_id:
                return message
            notifications.append(message)

    initialized = await request(
        1,
        "system.initialize",
        {},
    )
    assert initialized["result"]["protocol"] == NATIVE_PROTOCOL
    assert initialized["result"]["implementation"]["name"] == "sagent"

    opened = await request(2, "session.open", {})
    assert opened["result"]["open"] is True
    assert opened["result"]["snapshot"] == {"version": 1, "items": []}

    submitted = await request(
        3,
        "session.submit",
        {"text": "Say PONG", "agentLabel": "Agent", "mode": "immediate"},
    )
    assert submitted["result"]["accepted"] is True

    event_types: list[str] = []
    while "AgentIdle" not in event_types:
        message = await asyncio.wait_for(output.get(), timeout=2)
        if message.get("method") != NATIVE_EVENT_METHOD:
            notifications.append(message)
            continue
        event_type = message["params"].get("eventType")
        if isinstance(event_type, str):
            event_types.append(event_type)
    assert "ModelResponsePartial" in event_types
    assert "ModelResponseComplete" in event_types

    state = await request(4, "session.state", {})
    snapshot = state["result"]["snapshot"]
    assert snapshot["items"][-1]["type"] == "assistant"
    assert snapshot["items"][-1]["text"] == "PONG"
    assert state["result"]["records"]

    halted = await request(5, "session.halt", {"agentLabel": "Agent"})
    assert halted["result"] == {"cancelled": False}

    closed = await request(6, "session.close", {})
    assert closed["result"] == {"closed": True}
    shutdown = await request(7, "system.shutdown", {})
    assert shutdown["result"] == {"stopped": True}


@pytest.mark.asyncio
async def test_one_daemon_keeps_two_native_sessions_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening a second chat must not replace the first native session."""
    monkeypatch.setattr(
        "sagent.session.host.ModelCatalog",
        lambda: ModelCatalog(
            factories={
                "scripted": partial(
                    ScriptedProvider,
                    replies=[reply("PONG")],
                ),
            },
        ),
    )
    server = SagentServer(
        cwd=tmp_path,
        model_name="scripted/scripted",
        output=asyncio.Queue(),
    )

    first = await server._dispatch(
        "session.open",
        {"sessionPath": str(tmp_path / "first" / "session.jsonl")},
    )
    second = await server._dispatch(
        "session.open",
        {"sessionPath": str(tmp_path / "second" / "session.jsonl")},
    )
    first_id = first["sessionId"]
    second_id = second["sessionId"]
    assert isinstance(first_id, str)
    assert isinstance(second_id, str)
    assert first_id != second_id

    first_state = await server._dispatch("session.state", {"sessionId": first_id})
    second_state = await server._dispatch("session.state", {"sessionId": second_id})
    assert first_state["path"] == str(tmp_path / "first" / "session.jsonl")
    assert second_state["path"] == str(tmp_path / "second" / "session.jsonl")
    assert set(server._sessions.ids()) == {first_id, second_id}

    first_submission = await server._dispatch(
        "session.submit",
        {"sessionId": first_id, "text": "first"},
    )
    second_submission = await server._dispatch(
        "session.submit",
        {"sessionId": second_id, "text": "second"},
    )
    assert first_submission["accepted"] is True
    assert second_submission["accepted"] is True

    await server._dispatch("session.close", {"sessionId": first_id})
    assert first_id not in server._sessions.ids()
    assert second_id in server._sessions.ids()
    await server.close()


@pytest.mark.asyncio
async def test_canonical_protocol_dispatches_system_and_session_methods(
    tmp_path: Path,
) -> None:
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=asyncio.Queue(),
    )

    initialized = await server._dispatch(
        "system.initialize", {}
    )
    assert initialized["protocol"] == "sagent/v2"
    assert (await server._dispatch("session.state", {}))["open"] is False
    assert (await server._dispatch("session.team.state", {}))["open"] is False
    assert (await server._dispatch("session.list", {}))["sessions"] == []
    assert await server._dispatch("session.close", {}) == {"closed": True}


def test_native_wire_uses_treekle_for_native_values() -> None:
    message = _wire(
        {
            "value": types.ModelResponseThinking(text="think", record=1),
        },
    )
    assert str(message["value"]["py/object"]).endswith(
        ".ModelResponseThinking",
    )
    assert message["value"]["text"] == "think"
    assert message["value"]["record"] == 1


def test_native_protocol_contract_has_explicit_boundaries() -> None:
    assert NATIVE_PROTOCOL == "sagent/v2"
    assert NATIVE_EVENT_METHOD == "session.event"
    assert NATIVE_NOTIFICATION_METHOD == "session.notification"
    assert {"system.initialize", "system.shutdown"} == SYSTEM_METHODS
    assert {
        "session.open",
        "session.list",
        "session.state",
        "session.records",
        "session.team.state",
        "session.agent.state",
        "session.submit",
        "session.halt",
        "session.close",
    } == SESSION_METHODS
    assert {
        "catalog.skills",
        "catalog.modelSettings",
        "catalog.modelSettings.set",
        "catalog.usage",
    } == CATALOG_METHODS
    assert {"commands.list", "commands.execute"} == COMMAND_METHODS


@pytest.mark.skipif(
    not hasattr(types, "ToolCallStarted"),
    reason="ToolCallStarted streaming event is not available",
)
@pytest.mark.asyncio
async def test_server_observes_streaming_tool_call_and_result(tmp_path: Path) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=queue,
    )
    mock_session = MagicMock()
    server._session = mock_session
    server._active_agent_label = "Agent"

    # Tool call started
    call = ToolCall(
        call_id="call-42",
        name="run_command",
        arguments={"CommandLine": "echo hello", "toolSummary": "Echo greeting"},
    )
    tool_call_started_cls = getattr(types, "ToolCallStarted", None)
    assert tool_call_started_cls is not None
    server._observe(tool_call_started_cls(call=call, title="echo hello"), "Agent")

    notif1 = _wire(queue.get_nowait())
    assert notif1["method"] == NATIVE_EVENT_METHOD
    assert notif1["params"]["protocol"] == NATIVE_PROTOCOL
    assert notif1["params"]["eventType"] == "ToolCallStarted"
    assert notif1["params"]["agentLabel"] == "Agent"
    assert notif1["params"]["sequence"] == 1
    assert str(notif1["params"]["value"]["py/object"]).endswith(
        ".ToolCallStarted",
    )

    # Tool result completed
    result = ShellCommandResult(
        call_id="call-42",
        stdout="hello\n",
        exit_code=0,
        is_error=False,
    )
    server._observe(result, "Agent")

    notif2 = _wire(queue.get_nowait())
    assert notif2["method"] == NATIVE_EVENT_METHOD
    assert notif2["params"]["protocol"] == NATIVE_PROTOCOL
    assert notif2["params"]["eventType"] == "ShellCommandResult"
    assert notif2["params"]["agentLabel"] == "Agent"
    assert notif2["params"]["sequence"] == 2
    assert str(notif2["params"]["value"]["py/object"]).endswith(
        ".ShellCommandResult",
    )


def test_usage_payload_converts_provider_usage() -> None:
    assert _usage_payload(None) is None
    usage = ProviderUsage(
        provider="acp",
        account="dev",
        summary="Antigravity: 90% left",
        details=("Tier: free-tier", "gemini: 90% left"),
        raw={"key": "val"},
    )
    payload = _usage_payload(usage)
    assert payload == {
        "provider": "acp",
        "account": "dev",
        "summary": "Antigravity: 90% left",
        "details": ["Tier: free-tier", "gemini: 90% left"],
        "raw": {"key": "val"},
    }


@pytest.mark.asyncio
async def test_native_protocol_routes_submission_modes_records_and_halt(
    tmp_path: Path,
) -> None:
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=asyncio.Queue(),
    )
    agent = MagicMock()
    agent.session = MagicMock()
    agent.session.__iter__.side_effect = lambda: iter((UserMessage(content="history"),))
    agent.runtime.is_idle = False
    session = MagicMock()
    session.team = {"Agent": agent}
    session.agents = session.team
    session.controller = AgentSessionController(session.team)
    session.submit.side_effect = session.controller.send
    session.halt.side_effect = session.controller.halt
    server._session = session

    initialized = await server._dispatch("system.initialize", {})
    assert initialized["protocol"] == "sagent/v2"

    for mode in ("immediate", "queued", "deferred"):
        result = await server._dispatch(
            "session.submit", {"text": mode, "agentLabel": "Agent", "mode": mode}
        )
        assert result["accepted"] is True
        assert result["mode"] == mode

    records = await server._dispatch("session.records", {"agentLabel": "Agent"})
    assert records["records"][0]["recordType"] == "UserMessage"
    assert await server._dispatch("session.halt", {}) == {"cancelled": True}


@pytest.mark.asyncio
async def test_native_halt_targets_requested_child_agent(
    tmp_path: Path,
) -> None:
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=asyncio.Queue(),
    )
    child = MagicMock()
    child.runtime.is_idle = False
    session = MagicMock()
    session.agents = {"child": child}
    session.halt.return_value = True
    server._session = session

    result = await server._dispatch("session.halt", {"agentLabel": "child"})

    assert result == {"cancelled": True}
    session.halt.assert_called_once_with(agent_label="child")


@pytest.mark.asyncio
async def test_native_state_queries_serialize_live_agent_and_team_state(
    tmp_path: Path,
) -> None:
    server = SagentServer(
        cwd=tmp_path,
        model_name="default",
        output=asyncio.Queue(),
    )
    agent = MagicMock()
    agent.session = MagicMock()
    agent.session.__iter__.side_effect = lambda: iter((UserMessage(content="history"),))
    agent.session.journal.path = tmp_path / "agent.jsonl"
    agent.runtime.tool_context.label = "Agent"
    agent.runtime.conversation = "conversation-1"
    agent.runtime.is_idle = True
    agent.runtime.accepts_user_dispatch = True
    agent.runtime.accepts_deferred_dispatch = False
    agent.runtime.model_call = None
    agent.runtime.compact_task = None
    agent.runtime.cohort = set()
    agent.runtime.running_tools = {}
    agent.runtime.jobs = []
    agent.running = False
    agent.tools = {"Echo": object()}
    agent.model.capability.model_id = "model-1"
    agent.model.provider.name = "provider-1"
    agent.account = "default"
    agent.allowed = None
    agent.spend_usd = 0.0
    agent.max_spend_usd = 1.0

    team = MagicMock()
    team.__iter__.return_value = iter(("Agent",))
    team.__contains__.side_effect = {"Agent": True}.__contains__
    team.__getitem__.side_effect = {"Agent": agent}.__getitem__
    team.parent.return_value = None
    team.above.return_value = None
    team.children.return_value = []
    team.watched.return_value = None
    team.persistent.return_value = set()
    team.owes.return_value = {}

    session = MagicMock()
    session.path = tmp_path / "session" / "agent.jsonl"
    session.agent = agent
    session.team = team
    server._session = session

    await server._dispatch("system.initialize", {})
    state = await server._dispatch("session.state", {})
    agent_state = await server._dispatch("session.agent.state", {"label": "Agent"})

    assert state["records"][0]["recordType"] == "UserMessage"
    assert state["snapshot"]["version"] == 1
    assert state["snapshot"]["items"]
    assert state["team"]["agents"][0]["runtime"] == {
        "idle": True,
        "acceptsUserDispatch": True,
        "acceptsDeferredDispatch": False,
        "modelCall": False,
        "compacting": False,
        "cohortSize": 0,
        "runningTools": 0,
        "jobs": [],
    }
    assert agent_state["agent"]["conversationId"] == "conversation-1"
    assert agent_state["records"][0]["recordType"] == "UserMessage"
    assert agent_state["snapshot"]["version"] == 1
    assert agent_state["snapshot"]["items"]
