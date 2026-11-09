"""Tests for the Sagent-specific VS Code process protocol."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import asyncio
import os
import subprocess
import sys

from treekle import json_loads

import pytest

from sagent.model.capability import ModelCapability
from sagent.runtime import types
from sagent.session.types import AssistantMessage, ShellCommandResult, ToolCall
from sagent.vscode.server import (
    SagentVscodeServer,
    _model_settings_arguments,
    _record_metadata,
    _resolve_extra_prompt,
    _resolve_prompt_text,
    _setting_values,
    main,
    serve,
)


@pytest.mark.cli_python_subprocess
def test_module_entrypoint_uses_a_real_pipe(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repository), environment.get("PYTHONPATH")) if part
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "sagent.vscode"],
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
        process.stdin.write('{"id":1,"method":"initialize"}\n')
        process.stdin.flush()
        initialize = json_loads(process.stdout.readline())
        assert initialize["result"]["protocol"] == "sagent-vscode/1"
        process.stdin.write('{"id":2,"method":"shutdown"}\n')
        process.stdin.flush()
        shutdown = json_loads(process.stdout.readline())
        assert shutdown["result"]["stopped"] is True
        stopped = json_loads(process.stdout.readline())
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
async def test_protocol_serves_requests_until_shutdown(tmp_path: Path) -> None:
    incoming = StringIO(
        '{"id":1,"method":"initialize","params":{}}\n'
        '{"id":2,"method":"getTeam","params":{}}\n'
        '{"id":3,"method":"listSessions","params":{}}\n'
        '{"id":4,"method":"shutdown","params":{}}\n',
    )
    outgoing = StringIO()

    await serve(
        input_stream=incoming,
        output_stream=outgoing,
        cwd=tmp_path,
        model_name="default",
    )

    messages = [json_loads(line) for line in outgoing.getvalue().splitlines()]
    assert messages[0] == {
        "id": 1,
        "result": {
            "protocol": "sagent-vscode/1",
            "capabilities": {
                "streaming": True,
                "sessions": True,
                    "team": True,
                    "agentSessions": True,
                    "modelSettings": True,
                    "commands": True,
                    "nativeWorkspace": True,
                    "skills": True,
                },
        },
    }
    assert messages[1] == {"id": 2, "result": {"open": False, "agents": []}}
    assert messages[2] == {
        "id": 3,
        "result": {"activePath": None, "sessions": []},
    }
    assert messages[3] == {"id": 4, "result": {"stopped": True}}
    assert messages[4] == {"method": "sagent/stop", "params": {}}


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
        '{"id":1,"method":"getSkills","params":{}}\n'
        '{"id":2,"method":"shutdown","params":{}}\n',
    )
    outgoing = StringIO()

    await serve(
        input_stream=incoming,
        output_stream=outgoing,
        cwd=tmp_path,
        model_name="default",
        skill_directories=(skills,),
    )

    messages = [json_loads(line) for line in outgoing.getvalue().splitlines()]
    result = messages[0]["result"]
    assert "skills" in result
    assert result["skills"][0]["name"] == "trax"
    assert result["skills"][0]["description"] == "Track durable work."
    assert result["skills"][0]["source"] == "user"


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
    assert _resolve_extra_prompt("Inline instructions.", tmp_path) == "Inline instructions."


def test_resolve_extra_prompt_handles_empty_string(tmp_path: Path) -> None:
    assert _resolve_extra_prompt("", tmp_path) == ""


def test_server_initializes_system_and_append_prompts(tmp_path: Path) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentVscodeServer(
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
    server = SagentVscodeServer(
        cwd=tmp_path,
        model_name="default",
        extra_prompt="Legacy instructions.",
        output=queue,
    )
    assert server.system_prompt is None
    assert server.append_system_prompt == "Legacy instructions."


@pytest.mark.asyncio
async def test_server_observes_streaming_tool_call_and_result(tmp_path: Path) -> None:
    queue = asyncio.Queue[dict[str, object]]()
    server = SagentVscodeServer(
        cwd=tmp_path,
        model_name="default",
        output=queue,
    )
    server._active_turn_id = "turn-1"
    server._active_agent_label = "Agent"

    # Tool call started
    call = ToolCall(
        call_id="call-42",
        name="run_command",
        arguments={"CommandLine": "echo hello", "toolSummary": "Echo greeting"},
    )
    server._observe(types.ToolCallStarted(call=call, title="echo hello"), "Agent")

    notif1 = queue.get_nowait()
    assert notif1["method"] == "sagent/event"
    assert notif1["params"] == {
        "type": "tool.started",
        "turnId": "turn-1",
        "callId": "call-42",
        "name": "run_command",
        "arguments": {"CommandLine": "echo hello", "toolSummary": "Echo greeting"},
        "summary": "echo hello",
        "status": "running",
        "agentLabel": "Agent",
    }

    # Tool result completed
    result = ShellCommandResult(
        call_id="call-42",
        stdout="hello\n",
        exit_code=0,
        is_error=False,
    )
    server._observe(result, "Agent")

    notif2 = queue.get_nowait()
    assert notif2["method"] == "sagent/event"
    params2 = notif2["params"]
    assert isinstance(params2, dict)
    assert params2["type"] == "tool.completed"
    assert params2["callId"] == "call-42"
    assert params2["status"] == "completed"
    assert params2["text"] == "hello\n"
    assert params2["isError"] is False
