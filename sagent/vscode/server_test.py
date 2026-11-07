"""Tests for the Sagent-specific VS Code process protocol."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import os
import subprocess
import sys

from treekle import json_loads

import pytest

from sagent.model.capability import ModelCapability
from sagent.vscode.server import _model_settings_arguments, _setting_values, serve


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


def test_model_capability_values_keep_native_json_types() -> None:
    capability = ModelCapability(
        cache_ttl_sec=frozenset({0.0, 300.0}),
        manage_context_server_side=frozenset({False, True}),
    )

    assert _setting_values(capability, "cache_ttl_sec") == [0.0, 300.0]
    assert _setting_values(capability, "manage_context_server_side") == [False, True]


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
