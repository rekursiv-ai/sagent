"""Tests for the generic ACP v1 client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import sys

import pytest

from sagent.acp.client import (
    AcpClient,
    AcpError,
    PermissionDecision,
    image_content,
    text_content,
)


if TYPE_CHECKING:
    from pathlib import Path


_STUB_SERVER_SCRIPT = """
import sys
import json

def read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)

def write_msg(msg):
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\\n")
    sys.stdout.flush()

while True:
    req = read_msg()
    if req is None:
        break
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        write_msg({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": True, "audio": False, "embeddedContext": True},
                    "sessionCapabilities": {"close": True},
                },
                "agentInfo": {"name": "StubAgent", "version": "1.0"},
            }
        })
    elif method == "session/new":
        write_msg({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "sess-stub-001"}})
    elif method == "session/load":
        write_msg({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": params.get("sessionId", "sess-stub-001")}})
    elif method == "session/prompt":
        prompt_items = params.get("prompt", [])
        text_val = ""
        for p in prompt_items:
            if p.get("type") == "text":
                text_val = p.get("text", "")
        if text_val == "TRIGGER_PERMISSION":
            # Server sends a request to the client
            write_msg({
                "jsonrpc": "2.0",
                "id": "perm-req-1",
                "method": "session/request_permission",
                "params": {
                    "sessionId": params.get("sessionId"),
                    "options": [
                        {"optionId": "allow-1", "kind": "allow_once", "name": "Allow"},
                        {"optionId": "reject-1", "kind": "reject_once", "name": "Reject"},
                    ]
                }
            })
            perm_resp = read_msg()
            perm_result = perm_resp.get("result", {}) if perm_resp else {}
            outcome = perm_result.get("outcome", {}).get("outcome")
            write_msg({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": params.get("sessionId"),
                    "update": {"sessionUpdate": "agent_message_chunk", "content": f"permission was {outcome}"}
                }
            })
            write_msg({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})
        elif text_val == "HANG_UNTIL_CANCEL":
            write_msg({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": params.get("sessionId"),
                    "update": {"sessionUpdate": "agent_message_chunk", "content": "started"}
                }
            })
            # wait for cancel notification
            cancel_msg = read_msg()
            write_msg({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "cancelled"}})
        else:
            write_msg({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": params.get("sessionId"),
                    "update": {"sessionUpdate": "agent_message_chunk", "content": f"Echo: {text_val}"}
                }
            })
            write_msg({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})
    elif method == "session/close":
        write_msg({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "session/cancel":
        pass
"""


@pytest.fixture
def stub_server_cmd() -> tuple[str, ...]:
    return (sys.executable, "-u", "-c", _STUB_SERVER_SCRIPT)


@pytest.mark.asyncio
async def test_acp_client_initialize_and_session_lifecycle(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(command=stub_server_cmd, cwd=tmp_path)
    async with client:
        assert client.agent_info == {"name": "StubAgent", "version": "1.0"}
        assert client.agent_capabilities is not None
        assert client.agent_capabilities.load_session is True
        assert client.agent_capabilities.image_prompt is True
        assert client.agent_capabilities.audio_prompt is False
        assert client.agent_capabilities.embedded_context is True
        assert client.agent_capabilities.session_capabilities.get("close") is True

        session = await client.new_session()
        assert session.session_id == "sess-stub-001"

        loaded = await client.load_session("sess-stub-001")
        assert loaded.session_id == "sess-stub-001"

        await client.close_session(session)


@pytest.mark.asyncio
async def test_acp_client_prompt_and_updates(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(command=stub_server_cmd, cwd=tmp_path)
    async with client:
        session = await client.new_session()
        result = await client.prompt(session, "Hello world")
        assert result.stop_reason == "end_turn"
        assert len(result.updates) == 1
        assert result.updates[0].get("sessionUpdate") == "agent_message_chunk"
        assert result.updates[0].get("content") == "Echo: Hello world"


@pytest.mark.asyncio
async def test_acp_client_prompt_stream(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(command=stub_server_cmd, cwd=tmp_path)
    async with client:
        session = await client.new_session()
        updates = [
            dict(update)
            async for update in client.prompt_stream(session, "Stream test")
        ]
        assert len(updates) == 1
        assert updates[0].get("content") == "Echo: Stream test"


@pytest.mark.asyncio
async def test_acp_client_permission_auto_allow(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(
        command=stub_server_cmd,
        cwd=tmp_path,
        on_permission=PermissionDecision.auto_allow,
    )
    async with client:
        session = await client.new_session()
        result = await client.prompt(session, "TRIGGER_PERMISSION")
        assert result.stop_reason == "end_turn"
        assert any(
            "permission was selected" in str(u.get("content")) for u in result.updates
        )


@pytest.mark.asyncio
async def test_acp_client_permission_default_cancels(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(command=stub_server_cmd, cwd=tmp_path)
    async with client:
        session = await client.new_session()
        result = await client.prompt(session, "TRIGGER_PERMISSION")
        assert result.stop_reason == "end_turn"
        assert any(
            "permission was cancelled" in str(u.get("content")) for u in result.updates
        )


@pytest.mark.asyncio
async def test_acp_client_cancel(
    stub_server_cmd: tuple[str, ...], tmp_path: Path
) -> None:
    client = AcpClient(command=stub_server_cmd, cwd=tmp_path)
    async with client:
        session = await client.new_session()
        prompt_task = asyncio.create_task(client.prompt(session, "HANG_UNTIL_CANCEL"))
        await asyncio.sleep(0.05)
        await client.cancel(session)
        result = await prompt_task
        assert result.stop_reason == "cancelled"


def test_permission_decision_helpers() -> None:
    allow = PermissionDecision.allow("opt-allow")
    assert allow.to_json() == {
        "outcome": {"outcome": "selected", "optionId": "opt-allow"}
    }

    cancel = PermissionDecision.cancel()
    assert cancel.to_json() == {"outcome": {"outcome": "cancelled"}}

    bad_selected = PermissionDecision(outcome="selected", option_id=None)
    with pytest.raises(ValueError, match="requires an option_id"):
        bad_selected.to_json()

    # auto_allow with explicit allow option
    decision1 = PermissionDecision.auto_allow(
        {
            "options": [
                {"optionId": "opt-1", "kind": "reject_once"},
                {"optionId": "opt-2", "kind": "allow_once"},
            ]
        }
    )
    assert decision1.option_id == "opt-2"

    # auto_allow with optionId containing 'allow'
    decision2 = PermissionDecision.auto_allow(
        {
            "options": [
                {"optionId": "allow_edit", "name": "Allow edit"},
            ]
        }
    )
    assert decision2.option_id == "allow_edit"

    # auto_allow fallback to first option
    decision3 = PermissionDecision.auto_allow(
        {
            "options": [
                {"optionId": "custom_opt"},
            ]
        }
    )
    assert decision3.option_id == "custom_opt"

    # auto_allow with empty options cancels
    decision4 = PermissionDecision.auto_allow({"options": []})
    assert decision4.outcome == "cancelled"


def test_content_block_helpers() -> None:
    txt = text_content("hello")
    assert txt == {"type": "text", "text": "hello"}

    img = image_content(b"fake_bytes", mime_type="image/png", uri="file:///tmp/img.png")
    assert img["type"] == "image"
    assert img["mimeType"] == "image/png"
    assert img["uri"] == "file:///tmp/img.png"
    assert isinstance(img["data"], str)


@pytest.mark.asyncio
async def test_acp_client_protocol_version_mismatch(tmp_path: Path) -> None:
    mismatch_server = (
        sys.executable,
        "-u",
        "-c",
        """
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": 99}})+ "\\n")
sys.stdout.flush()
""",
    )
    client = AcpClient(command=mismatch_server, cwd=tmp_path)
    with pytest.raises(AcpError, match="unsupported ACP protocol version: 99"):
        async with client:
            pass


@pytest.mark.asyncio
async def test_acp_client_exit_with_error_reports_stderr(tmp_path: Path) -> None:
    crash_server = (
        sys.executable,
        "-u",
        "-c",
        """
import sys
sys.stderr.write("Fatal crash during startup\\n")
sys.stderr.flush()
sys.exit(42)
""",
    )
    client = AcpClient(command=crash_server, cwd=tmp_path)
    with pytest.raises(AcpError, match="Fatal crash during startup"):
        async with client:
            pass
