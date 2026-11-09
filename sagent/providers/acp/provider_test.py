"""Tests for the ACP provider in ModelCatalog."""

from __future__ import annotations

from pathlib import Path

import base64
import os
import shutil

import pytest

from sagent.acp.agy import find_agy_binary
from sagent.model.types import (
    Request,
    Response,
    TextChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from sagent.providers.acp.provider import Acp, AcpModel
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import Attachment, UserMessage


def test_acp_provider_resolution() -> None:
    provider = Acp()
    assert provider.name == "acp"
    assert "acp-agent" in provider.aliases
    assert "gemini-flash-3.8" in list(provider)
    assert "claude-sonnet-4.6" in list(provider)

    model = provider["agy"]
    assert isinstance(model, AcpModel)
    assert model.capability.model_id == "gemini-flash-3.8"

    default_model = provider["default"]
    assert isinstance(default_model, AcpModel)
    assert default_model.capability.model_id == "gemini-flash-3.8"


def test_acp_attachment_tokens() -> None:
    provider = Acp()
    model = provider["agy"]
    img_att = Attachment(mime_descriptor="image/png", data=b"fake")
    assert model.attachment_tokens(img_att) == 258

    bin_att = Attachment(mime_descriptor="application/octet-stream", data=b"x" * 100)
    assert model.attachment_tokens(bin_att) > 0


@pytest.mark.asyncio
async def test_acp_in_model_catalog() -> None:
    async with ModelCatalog() as catalog:
        assert "acp" in catalog.factories
        model = catalog["acp"]["agy"]
        assert isinstance(model, AcpModel)
        assert model.capability.model_id == "gemini-flash-3.8"
        claude = catalog["acp"]["claude-sonnet"]
        assert isinstance(claude, AcpModel)
        assert claude.capability.model_id == "claude-sonnet-4.6"


def _has_real_agy() -> bool:
    target = find_agy_binary("agy")
    return shutil.which(target) is not None or (
        Path(target).is_file() and os.access(target, os.X_OK)
    )


@pytest.mark.real_user_dirs
@pytest.mark.skipif(not _has_real_agy(), reason="agy binary not installed on system")
@pytest.mark.asyncio
async def test_acp_model_live_stream_text(tmp_path: Path) -> None:
    provider = Acp(cwd=tmp_path)
    async with provider:
        model = provider["agy"]
        request = Request(
            records=(UserMessage(content="Respond with only the word: PONG"),)
        )
        events = [event async for event in model.stream(request)]

        chunks = [e for e in events if isinstance(e, TextChunk)]
        responses = [e for e in events if isinstance(e, Response)]

        assert len(chunks) > 0
        assert len(responses) == 1
        full_text = "".join(c.text for c in chunks)
        assert "PONG" in full_text


@pytest.mark.real_user_dirs
@pytest.mark.skipif(not _has_real_agy(), reason="agy binary not installed on system")
@pytest.mark.asyncio
async def test_acp_model_live_stream_with_image(tmp_path: Path) -> None:
    provider = Acp(cwd=tmp_path)
    async with provider:
        model = provider["agy"]
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        att = Attachment(mime_descriptor="image/png", data=png_bytes)
        request = Request(
            records=(
                UserMessage(
                    content="State the dimensions of this attached image in 5 words or less.",
                    attachments=(att,),
                ),
            )
        )
        events = [event async for event in model.stream(request)]

        chunks = [e for e in events if isinstance(e, TextChunk)]
        assert len(chunks) > 0
        full_text = "".join(c.text for c in chunks)
        assert "1x1" in full_text or "pixel" in full_text.lower() or "1" in full_text


@pytest.mark.real_user_dirs
@pytest.mark.skipif(not _has_real_agy(), reason="agy binary not installed on system")
@pytest.mark.asyncio
async def test_acp_model_live_tool_call_streaming(tmp_path: Path) -> None:
    provider = Acp(cwd=tmp_path)
    async with provider:
        model = provider["agy"]
        request = Request(
            records=(
                UserMessage(
                    content="Run the command echo ACP_LIVE_TOOL_CALL_STREAMING_TEST and output its result.",
                ),
            )
        )
        events = [event async for event in model.stream(request)]

        tool_calls = [e for e in events if isinstance(e, ToolCallChunk)]
        tool_results = [e for e in events if isinstance(e, ToolResultChunk)]
        responses = [e for e in events if isinstance(e, Response)]

        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1
        assert len(responses) == 1

        call = tool_calls[0].call
        assert call.call_id
        assert call.name in ("run_command", "bash", "shell")

        res = tool_results[0].result
        assert res.call_id == call.call_id
        assert res.is_error is not True
        assert "ACP_LIVE_TOOL_CALL_STREAMING_TEST" in res.text()
