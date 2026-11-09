"""Tests for the Antigravity ACP integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import base64
import os
import shutil

import pytest

from sagent.acp.agy import AgyAcp, find_agy_binary
from sagent.acp.client import image_content, text_content


def test_find_agy_binary_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_which(_cmd: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _mock_which)
    fake_empty = tmp_path / "empty"
    fake_empty.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("sagent.acp.agy.data_dir", lambda: fake_empty / "share")
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    # With no fake binary, returns the entry name
    assert find_agy_binary("agy") == "agy"

    # With a fake executable in ~/.local/bin
    fake_bin = tmp_path / "bin" / "agy"
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    fake_share = tmp_path / "share"
    monkeypatch.setattr("sagent.acp.agy.data_dir", lambda: fake_share)
    assert find_agy_binary("agy") == str(fake_bin)


def test_agy_acp_resolved_command_flags(tmp_path: Path) -> None:
    agy = AgyAcp(
        command=("/custom/agy", "acp"),
        dangerously_skip_permissions=True,
        agent="custom-agent",
        add_dirs=(tmp_path / "extra1", tmp_path / "extra2"),
    )
    cmd = agy.resolved_command()
    assert cmd[0] == "/custom/agy"
    assert cmd[1] == "acp"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--agent") + 1] == "custom-agent"
    assert "--add-dir" in cmd
    assert str(tmp_path / "extra1") in cmd
    assert str(tmp_path / "extra2") in cmd


def _has_real_agy() -> bool:
    target = find_agy_binary("agy")
    return shutil.which(target) is not None or (
        Path(target).is_file() and os.access(target, os.X_OK)
    )


def _extract_text(update: Mapping[str, object]) -> str:
    if update.get("sessionUpdate") == "agent_message_chunk":
        content: object = update.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            content_map = cast(Mapping[str, object], content)
            val = content_map.get("text")
            if isinstance(val, str):
                return val
    return ""


@pytest.mark.real_user_dirs
@pytest.mark.skipif(not _has_real_agy(), reason="agy binary not installed on system")
@pytest.mark.asyncio
async def test_agy_acp_live_text_prompt(tmp_path: Path) -> None:
    agy = AgyAcp(dangerously_skip_permissions=True)
    result = await agy.prompt(
        "Reply with only the word: PONG",
        cwd=tmp_path,
    )
    assert result.stop_reason in ("end_turn", "completed")
    assert len(result.updates) > 0
    full_text = "".join(_extract_text(cast(Mapping[str, object], u)) for u in result.updates)
    assert "PONG" in full_text


@pytest.mark.real_user_dirs
@pytest.mark.skipif(not _has_real_agy(), reason="agy binary not installed on system")
@pytest.mark.asyncio
async def test_agy_acp_live_multimodal_image_prompt(tmp_path: Path) -> None:
    agy = AgyAcp(dangerously_skip_permissions=True)
    # 1x1 black png
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    img_block = image_content(png_bytes, mime_type="image/png")
    txt_block = text_content("State the dimensions of this image in 5 words or less.")

    result = await agy.prompt(
        [img_block, txt_block],
        cwd=tmp_path,
    )
    assert result.stop_reason in ("end_turn", "completed")
    full_text = "".join(_extract_text(cast(Mapping[str, object], u)) for u in result.updates)
    assert "1x1" in full_text or "pixel" in full_text.lower() or "1" in full_text
