"""Tests for tools.play_audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import platform
import shutil
import subprocess

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools import play_audio


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    """Write a minimal RIFF/WAV header to a file.

    The header is syntactically valid - platform playback commands
    parse it as a silent zero-length PCM file, which is enough to
    exercise the code paths without an actual audio device.
    """
    # 44-byte standard PCM WAV header, zero samples.
    header = (
        b"RIFF\x24\x00\x00\x00WAVE"
        b"fmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00"  # PCM, mono
        b"\x44\xac\x00\x00"  # 44_100 Hz
        b"\x88\x58\x01\x00"  # byte rate
        b"\x02\x00\x10\x00"  # block align, 16-bit
        b"data\x00\x00\x00\x00"
    )
    p = tmp_path / "silent.wav"
    p.write_bytes(header)
    return p


def _none(_exe: str) -> str | None:
    return None


def _fake_bin(exe: str) -> str:
    return f"/fake/bin/{exe}"


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestPlayAudio:
    @pytest.mark.anyio
    async def test_rejects_missing_path(self) -> None:
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": ""})))
        assert r.descriptor == "text/x-error"
        assert "required" in str(r.content)

    @pytest.mark.anyio
    async def test_not_found(self, tmp_path: Path) -> None:
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": str(tmp_path / "nope.wav")})))
        assert r.descriptor == "text/x-error"
        assert "Not found" in str(r.content)

    @pytest.mark.anyio
    async def test_rejects_non_wav(self, tmp_path: Path) -> None:
        mp3 = tmp_path / "foo.mp3"
        mp3.write_bytes(b"\x00")
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": str(mp3)})))
        assert r.descriptor == "text/x-error"
        assert "Unsupported format" in str(r.content)

    @pytest.mark.anyio
    async def test_soft_fails_when_no_backend(
        self,
        wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force every candidate backend to miss PATH; tool must return
        # success ("best effort") rather than raising.
        monkeypatch.setattr(shutil, "which", _none)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": str(wav_path)})))
        assert "best effort" in _text(r)
        assert "not on PATH" in _text(r)

    @pytest.mark.anyio
    async def test_happy_path_invokes_first_backend(
        self,
        wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(shutil, "which", _fake_bin)
        calls: list[list[str]] = []

        class _Done:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(argv: list[str], **_kwargs: Any) -> _Done:
            calls.append(argv)
            return _Done()

        monkeypatch.setattr(subprocess, "run", _run)
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": str(wav_path)})))
        assert "Played" in _text(r)
        assert len(calls) == 1
        assert calls[0][0] == "/fake/bin/aplay"
        assert calls[0][-1] == str(wav_path)

    @pytest.mark.anyio
    async def test_falls_through_to_next_backend(
        self,
        wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(shutil, "which", _fake_bin)
        calls: list[str] = []

        class _Done:
            def __init__(self, rc: int) -> None:
                self.returncode = rc
                self.stdout = ""
                self.stderr = ""

        def _run(argv: list[str], **_kwargs: Any) -> _Done:
            calls.append(argv[0])
            return _Done(0 if "paplay" in argv[0] else 1)

        monkeypatch.setattr(subprocess, "run", _run)
        tool = play_audio.PlayAudio()
        r = await tool.run(_msg(json_freeze({"path": str(wav_path)})))
        assert "Played" in _text(r)
        # Tried aplay first, then paplay.
        assert [c.rsplit("/", 1)[-1] for c in calls] == ["aplay", "paplay"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
