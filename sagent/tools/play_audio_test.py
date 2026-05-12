"""Tests for ``tools.play_audio``: cross-platform WAV playback shim."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import subprocess

import pytest

from sagent.testing import with_fake_agent
from sagent.tools import play_audio as pa
from sagent.tools.play_audio import PlayAudio


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """Create an empty .wav file so existence/suffix checks pass."""
    p = tmp_path / "beep.wav"
    p.write_bytes(b"RIFF")
    return p


@pytest.fixture
def play_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace ``_play`` with a stub that records each call and succeeds."""
    seen: list[Path] = []

    def fake_play(p: Path) -> str | None:
        seen.append(p)
        return None

    monkeypatch.setattr(pa, "_play", fake_play)
    return seen


def test_metadata_basics() -> None:
    t = PlayAudio()
    assert t.name == "PlayAudio"
    assert t.tool_id == "application/x-tool-playaudio"
    assert t.supports_microcompaction is True
    assert t.prompt() == ""
    assert t.summary_result(MagicMock()) is None


def test_summary_renders_basename() -> None:
    t = PlayAudio()
    assert t.summary({"path": "/x/y/beep.wav"}) == "PlayAudio beep.wav"
    assert t.summary({}) == "PlayAudio ?"


@pytest.mark.asyncio
async def test_run_missing_path() -> None:
    t = PlayAudio()
    with with_fake_agent():
        result = await t.run({"path": ""})
    assert result.is_error
    assert "path is required" in result.content


@pytest.mark.asyncio
async def test_run_not_found(tmp_path: Path) -> None:
    t = PlayAudio()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"path": "absent.wav"})
    assert result.is_error
    assert "Not found" in result.content


@pytest.mark.asyncio
async def test_run_not_a_file(tmp_path: Path) -> None:
    sub = tmp_path / "dir.wav"
    sub.mkdir()
    t = PlayAudio()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"path": str(sub)})
    assert result.is_error
    assert "Not a file" in result.content


@pytest.mark.asyncio
async def test_run_wrong_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "song.mp3"
    bad.write_bytes(b"ID3")
    t = PlayAudio()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(tmp_path)
        result = await t.run({"path": str(bad)})
    assert result.is_error
    assert "Unsupported format" in result.content


@pytest.mark.asyncio
async def test_run_relative_uses_bash_cwd(wav: Path, play_calls: list[Path]) -> None:
    t = PlayAudio()
    with with_fake_agent() as agent:
        agent.tool_state.bash_cwd = str(wav.parent)
        result = await t.run({"path": wav.name})
    assert not result.is_error
    assert "Played beep.wav" in result.content
    assert play_calls == [wav]


@pytest.mark.asyncio
async def test_run_absolute_path_success(wav: Path, play_calls: list[Path]) -> None:
    t = PlayAudio()
    with with_fake_agent():
        result = await t.run({"path": str(wav)})
    assert not result.is_error
    assert play_calls == [wav]


@pytest.mark.asyncio
async def test_run_play_soft_fail_reported(
    wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _soft_fail(_: Path) -> str:
        return "no audio backend worked"

    monkeypatch.setattr(pa, "_play", _soft_fail)
    t = PlayAudio()
    with with_fake_agent():
        result = await t.run({"path": str(wav)})
    # Soft fail returns success-ish text (not is_error) with reason embedded.
    assert not result.is_error
    assert "best effort" in result.content


def test_play_via_cmd_no_path_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _none(_: str) -> str | None:
        return None

    monkeypatch.setattr("shutil.which", _none)
    out = pa._play_via_cmd(Path("/x.wav"), ["aplay"])
    assert out is not None
    assert "not on PATH" in out


def test_play_via_cmd_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(exe: str) -> str:
        return f"/usr/bin/{exe}"

    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr("shutil.which", _which)
    monkeypatch.setattr("subprocess.run", _run)
    assert pa._play_via_cmd(Path("/x.wav"), ["aplay"]) is None


def test_play_via_cmd_nonzero_returns_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _which(exe: str) -> str:
        return f"/usr/bin/{exe}"

    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr=""
    )

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr("shutil.which", _which)
    monkeypatch.setattr("subprocess.run", _run)
    out = pa._play_via_cmd(Path("/x.wav"), ["aplay"])
    assert out is not None
    assert "exit 2" in out


def test_play_via_cmd_exception_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(exe: str) -> str:
        return f"/usr/bin/{exe}"

    def _boom(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("nope")

    monkeypatch.setattr("shutil.which", _which)
    monkeypatch.setattr("subprocess.run", _boom)
    out = pa._play_via_cmd(Path("/x.wav"), ["aplay"])
    assert out is not None
    assert "aplay" in out


def test_play_dispatch_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    captured: list[tuple[Path, tuple[list[str], ...]]] = []

    def fake_via_cmd(path: Path, *candidates: list[str]) -> str | None:
        captured.append((path, candidates))
        return None

    monkeypatch.setattr(pa, "_play_via_cmd", fake_via_cmd)
    assert pa._play(Path("/x.wav")) is None
    assert captured[0][0] == Path("/x.wav")
    assert [c[0] for c in captured[0][1]] == ["aplay", "paplay", "play"]


def test_play_dispatch_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    captured: list[list[str]] = []

    def fake_via_cmd(_: Path, *candidates: list[str]) -> str | None:
        captured.extend(candidates)
        return None

    monkeypatch.setattr(pa, "_play_via_cmd", fake_via_cmd)
    assert pa._play(Path("/x.wav")) is None
    assert captured == [["afplay"]]


def test_play_dispatch_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    def _stub(_: Path) -> str | None:
        return None

    monkeypatch.setattr(pa, "_play_windows", _stub)
    assert pa._play(Path("/x.wav")) is None


def test_play_windows_no_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: str) -> object:
        raise ImportError("not on POSIX")

    monkeypatch.setattr("importlib.import_module", boom)
    out = pa._play_windows(Path("/x.wav"))
    assert out is not None
    assert "winsound unavailable" in out


def test_play_windows_runtime_error() -> None:
    fake_ws = MagicMock()
    fake_ws.SND_FILENAME = 1
    fake_ws.SND_NODEFAULT = 2
    fake_ws.PlaySound.side_effect = RuntimeError("hw fail")
    with patch("importlib.import_module", return_value=fake_ws):
        out = pa._play_windows(Path("/x.wav"))
    assert out is not None
    assert "winsound.PlaySound failed" in out


def test_play_windows_success() -> None:
    fake_ws = MagicMock()
    fake_ws.SND_FILENAME = 1
    fake_ws.SND_NODEFAULT = 2
    with patch("importlib.import_module", return_value=fake_ws):
        assert pa._play_windows(Path("/x.wav")) is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
