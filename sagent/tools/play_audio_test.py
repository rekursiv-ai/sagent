"""Tests for PlayAudio, with stand-in players on ``PATH``: no sound is made."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
import platform
import re

import pytest

from sagent.tools.play_audio import PlayAudio
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def players(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return an empty directory that is the whole ``PATH``; players go in it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "ding.wav"
    path.write_bytes(b"RIFF")
    return path


def test_summary_names_the_file() -> None:
    assert PlayAudio().summary({"path": "/sounds/ding.wav"}) == "PlayAudio ding.wav"
    assert PlayAudio().summary({}) == "PlayAudio "


def test_playback_runs_beside_anything(tmp_path: Path) -> None:
    assert (
        PlayAudio().serialize_key(ToolContext(cwd=tmp_path), {"path": "a.wav"}) is None
    )


@pytest.mark.asyncio
async def test_the_first_player_that_works_plays_the_file(
    tmp_path: Path,
    players: Path,
    wav: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    _player(players, "paplay", log=tmp_path / "paplay.args")
    _player(players, "play", log=tmp_path / "play.args")
    result = await PlayAudio().call(ToolContext(cwd=tmp_path), {"path": "ding.wav"})
    assert (result.text(), result.is_error) == ("Played ding.wav.", None)
    assert result.call_id == ""
    assert (tmp_path / "paplay.args").read_text() == f"{wav}\n"
    assert not (tmp_path / "play.args").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system", "player", "args"),
    [
        ("Linux", "aplay", ["-q", "{wav}"]),
        ("Linux", "play", ["-q", "{wav}"]),
        ("Darwin", "afplay", ["{wav}"]),
        (
            "Windows",
            "powershell",
            [
                "-NoProfile",
                "-Command",
                "(New-Object Media.SoundPlayer '{quoted}').PlaySync()",
            ],
        ),
    ],
)
async def test_each_platform_runs_its_own_player(
    tmp_path: Path,
    players: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str,
    player: str,
    args: list[str],
) -> None:
    sound = tmp_path / "it's.wav"
    sound.write_bytes(b"RIFF")
    monkeypatch.setattr(platform, "system", lambda: system)
    _player(players, player, log=tmp_path / "args")
    result = await PlayAudio().call(ToolContext(cwd=tmp_path), {"path": str(sound)})
    assert result.text() == "Played it's.wav."
    quoted = str(sound).replace("'", "''")
    assert (tmp_path / "args").read_text().splitlines() == [
        a.format(wav=sound, quoted=quoted) for a in args
    ]


@pytest.mark.asyncio
async def test_a_host_with_no_working_player_still_succeeds_and_says_why(
    tmp_path: Path,
    players: Path,
    wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    _player(players, "paplay", code=1)
    (players / "play").write_text("#!/nonexistent/shell\n")
    (players / "play").chmod(0o755)
    with caplog.at_level(logging.WARNING):
        result = await PlayAudio().call(ToolContext(cwd=tmp_path), {"path": str(wav)})
    assert result.is_error is None
    assert re.fullmatch(
        r"Played ding\.wav \(best effort\): no audio backend worked: aplay \(not on"
        r" PATH\); paplay \(exit 1\); play \(\[Errno \d+\] .+\)",
        result.text(),
    )
    (warning,) = caplog.records
    assert warning.getMessage().startswith("PlayAudio: no audio backend worked: ")


@pytest.mark.asyncio
async def test_a_player_that_hangs_is_given_up_on(
    tmp_path: Path,
    players: Path,
    wav: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    (players / "afplay").write_text("#!/bin/sh\nexec /bin/sleep 5\n")
    (players / "afplay").chmod(0o755)
    result = await PlayAudio(timeout_sec=0.05).call(
        ToolContext(cwd=tmp_path),
        {"path": str(wav)},
    )
    assert result.text().startswith(
        "Played ding.wav (best effort): no audio backend worked: afplay (Command",
    )
    assert "timed out after 0.05 seconds)" in result.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("absent.wav", "Not a file: {}"),
        ("dir.wav", "Not a file: {}"),
        ("ding.mp3", "Unsupported format '.mp3'; only .wav is supported."),
    ],
)
async def test_what_is_no_wav_file_is_an_error(
    tmp_path: Path,
    players: Path,
    name: str,
    message: str,
) -> None:
    (tmp_path / "dir.wav").mkdir()
    (tmp_path / "ding.mp3").write_bytes(b"ID3")
    _player(players, "aplay", log=tmp_path / "args")
    result = await PlayAudio().call(ToolContext(cwd=tmp_path), {"path": name})
    assert result.is_error is True
    assert result.text() == message.format(tmp_path / name)
    assert not (tmp_path / "args").exists()


@pytest.mark.asyncio
async def test_an_upper_case_suffix_is_still_a_wav(
    tmp_path: Path,
    players: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    (tmp_path / "DING.WAV").write_bytes(b"RIFF")
    _player(players, "aplay")
    result = await PlayAudio().call(ToolContext(cwd=tmp_path), {"path": "DING.WAV"})
    assert result.text() == "Played DING.WAV."


@pytest.mark.asyncio
async def test_a_call_without_a_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing argument 'path'"):
        await PlayAudio().call(ToolContext(cwd=tmp_path), {})


def _player(
    bin_dir: Path,
    name: str,
    *,
    code: int = 0,
    log: Path | None = None,
) -> None:
    """Put a stand-in ``name`` on ``PATH`` that records its arguments and exits ``code``."""
    record = f"printf '%s\\n' \"$@\" > '{log}'\n" if log is not None else ""
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\n{record}exit {code}\n")
    script.chmod(0o755)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
