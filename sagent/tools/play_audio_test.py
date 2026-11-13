"""Contracts for PlayAudio decisions and its player boundary; no sound is made."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, call

import logging
import platform
import shutil
import subprocess

import pytest

from sagent.session.types import PlayAudioResult
from sagent.tools.play_audio import PlayAudio
from sagent.tools.types import ToolContext, settled


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


def test_the_first_player_that_works_stops_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    which = Mock(side_effect=[None, "/players/paplay"])
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    assert PlayAudio()._play(Path("/sounds/ding.wav")) == ""

    assert which.call_args_list == [call("aplay"), call("paplay")]
    run.assert_called_once_with(
        ["/players/paplay", "/sounds/ding.wav"],
        capture_output=True,
        timeout=10.0,
        check=False,
    )


@pytest.mark.parametrize(
    ("system", "player", "arguments"),
    [
        ("Linux", "aplay", ["-q", "/sounds/it's.wav"]),
        ("Linux", "play", ["-q", "/sounds/it's.wav"]),
        ("Darwin", "afplay", ["/sounds/it's.wav"]),
        (
            "Windows",
            "powershell",
            [
                "-NoProfile",
                "-Command",
                "(New-Object Media.SoundPlayer '/sounds/it''s.wav').PlaySync()",
            ],
        ),
    ],
)
def test_each_platform_runs_its_own_player(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    player: str,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)

    def find(name: str) -> str | None:
        return f"/players/{name}" if name == player else None

    which = Mock(side_effect=find)
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    assert PlayAudio()._play(Path("/sounds/it's.wav")) == ""

    run.assert_called_once_with(
        [f"/players/{player}", *arguments],
        capture_output=True,
        timeout=10.0,
        check=False,
    )


def test_unavailable_failing_and_unstartable_players_are_all_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    which = Mock(side_effect=["/players/aplay", "/players/paplay", None])
    run = Mock(side_effect=[OSError("audio down"), subprocess.CompletedProcess([], 1)])
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    failure = PlayAudio()._play(Path("/sounds/ding.wav"))

    assert failure == (
        "no audio backend worked: aplay (audio down); paplay (exit 1); play (not on PATH)"
    )
    assert which.call_args_list == [call("aplay"), call("paplay"), call("play")]
    assert run.call_args_list == [
        call(
            argv,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        for argv in (
            ["/players/aplay", "-q", "/sounds/ding.wav"],
            ["/players/paplay", "/sounds/ding.wav"],
        )
    ]


def test_a_player_that_times_out_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    which = Mock(return_value="/players/afplay")
    run = Mock(side_effect=subprocess.TimeoutExpired("afplay", 0.05))
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)

    failure = PlayAudio(timeout_sec=0.05)._play(Path("/sounds/ding.wav"))

    assert failure == (
        "no audio backend worked: afplay (Command 'afplay' timed out after 0.05 seconds)"
    )
    run.assert_called_once_with(
        ["/players/afplay", "/sounds/ding.wav"],
        capture_output=True,
        timeout=0.05,
        check=False,
    )


@pytest.mark.asyncio
async def test_a_host_with_no_working_player_still_succeeds_and_warns(
    tmp_path: Path,
    wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    play = Mock(return_value="unavailable")
    monkeypatch.setattr(PlayAudio, "_play", play)

    with caplog.at_level(logging.WARNING, logger="sagent.tools.play_audio"):
        result = await settled(
            PlayAudio().call(ToolContext(cwd=tmp_path), {"path": "ding.wav"}),
        )

    play.assert_called_once_with(wav)
    assert result.result == PlayAudioResult(path=str(wav))
    assert (result.text(), result.is_error, result.call_id) == (
        "Played ding.wav (best effort): unavailable",
        False,
        "",
    )
    assert caplog.messages == ["PlayAudio: unavailable"]


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
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    message: str,
) -> None:
    (tmp_path / "dir.wav").mkdir()
    (tmp_path / "ding.mp3").write_bytes(b"ID3")
    play = Mock(return_value="")
    monkeypatch.setattr(PlayAudio, "_play", play)

    result = await settled(PlayAudio().call(ToolContext(cwd=tmp_path), {"path": name}))

    assert result.is_error is True
    assert result.call_id == ""
    assert result.text() == message.format(tmp_path / name)
    play.assert_not_called()


@pytest.mark.asyncio
async def test_an_upper_case_suffix_is_still_a_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sound = tmp_path / "DING.WAV"
    sound.write_bytes(b"RIFF")
    play = Mock(return_value="")
    monkeypatch.setattr(PlayAudio, "_play", play)

    result = await settled(
        PlayAudio().call(ToolContext(cwd=tmp_path), {"path": "DING.WAV"}),
    )

    assert (result.text(), result.is_error, result.call_id) == (
        "Played DING.WAV.",
        False,
        "",
    )
    play.assert_called_once_with(sound)


@pytest.mark.asyncio
async def test_a_call_without_a_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing argument 'path'"):
        await PlayAudio().call(ToolContext(cwd=tmp_path), {}).result


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
