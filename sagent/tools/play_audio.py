"""PlayAudio: play a WAV file on the host, for a short notification.

Each platform's own player is run: PowerShell's ``SoundPlayer`` on Windows,
``afplay`` on macOS, and ``aplay``, ``paplay`` or ``play`` elsewhere. Playback is best
effort: a host with no audio, such as one over SSH, still succeeds.

Typical usage example:

  result = await PlayAudio().call(context, {"path": "done.wav"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import asyncio
import logging
import platform
import shutil
import subprocess

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "PlayAudio",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayAudio:
    """Play a WAV file on the host's speakers."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        path: Annotated[str, "Path to a .wav file (absolute or cwd-relative)."]

    name = "PlayAudio"
    description = (files(__package__) / "assets" / "playaudio.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    timeout_sec: float = 10.0
    """How long one player may run."""

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``PlayAudio <name>``."""
        return f"PlayAudio {Path(str(arguments.get('path', ''))).name}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: playback shares nothing."""
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Play the file at ``path``.

        Args:
          context: The calling agent's state.
          arguments: ``path``, relative to ``context.cwd``.

        Returns:
          result: What played, or why the file could not; a host with no audio
            still succeeds, saying so.

        """
        path = context.path(parse(arguments, PlayAudio.Arguments).path)
        if not path.is_file():
            return _error(f"Not a file: {path}")
        if path.suffix.lower() != ".wav":
            return _error(
                f"Unsupported format {path.suffix!r}; only .wav is supported.",
            )
        failure = await asyncio.to_thread(self._play, path)
        if failure:
            logger.warning("PlayAudio: %s", failure)
            return UncategorizedToolResult(
                call_id="",
                content=f"Played {path.name} (best effort): {failure}",
            )
        return UncategorizedToolResult(call_id="", content=f"Played {path.name}.")

    def _play(self, path: Path) -> str:
        """Play ``path``; return why it did not play, ``""`` when it did."""
        system = platform.system()
        if system == "Windows":
            quoted = str(path).replace("'", "''")
            players = [
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(New-Object Media.SoundPlayer '{quoted}').PlaySync()",
                ],
            ]
        elif system == "Darwin":
            players = [["afplay", str(path)]]
        else:
            players = [
                ["aplay", "-q", str(path)],
                ["paplay", str(path)],
                ["play", "-q", str(path)],
            ]
        tried: list[str] = []
        for argv in players:
            exe = shutil.which(argv[0])
            if exe is None:
                tried.append(f"{argv[0]} (not on PATH)")
                continue
            try:
                done = subprocess.run(  # noqa: S603 -- a fixed player and a checked local path.
                    [exe, *argv[1:]],
                    capture_output=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                tried.append(f"{argv[0]} ({e})")
                continue
            if done.returncode == 0:
                return ""
            tried.append(f"{argv[0]} (exit {done.returncode})")
        return "no audio backend worked: " + "; ".join(tried)


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
