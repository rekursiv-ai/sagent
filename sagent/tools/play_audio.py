"""PlayAudio tool: play a WAV file on the host, cross-platform.

Primary use case is AGENTS.md's completion-notification directive -
the agent rings a short WAV to get the user's attention when a task
finishes or it needs input. Works without external deps: each
platform's native audio command is shelled out via subprocess, or
Python's stdlib :mod:`winsound` on Windows.

Format support is WAV only. Other formats would require ``ffmpeg``
or a playback lib; a deliberate non-feature here.

Silent no-op (logged, tool returns success) when no audio subsystem
is reachable - typical for headless SSH / CI / container runs. The
agent shouldn't fail a model request because the host can't beep.
"""

from __future__ import annotations

from pathlib import Path

import importlib
import logging
import platform
import shutil
import subprocess

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SEC = 10.0


class PlayAudio:
    """Tool: play a WAV file on the host's audio output."""

    name: str = "PlayAudio"
    tool_id: str = "application/x-tool-playaudio"
    description: str = load_tool_description("PlayAudio")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a .wav file (absolute or cwd-relative).",
                },
            },
            "required": ["path"],
        }
    )

    def summary(self, msg: Message) -> str:
        """Return a short label showing the WAV filename.

        Args:
          msg: Tool call message.

        Returns:
          label: "PlayAudio <filename>".

        """
        directive = get_directive(msg)
        path = str(directive.get("path", ""))
        name = Path(path).name if path else "?"
        return f"PlayAudio {name}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return per-request system prompt text.

        Returns:
          prompt: Always empty for this tool.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Play the specified WAV file on the host.

        Args:
          msg: Tool call message with ``path`` field.

        Returns:
          result: Playback confirmation or error message.

        """
        directive = get_directive(msg)
        path = str(directive.get("path", ""))
        return await run_sync(self._run, parent_id=msg.id, path=path)

    def _run(self, *, path: str) -> str | Message:
        """Validate path and play the WAV file synchronously."""
        if not path:
            return TextMessage("path is required.", "text/x-error")
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(get_tool_state().bash_cwd) / p
        if not p.exists():
            return TextMessage(f"Not found: {p}", "text/x-error")
        if not p.is_file():
            return TextMessage(f"Not a file: {p}", "text/x-error")
        if p.suffix.lower() != ".wav":
            return TextMessage(
                f"Unsupported format {p.suffix!r}; only .wav is supported.",
                "text/x-error",
            )
        err = _play(p)
        if err is not None:
            logger.warning("PlayAudio: %s", err)
            # Audio playback is best-effort; a backend that fails or is
            # absent (headless host) shouldn't fail the tool call.
            return f"Played {p.name} (best effort): {err}"
        return f"Played {p.name}."


def _play(path: Path) -> str | None:
    """Play ``path`` on the host. Returns None on success, else a reason."""
    system = platform.system()
    if system == "Windows":
        return _play_windows(path)
    if system == "Darwin":
        return _play_via_cmd(path, ["afplay"])
    # Linux / other Unix - try ALSA first, then PulseAudio, then sox.
    return _play_via_cmd(path, ["aplay", "-q"], ["paplay"], ["play", "-q"])


def _play_windows(path: Path) -> str | None:
    # Imported lazily so the module still imports on POSIX hosts for
    # static analysis and cross-platform tests.
    try:
        ws = importlib.import_module("winsound")
    except ImportError as e:
        return f"winsound unavailable: {e}"
    try:
        ws.PlaySound(str(path), ws.SND_FILENAME | ws.SND_NODEFAULT)  # ty: ignore[unresolved-attribute] -- dynamic import; ty can't resolve winsound attrs
    except RuntimeError as e:
        return f"winsound.PlaySound failed: {e}"
    return None


def _play_via_cmd(path: Path, *candidates: list[str]) -> str | None:
    """Try each ``[exe, *flags]`` candidate; run the first one on PATH.

    Returns None on success, a reason string otherwise. A missing
    audio subsystem (no command on PATH, or the command exits
    non-zero because no device) is treated as a soft failure.
    """
    tried: list[str] = []
    for argv in candidates:
        exe = shutil.which(argv[0])
        if exe is None:
            tried.append(f"{argv[0]} (not on PATH)")
            continue
        try:
            result = subprocess.run(  # noqa: S603 -- argv built from static candidates + validated path
                [exe, *argv[1:], str(path)],
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT_SEC,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            tried.append(f"{argv[0]} ({e})")
            continue
        if result.returncode == 0:
            return None
        tried.append(f"{argv[0]} (exit {result.returncode})")
    return "no audio backend worked: " + "; ".join(tried)
