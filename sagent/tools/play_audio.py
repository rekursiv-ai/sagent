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

from collections.abc import Mapping
from pathlib import Path

import importlib
import logging
import platform
import shutil
import subprocess

from sagent.lib.json import JSON, json_freeze
from sagent.tools.core import (
    get_tool_state,
    load_tool_description,
    run_sync,
)
from sagent.types.history import ToolResult


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

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this audio-playback invocation.

        Args:
          args: Directive carrying the ``path`` string.

        Returns:
          label: ``PlayAudio <basename>`` line shown before invocation.

        """
        path = str(args.get("path", ""))
        name = Path(path).name if path else "?"
        return f"PlayAudio {name}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for PlayAudio.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for PlayAudio.

        Returns:
          contribution: Empty string.

        """
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Play a WAV file on the host's audio output.

        Args:
          args: Directive carrying the ``path`` to a ``.wav`` file.

        Returns:
          result: Confirmation message, or a non-error best-effort line
              when no audio subsystem is reachable.

        """
        path = str(args.get("path", ""))
        return await run_sync(self._run, path=path)

    def _run(self, *, path: str) -> str | ToolResult:
        """Validate the path and dispatch to the platform-specific player."""
        if not path:
            return ToolResult(call_id="", content="path is required.", is_error=True)
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(get_tool_state().bash_cwd) / p
        if not p.exists():
            return ToolResult(call_id="", content=f"Not found: {p}", is_error=True)
        if not p.is_file():
            return ToolResult(call_id="", content=f"Not a file: {p}", is_error=True)
        if p.suffix.lower() != ".wav":
            return ToolResult(
                call_id="",
                content=(f"Unsupported format {p.suffix!r}; only .wav is supported."),
                is_error=True,
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
    """Play ``path`` via ``winsound`` on Windows hosts."""
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
