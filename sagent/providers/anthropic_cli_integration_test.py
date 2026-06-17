"""Live integration tests for the AnthropicCLI provider.

These spawn the REAL ``claude`` binary and exercise the live
``stream-json`` wire protocol end-to-end -- the one thing the unit
suite (hand-authored fixtures) structurally cannot verify. They are
marked ``integration`` and DESELECTED by default (see ``pyproject.toml``
``addopts``); run explicitly with::

    uv run pytest -m integration sagent/providers/anthropic_cli_integration_test.py

Requirements (else the module skips):
  - ``claude`` on PATH,
  - valid CLI credentials (``claude login`` / ``~/.claude/.credentials.json``).

What they cover (the production-correctness gaps the unit suite leaves):
  - a basic turn produces a non-empty ``ModelResponse``;
  - a bridge-mounted tool round-trips (CLI -> ToolsBridge -> sagent Tool
    -> result -> model uses it);
  - session mode resumes: turn 1 mints ``--session-id``, turn 2
    ``--resume``s and sees turn-1 context.
"""

from __future__ import annotations

from pathlib import Path

import shutil
import uuid as _uuid

import pytest

from sagent.providers.anthropic_cli import AnthropicCLI
from sagent.providers.lib.oauth import credentials_path
from sagent.tools import tool
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


pytestmark = pytest.mark.integration


def _claude_available() -> bool:
    if shutil.which("claude") is None:
        return False
    creds = credentials_path(Path.home() / ".claude" / ".credentials.json", None)
    return creds.exists()


_SKIP_REASON = "requires `claude` on PATH + CLI credentials"
_requires_claude = pytest.mark.skipif(not _claude_available(), reason=_SKIP_REASON)

# The MCP bridge binds a local HTTP port; sandboxed/CI environments may
# forbid that. Treat a bridge-startup failure as an environment skip, not
# a provider failure -- these tests assert provider behavior, not that the
# host can bind loopback sockets.
_BRIDGE_UNAVAILABLE = "MCP bridge could not start in this environment"


def _skip_if_bridge_unavailable(exc: Exception) -> None:
    msg = str(exc)
    if isinstance(exc, TimeoutError) and "bridge" in msg.lower():
        pytest.skip(f"{_BRIDGE_UNAVAILABLE}: {msg}")


@_requires_claude
@pytest.mark.asyncio
async def test_basic_turn_returns_response() -> None:
    """A real ``claude --print`` turn yields a non-empty assistant message."""
    model = AnthropicCLI.from_credentials().model("claude-haiku-4-5")
    try:
        response = await model.stream(
            ModelRequest(
                messages=[UserMessage(text="Reply with exactly the word: pong")],
            ),
        )
    except TimeoutError as exc:
        _skip_if_bridge_unavailable(exc)
        raise
    finally:
        await model.close()
    assert response.message.text.strip(), "expected non-empty assistant text"


@_requires_claude
@pytest.mark.asyncio
async def test_session_resume_two_turns() -> None:
    """Turn 1 mints the session; turn 2 ``--resume``s it and recalls context.

    Verifies the live ``--session-id`` -> ``--resume`` lifecycle: the
    second turn must see the fact established in the first (proving the
    on-disk session was resumed, not started fresh).
    """
    sid = str(_uuid.uuid4())
    model = AnthropicCLI.from_credentials().model("claude-haiku-4-5", session_id=sid)
    try:
        await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(
                        text="Remember this codeword: BARRACUDA. Reply 'ok'.",
                    ),
                ],
            ),
        )
        assert model._session_initialized is True, "turn 1 must establish the session"

        response = await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(text="Remember this codeword: BARRACUDA. Reply 'ok'."),
                    # turn-1 assistant reply is on disk; only the new
                    # entry is fed via stdin.
                    UserMessage(
                        text="What was the codeword? Reply with just the word."
                    ),
                ],
            ),
        )
    except TimeoutError as exc:
        _skip_if_bridge_unavailable(exc)
        raise
    finally:
        await model.close()
    assert "BARRACUDA" in response.message.text.upper(), (
        f"resumed turn did not recall turn-1 context; got {response.message.text!r}"
    )


@_requires_claude
@pytest.mark.asyncio
async def test_bridge_tool_round_trips() -> None:
    """A bridge-mounted tool is dispatched by the real CLI and its result used.

    Exercises the full tool path: CLI emits a ``tool_use`` -> sagent's
    ``ToolsBridge`` runs the sagent ``Tool`` -> result returns -> the
    model incorporates it. Uses a trivial echo-style tool so the model
    has a single obvious action.
    """

    @tool(name="magic_word")
    def magic_word() -> str:
        """Return the secret magic word. Call it to learn the word."""
        return "FLAMINGO"

    model = AnthropicCLI.from_credentials().model("claude-haiku-4-5")
    try:
        response = await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(
                        text=(
                            "Call the magic_word tool, then reply with just the "
                            "word it returns."
                        ),
                    ),
                ],
                tools=[magic_word],
            ),
        )
    except TimeoutError as exc:
        _skip_if_bridge_unavailable(exc)
        raise
    finally:
        await model.close()
    assert "FLAMINGO" in response.message.text.upper(), (
        f"model did not use the tool result; got {response.message.text!r}"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
