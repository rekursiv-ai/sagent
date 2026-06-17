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

import asyncio
import shutil
import uuid as _uuid

import pytest

from sagent.providers.anthropic_cli import AnthropicCLI
from sagent.providers.lib.oauth import credentials_path
from sagent.tools import tool
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


# ``real_sleep``: these spawn the real CLI + the MCP bridge's uvicorn
# server, whose startup polls ``asyncio.sleep`` to yield to the serve
# task. The conftest's autouse ``_fast_sleep`` no-ops ``asyncio.sleep``,
# which starves uvicorn startup and trips the bridge's 10s deadline.
pytestmark = [pytest.mark.integration, pytest.mark.real_sleep]


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

    Uses sonnet, not haiku: haiku is unreliable at electing to use an
    MCP-mounted tool (observed ~1/3 of cold runs "searching" for it and
    declaring it absent instead of calling it). That is a model-capability
    fact, not a bridge defect -- the bridge dispatch is identical either
    way -- so this test pins the more capable model to stay deterministic.
    """

    @tool(name="magic_word")
    def magic_word() -> str:
        """Return the secret magic word. Call it to learn the word."""
        return "FLAMINGO"

    model = AnthropicCLI.from_credentials().model("claude-sonnet-4-6")
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


@_requires_claude
@pytest.mark.asyncio
async def test_detached_result_delivered_to_model_on_resume() -> None:
    """A completed detached tool result is delivered to the model on the
    next ``--resume`` turn and the model can read it.

    Tests the Model side of the bridge's internal cohort -- the live path
    that ``_detached_delivery_entry`` folds a finished background run into
    the next turn's input and the resumed CLI surfaces it to the model.
    Turn 1's detach is staged DIRECTLY through the bridge (``background:
    true``) rather than via a real ``claude`` turn: whether haiku elects
    to background a tool is model non-determinism already covered by the
    unit tests; what needs live verification is that the staged result
    rides ``--resume`` into the model's context. Keeps the test fast and
    deterministic.
    """

    @tool(name="slow_oracle")
    async def slow_oracle() -> str:
        """Return the oracle's secret word."""
        return "PELICAN"

    sid = str(_uuid.uuid4())
    model = AnthropicCLI.from_credentials().model("claude-haiku-4-5", session_id=sid)
    try:
        # Establish the session + bridge with one trivial real turn.
        await model.stream(
            ModelRequest(
                messages=[UserMessage(text="Reply with the word: ready")],
                tools=[slow_oracle],
            ),
        )
        bridge = model._tools_bridge
        assert bridge is not None, "first turn must have created the bridge"

        # Stage a detached run directly through the bridge (the same call
        # the CLI makes when the model passes ``background: true``); wait
        # for it to finish so a real result is queued for delivery.
        blocks = await bridge._call_tool("slow_oracle", {"background": True})
        assert "[detached" in str(blocks[0])
        for _ in range(100):
            if not bridge.has_pending_detached():
                break
            await asyncio.sleep(0.02)
        assert not bridge.has_pending_detached(), "detached task never completed"

        # Next turn: ``_detached_delivery_entry`` folds the finished result
        # into the resumed turn's input, so the model sees PELICAN.
        response = await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(text="Reply with the word: ready"),
                    UserMessage(
                        text=(
                            "A detached tool result was just delivered to you. "
                            "What word did it contain? Reply with just that word."
                        ),
                    ),
                ],
                tools=[slow_oracle],
            ),
        )
    except TimeoutError as exc:
        _skip_if_bridge_unavailable(exc)
        raise
    finally:
        await model.close()
    assert "PELICAN" in response.message.text.upper(), (
        f"model never received the detached result; got {response.message.text!r}"
    )


@_requires_claude
@pytest.mark.asyncio
async def test_real_claude_drives_full_detach_path() -> None:
    """End-to-end proof that REAL ``claude`` drives the bridge's detach
    path on its own -- the link the staged test deliberately skips.

    Turn 1: claude is handed the bg-augmented schema and asked to run the
    tool in the background; it must emit ``background: true``, the bridge
    returns the ``[detached: ...]`` placeholder, and claude ends the turn
    on it ("comes up for air") WITHOUT the answer. Turn 2: the finished
    result rides ``--resume`` and claude reads it.

    Uses sonnet, not haiku: haiku is unreliable at electing to background
    a tool (observed running it inline or hallucinating it absent),
    which is a model-capability fact, not a provider defect. This is the
    committed analogue of the standalone MCP experiment.
    """

    @tool(name="slow_oracle")
    async def slow_oracle() -> str:
        """Return the oracle's secret word. Slow; prefer backgrounding it."""
        await asyncio.sleep(0.5)
        return "PELICAN"

    sid = str(_uuid.uuid4())
    model = AnthropicCLI.from_credentials().model("claude-sonnet-4-6", session_id=sid)
    try:
        # Turn 1: claude must background the tool and end the turn on the
        # detached placeholder, with no answer yet.
        turn1 = await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(
                        text=(
                            "Call the slow_oracle tool, passing background set "
                            "to true so it runs without blocking. Do NOT wait "
                            "for its result. Just confirm you started it."
                        ),
                    ),
                ],
                tools=[slow_oracle],
            ),
        )
        bridge = model._tools_bridge
        assert bridge is not None, "turn 1 must have created the bridge"
        for _ in range(200):
            if not bridge.has_pending_detached():
                break
            await asyncio.sleep(0.05)
        assert not bridge.has_pending_detached(), "detached task never completed"
        # Real claude actually drove the detach: a background task ran and
        # produced a result. If empty, claude declined to background -- the
        # very behavior this test exists to verify, surfaced clearly.
        assert bridge._bg_done, (
            "real claude did not background the tool (ran it inline or "
            f"skipped it); turn-1 text was {turn1.message.text!r}"
        )
        # And it did NOT already have the answer (it came up for air).
        assert "PELICAN" not in turn1.message.text.upper(), (
            f"turn 1 should not contain the answer yet; got {turn1.message.text!r}"
        )

        # Turn 2: the detached result rides --resume; claude reads it.
        turn2 = await model.stream(
            ModelRequest(
                messages=[
                    UserMessage(
                        text=(
                            "Call the slow_oracle tool, passing background set "
                            "to true so it runs without blocking. Do NOT wait "
                            "for its result. Just confirm you started it."
                        ),
                    ),
                    UserMessage(
                        text=(
                            "What word did the slow_oracle tool return? It was "
                            "delivered to you as a detached tool result. Reply "
                            "with just that word."
                        ),
                    ),
                ],
                tools=[slow_oracle],
            ),
        )
    except TimeoutError as exc:
        _skip_if_bridge_unavailable(exc)
        raise
    finally:
        await model.close()
    assert "PELICAN" in turn2.message.text.upper(), (
        f"model never received the detached result; got {turn2.message.text!r}"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
