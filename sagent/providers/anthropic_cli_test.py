"""Tests for ``providers.anthropic_cli``: wire-format + lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json

import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.custom_types import ModelRequest
from sagent.lib.json import MutableJSON
from sagent.providers.anthropic import Anthropic
from sagent.providers.anthropic_cli import (
    AnthropicCLI,
    _AnthropicCLIModel,
    _build_anthropic_argv,
    _build_model_response,
    _dispatch_stream_event,
    _hash_system,
    _serialize_for_stdin,
    _user_line,
)
from sagent.providers.lib.subproc import Subproc


_CRED_PAYLOAD: dict[str, object] = {
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-test",
        "refreshToken": "sk-ant-ort01-test",
        "expiresAt": 9_000_000_000_000,
        "scopes": ["user:profile", "user:inference"],
        "subscriptionType": "max",
        "rateLimitTier": "default",
    }
}


def _write_creds(tmp_path: Path) -> Path:
    """Write a fake credentials file into ``tmp_path`` and return its path."""
    path = tmp_path / ".credentials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CRED_PAYLOAD), encoding="utf-8")
    return path


def test_from_cli_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` raises ``FileNotFoundError`` if no creds file exists."""
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError, match="no credentials"):
        AnthropicCLI.from_credentials()


def test_from_cli_with_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` returns a configured provider when both creds + CLI exist."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        creds,
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli.shutil.which",
        _which_claude,
    )
    provider = AnthropicCLI.from_credentials()
    assert provider.account is None


def test_from_key_delegates_to_anthropic() -> None:
    """``AnthropicCLI.from_key`` returns a plain ``Anthropic`` for API keys."""
    fallback = AnthropicCLI.from_key("sk-ant-test")
    assert isinstance(fallback, Anthropic)
    assert not isinstance(fallback, AnthropicCLI)


def test_from_cli_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` fails fast if ``claude`` is not on ``PATH``."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        creds,
    )

    def _which_missing(name: str) -> str | None:
        del name
        return None

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli.shutil.which",
        _which_missing,
    )
    with pytest.raises(RuntimeError, match="not on PATH"):
        AnthropicCLI.from_credentials()


def test_model_unknown_raises() -> None:
    """An unsupported model id surfaces with the known-model list."""
    provider = AnthropicCLI()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = provider.model("not-a-claude")


def test_model_resolves_context_tag_to_profile() -> None:
    """``claude-sonnet-4-5+1m`` finds the same profile as ``claude-sonnet-4-5``."""
    provider = AnthropicCLI()
    model = provider.model("claude-sonnet-4-5+1m")
    assert model.model_id == "claude-sonnet-4-5+1m"
    assert model.max_request_tokens == 1_000_000


def test_utility_model_picks_haiku() -> None:
    """``utility_model`` returns the cheapest Claude in ``KNOWN_MODELS``."""
    provider = AnthropicCLI()
    model = provider.utility_model()
    assert model.model_id == "claude-haiku-4-5"


def test_model_capabilities() -> None:
    """The wrapped backend declares the supported-flag surface from the spec."""
    provider = AnthropicCLI()
    model = provider.model("claude-sonnet-4-5")
    assert model.supports_streaming is True
    assert model.supports_thinking is True
    assert model.supports_effort is False
    assert model.supports_cache_control is False
    assert model.supports_context_management is True
    assert model.supports_persistent_retry is False
    assert model.supports_account_auth is True


def test_is_context_overflow_text_markers() -> None:
    """Overflow classification is body-text driven, not status-code driven."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert model.is_context_overflow(RuntimeError("prompt is too long")) is True
    assert model.is_context_overflow(RuntimeError("exceeds context window")) is True
    assert model.is_context_overflow(RuntimeError("network error")) is False


def test_argv_contains_required_flags() -> None:
    """The spawn recipe sets every knob the CLI needs for stream-json mode."""
    argv = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="be brief",
        bridge_url="http://127.0.0.1:1234/mcp",
        bridge_server_name="sagent",
    )
    assert argv[0] == "claude"
    assert "--print" in argv
    assert "stream-json" in argv
    assert "--include-partial-messages" in argv
    assert "--no-session-persistence" in argv
    assert "--disable-slash-commands" in argv
    assert "bypassPermissions" in argv
    mcp_idx = argv.index("--mcp-config")
    cfg = json.loads(argv[mcp_idx + 1])
    assert cfg["mcpServers"]["sagent"]["type"] == "http"
    assert cfg["mcpServers"]["sagent"]["url"].endswith("/mcp")


def test_user_line_text_only() -> None:
    """A plain ``UserMessage`` becomes a single ``content: str`` line."""
    line = _user_line(UserMessage(text="hello"), max_image_dim=8000)
    assert line == {"type": "user", "message": {"role": "user", "content": "hello"}}


def test_serialize_for_stdin_rejects_tool_result() -> None:
    """Tool results never traverse stdin -- the MCP bridge handles them."""
    with pytest.raises(RuntimeError, match="ToolResult in history"):
        _ = _serialize_for_stdin(
            ToolResult(call_id="x", content="done"), max_image_dim=8000
        )


def test_dispatch_stream_event_routes_text_and_thinking() -> None:
    """``content_block_delta`` events fan text/thinking into separate buckets."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    text_event = cast(MutableJSON, {"delta": {"type": "text_delta", "text": "hello"}})
    thinking_event = cast(
        MutableJSON,
        {"delta": {"type": "thinking_delta", "thinking": "reflecting"}},
    )
    _dispatch_stream_event(
        text_event,
        text_parts,
        thinking_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    _dispatch_stream_event(
        thinking_event,
        text_parts,
        thinking_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    assert text_parts == ["hello"]
    assert thinking_parts == ["reflecting"]
    assert text_chunks == ["hello"]
    assert thinking_chunks == ["reflecting"]


def test_dispatch_stream_event_ignores_unknown_delta_types() -> None:
    """A non-text/thinking delta does not perturb the accumulators."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    _dispatch_stream_event(
        cast(MutableJSON, {"delta": {"type": "signature_delta", "signature": "x"}}),
        text_parts,
        thinking_parts,
        on_text=None,
        on_thinking=None,
    )
    assert text_parts == []
    assert thinking_parts == []


def test_build_model_response_sums_model_usage_rows() -> None:
    """Costs sum across all ``modelUsage`` rows (Sonnet + Haiku classifier)."""
    usage_event = cast(
        MutableJSON,
        {
            "type": "result",
            "stop_reason": "end_turn",
            "is_error": False,
            "modelUsage": {
                "claude-sonnet-4-6": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "costUSD": 0.001,
                },
                "claude-haiku-4-5": {
                    "inputTokens": 200,
                    "outputTokens": 10,
                    "costUSD": 0.0002,
                },
            },
            "session_id": "sid-x",
        },
    )
    response = _build_model_response(
        usage_event=usage_event,
        text="reply",
        thinking_parts=["thought"],
        stop_reason="end_turn",
        fallback_message_id="fallback",
    )
    assert response.message.text == "reply"
    assert response.tokens.input_tokens == 300
    assert response.tokens.output_tokens == 60
    assert response.total_cost == pytest.approx(0.0012)
    assert response.stop_reason == "model_finished"
    assert response.message_id == "sid-x"
    assert len(response.message.thinking_blocks) == 1


def test_build_model_response_falls_back_to_total_cost_usd() -> None:
    """``total_cost_usd`` is used when ``modelUsage`` is empty (older CLIs)."""
    usage_event = cast(
        MutableJSON,
        {
            "type": "result",
            "stop_reason": "end_turn",
            "modelUsage": {},
            "total_cost_usd": 0.005,
        },
    )
    response = _build_model_response(
        usage_event=usage_event,
        text="",
        thinking_parts=[],
        stop_reason="end_turn",
        fallback_message_id="m",
    )
    assert response.total_cost == pytest.approx(0.005)


def test_hash_system_stable_and_distinguishes() -> None:
    """``_hash_system`` is deterministic and distinguishes different prompts."""
    a = _hash_system("be brief")
    b = _hash_system("be brief")
    c = _hash_system("be verbose")
    d = _hash_system(None)
    assert a == b
    assert a != c
    assert d == _hash_system("")


def test_should_respawn_triggers() -> None:
    """Each respawn trigger from §1.4 is detected on the next request."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert isinstance(model, _AnthropicCLIModel)

    class _DummyActive:
        pass

    # Pretend the hot spare has a live active subprocess; the test only
    # needs ``_active`` to be non-None to exercise the respawn branch.
    model._hot_spare._active = cast(Subproc, _DummyActive())
    user = UserMessage(text="hi")
    request = ModelRequest(messages=[user])

    # System hash mismatch -> respawn.
    model._system_hash = _hash_system("different")
    assert model._should_respawn(request) is True

    # Sync hash; head mismatch -> respawn.
    model._system_hash = _hash_system(None)
    model._sent_history_head = UserMessage(text="something else")
    assert model._should_respawn(request) is True

    # Sync head; turn-count safety valve -> respawn.
    model._sent_history_head = user
    model._turn_count = 200
    assert model._should_respawn(request) is True

    # Sync turn-count; input-token safety valve -> respawn.
    model._turn_count = 0
    model._last_input_tokens = model._max_request_tokens
    assert model._should_respawn(request) is True

    # All clear -> no respawn.
    model._last_input_tokens = 0
    assert model._should_respawn(request) is False


def test_should_respawn_skips_when_no_active() -> None:
    """Before the first acquire, the model has no active and skips respawn."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert model._should_respawn(ModelRequest(messages=[UserMessage(text="hi")])) is (
        False
    )


def test_serialize_for_stdin_user_passthrough() -> None:
    """``UserMessage`` falls through ``_serialize_for_stdin`` to ``_user_line``."""
    line = _serialize_for_stdin(UserMessage(text="ping"), max_image_dim=8000)
    assert line["type"] == "user"


_ = AssistantMessage  # touch the optional import so static checks pass
_ = ToolCall


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
