"""Tests for ``providers.google_cli``: wire-format + lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import time

import pytest

from sagent.agent.runtime import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.custom_types import ModelRequest
from sagent.lib.json import MutableJSON
from sagent.providers.google import Google
from sagent.providers.google_cli import (
    GoogleCLI,
    _dispatch_session_update,
    _estimate_input_tokens,
    _GoogleCLIModel,
    _hash_system,
    _read_expiry,
    _serialize_prompt_blocks,
    _user_prompt_blocks,
)
from sagent.providers.lib.subproc import _Subproc


_CRED_PAYLOAD: dict[str, object] = {
    "access_token": "test-at",
    "refresh_token": "test-rt",
    "expiry_date": (time.time() + 3600) * 1000.0,
}


def _write_creds(tmp_path: Path) -> Path:
    """Write a fake credentials file into ``tmp_path`` and return its path."""
    path = tmp_path / "oauth_creds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CRED_PAYLOAD), encoding="utf-8")
    return path


def test_from_cli_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` raises ``FileNotFoundError`` if no creds file exists."""
    monkeypatch.setattr(
        "sagent.providers.google_cli._CREDS_PATH",
        tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError, match="no credentials"):
        GoogleCLI.from_credentials()


def test_from_cli_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` fails fast if ``gemini`` is not on ``PATH``."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.google_cli._CREDS_PATH",
        creds,
    )

    def _which_missing(name: str) -> str | None:
        del name
        return None

    monkeypatch.setattr(
        "sagent.providers.google_cli.shutil.which",
        _which_missing,
    )
    with pytest.raises(RuntimeError, match="not on PATH"):
        GoogleCLI.from_credentials()


def test_from_cli_with_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` returns a configured provider when both creds + CLI exist."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.google_cli._CREDS_PATH",
        creds,
    )

    def _which_gemini(name: str) -> str | None:
        del name
        return "/usr/bin/gemini"

    monkeypatch.setattr(
        "sagent.providers.google_cli.shutil.which",
        _which_gemini,
    )
    provider = GoogleCLI.from_credentials()
    assert provider.account is None
    assert provider.api_key == ""


def test_from_key_delegates_to_google() -> None:
    """``GoogleCLI.from_key`` returns a plain ``Google`` provider for API keys."""
    fallback = GoogleCLI.from_key("AIza-not-real")
    assert isinstance(fallback, Google)
    assert not isinstance(fallback, GoogleCLI)


def test_model_unknown_raises() -> None:
    """An unsupported model id surfaces with the known-model list."""
    provider = GoogleCLI()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = provider.model("not-a-gemini")


def test_model_uses_default_when_unset() -> None:
    """``provider.model()`` picks ``DEFAULT_MODEL``."""
    provider = GoogleCLI()
    model = provider.model()
    assert model.model_id == GoogleCLI.DEFAULT_MODEL


def test_utility_model_picks_flash_lite() -> None:
    """``utility_model`` returns the cheapest Gemini in ``KNOWN_MODELS``."""
    provider = GoogleCLI()
    model = provider.utility_model()
    assert model.model_id == GoogleCLI.DEFAULT_UTILITY_MODEL


def test_model_capabilities() -> None:
    """The wrapped backend declares the supported-flag surface from the spec."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert model.supports_streaming is True
    assert model.supports_thinking is True
    assert model.supports_effort is False
    assert model.supports_cache_control is False
    assert model.supports_context_management is True
    assert model.supports_account_auth is True


def test_max_image_limits() -> None:
    """Vision limits match the Gemini API recipe."""
    model = GoogleCLI().model("gemini-2.5-flash")
    assert model.max_image_dim == 3072
    assert model.max_image_bytes == 20 * 1024 * 1024


def test_is_context_overflow_text_markers() -> None:
    """Overflow classification covers Gemini's documented overflow strings."""
    model = GoogleCLI().model("gemini-2.5-flash")
    assert model.is_context_overflow(RuntimeError("Input too long")) is True
    assert model.is_context_overflow(RuntimeError("exceeds the maximum")) is True
    assert model.is_context_overflow(RuntimeError("other")) is False


def test_user_prompt_blocks_text_only() -> None:
    """Plain text turns into a single ``{type:text}`` block."""
    blocks = _user_prompt_blocks(UserMessage(text="hi"), max_image_dim=3072)
    assert blocks == [{"type": "text", "text": "hi"}]


def test_user_prompt_blocks_empty_user_emits_placeholder() -> None:
    """An empty user message still emits one block to satisfy ACP shape."""
    blocks = _user_prompt_blocks(UserMessage(text=""), max_image_dim=3072)
    assert blocks == [{"type": "text", "text": ""}]


def test_serialize_prompt_blocks_rejects_tool_result() -> None:
    """Tool results never traverse stdin -- the MCP bridge handles them."""
    with pytest.raises(RuntimeError, match="ToolResult in history"):
        _ = _serialize_prompt_blocks(
            ToolResult(call_id="x", content="done"), max_image_dim=3072
        )


def test_dispatch_session_update_routes_text_and_thinking() -> None:
    """``agent_message_chunk`` / ``agent_thought_chunk`` fan to the callbacks."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []

    _dispatch_session_update(
        cast(
            MutableJSON,
            {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "hello"},
                }
            },
        ),
        text_parts,
        thinking_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    _dispatch_session_update(
        cast(
            MutableJSON,
            {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "thinking..."},
                }
            },
        ),
        text_parts,
        thinking_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    assert text_parts == ["hello"]
    assert thinking_parts == ["thinking..."]
    assert text_chunks == ["hello"]
    assert thinking_chunks == ["thinking..."]


def test_dispatch_session_update_ignores_unknown_kinds() -> None:
    """``tool_call_update`` and other kinds are dropped without side effects."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    _dispatch_session_update(
        cast(
            MutableJSON,
            {"update": {"sessionUpdate": "tool_call_update", "id": 1}},
        ),
        text_parts,
        thinking_parts,
        on_text=None,
        on_thinking=None,
    )
    assert text_parts == []
    assert thinking_parts == []


def test_estimate_input_tokens_sums_user_and_assistant() -> None:
    """Token estimation walks both user and assistant entries."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    history: list[HistoryEntry] = [
        UserMessage(text="aaaa" * 4),
        AssistantMessage(text="bbbb"),
        UserMessage(text="cccc"),
    ]
    estimate = _estimate_input_tokens(history, model)
    # ``len("a"*16) // 4 = 4`` + ``4 // 4 = 1`` + ``4 // 4 = 1`` = 6.
    assert estimate == 6


def test_hash_system_stable_and_distinguishes() -> None:
    """``_hash_system`` matches the AnthropicCLI helper's stability guarantees."""
    a = _hash_system("be brief")
    b = _hash_system("be brief")
    c = _hash_system("be verbose")
    d = _hash_system(None)
    assert a == b
    assert a != c
    assert d == _hash_system("")


def test_read_expiry_handles_missing_and_invalid(tmp_path: Path) -> None:
    """``_read_expiry`` returns ``0`` for missing, malformed, or non-numeric files."""
    target = tmp_path / "creds.json"
    target.write_text("{}", encoding="utf-8")
    assert _read_expiry(target) == 0.0
    target.write_text("not json", encoding="utf-8")
    assert _read_expiry(target) == 0.0
    target.write_text(json.dumps({"expiry_date": 12345.0}), encoding="utf-8")
    assert _read_expiry(target) == pytest.approx(12345.0)


def test_should_respawn_triggers() -> None:
    """Each respawn trigger from §1.4 is detected on the next request."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)

    class _DummyActive:
        pass

    model._hot_spare._active = cast(_Subproc, _DummyActive())
    user = UserMessage(text="hi")
    request = ModelRequest(messages=[user])

    model._system_hash = _hash_system("different")
    assert model._should_respawn(request) is True

    model._system_hash = _hash_system(None)
    model._sent_history_head = UserMessage(text="other")
    assert model._should_respawn(request) is True

    model._sent_history_head = user
    model._turn_count = 200
    assert model._should_respawn(request) is True

    model._turn_count = 0
    model._last_input_tokens = model._max_request_tokens
    assert model._should_respawn(request) is True

    model._last_input_tokens = 0
    assert model._should_respawn(request) is False


def test_should_respawn_skips_when_no_active() -> None:
    """Before the first acquire, the model has no active and skips respawn."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert model._should_respawn(ModelRequest(messages=[UserMessage(text="hi")])) is (
        False
    )


def test_build_response_estimates_tokens_and_cost() -> None:
    """The Code Assist response is reconstructed with estimated usage + cost."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    request = ModelRequest(messages=[UserMessage(text="aaaa" * 4)])
    response = model._build_response(
        text_parts=["hello"],
        thinking_parts=["thinking..."],
        stop_reason="STOP",
        request=request,
    )
    assert response.message.text == "hello"
    assert response.stop_reason == "model_finished"
    assert response.tokens.input_tokens == 4
    assert response.tokens.output_tokens == 1
    assert response.total_cost >= 0.0
    assert len(response.message.thinking_blocks) == 1


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
