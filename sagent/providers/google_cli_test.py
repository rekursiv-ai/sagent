"""Tests for ``providers.google_cli``: wire-format + lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import asyncio
import inspect
import json
import os
import time

import pytest

from sagent.lib.json import MutableJSON
from sagent.providers import google_cli
from sagent.providers.google import Google
from sagent.providers.google_cli import (
    GoogleCLI,
    _dispatch_session_update,
    _GoogleCLIModel,
    _GoogleCLIProcState,
    _hash_system,
    _read_expiry,
    _serialize_prompt_blocks,
    _user_prompt_blocks,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)


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


def test_google_cli_does_not_import_subscription_provider() -> None:
    source = inspect.getsource(google_cli)
    assert "providers.google_sub" not in source


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


def test_from_cli_rejects_malformed_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    creds = tmp_path / "oauth_creds.json"
    creds.write_text("", encoding="utf-8")
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
    with pytest.raises(ValueError, match="Invalid credentials"):
        GoogleCLI.from_credentials()


def test_from_cli_rejects_credentials_missing_oauth_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    creds = tmp_path / "oauth_creds.json"
    creds.write_text(json.dumps({}), encoding="utf-8")
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
    with pytest.raises(ValueError, match="Invalid credentials"):
        GoogleCLI.from_credentials()


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


def test_default_model_inherits_from_google() -> None:
    """``GoogleCLI`` defers to ``Google.DEFAULT_MODEL``."""
    assert GoogleCLI.DEFAULT_MODEL == Google.DEFAULT_MODEL


def test_default_utility_model_inherits_from_google() -> None:
    """``GoogleCLI`` defers to ``Google.DEFAULT_UTILITY_MODEL``."""
    assert GoogleCLI.DEFAULT_UTILITY_MODEL == Google.DEFAULT_UTILITY_MODEL


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


def test_google_cli_legacy_model_does_not_support_thinking() -> None:
    """Legacy ``gemini-1.5-*`` snapshots can't accept ``thinkingConfig``.

    Even via ACP the CLI cannot enable thinking on these models, so capability
    advertisement must honor the per-model profile flag.
    """
    provider = GoogleCLI()
    assert provider.model("gemini-1.5-flash").supports_thinking is False
    assert provider.model("gemini-1.5-pro").supports_thinking is False


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


def test_approx_request_tokens_sums_user_and_assistant() -> None:
    """The walker counts both user and assistant text entries."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    request = ModelRequest(
        messages=[
            UserMessage(text="aaaa" * 4),
            AssistantMessage(text="bbbb"),
            UserMessage(text="cccc"),
        ],
    )
    estimate = model.approx_request_tokens(request)
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

    model._hot_spare._active = cast(Subproc, _DummyActive())
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


@pytest.mark.asyncio
async def test_stream_eof_respawns_and_resets_sent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess EOF is transport failure and resets Google send state."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    respawn_count = 0

    class _DeadProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> None:
            del skip_non_json

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _DeadProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _DeadProc())

    model._system_hash = _hash_system(None)
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])
    with pytest.raises(Exception, match="stdout closed"):
        await model.stream(request)
    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


@pytest.mark.asyncio
async def test_stream_read_timeout_respawns_and_resets_sent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    respawn_count = 0

    class _StalledProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            raise SubprocessTransportError("subprocess stdout idle timeout")

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _StalledProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _StalledProc())

    model._system_hash = _hash_system(None)
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])
    with pytest.raises(Exception, match="stdout idle timeout"):
        await model.stream(request)
    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


@pytest.mark.asyncio
async def test_stream_repeated_transport_failures_trip_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated subprocess transport failures stop respawning indefinitely."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    respawn_count = 0

    class _DeadProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> None:
            del skip_non_json

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _DeadProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            if respawn_count == 2:
                raise RuntimeError("transport failure budget exhausted")
            return cast(Subproc, _DeadProc())

    model._system_hash = _hash_system(None)
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])

    with pytest.raises(Exception, match="stdout closed"):
        await model.stream(request)
    with pytest.raises(RuntimeError, match="transport failure budget exhausted"):
        await model.stream(request)

    assert respawn_count == 2
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


def test_should_respawn_skips_when_no_active() -> None:
    """Before the first acquire, the model has no active and skips respawn."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert model._should_respawn(ModelRequest(messages=[UserMessage(text="hi")])) is (
        False
    )


@pytest.mark.asyncio
async def test_stream_system_change_discards_warmed_old_system_spare() -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    spawned_systems: list[str] = []
    used_systems: list[str] = []
    warmed = asyncio.Event()

    class _Proc:
        async def close(self) -> None:
            pass

    async def spawn_initialized() -> Subproc:
        proc = cast(Subproc, _Proc())
        spawned_systems.append(model._pending_system)
        if len(spawned_systems) == 2:
            warmed.set()
        state = _GoogleCLIProcState(
            proc=proc,
            session_id=f"session-{model._pending_system}",
            tmpdir=Path.cwd(),
            system_hash=_hash_system(model._pending_system),
        )
        _GoogleCLIModel._attach_proc_state(state)
        return proc

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, text_parts, thinking_parts, on_text, on_thinking
        used_systems.append(model._system_hash)
        return "STOP"

    model._send_prompt = send_prompt  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.
    model._hot_spare = HotSpare(spawn_initialized)

    _ = await model.stream(ModelRequest(messages=[UserMessage(text="a")], system="A"))
    await warmed.wait()
    _ = await model.stream(ModelRequest(messages=[UserMessage(text="b")], system="B"))

    assert used_systems == [_hash_system("A"), _hash_system("B")]
    assert spawned_systems == ["A", "A", "B"]
    await model.close()


@pytest.mark.asyncio
async def test_hot_spare_warmup_does_not_overwrite_active_session_id() -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    ready = asyncio.Event()
    warmed = asyncio.Event()

    class _DummyProc:
        session_ids: list[str]

        def __init__(self) -> None:
            self.session_ids = []

        async def close(self) -> None:
            pass

    async def spawn_initialized() -> Subproc:
        proc = cast(Subproc, _DummyProc())
        session_id = "active-session"
        if not ready.is_set():
            ready.set()
        else:
            session_id = "spare-session"
            warmed.set()
        state = _GoogleCLIProcState(
            proc=proc,
            session_id=session_id,
            tmpdir=Path.cwd(),
            system_hash=_hash_system(None),
        )
        _GoogleCLIModel._attach_proc_state(state)
        return proc

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del prompt_blocks, text_parts, thinking_parts, on_text, on_thinking
        cast(_DummyProc, proc).session_ids.append(model._session_id)
        return "STOP"

    model._send_prompt = send_prompt  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.
    model._hot_spare = HotSpare(spawn_initialized)
    proc = await model._hot_spare.acquire()
    await warmed.wait()
    response = await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))
    assert cast(_DummyProc, proc).session_ids == ["active-session"]
    assert response.message_id == "active-session"
    await model.close()


@pytest.mark.asyncio
async def test_exchange_turn_skips_assistant_replay() -> None:
    """Respawn replay sends only user-like entries to the CLI subprocess."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    prompts: list[list[MutableJSON]] = []

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del proc, text_parts, thinking_parts, on_text, on_thinking
        prompts.append(prompt_blocks)
        return "STOP"

    model._send_prompt = send_prompt  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.
    response = await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(
            messages=[
                UserMessage(text="first"),
                AssistantMessage(text="hidden"),
                UserMessage(text="second"),
            ]
        ),
        on_text=None,
        on_thinking=None,
    )

    assert [[block["text"] for block in prompt] for prompt in prompts] == [
        ["first"],
        ["second"],
    ]
    assert response.stop_reason == "model_finished"


@pytest.mark.asyncio
async def test_stream_writeback_failure_returns_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    success_count = 0

    class _Proc:
        pass

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _Proc())

        def record_success(self) -> None:
            nonlocal success_count
            success_count += 1

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, thinking_parts, on_text, on_thinking
        text_parts.append("ok")
        return "STOP"

    async def writeback_credentials() -> None:
        raise OSError("credential writeback failed")

    user = UserMessage(text="hi")
    model._system_hash = _hash_system(None)
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    monkeypatch.setattr(model, "_writeback_credentials", writeback_credentials)

    response = await model.stream(ModelRequest(messages=[user]))

    assert response.message.text == "ok"
    assert response.message_id == "session"
    assert success_count == 1
    assert model._last_sent_index == 1
    assert model._sent_history_head is user
    assert model._turn_count == 1


@pytest.mark.asyncio
async def test_respawn_resets_active_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    model._turn_count = 100
    model._last_input_tokens = model._max_request_tokens

    class _Proc:
        pass

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _Proc())

        async def respawn(self) -> Subproc:
            return cast(Subproc, _Proc())

        def record_success(self) -> None:
            pass

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, text_parts, thinking_parts, on_text, on_thinking
        return "STOP"

    model._system_hash = _hash_system(None)
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_prompt", send_prompt)

    request = ModelRequest(messages=[UserMessage(text="hi")])
    response = await model.stream(request)

    assert response.message_id == "session"
    assert model._turn_count == 1
    assert model._last_input_tokens == 0
    assert model._should_respawn(request) is False


@pytest.mark.asyncio
async def test_exchange_turn_returns_current_output_only() -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    text_callbacks: list[str] = []

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        del proc, thinking_parts, on_thinking
        text = cast(str, prompt_blocks[0]["text"])
        text_parts.append(text)
        if on_text is not None:
            on_text(text)
        return "STOP"

    model._send_prompt = send_prompt  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.
    response = await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(messages=[UserMessage(text="first"), UserMessage(text="current")]),
        on_text=text_callbacks.append,
        on_thinking=None,
    )

    assert response.message.text == "current"
    assert text_callbacks == ["current"]


@pytest.mark.asyncio
async def test_terminal_json_rpc_error_respawns_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    respawn_count = 0

    class _ErrorProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> MutableJSON:
            del skip_non_json
            return cast(MutableJSON, {"id": 1, "error": {"message": "boom"}})

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _ErrorProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _ErrorProc())

    user = UserMessage(text="hi")
    model._system_hash = _hash_system(None)
    model._session_id = "session"
    model._last_sent_index = 0
    model._sent_history_head = user
    model._turn_count = 9
    model._last_input_tokens = 123
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())

    with pytest.raises(SubprocessTransportError, match="JSON-RPC error"):
        _ = await model.stream(ModelRequest(messages=[user]))

    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None
    assert model._turn_count == 0
    assert model._last_input_tokens == 0


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


@pytest.mark.asyncio
async def test_writeback_credentials_atomic_and_0o600(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Writeback is atomic (failure leaves target untouched) and target is ``0o600``.

    With ``shutil.copyfile`` the destination is created and partially
    written before any failure; a mid-write crash leaves a corrupt
    file. ``atomic_write_bytes`` writes to a tmp sibling and renames,
    so a mid-write crash never disturbs ``target``.
    """
    monkeypatch.setattr(
        "sagent.providers.google_cli._CREDS_PATH",
        tmp_path / "home" / ".gemini" / "oauth_creds.json",
    )
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    src_dir = tmp_path / "sandbox" / ".gemini"
    src_dir.mkdir(parents=True)
    src = src_dir / "oauth_creds.json"
    src.write_text(json.dumps(_CRED_PAYLOAD), encoding="utf-8")
    model._tmpdir = tmp_path / "sandbox"
    target = tmp_path / "home" / ".gemini" / "oauth_creds.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    sentinel = json.dumps({"existing": "do not clobber"})
    target.write_text(sentinel, encoding="utf-8")
    target.chmod(0o600)
    later = (time.time() + 7200) * 1000.0
    src.write_text(
        json.dumps({**_CRED_PAYLOAD, "expiry_date": later}), encoding="utf-8"
    )

    real_open = os.open

    def _flaky_open(path: str, flags: int, mode: int = 0o777) -> int:
        if str(path).startswith(str(target)):
            raise OSError("simulated disk full")
        return real_open(path, flags, mode)

    with monkeypatch.context() as fail_open:
        fail_open.setattr(
            "sagent.lib.atomic_file.os.open",
            _flaky_open,
        )
        with pytest.raises(OSError, match="simulated disk full"):
            await model._writeback_credentials()
    assert target.read_text(encoding="utf-8") == sentinel
    leftover = list(target.parent.glob(f"{target.name}.tmp.*"))
    assert leftover == [], f"non-atomic writeback left tmp files: {leftover}"

    await model._writeback_credentials()
    assert (target.stat().st_mode & 0o777) == 0o600
    assert json.loads(target.read_text(encoding="utf-8"))["expiry_date"] == (
        pytest.approx(later)
    )


def test_writeback_credentials_works_across_event_loops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model whose lock was first used in loop A still works in loop B."""
    monkeypatch.setattr(
        "sagent.providers.google_cli._CREDS_PATH",
        tmp_path / "home" / ".gemini" / "oauth_creds.json",
    )
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert isinstance(model, _GoogleCLIModel)
    src_dir = tmp_path / "sandbox" / ".gemini"
    src_dir.mkdir(parents=True)
    src = src_dir / "oauth_creds.json"
    src.write_text(json.dumps(_CRED_PAYLOAD), encoding="utf-8")
    model._tmpdir = tmp_path / "sandbox"

    asyncio.run(model._writeback_credentials())
    payload = dict(_CRED_PAYLOAD)
    payload["expiry_date"] = (time.time() + 7200) * 1000.0
    src.write_text(json.dumps(payload), encoding="utf-8")
    asyncio.run(model._writeback_credentials())

    target = tmp_path / "home" / ".gemini" / "oauth_creds.json"
    assert json.loads(target.read_text(encoding="utf-8"))["expiry_date"] == (
        pytest.approx(payload["expiry_date"])
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
