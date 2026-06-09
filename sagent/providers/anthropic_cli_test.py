"""Tests for ``providers.anthropic_cli``: wire-format + lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import asyncio
import inspect
import json

import pytest

from sagent.lib.json import MutableJSON
from sagent.providers import anthropic_cli
from sagent.providers.anthropic import Anthropic
from sagent.providers.anthropic_cli import (
    AnthropicCLI,
    _AnthropicCLIModel,
    _build_anthropic_argv,
    _build_model_response,
    _dispatch_stream_event,
    _hash_system,
    _round_context_tokens,
    _serialize_for_stdin,
    _user_line,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


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


def test_anthropic_cli_does_not_import_subscription_provider() -> None:
    source = inspect.getsource(anthropic_cli)
    assert "providers.anthropic_sub" not in source


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


def test_from_cli_rejects_malformed_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    creds = tmp_path / ".credentials.json"
    creds.write_text("", encoding="utf-8")
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
    with pytest.raises(ValueError, match="Invalid credentials"):
        AnthropicCLI.from_credentials()


def test_from_cli_rejects_credentials_missing_oauth_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps({"claudeAiOauth": {}}), encoding="utf-8")
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
    with pytest.raises(ValueError, match="Invalid credentials"):
        AnthropicCLI.from_credentials()


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


def test_default_model_inherits_from_anthropic() -> None:
    """``AnthropicCLI`` defers to ``Anthropic.DEFAULT_MODEL``.

    Vendor-base classes (``Anthropic``, ``Google``, ``OpenAI``) own the
    catalog; auth subclasses (``CLI`` / ``Subscription``) should not
    fork the default unless the transport genuinely demands it.
    """
    assert AnthropicCLI.DEFAULT_MODEL == Anthropic.DEFAULT_MODEL


def test_default_utility_model_inherits_from_anthropic() -> None:
    """``AnthropicCLI`` defers to ``Anthropic.DEFAULT_UTILITY_MODEL``."""
    assert AnthropicCLI.DEFAULT_UTILITY_MODEL == Anthropic.DEFAULT_UTILITY_MODEL


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


def test_build_model_response_normalizes_input_to_last_round() -> None:
    """Input-side tokens report the LAST round's context, not the sum.

    One ``claude --print`` turn = N internal API rounds; the terminal
    ``result`` usage sums input + cache across ALL of them, over-counting
    the live context window by ~the round count (live 2026-06-09: a
    69-round turn summed to 5.6M against a 200k window, spuriously
    tripping the Agent's proactive compaction gate). The provider must
    normalize at its boundary so ``response.tokens`` means what the
    direct-API provider's per-request usage means. Output stays
    cumulative (it genuinely accumulates) and billing rides ``costUSD``
    (computed by the CLI from full cumulative usage) untouched.
    """
    response = _build_model_response(
        usage_event=cast(
            MutableJSON,
            {
                "type": "result",
                "modelUsage": {
                    "claude-opus-4-6": {
                        "inputTokens": 5_600_000,  # cumulative across rounds
                        "outputTokens": 450,
                        "cacheCreationInputTokens": 90_000,
                        "cacheReadInputTokens": 5_400_000,
                        "costUSD": 1.25,
                    }
                },
            },
        ),
        # Raw Anthropic API shape off the last ``message_start`` —
        # snake_case, unlike the camelCase ``modelUsage`` rows above.
        last_round_usage=cast(
            MutableJSON,
            {
                "input_tokens": 3,
                "cache_creation_input_tokens": 1_200,
                "cache_read_input_tokens": 96_000,
            },
        ),
        text="done",
        thinking_parts=[],
        signature_parts=[],
        stop_reason="end_turn",
        fallback_message_id="m1",
    )
    assert response.tokens.input_tokens == 3
    assert response.tokens.cache_creation_tokens == 1_200
    assert response.tokens.cache_read_tokens == 96_000
    assert response.tokens.output_tokens == 450
    assert response.total_cost == pytest.approx(1.25)


def test_round_context_tokens_sums_cache_pools() -> None:
    """Context footprint = non-cached input + both cache pools; None → 0."""
    assert _round_context_tokens(None) == 0
    assert (
        _round_context_tokens(
            cast(
                MutableJSON,
                {
                    "input_tokens": 3,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 90,
                },
            )
        )
        == 100
    )


@pytest.mark.asyncio
async def test_drain_captures_last_round_usage_for_context_anchor() -> None:
    """End-to-end through ``_drain_until_result``: the LAST round's
    ``message_start`` usage feeds both ``response.tokens``'s input side
    and the model's ``_last_input_tokens`` respawn/compaction anchor —
    while the cumulative ``result`` totals do not.
    """

    def _msg_start(input_tokens: int, cache_read: int) -> MutableJSON:
        return cast(
            MutableJSON,
            {
                "type": "stream_event",
                "event": {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": input_tokens,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": cache_read,
                        }
                    },
                },
            },
        )

    events = [
        _msg_start(50_000, 0),  # round 1: cold prompt
        _msg_start(3, 96_000),  # round 2 (final): cached context
        cast(
            MutableJSON,
            {
                "type": "result",
                "stop_reason": "end_turn",
                "is_error": False,
                "modelUsage": {
                    "claude-haiku-4-5": {
                        "inputTokens": 146_003,  # cumulative
                        "outputTokens": 70,
                        "cacheReadInputTokens": 96_000,
                        "costUSD": 0.01,
                    }
                },
            },
        ),
    ]

    class _Proc:
        async def read_json_line(
            self, *, skip_non_json: bool = False
        ) -> MutableJSON | None:
            del skip_non_json
            return events.pop(0) if events else None

    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    response = await model._drain_until_result(
        cast(Subproc, _Proc()), on_text=None, on_thinking=None
    )
    # Input side = round 2's context footprint, not the 146k sum.
    assert response.tokens.input_tokens == 3
    assert response.tokens.cache_read_tokens == 96_000
    assert response.tokens.output_tokens == 70
    assert model._last_input_tokens == 96_003


def test_dispatch_stream_event_routes_text_and_thinking() -> None:
    """``content_block_delta`` events fan text/thinking into separate buckets."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
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
        signature_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    _dispatch_stream_event(
        thinking_event,
        text_parts,
        thinking_parts,
        signature_parts,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    assert text_parts == ["hello"]
    assert thinking_parts == ["reflecting"]
    assert text_chunks == ["hello"]
    assert thinking_chunks == ["reflecting"]


def test_dispatch_stream_event_captures_signature_delta() -> None:
    """``signature_delta`` accumulates the thought signature.

    This test's predecessor used ``signature_delta`` as its example of
    an *ignorable* delta type — it is not ignorable: the signature must
    be carried alongside the thinking body, or any later wire re-send
    of the thinking block is rejected with HTTP 400
    ``thinking.signature: Field required``.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
    _dispatch_stream_event(
        cast(MutableJSON, {"delta": {"type": "signature_delta", "signature": "x"}}),
        text_parts,
        thinking_parts,
        signature_parts,
        on_text=None,
        on_thinking=None,
    )
    assert text_parts == []
    assert thinking_parts == []
    assert signature_parts == ["x"]


def test_dispatch_stream_event_ignores_unknown_delta_types() -> None:
    """A genuinely unknown delta does not perturb the accumulators."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
    _dispatch_stream_event(
        cast(MutableJSON, {"delta": {"type": "citation_delta", "citation": "x"}}),
        text_parts,
        thinking_parts,
        signature_parts,
        on_text=None,
        on_thinking=None,
    )
    assert text_parts == []
    assert thinking_parts == []
    assert signature_parts == []


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
        last_round_usage=None,
        text="reply",
        thinking_parts=["thought"],
        signature_parts=["sig-bytes"],
        stop_reason="end_turn",
        fallback_message_id="fallback",
    )
    assert response.message.text == "reply"
    # No ``message_start`` observed → input side is 0 ("unknown — estimate
    # instead"), NOT the cumulative 300 (see the normalization tests below).
    assert response.tokens.input_tokens == 0
    # Output IS cumulative: every internal round's generation was produced.
    assert response.tokens.output_tokens == 60
    assert response.total_cost == pytest.approx(0.0012)
    assert response.stop_reason == "model_finished"
    assert response.message_id == "sid-x"
    assert len(response.message.thinking_blocks) == 1
    # Signature MUST be carried alongside the thinking body — otherwise
    # a subsequent wire send fails with HTTP 400
    # ``thinking.signature: Field required``.
    block = response.message.thinking_blocks[0]
    assert block.get("type") == "thinking"
    assert block.get("thinking") == "thought"
    assert block.get("signature") == "sig-bytes"


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
        last_round_usage=None,
        text="",
        thinking_parts=[],
        signature_parts=[],
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


@pytest.mark.asyncio
async def test_stream_eof_respawns_and_resets_sent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess EOF is transport failure and resets Anthropic send state."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
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
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
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
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
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
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])

    with pytest.raises(Exception, match="stdout closed"):
        await model.stream(request)
    with pytest.raises(RuntimeError, match="transport failure budget exhausted"):
        await model.stream(request)

    assert respawn_count == 2
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


@pytest.mark.asyncio
async def test_stream_application_error_does_not_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application ``ValueError`` escapes without subprocess respawn."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    respawn_count = 0

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, object())

        async def respawn(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, object())

    async def _send_entry(proc: Subproc, history: object) -> None:
        del proc, history
        raise ValueError("application bug")

    async def _drain_until_result(
        proc: Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        del proc, on_text, on_thinking
        raise AssertionError("unreachable")

    model._system_hash = _hash_system(None)
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_entry", _send_entry)
    monkeypatch.setattr(model, "_drain_until_result", _drain_until_result)
    request = ModelRequest(messages=[UserMessage(text="hi")])
    with pytest.raises(ValueError, match="application bug"):
        await model.stream(request)
    assert respawn_count == 0


def test_should_respawn_skips_when_no_active() -> None:
    """Before the first acquire, the model has no active and skips respawn."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert model._should_respawn(ModelRequest(messages=[UserMessage(text="hi")])) is (
        False
    )


@pytest.mark.asyncio
async def test_stream_failed_turn_keeps_proven_system_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    system = "attempted system"

    class _Proc:
        pass

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _Proc())

        async def discard_spare(self) -> None:
            pass

        async def respawn(self) -> Subproc:
            return cast(Subproc, _Proc())

        async def respawn_after_transport_failure(self) -> Subproc:
            return cast(Subproc, _Proc())

    async def _exchange_turn(
        proc: Subproc,
        request: ModelRequest,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        del proc, request, on_text, on_thinking
        raise SubprocessTransportError("boom")

    model._system_hash = _hash_system("proven system")
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_exchange_turn", _exchange_turn)

    with pytest.raises(SubprocessTransportError, match="boom"):
        _ = await model.stream(
            ModelRequest(messages=[UserMessage(text="hi")], system=system)
        )

    assert model._system_hash == _hash_system("proven system")


@pytest.mark.asyncio
async def test_stream_same_system_after_first_acquire_does_not_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    respawn_count = 0
    system = "be precise"

    class _Proc:
        pass

    class _HotSpare:
        def __init__(self) -> None:
            self.active: Subproc | None = None

        async def acquire(self) -> Subproc:
            if self.active is None:
                self.active = cast(Subproc, _Proc())
            return self.active

        async def respawn(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            self.active = cast(Subproc, _Proc())
            return self.active

        async def discard_spare(self) -> None:
            pass

        def record_success(self) -> None:
            pass

    async def _send_entry(proc: Subproc, entry: TapeEvent) -> None:
        del proc, entry

    async def _drain_until_result(
        proc: Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        del proc, on_text, on_thinking
        return ModelResponse(message=AssistantMessage(text="ok"))

    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_entry", _send_entry)
    monkeypatch.setattr(model, "_drain_until_result", _drain_until_result)

    first = UserMessage(text="first")
    second = UserMessage(text="second")
    _ = await model.stream(ModelRequest(messages=[first], system=system))
    _ = await model.stream(ModelRequest(messages=[first, second], system=system))

    assert respawn_count == 0
    assert model._system_hash == _hash_system(system)


@pytest.mark.asyncio
async def test_stream_system_change_discards_warmed_old_system_spare() -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    spawned_systems: list[str] = []
    used_systems: list[str] = []
    warmed = asyncio.Event()

    class _Proc:
        def __init__(self, system: str) -> None:
            self.system = system

        async def close(self) -> None:
            pass

    async def spawn_initialized() -> Subproc:
        spawned_systems.append(model._pending_system)
        if len(spawned_systems) == 2:
            warmed.set()
        return cast(Subproc, _Proc(model._pending_system))

    async def send_entry(proc: Subproc, entry: TapeEvent) -> None:
        del entry
        used_systems.append(cast(_Proc, proc).system)

    async def drain_until_result(
        proc: Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
        *,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        del proc, on_text, on_thinking, update_input_tokens
        return ModelResponse(message=AssistantMessage(text="ok"))

    model._hot_spare = HotSpare(spawn_initialized)
    model._send_entry = send_entry  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.
    model._drain_until_result = drain_until_result  # ty: ignore[invalid-assignment] -- test replaces method with same call shape bound as a plain function.

    _ = await model.stream(ModelRequest(messages=[UserMessage(text="a")], system="A"))
    await warmed.wait()
    _ = await model.stream(ModelRequest(messages=[UserMessage(text="b")], system="B"))

    assert used_systems == ["A", "B"]
    assert spawned_systems == ["A", "A", "B"]
    await model.close()


@pytest.mark.asyncio
async def test_exchange_turn_skips_assistant_replay() -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    lines: list[str] = []

    class _Proc:
        async def write_line(self, line: str) -> None:
            lines.append(line)

        async def read_json_line(self, *, skip_non_json: bool = False) -> MutableJSON:
            del skip_non_json
            return cast(MutableJSON, {"type": "result", "usage": {}})

    response = await model._exchange_turn(
        cast(Subproc, _Proc()),
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

    payloads = [json.loads(line) for line in lines]
    assert response.message.text == ""
    assert [payload["message"]["content"] for payload in payloads] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_respawn_resets_active_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    model._turn_count = 100
    model._last_input_tokens = model._max_request_tokens
    response = ModelResponse(message=AssistantMessage(text="ok"))

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

    async def _send_entry(proc: Subproc, history: TapeEvent) -> None:
        del proc, history

    async def _drain_until_result(
        proc: Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        del proc, on_text, on_thinking
        return response

    model._system_hash = _hash_system(None)
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_entry", _send_entry)
    monkeypatch.setattr(model, "_drain_until_result", _drain_until_result)

    request = ModelRequest(messages=[UserMessage(text="hi")])
    _ = await model.stream(request)

    assert model._turn_count == 1
    assert model._last_input_tokens == 0
    assert model._should_respawn(request) is False


@pytest.mark.asyncio
async def test_background_spare_warm_does_not_reset_active_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")

    async def _spawn_initialized() -> Subproc:
        return cast(Subproc, object())

    monkeypatch.setattr(model, "_spawn_initialized", _spawn_initialized)
    model._turn_count = 7
    model._last_input_tokens = 123

    await model._spawn_spare_initialized()

    assert model._turn_count == 7
    assert model._last_input_tokens == 123


@pytest.mark.asyncio
async def test_terminal_is_error_respawns_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    respawn_count = 0

    class _ErrorProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> MutableJSON:
            del skip_non_json
            return cast(MutableJSON, {"type": "result", "is_error": True})

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
    model._last_sent_index = 0
    model._sent_history_head = user
    model._turn_count = 9
    model._last_input_tokens = 123
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())

    with pytest.raises(SubprocessTransportError, match="result is_error"):
        _ = await model.stream(ModelRequest(messages=[user]))

    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None
    assert model._turn_count == 0
    assert model._last_input_tokens == 0


@pytest.mark.asyncio
async def test_exchange_turn_drains_each_user_like_entry() -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")

    class _Proc:
        pending = False

        async def write_line(self, line: str) -> None:
            del line
            assert not self.pending
            self.pending = True

        async def read_json_line(self, *, skip_non_json: bool = False) -> MutableJSON:
            del skip_non_json
            assert self.pending
            self.pending = False
            return cast(MutableJSON, {"type": "result", "usage": {}})

    proc = _Proc()

    _ = await model._exchange_turn(
        cast(Subproc, proc),
        ModelRequest(
            messages=[
                UserMessage(text="first"),
                AssistantMessage(text="hidden"),
                UserMessage(text="current"),
            ]
        ),
        on_text=None,
        on_thinking=None,
    )

    assert proc.pending is False


@pytest.mark.asyncio
async def test_exchange_turn_replay_drain_does_not_update_input_tokens() -> None:
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    model._last_input_tokens = 7
    drain_count = 0

    class _Proc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(
            self, *, skip_non_json: bool = False
        ) -> MutableJSON | None:
            del skip_non_json
            nonlocal drain_count
            drain_count += 1
            if drain_count == 1:
                return cast(
                    MutableJSON,
                    {
                        "type": "result",
                        "usage": {"input_tokens": model._max_request_tokens},
                    },
                )
            return None

    with pytest.raises(SubprocessTransportError, match="stdout closed"):
        _ = await model._exchange_turn(
            cast(Subproc, _Proc()),
            ModelRequest(
                messages=[UserMessage(text="replay"), UserMessage(text="current")]
            ),
            on_text=None,
            on_thinking=None,
        )

    assert model._last_input_tokens == 7


def test_serialize_for_stdin_user_passthrough() -> None:
    """``UserMessage`` falls through ``_serialize_for_stdin`` to ``_user_line``."""
    line = _serialize_for_stdin(UserMessage(text="ping"), max_image_dim=8000)
    assert line["type"] == "user"


_ = ToolCall


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
