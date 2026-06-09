"""Tests for ``providers.anthropic_cli``: wire-format + lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import asyncio
import inspect
import json
import os

import pytest

from sagent.agent.runtime import cli_publish_var
from sagent.lib.json import MutableJSON
from sagent.providers import anthropic_cli
from sagent.providers.anthropic import Anthropic
from sagent.providers.anthropic_cli import (
    AnthropicCLI,
    AnthropicCLIRetryableError,
    _anthropic_subprocess_env,
    _AnthropicCLIModel,
    _build_anthropic_argv,
    _build_model_response,
    _dispatch_stream_event,
    _extract_retry_after_ms,
    _hash_system,
    _is_event_retryable,
    _round_context_tokens,
    _serialize_for_stdin,
    _session_jsonl_exists,
    _user_line,
)
from sagent.providers.anthropic_cli_session import session_jsonl_path
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolLabel,
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


def test_argv_session_id_swaps_no_persistence_for_session_id_flag() -> None:
    """First-turn argv carries ``--session-id <uuid>`` instead of
    ``--no-session-persistence``.
    """
    argv = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="be brief",
        bridge_url="http://127.0.0.1:1234/mcp",
        bridge_server_name="sagent",
        session_id="deadbeef-1234-5678-9abc-deadbeef1234",
        resume_existing=False,
    )
    assert "--no-session-persistence" not in argv
    assert "--session-id" in argv
    assert "--resume" not in argv
    sid_idx = argv.index("--session-id")
    assert argv[sid_idx + 1] == "deadbeef-1234-5678-9abc-deadbeef1234"


def test_argv_session_id_resume_existing_uses_resume_flag() -> None:
    """``resume_existing=True`` → ``--resume <uuid>``, no
    ``--session-id``.
    """
    argv = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="be brief",
        bridge_url="http://127.0.0.1:1234/mcp",
        bridge_server_name="sagent",
        session_id="deadbeef-1234-5678-9abc-deadbeef1234",
        resume_existing=True,
    )
    assert "--no-session-persistence" not in argv
    assert "--session-id" not in argv
    assert "--resume" in argv
    r_idx = argv.index("--resume")
    assert argv[r_idx + 1] == "deadbeef-1234-5678-9abc-deadbeef1234"


def test_argv_default_session_id_none_preserves_stateless_flag() -> None:
    """No session_id → existing ``--no-session-persistence`` behaviour."""
    argv = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="x",
        bridge_url="http://x",
        bridge_server_name="sagent",
    )
    assert "--no-session-persistence" in argv
    assert "--session-id" not in argv
    assert "--resume" not in argv


def test_argv_denies_sendmessage_builtin() -> None:
    """``SendMessage`` (Claude-Teams built-in) is denied in every mode.

    In session-persistence mode we omit ``--tools`` so the CLI's default
    allowlist applies -- which includes ``SendMessage``, a tool that
    routes to Claude Teams' private (here EMPTY) agent registry and
    silently drops the message. Agents must use
    ``mcp__sagent_chat__sagent_send`` instead. Regression guard for the
    2026-06-09 lost-PR-notification bug (SWE reached for ``SendMessage``
    to tell TL a PR was open; the message vanished).
    """

    def _denied(argv: list[str]) -> bool:
        return (
            "--disallowedTools" in argv
            and argv[argv.index("--disallowedTools") + 1] == "SendMessage"
        )

    stateless = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="be brief",
        bridge_url="http://127.0.0.1:1234/mcp",
        bridge_server_name="sagent",
    )
    persistent = _build_anthropic_argv(
        model_id="claude-sonnet-4-5",
        system_prompt="be brief",
        bridge_url="http://127.0.0.1:1234/mcp",
        bridge_server_name="sagent",
        session_id="deadbeef-1234-5678-9abc-deadbeef1234",
        resume_existing=True,
    )
    assert _denied(stateless)
    assert _denied(persistent)


def test_model_session_id_initialises_session_persistent_mode() -> None:
    """``AnthropicCLI.model(session_id=...)`` flips the mode flag,
    bypasses HotSpare, and starts uninitialised.
    """
    provider = AnthropicCLI()
    sid = "deadbeef-1234-5678-9abc-deadbeef1234"
    m = provider.model("claude-haiku-4-5", session_id=sid)
    assert m._session_id == sid
    assert m._session_initialized is False
    assert m._hot_spare is None
    assert m._active_proc is None
    # Stateless companion still works.
    m_stateless = provider.model("claude-haiku-4-5")
    assert m_stateless._session_id is None
    assert m_stateless._hot_spare is not None


def test_session_jsonl_exists_is_cwd_aware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resume-vs-mint probe only sees sessions under THIS cwd's
    encoded project dir.

    Claude indexes sessions per encoded-cwd project dir and ``--resume``
    cannot reach across. The probe used to glob across ALL project dirs,
    which broke the moment two deployments derived the same
    deterministic per-role uuid: live repro 2026-06-09 — a second
    server instance launched from a scratch cwd found the primary
    deployment's JSONL via the glob, chose ``--resume``, and claude
    exited ``No conversation found``, wedging warmup for all five
    agents.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    sid = "deadbeef-1234-5678-9abc-deadbeef1234"

    cwd_a = tmp_path / "deploy-a"
    cwd_b = tmp_path / "deploy-b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    # Record a session under deploy-a's encoded dir only.
    jsonl_a = session_jsonl_path(sid, cwd=cwd_a)
    jsonl_a.parent.mkdir(parents=True, exist_ok=True)
    jsonl_a.write_text("{}\n")

    monkeypatch.chdir(cwd_a)
    assert _session_jsonl_exists(sid) is True  # same cwd → resume
    monkeypatch.chdir(cwd_b)
    assert _session_jsonl_exists(sid) is False  # other cwd → fresh session


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


def test_anthropic_subprocess_env_overrides_home_when_tmpdir_set(
    tmp_path: Path,
) -> None:
    """Stateless mode (or session-persistent + per-account): tmpdir
    becomes HOME so the renamed credentials file is found.
    """
    env = _anthropic_subprocess_env(tmp_path)
    assert env["HOME"] == str(tmp_path)
    assert env["USERPROFILE"] == str(tmp_path)
    # Stateless default has CLAUDE_CODE_SKIP_PROMPT_HISTORY set.
    assert env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"


def test_anthropic_subprocess_env_skip_history_off_when_persistent(
    tmp_path: Path,
) -> None:
    """Session-persistent mode keeps the SKIP_PROMPT_HISTORY var unset
    so claude actually writes its session JSONL.
    """
    env = _anthropic_subprocess_env(tmp_path, persist_session=True)
    assert "CLAUDE_CODE_SKIP_PROMPT_HISTORY" not in env


def test_anthropic_subprocess_env_disables_autocompact_in_materialize_mode(
    tmp_path: Path,
) -> None:
    """v2.1-α: ``materialize_session=True`` must disable claude's auto-compact.

    Rationale: in materialize mode sagent's tape is the source of truth
    and overwrites the on-disk JSONL each turn. If claude auto-compacts
    mid-session, it writes a ``system/compact_boundary`` to the file
    that sagent's tape doesn't carry. The next materialize-overwrite
    cycle would clobber claude's compaction, restoring the full
    pre-compaction history and re-triggering the blocking_limit. So we
    suppress claude's compactor and rely on sagent's own (which
    records compactions as ``ContextSplice`` on the tape, which the
    materializer renders as the resolved view).
    """
    # v2 baseline: session-persistent + NOT materialize → auto-compact ENABLED
    env_v2 = _anthropic_subprocess_env(
        tmp_path, persist_session=True, materialize_session=False
    )
    assert "DISABLE_AUTO_COMPACT" not in env_v2

    # v2.1-α: session-persistent + materialize → auto-compact DISABLED
    env_v21 = _anthropic_subprocess_env(
        tmp_path, persist_session=True, materialize_session=True
    )
    assert env_v21.get("DISABLE_AUTO_COMPACT") == "1"

    # Stateless mode (regardless of materialize flag) always disables
    env_stateless = _anthropic_subprocess_env(tmp_path, persist_session=False)
    assert env_stateless.get("DISABLE_AUTO_COMPACT") == "1"


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


def test_seed_session_marks_prefix_synced() -> None:
    """``seed_session``: the rehydration handshake after ``replay_tape``.

    Declares the on-disk session JSONL already holds the first N tape
    entries, so the next spawn resumes with ``--resume`` instead of
    minting a new session, and the prefix is not re-fed via stdin.
    Public API so host applications (e.g. the blackjax-ai-devs-channel
    example) never touch provider-private attributes. No-op in
    stateless mode.
    """
    provider = AnthropicCLI()
    seeded = provider.model(
        "claude-haiku-4-5",
        session_id="deadbeef-1234-5678-9abc-deadbeef1234",
    )
    assert seeded.session_id == "deadbeef-1234-5678-9abc-deadbeef1234"
    seeded.seed_session(42)
    assert seeded._last_sent_index == 42
    assert seeded._session_initialized is True

    stateless = provider.model("claude-haiku-4-5")
    assert stateless.session_id is None
    stateless.seed_session(42)  # no session to seed → no-op
    assert stateless._last_sent_index == 0


def test_model_accepts_subprocess_read_timeout_kwarg() -> None:
    """``AnthropicCLI.model(subprocess_read_timeout_sec=…)`` plumbs to the model.

    The v2.1-β.2 lever: bump beyond the 60s Subproc default so
    long-running synchronous Bash tools (``pre-commit run``, ``ty
    check``, JAX warmup compiles) don't trip the divergence cascade.
    Verified live 2026-06-09 on SWE's PR3 ``pre-commit + git commit``
    chain that was 7-times-eaten by ``send_with_retry`` divergence.
    """
    provider = AnthropicCLI()
    model = provider.model(
        "claude-haiku-4-5",
        subprocess_read_timeout_sec=300.0,
    )
    assert model._subprocess_read_timeout_sec == 300.0  # type: ignore[attr-defined]

    # Default path: ``None`` defers to ``Subproc``'s own default.
    default_model = provider.model("claude-haiku-4-5")
    assert default_model._subprocess_read_timeout_sec is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_persistent_stream_returns_empty_when_history_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``agent.clear()``, sagent's runtime calls ``stream()`` with
    an empty ``request.messages`` until new input arrives. The
    session-persistent path detects this (``_last_sent_index > len``),
    resets its counters, deletes the stale on-disk session JSONL so
    the next ``--session-id`` call works, and returns a no-op
    ``ModelResponse`` rather than crashing the runtime turn.

    Regression for 2026-06-03 07:20 ``RuntimeError: stream() called
    with no new user-like entries to send`` triggered by /api/restart
    on TL.
    """
    _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic_cli.shutil.which",
        _which_claude,
    )
    # Point HOME at tmp_path so the JSONL-cleanup glob is sandboxed.
    monkeypatch.setenv("HOME", str(tmp_path))

    provider = AnthropicCLI.from_credentials()
    sid = "deadbeef-1234-5678-9abc-deadbeef1234"
    model = provider.model("claude-haiku-4-5", session_id=sid)

    # Stage 1: pretend two turns of conversation already happened
    # (``_last_sent_index == 2``, on-disk session JSONL present).
    model._last_sent_index = 2
    model._session_initialized = True
    proj_dir = tmp_path / ".claude" / "projects" / "-some-cwd"
    proj_dir.mkdir(parents=True)
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    assert jsonl.exists()

    # Stage 2: simulate post-``agent.clear()`` call: history is empty
    # but the provider's counters still think 2 messages were sent.
    request = ModelRequest(
        system="terse",
        messages=[],
        tools=[],
    )
    response = await model.stream(
        request,
        on_text=None,
        on_thinking=None,
    )

    # The response is a no-op (empty assistant text, no tools, zero
    # cost) so the runtime gets a clean "model said nothing" turn.
    assert response.message.text == ""
    assert response.message.tool_calls == ()
    assert response.tokens.input_tokens == 0
    assert response.tokens.output_tokens == 0

    # Provider state was reset: next real call will use ``--session-id``
    # (not ``--resume``).
    assert model._last_sent_index == 0
    assert model._session_initialized is False

    # The stale on-disk JSONL was removed so the next ``--session-id``
    # spawn doesn't error with "Session ID is already in use".
    assert not jsonl.exists()


@pytest.mark.asyncio
async def test_session_persistent_advances_sent_index_per_entry_on_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for 2026-06-03 ~14:30 SWE bug: when multiple new
    user-like entries are queued and the drain aborts partway through,
    ``_last_sent_index`` must reflect every entry already WRITTEN to
    stdin -- not just the entries whose drain completed.

    Otherwise the next ``--resume`` re-writes the entries that already
    landed in claude's session JSONL (producing duplicates) AND fails
    to reach the entries that came after the abort point (silently
    dropping them). The SWE symptom was an early "Great smoke" inbound
    appearing 3× in the session JSONL while five subsequent TL STOP
    directives never appeared at all.
    """
    _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli.shutil.which",
        lambda _name: "/usr/bin/claude",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    sid = "deadbeef-1234-5678-9abc-deadbeef1234"
    provider = AnthropicCLI.from_credentials()
    model = provider.model("claude-haiku-4-5", session_id=sid)
    # Pretend a turn already landed so we're past the first-spawn case.
    model._session_initialized = True
    model._last_sent_index = 5

    # Replace the heavy I/O with mocks:
    #   * _ensure_tools_bridge / _sync_tools_bridge: no-op
    #   * _spawn_initialized: returns a fake proc
    #   * _send_entry: records the entry written
    #   * _drain_until_result: returns OK on the first call (entry index
    #     5 succeeds end-to-end), raises SubprocessTransportError on the
    #     second (simulating aborted_streaming on the second entry's
    #     model_call)
    bridge_calls: list[object] = []
    sent_entries: list[TapeEvent] = []

    async def _ensure() -> None:
        bridge_calls.append("ensure")

    model._ensure_tools_bridge = _ensure  # ty: ignore[invalid-assignment]
    model._sync_tools_bridge = lambda r: bridge_calls.append(("sync", r))  # ty: ignore[invalid-assignment]

    fake_proc = MagicMock()
    fake_proc.close = AsyncMock()
    model._spawn_initialized = AsyncMock(return_value=fake_proc)  # ty: ignore[invalid-assignment]

    async def _send_entry(proc: object, entry: TapeEvent) -> None:
        del proc
        sent_entries.append(entry)

    model._send_entry = _send_entry  # ty: ignore[invalid-assignment]

    drain_calls = 0

    async def _drain(
        proc: object, on_text=None, on_thinking=None, update_input_tokens: bool = True
    ):
        del proc, on_text, on_thinking, update_input_tokens
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 1:
            # First drain (for the FIRST entry) succeeds: this would be
            # the equivalent of "Great smoke" being acknowledged.
            return ModelResponse(
                message=AssistantMessage(text="ack 1", tool_calls=()),
                stop_reason="model_finished",
            )
        # Second drain (for the SECOND entry) aborts mid-stream: this
        # is the equivalent of the aborted_streaming on the abort cycle
        # that prevented TL's STOP from being processed in production.
        raise SubprocessTransportError("simulated abort on entry 2")

    model._drain_until_result = _drain  # ty: ignore[invalid-assignment]

    # Three entries queued. _last_sent_index = 5 means request.messages
    # has 8 entries; entries 5, 6, 7 are the new user-like ones.
    msg_E1 = AgentSendMessage(source="tl", text="entry 1 — should land cleanly")
    msg_E2 = AgentSendMessage(source="tl", text="entry 2 — drain aborts on this one")
    msg_E3 = AgentSendMessage(
        source="tl", text="entry 3 — STOP directive that must NOT be lost"
    )
    request = ModelRequest(
        system="x",
        messages=[
            UserMessage(text="old turn 1"),
            UserMessage(text="old turn 2"),
            UserMessage(text="old turn 3"),
            UserMessage(text="old turn 4"),
            UserMessage(text="old turn 5"),
            msg_E1,
            msg_E2,
            msg_E3,
        ],
        tools=[],
    )

    with pytest.raises(SubprocessTransportError):
        await model.stream(request, on_text=None, on_thinking=None)

    # Both E1 and E2 were written to stdin before the drain raised on
    # E2's model_call. E3 was NOT written -- the loop short-circuited.
    assert sent_entries == [msg_E1, msg_E2]

    # The CRITICAL regression assertion: _last_sent_index now reflects
    # both writes (5 + 2 entries = position 7, pointing at E3 which is
    # the FIRST entry the next --resume must deliver).
    # Pre-fix behaviour would have left _last_sent_index at 5, causing
    # the next retry to re-write E1 (duplicate in claude's session) and
    # re-attempt the same drain abort sequence, leaving E3 perpetually
    # stranded.
    assert model._last_sent_index == 7


def test_is_event_retryable_classifies_organic_shapes() -> None:
    """Direct unit test of the catalog. Examples are taken from
    actual ``is_error: True`` events captured 2026-06-03/04 from the
    multi-agent server.
    """
    # 1. The dominant aborted_streaming + ede_diagnostic shape (TL,
    #    2026-06-03 10:17:53 — 418k cache reads attempt that died on
    #    a tool_use boundary). Retryable.
    aborted_streaming = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "stop_reason": "tool_use",
        "terminal_reason": "aborted_streaming",
        "errors": [
            "[ede_diagnostic] result_type=user last_content_type=n/a "
            "stop_reason=tool_use",
        ],
    }
    assert _is_event_retryable(aborted_streaming) is True

    # 2. ede_diagnostic without aborted_streaming terminal_reason --
    #    still retryable (same root cause, different surface).
    ede_only = {
        "is_error": True,
        "terminal_reason": "completed",
        "errors": ["[ede_diagnostic] mid-stream cut"],
    }
    assert _is_event_retryable(ede_only) is True

    # 3. Context overflow -- NOT retryable. The next attempt would hit
    #    the same wall; operator must clear the session.
    blocking_limit = {
        "is_error": True,
        "terminal_reason": "blocking_limit",
        "result": "Prompt is too long",
        "stop_reason": "stop_sequence",
    }
    assert _is_event_retryable(blocking_limit) is False

    # 4. Empty errors list, no special terminal_reason -- not retryable
    #    by default.
    unknown_error = {"is_error": True, "errors": [], "stop_reason": "end_turn"}
    assert _is_event_retryable(unknown_error) is False


def test_extract_retry_after_ms_handles_both_key_names() -> None:
    """``send_with_retry`` falls back to exponential backoff when no
    hint is present, so ``None`` is a valid return; but when the CLI
    emits a hint (either ``retry_after_ms`` or ``retry_delay_ms``) we
    forward it.
    """
    assert _extract_retry_after_ms({"retry_after_ms": 506}) == 506.0
    assert _extract_retry_after_ms({"retry_delay_ms": 1247.5}) == 1247.5
    assert _extract_retry_after_ms({}) is None
    # Negative or non-numeric values are ignored (defensive).
    assert _extract_retry_after_ms({"retry_after_ms": -1}) is None
    assert _extract_retry_after_ms({"retry_after_ms": "soon"}) is None


def test_is_retryable_provider_error_session_persistent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``send_with_retry`` consults this method to decide whether to
    sleep + retry vs let the runtime see the error. Session-persistent
    mode answers True for :class:`AnthropicCLIRetryableError`;
    stateless mode keeps the historical False (its HotSpare
    subprocess has already consumed the stdin lines we wrote, so a
    same-call retry would duplicate or stall).
    """
    _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic_cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    provider = AnthropicCLI.from_credentials()

    # Stateless mode (no session_id) -- never retry at this layer.
    stateless = provider.model("claude-haiku-4-5")
    retryable_exc = AnthropicCLIRetryableError("aborted_streaming")
    assert stateless.is_retryable_provider_error(retryable_exc) is False

    # Session-persistent mode (session_id set) -- retry the retryable type.
    persistent = provider.model(
        "claude-haiku-4-5",
        session_id="deadbeef-1234-5678-9abc-deadbeef1234",
    )
    assert persistent.is_retryable_provider_error(retryable_exc) is True

    # Both modes still propagate non-retryable subprocess errors:
    plain_exc = SubprocessTransportError("subprocess stdout closed before result")
    assert stateless.is_retryable_provider_error(plain_exc) is False
    assert persistent.is_retryable_provider_error(plain_exc) is False

    # Random Exception that isn't a subprocess error: never retried.
    assert persistent.is_retryable_provider_error(RuntimeError("oops")) is False


def test_model_session_initialized_probes_disk_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host-application restart should pick up prior conversations
    transparently: at construction time the provider probes for the
    session JSONL under THIS cwd's encoded project dir and sets
    ``_session_initialized = True`` on a hit -- so the first spawn
    uses ``--resume`` (not ``--session-id``, which would error with
    "Session ID is already in use"). The probe is cwd-aware: a JSONL
    for the same uuid under a DIFFERENT cwd's project dir must not
    count (claude's ``--resume`` can't see it; see
    ``test_session_jsonl_exists_is_cwd_aware``).
    """
    sid = "deadbeef-1234-5678-9abc-deadbeef1234"
    other = "1c0705bd-ecf6-55a2-91cc-9d519e9ca6f6"

    # Stage 1: empty $HOME -> session_initialized is False.
    monkeypatch.setenv("HOME", str(tmp_path))
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5", session_id=sid)
    assert model._session_initialized is False

    # Stage 2: drop a session JSONL for OUR sid under THIS cwd's
    # encoded project dir -> session_initialized flips to True.
    ours = session_jsonl_path(sid, cwd=workdir)
    ours.parent.mkdir(parents=True, exist_ok=True)
    ours.write_text("{}\n", encoding="utf-8")
    model = provider.model("claude-haiku-4-5", session_id=sid)
    assert model._session_initialized is True

    # Stage 3: jsonl exists for a DIFFERENT uuid but not ours -> still False.
    other_jsonl = session_jsonl_path(other, cwd=workdir)
    other_jsonl.write_text("{}\n", encoding="utf-8")
    ours.unlink()
    model = provider.model("claude-haiku-4-5", session_id=sid)
    assert model._session_initialized is False

    # Stage 4: stateless mode (session_id=None) is always False.
    model = provider.model("claude-haiku-4-5")
    assert model._session_initialized is False


def test_anthropic_subprocess_env_inherits_real_home_when_tmpdir_none() -> None:
    """Session-persistent + single-account: ``tmpdir=None`` means the
    subprocess inherits the operator's real HOME so native tools (Bash,
    gh, git) find ``~/.config/`` and ``~/.gitconfig``.
    """
    operator_home = os.environ.get("HOME", "")
    env = _anthropic_subprocess_env(None, persist_session=True)
    # HOME comes through unchanged (inherited from os.environ).
    assert env.get("HOME") == operator_home
    # No USERPROFILE override either.
    if "USERPROFILE" not in os.environ:
        assert "USERPROFILE" not in env


def test_dispatch_stream_event_routes_text_and_thinking() -> None:
    """``content_block_delta`` events fan text/thinking into separate buckets."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    tool_use_blocks: dict[int, dict[str, object]] = {}
    text_event = cast(
        MutableJSON,
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        },
    )
    thinking_event = cast(
        MutableJSON,
        {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "reflecting"},
        },
    )
    _dispatch_stream_event(
        text_event,
        text_parts,
        thinking_parts,
        signature_parts,
        tool_use_blocks,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    _dispatch_stream_event(
        thinking_event,
        text_parts,
        thinking_parts,
        signature_parts,
        tool_use_blocks,
        on_text=text_chunks.append,
        on_thinking=thinking_chunks.append,
    )
    assert text_parts == ["hello"]
    assert thinking_parts == ["reflecting"]
    assert signature_parts == []  # no signature_delta yet
    assert text_chunks == ["hello"]
    assert thinking_chunks == ["reflecting"]


def test_dispatch_stream_event_captures_signature_delta() -> None:
    """``signature_delta`` events feed the signature accumulator.

    Required for v2.1-α materializer mode: claude's stream emits a
    ``signature_delta`` event after the thinking body, carrying the
    opaque thought signature. The stream parser MUST capture it —
    without it, ``AssistantMessage.thinking_blocks`` carry blocks
    with no signature, the materializer writes them unsigned to
    JSONL, and Anthropic's API rejects on the next ``--resume`` wire
    send with ``HTTP 400 thinking.signature: Field required``.
    Verified live 2026-06-09 in statistician session
    ``b4fe5972-...`` after TL sent a thank-you AgentSendMessage.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
    tool_use_blocks: dict[int, dict[str, object]] = {}
    sig_event = cast(
        MutableJSON,
        {
            "type": "content_block_delta",
            "delta": {"type": "signature_delta", "signature": "abc123"},
        },
    )
    _dispatch_stream_event(
        sig_event,
        text_parts,
        thinking_parts,
        signature_parts,
        tool_use_blocks,
        on_text=None,
        on_thinking=None,
    )
    assert signature_parts == ["abc123"]
    assert text_parts == []
    assert thinking_parts == []


def test_dispatch_stream_event_publishes_rich_tool_label_at_stop() -> None:
    """tool_use is published at content_block_stop with name + arg
    summary, after the streamed input_json_delta has been accumulated.
    """
    published: list[object] = []
    token = cli_publish_var.set(published.append)
    try:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        signature_parts: list[str] = []
        tool_use_blocks: dict[int, dict[str, object]] = {}
        # 1) start: registers tool_use at index 0 -- NO label published yet
        _dispatch_stream_event(
            cast(
                MutableJSON,
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "Bash",
                        "input": {},
                    },
                },
            ),
            text_parts,
            thinking_parts,
            signature_parts,
            tool_use_blocks,
            on_text=None,
            on_thinking=None,
        )
        assert published == []  # nothing yet -- we wait for args
        # 2) deltas: stream the JSON in two chunks
        for partial in ('{"command":"ls', ' -la"}'):
            _dispatch_stream_event(
                cast(
                    MutableJSON,
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": partial},
                    },
                ),
                text_parts,
                thinking_parts,
                signature_parts,
                tool_use_blocks,
                on_text=None,
                on_thinking=None,
            )
        assert published == []  # still nothing
        # 3) stop: now we publish the rich label
        _dispatch_stream_event(
            cast(
                MutableJSON,
                {"type": "content_block_stop", "index": 0},
            ),
            text_parts,
            thinking_parts,
            signature_parts,
            tool_use_blocks,
            on_text=None,
            on_thinking=None,
        )
    finally:
        cli_publish_var.reset(token)
    assert len(published) == 1
    label = published[0]
    assert isinstance(label, ToolLabel)
    # Includes both the tool name and the command arg
    assert label.text == "Bash ls -la"
    assert label.call_id == "toolu_abc123"


def test_dispatch_stream_event_no_label_for_text_block_start() -> None:
    """``content_block_start`` for ``text`` does NOT publish a ToolLabel."""
    published: list[object] = []
    token = cli_publish_var.set(published.append)
    try:
        tool_use_blocks: dict[int, dict[str, object]] = {}
        _dispatch_stream_event(
            cast(
                MutableJSON,
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            [],
            [],
            [],
            tool_use_blocks,
            on_text=None,
            on_thinking=None,
        )
        # And a content_block_stop on the text block: no label.
        _dispatch_stream_event(
            cast(MutableJSON, {"type": "content_block_stop", "index": 1}),
            [],
            [],
            [],
            tool_use_blocks,
            on_text=None,
            on_thinking=None,
        )
    finally:
        cli_publish_var.reset(token)
    assert published == []


def test_dispatch_stream_event_ignores_unknown_delta_types() -> None:
    """A delta type we don't recognize does not perturb any accumulator.

    Historical note: this test previously used ``signature_delta`` as
    the exemplar "unknown" type because the parser intentionally
    dropped it. Capturing the signature became load-bearing once
    v2.1-α materialize mode started re-sending history via wire, so
    ``signature_delta`` moved into the known set. We use a genuinely
    unknown type (``fake_future_delta``) to keep the contract
    semantics — the parser stays inert on shapes it doesn't know.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    signature_parts: list[str] = []
    tool_use_blocks: dict[int, dict[str, object]] = {}
    _dispatch_stream_event(
        cast(
            MutableJSON,
            {
                "type": "content_block_delta",
                "delta": {"type": "fake_future_delta", "payload": "x"},
            },
        ),
        text_parts,
        thinking_parts,
        signature_parts,
        tool_use_blocks,
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
    assert model._hot_spare is not None
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


def test_cancel_in_flight_returns_false_when_no_active_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No active hot-spare ⇒ ``cancel_in_flight`` returns False without raising."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr("sagent.providers.anthropic_cli._CREDS_PATH", creds)
    provider = AnthropicCLI.from_credentials()
    model = provider.model("claude-opus-4-7")
    # Hot spare has not been acquired -> ``active`` is None.
    assert model._hot_spare.active is None
    assert model.cancel_in_flight() is False


def test_cancel_in_flight_signals_active_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Active hot-spare ⇒ ``cancel_in_flight`` forwards to ``Subproc.interrupt``."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr("sagent.providers.anthropic_cli._CREDS_PATH", creds)
    provider = AnthropicCLI.from_credentials()
    model = provider.model("claude-opus-4-7")

    class _FakeActiveSubproc:
        def __init__(self) -> None:
            self.interrupt_calls = 0

        def interrupt(self) -> bool:
            self.interrupt_calls += 1
            return True

    fake = _FakeActiveSubproc()
    monkeypatch.setattr(model._hot_spare, "_active", fake)
    assert model.cancel_in_flight() is True
    assert fake.interrupt_calls == 1


_ = ToolCall


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
