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
import re

import pytest

from sagent.lib.custom_json import MutableJSON
from sagent.providers.anthropic import cli as anthropic_cli
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.anthropic.cli import (
    AnthropicCLI,
    AnthropicCLIRetryableError,
    _anthropic_subprocess_env,
    _AnthropicCLIModel,
    _build_anthropic_argv,
    _build_model_response,
    _claude_auth_status,
    _dispatch_stream_event,
    _extract_retry_after_ms,
    _hash_system,
    _is_event_retryable,
    _real_home,
    _round_context_tokens,
    _serialize_for_stdin,
    _session_jsonl_path,
    _user_line,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolLabel,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


def _noop_sync_tools_bridge(
    request: ModelRequest, publish: Callable[[RuntimeEvent], None] | None = None
) -> None:
    del request, publish


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


def _which_claude_stub(_name: str) -> str | None:
    """Pretend ``claude`` is installed (monkeypatched ``shutil.which``)."""
    return "/usr/bin/claude"


def _auth_status_unknown(_binary: str) -> bool | None:
    """Pretend the installed CLI predates the native status command."""
    return None


def _auth_status_logged_in(_binary: str) -> bool | None:
    """Pretend the native CLI reports an active login."""
    return True


def _auth_status_logged_out(_binary: str) -> bool | None:
    """Pretend the native CLI reports no active login."""
    return False


def test_anthropic_cli_does_not_import_subscription_provider() -> None:
    source = inspect.getsource(anthropic_cli)
    assert "providers.anthropic_sub" not in source


def test_from_cli_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy CLIs without auth status still require the credentials file."""
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._claude_auth_status",
        _auth_status_unknown,
    )
    with pytest.raises(FileNotFoundError, match="no credentials"):
        AnthropicCLI.from_credentials()


def test_from_cli_accepts_native_login_without_credentials_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The native Keychain login is authoritative without the legacy file."""
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._claude_auth_status",
        _auth_status_logged_in,
    )

    provider = AnthropicCLI.from_credentials()

    assert provider.account is None


def test_from_cli_treats_explicit_default_account_as_native_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--account default`` must not bypass the macOS Keychain login."""
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._claude_auth_status",
        _auth_status_logged_in,
    )

    provider = AnthropicCLI.from_credentials(account="default")

    assert provider.account is None


def test_from_cli_missing_named_account_does_not_suggest_native_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Native login cannot create Sagent's legacy named credential file."""
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )

    with pytest.raises(FileNotFoundError) as exc:
        AnthropicCLI.from_credentials(account="work")

    assert "named-account credentials" in str(exc.value)
    assert "claude auth login" not in str(exc.value)


def test_from_cli_rejects_native_logged_out_state_even_with_stale_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale legacy JSON file must not override an explicit logged-out state."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr("sagent.providers.anthropic.cli._CREDS_PATH", creds)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._claude_auth_status",
        _auth_status_logged_out,
    )

    with pytest.raises(FileNotFoundError, match=r"claude auth login"):
        AnthropicCLI.from_credentials()


def test_claude_auth_status_scrubs_non_subscription_auth_and_reads_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native status checks subscription login without leaking API-key auth."""
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return MagicMock(
            stdout=(
                '{"loggedIn": true, "authMethod": "claude.ai", '
                '"apiProvider": "firstParty"}'
            ),
            returncode=0,
        )

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://synthetic.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-test-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret-test-token")
    monkeypatch.setenv("ANTHROPIC_AWS_API_KEY", "secret-aws-key")
    monkeypatch.setenv("ANTHROPIC_UNIX_SOCKET", "/synthetic/claude.sock")
    monkeypatch.setenv("CLAUDE_CODE_API_BASE_URL", "https://synthetic.invalid")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-oauth-token")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ACCESS_TOKEN", "secret-session-token")
    monkeypatch.setenv("CLAUDE_CODE_USE_ANTHROPIC_AWS", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_GATEWAY", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_MANTLE", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setattr("sagent.providers.anthropic.cli.subprocess.run", fake_run)

    assert _claude_auth_status("/opt/homebrew/bin/claude") is True
    assert captured["args"] == (
        [
            "/opt/homebrew/bin/claude",
            "auth",
            "status",
            "--json",
        ],
    )
    env = cast(dict[str, str], captured["env"])
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_AWS_API_KEY",
        "ANTHROPIC_UNIX_SOCKET",
        "CLAUDE_CODE_API_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_GATEWAY",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        assert key not in env


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (
            {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "apiProvider": "firstParty",
            },
            True,
        ),
        ({"loggedIn": True, "authMethod": "console"}, False),
        ({"loggedIn": True, "authMethod": "api_key"}, False),
        ({"loggedIn": True}, None),
        ({"loggedIn": False}, False),
    ],
)
def test_claude_auth_status_requires_subscription_method(
    monkeypatch: pytest.MonkeyPatch,
    status: dict[str, object],
    expected: bool | None,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> object:
        return MagicMock(
            stdout=json.dumps(status),
            returncode=0,
        )

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.subprocess.run",
        fake_run,
    )

    assert _claude_auth_status("/opt/homebrew/bin/claude") is expected


def test_login_runs_native_claudeai_flow_with_scrubbed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return MagicMock(returncode=0)

    def fake_which(_name: str) -> str:
        return "/opt/homebrew/bin/claude"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-test-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-oauth-token")
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("sagent.providers.anthropic.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._claude_auth_status",
        _auth_status_logged_in,
    )

    AnthropicCLI.login()

    assert captured["args"] == (
        [
            "/opt/homebrew/bin/claude",
            "auth",
            "login",
            "--claudeai",
        ],
    )
    env = cast(dict[str, str], captured["env"])
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_login_rejects_named_anthropic_account() -> None:
    with pytest.raises(ValueError, match="named accounts"):
        AnthropicCLI.login(account="work")


def test_login_reports_native_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(_name: str) -> str:
        return "/opt/homebrew/bin/claude"

    def fake_run(*_args: object, **_kwargs: object) -> object:
        return MagicMock(returncode=7)

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        AnthropicCLI.login()


def test_from_cli_with_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``from_credentials`` returns a configured provider when both creds + CLI exist."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        creds,
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        creds,
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        creds,
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        creds,
    )

    def _which_missing(name: str) -> str | None:
        del name
        return None

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
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
    assert model.tagged_model_id == "claude-sonnet-4-5+1m"
    assert model.max_request_tokens == 1_000_000


def test_model_rejects_an_unknown_tag() -> None:
    """``+fast`` is not a context tag, so it stays part of the base id."""
    provider = AnthropicCLI()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = provider.model("claude-opus-4-8+fast")


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
    assert model.capability.model_id == "claude-haiku-4-5"


def test_model_capabilities() -> None:
    """The CLI transport narrows the row it inherits from the API."""
    provider = AnthropicCLI()
    model = provider.model("claude-sonnet-4-5")
    assert model.capability.thinking_budget != frozenset({"none"})
    assert model.capability.thinking_effort == frozenset({"none"})
    assert model.capability.cache_ttl_sec == 0.0
    assert model.capability.manage_context_server_side == frozenset({True})
    assert model.capability.retries_internally is False


def test_is_context_overflow_text_markers() -> None:
    """Overflow classification is body-text driven, not status-code driven."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert model.is_context_overflow(RuntimeError("prompt is too long")) is True
    assert model.is_context_overflow(RuntimeError("exceeds context window")) is True
    assert model.is_context_overflow(RuntimeError("network error")) is False


def test_byte_limit_not_classified_as_context_overflow() -> None:
    """A request-byte-limit error must not classify as token overflow.

    Uniform with the HTTP providers: the byte wire-limit routes to
    byte-overflow recovery, never the ``/model`` larger-window remediation
    a larger window cannot satisfy.
    """
    model = AnthropicCLI().model("claude-haiku-4-5")
    assert model.is_context_overflow(RuntimeError("request entity too large")) is False


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
    # ``--tools ""`` disables the MCP bridge catalog (the CLI reads
    # empty-string as "allow no tools, MCP included"); never emit it.
    assert "--tools" not in argv
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


def test_session_jsonl_path_is_cwd_aware(tmp_path: Path) -> None:
    """The session path is keyed by THIS cwd's encoded project dir.

    Claude indexes sessions per encoded-cwd project dir and ``--resume``
    cannot reach across. The path encoding maps every non-``[A-Za-z0-9-]``
    char of the RESOLVED cwd to ``-``, so two different cwds produce two
    different project dirs: live repro 2026-06-09 — a second server
    instance launched from a scratch cwd resumed the primary
    deployment's JSONL and claude exited ``No conversation found``,
    wedging warmup for all five agents.
    """
    h = tmp_path / "home"
    sid = "deadbeef-1234-5678-9abc-deadbeef1234"

    cwd_a = tmp_path / "deploy-a"
    cwd_b = tmp_path / "deploy-b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    path_a = _session_jsonl_path(sid, cwd=cwd_a, home=h)
    path_b = _session_jsonl_path(sid, cwd=cwd_b, home=h)

    # Different cwds -> different encoded project dirs -> different paths.
    assert path_a != path_b

    encoded = re.sub(r"[^A-Za-z0-9-]", "-", str(cwd_a.resolve()))
    assert path_a == h / ".claude" / "projects" / encoded / f"{sid}.jsonl"


def test_user_line_text_only() -> None:
    """A plain ``UserMessage`` becomes a single ``content: str`` line."""
    line = _user_line(
        UserMessage(text="hello"), max_image_dim=8000, max_image_bytes=5 * 1024 * 1024
    )
    assert line == {"type": "user", "message": {"role": "user", "content": "hello"}}


def test_serialize_for_stdin_rejects_tool_result() -> None:
    """Tool results never traverse stdin -- the MCP bridge handles them."""
    with pytest.raises(RuntimeError, match="ToolResult in history"):
        _ = _serialize_for_stdin(
            ToolResult(call_id="x", content="done"),
            max_image_dim=8000,
            max_image_bytes=5 * 1024 * 1024,
        )


def test_anthropic_subprocess_env_overrides_home_when_tmpdir_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stateless mode (or session-persistent + per-account): tmpdir
    becomes HOME so the renamed credentials file is found.
    """
    monkeypatch.setenv(
        "CLAUDE_CONFIG_DIR",
        "/operator/.claude",
    )
    env = _anthropic_subprocess_env(tmp_path)
    assert env["HOME"] == str(tmp_path)
    assert env["USERPROFILE"] == str(tmp_path)
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude")
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


def test_anthropic_subprocess_env_strips_non_subscription_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_AWS_API_KEY",
        "ANTHROPIC_BEDROCK_MANTLE_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_UNIX_SOCKET",
        "AWS_BEARER_TOKEN_BEDROCK",
        "CLAUDE_CODE_API_BASE_URL",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_GATEWAY",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    )
    for key in keys:
        monkeypatch.setenv(key, "synthetic-test-value")

    env = _anthropic_subprocess_env(None)

    for key in keys:
        assert key not in env


def test_anthropic_subprocess_env_always_disables_autocompact(
    tmp_path: Path,
) -> None:
    """Auto-compact is disabled in both session and stateless modes.

    Sagent owns history in both modes, so claude's own compactor (which
    would write a ``system/compact_boundary`` to a file sagent
    overwrites next turn) must stay off.
    """
    env_persistent = _anthropic_subprocess_env(tmp_path, persist_session=True)
    assert env_persistent.get("DISABLE_AUTO_COMPACT") == "1"

    env_stateless = _anthropic_subprocess_env(tmp_path, persist_session=False)
    assert env_stateless.get("DISABLE_AUTO_COMPACT") == "1"


@pytest.mark.asyncio
async def test_stateless_default_account_inherits_native_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default-account subprocesses retain macOS Keychain-backed CLI auth."""
    operator_home = tmp_path / "operator-home"
    monkeypatch.setenv("HOME", str(operator_home))
    captured: dict[str, object] = {}

    class _CaptureSubproc:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured["argv"] = argv
            captured.update(kwargs)

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(anthropic_cli, "Subproc", _CaptureSubproc)
    model = AnthropicCLI().model("claude-haiku-4-5")
    bridge = MagicMock()
    bridge.url = "http://127.0.0.1:1234/mcp"
    bridge.server_name = "sagent_test"
    bridge.has_tools = False
    model._tools_bridge = cast(ToolsBridge, bridge)

    await model._spawn_initialized()

    env = cast(dict[str, str], captured["env"])
    assert env["HOME"] == str(operator_home)
    assert captured["tmpdir"] is None


@pytest.mark.asyncio
async def test_stateless_named_account_uses_isolated_file_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Named accounts retain the explicit file-isolation behavior."""
    base = tmp_path / ".credentials.json"
    named = tmp_path / ".credentials-work.json"
    named.write_text(json.dumps(_CRED_PAYLOAD), encoding="utf-8")
    isolated = tmp_path / "isolated-home"
    monkeypatch.setattr("sagent.providers.anthropic.cli._CREDS_PATH", base)

    def fake_mkdtemp(**_kwargs: object) -> str:
        return str(isolated)

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.tempfile.mkdtemp",
        fake_mkdtemp,
    )
    captured: dict[str, object] = {}

    class _CaptureSubproc:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured["argv"] = argv
            captured.update(kwargs)

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(anthropic_cli, "Subproc", _CaptureSubproc)
    model = AnthropicCLI(account="work").model("claude-haiku-4-5")
    bridge = MagicMock()
    bridge.url = "http://127.0.0.1:1234/mcp"
    bridge.server_name = "sagent_test"
    bridge.has_tools = False
    model._tools_bridge = cast(ToolsBridge, bridge)

    await model._spawn_initialized()

    env = cast(dict[str, str], captured["env"])
    assert env["HOME"] == str(isolated)
    assert captured["tmpdir"] == isolated
    assert (isolated / ".claude" / ".credentials.json").exists()


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
    assert response.tokens.request == 3
    assert response.tokens.cache_write == 1_200
    assert response.tokens.cache_read == 96_000
    assert response.tokens.response == 450
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
    response = await model._drain_until_result(cast(Subproc, _Proc()), publish=None)
    # Input side = round 2's context footprint, not the 146k sum.
    assert response.tokens.request == 3
    assert response.tokens.cache_read == 96_000
    assert response.tokens.response == 70
    assert model._last_input_tokens == 96_003


@pytest.mark.asyncio
async def test_drain_zero_round_preserves_prior_input_token_anchor() -> None:
    """A zero-round drain (no ``message_start``) must not zero the anchor.

    ``_last_input_tokens`` drives the context-fraction respawn gate. A turn
    that emits a ``result`` with no preceding ``message_start`` carries no new
    footprint; overwriting the last known value with 0 would read as "context
    empty" and suppress a respawn the still-full context needs. The prior
    anchor must survive.
    """
    events = [
        cast(
            MutableJSON,
            {
                "type": "result",
                "stop_reason": "end_turn",
                "is_error": False,
                "modelUsage": {
                    "claude-haiku-4-5": {
                        "inputTokens": 0,
                        "outputTokens": 0,
                        "costUSD": 0.0,
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
    model._last_input_tokens = 180_000  # a genuinely full prior context
    _ = await model._drain_until_result(cast(Subproc, _Proc()), publish=None)
    assert model._last_input_tokens == 180_000, "zero-round drain wiped the anchor"


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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )

    def _which_claude(name: str) -> str | None:
        del name
        return "/usr/bin/claude"

    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
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
    # Place the JSONL where the provider computes the path: the resolved
    # cwd encoded into the project dir, under the tmp HOME.
    jsonl = _session_jsonl_path(sid, cwd=Path.cwd(), home=tmp_path)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("{}\n", encoding="utf-8")
    assert jsonl.exists()

    # Stage 2: simulate post-``agent.clear()`` call: history is empty
    # but the provider's counters still think 2 messages were sent.
    request = ModelRequest(
        system="terse",
        messages=[],
        tools=[],
    )
    response = await model.stream(request, publish=None)

    # The response is a no-op (empty assistant text, no tools, zero
    # cost) so the runtime gets a clean "model said nothing" turn.
    assert response.message.text == ""
    assert response.message.tool_calls == ()
    assert response.tokens.request == 0
    assert response.tokens.response == 0

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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
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
    model._sync_tools_bridge = lambda request, publish=None: bridge_calls.append(  # ty: ignore[invalid-assignment]
        ("sync", request, publish)
    )

    fake_proc = MagicMock()
    fake_proc.close = AsyncMock()
    model._spawn_initialized = AsyncMock(return_value=fake_proc)

    async def _send_entry(proc: object, entry: TapeEvent) -> None:
        del proc
        sent_entries.append(entry)

    model._send_entry = _send_entry  # ty: ignore[invalid-assignment]

    drain_calls = 0

    async def _drain(
        proc: object,
        publish: object = None,
        update_input_tokens: bool = True,
    ):
        del proc, publish, update_input_tokens
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
        await model.stream(request, publish=None)

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


class _FakeBridge:
    """Minimal ``ToolsBridge`` stand-in exposing the detached-drain surface.

    The provider only touches ``drain_detached_results`` /
    ``update_tools`` / ``start`` / ``stop`` (and, for the MCP-connect
    gate, ``has_tools`` / ``listed_snapshot`` / ``wait_listed``) here, so
    we stub exactly those rather than binding a real loopback socket.
    """

    def __init__(
        self,
        pending: list[ToolResult],
        *,
        has_tools: bool = False,
        will_list: bool = True,
    ) -> None:
        self._pending = pending
        self.url = "http://127.0.0.1:0/mcp"
        self.server_name = "sagent"
        self.drain_calls = 0
        self._has_tools = has_tools
        self._will_list = will_list
        self.wait_calls: list[int] = []

    def drain_detached_results(self) -> list[ToolResult]:
        self.drain_calls += 1
        out, self._pending = self._pending, []
        return cast(list[ToolResult], out)  # pyright: ignore[reportUnnecessaryCast] -- ty needs the cast; pyright resolves the type

    def update_tools(self, tools: object) -> None:
        del tools

    def set_publish(self, publish: object) -> None:
        del publish

    @property
    def has_tools(self) -> bool:
        return self._has_tools

    def listed_snapshot(self) -> int:
        return 0

    async def wait_listed(self, since: int, timeout_sec: float) -> bool:
        del timeout_sec
        self.wait_calls.append(since)
        return self._will_list

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_await_mcp_listed_raises_when_catalog_never_connects() -> None:
    """When tools are expected but the CLI never fetches the catalog,
    ``_await_mcp_listed`` raises ``SubprocessTransportError`` so the turn
    respawns instead of silently running tool-less.
    """
    model = AnthropicCLI().model("claude-haiku-4-5")
    bridge = _FakeBridge([], has_tools=True, will_list=False)
    model._tools_bridge = cast(ToolsBridge, bridge)
    proc = cast(Subproc, object())
    with pytest.raises(SubprocessTransportError, match=r"never.*connected"):
        await model._await_mcp_listed(proc)


@pytest.mark.asyncio
async def test_await_mcp_listed_noop_without_tools() -> None:
    """No tools advertised -> no wait, no raise (the gate is opt-in)."""
    model = AnthropicCLI().model("claude-haiku-4-5")
    bridge = _FakeBridge([], has_tools=False)
    model._tools_bridge = cast(ToolsBridge, bridge)
    proc = cast(Subproc, object())
    await model._await_mcp_listed(proc)  # must not raise
    assert bridge.wait_calls == []  # never even waited


@pytest.mark.asyncio
async def test_await_mcp_listed_uses_per_proc_baseline() -> None:
    """Each proc's wait uses ITS OWN snapshot, not a shared field clobbered
    by a concurrent spare-warm.
    """
    model = AnthropicCLI().model("claude-haiku-4-5")
    bridge = _FakeBridge([], has_tools=True, will_list=True)
    model._tools_bridge = cast(ToolsBridge, bridge)
    proc_a = cast(Subproc, object())
    proc_b = cast(Subproc, object())
    # Two procs spawned with different recorded baselines.
    model._mcp_baseline_by_proc[id(proc_a)] = 5
    model._mcp_baseline_by_proc[id(proc_b)] = 9
    await model._await_mcp_listed(proc_a)
    await model._await_mcp_listed(proc_b)
    # Each wait used its own proc's baseline, in order.
    assert bridge.wait_calls == [5, 9]
    # Consumed baselines are pruned.
    assert id(proc_a) not in model._mcp_baseline_by_proc
    assert id(proc_b) not in model._mcp_baseline_by_proc


def test_detached_delivery_entry_folds_results_into_one_user_message() -> None:
    """``_detached_delivery_entry`` folds drained detached results into one
    ``UserMessage``; errors are tagged; ``None`` when nothing is pending.
    """
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5", session_id="sess-detached-1")

    # No bridge yet -> nothing to deliver.
    assert model._tools_bridge is None
    assert model._detached_delivery_entry() is None

    bridge = _FakeBridge(
        [
            ToolResult(call_id="", content="websearch: 3 hits"),
            ToolResult(call_id="", content="paperfetch failed", is_error=True),
        ]
    )
    model._tools_bridge = cast(ToolsBridge, bridge)

    entry = model._detached_delivery_entry()
    assert isinstance(entry, UserMessage)
    assert entry.text == (
        "[detached tool result] websearch: 3 hits\n\n"
        "[detached tool result] paperfetch failed (error)"
    )
    # Drained from the bridge exactly once.
    assert bridge.drain_calls == 1


def test_detached_delivery_entry_holds_buffer_until_turn_succeeds() -> None:
    """The drained text is held (not re-drained) across calls until the
    delivering turn clears it.

    ``drain_detached_results`` is destructive, so a failed-then-retried
    delivery turn must re-present the SAME results rather than losing them
    (the model was promised them). The buffer survives repeat calls and is
    only dropped when ``_pending_detached_text`` is reset on turn success.
    """
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5", session_id="sess-detached-2")
    bridge = _FakeBridge([ToolResult(call_id="", content="staged")])
    model._tools_bridge = cast(ToolsBridge, bridge)

    first = model._detached_delivery_entry()
    assert isinstance(first, UserMessage)
    assert first.text == "[detached tool result] staged"
    assert bridge.drain_calls == 1

    # Simulate a failed delivery turn: the buffer is still held, so a
    # retry re-presents the same entry WITHOUT re-draining the (now empty)
    # bridge.
    second = model._detached_delivery_entry()
    assert isinstance(second, UserMessage)
    assert second.text == "[detached tool result] staged"
    assert bridge.drain_calls == 1  # not re-drained

    # Turn succeeds -> buffer cleared -> nothing pending.
    model._pending_detached_text = None
    assert model._detached_delivery_entry() is None
    assert bridge.drain_calls == 2  # now consults the (empty) bridge again


def test_clear_drops_pending_detached_buffer() -> None:
    """``agent.clear()`` must wipe a buffered detached result.

    The buffer holds tool results the model was promised. After a clear the
    model no longer knows those tool calls, so re-injecting the buffer would
    surface a phantom ``[detached tool result]`` referencing calls that no
    longer exist in context. Both reset boundaries (session clear and respawn)
    must drop it.
    """
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5", session_id="sess-clear-detached")
    model._pending_detached_text = "[detached tool result] staged"
    model._reset_for_clear()
    assert model._pending_detached_text is None, "session clear must drop the buffer"

    model._pending_detached_text = "[detached tool result] staged"
    model._reset_active_state()
    assert model._pending_detached_text is None, "respawn must drop the buffer"


@pytest.mark.asyncio
async def test_session_persistent_delivers_detached_result_as_trailing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn with no new tape entries but a pending detached result still
    spawns: the result is sent as a trailing synthetic entry that does
    NOT advance ``_last_sent_index`` (it is not part of
    ``request.messages``).

    This is the Model side of the bridge's internal cohort: a background
    tool that finished since the last turn is fed back as ordinary turn
    input via ``--resume``, fulfilling the tool contract without the
    runtime ever seeing the detached ``ToolCall``.
    """
    _write_creds(tmp_path)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    sid = "deadbeef-1234-5678-9abc-detached0001"
    provider = AnthropicCLI.from_credentials()
    model = provider.model("claude-haiku-4-5", session_id=sid)
    # A prior turn already established the session; history is fully sent.
    model._session_initialized = True
    model._last_sent_index = 1

    # Bridge already exists with one completed detached run pending.
    bridge = _FakeBridge([ToolResult(call_id="", content="BACKGROUND DONE")])
    model._tools_bridge = cast(ToolsBridge, bridge)

    # ``_ensure_tools_bridge`` must not re-create the bridge; spawn/send/drain
    # are stubbed so no real subprocess starts.
    async def _ensure() -> None:
        return None

    monkeypatch.setattr(model, "_ensure_tools_bridge", _ensure)
    monkeypatch.setattr(model, "_sync_tools_bridge", _noop_sync_tools_bridge)

    fake_proc = MagicMock()
    fake_proc.close = AsyncMock()
    monkeypatch.setattr(model, "_spawn_initialized", AsyncMock(return_value=fake_proc))

    sent_entries: list[TapeEvent] = []

    async def _send_entry(proc: object, entry: TapeEvent) -> None:
        del proc
        sent_entries.append(entry)

    monkeypatch.setattr(model, "_send_entry", _send_entry)

    async def _drain(
        proc: object,
        publish: object = None,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        del proc, publish, update_input_tokens
        return ModelResponse(
            message=AssistantMessage(text="ack", tool_calls=()),
            stop_reason="model_finished",
        )

    monkeypatch.setattr(model, "_drain_until_result", _drain)

    # ``request.messages`` carries only the already-sent history (no NEW
    # tape entry), but a detached result is pending.
    request = ModelRequest(
        system="x",
        messages=[UserMessage(text="old turn 1")],
        tools=[],
    )
    response = await model.stream(request, publish=None)

    # The detached result was delivered as the sole trailing entry.
    assert len(sent_entries) == 1
    delivered = sent_entries[0]
    assert isinstance(delivered, UserMessage)
    assert delivered.text == "[detached tool result] BACKGROUND DONE"

    # It did NOT advance the cumulative counter past the real history
    # length (the synthetic entry is not part of ``request.messages``).
    assert model._last_sent_index == len(request.messages)
    assert response.message.text == "ack"


@pytest.mark.asyncio
async def test_stateless_exchange_delivers_pending_detached_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The STATELESS path also drains + delivers detached results.

    The bridge advertises ``background``/``delay`` in both modes, so a
    stateless turn can stage a detached run whose result must be fed back.
    ``_exchange_turn`` appends the drained result as a trailing synthetic
    user entry; without this the model's promised delivery never arrives
    and ``_bg_done`` leaks.
    """
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    assert model._session_id is None  # stateless

    bridge = _FakeBridge([ToolResult(call_id="", content="BG RESULT")])
    model._tools_bridge = cast(ToolsBridge, bridge)

    sent_entries: list[TapeEvent] = []

    async def _send_entry(proc: object, entry: TapeEvent) -> None:
        del proc
        sent_entries.append(entry)

    async def _drain(
        proc: object,
        publish: object = None,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        del proc, publish, update_input_tokens
        return ModelResponse(
            message=AssistantMessage(text="ack", tool_calls=()),
            stop_reason="model_finished",
        )

    monkeypatch.setattr(model, "_send_entry", _send_entry)
    monkeypatch.setattr(model, "_drain_until_result", _drain)

    # A real user turn plus a pending detached result.
    request = ModelRequest(
        system="x",
        messages=[UserMessage(text="hello")],
        tools=[],
    )
    fake_proc = cast(Subproc, object())
    response = await model._exchange_turn(fake_proc, request, publish=None)

    # Both the real user entry and the detached delivery were sent, in
    # order, with the detached result trailing.
    assert len(sent_entries) == 2
    assert isinstance(sent_entries[0], UserMessage)
    assert sent_entries[0].text == "hello"
    assert isinstance(sent_entries[1], UserMessage)
    assert sent_entries[1].text == "[detached tool result] BG RESULT"
    assert response.message.text == "ack"


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
            (
                "[ede_diagnostic] result_type=user last_content_type=n/a "
                "stop_reason=tool_use"
            ),
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
    unknown_error: dict[str, object] = {
        "is_error": True,
        "errors": [],
        "stop_reason": "end_turn",
    }
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
        "sagent.providers.anthropic.cli._CREDS_PATH",
        tmp_path / ".credentials.json",
    )
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
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

    def publish(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponsePartial):
            text_chunks.append(ev.text)
        elif isinstance(ev, ModelResponseThinking):
            thinking_chunks.append(ev.text)

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
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=publish,
    )
    _dispatch_stream_event(
        thinking_event,
        text_parts,
        thinking_parts,
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=publish,
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
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=None,
    )
    assert signature_parts == ["abc123"]
    assert text_parts == []
    assert thinking_parts == []


def test_dispatch_stream_event_publishes_rich_tool_label_at_stop() -> None:
    """tool_use is published at content_block_stop with name + arg
    summary, after the streamed input_json_delta has been accumulated.
    """
    published: list[RuntimeEvent] = []
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
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=published.append,
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
            signature_parts=signature_parts,
            tool_use_blocks=tool_use_blocks,
            publish=published.append,
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
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=published.append,
    )
    assert len(published) == 1
    label = published[0]
    assert isinstance(label, ToolLabel)
    # Includes both the tool name and the command arg
    assert label.text == "Bash ls -la"
    assert label.call_id == "toolu_abc123"


def test_dispatch_stream_event_no_label_for_text_block_start() -> None:
    """``content_block_start`` for ``text`` does NOT publish a ToolLabel."""
    published: list[RuntimeEvent] = []
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
        signature_parts=[],
        tool_use_blocks=tool_use_blocks,
        publish=published.append,
    )
    # And a content_block_stop on the text block: no label.
    _dispatch_stream_event(
        cast(MutableJSON, {"type": "content_block_stop", "index": 1}),
        [],
        [],
        signature_parts=[],
        tool_use_blocks=tool_use_blocks,
        publish=published.append,
    )
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
        signature_parts=signature_parts,
        tool_use_blocks=tool_use_blocks,
        publish=None,
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
    assert response.tokens.request == 0
    # Output IS cumulative: every internal round's generation was produced.
    assert response.tokens.response == 60
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
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        del proc, publish
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
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        del proc, request, publish
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
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        del proc, publish
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
        publish: Callable[[RuntimeEvent], None] | None,
        *,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        del proc, publish, update_input_tokens
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
        publish=None,
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
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        del proc, publish
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
        publish=None,
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
            publish=None,
        )

    assert model._last_input_tokens == 7


def test_serialize_for_stdin_user_passthrough() -> None:
    """``UserMessage`` falls through ``_serialize_for_stdin`` to ``_user_line``."""
    line = _serialize_for_stdin(
        UserMessage(text="ping"), max_image_dim=8000, max_image_bytes=5 * 1024 * 1024
    )
    assert line["type"] == "user"


def test_interrupt_active_proc_returns_false_when_no_active_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No active hot-spare ⇒ ``_interrupt_active_proc`` returns False without raising."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr("sagent.providers.anthropic.cli._CREDS_PATH", creds)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
    provider = AnthropicCLI.from_credentials()
    model = provider.model("claude-opus-4-7")
    # Hot spare has not been acquired -> ``active`` is None.
    assert model._hot_spare is not None
    assert model._hot_spare.active is None
    assert model._interrupt_active_proc() is False


def test_interrupt_active_proc_signals_active_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Active hot-spare ⇒ ``_interrupt_active_proc`` forwards to ``Subproc.interrupt``."""
    creds = _write_creds(tmp_path)
    monkeypatch.setattr("sagent.providers.anthropic.cli._CREDS_PATH", creds)
    monkeypatch.setattr(
        "sagent.providers.anthropic.cli.shutil.which",
        _which_claude_stub,
    )
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
    assert model._interrupt_active_proc() is True
    assert fake.interrupt_calls == 1


_ = ToolCall


# ----------------------------------------------------------------------
# Cancel-through-stream(): the mid-turn preempt path.
#
# The runtime preempts by cancelling the model-call task, which raises
# ``CancelledError`` INTO ``model.stream``. The provider must SIGINT the
# subprocess (``_interrupt_active_proc``) and re-raise so the runtime's
# cancellation path runs. Earlier tests only exercised the leaf helper;
# these drive the real ``stream`` branches.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stateless_stream_cancelled_interrupts_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CancelledError`` mid-stream → SIGINT the active proc, respawn, re-raise."""
    provider = AnthropicCLI()
    model = provider.model("claude-haiku-4-5")
    interrupted = False
    respawn_count = 0

    class _CancelProc:
        def interrupt(self) -> bool:
            nonlocal interrupted
            interrupted = True
            return True

        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            raise asyncio.CancelledError

    proc = _CancelProc()

    class _HotSpare:
        active = cast(Subproc | None, proc)

        async def acquire(self) -> Subproc:
            return cast(Subproc, proc)

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, proc)

    model._system_hash = _hash_system(None)
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])

    with pytest.raises(asyncio.CancelledError):
        await model.stream(request)

    assert interrupted is True, "stream() must SIGINT the active proc on cancel"
    assert respawn_count == 1, "stream() must respawn after a cancelled turn"
    assert model._last_sent_index == 0


@pytest.mark.asyncio
async def test_session_persistent_stream_cancelled_interrupts_and_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-mode ``CancelledError`` → SIGINT active proc, close, re-raise."""
    monkeypatch.setenv("HOME", str(tmp_path))
    provider = AnthropicCLI()
    sid = "11111111-2222-3333-4444-555555555555"
    model = provider.model("claude-haiku-4-5", session_id=sid)
    interrupted = False
    closed = False

    class _CancelProc:
        def interrupt(self) -> bool:
            nonlocal interrupted
            interrupted = True
            return True

        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            raise asyncio.CancelledError

        async def close(self) -> None:
            nonlocal closed
            closed = True

    proc = _CancelProc()

    async def _fake_spawn() -> Subproc:
        return cast(Subproc, proc)

    monkeypatch.setattr(model, "_spawn_initialized", _fake_spawn)
    monkeypatch.setattr(model, "_ensure_tools_bridge", AsyncMock())
    monkeypatch.setattr(model, "_sync_tools_bridge", _noop_sync_tools_bridge)
    request = ModelRequest(messages=[UserMessage(text="hi")])

    with pytest.raises(asyncio.CancelledError):
        await model.stream(request)

    assert interrupted is True, "session stream must SIGINT the active proc on cancel"
    assert closed is True, "session stream must close the proc on cancel"
    assert model._active_proc is None


# ----------------------------------------------------------------------
# Path resolution: CLAUDE_CONFIG_DIR + the session-jsonl encoded path.
# ----------------------------------------------------------------------


def test_real_home_honors_claude_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_real_home`` returns the parent of ``CLAUDE_CONFIG_DIR`` when set.

    claude stores ``projects/`` under ``$CLAUDE_CONFIG_DIR`` (treated as
    the ``.claude`` dir). ``_real_home`` returns its parent so the shared
    ``/.claude/projects`` suffix in ``_session_jsonl_path`` resolves to
    claude's actual projects root.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/custom/cfg/.claude")
    assert _real_home() == Path("/custom/cfg")


def test_real_home_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``CLAUDE_CONFIG_DIR``, ``_real_home`` uses ``$HOME``."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/operator")
    assert _real_home() == Path("/home/operator")


def test_session_jsonl_path_under_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: with ``CLAUDE_CONFIG_DIR`` the session path lands under it."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/custom/cfg/.claude")
    sid = "abc12345-aaaa-bbbb-cccc-dddddddddddd"
    path = _session_jsonl_path(sid, cwd=Path("/work/repo"), home=_real_home())
    assert path == Path(
        f"/custom/cfg/.claude/projects/-work-repo/{sid}.jsonl",
    )


# ----------------------------------------------------------------------
# Multi-turn session lifecycle: turn 1 mints (--session-id), turn 2
# resumes (--resume). Verifies the _session_initialized flip + that the
# argv switches accordingly across two real stream() turns.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_persistent_two_turns_mint_then_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn 1 uses ``--session-id`` (mint); turn 2 uses ``--resume``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    provider = AnthropicCLI()
    sid = "22222222-3333-4444-5555-666666666666"
    model = provider.model("claude-haiku-4-5", session_id=sid)
    resume_existing_seen: list[bool] = []

    class _OkProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            # Minimal terminal result event: one round, clean stop.
            return {
                "type": "result",
                "stop_reason": "end_turn",
                "modelUsage": {},
                "session_id": "s",
            }

        async def close(self) -> None:
            return

    # Capture the resume_existing flag the argv builder would receive by
    # intercepting the spawn (which reads model._session_initialized).
    async def _fake_spawn() -> Subproc:
        resume_existing_seen.append(model._session_initialized)
        return cast(Subproc, _OkProc())

    monkeypatch.setattr(model, "_spawn_initialized", _fake_spawn)
    monkeypatch.setattr(model, "_ensure_tools_bridge", AsyncMock())
    monkeypatch.setattr(model, "_sync_tools_bridge", _noop_sync_tools_bridge)

    # Turn 1: fresh session → mint (_session_initialized False at spawn).
    await model.stream(ModelRequest(messages=[UserMessage(text="one")]))
    # Turn 2: same session, one new entry → resume.
    await model.stream(
        ModelRequest(
            messages=[
                UserMessage(text="one"),
                AssistantMessage(text="ok"),
                UserMessage(text="two"),
            ],
        ),
    )

    assert resume_existing_seen == [False, True], (
        "turn 1 must mint (--session-id), turn 2 must resume (--resume); "
        f"got {resume_existing_seen}"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
