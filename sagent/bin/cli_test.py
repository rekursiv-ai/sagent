"""Tests for ``bin.cli``: arg parsing + event serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import argparse
import asyncio
import dataclasses
import io
import json
import logging
import os
import subprocess
import sys

import pytest

from sagent.agent import Agent
from sagent.agent.session_io import (
    PersistentAgentRecord,
    SessionMeta,
)
from sagent.agent.state import agent_registry
from sagent.bin.cli import (
    _DEFAULT_PROVIDER,
    DEFAULT_TOOLS,
    _apply_resume_model_defaults,
    _build_persistent_child,
    _build_provider_model_once,
    _cli_provider_options,
    _configure_logging,
    _default_allow_providers,
    _event_to_json_record,
    _install_repl_logging,
    _last_assistant_text,
    _parse_allow_providers,
    _parse_cli_args,
    _parse_stream_json,
    _resolve_cli_thinking_state,
    _resolve_provider_and_allow,
    _resolve_session_dir,
    _resume_label,
    _run_headless,
    parse_agent_args,
    resolve_tools,
)
from sagent.providers import PROVIDER_NAMES
from sagent.testing import FakeAgent
from sagent.types.providers import ProviderOptions
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ModelServiceSuspended,
    ServiceErrorSnapshot,
    ToolLabel,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


def _parse(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    namespace, _ = _parse_cli_args(parser, list(args))
    return namespace


def test_parse_cli_args_defaults() -> None:
    ns = _parse([])
    assert ns.provider  # any non-empty default
    assert ns.auth  # any non-empty default
    assert ns.output_format == "text"
    assert ns.input_format == "text"
    assert ns.continue_ is False
    assert ns.resume is None
    assert ns.tools is None
    assert ns.add_dir == []
    assert ns.compact is True


def test_parse_cli_args_no_compact() -> None:
    ns = _parse(["--no-compact"])
    assert ns.compact is False


def test_parse_cli_args_thinking_default() -> None:
    ns = _parse([])
    assert ns.thinking == "default"
    assert _resolve_cli_thinking_state(ns) is None


def test_parse_cli_args_thinking_full_state() -> None:
    ns = _parse(["--thinking", "on-show"])
    assert _resolve_cli_thinking_state(ns) == "on-show"


def test_cli_provider_options_default_unset() -> None:
    ns = _parse([])
    assert ns.server_side_context_management is None
    assert _cli_provider_options(ns).set_fields() == {}


def test_cli_provider_options_server_side_context_management_flag() -> None:
    ns = _parse(["--server-side-context-management"])
    assert _cli_provider_options(ns).set_fields() == {
        "server_side_context_management": True,
    }


def test_cli_provider_options_negated_flag_is_explicit_false() -> None:
    ns = _parse(["--no-server-side-context-management"])
    assert _cli_provider_options(ns).set_fields() == {
        "server_side_context_management": False,
    }


@dataclasses.dataclass(slots=True, kw_only=True)
class _ChildStubModel:
    """Minimal rich-model stand-in for ``_build_persistent_child`` tests."""

    model_id: str = "claude-opus-4-8"
    max_request_tokens: int = 100_000
    max_response_tokens: int = 1_024
    supports_thinking: bool = True
    supports_effort: bool = False
    valid_efforts: tuple[str, ...] = ()
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()


def _child_record(**overrides: object) -> PersistentAgentRecord:
    record = PersistentAgentRecord(
        label="fix-tools",
        run_id="run-1",
        session_dir="",
        state="running",
        provider="Anthropic",
        auth="env",
        account=None,
        model_id="claude-opus-4-8",
        tools=(),
        system="system text",
        notify_on_asleep=True,
    )
    return dataclasses.replace(record, **overrides)


def _capture_build_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, ProviderOptions | None]:
    """Stub ``cli.build_provider``; capture the forwarded ``options``."""
    captured: dict[str, ProviderOptions | None] = {}

    def fake_build_provider(
        provider_name: str,
        auth: str = "env",
        *,
        account: str | None = None,
        options: ProviderOptions | None = None,
    ) -> object:
        del provider_name, auth, account
        captured["options"] = options

        class _Provider:
            def model(self, model_id: str | None = None) -> _ChildStubModel:
                return _ChildStubModel(model_id=model_id or "claude-opus-4-8")

        return _Provider()

    monkeypatch.setattr(
        "sagent.bin.cli.build_provider",
        fake_build_provider,
    )
    return captured


def test_build_persistent_child_forwards_provider_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume must rebuild the provider with the record's construction options."""
    captured = _capture_build_provider(monkeypatch)
    record = _child_record(
        provider_options=ProviderOptions(server_side_context_management=True),
    )
    child = _build_persistent_child(record, allow_providers=(), parent_label="parent")
    assert captured["options"] == ProviderOptions(server_side_context_management=True)
    assert child.provider_options == record.provider_options


def test_build_persistent_child_restores_thinking_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical thinking state round-trips, including derived redaction."""
    captured = _capture_build_provider(monkeypatch)
    record = _child_record(thinking="adaptive", thinking_state="redact-hide")
    child = _build_persistent_child(record, allow_providers=(), parent_label="parent")
    assert child.thinking_state == "redact-hide"
    assert child.show_thinking is False
    assert child.thinking == "adaptive"
    # Anthropic supports the redact option, so the provider rebuild
    # derives it from the restored thinking state.
    options = captured["options"]
    assert options is not None
    assert options.redact_thinking is True


def test_default_allow_providers_leads_with_default_provider() -> None:
    """Default allow-list's first entry is the zero-flag default provider."""
    out = _default_allow_providers()
    assert out[0] == _DEFAULT_PROVIDER
    assert set(out) == set(PROVIDER_NAMES)
    assert len(out) == len(PROVIDER_NAMES)  # no dup


def test_implicit_provider_derives_auth_from_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth follows the resolved provider even when ``--auth`` is implicit.

    The default provider now comes from the allow-list, so its auth must
    be derived from the provider rather than the standalone ``--auth``
    default (which would mismatch, e.g. ``Anthropic`` wants ``env`` but
    the default auth is ``credentials``).
    """
    ns = _parse([])
    ns.provider = "Anthropic"  # simulate allow-list default selection

    class _Provider:
        def model(self, model_id: str | None = None) -> object:
            return argparse.Namespace(model_id=model_id or "m")

    calls: list[tuple[str, str]] = []

    def fake_build_provider(
        provider_name: str,
        auth: str,
        *,
        account: str | None = None,
        **extra: object,
    ) -> object:
        del account, extra
        calls.append((provider_name, auth))
        return _Provider()

    monkeypatch.setattr(
        "sagent.bin.cli.build_provider",
        fake_build_provider,
    )

    _, _, auth = _build_provider_model_once(ns, None)

    assert calls == [("Anthropic", "env")]
    assert auth == "env"


def test_parse_cli_args_session_paths() -> None:
    ns = _parse(["--session", "/tmp/x"])  # noqa: S108 -- test arg string
    assert ns.session == "/tmp/x"  # noqa: S108 -- test arg string


def test_parse_cli_args_continue() -> None:
    ns = _parse(["--continue"])
    assert ns.continue_ is True


def test_parse_cli_args_resume_no_value() -> None:
    ns = _parse(["--resume"])
    assert ns.resume is True


def test_parse_cli_args_resume_with_hash() -> None:
    ns = _parse(["--resume", "abc123"])
    assert ns.resume == "abc123"


def test_parse_cli_args_resume_persistent_defaults_enabled() -> None:
    ns = _parse([])
    assert ns.resume_persistent is True


def test_parse_cli_args_no_resume_persistent() -> None:
    ns = _parse(["--no-resume-persistent"])
    assert ns.resume_persistent is False


def test_resume_label_renames_collision() -> None:
    agent_registry["fix-tools"] = FakeAgent()
    try:
        assert _resume_label("fix-tools") == "fix-tools_1"
    finally:
        agent_registry.pop("fix-tools", None)


def test_resume_model_defaults_use_session_meta_when_no_explicit_flags() -> None:
    ns = _parse(["--resume", "abc123"])
    meta = SessionMeta(
        provider="AnthropicCLI",
        auth="subprocess",
        model_id="opus-4-7+1m",
        account="work",
    )
    _apply_resume_model_defaults(ns, meta)
    assert ns.provider == "AnthropicCLI"
    assert ns.auth == "subprocess"
    assert ns.model == "opus-4-7+1m"
    assert ns.account == "work"


def test_resume_model_defaults_explicit_provider_uses_provider_default_model() -> None:
    ns = _parse(["--resume", "abc123", "--provider", "OpenAISubscription"])
    meta = SessionMeta(
        provider="AnthropicCLI",
        auth="subprocess",
        model_id="opus-4-7+1m",
        account="work",
    )
    _apply_resume_model_defaults(ns, meta)
    assert ns.provider == "OpenAISubscription"
    assert ns.auth == "subprocess"
    assert ns.model is None
    assert ns.account == "work"


def test_resume_model_defaults_explicit_model_overrides_session_meta() -> None:
    ns = _parse(
        ["--resume", "abc123", "--provider", "OpenAISubscription", "--model", "gpt-5.5"]
    )
    meta = SessionMeta(
        provider="AnthropicCLI",
        auth="subprocess",
        model_id="opus-4-7+1m",
    )
    _apply_resume_model_defaults(ns, meta)
    assert ns.provider == "OpenAISubscription"
    assert ns.model == "gpt-5.5"


def test_resume_model_defaults_explicit_provider_keeps_fast_tagged_model() -> None:
    """A ``+fast`` id is not a catalog key; the check must strip it.

    Regression: exact ``KNOWN_MODELS`` membership nulled the persisted
    model on an explicit same-provider resume, silently dropping the
    fast tag (and the model choice) in favor of the provider default.
    """
    ns = _parse(["--resume", "abc123", "--provider", "Anthropic"])
    meta = SessionMeta(
        provider="Anthropic",
        auth="env",
        model_id="claude-opus-4-8+1m+fast",
    )
    _apply_resume_model_defaults(ns, meta)
    assert ns.model == "claude-opus-4-8+1m+fast"


def test_parse_agent_args_known_unknown_split() -> None:
    parser = argparse.ArgumentParser()
    ns, unknown = parse_agent_args(parser, ["--model", "x", "extra-token"])
    assert ns.model == "x"
    assert unknown == ["extra-token"]


def test_parse_cli_args_output_format_choices() -> None:
    ns = _parse(["--output-format", "json"])
    assert ns.output_format == "json"


def test_parse_cli_args_invalid_output_format() -> None:
    with pytest.raises(SystemExit):
        _parse(["--output-format", "weird"])


def test_resolve_tools_none_returns_empty() -> None:
    assert resolve_tools(["none"]) == []


def test_resolve_tools_unknown_raises() -> None:
    with pytest.raises(SystemExit, match="unknown tool"):
        _ = resolve_tools(["NoSuchTool"])


def test_resolve_tools_known_subset_includes_bash() -> None:
    tools = resolve_tools(["Read", "Bash"])
    assert len(tools) == 2
    names = [t.name for t in tools]
    assert "Read" in names
    assert "Bash" in names
    # Bash is constructed last regardless of position; the order in
    # ``names`` follows the request order.
    assert names == ["Read", "Bash"]


def test_last_assistant_text_empty() -> None:
    assert _last_assistant_text([]) == ""


def test_last_assistant_text_returns_most_recent() -> None:
    history: list[ModelContextEvent] = [
        AssistantMessage(text="first"),
        UserMessage(text="user"),
        AssistantMessage(text="latest"),
    ]
    assert _last_assistant_text(history) == "latest"


def test_last_assistant_text_no_assistant_returns_empty() -> None:
    history: list[ModelContextEvent] = [UserMessage(text="hi")]
    assert _last_assistant_text(history) == ""


def test_event_to_json_record_partial_text() -> None:
    rec = _event_to_json_record(ModelResponsePartial(text="hello"))
    assert rec == {"descriptor": "text/plain", "content": "hello"}


def test_event_to_json_record_thinking() -> None:
    rec = _event_to_json_record(ModelResponseThinking(text="hmm"))
    assert rec == {"descriptor": "text/x-thinking", "content": "hmm"}


def test_event_to_json_record_tool_label() -> None:
    rec = _event_to_json_record(ToolLabel(call_id="c1", text="Bash"))
    assert rec == {
        "descriptor": "text/x-tool-label",
        "content": "Bash",
        "call_id": "c1",
    }


def test_event_to_json_record_tool_result() -> None:
    rec = _event_to_json_record(ToolResult(call_id="c1", content="ok", is_error=True))
    assert rec == {
        "descriptor": "application/x-tool-result",
        "call_id": "c1",
        "content": "ok",
        "is_error": True,
    }


def test_event_to_json_record_model_error() -> None:
    rec = _event_to_json_record(
        ModelResponseError(exception=RuntimeError("creds")),
    )
    assert rec == {
        "descriptor": "application/x-error",
        "content": "RuntimeError: creds",
    }


class _HeadlessErrorAgent:
    def __init__(self) -> None:
        self.history: list[ModelContextEvent] = []

    async def run(
        self,
        message: UserMessage,
    ) -> AsyncIterator[ModelResponseError]:
        del message
        yield ModelResponseError(exception=RuntimeError("boom"))


@pytest.mark.parametrize(
    ("output_format", "stdout", "stderr"),
    [
        ("text", "", "Error: RuntimeError: boom\n"),
        ("json", '{"error": "RuntimeError: boom"}\n', ""),
        (
            "stream-json",
            '{"descriptor": "application/x-error", "content": "RuntimeError: boom"}\n',
            "",
        ),
    ],
)
def test_run_headless_model_error_exits_nonzero(
    output_format: str,
    stdout: str,
    stderr: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello"))
    agent = cast(Agent, _HeadlessErrorAgent())

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(
            _run_headless(
                agent,
                input_format="text",
                output_format=output_format,
            )
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == stdout
    assert captured.err == stderr


def test_event_to_json_record_model_service_suspended() -> None:
    rec = _event_to_json_record(
        ModelServiceSuspended(
            provider="anthropic",
            auth="key",
            account="default",
            model_id="claude-test",
            retry_at=12345.0,
            delay_sec=60.0,
            server_supplied=True,
            error=ServiceErrorSnapshot(
                type_name="RateLimitError", message="429", status=429
            ),
        )
    )
    assert rec == {
        "descriptor": "application/x-model-service-suspended",
        "provider": "anthropic",
        "auth": "key",
        "account": "default",
        "model_id": "claude-test",
        "retry_at": 12345.0,
        "delay_sec": 60.0,
        "server_supplied": True,
        "error": {
            "type_name": "RateLimitError",
            "message": "429",
            "status": 429,
        },
    }


def test_event_to_json_record_unknown_returns_none() -> None:
    # ``UserMessage`` is a runtime event but not serialized for stream-json.
    assert _event_to_json_record(UserMessage(text="hi")) is None


def test_parse_stream_json_single_prompt() -> None:
    raw = json.dumps({"prompt": "do it"})
    assert _parse_stream_json(raw) == "do it"


def test_parse_stream_json_multi_lines_joined() -> None:
    raw = "\n".join(
        [
            json.dumps({"prompt": "first"}),
            "",
            json.dumps({"prompt": "second"}),
        ]
    )
    assert _parse_stream_json(raw) == "first\n\nsecond"


def test_parse_stream_json_invalid_json_raises() -> None:
    with pytest.raises(ValueError, match="invalid JSON line"):
        _ = _parse_stream_json("not json\n")


def test_parse_stream_json_non_object_raises() -> None:
    with pytest.raises(TypeError, match="JSON objects per line"):
        _ = _parse_stream_json(json.dumps(["a", "list"]))


def test_parse_stream_json_missing_prompt_field_silently_skipped() -> None:
    # No ``prompt`` key -> contributes nothing to the joined result.
    assert _parse_stream_json(json.dumps({"other": "data"})) == ""


def test_configure_logging_no_op_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    # Should not raise; should not configure root logger forcibly.
    _configure_logging(None)


def test_configure_logging_invalid_level_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    with pytest.raises(SystemExit, match="invalid log level"):
        _configure_logging("NONSENSE")


def test_configure_logging_env_var_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGENT_LOG_LEVEL", "DEBUG")
    _configure_logging(None)
    # No assertion on logger state -- just verify no raise.


def test_install_repl_logging_silences_stderr(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``_install_repl_logging`` must prevent log records reaching stderr.

    The REPL renders via prompt-toolkit; any logger output to stderr
    corrupts the display. Policy: stderr is for headless mode only.
    This test pins the contract that calling ``_install_repl_logging``
    suppresses both Python's ``lastResort`` stderr fallback and any
    pre-installed stderr-bound handlers on root.
    """
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SAGENT_LOG_FILE", raising=False)

    # Pre-state: a fake stderr handler on root (simulates a prior
    # ``basicConfig`` from headless config bleeding into REPL).
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_last_resort = logging.lastResort
    try:
        root.handlers.clear()
        stderr_handler = logging.StreamHandler()  # defaults to stderr
        root.addHandler(stderr_handler)

        _install_repl_logging(session_dir=tmp_path)

        logging.getLogger("sagent.test").warning("nope")
        captured = capfd.readouterr()
        assert captured.err == "", (
            f"REPL mode must not write to stderr; got {captured.err!r}"
        )
    finally:
        current_handlers = list(root.handlers)
        root.handlers.clear()
        for handler in current_handlers:
            if handler not in saved_handlers:
                handler.close()
        for h in saved_handlers:
            root.addHandler(h)
        logging.lastResort = saved_last_resort


def test_install_repl_logging_routes_debug_to_file_by_default(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REPL mode saves DEBUG diagnostics to a file by default."""
    log_file = tmp_path / "repl.log"
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SAGENT_LOG_FILE", raising=False)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_last_resort = logging.lastResort
    try:
        root.handlers.clear()
        _install_repl_logging(session_dir=tmp_path)

        logger = logging.getLogger("sagent.test")
        logger.debug("hello-file")

        captured = capfd.readouterr()
        assert captured.err == "", (
            f"REPL mode must never write to stderr; got {captured.err!r}"
        )
        # Flush handlers so the file content is visible.
        for h in root.handlers:
            h.flush()
        assert "hello-file" in log_file.read_text(), (
            f"expected log content in {log_file}; got {log_file.read_text()!r}"
        )
    finally:
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        logging.lastResort = saved_last_resort


def test_install_repl_logging_cli_level_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI log level still controls the file-backed REPL threshold."""
    log_file = tmp_path / "repl.log"
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    monkeypatch.delenv("SAGENT_LOG_FILE", raising=False)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_last_resort = logging.lastResort
    try:
        root.handlers.clear()
        _install_repl_logging("INFO", session_dir=tmp_path)

        logger = logging.getLogger("sagent.test")
        logger.debug("debug-hidden")
        logger.info("info-visible")

        for h in root.handlers:
            h.flush()
        text = log_file.read_text()
        assert "info-visible" in text
        assert "debug-hidden" not in text
    finally:
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        logging.lastResort = saved_last_resort


def test_install_repl_logging_env_file_overrides_session_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``SAGENT_LOG_FILE`` remains an explicit override for REPL logs."""
    session_dir = tmp_path / "session"
    override = tmp_path / "override" / "repl.log"
    monkeypatch.delenv("SAGENT_LOG_LEVEL", raising=False)
    monkeypatch.setenv("SAGENT_LOG_FILE", str(override))

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_last_resort = logging.lastResort
    try:
        root.handlers.clear()
        _install_repl_logging(session_dir=session_dir)

        logging.getLogger("sagent.test").debug("override-visible")

        for h in root.handlers:
            h.flush()
        assert "override-visible" in override.read_text()
        assert not (session_dir / "repl.log").exists()
    finally:
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        logging.lastResort = saved_last_resort


def test_default_tools_constant() -> None:
    assert "Bash" in DEFAULT_TOOLS
    assert "Read" in DEFAULT_TOOLS


def test_resolve_session_dir_explicit_session() -> None:
    ns = _parse(["--session", "/tmp/explicit"])  # noqa: S108 -- test arg
    assert _resolve_session_dir(ns) == "/tmp/explicit"  # noqa: S108 -- test arg


def test_resolve_session_dir_fresh_when_no_flags() -> None:
    ns = _parse([])
    out = _resolve_session_dir(ns)
    assert isinstance(out, str)
    # Fresh session under the standard projects dir.
    assert out  # non-empty path string


@pytest.mark.ci_smoke
def test_direct_script_bootstraps_dependencies() -> None:
    """Polyglot-shebang `cli.py --help` runs to completion and prints help."""
    script = Path(__file__).resolve().parent / "cli.py"
    proc = subprocess.run(  # noqa: S603 -- known script, hard-coded args
        [str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "UV_FROZEN": "1"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: cli.py")


def test_parse_allow_providers_ok() -> None:
    """A well-formed CSV returns the parsed tuple."""
    assert _parse_allow_providers("Anthropic,OpenAI") == ("Anthropic", "OpenAI")
    assert _parse_allow_providers("  Anthropic ,  OpenAI ") == (
        "Anthropic",
        "OpenAI",
    )


def test_parse_allow_providers_empty_exits() -> None:
    """Empty CSV exits rather than silently allowing only the parent provider."""
    with pytest.raises(SystemExit) as exc:
        _parse_allow_providers("")
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc2:
        _parse_allow_providers(", ,")
    assert exc2.value.code == 1


def test_parse_allow_providers_unknown_exits() -> None:
    """Unknown provider names exit with a clear error."""
    with pytest.raises(SystemExit) as exc:
        _parse_allow_providers("Anthropic,FooBar")
    assert exc.value.code == 1


def test_resolve_provider_and_allow_default_picks_first_allowed() -> None:
    """Default (``primary=None``) provider is the first allowed entry."""
    provider, out = _resolve_provider_and_allow("OpenAI,Anthropic", primary=None)
    assert provider == "OpenAI"
    assert out == ("OpenAI", "Anthropic")


def test_resolve_provider_and_allow_no_dup_when_already_present() -> None:
    """Idempotent: explicit primary already in CSV is a no-op."""
    provider, out = _resolve_provider_and_allow(
        "Anthropic,OpenAI",
        primary="Anthropic",
    )
    assert provider == "Anthropic"
    assert out == ("Anthropic", "OpenAI")


def test_resolve_provider_and_allow_rejects_unknown_explicit_primary() -> None:
    """Unknown explicit primary still fails ``_parse_allow_providers`` validation."""
    with pytest.raises(SystemExit) as exc:
        _resolve_provider_and_allow("Anthropic,OpenAI", primary="NopeNotReal")
    assert exc.value.code == 1


def test_resolve_provider_and_allow_rejects_empty_spec() -> None:
    """Empty allow-list exits regardless of ``primary``."""
    with pytest.raises(SystemExit) as exc:
        _resolve_provider_and_allow("", primary=None)
    assert exc.value.code == 1


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
