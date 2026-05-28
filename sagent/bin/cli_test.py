"""Tests for ``bin.cli``: arg parsing + event serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import argparse
import json
import logging
import os
import subprocess

import pytest

from sagent.agent.session_io import SessionMeta
from sagent.agent.state import agent_registry
from sagent.bin.cli import (
    DEFAULT_TOOLS,
    _apply_resume_model_defaults,
    _configure_logging,
    _event_to_json_record,
    _install_repl_logging,
    _last_assistant_text,
    _parse_allow_providers,
    _parse_cli_args,
    _parse_stream_json,
    _provider_kwargs,
    _resolve_cli_thinking_state,
    _resolve_session_dir,
    _resume_label,
    parse_agent_args,
    resolve_tools,
)
from sagent.testing import FakeAgent
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
    from collections.abc import Sequence


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


def test_parse_cli_args_thinking_precedes_provider_arg() -> None:
    ns = _parse(
        [
            "--provider",
            "OpenAISubscription",
            "--thinking",
            "adaptive-show",
            "--provider-arg",
            "OpenAISubscription.thinking=redact",
        ]
    )
    assert _resolve_cli_thinking_state(ns) == "adaptive-show"


def test_parse_cli_args_provider_arg_thinking_when_cli_default() -> None:
    ns = _parse(
        [
            "--provider",
            "OpenAISubscription",
            "--provider-arg",
            "OpenAISubscription.thinking=redact",
        ]
    )
    assert _resolve_cli_thinking_state(ns) == "redact-hide"


def test_provider_kwargs_removes_thinking_pseudo_arg() -> None:
    assert _provider_kwargs({"thinking": "redact", "redact_thinking": True}) == {
        "redact_thinking": True
    }


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
        root.handlers.clear()
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
    assert "CLI agent (REPL or headless)." in proc.stdout


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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
