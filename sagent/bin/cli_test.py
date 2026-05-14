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

from sagent.agent.runtime import (
    AssistantMessage,
    ModelResponseError,
    ModelResponsePartial,
    ModelResponseThinking,
    ToolLabel,
    ToolResult,
    UserMessage,
)
from sagent.bin.cli import (
    DEFAULT_TOOLS,
    _configure_logging,
    _event_to_json_record,
    _install_repl_logging,
    _last_assistant_text,
    _parse_cli_args,
    _parse_stream_json,
    _resolve_session_dir,
    parse_agent_args,
    resolve_tools,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.agent.runtime import HistoryEntry


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
    history: list[HistoryEntry] = [
        AssistantMessage(text="first"),
        UserMessage(text="user"),
        AssistantMessage(text="latest"),
    ]
    assert _last_assistant_text(history) == "latest"


def test_last_assistant_text_no_assistant_returns_empty() -> None:
    history: list[HistoryEntry] = [UserMessage(text="hi")]
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
    capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
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

        _install_repl_logging()

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


def test_install_repl_logging_routes_to_file_when_log_level_set(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``SAGENT_LOG_LEVEL`` in REPL mode routes records to a file, not stderr."""
    log_file = tmp_path / "repl.log"
    monkeypatch.setenv("SAGENT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SAGENT_LOG_FILE", str(log_file))

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_last_resort = logging.lastResort
    try:
        root.handlers.clear()
        _install_repl_logging()

        logger = logging.getLogger("sagent.test")
        logger.setLevel(logging.DEBUG)
        logger.warning("hello-file")

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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
