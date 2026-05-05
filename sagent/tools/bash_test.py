"""Tests for the Bash tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools.bash import (
    BASH_DEFAULT_TIMEOUT_MS,
    BASH_MAX_TIMEOUT_MS,
    Bash,
    _render_bash_description,
)
from sagent.tools.core import ToolState, tool_state_context
from sagent.tools.grep import Grep
from sagent.tools.lib import bash as bash_mod
from sagent.tools.lib.bash import Node


bash = Bash()


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-x"),),
        "multipart/x-tool-call",
    )


def _text(r: Message) -> str:
    if isinstance(r, TextMessage):
        return r.content
    if isinstance(r, MultipartMessage):
        for p in r.content:
            if isinstance(p, TextMessage) and p.descriptor == "text/plain":
                return p.content
    return ""


class TestBash:
    def test_description_substitutes_timeout_templates(self) -> None:
        text = (
            "${GET_MAX_TIMEOUT_MS()} "
            "${GET_MAX_TIMEOUT_MS()/60000} "
            "${GET_DEFAULT_TIMEOUT_MS()} "
            "${GET_DEFAULT_TIMEOUT_MS()/60000}"
        )
        assert _render_bash_description(text) == (
            f"{BASH_MAX_TIMEOUT_MS} "
            f"{BASH_MAX_TIMEOUT_MS // 60_000} "
            f"{BASH_DEFAULT_TIMEOUT_MS} "
            f"{BASH_DEFAULT_TIMEOUT_MS // 60_000}"
        )

    def test_description_has_no_unrendered_templates(self) -> None:
        assert "${" not in Bash.description

    @pytest.mark.anyio
    async def test_simple_command(self) -> None:
        response = await bash.run(_msg(json_freeze({"command": "echo hello"})))
        assert _text(response) == "hello"

    @pytest.mark.anyio
    async def test_exit_code(self) -> None:
        response = await bash.run(_msg(json_freeze({"command": "false"})))
        assert "[exit code: 1]" in _text(response)


class TestBashEdgeCases:
    @pytest.mark.anyio
    async def test_run_in_background(self) -> None:
        response = await bash.run(
            _msg(json_freeze({"command": "echo bg", "run_in_background": True}))
        )
        assert "background" in _text(response).lower()

    @pytest.mark.anyio
    async def test_no_output(self) -> None:
        response = await bash.run(_msg(json_freeze({"command": "true"})))
        assert _text(response) == "(no output)"

    @pytest.mark.anyio
    async def test_cwd_tracking(self, tmp_path: Path) -> None:
        response = await bash.run(
            _msg(json_freeze({"command": f"cd {tmp_path} && pwd"}))
        )
        assert str(tmp_path) in _text(response)


class TestBashStderr:
    @pytest.mark.anyio
    async def test_stderr_appended(self) -> None:
        response = await bash.run(_msg(json_freeze({"command": "echo errout >&2"})))
        assert "errout" in _text(response)


class TestBashLint:
    """Verify ``Bash(peers=...)`` routes commands through peer matchers."""

    @pytest.mark.anyio
    async def test_fires_when_peer_matches(self, tmp_path: Path) -> None:
        f = tmp_path / "hi.txt"
        f.write_text("hello world\n")
        b = Bash(peers=(Grep(),))
        response = await b.run(_msg(json_freeze({"command": f"grep hello {f}"})))
        assert "<system-reminder>" in _text(response)
        assert "[bash-lint]" in _text(response)
        assert "grep via Bash is a bad UX. Use the Grep tool." in _text(response)
        # Actual output still present.
        assert "hello world" in _text(response)

    @pytest.mark.anyio
    async def test_silent_when_peer_bails(self) -> None:
        b = Bash(peers=(Grep(),))
        response = await b.run(_msg(json_freeze({"command": "echo hi"})))
        assert "[bash-lint]" not in _text(response)

    @pytest.mark.anyio
    async def test_no_peers_no_lint(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("x\n")
        b = Bash()  # no peers
        response = await b.run(_msg(json_freeze({"command": f"grep x {f}"})))
        assert "[bash-lint]" not in _text(response)

    @pytest.mark.anyio
    async def test_parse_error_no_lint(self) -> None:
        b = Bash(peers=(Grep(),))
        response = await b.run(_msg(json_freeze({"command": "echo 'unterminated"})))
        # Should not crash; may error from shell but no lint banner.
        assert "[bash-lint]" not in _text(response)


class TestBashParseCache:
    """Verify the per-request bashlex parse cache is shared between
    matcher dispatch and any other call site (e.g. agent's
    concurrency check).
    """

    @pytest.mark.anyio
    async def test_lint_uses_state_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``Bash._collect_nudges`` consults ``ToolState.bash_parse_cache``
        instead of re-parsing. Pre-seed the cache with the real
        parsed trees, then monkeypatch ``parse_bash`` to a counter;
        confirm the counter stays at zero during lint dispatch.
        """
        real_parse = bash_mod.parse_bash

        f = tmp_path / "hi.txt"
        _ = f.write_text("hello\n")
        cmd = f"grep hello {f}"

        state = ToolState()
        # Pre-seed as the agent's pre-dispatch concurrency check would.
        state.bash_parse_cache[cmd] = real_parse(cmd)

        call_count = 0

        def counting_parse(command: str) -> tuple[Node, ...] | None:
            nonlocal call_count
            call_count += 1
            return real_parse(command)

        monkeypatch.setattr(bash_mod, "parse_bash", counting_parse)

        with tool_state_context(state):
            b = Bash(peers=(Grep(),))
            response = await b.run(_msg(json_freeze({"command": cmd})))

        # Lint ran through the cache - no re-parse.
        assert call_count == 0
        # Sanity: lint actually fired (not a silent skip).
        assert "[bash-lint]" in _text(response)
