"""Tests for sagent.cli."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import argparse
import os
import subprocess

import pytest

from sagent import sessions
from sagent.bin.cli import (
    _build_provider_model,
    _configure_logging,
    _parse_cli_args,
    _resolve_continue,
    _resolve_resume,
    _resolve_session_dir,
    main,
    resolve_tools,
)
from sagent.providers import build_provider
from sagent.providers.anthropic import Anthropic
from sagent.tools.advisor import Advisor
from sagent.tools.bash import Bash


def _provider_patches(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch CLI defaults to Anthropic/env and return mock provider."""
    monkeypatch.setattr("sagent.bin.cli._DEFAULT_PROVIDER", "Anthropic")
    monkeypatch.setattr("sagent.bin.cli._DEFAULT_AUTH", "env")
    mock_prov = MagicMock()
    mock_model = MagicMock()
    mock_model.max_request_tokens = 100_000
    mock_model.max_response_tokens = 8_192
    mock_model.model_id = "claude-sonnet-4-6"
    mock_prov.model.return_value = mock_model
    return mock_prov


def _parse(argv: list[str]) -> argparse.Namespace:
    """Build a fresh parser + run _parse_cli_args, return the namespace."""
    parser = argparse.ArgumentParser()
    ns, _ = _parse_cli_args(parser, argv)
    return ns


class TestParseCliArgs:
    @pytest.mark.ci_smoke
    def test_direct_script_bootstraps_dependencies(self, tmp_path: Path) -> None:
        script = Path(__file__).with_name("cli.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["UV_FROZEN"] = "1"
        env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / ".venv")

        proc = subprocess.run(  # noqa: S603 - Exercises the trusted local wrapper.
            [str(script), "--help"],
            cwd=script.parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr
        assert "Interactive CLI agent." in proc.stdout

    def test_max_request_tokens(self) -> None:
        assert _parse(["--max-request-tokens", "8192"]).max_request_tokens == 8192

    def test_max_response_tokens(self) -> None:
        assert _parse(["--max-response-tokens", "12"]).max_response_tokens == 12

    def test_log_level(self) -> None:
        assert _parse(["--log-level", "DEBUG"]).log_level == "DEBUG"

    def test_configure_logging_rejects_invalid_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAGENT_LOG_LEVEL", "LOUD")
        with pytest.raises(SystemExit, match="invalid log level"):
            _configure_logging(None)


class TestBuildProvider:
    def test_anthropic_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("sagent.providers.Anthropic") as mock:
            mock_prov = MagicMock()
            mock.from_env.return_value = mock_prov
            result = build_provider("Anthropic", "env")
            mock.from_env.assert_called_once_with()
            assert result is mock_prov

    def test_openai_env(self) -> None:
        with patch("sagent.providers.OpenAI") as mock:
            mock_prov = MagicMock()
            mock.from_env.return_value = mock_prov
            result = build_provider("OpenAI", "env")
            mock.from_env.assert_called_once_with()
            assert result is mock_prov

    def test_google_env(self) -> None:
        with patch("sagent.providers.Google") as mock:
            mock_prov = MagicMock()
            mock.from_env.return_value = mock_prov
            result = build_provider("Google", "env")
            mock.from_env.assert_called_once_with()
            assert result is mock_prov

    def test_unknown_provider_class(self) -> None:
        with pytest.raises(AttributeError):
            build_provider("NotAProvider", "env")

    def test_unknown_auth_treated_as_literal_key(self) -> None:
        prov = build_provider("Anthropic", "sk-ant-literal")
        assert isinstance(prov, Anthropic)
        assert prov._api_key == "sk-ant-literal"

    def test_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            build_provider("Anthropic", "env")

    def test_selfhosted_model_path_is_load_path(self) -> None:
        args = argparse.Namespace(
            provider="SelfHosted",
            auth="env",
            account=None,
            model="/opt/models/qwen3.6-27b",
        )
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_model.model_id = "/opt/models/qwen3.6-27b"
        mock_provider.model.return_value = mock_model
        with patch(
            "sagent.bin.cli.build_provider",
            return_value=mock_provider,
        ) as build:
            provider, model, auth = _build_provider_model(args)

        build.assert_called_once_with(
            "SelfHosted",
            "/opt/models/qwen3.6-27b",
            account=None,
        )
        mock_provider.model.assert_called_once_with(None)
        assert provider is mock_provider
        assert model is mock_model
        assert auth == "/opt/models/qwen3.6-27b"

    def test_selfhosted_model_options_stay_in_load_path(self) -> None:
        args = argparse.Namespace(
            provider="SelfHosted",
            auth="env",
            account=None,
            model="Qwen/Qwen3.6-27B+bfloat16+cuda",
        )
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_provider.model.return_value = mock_model
        with patch(
            "sagent.bin.cli.build_provider",
            return_value=mock_provider,
        ) as build:
            provider, model, auth = _build_provider_model(args)

        build.assert_called_once_with(
            "SelfHosted",
            "Qwen/Qwen3.6-27B+bfloat16+cuda",
            account=None,
        )
        mock_provider.model.assert_called_once_with(None)
        assert provider is mock_provider
        assert model is mock_model
        assert auth == "Qwen/Qwen3.6-27B+bfloat16+cuda"

    def test_selfhosted_defaults_to_env_auth(self) -> None:
        args = argparse.Namespace(
            provider="SelfHosted",
            auth="credentials",
            account=None,
            model=None,
        )
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_model.model_id = "Qwen/Qwen3.6-27B"
        mock_provider.model.return_value = mock_model
        with patch(
            "sagent.bin.cli.build_provider",
            return_value=mock_provider,
        ) as build:
            provider, model, auth = _build_provider_model(args)

        build.assert_called_once_with("SelfHosted", "env", account=None)
        mock_provider.model.assert_called_once_with(None)
        assert provider is mock_provider
        assert model is mock_model
        assert auth == "env"

    def test_llamacpp_model_is_endpoint_model_not_load_path(self) -> None:
        args = argparse.Namespace(
            provider="LlamaCpp",
            auth="/models/qwen.gguf",
            account=None,
            model="qwen3.6-27b-12gb",
        )
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_model.model_id = "qwen3.6-27b-12gb"
        mock_provider.model.return_value = mock_model
        with patch(
            "sagent.bin.cli.build_provider",
            return_value=mock_provider,
        ) as build:
            provider, model, auth = _build_provider_model(args)

        build.assert_called_once_with("LlamaCpp", "/models/qwen.gguf", account=None)
        mock_provider.model.assert_called_once_with("qwen3.6-27b-12gb")
        assert provider is mock_provider
        assert model is mock_model
        assert auth == "/models/qwen.gguf"


class TestResolveSessionDir:
    """``_resolve_session_dir`` precedence: --session > --continue > --resume > fresh."""

    def test_explicit_session_wins(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            session=tmp_path / "explicit",
            continue_=True,  # ignored because --session takes precedence
            continue_all=False,
            resume=True,
            resume_all=False,
        )
        assert _resolve_session_dir(args) == str(tmp_path / "explicit")

    def test_continue_routes_to_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        called = {}

        def _fake_continue(cwd: Path) -> str:
            called["cwd"] = cwd
            return str(tmp_path / "resumed")

        monkeypatch.setattr(
            "sagent.bin.cli._resolve_continue",
            _fake_continue,
        )
        args = argparse.Namespace(
            session=None,
            continue_=True,
            continue_all=False,
            resume=False,
            resume_all=False,
        )
        result = _resolve_session_dir(args)
        assert result == str(tmp_path / "resumed")
        assert called["cwd"] == Path.cwd()

    def test_resume_routes_to_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _resume(_cwd: Path) -> str:
            return str(tmp_path / "picked")

        monkeypatch.setattr(
            "sagent.bin.cli._resolve_resume",
            _resume,
        )
        args = argparse.Namespace(
            session=None,
            continue_=False,
            continue_all=False,
            resume=True,
            resume_all=False,
        )
        assert _resolve_session_dir(args) == str(tmp_path / "picked")

    def test_no_flags_creates_fresh(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _fresh(_cwd: Path) -> Path:
            return tmp_path / "fresh"

        monkeypatch.setattr(sessions, "new_session_dir", _fresh)
        args = argparse.Namespace(
            session=None,
            continue_=False,
            continue_all=False,
            resume=False,
            resume_all=False,
        )
        assert _resolve_session_dir(args) == str(tmp_path / "fresh")


class TestResolveContinue:
    def test_uses_latest_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        prior = MagicMock()
        prior.path = tmp_path / "prior"

        def _latest(_cwd: Path) -> MagicMock:
            return prior

        monkeypatch.setattr(sessions, "latest_session", _latest)
        assert _resolve_continue(tmp_path) == str(prior.path)

    def test_falls_back_to_fresh(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _latest(_cwd: Path) -> None:
            return None

        def _fresh(_cwd: Path) -> Path:
            return tmp_path / "fresh"

        monkeypatch.setattr(sessions, "latest_session", _latest)
        monkeypatch.setattr(sessions, "new_session_dir", _fresh)
        assert _resolve_continue(tmp_path) == str(tmp_path / "fresh")


class TestResolveResume:
    def test_empty_list_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _list(_cwd: Path) -> list[sessions.SessionInfo]:
            return []

        def _fresh(_cwd: Path) -> Path:
            return tmp_path / "fresh"

        monkeypatch.setattr(sessions, "list_sessions", _list)
        monkeypatch.setattr(sessions, "new_session_dir", _fresh)
        assert _resolve_resume(tmp_path) == str(tmp_path / "fresh")

    def test_picked_session_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        avail = [MagicMock()]
        choice = MagicMock()
        choice.path = tmp_path / "picked"

        def _list(_cwd: Path) -> list[MagicMock]:
            return avail

        def _pick(_av: list[sessions.SessionInfo]) -> MagicMock:
            return choice

        monkeypatch.setattr(sessions, "list_sessions", _list)
        monkeypatch.setattr(sessions, "pick_session", _pick)
        assert _resolve_resume(tmp_path) == str(tmp_path / "picked")

    def test_no_pick_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _list(_cwd: Path) -> list[MagicMock]:
            return [MagicMock()]

        def _pick(_av: list[sessions.SessionInfo]) -> None:
            return None

        def _fresh(_cwd: Path) -> Path:
            return tmp_path / "fresh"

        monkeypatch.setattr(sessions, "list_sessions", _list)
        monkeypatch.setattr(sessions, "pick_session", _pick)
        monkeypatch.setattr(sessions, "new_session_dir", _fresh)
        assert _resolve_resume(tmp_path) == str(tmp_path / "fresh")


class TestResolveTools:
    def test_explicit_wiki_tool_still_resolves(self) -> None:
        tools_list = resolve_tools(["Wiki"])

        assert [t.name for t in tools_list] == ["Wiki"]

    def test_none_disables_tools(self) -> None:
        assert resolve_tools(["none"]) == []

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(SystemExit, match="unknown tool"):
            resolve_tools(["NotARealTool"])

    def test_order_preserved(self) -> None:
        tools_list = resolve_tools(["Read", "Bash", "Grep"])
        names = [t.name for t in tools_list]
        assert names == ["Read", "Bash", "Grep"]

    def test_bash_receives_peers(self) -> None:
        # Bash constructed last with peer matchers from Read/Grep.
        tools_list = resolve_tools(["Bash", "Read", "Grep"])
        bash = tools_list[0]
        assert isinstance(bash, Bash)
        # Has matchers wired from both peers.
        assert len(bash._peer_matchers) == 2


class TestMain:
    def test_main_smoke(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_prov = _provider_patches(monkeypatch)
        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch("sys.argv", ["cli.py"]),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()

    def test_max_request_tokens_sets_agent_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_prov = _provider_patches(monkeypatch)
        agent = MagicMock()
        agent.tool_state = MagicMock()
        agent.tool_state.additional_dirs = []

        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.bin.cli.Agent", return_value=agent),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch("sys.argv", ["cli.py", "--max-request-tokens", "8192"]),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()

        assert agent.max_request_tokens == 8192

    def test_no_session_persistence_disables_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_prov = _provider_patches(monkeypatch)
        captured: dict[str, object] = {}

        def _capture_agent(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.bin.cli.Agent", side_effect=_capture_agent),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch("sys.argv", ["cli.py", "--no-session-persistence"]),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()
        system = cast("dict[str, object]", captured["system"])
        assert "memory" not in system


class TestAdvisorFlag:
    def test_defaults_off(self) -> None:
        ns = _parse([])
        assert ns.advisor is None
        assert ns.advisor_max_uses is None

    def test_parses_advisor_model(self) -> None:
        ns = _parse(["--advisor", "claude-opus-4-7"])
        assert ns.advisor == "claude-opus-4-7"

    def test_parses_max_uses(self) -> None:
        ns = _parse(["--advisor", "claude-opus-4-7", "--advisor-max-uses", "3"])
        assert ns.advisor_max_uses == 3

    def test_main_wires_advisor_into_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_prov = _provider_patches(monkeypatch)
        captured: dict[str, object] = {}

        def _capture_agent(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.bin.cli.Agent", side_effect=_capture_agent),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch(
                "sys.argv",
                [
                    "cli.py",
                    "--model",
                    "claude-sonnet-4-6",
                    "--advisor",
                    "claude-opus-4-7",
                    "--advisor-max-uses",
                    "2",
                ],
            ),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()
        tool_list = cast("list[object]", captured["tools"])
        advisors = [t for t in tool_list if isinstance(t, Advisor)]
        assert len(advisors) == 1
        assert advisors[0]._max_uses == 2

    def test_main_no_advisor_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_prov = _provider_patches(monkeypatch)
        captured: dict[str, object] = {}

        def _capture_agent(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.bin.cli.Agent", side_effect=_capture_agent),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch("sys.argv", ["cli.py"]),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()
        tool_list = cast("list[object]", captured["tools"])
        assert not any(isinstance(t, Advisor) for t in tool_list)


class TestAddDirFlag:
    def test_defaults_empty(self) -> None:
        ns = _parse([])
        assert ns.add_dir == []

    def test_parses_single(self) -> None:
        ns = _parse(["--add-dir", "/src/a"])
        assert ns.add_dir == ["/src/a"]

    def test_parses_multiple(self) -> None:
        ns = _parse(["--add-dir", "/src/a", "/src/b", "/src/c"])
        assert ns.add_dir == ["/src/a", "/src/b", "/src/c"]

    def test_main_wires_into_tool_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_prov = _provider_patches(monkeypatch)
        captured: dict[str, object] = {}

        def _capture_agent(**kwargs: object) -> MagicMock:
            m = MagicMock()
            m.tool_state = MagicMock()
            m.tool_state.additional_dirs = []
            captured["agent"] = m
            captured["kwargs"] = kwargs
            return m

        with (
            patch("sagent.bin.cli.asyncio.run"),
            patch("sagent.bin.cli.Agent", side_effect=_capture_agent),
            patch("sagent.providers.Anthropic") as mock_ant,
            patch("sys.argv", ["cli.py", "--add-dir", "/src/x", "/src/y"]),
        ):
            mock_ant.from_env.return_value = mock_prov
            main()
        agent = cast("MagicMock", captured["agent"])
        assert agent.tool_state.additional_dirs == ["/src/x", "/src/y"]


class TestAccountFlag:
    def test_default_is_none(self) -> None:
        ns = _parse([])
        assert ns.account is None

    def test_parses_account(self) -> None:
        ns = _parse(["--account", "work"])
        assert ns.account == "work"

    def test_build_provider_skips_account_for_incompatible_factory(self) -> None:
        """Providers whose ``from_<auth>`` doesn't accept ``account`` are unaffected."""
        captured: dict[str, object] = {}

        def _factory_no_account() -> MagicMock:
            captured["called"] = True
            return MagicMock()

        with patch("sagent.providers.Anthropic") as mock:
            mock.from_env = _factory_no_account
            build_provider("Anthropic", "env", account="work")
        # Factory called without ``account`` - no TypeError crash.
        assert captured == {"called": True}


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
