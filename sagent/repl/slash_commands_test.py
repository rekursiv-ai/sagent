"""Tests for repl.slash_commands."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console

from sagent.custom_types import ModelSpec
from sagent.lib.asyncio_collections import Deque
from sagent.repl.slash_commands import (
    handle_slash_clear,
    handle_slash_command,
    handle_slash_login,
    handle_slash_model,
)

import sagent.providers as providers_mod


def _agent(**overrides: object) -> MagicMock:
    a = MagicMock()
    a.inbox = Deque[str]()
    a.model_spec = overrides.get(
        "model_spec",
        ModelSpec(provider="Anthropic", auth="env", model_id="old-model"),
    )
    a.model = MagicMock(model_id="old-model")
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, width=80, force_terminal=False)
    return c, buf


class TestHandleSlashCommand:
    def test_model_dispatches(self) -> None:
        a = _agent()
        c, _ = _console()
        assert handle_slash_command(a, c, "/model") is True

    def test_clear_dispatches(self) -> None:
        a = _agent()
        c, _ = _console()
        assert handle_slash_command(a, c, "/clear") is True

    def test_clear_with_args_dispatches(self) -> None:
        a = _agent()
        c, _ = _console()
        assert handle_slash_command(a, c, "/clear fresh") is True

    def test_login_dispatches(self) -> None:
        a = _agent(model_spec=None)
        c, buf = _console()
        assert handle_slash_command(a, c, "/login") is True
        assert "[/login]" in buf.getvalue()

    def test_unknown_returns_false(self) -> None:
        a = _agent()
        c, _ = _console()
        assert handle_slash_command(a, c, "/unknown") is False


class TestHandleSlashModel:
    def test_parse_error(self) -> None:
        a = _agent()
        c, buf = _console()
        assert handle_slash_model(a, c, " 'unterminated") is True
        assert "parse error" in buf.getvalue()

    def test_no_model_spec(self) -> None:
        a = _agent(model_spec=None)
        c, buf = _console()
        assert handle_slash_model(a, c, " foo") is True
        assert "no model spec" in buf.getvalue()

    def test_account_flag(self) -> None:
        a = _agent()
        c, _ = _console()
        with patch("sagent.repl.slash_commands.build_provider") as mock_bp:
            mock_model = MagicMock(model_id="new-model")
            mock_bp.return_value.model.return_value = mock_model
            handle_slash_model(a, c, " --account work new-model")
        spec = a.swap_model.call_args.kwargs["spec"]
        assert spec.account == "work"

    def test_unknown_flag(self) -> None:
        a = _agent()
        c, buf = _console()
        assert handle_slash_model(a, c, " --bogus") is True
        assert "unknown flag" in buf.getvalue()

    def test_no_model_id_same_provider_shows_usage(self) -> None:
        a = _agent()
        c, buf = _console()
        assert handle_slash_model(a, c, " --auth api") is True
        assert "usage:" in buf.getvalue()

    def test_infer_provider_called(self) -> None:
        a = _agent()
        c, _ = _console()
        with (
            patch(
                "sagent.repl.slash_commands.infer_provider",
                return_value=("Google", "env"),
            ) as mock_ip,
            patch("sagent.repl.slash_commands.build_provider") as mock_bp,
        ):
            mock_bp.return_value.model.return_value = MagicMock(model_id="gemini-2")
            handle_slash_model(a, c, " gemini-2")
        mock_ip.assert_called_once_with("gemini-2", "Anthropic")
        mock_bp.assert_called_once_with("Google", "env", account=None)

    def test_infer_provider_returns_none(self) -> None:
        a = _agent()
        c, _ = _console()
        with (
            patch(
                "sagent.repl.slash_commands.infer_provider",
                return_value=None,
            ),
            patch("sagent.repl.slash_commands.build_provider") as mock_bp,
        ):
            mock_bp.return_value.model.return_value = MagicMock(model_id="custom")
            handle_slash_model(a, c, " custom")
        mock_bp.assert_called_once_with("Anthropic", "env", account=None)

    def test_selfhosted_local_path_updates_auth(self) -> None:
        a = _agent(
            model_spec=ModelSpec(
                provider="SelfHosted",
                auth="/old/model",
                model_id="/old/model",
            ),
            model=MagicMock(model_id="/old/model"),
        )
        c, _ = _console()
        with patch("sagent.repl.slash_commands.build_provider") as mock_bp:
            mock_bp.return_value.model.return_value = MagicMock(model_id="/new/model")
            handle_slash_model(a, c, " /new/model")
        mock_bp.assert_called_once_with("SelfHosted", "/new/model", account=None)
        spec = a.swap_model.call_args.kwargs["spec"]
        assert spec.provider == "SelfHosted"
        assert spec.auth == "/new/model"
        assert spec.model_id == "/new/model"

    def test_build_provider_error(self) -> None:
        a = _agent()
        c, buf = _console()
        with patch(
            "sagent.repl.slash_commands.build_provider",
            side_effect=ValueError("bad model"),
        ):
            assert handle_slash_model(a, c, " nope") is True
        assert "bad model" in buf.getvalue()

    def test_cross_provider_label(self) -> None:
        a = _agent()
        c, buf = _console()
        with patch("sagent.repl.slash_commands.build_provider") as mock_bp:
            mock_bp.return_value.model.return_value = MagicMock(model_id="gemini-2")
            handle_slash_model(a, c, " --provider Google gemini-2")
        text = buf.getvalue()
        assert "Anthropic/old-model" in text
        assert "Google/gemini-2" in text


class TestHandleSlashLogin:
    def test_no_model_spec(self) -> None:
        a = _agent(model_spec=None)
        c, buf = _console()
        handle_slash_login(a, c)
        assert "no model spec" in buf.getvalue()

    def test_unknown_provider(self) -> None:
        a = _agent(
            model_spec=ModelSpec(provider="NoSuchProvider99", auth="env", model_id="m")
        )
        c, buf = _console()
        handle_slash_login(a, c)
        assert "unknown provider" in buf.getvalue()

    def test_no_login_method(self) -> None:
        a = _agent(
            model_spec=ModelSpec(provider="_TestNoLogin", auth="env", model_id="m")
        )
        c, buf = _console()
        stub = SimpleNamespace()
        with patch.object(providers_mod, "_TestNoLogin", stub, create=True):
            handle_slash_login(a, c)
        assert "no login method" in buf.getvalue()

    def test_login_success(self) -> None:
        a = _agent(
            model_spec=ModelSpec(provider="_TestLogin", auth="env", model_id="m")
        )
        c, buf = _console()
        login = MagicMock()
        stub = SimpleNamespace(login=login)
        with patch.object(providers_mod, "_TestLogin", stub, create=True):
            handle_slash_login(a, c)
        login.assert_called_once()
        assert "re-authenticated" in buf.getvalue()

    def test_login_error(self) -> None:
        a = _agent(
            model_spec=ModelSpec(provider="_TestLoginErr", auth="env", model_id="m")
        )
        c, buf = _console()
        login = MagicMock(side_effect=RuntimeError("auth failed"))
        stub = SimpleNamespace(login=login)
        with patch.object(providers_mod, "_TestLoginErr", stub, create=True):
            handle_slash_login(a, c)
        assert "auth failed" in buf.getvalue()


class TestHandleSlashClear:
    def test_clear_no_reason(self) -> None:
        a = _agent()
        c, buf = _console()
        assert handle_slash_clear(a, c, "") is True
        assert a.inbox.drain() == ["/clear"]
        assert "queued" in buf.getvalue()

    def test_clear_with_reason(self) -> None:
        a = _agent()
        c, buf = _console()
        assert handle_slash_clear(a, c, " fresh start") is True
        assert a.inbox.drain() == ["/clear fresh start"]
        assert "fresh start" in buf.getvalue()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
