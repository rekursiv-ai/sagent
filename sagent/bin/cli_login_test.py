"""Tests for the ``sagent login`` subcommand."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import argparse

import pytest

from sagent.bin.cli import (
    _parse_cli_args,
    main,
)
from sagent.providers import build_provider


class TestBuildProviderSubscription:
    def test_credentials_calls_from_credentials(self) -> None:
        with patch(
            "sagent.providers.OpenAISubscription",
        ) as mock:
            mock_prov = MagicMock()
            mock.from_credentials.return_value = mock_prov
            result = build_provider("OpenAISubscription", "credentials")
            mock.from_credentials.assert_called_once_with()
            assert result is mock_prov

    def test_build_provider_forwards_account(self) -> None:
        """``build_provider(..., account="work")`` reaches ``from_credentials``."""
        captured: dict[str, object] = {}

        def _factory_with_account(
            creds: object = None, *, account: str | None = None
        ) -> MagicMock:
            del creds
            captured["account"] = account
            return MagicMock()

        with patch(
            "sagent.providers.OpenAISubscription",
        ) as mock:
            mock.from_credentials = _factory_with_account
            build_provider("OpenAISubscription", "credentials", account="work")
        assert captured == {"account": "work"}


class TestLoginSubcommand:
    def test_parses_login_with_account(self) -> None:
        parser = argparse.ArgumentParser()
        ns, remaining = _parse_cli_args(parser, ["login", "--account", "work"])
        assert remaining == ["login"]
        assert ns.account == "work"

    def test_parses_login_default(self) -> None:
        parser = argparse.ArgumentParser()
        _, remaining = _parse_cli_args(parser, ["login"])
        assert remaining == ["login"]

    def test_parses_login_headless(self) -> None:
        parser = argparse.ArgumentParser()
        ns, remaining = _parse_cli_args(parser, ["login", "--headless"])
        assert remaining == ["login"]
        assert ns.headless is True

    def test_login_invokes_save(self) -> None:
        fake_creds = {
            "access_token": "t",
            "refresh_token": "r",
            "account_id": "acct",
            "expires_at": 0.0,
        }
        with (
            patch(
                "sagent.providers.OpenAISubscription",
            ) as mock,
            patch(
                "sys.argv",
                [
                    "cli.py",
                    "login",
                    "--provider",
                    "OpenAISubscription",
                    "--account",
                    "work",
                ],
            ),
        ):
            mock.login.return_value = fake_creds
            main()
        mock.login.assert_called_once()
        assert mock.login.call_args.kwargs["account"] == "work"
        assert mock.login.call_args.kwargs["manual"] is False
        mock.save.assert_called_once_with(fake_creds, account="work")

    def test_login_forwards_headless(self) -> None:
        fake_creds = {
            "access_token": "t",
            "refresh_token": "r",
            "account_id": "acct",
            "expires_at": 0.0,
        }
        with (
            patch(
                "sagent.providers.OpenAISubscription",
            ) as mock,
            patch(
                "sys.argv",
                [
                    "cli.py",
                    "login",
                    "--provider",
                    "OpenAISubscription",
                    "--headless",
                ],
            ),
        ):
            mock.login.return_value = fake_creds
            main()
        assert mock.login.call_args.kwargs["manual"] is True

    def test_login_rejects_provider_without_login(self) -> None:
        # ``OpenAI`` is the API-key class without a ``login`` classmethod;
        # ``--provider OpenAI login`` should fail fast with SystemExit
        # and a helpful stderr message.
        with (
            patch(
                "sagent.providers.OpenAI",
                spec=object,
            ),
            patch("sys.argv", ["cli.py", "login", "--provider", "OpenAI"]),
            pytest.raises(SystemExit),
        ):
            main()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
