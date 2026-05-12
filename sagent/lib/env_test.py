"""Tests for ``lib.env``: environment variable helpers."""

from __future__ import annotations

import pytest

from sagent.lib.env import env_truthy


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "YES", "On"])
def test_env_truthy_recognizes_truthy(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAGENT_TEST_VAR", value)
    assert env_truthy("SAGENT_TEST_VAR")


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "anything"])
def test_env_truthy_recognizes_falsy(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAGENT_TEST_VAR", value)
    assert not env_truthy("SAGENT_TEST_VAR")


def test_env_truthy_unset_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGENT_TEST_VAR", raising=False)
    assert not env_truthy("SAGENT_TEST_VAR")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
