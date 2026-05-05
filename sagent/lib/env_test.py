"""Tests for lib.env."""

from __future__ import annotations

import pytest

from sagent.lib.env import env_truthy


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", "on", "ON"])
def test_truthy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SAGENT_TEST_VAR", val)
    assert env_truthy("SAGENT_TEST_VAR")


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage"])
def test_falsy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SAGENT_TEST_VAR", val)
    assert not env_truthy("SAGENT_TEST_VAR")


def test_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGENT_TEST_VAR", raising=False)
    assert not env_truthy("SAGENT_TEST_VAR")
