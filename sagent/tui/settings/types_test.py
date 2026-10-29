"""Defaults and contracts of ``Setting``, ``SettingError`` and ``HasSettings``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses

import pytest

from sagent.runtime.exceptions import UserFacingError
from sagent.tui.settings.types import HasSettings, Setting, SettingError


if TYPE_CHECKING:
    from collections.abc import Sequence


async def _echo(text: str) -> str:
    return text


def test_setting_defaults() -> None:
    setting = Setting(name=("clear",), help_short="clear", apply=_echo)
    assert setting.help_long == ""
    assert setting.usage == ""
    assert setting.value is None
    assert setting.options() == ()
    assert setting.cli is False
    assert setting.inherit is None


def test_setting_is_frozen_and_keyword_only() -> None:
    setting = Setting(name=("a", "b"), help_short="h", apply=_echo, cli=True)
    assert setting.cli
    assert all(field.kw_only for field in dataclasses.fields(setting))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setting.__setattr__("cli", False)


def test_setting_error_is_user_facing_value_error() -> None:
    assert issubclass(SettingError, UserFacingError)
    assert issubclass(SettingError, ValueError)
    with pytest.raises(ValueError, match="bad level"):
        raise SettingError("bad level")


class _Owner:
    def settings(self) -> Sequence[Setting]:
        return [Setting(name=("x",), help_short="x", apply=_echo)]


def test_has_settings_is_runtime_checkable() -> None:
    assert isinstance(_Owner(), HasSettings)
    assert not isinstance(object(), HasSettings)
    assert HasSettings.settings(_Owner()) is None


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
