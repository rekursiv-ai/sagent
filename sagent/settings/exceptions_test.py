"""Contracts of what settings raise."""

from __future__ import annotations

import pytest

from sagent.runtime.exceptions import UserFacingError
from sagent.settings.exceptions import HelpRequestedError, SettingError


def test_setting_error_is_user_facing_value_error() -> None:
    assert issubclass(SettingError, UserFacingError)
    assert issubclass(SettingError, ValueError)
    with pytest.raises(ValueError, match="bad level"):
        raise SettingError("bad level")


def test_asking_for_help_is_no_mistake() -> None:
    """``except ValueError``, as a setting's refusal is caught, must not take it."""
    assert not issubclass(HelpRequestedError, ValueError)
    assert not issubclass(HelpRequestedError, UserFacingError)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
