"""Tests for harvesting: every owner's settings, under where it is mounted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import re

import pytest

from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.types import Setting


if TYPE_CHECKING:
    from collections.abc import Sequence


async def _ok(args: str) -> str:
    return args


def _setting(*name: str) -> Setting:
    return Setting(name=name, help_short=".".join(name), apply=_ok)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Owner:
    """Exports ``exported``, as given."""

    exported: tuple[Setting, ...]

    def settings(self) -> Sequence[Setting]:
        return self.exported


def test_each_setting_is_named_under_its_owner_in_order() -> None:
    effort = _setting("thinking", "effort")
    harvested = harvest(
        [
            ((), _Owner(exported=(_setting("clear"),))),
            (("model",), _Owner(exported=(effort, _setting("context")))),
            (("tool", "Bash"), _Owner(exported=(_setting("display"),))),
        ],
    )
    assert [s.name for s in harvested] == [
        ("clear",),
        ("model", "thinking", "effort"),
        ("model", "context"),
        ("tool", "Bash", "display"),
    ]
    assert harvested[1].apply is effort.apply
    assert harvested[1].help_short == "thinking.effort"


def test_an_owner_exporting_nothing_adds_nothing() -> None:
    assert harvest([(("screen",), _Owner(exported=()))]) == []


def test_two_settings_of_one_full_name_are_refused() -> None:
    with pytest.raises(ValueError, match=r"^two settings are named 'model\.context'$"):
        harvest(
            [
                (("model",), _Owner(exported=(_setting("context"),))),
                ((), _Owner(exported=(_setting("model", "context"),))),
            ],
        )


@pytest.mark.parametrize(
    ("prefix", "name", "shown"),
    [
        ((), (), "()"),
        (("model",), ("",), "('model', '')"),
        (("model",), ("thinking.effort",), "('model', 'thinking.effort')"),
        (("tool", "my tool"), ("display",), "('tool', 'my tool', 'display')"),
        ((), ("a=b",), "('a=b',)"),
        (("x\t",), ("y",), "('x\\t', 'y')"),
    ],
    ids=["empty", "blank-segment", "dot", "space", "equals", "tab"],
)
def test_a_name_a_user_could_not_type_is_refused(
    prefix: tuple[str, ...],
    name: tuple[str, ...],
    shown: str,
) -> None:
    owner = _Owner(exported=(Setting(name=name, help_short="x", apply=_ok),))
    message = f"setting name {shown} needs segments of letters, digits, '_' or '-'"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        harvest([(prefix, owner)])


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
