"""What a user may reach by name, and what exports it.

Each ``Setting`` is one leaf: a value to read and change, such as
``model.thinking.effort``, or an action to run, such as ``clear``. Its owner,
a ``HasSettings``, exports it under the owner's own name; the prompt and the
command line reach the same leaves, so neither keeps a list of its own. Two
leaves may not share a full name.

Typical usage example:

  class Screen:
      show_thinking: bool = True

      def settings(self) -> Sequence[Setting]:
          return [Setting(name=("thinking",),
                          help_short="show or hide reasoning",
                          usage="show|hide",
                          value=lambda: "show" if self.show_thinking else "hide",
                          options=lambda: ("show", "hide"),
                          apply=self._set)]

  /screen.thinking           # [/screen.thinking] show   (show, hide)
  /screen.thinking hide      # Typed at the prompt.
  --screen.thinking hide     # Given at startup, when ``cli``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sagent.runtime.exceptions import UserFacingError


__all__ = [
    "HasSettings",
    "Setting",
    "SettingError",
]


class SettingError(UserFacingError, ValueError):
    """What a user gave a setting does not fit it; the message says what would."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Setting:
    """One thing a user reaches as ``/name ARGS``, or as ``--name ARGS`` when ``cli``."""

    name: tuple[str, ...]
    """One segment per level, such as ``("model", "thinking", "effort")``; an
    owner's are relative to it, and harvesting prefixes where it is mounted."""

    help_short: str
    """One line saying what it is; ``/help`` lists it."""

    help_long: str = ""
    """What one line cannot say, such as a grammar; ``/help NAME`` adds it."""

    usage: str = ""
    """What may follow the name, such as ``LEVEL``; ``""`` for nothing."""

    value: Callable[[], str] | None = None
    """Its current value; ``None`` for an action, which a bare name runs."""

    options: Callable[[], Sequence[str]] = tuple
    """What it may take, least committing first; ``()`` when unlisted, as a name."""

    apply: Callable[[str], Awaitable[str]]
    """Change it to, or run it with, what followed the name; return what to print.
    Raises ``ValueError``, saying why, having changed nothing."""

    cli: bool = False
    """Whether the command line takes it before the session opens."""

    inherit: Callable[[str], Awaitable[str]] | None = None
    """Take what an ancestor chose: the closest value this one offers, rather than
    a refusal; ``None`` takes it as ``apply`` does."""


@runtime_checkable
class HasSettings(Protocol):
    """Exports what a user may reach, each name relative to where it is mounted."""

    def settings(self) -> Sequence[Setting]:
        """Return each leaf, in the order an abbreviation that matches several picks from."""
        ...
