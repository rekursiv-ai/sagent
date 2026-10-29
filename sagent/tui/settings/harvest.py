"""Every owner's settings, each named under where its owner is mounted.

Typical usage example:

  settings = harvest([((), commands), (("model",), model), (("screen",), printer)])
  [".".join(s.name) for s in settings]     # ["clear", "model.context", ...]
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import re


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.tui.settings.types import HasSettings, Setting


__all__ = [
    "harvest",
]


def harvest(owners: Sequence[tuple[tuple[str, ...], HasSettings]]) -> list[Setting]:
    """Return each owner's settings, its mount put before each name, in order.

    Args:
      owners: Each owner, after the name it is mounted at; ``()`` for the top.

    Returns:
      settings: Owner by owner, each in the order it exported them.

    Raises:
      ValueError: A name has a segment a user could not type, or two settings
        share a full name.

    """
    harvested: list[Setting] = []
    seen: set[tuple[str, ...]] = set()
    for mount, owner in owners:
        for setting in owner.settings():
            name = (*mount, *setting.name)
            # A dot would split a segment; whitespace or ``=`` would end the name.
            if not name or not all(re.fullmatch(r"[\w-]+", p, re.ASCII) for p in name):
                raise ValueError(
                    f"setting name {name!r} needs segments of letters, digits,"
                    " '_' or '-'",
                )
            if name in seen:
                raise ValueError(f"two settings are named {'.'.join(name)!r}")
            seen.add(name)
            harvested.append(replace(setting, name=name))
    return harvested
