"""Where Recall searches as settings, each change replacing the tool the agent reads.

Typical usage example:

  owners = [..., (("tool", "Recall"), RecallSettings(tools=tools))]
  /tool.Recall.url https://trax.example
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from sagent.settings.types import Setting
from sagent.tools.recall import Recall
from sagent.tools.types import Tool


__all__ = [
    "RecallSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallSettings:
    """Exports where the ``Recall`` in ``tools`` searches: ``url`` and ``session``.

    Its key is no setting: a secret is not typed where a screen shows it. A change
    of server drops the key the tool held, which was for the server it was given with.
    """

    tools: MutableMapping[str, Tool]
    """Where the agent reads its tools; a change replaces ``Recall`` here."""

    def settings(self) -> list[Setting]:
        """Return the server and the session Recall searches, as settings.

        Returns:
          settings: Each, named relative to where the tool is mounted.

        """
        return [
            Setting(
                name=("url",),
                help_short=(
                    "the trackinizer server Recall asks; empty reads $TRACKINIZER_URL"
                ),
                usage="URL",
                value=partial(self._value, "url"),
                options=lambda: ("default",),
                apply=partial(self._apply, "url"),
                cli=True,
            ),
            Setting(
                name=("session",),
                help_short=(
                    "the stored session Recall searches near; empty reads $TRAX_SESSION"
                ),
                usage="ID",
                value=partial(self._value, "session"),
                options=lambda: ("default",),
                apply=partial(self._apply, "session"),
                cli=True,
            ),
        ]

    def _value(self, name: Literal["url", "session"]) -> str:
        tool = self.tools["Recall"]
        assert isinstance(tool, Recall)
        return {"url": tool.url, "session": tool.session}[name]

    async def _apply(self, name: Literal["url", "session"], args: str) -> str:
        tool = self.tools["Recall"]
        assert isinstance(tool, Recall)
        wanted = "" if args == "default" else args
        match name:
            case "url":
                self.tools["Recall"] = replace(tool, url=wanted, key="")
            case "session":
                self.tools["Recall"] = replace(tool, session=wanted)
        return self._value(name)
