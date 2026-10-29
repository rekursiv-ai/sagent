"""The runtime as settings: when a waited-on call detaches, and how a rate limit waits.

Typical usage example:

  owners = [..., (("runtime",), RuntimeSettings(runtime=agent.runtime))]
  /runtime.detach 120        # A call still running after 2 minutes detaches.
  /runtime.retry off         # A long rate limit ends the turn.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

from sagent.runtime.runtime import Runtime
from sagent.tui.settings.types import Setting, SettingError


__all__ = [
    "RuntimeSettings",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeSettings:
    """Exports ``detach`` and ``retry``: how ``runtime`` waits on its calls."""

    runtime: Runtime

    def settings(self) -> list[Setting]:
        """Return the runtime's settings.

        Returns:
          settings: ``detach``, off or seconds; ``retry``, persistent or off.

        """
        return [
            Setting(
                name=("detach",),
                help_short="detach a call the model waits on after this long",
                help_long="A detached call runs on; its result arrives later as a"
                " message, and the model works meanwhile.",
                usage="off|SECONDS",
                value=self._detach,
                options=lambda: ("off",),
                apply=self._set_detach,
                cli=True,
            ),
            Setting(
                name=("retry",),
                help_short="wait out a long rate limit, or end the turn on it",
                usage="persistent|off",
                value=self._retry,
                options=lambda: ("persistent", "off"),
                apply=self._set_retry,
                cli=True,
            ),
        ]

    def _detach(self) -> str:
        after = self.runtime.detach_after_sec
        return "off (default)" if after is None else f"{after:g}s"

    async def _set_detach(self, args: str) -> str:
        if args in ("off", "default"):
            self.runtime.detach_after_sec = None
            return self._detach()
        try:
            seconds = float(args.removesuffix("s"))
        except ValueError:
            seconds = math.nan
        if not math.isfinite(seconds) or seconds <= 0:
            raise SettingError(f"{args!r} is not off or a number of seconds above 0")
        self.runtime.detach_after_sec = seconds
        return self._detach()

    def _retry(self) -> str:
        return "persistent" if self.runtime.persistent_retry else "off"

    async def _set_retry(self, args: str) -> str:
        if args not in ("persistent", "off"):
            raise SettingError(f"{args!r} is not one of: persistent, off")
        self.runtime.persistent_retry = args == "persistent"
        return self._retry()
