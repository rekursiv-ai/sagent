"""The runtime as settings: when long calls wake the model, and how a rate limit waits.

Typical usage example:

  owners = [..., (("runtime",), RuntimeSettings(runtime=agent.runtime))]
  /runtime.detach 15 1 300     # Wake at 15 s, then double, at most every 5 min.
  /runtime.detach.Bash off     # Wait on every Bash call to its end.
  /runtime.retry off           # A long rate limit ends the turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import re

from sagent.runtime.runtime import Runtime
from sagent.runtime.types import WakeSchedule
from sagent.settings.exceptions import SettingError
from sagent.settings.types import Setting


__all__ = [
    "RuntimeSettings",
    "wake_schedule",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeSettings:
    """Exports ``detach``, one ``detach.TOOL`` per tool, and ``retry``."""

    runtime: Runtime

    def settings(self) -> list[Setting]:
        """Return the runtime's settings.

        Returns:
          settings: ``detach`` and each tool's ``detach.TOOL``, a wake schedule
            or off; ``retry``, persistent or off.

        """
        return [
            Setting(
                name=("detach",),
                help_short="when a long call wakes the model, and detaches",
                help_long="The first wake is FIRST seconds in; each gap after is"
                " GROWTH times how long the call has run, at least FIRST, at most"
                " MAX_GAP and the prompt cache's lifetime. The first wake answers"
                " the calls with what they wrote and lets them run on; each later"
                " one shows the model what is new.",
                usage="FIRST GROWTH MAX_GAP|off|default",
                value=self._detach,
                options=lambda: ("off", "default"),
                apply=self._set_detach,
                cli=True,
            ),
            *(
                Setting(
                    name=("detach", tool),
                    help_short=f"when a long {tool} call wakes the model",
                    usage="FIRST GROWTH MAX_GAP|off|default",
                    value=partial(self._tool_detach, tool),
                    options=lambda: ("off", "default"),
                    apply=partial(self._set_tool_detach, tool),
                    cli=True,
                )
                for tool in self.runtime.tools
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
        default = self.runtime.wake_schedules.default
        text = "off" if default is None else str(default)
        return f"{text} (default)" if default == WakeSchedule() else text

    async def _set_detach(self, args: str) -> str:
        default = WakeSchedule() if args == "default" else wake_schedule(args)
        self.runtime.wake_schedules = replace(
            self.runtime.wake_schedules,
            default=default,
        )
        return self._detach()

    def _tool_detach(self, tool: str) -> str:
        per_tool = self.runtime.wake_schedules.per_tool
        if tool not in per_tool:
            return "default"
        own = per_tool[tool]
        return "off" if own is None else str(own)

    async def _set_tool_detach(self, tool: str, args: str) -> str:
        schedule = self.runtime.wake_schedules
        per_tool = {k: v for k, v in schedule.per_tool.items() if k != tool}
        if args != "default":
            per_tool[tool] = wake_schedule(args)
        self.runtime.wake_schedules = replace(schedule, per_tool=per_tool)
        return self._tool_detach(tool)

    def _retry(self) -> str:
        return "persistent" if self.runtime.persistent_retry else "off"

    async def _set_retry(self, args: str) -> str:
        if args not in ("persistent", "off"):
            raise SettingError(f"{args!r} is not one of: persistent, off")
        self.runtime.persistent_retry = args == "persistent"
        return self._retry()


def wake_schedule(args: str) -> WakeSchedule | None:
    """Return the schedule ``FIRST GROWTH MAX_GAP`` names; ``None`` for ``off``.

    Args:
      args: Three numbers, apart by spaces or commas, a time maybe ending in
        ``s``; or ``off``.

    Returns:
      schedule: The schedule, or ``None`` for no wakes.

    Raises:
      SettingError: ``args`` is neither, or names a schedule that never wakes.

    """
    if args == "off":
        return None
    words = [w for w in re.split(r"[\s,]+", args.strip()) if w]
    try:
        first, gap_growth, max_gap = (float(w.removesuffix("s")) for w in words)
    except ValueError:
        raise SettingError(
            f"{args!r} is not FIRST GROWTH MAX_GAP, off, or default",
        ) from None
    try:
        return WakeSchedule(
            first_gap_sec=first,
            gap_growth=gap_growth,
            max_gap_sec=max_gap,
        )
    except ValueError as e:
        raise SettingError(str(e)) from None
