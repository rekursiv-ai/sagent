"""The ``tool_volume`` mode: a trial flags when the producer makes too many tool calls.

The guarded behavior is over-verification: surveying with reads, greps, and
commands to answer what the task's own text already settles. The producer has
every tool, ``Bash`` among them, because the behavior is reaching for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import logging

from sagent.journal.types import ToolCall
from sagent.scripts.prompt_debug.harness import converse, producer, workspace
from sagent.tools.bash import Bash
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.read import Read
from sagent.tools.skill import Skill


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.scripts.prompt_debug.harness import Case, Run


__all__ = ["ToolVolumeMode"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolVolumeMode:
    """A trial flags when the producer makes more than ``max_tool_calls`` calls.

    Attributes:
      max_tool_calls: The most calls a correct answer needs.

    """

    max_tool_calls: int

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Self:
        """Return the mode a case file says: ``max_tool_calls``.

        Args:
          meta: The frontmatter.
          sections: The sections, by name; unused.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        Raises:
          TypeError: ``max_tool_calls`` is not an integer.
          ValueError: ``max_tool_calls`` is under 1.

        """
        del sections
        most = meta.get("max_tool_calls")
        if isinstance(most, bool) or not isinstance(most, int):
            raise TypeError(f"{where}: max_tool_calls must be an integer.")
        if most < 1:
            raise ValueError(f"{where}: tool_volume mode requires max_tool_calls >= 1")
        return cls(max_tool_calls=most)

    async def trial(self, case: Case, run: Run) -> bool:
        """Have a producer, working in the trial's folder, answer; count its calls.

        Args:
          case: The case.
          run: The models and prompt.

        Returns:
          flagged: Whether it made more than ``max_tool_calls`` calls.

        """
        tools = [Read(), Grep(), Glob(), List(), Bash(), Skill()]
        with workspace(case) as root:
            async with producer(run, tools=tools, cwd=root) as agent:
                _ = await converse(agent, case, root)
                calls = sum(isinstance(r, ToolCall) for r in agent.session.history())
        # The count, not just the verdict: a fix that halves the calls but stays over
        # the bound reads the same in the verdict.
        logger.info("tool_volume trial: %d calls (max %d)", calls, self.max_tool_calls)
        return calls > self.max_tool_calls
