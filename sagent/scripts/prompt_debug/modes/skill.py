"""The ``skill`` mode: a trial flags when the producer never invoked a skill.

The producer gets the tools the ``code`` mode gives, ``Skill`` among them, so the
skills its catalog lists are offered as they are in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from sagent.journal.types import ToolCall
from sagent.scripts.prompt_debug.harness import converse, producer, workspace
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.tools.skill import Skill


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.scripts.prompt_debug.harness import Case, Run


__all__ = ["SkillMode"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillMode:
    """A trial flags when the producer never called ``Skill`` for ``expect_skill``.

    Attributes:
      expect_skill: The skill the task should make it load.

    """

    expect_skill: str

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Self:
        """Return the mode a case file says: ``expect_skill``.

        Args:
          meta: The frontmatter.
          sections: The sections, by name; unused.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        Raises:
          ValueError: ``expect_skill`` is missing or not a string.

        """
        del sections
        skill = meta.get("expect_skill")
        if not isinstance(skill, str) or not skill:
            raise ValueError(f"{where}: skill mode requires a non-empty expect_skill")
        return cls(expect_skill=skill)

    async def trial(self, case: Case, run: Run) -> bool:
        """Have a producer do the task; look for the skill's invocation.

        Args:
          case: The case.
          run: The models and prompt.

        Returns:
          flagged: Whether the skill was never invoked.

        """
        tools = [Read(), Edit(), Glob(), Grep(), Skill()]
        with workspace(case) as root:
            async with producer(run, tools=tools) as agent:
                _ = await converse(agent, case, root)
                return not any(
                    isinstance(record, ToolCall)
                    and record.name == Skill.name
                    and record.arguments.get("skill") == self.expect_skill
                    for record in agent.session.history()
                )
