"""The ``code`` mode: the file the producer edited is matched against a regex.

The producer may read, edit, and load skills, as it does in the CLI, so a
behavior a STYLE.md or a skill guards (which the system prompt only names)
shows as it would there. The verdict is structural: no judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import re

from sagent.model.exceptions import RetryableError
from sagent.scripts.prompt_debug.harness import converse, producer, workspace
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.tools.skill import Skill


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.scripts.prompt_debug.harness import Case, Run


__all__ = ["CodeMode"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeMode:
    """A trial flags when ``file``, once edited, matches ``flag_regex``.

    Attributes:
      flag_regex: What the guarded behavior writes, e.g. a one-line delegator.
      file: The seeded file the producer edits, relative to the trial's folder.

    """

    flag_regex: str
    file: str

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Self:
        """Return the mode a case file says: ``flag_regex`` and ``file``.

        Args:
          meta: The frontmatter.
          sections: The sections, by name; one must seed ``file``.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        Raises:
          ValueError: A key is missing, the regex is invalid, or no section seeds
            ``file``.
          TypeError: A key is not a string.

        """
        flag_regex = meta.get("flag_regex", "")
        file = meta.get("file", "")
        if not isinstance(flag_regex, str) or not isinstance(file, str):
            raise TypeError(f"{where}: flag_regex and file must be strings.")
        if not flag_regex or not file:
            raise ValueError(f"{where}: code mode requires flag_regex and file.")
        if f"file: {file}" not in sections:
            raise ValueError(f"{where}: no '## file: {file}' section seeds {file}.")
        try:
            _ = re.compile(flag_regex)
        except re.error as e:
            raise ValueError(f"{where}: invalid flag_regex {flag_regex!r}: {e}") from e
        return cls(flag_regex=flag_regex, file=file)

    async def trial(self, case: Case, run: Run) -> bool:
        """Have a producer edit ``file``; match what it wrote.

        Args:
          case: The case.
          run: The models and prompt.

        Returns:
          flagged: Whether the edited file matches ``flag_regex``.

        Raises:
          RetryableError: The producer left the file as it was, so the trial
            measured nothing.

        """
        tools = [Read(), Edit(), Glob(), Grep(), Skill()]
        with workspace(case) as root:
            async with producer(run, tools=tools) as agent:
                _ = await converse(agent, case, root)
            edited = (root / self.file).read_text(encoding="utf-8")
        if edited == case.files[self.file]:
            raise RetryableError(f"void trial: producer left {self.file} unchanged")
        return re.search(self.flag_regex, edited) is not None
