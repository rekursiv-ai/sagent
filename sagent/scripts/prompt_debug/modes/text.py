"""The ``text`` mode: a blind judge scores the producer's answer against a rubric.

The judge sees only the answer, never the task or the producer's prompt: the
behaviors this mode guards, such as verbosity or tone, are ones a producer is
blind to in its own output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from sagent.scripts.prompt_debug.harness import converse, producer, workspace
from sagent.tools.bash import Bash
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.list import List
from sagent.tools.read import Read


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.scripts.prompt_debug.harness import Case, Run
    from sagent.tools.types import Tool


__all__ = ["TextMode"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TextMode:
    """A trial flags when the judge says the answer shows what the rubric names.

    Attributes:
      judge_rubric: What the judge scores against; its ``## judge_rubric``.
      use_tools: Whether the producer may read and run things. A behavior that
        comes from misreading a tool's result cannot show without tools.

    """

    judge_rubric: str
    use_tools: bool = False

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Self:
        """Return the mode a case file says: its rubric, and ``use_tools``.

        Args:
          meta: The frontmatter.
          sections: The sections, by name.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        Raises:
          ValueError: The rubric is missing.
          TypeError: ``use_tools`` is not a boolean.

        """
        rubric = sections.get("judge_rubric", "").strip()
        if not rubric:
            raise ValueError(f"{where}: text mode requires a non-empty judge_rubric")
        use_tools = meta.get("use_tools", False)
        if not isinstance(use_tools, bool):
            raise TypeError(f"{where}: use_tools must be a boolean.")
        return cls(judge_rubric=rubric, use_tools=use_tools)

    async def trial(self, case: Case, run: Run) -> bool:
        """Have a producer answer, then the judge score the answer.

        Args:
          case: The case.
          run: The models and prompt.

        Returns:
          flagged: Whether the judge said yes.

        """
        tools: list[Tool] = (
            [Read(), Grep(), Glob(), List(), Bash()] if self.use_tools else []
        )
        with workspace(case) as root:
            async with producer(run, tools=tools) as agent:
                answer = await converse(agent, case, root)
        judge = replace(
            run,
            model=run.judge_model,
            system=(
                "You are a strict classifier. Read the answer text and the rubric,"
                " then reply with exactly one line: 'VERDICT: YES' or 'VERDICT: NO'."
                " Judge only by the rubric; add nothing else."
            ),
            cli_context=False,
        )
        async with producer(judge, tools=[]) as agent:
            said = await agent.send(
                f"RUBRIC:\n{self.judge_rubric}\n\nANSWER:\n{answer}",
            )
        return _verdict(said)


# The last verdict line, not a substring: a judge that reasons aloud ("VERDICT: YES
# if ..., but VERDICT: NO") is scored by its final word.
def _verdict(text: str) -> bool:
    """Return whether the last ``VERDICT:`` line of ``text`` says YES."""
    verdicts = [
        line
        for line in text.upper().splitlines()
        if line.strip().startswith("VERDICT:")
    ]
    if not verdicts:
        raise ValueError(f"judge emitted no VERDICT line: {text!r}")
    return "YES" in verdicts[-1]
