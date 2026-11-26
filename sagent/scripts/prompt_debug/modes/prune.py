"""The ``prune`` mode: a trial flags when the producer drops a rule file it follows.

A rule file's read is what goes first: it is old and seldom named again, though
every later edit still obeys it. The producer works in the trial's folder, which
holds the case's files (an AGENTS.md saying to follow STYLE.md, and that
STYLE.md) and four modules, each long enough that a stub shortens its read. It
answers the case's turns with ``AgentSession`` among its tools; a trial flags when
a read whose path matches ``flag_regex`` ends up queued to drop, or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Self

import logging
import re

from sagent.agent.agent import NoShrinker
from sagent.agent.compaction.pruner import Pruner
from sagent.agent.team import Team
from sagent.agent.types import Factory, Shrinker
from sagent.journal.types import ToolCall, ToolMessage
from sagent.model.capability import PruningPolicy
from sagent.model.types import Response
from sagent.providers.catalog import ModelCatalog
from sagent.scripts.prompt_debug.harness import converse, producer, workspace
from sagent.tools.agent_session import AgentSession
from sagent.tools.edit import Edit
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.agent.agent import Agent
    from sagent.scripts.prompt_debug.harness import Case, Run


__all__ = ["PruneMode"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PruneMode:
    """A trial flags when a read of a file matching ``flag_regex`` is queued or dropped.

    Attributes:
      flag_regex: The paths of the rule files.
      pruner: What prunes the producer's context. In a case file, ``pruner`` is
        ``none`` (nothing prunes on its own), ``shipped`` (sagent's pruner), or
        ``cold`` (sagent's pruner, its cache cold once the model has answered and
        every result it may drop dead, so a two-turn trial reaches the cut).
      session_description: What the model reads of ``AgentSession``; its own when
        empty.

    """

    flag_regex: str
    pruner: Factory[Shrinker]
    session_description: str = ""

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Self:
        """Return the mode a case file says: ``flag_regex`` and ``pruner``.

        Args:
          meta: The frontmatter.
          sections: The sections, by name; unused.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        Raises:
          ValueError: A key is missing, unknown, or an invalid regex.

        """
        del sections
        flag_regex = meta.get("flag_regex")
        if not isinstance(flag_regex, str) or not flag_regex:
            raise ValueError(f"{where}: prune mode requires a non-empty flag_regex")
        try:
            _ = re.compile(flag_regex)
        except re.error as e:
            raise ValueError(f"{where}: invalid flag_regex {flag_regex!r}: {e}") from e
        pruners: dict[object, Factory[Shrinker]] = {
            "none": NoShrinker,
            "shipped": Pruner,
            # Each reading leaps past any cache's lifetime.
            "cold": partial(
                Pruner,
                clock=partial(next, count(0.0, 1e9)),
                policy=PruningPolicy(keep_recent=1, expiry=0.0),
            ),
        }
        name = meta.get("pruner", "none")
        if name not in pruners:
            raise ValueError(
                f"{where}: pruner must be none, shipped, or cold, got {name!r}.",
            )
        return cls(flag_regex=flag_regex, pruner=pruners[name])

    async def trial(self, case: Case, run: Run) -> bool:
        """Have a producer read the folder and answer; look for a rule file dropped.

        Args:
          case: The case.
          run: The models and prompt.

        Returns:
          flagged: Whether a rule file's read was queued or dropped.

        """
        team = Team(catalog=ModelCatalog(factories={}))
        session = AgentSession(team=team)
        if self.session_description:
            session = replace(session, description=self.session_description)
        tools = [Read(), Edit(), Grep(), Glob(), session]
        with workspace(case) as root:
            for name in ("orders", "ledger", "report", "audit"):
                _ = (root / f"{name}.py").write_text(_source(name), encoding="utf-8")
            async with producer(
                run,
                tools=tools,
                cwd=root,
                pruner=self.pruner,
            ) as agent:
                _ = team.join(agent)
                _ = await converse(agent, case, root)
                gone = _pruned_reads(agent)
                why = [
                    r.arguments.get("reason")
                    for r in agent.session.history()
                    if isinstance(r, ToolCall) and r.name == AgentSession.name
                ]
        logger.info("prune trial: dropped or queued %s; said %s", gone, why)
        return any(re.search(self.flag_regex, path) for path in gone)


def _source(name: str) -> str:
    """Return a module of a dozen small functions, named for ``name``."""
    return "".join(
        f"def {name}_{i}(rows: list[dict[str, int]]) -> int:\n"
        f"    total = 0\n"
        f"    for row in rows:\n"
        f"        total += row.get('{name}_{i}', 0) * {i + 1}\n"
        f"    return total\n\n\n"
        for i in range(12)
    )


def _pruned_reads(agent: Agent) -> list[str]:
    """Return the file each read the agent queued to drop, or dropped, was of."""
    runtime = agent.runtime
    context = runtime.tape.context().messages
    reads = {
        r.call_id: str(r.arguments.get("file_path", ""))
        for e in context
        if isinstance(e, Response)
        for r in e.records
        if isinstance(r, ToolCall) and r.name == Read.name
    }
    return [
        reads[e.call_id]
        for e in context
        if isinstance(e, ToolMessage)
        and e.call_id in reads
        and (e.call_id in runtime.queued or e.text() == runtime.stubs.pruned)
    ]
