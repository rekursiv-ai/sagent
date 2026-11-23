"""Prove the prune note keeps the rules an agent was told to follow.

The bug: the pruner dropped tool results by its cost rule alone. A STYLE.md read
early and never named again was the first to go, though every later edit still
obeyed it. The fix: the pruner drops nothing itself; it notes what a drop would
pay for, and the model queues what it no longer needs.

Each trial is the ``prune-keeps-rules`` case: a fresh agent in a folder whose
AGENTS.md says to read and follow STYLE.md reads both and four code files, then
gets a question no edit follows, the moment the style rules look spent. A trial
flags when its STYLE.md read ends up dropped or queued. Its AGENTS.md stands in
the system prompt too, so dropping that read loses nothing and is not flagged.

Arms:
  auto  The old pruner: drops what its cost rule picks, asking no one.
  note  The shipped pruner: notes what would pay; the model queues.
  old   No pruner; the user asks it to prune; AgentSession described as before.

``auto`` and ``note`` run with the cache cold once the model has answered and
every result the policy lets go dead, so a two-turn trial reaches the cut; they
differ only in the pruner.

Results, 2026-11-23, the subscription Anthropic provider's default model:
  auto 8/8, note 0/10 (one-sided Fisher p ~ 2e-5); old 0/8. Asked, the model
  keeps STYLE.md "for future edits": the bug was the cost rule, not its judgment.

Examples:
  python -m sagent.scripts.prompt_debug experiment prune_note auto 8
  python -m sagent.scripts.prompt_debug experiment prune_note note 10

"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from importlib.resources import files
from itertools import count
from typing import TYPE_CHECKING, override

import logging

from sagent.agent.agent import NoShrinker
from sagent.agent.compaction.pruner import Pruner
from sagent.model.capability import PruningPolicy
from sagent.scripts.prompt_debug import cases
from sagent.scripts.prompt_debug.harness import parse_case, run_case
from sagent.scripts.prompt_debug.modes.prune import PruneMode


if TYPE_CHECKING:
    from sagent.agent.types import Factory, Shrinker
    from sagent.providers.types import Model
    from sagent.scripts.prompt_debug.harness import CaseResult


__all__ = ["run"]


logger = logging.getLogger(__name__)


async def run(arm: str, *, n: int, model: Model) -> CaseResult:
    """Run ``n`` trials of ``arm`` on ``model``; return each trial's verdict.

    Args:
      arm: ``auto``, ``note``, or ``old``, as the module docstring says.
      n: How many trials.
      model: What answers.

    Returns:
      result: Each trial's verdict: whether its STYLE.md read was dropped.

    Raises:
      ValueError: ``arm`` is none of those.
      TypeError: The guard case is not a ``prune`` case.

    """
    case = parse_case(files(cases) / "prune_keeps_rules.md")
    mode = case.mode
    if not isinstance(mode, PruneMode):
        raise TypeError(f"{case.name} is not a prune case.")
    pruners: dict[str, Factory[Shrinker]] = {
        # The guard's cold pruner, dropping what its cost rule picks.
        "auto": partial(
            _OldPruner,
            clock=partial(next, count(0.0, 1e9)),
            policy=PruningPolicy(keep_recent=1, expiry=0.0),
        ),
        "note": mode.pruner,
        "old": NoShrinker,
    }
    if arm not in pruners:
        raise ValueError(f"arm must be auto, note, or old, got {arm!r}.")
    task = case.producer_task
    asked = arm == "old"
    arm_case = replace(
        case,
        name=f"prune-note-{arm}",
        n=n,
        mode=replace(
            mode,
            pruner=pruners[arm],
            session_description=_old_description() if asked else "",
        ),
        producer_task=(
            "Your context is getting large. Prune the tool results you no longer"
            f" need, then tell me {task[0].lower()}{task[1:]}"
            if asked
            else task
        ),
    )
    return await run_case(
        arm_case,
        model=model,
        max_concurrency=2,
        progress=_progress,
        stop_early=False,
    )


class _OldPruner(Pruner):
    """The pruner before the queue: it drops what its cost rule picks, asking no one."""

    @override
    def prune(self) -> bool:
        self._agent.runtime.drop(self._plan())
        return False


def _progress(done: int, total: int, flagged: bool) -> None:
    """Log each trial as it ends."""
    logger.info("%d/%d flagged=%s", done, total, flagged)


def _old_description() -> str:
    """Return AgentSession's description before the queue, its prune action named as now."""
    return """Read and edit an agent's session: its context and everything before it.
Set `agent` to a label or relation; omit it for your own session.

Actions:
- `list` -- each entry of your context, one line: its id, what it is, its tokens, and how it starts. Use it to find what you no longer need.
- `read` -- the entries `ids`, whole; or the records `records`, by the number `search` or a watch feed gives, whole even if pruned or compacted away. A call's record comes with its result.
- `search` -- find `text` (case ignored) in the whole session, compacted windows included.
- `estimate` -- what pruning the tool outputs `ids` (results, or a long call's news) would save, and what it costs: dropping a result breaks the prompt cache at it, so everything after it is written once more.
- `prune_queue` -- replace the tool outputs `ids` with a short stub. Say why in `reason`. Prune what you will not need again; the result is gone from your context, though the session's journal keeps it.
- `compact` -- start over from a summary of the session: yours after this call, a descendant's before it returns; `prompt` says what it must keep. `style` is `summary` (condense what you see) or `session-audit` (audit the whole session: every user message verbatim, the steering, the decisions, the work, and what is pending); omitted, your compactor's style. The style chosen stays chosen.
- `clear` -- start over with nothing: yours after this call, a descendant's before it returns. Rare and destructive.

Mutations and estimates may target yourself or descendants; list, read, and
search may target any agent in your team. Your permissions apply, and the
target need not have AgentSession.

Your context is compacted and pruned automatically as it fills; do not do either defensively. Use these when the user asks, or when you know a result is spent.
"""
