"""Load prompt-regression cases and score them against the real system.

A case (a ``.md`` file) pins a producer task and the most a guarded behavior may
show. :func:`run_case` runs fresh producer agents on the system prompt the sagent
CLI assembles, and its mode scores each one: a trial flags when the behavior
shows. A mode is a class with ``parse`` and ``trial``, named by the case:

- ``text``: a blind judge scores the producer's answer against a rubric.
- ``code``: the file the producer edited is matched against a regex.
- ``skill``: the trial flags when the producer never invoked a skill.
- ``tool_volume``: the trial flags when the producer made too many tool calls.
- ``prune``: the trial flags when the producer drops a rule file it was told to
  follow.
- ``package.module:Class``: any other mode, imported by name.

A case file is TOML frontmatter between ``---`` lines, then ``## <name>``
sections. The frontmatter has ``name``, ``n``, ``max_flag_rate``, ``mode``
(``text`` by default), and the mode's own keys; a mode ignores keys it does not
know. The sections are ``## producer_task``, an optional ``## setup_turn`` sent
first, any ``## file: <path>`` seeded into the trial's folder, and the mode's
own. ``{DIR}`` in either turn names that folder.

Typical usage example:

  async with ModelCatalog() as catalog:
      for case in load_cases():
          result = await run_case(case, model=catalog["anthropic"]["default"])
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import asyncio
import importlib
import tempfile
import tomllib

from treekle import from_plain

from sagent.agent.agent import Agent, NoShrinker, default_system_prompt
from sagent.agent.agents_md import AgentsMdConfig
from sagent.model.exceptions import RetryableError
from sagent.providers.types import Model
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence
    from importlib.resources.abc import Traversable

    from sagent.agent.types import Factory, Shrinker
    from sagent.tools.types import Tool


__all__ = [
    "Case",
    "CaseResult",
    "Mode",
    "ModeKind",
    "Run",
    "converse",
    "load_cases",
    "parse_case",
    "producer",
    "run_case",
    "workspace",
]


class Mode(Protocol):
    """How a case's trials run and are scored."""

    async def trial(self, case: Case, run: Run) -> bool:
        """Run one trial of ``case``; return whether the guarded behavior showed.

        Args:
          case: The case.
          run: The models and prompt the trial runs on.

        Returns:
          flagged: Whether the trial showed the behavior.

        """
        ...


class ModeKind(Protocol):
    """A mode class: builds its mode from a case file."""

    def parse(
        self,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> Mode:
        """Return the mode a case file's frontmatter and sections say.

        Args:
          meta: The frontmatter; keys the mode does not know are ignored.
          sections: The sections, by name.
          where: The file's name, for errors.

        Returns:
          mode: The mode.

        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Case:
    """One prompt-regression case.

    Attributes:
      name: Its name.
      n: How many trials.
      max_flag_rate: The most of ``n`` that may flag for the case to pass.
      mode: How each trial runs and is scored.
      producer_task: What the producer is asked.
      setup_turn: What it is asked first, its answer not scored; empty for none.
        A behavior that shows only once the agent holds a position needs the turn
        that sets it.
      files: What the trial's folder holds, by path relative to it.

    """

    name: str
    n: int
    max_flag_rate: float
    mode: Mode
    producer_task: str
    setup_turn: str = ""
    files: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    """What every trial of a run shares.

    Attributes:
      model: What the producers run on.
      judge_model: What a judge runs on.
      system: The producers' system prompt.
      cli_context: Whether the producers are told what the sagent CLI tells: the
        AGENTS.md of their folder and the environment they run in. An override
        prompt is the whole prompt under test, so it gets neither.

    """

    model: Model
    judge_model: Model
    system: str
    cli_context: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CaseResult:
    """The trials a case ran, each flagged or not."""

    case: Case
    verdicts: tuple[bool, ...]

    @property
    def flagged(self) -> int:
        """Return how many trials flagged."""
        return sum(self.verdicts)

    @property
    def rate(self) -> float:
        """Return the fraction of trials run that flagged."""
        return self.flagged / len(self.verdicts) if self.verdicts else 0.0

    @property
    def passed(self) -> bool:
        """Return whether the flagged count is within the case's tolerance.

        Scored against ``case.n``, not the trials run, so a run stopped early,
        once the outcome was certain, scores as a full one would.
        """
        return self.flagged <= self.case.max_flag_rate * self.case.n


def load_cases(package: str = f"{__package__}.cases") -> list[Case]:
    """Parse every ``.md`` case in ``package``.

    Args:
      package: The package whose case files to load.

    Returns:
      cases: The cases, by file name.

    Raises:
      ValueError: Two cases share a name.

    """
    found: dict[str, Traversable] = {
        p.name: p for p in files(package).iterdir() if p.name.endswith(".md")
    }
    cases = [parse_case(found[name]) for name in sorted(found)]
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case name(s): {', '.join(duplicates)}.")
    return cases


def parse_case(path: Traversable) -> Case:
    """Parse one case file, as the module docstring describes it.

    Args:
      path: The file.

    Returns:
      case: The case.

    Raises:
      ValueError: The file is malformed or names an unknown mode.
      TypeError: A frontmatter value has the wrong type.

    """
    where = path.name
    _, sep, rest = path.read_text(encoding="utf-8").partition("---\n")
    if not sep:
        raise ValueError(f"{where}: missing TOML frontmatter.")
    front, sep, body = rest.partition("\n---\n")
    if not sep:
        raise ValueError(f"{where}: missing frontmatter terminator.")
    meta = from_plain(tomllib.loads(front), dict[str, object])
    sections = _split_sections(body)
    mode = meta.get("mode", "text")
    if not isinstance(mode, str):
        raise TypeError(f"{where}: mode must be a string.")
    task = sections.get("producer_task", "").strip()
    if not task:
        raise ValueError(f"{where}: producer_task section is required.")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: name is required.")
    n = meta.get("n")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"{where}: n must be an integer.")
    rate = meta.get("max_flag_rate")
    if isinstance(rate, bool) or not isinstance(rate, int | float):
        raise TypeError(f"{where}: max_flag_rate must be numeric.")
    return Case(
        name=name.strip(),
        n=n,
        max_flag_rate=float(rate),
        mode=_kind(mode, where=where).parse(meta, sections, where=where),
        producer_task=task,
        setup_turn=sections.get("setup_turn", "").strip(),
        files={
            key.removeprefix("file:").strip(): text.strip() + "\n"
            for key, text in sections.items()
            if key.startswith("file:")
        },
    )


async def run_case(
    case: Case,
    *,
    model: Model,
    judge_model: Model | None = None,
    producer_system: str | None = None,
    max_concurrency: int = 4,
    trial_attempts: int = 4,
    progress: Callable[[int, int, bool], None] | None = None,
    stop_early: bool = True,
) -> CaseResult:
    """Run up to ``case.n`` trials of ``case``; return each one's verdict.

    Each trial is a whole multi-turn agent, so trials run ``max_concurrency`` at a
    time. More at once draw more dropped connections, each retried: on the skill
    case, 1 at a time took 90s, 4 took 31s, and 6 took 34s.

    A run stops once its outcome is certain: failed once the flagged count
    exceeds ``max_flag_rate * n``, passed once even every trial left flagging
    could not. A case can so carry a large ``n`` while a healthy run pays only
    the trials that prove it passes.

    Args:
      case: The case.
      model: What the producers run on.
      judge_model: What a judge runs on; ``model`` when ``None``.
      producer_system: The producers' system prompt, as ``--system-prompt``
        replaces the CLI's; ``None`` for the CLI's own, with its context.
      max_concurrency: The most trials at once.
      trial_attempts: Tries per trial: a :class:`RetryableError` retries it.
      progress: Told ``(done, n, verdict)`` as each trial ends.
      stop_early: Whether to stop once the outcome is certain; ``False``
        measures a whole arm.

    Returns:
      result: The verdicts of the trials run.

    """
    run = Run(
        model=model,
        judge_model=model if judge_model is None else judge_model,
        system=default_system_prompt() if producer_system is None else producer_system,
        cli_context=producer_system is None,
    )
    tally = _Tally()
    _ = await asyncio.gather(
        *(
            _work(
                case,
                run,
                tally,
                attempts=trial_attempts,
                progress=progress,
                stop_early=stop_early,
            )
            for _ in range(min(max_concurrency, case.n))
        ),
    )
    return CaseResult(case=case, verdicts=tuple(tally.verdicts))


def producer(
    run: Run,
    *,
    tools: Sequence[Tool],
    cwd: Path | None = None,
    pruner: Factory[Shrinker] = NoShrinker,
) -> Agent:
    """Return a producer agent on ``run``'s model and prompt.

    Args:
      run: Its model, prompt, and whether it is told the CLI's context.
      tools: What it may call.
      cwd: Where it works; the process's folder when ``None``, whose AGENTS.md
        and skills it then loads.
      pruner: What prunes its context.

    Returns:
      agent: The producer, not yet open.

    """
    return Agent(
        model=run.model,
        tools=tools,
        system_prompt=run.system,
        tool_context=ToolContext() if cwd is None else ToolContext(cwd=cwd),
        agents_md=None if run.cli_context else AgentsMdConfig(tiers=frozenset()),
        environment=run.cli_context,
        pruner=pruner,
    )


@contextmanager
def workspace(case: Case) -> Generator[Path]:
    """Yield a fresh folder holding ``case.files``; remove it after.

    Args:
      case: Whose files.

    Yields:
      root: The folder, outside any checkout.

    """
    with tempfile.TemporaryDirectory(prefix="prompt_debug_") as folder:
        root = Path(folder)
        for relative, text in case.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(text, encoding="utf-8")
        yield root


async def converse(agent: Agent, case: Case, root: Path) -> str:
    """Send an open ``agent`` the case's turns, ``{DIR}`` naming ``root``.

    Args:
      agent: The producer, open.
      case: Its setup turn, if any, then its task.
      root: The trial's folder.

    Returns:
      reply: What the producer said last, answering the task.

    """
    if case.setup_turn:
        _ = await agent.send(case.setup_turn.replace("{DIR}", str(root)))
    return await agent.send(case.producer_task.replace("{DIR}", str(root)))


def _kind(name: str, *, where: str) -> ModeKind:
    """Return the mode class ``name`` says: a built-in, or ``package.module:Class``."""
    modes = f"{__package__}.modes"
    spec = {
        "text": f"{modes}.text:TextMode",
        "code": f"{modes}.code:CodeMode",
        "skill": f"{modes}.skill:SkillMode",
        "tool_volume": f"{modes}.tool_volume:ToolVolumeMode",
        "prune": f"{modes}.prune:PruneMode",
    }.get(name, name)
    module, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(
            f"{where}: mode must be text, code, skill, tool_volume, prune,"
            f" or package.module:Class, got {name!r}.",
        )
    try:
        return cast(ModeKind, getattr(importlib.import_module(module), attr))
    except (ImportError, AttributeError) as e:
        raise ValueError(f"{where}: mode {name!r} not found: {e}") from e


@dataclass(slots=True, kw_only=True)
class _Tally:
    """The trials a run's workers have started and ended, shared among them."""

    verdicts: list[bool] = field(default_factory=list)
    started: int = 0
    settled: bool = False


async def _work(
    case: Case,
    run: Run,
    tally: _Tally,
    *,
    attempts: int,
    progress: Callable[[int, int, bool], None] | None,
    stop_early: bool,
) -> None:
    """Run trials of ``case`` until ``n`` have started or the outcome is settled."""
    fail_at = case.max_flag_rate * case.n
    while not tally.settled and tally.started < case.n:
        tally.started += 1
        verdict = await _with_retries(
            lambda: case.mode.trial(case, run),
            attempts=attempts,
        )
        tally.verdicts.append(verdict)
        done = len(tally.verdicts)
        if progress is not None:
            progress(done, case.n, verdict)
        flagged = sum(tally.verdicts)
        tally.settled = stop_early and (
            flagged > fail_at or flagged + case.n - done <= fail_at
        )


# A dropped connection clears with time; an immediate retry meets the same one.
async def _with_retries(fn: Callable[[], Awaitable[bool]], *, attempts: int) -> bool:
    """Await ``fn``, retrying a ``RetryableError`` after 1s, 2s, 4s, ...; once at least."""
    for attempt in range(attempts - 1):
        try:
            return await fn()
        except RetryableError:
            await asyncio.sleep(2.0**attempt)
    return await fn()


def _split_sections(body: str) -> dict[str, str]:
    """Return a case body's ``## <name>`` sections, by name; text before the first is dropped."""
    out: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current is not None:
                out[current] = "\n".join(lines)
            current = line[3:].strip()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        out[current] = "\n".join(lines)
    return out
