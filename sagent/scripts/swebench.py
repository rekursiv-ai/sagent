#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Run SWE-bench Verified with sagent, Claude Code, and Codex; grade and compare them.

Each agent solves each task alone, headless, in a fresh checkout at the task's
base commit, and its patch is what it changed there. Grading runs the SWE-bench
harness, which applies each patch in the task's Docker image and runs its
tests. The agents run with their permission prompts off, so run this where
that is safe.

  run:    (the default) every agent at once, --jobs tasks each at a time;
          then grade --workers tasks at a time, removing each batch's Docker
          images; then print the report. Rerun the same command to resume:
          what is solved and graded is skipped.
  report: per agent, tasks resolved of tasks graded, and what they cost; per
          pair, the same on the tasks both had graded, and where they differ.

The flags name the experiment's directory, so the same flags always find it;
sagent-noskills is sagent without its Skill tool, and joins the same report.

Examples:
  sh sagent/scripts/swebench.py
  sh sagent/scripts/swebench.py --agent sagent-noskills
  sh sagent/scripts/swebench.py report
  sh sagent/scripts/swebench.py --agent sagent --instance django__django-11099

'''
# fmt: on

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol, TypedDict, cast

import argparse
import hashlib
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading

from treekle import PlainTree, from_plain, json_dumps, json_loads

from sagent.session.jsonl import read_record
from sagent.session.types import TokenUsage


_CWD: Final = Path(__file__).resolve().parent
_DOCKER_SOCKET: Final = Path("/var/run/docker.sock")
_DEVNULL: Final = Path(os.devnull)
_OUT: Final = Path("/opt/scratch/artifacts/sagent/swebench")
_WORK: Final = Path("/opt/scratch/runs/swebench")
_DATASET: Final = "SWE-bench/SWE-bench_Verified"

# sagent's every tool but Skill; its --tools names what to load, not what to drop.
_NO_SKILL_TOOLS: Final = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Grep",
    "Glob",
    "List",
    "WebSearch",
    "WebFetch",
    "PaperSearch",
    "PaperDetails",
    "PaperAuthor",
    "PaperFetch",
    "Memory",
    "PlayAudio",
    "AgentSpawn",
    "AgentSend",
    "AgentSelf",
    "Session",
    "BackgroundTask",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import FrameType

    from datasets import load_dataset
else:
    from wrapt import lazy_import

    load_dataset = lazy_import("datasets", "load_dataset")  # Seconds cold.


__all__ = [
    "AGENTS",
    "DEFAULT_AGENTS",
    "Agent",
    "Mirrors",
    "Task",
    "Usage",
    "command",
    "experiment",
    "main",
    "prompt",
    "report",
    "run",
    "solve",
    "stratified",
    "usage",
]


logger = logging.getLogger(__name__)

type Agent = Literal[
    "sagent",
    "claude",
    "codex",
    "sagent-noskills",
]

AGENTS: Final[tuple[Agent, ...]] = ("sagent", "claude", "codex", "sagent-noskills")
DEFAULT_AGENTS: Final[tuple[Agent, ...]] = ("sagent", "claude", "codex")

type Runner = Callable[[Sequence[str], Path, str, float, Path | None], int]
"""Run argv in a directory with stdin, a time limit in seconds, and the file its
output goes to (inherited when ``None``); return its exit code."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """One SWE-bench task: a repository at a commit, and the issue to resolve."""

    instance_id: str
    repo: str
    """``owner/name`` on GitHub."""

    base_commit: str
    problem_statement: str
    image: str = ""
    """The Docker image the harness grades it in; none named in older datasets."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """What one agent spent on one task, as its own logs report it."""

    cost_usd: float | None
    """``None`` when the agent reports tokens but no price."""

    input_tokens: int
    """Every prompt token, cached or not."""

    output_tokens: int


@dataclass(slots=True, kw_only=True)
class Mirrors:
    """Each repository mirrored once under ``root``, fetched once a run, for every thread."""

    root: Path
    _locks: dict[str, threading.Lock] = field(default_factory=dict, init=False)
    _fresh: set[str] = field(default_factory=set, init=False)
    _guard: threading.Lock = field(default_factory=threading.Lock, init=False)

    def get(self, repo: str, run: Runner) -> Path:
        """Return the mirror of ``repo``, cloning or fetching it the first time it is asked.

        Args:
          repo: ``owner/name`` on GitHub.
          run: What runs git.

        Returns:
          mirror: The bare mirror, current as of this run.

        """
        name = repo.replace("/", "__")
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        mirror = self.root / f"{name}.git"
        with lock:
            if name in self._fresh:
                return mirror
            if mirror.exists():
                argv = ["git", "-C", str(mirror), "fetch", "--quiet"]
                _must(run(argv, self.root, "", 600.0, None))
            else:
                self.root.mkdir(parents=True, exist_ok=True)
                url = f"https://github.com/{repo}.git"
                argv = ["git", "clone", "--quiet", "--mirror", url, str(mirror)]
                _must(run(argv, self.root, "", 3600.0, None))
            self._fresh.add(name)
        return mirror


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command ``argv`` names.

    Args:
      argv: The command line, less the program; ``sys.argv``'s when ``None``.

    Returns:
      code: The process exit code.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    name = experiment(
        dataset=flags.dataset,
        instances=flags.instance,
        sample=flags.sample,
        seed=flags.seed,
        effort=flags.effort,
        model=flags.model,
    )
    out = flags.out or _OUT / name
    if flags.command == "report":
        named: list[Agent] = [a for a in AGENTS if a in flags.agent]
        sys.stdout.write(f"{out}\n{report(out, agents=named or _predicted(out))}\n")
        return 0
    agents: list[Agent] = [a for a in AGENTS if a in (flags.agent or DEFAULT_AGENTS)]
    # SIGTERM kills without unwinding; as an exit it runs the image cleanup.
    _ = signal.signal(signal.SIGTERM, _exit_on_term)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.info("experiment %s: %s", out, ", ".join(agents))
    tasks = _tasks(
        flags.dataset,
        instances=flags.instance,
        sample=flags.sample,
        seed=flags.seed,
    )
    _ = run(
        agents,
        tasks,
        out=out,
        mirrors=Mirrors(root=flags.mirrors),
        work=flags.work or _WORK / name,
        model=flags.model,
        effort=flags.effort,
        timeout_sec=flags.timeout_sec,
        jobs=flags.jobs,
        dataset=flags.dataset,
        workers=flags.workers,
        runner=_run,
    )
    sys.stdout.write(f"{out}\n{report(out, agents=_predicted(out))}\n")
    return 0


def experiment(
    *,
    dataset: str,
    instances: Sequence[str],
    sample: int,
    seed: int,
    effort: str,
    model: str | None,
) -> str:
    """Return the directory name the flags give an experiment, the same for the same flags.

    Args:
      dataset: The tasks' dataset; named only when not the default.
      instances: The tasks named, if any; named by a hash of them.
      sample: How many tasks are drawn.
      seed: Which draw.
      effort: Every agent's reasoning effort.
      model: Every agent's model, if one is given.

    Returns:
      name: E.g. ``n50-seed0-medium``.

    """
    parts = [] if dataset == _DATASET else [dataset.replace("/", "_")]
    if instances:
        ids = "\n".join(sorted(instances)).encode()
        parts.append(f"ids{hashlib.sha256(ids).hexdigest()[:8]}")
    parts += [f"n{sample}", f"seed{seed}", effort]
    if model is not None:
        parts.append(model.replace("/", "_"))
    return "-".join(parts)


class Flags(Protocol):
    """Parsed command-line flags."""

    command: str
    agent: list[Agent]
    dataset: str
    instance: list[str]
    sample: int
    seed: int
    model: str | None
    effort: str
    timeout_sec: float
    jobs: int
    workers: int
    out: Path | None
    mirrors: Path
    work: Path | None


def run(
    agents: Sequence[Agent],
    tasks: Sequence[Task],
    *,
    out: Path,
    mirrors: Mirrors,
    work: Path,
    model: str | None,
    effort: str,
    timeout_sec: float,
    jobs: int,
    dataset: str,
    workers: int,
    runner: Runner,
) -> dict[Agent, int]:
    """Have every agent solve ``tasks`` at once, then grade them a batch at a time.

    Each batch is ``workers`` tasks, graded for every agent; then the Docker
    images it pulled are removed, so the disk holds one batch of images at a
    time. An image there before the batch is kept, as is one a container uses.

    Args:
      agents: Who solves.
      tasks: What each solves.
      out: Where each agent's predictions, logs, and grades go.
      mirrors: The repositories, shared by every agent.
      work: Where each agent checks out each task.
      model: Every agent's model; each one's own default when ``None``.
      effort: Every agent's reasoning effort.
      timeout_sec: How long an agent may take on one task.
      jobs: Tasks each agent solves at once.
      dataset: The tasks' dataset, for the harness.
      workers: Tasks the harness grades at once, and in a batch.
      runner: What runs the agents, git, docker, and the harness.

    Returns:
      codes: Each agent's worst harness exit code.

    """

    def one(agent: Agent) -> None:
        solved = solve(
            agent,
            tasks,
            out=out,
            mirrors=mirrors,
            work=work / agent,
            model=model,
            effort=effort,
            timeout_sec=timeout_sec,
            jobs=jobs,
            run=runner,
        )
        logger.info("%s: %d solved", agent, solved)

    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        _ = list(pool.map(one, agents))
    codes: dict[Agent, int] = dict.fromkeys(agents, 0)
    for start in range(0, len(tasks), workers):
        batch = tasks[start : start + workers]
        images = sorted({t.image for t in batch if t.image})
        present = {
            i
            for i in images
            if runner(
                _docker(["docker", "image", "inspect", i]),
                out,
                "",
                600.0,
                _DEVNULL,
            )
            == 0
        }
        try:
            for agent in agents:
                code = _grade(
                    agent,
                    [t.instance_id for t in batch],
                    out=out,
                    dataset=dataset,
                    workers=workers,
                    run=runner,
                )
                codes[agent] = codes[agent] or code
        finally:
            pulled = [i for i in images if i not in present]
            if pulled:
                # No --force: an image a container still uses stays.
                _ = runner(
                    _docker(["docker", "rmi", *pulled]),
                    out,
                    "",
                    600.0,
                    _DEVNULL,
                )
        logger.info("graded %d of %d tasks", start + len(batch), len(tasks))
    return codes


def solve(
    agent: Agent,
    tasks: Iterable[Task],
    *,
    out: Path,
    mirrors: Mirrors,
    work: Path,
    model: str | None,
    effort: str,
    timeout_sec: float,
    jobs: int,
    run: Runner,
) -> int:
    """Have ``agent`` solve each task not yet in its predictions; return how many it did.

    Args:
      agent: Who solves.
      tasks: What to solve.
      out: Where ``<agent>/predictions.jsonl`` and each task's log go.
      mirrors: The repositories each checkout is cloned from.
      work: Where each task is checked out, and removed once its patch is kept.
      model: The agent's model; its own default when ``None``.
      effort: The agent's reasoning effort.
      timeout_sec: How long the agent may take on one task.
      jobs: Tasks solved at once.
      run: What runs the agent and git.

    Returns:
      solved: The tasks this call wrote a prediction for.

    """
    root = out / agent
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.jsonl"
    done = _done(predictions)
    pending = [t for t in tasks if t.instance_id not in done]
    written = threading.Lock()

    def one(task: Task) -> None:
        checkout = _checkout(
            task,
            mirror=mirrors.get(task.repo, run),
            work=work,
            run=run,
        )
        log = root / "logs" / task.instance_id
        log.mkdir(parents=True, exist_ok=True)
        argv = command(agent, task, log=log, model=model, effort=effort)
        stdin = prompt(task) if agent.startswith("sagent") else ""
        try:
            code = run(argv, checkout, stdin, timeout_sec, log / "stdout.txt")
            patch = _patch(checkout, log=log, run=run)
        finally:
            shutil.rmtree(checkout, ignore_errors=True)
        record: dict[str, PlainTree] = {
            "instance_id": task.instance_id,
            "model_name_or_path": agent,
            "model_patch": patch,
            "exit_code": code,
        }
        with written, predictions.open("a", encoding="utf-8") as file:
            _ = file.write(json_dumps(record) + "\n")
        logger.info(
            "%s: %s exit %d, %d-byte patch",
            agent,
            task.instance_id,
            code,
            len(patch),
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        _ = list(pool.map(one, pending))
    return len(pending)


def command(
    agent: Agent,
    task: Task,
    *,
    log: Path,
    model: str | None,
    effort: str,
) -> list[str]:
    """Return how ``agent`` is run headless on ``task``, its permission prompts off.

    Args:
      agent: Who solves.
      task: What it solves; Claude Code and Codex take the prompt as an argument.
      log: Where the agent's own session record goes.
      model: The agent's model; its own default when ``None``.
      effort: The agent's reasoning effort, the same for every agent.

    Returns:
      argv: The command; sagent reads its prompt from stdin.

    """
    chosen = ["--model", model] if model is not None else []
    if agent in ("sagent", "sagent-noskills"):
        main_py = _CWD.parent / "__main__.py"
        tools = ["--tools", *_NO_SKILL_TOOLS] if agent == "sagent-noskills" else []
        return [
            "sh",
            str(main_py),
            "--session",
            str(log / "session.jsonl"),
            "--effort",
            effort,
            *chosen,
            *tools,
        ]
    if agent == "claude":
        return [
            "claude",
            "-p",
            prompt(task),
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--effort",
            effort,
            *chosen,
        ]
    return [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(log / "last_message.txt"),
        "-c",
        f"model_reasoning_effort={effort}",
        *chosen,
        prompt(task),
    ]


def prompt(task: Task) -> str:
    """Return what every agent is asked: the issue, and to fix it here."""
    return (
        f"{task.problem_statement.strip()}\n\n"
        "Resolve the issue above by editing the repository in the current directory."
        " Change only what the fix needs. Do not commit."
    )


def report(out: Path, *, agents: Sequence[Agent]) -> str:
    """Return a table of what each agent resolved and spent, one row per agent.

    Args:
      out: Where ``run`` left each agent's predictions, logs, and grades.
      agents: Whose results to read, in order.

    Returns:
      text: A Markdown table of per-task means, so agents graded on different
        numbers of tasks compare; cost is ``?`` unless every task has a price.

    """
    rows = [["agent", "resolved", "graded", "rate", "$/task", "in/task", "out/task"]]
    for a in agents:
        graded = _graded(out, a)
        if not graded:
            rows.append([a, "-", "0", "-", "-", "-", "-"])
            continue
        resolved = sum(graded.values())
        known = [u for i in graded if (u := usage(a, out / a / "logs" / i)) is not None]
        costs = [u.cost_usd for u in known if u.cost_usd is not None]
        priced = bool(costs) and len(costs) == len(known) == len(graded)
        n = len(known) or 1
        rows.append(
            [
                a,
                str(resolved),
                str(len(graded)),
                f"{100.0 * resolved / len(graded):.0f}%",
                f"${sum(costs) / len(costs):.3f}" if priced else "?",
                f"{sum(u.input_tokens for u in known) / n / 1e3:.0f}k",
                f"{sum(u.output_tokens for u in known) / n / 1e3:.1f}k",
            ],
        )
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]

    def line(row: Sequence[str]) -> str:
        cells = [
            row[0].ljust(widths[0]),
            *(v.rjust(w) for v, w in zip(row[1:], widths[1:], strict=True)),
        ]
        return "| " + " | ".join(cells) + " |"

    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(rows[0]), rule, *map(line, rows[1:])])


def usage(agent: Agent, log: Path) -> Usage | None:
    """Return what ``agent`` spent on the task logged in ``log``; ``None`` without a log.

    Args:
      agent: Whose log it is: sagent's session, or the others' printed output.
      log: The task's log directory.

    Returns:
      usage: Its cost and tokens; Codex prints no cost.

    """
    if agent in ("sagent", "sagent-noskills"):
        path = log / "session.jsonl"
        if not path.exists():
            return None
        infos = [
            from_plain(dict(record.info), _SagentInfo)
            for line in path.read_bytes().splitlines()
            if isinstance(record := read_record(line), TokenUsage)
        ]
        return Usage(
            cost_usd=sum(i["spend_usd"] for i in infos),
            input_tokens=sum(
                i["request"] + i["cache_write"] + i["cache_write_1h"] + i["cache_read"]
                for i in infos
            ),
            output_tokens=sum(i["response"] for i in infos),
        )
    path = log / "stdout.txt"
    if not path.exists():
        return None
    lines = [
        s for s in path.read_text(encoding="utf-8").splitlines() if s.startswith("{")
    ]
    if agent == "claude":
        found = [
            result
            for s in lines
            if '"total_cost_usd"' in s and (result := _read(s, _ClaudeResult))
        ]
        if not found:
            return None
        result = found[-1]
        u = result["usage"]
        return Usage(
            cost_usd=result["total_cost_usd"],
            input_tokens=u["input_tokens"]
            + u["cache_creation_input_tokens"]
            + u["cache_read_input_tokens"],
            output_tokens=u["output_tokens"],
        )
    turns = [
        turn["usage"]
        for s in lines
        if '"turn.completed"' in s and (turn := _read(s, _CodexTurn))
    ]
    if not turns:
        return None
    return Usage(
        cost_usd=None,
        input_tokens=sum(t["input_tokens"] for t in turns),
        output_tokens=sum(t["output_tokens"] for t in turns),
    )


def stratified(tasks: Sequence[Task], size: int, *, seed: int) -> list[Task]:
    """Return ``size`` of ``tasks`` at random, each repository's share as in ``tasks``.

    Each repository gets the floor of its share; what is left goes to the
    largest remainders, so the shares sum to ``size``.

    Args:
      tasks: What to draw from.
      size: How many to draw; all of ``tasks`` when it is at least their number.
      seed: Which draw: the same seed draws the same tasks.

    Returns:
      sample: The tasks drawn, in the order of ``tasks``.

    """
    if size >= len(tasks):
        return list(tasks)
    by_repo: dict[str, list[Task]] = {}
    for task in tasks:
        by_repo.setdefault(task.repo, []).append(task)
    repos = sorted(by_repo)
    shares = {r: size * len(by_repo[r]) / len(tasks) for r in repos}
    counts = {r: int(shares[r]) for r in repos}
    left = size - sum(counts.values())
    for r in sorted(repos, key=lambda r: counts[r] - shares[r])[:left]:
        counts[r] += 1

    def rank(task: Task) -> str:
        return hashlib.sha256(f"{seed}:{task.instance_id}".encode()).hexdigest()

    chosen = {
        t.instance_id for r in repos for t in sorted(by_repo[r], key=rank)[: counts[r]]
    }
    return [t for t in tasks if t.instance_id in chosen]


class _SagentInfo(TypedDict):
    request: int
    response: int
    cache_write: int
    cache_write_1h: int
    cache_read: int
    spend_usd: float


class _ClaudeUsage(TypedDict):
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int


class _ClaudeResult(TypedDict):
    total_cost_usd: float
    usage: _ClaudeUsage


class _CodexUsage(TypedDict):
    input_tokens: int
    output_tokens: int


class _CodexTurn(TypedDict):
    usage: _CodexUsage


class _Report(TypedDict):
    resolved: bool


# A run killed mid-write leaves its last line torn.
def _read[T](line: str, kind: type[T]) -> T | None:
    """Return ``line`` as ``kind``; ``None`` when it is no whole ``kind``."""
    try:
        return json_loads(line, kind)
    except ValueError:
        return None


def _graded(out: Path, agent: Agent) -> dict[str, bool]:
    """Return each task the harness finished for ``agent``, and whether it was resolved."""
    root = out / agent / "logs" / "run_evaluation" / agent / agent
    found: dict[str, bool] = {}
    for path in sorted(root.glob("*/report.json")):
        try:
            entries = json_loads(path.read_text(encoding="utf-8"), dict[str, _Report])
        # Malformed JSON or the wrong shape: a report the harness never finished.
        except ValueError:
            continue
        found.update({i: r["resolved"] for i, r in entries.items()})
    return found


def _done(predictions: Path) -> set[str]:
    """Return the tasks ``predictions`` already holds; raise on a line that is none."""
    if not predictions.exists():
        return set()
    return {
        from_plain(json_loads(line, dict[str, object]).get("instance_id"), str)
        for line in predictions.read_bytes().decode().splitlines()
        if line.strip()
    }


def _checkout(task: Task, *, mirror: Path, work: Path, run: Runner) -> Path:
    """Return a fresh checkout of the task's repository at its base commit."""
    checkout = work / task.instance_id
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(checkout, ignore_errors=True)
    argv = ["git", "clone", "--quiet", "--shared", str(mirror), str(checkout)]
    _must(run(argv, work, "", 600.0, None))
    _must(
        run(
            ["git", "checkout", "--quiet", task.base_commit],
            checkout,
            "",
            600.0,
            None,
        ),
    )
    return checkout


def _patch(checkout: Path, *, log: Path, run: Runner) -> str:
    """Return what was changed in ``checkout``, new files included, as a diff."""
    _must(run(["git", "add", "--all"], checkout, "", 600.0, None))
    diff = log / "patch.diff"
    argv = ["git", "diff", "--cached", "--binary", f"--output={diff}"]
    _must(run(argv, checkout, "", 600.0, None))
    return diff.read_text(encoding="utf-8", errors="replace")


# The harness skips a task it already has a report for.
def _grade(
    agent: Agent,
    instances: Sequence[str],
    *,
    out: Path,
    dataset: str,
    workers: int,
    run: Runner,
) -> int:
    """Run the SWE-bench harness on ``agent``'s predictions of ``instances``; return its code."""
    root = out / agent
    argv = [
        "uvx",
        "--from",
        "swebench",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--predictions_path",
        str(root / "predictions.jsonl"),
        "--max_workers",
        str(workers),
        "--run_id",
        agent,
        "--instance_ids",
        *instances,
    ]
    return run(_docker(argv), root, "", 24 * 3600.0, root / "grade.log")


def _docker(argv: list[str]) -> list[str]:
    """Return ``argv`` run in the ``docker`` group when this process is not yet in it."""
    if os.access(_DOCKER_SOCKET, os.W_OK):
        return argv
    # Joined to the group since login: only sg's new group set reaches the socket.
    return ["/usr/bin/sg", "docker", "-c", shlex.join(argv)]


def _exit_on_term(signum: int, frame: FrameType | None) -> NoReturn:
    """Exit as on Ctrl-C, unwinding, so cleanup in ``finally`` runs."""
    del frame
    raise SystemExit(128 + signum)


def _must(code: int) -> None:
    """Raise unless a command succeeded."""
    if code != 0:
        raise RuntimeError(f"a setup command failed with exit code {code}")


def _run(
    argv: Sequence[str],
    cwd: Path,
    stdin: str,
    timeout_sec: float,
    output: Path | None,
) -> int:
    """Run ``argv`` in ``cwd``; return its exit code, or 124 when it ran out of time."""
    with output.open("w", encoding="utf-8") if output else nullcontext() as sink:
        try:
            done = subprocess.run(  # noqa: S603 -- The argv is built here, never from input.
                list(argv),
                cwd=cwd,
                input=stdin,
                stdout=sink,
                stderr=subprocess.STDOUT if sink else None,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124
    return done.returncode


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the commands and their flags."""
    parser.add_argument("command", nargs="?", default="run", choices=["run", "report"])
    parser.add_argument(
        "--agent",
        action="append",
        choices=AGENTS,
        default=[],
        help=(
            "Who solves; repeat for more (default: sagent, claude, codex)."
            " For report, whose results (default: every agent in the experiment)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=_DATASET,
        help="The Hugging Face dataset of tasks (default: SWE-bench Verified).",
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        metavar="ID",
        help="A task to solve; repeat for more. Every task when none is given.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help=(
            "Solve this many tasks, each repository's share as in the dataset;"
            " 500 is all of Verified (default: 50)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Which sample --sample draws (default: 0).",
    )
    parser.add_argument(
        "--model",
        help="Every agent's model; its default when omitted.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Every agent's reasoning effort (default: medium).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=1800.0,
        help="How long an agent may take on one task (default: 1800).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Tasks each agent solves at once (default: 2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Tasks graded at once, and Docker images kept at once (default: 2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=f"The experiment's directory (default: {_OUT}/<named by the flags>).",
    )
    parser.add_argument(
        "--mirrors",
        type=Path,
        default=Path("/opt/scratch/caches/swebench/repos"),
        help="Where each repository is mirrored once.",
    )
    parser.add_argument(
        "--work",
        type=Path,
        help=f"Where tasks are checked out while solved (default: {_WORK}/<experiment>).",
    )


def _tasks(
    dataset: str,
    *,
    instances: Sequence[str],
    sample: int,
    seed: int,
) -> list[Task]:
    """Return ``sample`` of the dataset's tasks: of those named, else of all."""
    rows = cast("Iterable[dict[str, object]]", load_dataset(dataset, split="test"))
    tasks = [
        Task(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row["problem_statement"]),
            image=str(row.get("image") or ""),
        )
        for row in rows
        if not instances or row["instance_id"] in instances
    ]
    return stratified(tasks, sample, seed=seed)


def _predicted(out: Path) -> list[Agent]:
    """Return each agent with predictions under ``out``."""
    return [a for a in AGENTS if (out / a / "predictions.jsonl").exists()]


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
