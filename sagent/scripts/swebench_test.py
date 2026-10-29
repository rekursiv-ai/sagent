"""Tests for the SWE-bench driver: solving, grading, and the report, no network or Docker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import os
import signal
import threading
import time

from treekle import PlainTree, ReadError, json_dumps, json_loads

import pytest

from sagent.scripts import swebench
from sagent.scripts.swebench import (
    Agent,
    Mirrors,
    Task,
    Usage,
    command,
    prompt,
    report,
    run,
    solve,
    usage,
)
from sagent.session.jsonl import record_line
from sagent.session.types import TokenUsage, UserMessage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sagent.session.types import SessionRecord


_TASK = Task(
    instance_id="o__r-1",
    repo="o/r",
    base_commit="abc",
    problem_statement="  The bug.\n",
)


@dataclass(slots=True, kw_only=True)
class _Git:
    """Plays git, the agents, and the harness: clones make a directory, a diff a patch."""

    calls: list[tuple[list[str], Path, str, Path | None]] = field(default_factory=list)
    patch: str = "diff --git a/x b/x\n"
    fail: str = ""
    """A command word that makes the call fail."""

    meet: threading.Barrier | None = None
    """When set, every agent call waits here until as many have started."""

    meet_sec: float = 5.0
    """How long an agent call waits at ``meet`` before the barrier breaks."""

    images: set[str] = field(default_factory=set)
    """The Docker images already on the host."""

    crash: str = ""
    """A command word that makes the call raise, as a killed or broken run would."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(
        self,
        argv: Sequence[str],
        cwd: Path,
        stdin: str,
        timeout_sec: float,
        output: Path | None,
    ) -> int:
        del timeout_sec
        with self.lock:
            self.calls.append((list(argv), cwd, stdin, output))
        if self.crash and self.crash in argv:
            raise OSError(self.crash)
        if self.fail and self.fail in argv:
            return 1
        if argv[:3] == ["docker", "image", "inspect"]:
            return 0 if argv[3] in self.images else 1
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1]).mkdir(parents=True)
        for arg in argv:
            if arg.startswith("--output="):
                _ = Path(arg.removeprefix("--output=")).write_text(self.patch)
        if argv[0] in ("sh", "claude", "codex"):
            if self.meet is not None:
                _ = self.meet.wait(timeout=self.meet_sec)
            return 7
        return 0

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _, _, _ in self.calls]


def _solve(
    tmp_path: Path,
    git: _Git,
    *tasks: Task,
    model: str | None = None,
    jobs: int = 1,
) -> int:
    return solve(
        "sagent",
        tasks or (_TASK,),
        out=tmp_path / "out",
        mirrors=Mirrors(root=tmp_path / "mirrors"),
        work=tmp_path / "work",
        model=model,
        effort="medium",
        timeout_sec=5.0,
        jobs=jobs,
        run=git,
    )


def _tasks_of(**sizes: int) -> list[Task]:
    return [
        Task(
            instance_id=f"{repo}-{n}",
            repo=repo,
            base_commit="c",
            problem_statement="p",
        )
        for repo, size in sizes.items()
        for n in range(size)
    ]


def _graded(out: Path, agent: str, **resolved: bool) -> None:
    for instance, ok in resolved.items():
        root = out / agent / "logs" / "run_evaluation" / agent / agent / instance
        root.mkdir(parents=True)
        _ = (root / "report.json").write_text(
            json_dumps({instance: {"resolved": ok, "tests_status": {}}}),
        )


def _logged(out: Path, agent: str, instance: str, name: str, *lines: PlainTree) -> None:
    log = out / agent / "logs" / instance
    log.mkdir(parents=True, exist_ok=True)
    _ = (log / name).write_text("".join(json_dumps(line) + "\n" for line in lines))


def _session(out: Path, instance: str, *records: SessionRecord) -> None:
    log = out / "sagent" / "logs" / instance
    log.mkdir(parents=True, exist_ok=True)
    _ = (log / "session.jsonl").write_bytes(b"".join(record_line(r) for r in records))


def _sagent_usage(spend: float, prompt_tokens: int, response: int) -> TokenUsage:
    return TokenUsage(
        info={
            "request": prompt_tokens - 3,
            "response": response,
            "cache_write": 1,
            "cache_write_1h": 1,
            "cache_read": 1,
            "spend_usd": spend,
        },
    )


def _claude_result(cost: float, prompt_tokens: int, output: int) -> PlainTree:
    return {
        "type": "result",
        "total_cost_usd": cost,
        "usage": {
            "input_tokens": prompt_tokens - 2,
            "cache_creation_input_tokens": 1,
            "cache_read_input_tokens": 1,
            "output_tokens": output,
        },
    }


def test_every_agent_is_asked_the_same() -> None:
    assert prompt(_TASK) == (
        "The bug.\n\nResolve the issue above by editing the repository in the current"
        " directory. Change only what the fix needs. Do not commit."
    )


def test_each_agent_runs_headless_with_its_permission_prompts_off(
    tmp_path: Path,
) -> None:
    main_py = Path(swebench.__file__).resolve().parents[1] / "__main__.py"
    assert command("sagent", _TASK, log=tmp_path, model=None, effort="low") == [
        "sh",
        str(main_py),
        "--session",
        str(tmp_path / "session.jsonl"),
        "--effort",
        "low",
    ]
    assert command("claude", _TASK, log=tmp_path, model="opus", effort="low") == [
        "claude",
        "-p",
        prompt(_TASK),
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
        "--effort",
        "low",
        "--model",
        "opus",
    ]
    assert command("codex", _TASK, log=tmp_path, model=None, effort="low") == [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(tmp_path / "last_message.txt"),
        "-c",
        "model_reasoning_effort=low",
        prompt(_TASK),
    ]


def test_a_task_is_solved_in_a_fresh_checkout_and_its_patch_kept(
    tmp_path: Path,
) -> None:
    git = _Git()

    assert _solve(tmp_path, git, model="m") == 1

    mirror = tmp_path / "mirrors" / "o__r.git"
    checkout = tmp_path / "work" / "o__r-1"
    log = tmp_path / "out" / "sagent" / "logs" / "o__r-1"
    assert git.argvs()[:3] == [
        [
            "git",
            "clone",
            "--quiet",
            "--mirror",
            "https://github.com/o/r.git",
            str(mirror),
        ],
        ["git", "clone", "--quiet", "--shared", str(mirror), str(checkout)],
        ["git", "checkout", "--quiet", "abc"],
    ]
    agent, cwd, stdin, output = git.calls[3]
    assert agent[-4:] == ["--effort", "medium", "--model", "m"]
    assert (cwd, stdin, output) == (checkout, prompt(_TASK), log / "stdout.txt")
    assert git.argvs()[4:] == [
        ["git", "add", "--all"],
        ["git", "diff", "--cached", "--binary", f"--output={log / 'patch.diff'}"],
    ]
    (line,) = (
        (tmp_path / "out" / "sagent" / "predictions.jsonl").read_text().splitlines()
    )
    assert json_loads(line, dict[str, object]) == {
        "instance_id": "o__r-1",
        "model_name_or_path": "sagent",
        "model_patch": "diff --git a/x b/x\n",
        "exit_code": 7,
    }
    assert not checkout.exists()


def test_a_task_already_predicted_is_skipped_and_a_mirror_fetched_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out" / "sagent"
    root.mkdir(parents=True)
    _ = (root / "predictions.jsonl").write_text(
        json_dumps({"instance_id": "o__r-1"}) + "\n\n",
    )
    (tmp_path / "mirrors" / "o__r.git").mkdir(parents=True)
    more = [
        Task(
            instance_id=f"o__r-{n}",
            repo="o/r",
            base_commit="d",
            problem_statement="x",
        )
        for n in (2, 3)
    ]
    git = _Git()

    assert _solve(tmp_path, git, _TASK, *more) == 2

    fetch = ["git", "-C", str(tmp_path / "mirrors" / "o__r.git"), "fetch", "--quiet"]
    assert git.argvs().count(fetch) == 1
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 4


def test_an_agent_solves_jobs_tasks_at_once(tmp_path: Path) -> None:
    git = _Git(meet=threading.Barrier(3))

    assert _solve(tmp_path, git, *_tasks_of(a=2, b=1), jobs=3) == 3

    lines = (tmp_path / "out" / "sagent" / "predictions.jsonl").read_text().splitlines()
    assert sorted(
        json_loads(s, dict[str, str | int])["instance_id"] for s in lines
    ) == [
        "a-0",
        "a-1",
        "b-0",
    ]


def test_tasks_solved_one_at_a_time_never_meet(tmp_path: Path) -> None:
    """The barrier above proves concurrency: serially it breaks."""
    git = _Git(meet=threading.Barrier(2), meet_sec=0.01)
    with pytest.raises(threading.BrokenBarrierError):
        _ = _solve(tmp_path, git, *_tasks_of(a=2), jobs=1)


@pytest.mark.parametrize("line", ["[]", "{}", '{"instance_id": 1}'])
def test_a_predictions_file_with_a_line_that_is_none_stops_the_run(
    tmp_path: Path,
    line: str,
) -> None:
    root = tmp_path / "out" / "sagent"
    root.mkdir(parents=True)
    _ = (root / "predictions.jsonl").write_text(f"{line}\n")
    with pytest.raises(ReadError):
        _ = _solve(tmp_path, _Git())


def test_a_failed_setup_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code 1"):
        _ = _solve(tmp_path, _Git(fail="checkout"))


def test_every_agent_solves_at_once_then_is_graded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git(meet=threading.Barrier(4))
    out = tmp_path / "out"

    codes = _run_all(tmp_path, git, [_TASK], agents=swebench.AGENTS)

    assert codes == {"sagent": 0, "claude": 0, "codex": 0, "sagent-noskills": 0}
    clones = [a[-1] for a in git.argvs() if a[:3] == ["git", "clone", "--quiet"]]
    assert clones.count(str(tmp_path / "mirrors" / "o__r.git")) == 1
    for agent in swebench.AGENTS:
        assert str(tmp_path / "work" / agent / "o__r-1") in clones
    grades = {c[0][13]: c for c in git.calls if "--run_id" in c[0]}
    assert sorted(grades) == ["claude", "codex", "sagent", "sagent-noskills"]
    argv, cwd, _, output = grades["codex"]
    assert argv == [
        "uvx",
        "--from",
        "swebench",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "d",
        "--predictions_path",
        str(out / "codex" / "predictions.jsonl"),
        "--max_workers",
        "2",
        "--run_id",
        "codex",
        "--instance_ids",
        "o__r-1",
    ]
    assert (cwd, output) == (out / "codex", out / "codex" / "grade.log")
    sagents = {c[0][3]: (c[0], c[2]) for c in git.calls if c[0][0] == "sh"}
    plain, plain_stdin = sagents[
        str(out / "sagent" / "logs" / "o__r-1" / "session.jsonl")
    ]
    bare, bare_stdin = sagents[
        str(out / "sagent-noskills" / "logs" / "o__r-1" / "session.jsonl")
    ]
    assert plain_stdin == bare_stdin == prompt(_TASK)
    assert plain[-2:] == ["--effort", "high"]
    assert bare[: len(plain)] == [*plain[:3], bare[3], "--effort", "high"]
    assert bare[len(plain)] == "--tools"
    assert "Skill" not in bare
    assert {"Bash", "Read", "Edit"} <= set(bare)


def _run_all(
    tmp_path: Path,
    git: _Git,
    tasks: Sequence[Task],
    *,
    agents: Sequence[swebench.Agent] = swebench.DEFAULT_AGENTS,
) -> dict[swebench.Agent, int]:
    return run(
        agents,
        tasks,
        out=tmp_path / "out",
        mirrors=Mirrors(root=tmp_path / "mirrors"),
        work=tmp_path / "work",
        model=None,
        effort="high",
        timeout_sec=5.0,
        jobs=1,
        dataset="d",
        workers=2,
        runner=git,
    )


def _imaged(*names: str) -> list[Task]:
    return [
        Task(
            instance_id=n,
            repo="o/r",
            base_commit="c",
            problem_statement="p",
            image=f"img-{n}",
        )
        for n in names
    ]


def test_each_batch_is_graded_for_every_agent_then_its_new_images_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git(images={"img-b"})

    _ = _run_all(tmp_path, git, _imaged("a", "b", "c"))

    steps = [
        a[a.index("--run_id") + 1 :] if "--run_id" in a else a
        for a in git.argvs()
        if a[0] in ("uvx", "docker") and a[1] != "image"
    ]
    assert steps == [
        ["sagent", "--instance_ids", "a", "b"],
        ["claude", "--instance_ids", "a", "b"],
        ["codex", "--instance_ids", "a", "b"],
        ["docker", "rmi", "img-a"],
        ["sagent", "--instance_ids", "c"],
        ["claude", "--instance_ids", "c"],
        ["codex", "--instance_ids", "c"],
        ["docker", "rmi", "img-c"],
    ]


def test_a_batch_that_fails_to_grade_still_removes_its_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git(crash="--run_id")

    with pytest.raises(OSError, match="--run_id"):
        _ = _run_all(tmp_path, git, _imaged("a"))

    assert git.argvs()[-1] == ["docker", "rmi", "img-a"]


def test_a_checkout_is_removed_when_its_agent_crashes(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="sh"):
        _ = _solve(tmp_path, _Git(crash="sh"))
    assert not (tmp_path / "work" / "o__r-1").exists()


def test_grading_joins_the_docker_group_only_when_it_must(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path / "absent.sock")
    assert swebench._docker(["uvx", "a b"]) == [
        "/usr/bin/sg",
        "docker",
        "-c",
        "uvx 'a b'",
    ]
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    assert swebench._docker(["uvx", "a b"]) == ["uvx", "a b"]


def test_what_each_agent_spent_is_read_from_its_own_log(tmp_path: Path) -> None:
    _session(
        tmp_path,
        "i",
        UserMessage(content="x"),
        _sagent_usage(0.25, 100, 10),
        _sagent_usage(0.5, 200, 20),
    )
    _logged(
        tmp_path,
        "claude",
        "i",
        "stdout.txt",
        {"type": "system"},
        _claude_result(0.75, 300, 30),
    )
    _logged(
        tmp_path,
        "codex",
        "i",
        "stdout.txt",
        {"type": "thread.started"},
        {"type": "turn.completed", "usage": {"input_tokens": 400, "output_tokens": 40}},
        {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}},
    )

    assert usage("sagent", tmp_path / "sagent" / "logs" / "i") == Usage(
        cost_usd=0.75,
        input_tokens=300,
        output_tokens=30,
    )
    assert usage("claude", tmp_path / "claude" / "logs" / "i") == Usage(
        cost_usd=0.75,
        input_tokens=300,
        output_tokens=30,
    )
    assert usage("codex", tmp_path / "codex" / "logs" / "i") == Usage(
        cost_usd=None,
        input_tokens=405,
        output_tokens=41,
    )
    assert usage("sagent-noskills", tmp_path / "sagent" / "logs" / "i") == usage(
        "sagent",
        tmp_path / "sagent" / "logs" / "i",
    )
    for agent in swebench.AGENTS:
        assert usage(agent, tmp_path / "none") is None


@pytest.mark.parametrize("agent", ["claude", "codex"])
def test_a_log_that_never_says_what_it_spent_is_no_usage(
    tmp_path: Path,
    agent: Agent,
) -> None:
    _logged(tmp_path, agent, "i", "stdout.txt", {"type": "thread.started"})
    assert usage(agent, tmp_path / agent / "logs" / "i") is None


@pytest.mark.parametrize(
    ("agent", "cut"),
    [
        ("claude", '{"type": "result", "total_cost_usd": 0.5, "usage": {'),
        ("codex", '{"type": "turn.completed", "usage": {"input_tok'),
    ],
)
def test_a_log_cut_short_reads_what_it_finished(
    tmp_path: Path,
    agent: Agent,
    cut: str,
) -> None:
    """A run killed mid-write leaves a torn last line; the report still reads the rest."""
    whole = (
        _claude_result(0.25, 9, 1)
        if agent == "claude"
        else {
            "type": "turn.completed",
            "usage": {"input_tokens": 9, "output_tokens": 1},
        }
    )
    _logged(tmp_path, agent, "i", "stdout.txt", whole)
    with (tmp_path / agent / "logs" / "i" / "stdout.txt").open("a") as log:
        _ = log.write(cut + "\n")

    found = usage(agent, tmp_path / agent / "logs" / "i")

    assert found is not None
    assert (found.input_tokens, found.output_tokens) == (9, 1)


def test_a_report_the_harness_left_unreadable_is_ungraded(tmp_path: Path) -> None:
    _graded(tmp_path, "sagent", a=True, b=True)
    root = tmp_path / "sagent" / "logs" / "run_evaluation" / "sagent" / "sagent"
    _ = (root / "b" / "report.json").write_text("{cut short")

    rows = report(tmp_path, agents=("sagent",)).splitlines()[2:]

    assert [r.split("|")[2:4] for r in rows] == [["        1 ", "      1 "]]


def test_sigterm_exits_as_ctrl_c_would_with_its_signal_s_code() -> None:
    with pytest.raises(SystemExit) as exited:
        swebench._exit_on_term(signal.SIGTERM, None)
    assert exited.value.code == 128 + signal.SIGTERM


def test_the_report_is_a_row_per_agent(tmp_path: Path) -> None:
    _graded(tmp_path, "sagent", a=True, b=False, c=True)
    _graded(tmp_path, "claude", a=True)
    for instance in ("a", "b", "c"):
        _session(tmp_path, instance, _sagent_usage(1.0, 1_000_000, 1000))
    _logged(tmp_path, "claude", "a", "stdout.txt", _claude_result(0.5, 500_000, 2000))

    text = report(tmp_path, agents=("sagent", "claude", "codex"))

    assert text.splitlines() == [
        "| agent  | resolved | graded | rate | $/task | in/task | out/task |",
        "|--------|----------|--------|------|--------|---------|----------|",
        "| sagent |        2 |      3 |  67% | $1.000 |   1000k |     1.0k |",
        "| claude |        1 |      1 | 100% | $0.500 |    500k |     2.0k |",
        "| codex  |        - |      0 |    - |      - |       - |        - |",
    ]


@pytest.mark.parametrize("logged", [False, True], ids=["no-log", "no-price"])
def test_a_cost_not_known_for_every_task_is_not_summed(
    tmp_path: Path,
    logged: bool,
) -> None:
    _graded(tmp_path, "codex", a=True, b=True)
    turn = {"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 9}}
    _logged(tmp_path, "codex", "a", "stdout.txt", turn)
    _graded(tmp_path, "claude", a=True, b=True)
    _logged(tmp_path, "claude", "a", "stdout.txt", _claude_result(0.5, 9, 9))
    if logged:
        _logged(tmp_path, "claude", "b", "stdout.txt", {"type": "result"})

    rows = report(tmp_path, agents=("codex", "claude")).splitlines()[2:]

    assert [r.split("|")[5].strip() for r in rows] == ["?", "?"]


def test_a_sample_keeps_each_repositorys_share() -> None:
    tasks = _tasks_of(big=60, mid=30, small=9, tiny=1)

    drawn = swebench.stratified(tasks, 10, seed=0)

    counts = {
        r: sum(t.repo == r for t in drawn) for r in ("big", "mid", "small", "tiny")
    }
    # Shares 6, 3, 0.9, 0.1: the one left over goes to the larger remainder.
    assert counts == {"big": 6, "mid": 3, "small": 1, "tiny": 0}
    assert drawn == [t for t in tasks if t in drawn]


def test_a_sample_is_the_same_for_a_seed_and_differs_across_seeds() -> None:
    tasks = _tasks_of(a=50, b=50)

    first = swebench.stratified(tasks, 10, seed=1)

    assert first == swebench.stratified(tasks, 10, seed=1)
    assert first != swebench.stratified(tasks, 10, seed=2)
    assert first != tasks[:5] + tasks[50:55]


def test_a_sample_as_large_as_the_tasks_is_all_of_them() -> None:
    tasks = _tasks_of(a=2, b=1)
    assert swebench.stratified(tasks, 5, seed=0) == tasks


def test_tasks_are_read_from_the_dataset_named_and_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[str, str]] = []

    def load(name: str, *, split: str) -> list[dict[str, object]]:
        asked.append((name, split))
        return [
            {
                "instance_id": f"i{n}",
                "repo": "o/r",
                "base_commit": "c",
                "problem_statement": "p",
            }
            for n in range(3)
        ]

    monkeypatch.setattr(swebench, "load_dataset", load)

    every = swebench._tasks("d", instances=(), sample=50, seed=0)
    named = swebench._tasks("d", instances=("i2", "i0"), sample=50, seed=0)
    drawn = swebench._tasks("d", instances=(), sample=2, seed=0)

    assert asked == [("d", "test")] * 3
    assert [t.instance_id for t in every] == ["i0", "i1", "i2"]
    assert [t.instance_id for t in named] == ["i0", "i2"]
    assert drawn == swebench.stratified(every, 2, seed=0)
    assert every[0] == Task(
        instance_id="i0",
        repo="o/r",
        base_commit="c",
        problem_statement="p",
    )


def test_a_command_says_how_it_ended_and_keeps_what_it_printed(tmp_path: Path) -> None:
    script = ["sh", "-c", "read x; echo out $x; echo err >&2; test $x = y"]
    output = tmp_path / "output.txt"
    assert swebench._run(script, tmp_path, "y\n", 5.0, output) == 0
    assert output.read_text() == "out y\nerr\n"
    assert swebench._run(["false"], tmp_path, "", 5.0, None) == 1
    assert swebench._run(["sleep", "5"], tmp_path, "", 0.01, None) == 124


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default experiment and work roots into ``tmp_path``."""
    monkeypatch.setattr(swebench, "_OUT", tmp_path / "out")
    monkeypatch.setattr(swebench, "_WORK", tmp_path / "work")
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    return tmp_path


def test_an_experiment_is_named_by_its_flags() -> None:
    def name(
        *,
        dataset: str = "SWE-bench/SWE-bench_Verified",
        instances: Sequence[str] = (),
        sample: int = 50,
        seed: int = 0,
        effort: str = "medium",
        model: str | None = None,
    ) -> str:
        return swebench.experiment(
            dataset=dataset,
            instances=instances,
            sample=sample,
            seed=seed,
            effort=effort,
            model=model,
        )

    assert name() == "n50-seed0-medium"
    assert name(sample=500, seed=2, effort="high") == "n500-seed2-high"
    assert name(model="oaisub/gpt") == "n50-seed0-medium-oaisub_gpt"
    assert name(dataset="o/lite") == "o_lite-n50-seed0-medium"
    assert name(instances=("b", "a")) == name(instances=("a", "b"))
    assert name(instances=("a",)).startswith("ids")
    assert name(instances=("a",)) != name(instances=("b",))


def test_the_bare_command_runs_the_default_agents_where_report_finds_them(
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git = _Git()
    monkeypatch.setattr(swebench, "_run", git)
    asked: list[tuple[str, list[str], int, int]] = []

    def tasks(
        dataset: str,
        *,
        instances: Sequence[str],
        sample: int,
        seed: int,
    ) -> list[Task]:
        asked.append((dataset, list(instances), sample, seed))
        return [_TASK]

    monkeypatch.setattr(swebench, "_tasks", tasks)
    mirrors = ["--mirrors", str(scratch / "m")]

    assert swebench.main(mirrors) == 0
    assert swebench.main(["--agent", "sagent-noskills", *mirrors]) == 0
    ran = capsys.readouterr().out
    assert swebench.main(["report"]) == 0

    out = scratch / "out" / "n50-seed0-medium"
    assert asked == [("SWE-bench/SWE-bench_Verified", [], 50, 0)] * 2
    solvers = [a[0] for a in git.argvs() if a[0] in ("sh", "claude", "codex")]
    assert sorted(solvers) == ["claude", "codex", "sh", "sh"]
    checkouts = {a[-1] for a in git.argvs() if a[:2] == ["git", "clone"]}
    assert str(scratch / "work" / "n50-seed0-medium" / "codex" / "o__r-1") in checkouts
    last = [c.strip() for c in ran.splitlines()[-1].split("|")[1:-1]]
    assert last == ["sagent-noskills", "-", "0", "-", "-", "-", "-"]
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == str(out)
    assert [r.split("|")[1].strip() for r in shown[3:]] == [
        "sagent",
        "claude",
        "codex",
        "sagent-noskills",
    ]


def test_a_run_killed_by_sigterm_still_removes_its_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed while grading: the handler unwinds, so the pulled image is removed."""
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git()

    def killed(
        argv: Sequence[str],
        cwd: Path,
        stdin: str,
        timeout_sec: float,
        output: Path | None,
    ) -> int:
        code = git(argv, cwd, stdin, timeout_sec, output)
        if "--run_id" in argv:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        return code

    previous = signal.signal(signal.SIGTERM, swebench._exit_on_term)
    try:
        with pytest.raises(SystemExit) as exited:
            _ = run(
                ["codex"],
                _imaged("a"),
                out=tmp_path / "out",
                mirrors=Mirrors(root=tmp_path / "mirrors"),
                work=tmp_path / "work",
                model=None,
                effort="low",
                timeout_sec=5.0,
                jobs=1,
                dataset="d",
                workers=1,
                runner=killed,
            )
    finally:
        _ = signal.signal(signal.SIGTERM, previous)
    assert exited.value.code == 128 + signal.SIGTERM
    assert git.argvs()[-1] == ["docker", "rmi", "img-a"]


def test_a_report_names_only_the_agents_asked(
    scratch: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = scratch / "out" / "n50-seed0-medium"
    for agent in ("sagent", "codex"):
        (out / agent).mkdir(parents=True)
        _ = (out / agent / "predictions.jsonl").write_text("")
    assert swebench.main(["report", "--agent", "codex"]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == str(out)
    assert [r.split("|")[1].strip() for r in shown[3:]] == ["codex"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
