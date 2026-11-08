"""Tests for the SWE-bench driver: solving, grading, and the report, no network or Docker."""

from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast
from unittest.mock import Mock, patch

import argparse
import inspect
import logging
import os
import shutil
import signal
import subprocess
import threading

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

    calls: list[tuple[list[str], Path, str, float, Path | None]] = field(
        default_factory=list,
    )
    patch: bytes = b"diff --git a/x b/x\n"
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
        with self.lock:
            self.calls.append((list(argv), cwd, stdin, timeout_sec, output))
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
                _ = Path(arg.removeprefix("--output=")).write_bytes(self.patch)
        if argv[0] in ("sh", "claude", "codex"):
            if self.meet is not None:
                _ = self.meet.wait(timeout=self.meet_sec)
            return 7
        return 0

    def argvs(self) -> list[list[str]]:
        return [call[0] for call in self.calls]


def _solve(
    tmp_path: Path,
    git: _Git,
    *tasks: Task,
    model: str | None = None,
    jobs: int = 1,
    agent: Agent = "sagent",
) -> int:
    return solve(
        agent,
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


def _sized_tasks(**sizes: int) -> list[Task]:
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
    bare = command("sagent-noskills", _TASK, log=tmp_path, model=None, effort="high")
    assert bare[:7] == [
        "sh",
        str(main_py),
        "--session",
        str(tmp_path / "session.jsonl"),
        "--effort",
        "high",
        "--tools",
    ]
    assert bare[7:] == [
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
        "AgentSession",
        "BackgroundTask",
    ]


@pytest.mark.parametrize("agent", ["sagent-noskills", "claude"])
def test_a_task_is_solved_in_a_fresh_checkout_and_its_patch_kept(
    tmp_path: Path,
    agent: Agent,
) -> None:
    git = _Git()

    assert _solve(tmp_path, git, model="m", agent=agent) == 1

    mirror = tmp_path / "mirrors" / "o__r.git"
    checkout = tmp_path / "work" / "o__r-1"
    log = tmp_path / "out" / agent / "logs" / "o__r-1"
    argv = command(agent, _TASK, log=log, model="m", effort="medium")
    url = "https://github.com/o/r.git"
    diff = f"--output={log / 'patch.diff'}"
    assert git.calls == [
        (
            ["git", "clone", "--quiet", "--mirror", url, str(mirror)],
            tmp_path / "mirrors",
            "",
            3600.0,
            None,
        ),
        (
            ["git", "clone", "--quiet", "--shared", str(mirror), str(checkout)],
            tmp_path / "work",
            "",
            600.0,
            None,
        ),
        (["git", "checkout", "--quiet", "abc"], checkout, "", 600.0, None),
        (
            argv,
            checkout,
            prompt(_TASK) if agent == "sagent-noskills" else "",
            5.0,
            log / "stdout.txt",
        ),
        (["git", "add", "--all"], checkout, "", 600.0, None),
        (["git", "diff", "--cached", "--binary", diff], checkout, "", 600.0, None),
    ]
    (line,) = (tmp_path / "out" / agent / "predictions.jsonl").read_text().splitlines()
    assert json_loads(line, dict[str, object]) == {
        "instance_id": "o__r-1",
        "model_name_or_path": agent,
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
    assert git.calls[0] == (fetch, tmp_path / "mirrors", "", 600.0, None)
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 4


def test_an_agent_solves_jobs_tasks_at_once(tmp_path: Path) -> None:
    git = _Git(meet=threading.Barrier(3))

    assert _solve(tmp_path, git, *_sized_tasks(a=2, b=1), jobs=3) == 3

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
        _ = _solve(tmp_path, git, *_sized_tasks(a=2), jobs=1)


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


def test_a_task_logged_by_a_run_that_died_is_solved_again(tmp_path: Path) -> None:
    (tmp_path / "out" / "sagent" / "logs" / "o__r-1").mkdir(parents=True)
    assert _solve(tmp_path, _Git()) == 1


def test_a_checkout_its_agent_removed_is_no_error(tmp_path: Path) -> None:
    git = _Git()

    def removing(
        argv: Sequence[str],
        cwd: Path,
        stdin: str,
        timeout_sec: float,
        output: Path | None,
    ) -> int:
        code = git(argv, cwd, stdin, timeout_sec, output)
        if argv[1] == "diff":
            shutil.rmtree(cwd)
        return code

    assert (
        solve(
            "sagent",
            [_TASK],
            out=tmp_path / "out",
            mirrors=Mirrors(root=tmp_path / "mirrors"),
            work=tmp_path / "work",
            model=None,
            effort="low",
            timeout_sec=5.0,
            jobs=1,
            run=removing,
        )
        == 1
    )


def test_a_patch_that_is_not_utf_8_is_kept_as_far_as_it_reads(tmp_path: Path) -> None:
    assert _solve(tmp_path, _Git(patch=b"+calf\xe9\n")) == 1

    line = (tmp_path / "out" / "sagent" / "predictions.jsonl").read_bytes()
    assert json_loads(line, dict[str, object])["model_patch"] == "+calf\ufffd\n"


def test_a_mirror_root_is_made_with_its_parents(tmp_path: Path) -> None:
    root = tmp_path / "cache" / "mirrors"
    assert Mirrors(root=root).get("o/r", _Git()) == root / "o__r.git"


def test_two_repositories_are_mirrored_at_once(tmp_path: Path) -> None:
    meet = threading.Barrier(2)
    git = _Git()

    def meeting(
        argv: Sequence[str],
        cwd: Path,
        stdin: str,
        timeout_sec: float,
        output: Path | None,
    ) -> int:
        _ = meet.wait(timeout=5.0)
        return git(argv, cwd, stdin, timeout_sec, output)

    mirrors = Mirrors(root=tmp_path)

    def mirror(repo: str) -> Path:
        return mirrors.get(repo, meeting)

    with ThreadPoolExecutor(max_workers=2) as pool:
        got = list(pool.map(mirror, ["a/x", "b/y"]))

    assert got == [tmp_path / "a__x.git", tmp_path / "b__y.git"]


def test_a_failed_setup_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exit code 1"):
        _ = _solve(tmp_path, _Git(fail="checkout"))


def test_every_agent_solves_at_once_then_is_graded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git()
    out = tmp_path / "out"
    meet = threading.Barrier(4)

    def solving(*args: object, **kwargs: object) -> int:
        del args, kwargs
        return meet.wait(timeout=5.0)

    with patch.object(swebench, "solve", side_effect=solving) as solve_tasks:
        codes = _run_all(tmp_path, git, [_TASK], agents=swebench.AGENTS)

    assert codes == {"sagent": 0, "claude": 0, "codex": 0, "sagent-noskills": 0}
    assert solve_tasks.call_count == 4
    calls = {cast(Agent, call.args[0]): call for call in solve_tasks.call_args_list}
    assert set(calls) == set(swebench.AGENTS)
    for agent in swebench.AGENTS:
        invocation = calls[agent]
        assert invocation.args == (agent, [_TASK])
        configuration = cast(dict[str, object], invocation.kwargs)
        mirrors = configuration["mirrors"]
        assert isinstance(mirrors, Mirrors)
        assert mirrors.root == tmp_path / "mirrors"
        assert configuration == {
            "out": out,
            "mirrors": mirrors,
            "work": tmp_path / "work" / agent,
            "model": None,
            "effort": "high",
            "timeout_sec": 5.0,
            "jobs": 1,
            "run": git,
        }
    grades = {c[0][13]: c for c in git.calls if "--run_id" in c[0]}
    assert sorted(grades) == ["claude", "codex", "sagent", "sagent-noskills"]
    argv, cwd, stdin, timeout_sec, output = grades["codex"]
    assert (stdin, timeout_sec) == ("", 24 * 3600.0)
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


def _run_all(
    tmp_path: Path,
    git: _Git,
    tasks: Sequence[Task],
    *,
    agents: Sequence[swebench.Agent] = swebench.DEFAULT_AGENTS,
    model: str | None = None,
    jobs: int = 1,
) -> dict[swebench.Agent, int]:
    return run(
        agents,
        tasks,
        out=tmp_path / "out",
        mirrors=Mirrors(root=tmp_path / "mirrors"),
        work=tmp_path / "work",
        model=model,
        effort="high",
        timeout_sec=5.0,
        jobs=jobs,
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

    with patch.object(swebench, "solve", return_value=0):
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
    out = tmp_path / "out"
    devnull = Path(os.devnull)
    docker = [c for c in git.calls if c[0][0] == "docker"]
    assert docker[:3] == [
        (["docker", "image", "inspect", "img-a"], out, "", 600.0, devnull),
        (["docker", "image", "inspect", "img-b"], out, "", 600.0, devnull),
        (["docker", "rmi", "img-a"], out, "", 600.0, devnull),
    ]


def test_an_agent_given_no_tasks_has_graded_cleanly(tmp_path: Path) -> None:
    git = _Git()
    with patch.object(swebench, "solve", return_value=0) as solve_tasks:
        assert _run_all(tmp_path, git, [], agents=["codex"]) == {"codex": 0}
    assert solve_tasks.call_count == 1
    assert solve_tasks.call_args.args == ("codex", [])
    assert git.calls == []


def test_a_harness_that_fails_is_each_agents_code_and_progress_is_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)

    with (
        patch.object(swebench, "solve", return_value=1),
        caplog.at_level(logging.INFO, logger=swebench.__name__),
    ):
        codes = _run_all(tmp_path, _Git(fail="--run_id"), [_TASK], agents=["sagent"])

    assert codes == {"sagent": 1}
    assert caplog.messages == ["sagent: 1 solved", "graded 1 of 1 tasks"]


def test_each_agent_runs_with_the_model_and_time_it_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git()

    with patch.object(swebench, "solve", return_value=1) as solve_tasks:
        _ = _run_all(tmp_path, git, [_TASK], agents=["claude"], model="m")

    assert solve_tasks.call_count == 1
    assert solve_tasks.call_args.args == ("claude", [_TASK])
    assert solve_tasks.call_args.kwargs["model"] == "m"
    assert solve_tasks.call_args.kwargs["timeout_sec"] == 5.0
    assert solve_tasks.call_args.kwargs["work"] == tmp_path / "work" / "claude"


def test_a_run_passes_the_job_limit_to_each_solver(tmp_path: Path) -> None:
    with patch.object(swebench, "solve", return_value=2) as solve_tasks:
        _ = _run_all(tmp_path, _Git(), _sized_tasks(a=2), agents=["sagent"], jobs=2)
    assert solve_tasks.call_count == 1
    assert solve_tasks.call_args.kwargs["jobs"] == 2


def test_a_batch_that_fails_to_grade_still_removes_its_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git(crash="--run_id")

    with (
        patch.object(swebench, "solve", return_value=1),
        pytest.raises(OSError, match="--run_id"),
    ):
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


@pytest.mark.parametrize("text", ["{cut short", '{"a": {"resolved": "yes"}}'])
def test_a_report_the_harness_left_unreadable_is_ungraded(
    tmp_path: Path,
    text: str,
) -> None:
    _graded(tmp_path, "sagent", a=True, b=True)
    root = tmp_path / "sagent" / "logs" / "run_evaluation" / "sagent" / "sagent"
    _ = (root / "a" / "report.json").write_text(text)

    rows = report(tmp_path, agents=("sagent",)).splitlines()[2:]

    assert [r.split("|")[2:4] for r in rows] == [["        1 ", "      1 "]]


def test_a_result_line_of_the_wrong_shape_is_no_usage(tmp_path: Path) -> None:
    _logged(
        tmp_path,
        "claude",
        "i",
        "stdout.txt",
        _claude_result(0.25, 9, 1),
        {"type": "result", "total_cost_usd": 0.5},
    )
    assert usage("claude", tmp_path / "claude" / "logs" / "i") == Usage(
        cost_usd=0.25,
        input_tokens=9,
        output_tokens=1,
    )


def test_sigterm_exits_as_ctrl_c_would_with_its_signal_s_code() -> None:
    with pytest.raises(SystemExit) as exited:
        swebench._exit_on_term(signal.SIGTERM, None)
    assert exited.value.code == 128 + signal.SIGTERM


def test_the_report_is_a_row_per_agent(tmp_path: Path) -> None:
    _graded(tmp_path, "sagent", a=True, b=False, c=True)
    _graded(tmp_path, "claude", a=True)
    for instance in ("a", "b", "c"):
        _session(tmp_path, instance, _sagent_usage(1.0, 1_000_000, 1000))
    _logged(tmp_path, "claude", "a", "stdout.txt", _claude_result(0.5, 500_000, 1051))
    _graded(tmp_path, "sagent-noskills", a=False)

    text = report(tmp_path, agents=("sagent", "claude", "codex", "sagent-noskills"))

    assert text.splitlines() == [
        "| agent           | resolved | graded | rate | $/task | in/task | out/task |",
        "|-----------------|----------|--------|------|--------|---------|----------|",
        "| sagent          |        2 |      3 |  67% | $1.000 |   1000k |     1.0k |",
        "| claude          |        1 |      1 | 100% | $0.500 |    500k |     1.1k |",
        "| codex           |        - |      0 |    - |      - |       - |        - |",
        "| sagent-noskills |        0 |      1 |   0% |      ? |       ? |        ? |",
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
    tasks = _sized_tasks(big=6, mid=3, small=2, tiny=1)

    drawn = swebench.stratified(tasks, 5, seed=0)

    counts = {
        r: sum(t.repo == r for t in drawn) for r in ("big", "mid", "small", "tiny")
    }
    # The two spare draws go to small (5/6) and big (1/2).
    assert counts == {"big": 3, "mid": 1, "small": 1, "tiny": 0}
    assert drawn == [t for t in tasks if t in drawn]


def test_a_remainder_is_added_to_a_repositorys_floor() -> None:
    drawn = swebench.stratified(_sized_tasks(a=3, b=1), 2, seed=0)

    # Shares 1.5 and 0.5 tie; the one left over goes to the first.
    assert [sum(t.repo == r for t in drawn) for r in ("a", "b")] == [2, 0]


def test_a_sample_is_the_same_for_a_seed_and_differs_across_seeds() -> None:
    tasks = _sized_tasks(a=3, b=3)

    first = swebench.stratified(tasks, 2, seed=2)

    assert first == swebench.stratified(tasks, 2, seed=2)
    assert first != swebench.stratified(tasks, 2, seed=1)
    assert first != [tasks[0], tasks[3]]


def test_a_sample_as_large_as_the_tasks_is_all_of_them() -> None:
    tasks = _sized_tasks(a=2, b=1)
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
                "image": f"img{n}" if n else None,
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
    assert [t.image for t in every] == ["", "img1", "img2"]


def test_a_command_says_how_it_ended_and_keeps_what_it_printed(tmp_path: Path) -> None:
    script = ["agent", "solve"]
    output = tmp_path / "output.txt"
    output.write_text("stale")
    with patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(script, 7),
    ) as execute:
        assert swebench._run(script, tmp_path, "question\n", 5.0, output) == 7
    sink = cast(TextIO, execute.call_args.kwargs["stdout"])
    assert sink.closed
    assert sink.encoding == "utf-8"
    assert output.read_text() == ""
    execute.assert_called_once_with(
        script,
        cwd=tmp_path,
        input="question\n",
        stdout=sink,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5.0,
        check=False,
    )


def test_a_command_out_of_time_returns_124_and_closes_output(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired("agent", 0.01),
    ) as execute:
        assert swebench._run(["agent"], tmp_path, "", 0.01, output) == 124
    sink = cast(TextIO, execute.call_args.kwargs["stdout"])
    assert sink.closed
    execute.assert_called_once_with(
        ["agent"],
        cwd=tmp_path,
        input="",
        stdout=sink,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=0.01,
        check=False,
    )


def test_a_command_without_an_output_file_prints_where_this_process_does(
    tmp_path: Path,
) -> None:
    script = ["agent"]
    with patch.object(
        subprocess,
        "run",
        side_effect=[
            subprocess.CompletedProcess(script, 0),
            subprocess.CompletedProcess(script, 1),
        ],
    ) as execute:
        assert swebench._run(script, tmp_path, "", 5.0, None) == 0
        assert swebench._run(script, tmp_path, "", 5.0, None) == 1
    assert execute.call_count == 2
    execute.assert_called_with(
        script,
        cwd=tmp_path,
        input="",
        stdout=None,
        stderr=None,
        text=True,
        timeout=5.0,
        check=False,
    )


@pytest.fixture
def scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Path, None, None]:
    """Isolate CLI paths, signal handling, and logging configuration."""
    monkeypatch.setattr(swebench, "_OUT", tmp_path / "out")

    def basic_config(**kwargs: object) -> None:
        del kwargs

    monkeypatch.setattr(swebench, "_WORK", tmp_path / "work")
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    previous = signal.getsignal(signal.SIGTERM)
    loggers = [
        logging.getLogger(name) for name in ("httpx", "huggingface_hub", "datasets")
    ]
    levels = [logger.level for logger in loggers]
    try:
        yield tmp_path
    finally:
        _ = signal.signal(signal.SIGTERM, previous)
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def test_cli_fixture_restores_signal_and_logging_after_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels = [logging.DEBUG, logging.ERROR, logging.CRITICAL]
    setters = [Mock() for _ in levels]
    loggers = [
        Mock(spec=logging.Logger, level=level, setLevel=setter)
        for level, setter in zip(levels, setters, strict=True)
    ]
    unwrapped = cast(
        Callable[[Path, pytest.MonkeyPatch], Generator[Path, None, None]],
        inspect.unwrap(scratch),
    )
    error = RuntimeError("interrupted test")
    with (
        monkeypatch.context() as local,
        patch.object(signal, "getsignal", return_value=signal.SIG_IGN),
        patch.object(signal, "signal") as restore_signal,
        patch.object(logging, "getLogger", side_effect=loggers),
    ):
        lifecycle = unwrapped(tmp_path, local)
        assert next(lifecycle) == tmp_path
        restore_signal.assert_not_called()
        for logger in loggers:
            logger.level = logging.WARNING
        with pytest.raises(RuntimeError, match="interrupted test") as raised:
            lifecycle.throw(error)
    assert raised.value is error
    restore_signal.assert_called_once_with(signal.SIGTERM, signal.SIG_IGN)
    for setter, level in zip(setters, levels, strict=True):
        setter.assert_called_once_with(level)


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
    assert name(instances=("b", "a")) == "ids7e18f737-n50-seed0-medium"
    assert name(instances=("b", "a")) == name(instances=("a", "b"))
    assert name(instances=("a",)).startswith("ids")
    assert name(instances=("a",)) != name(instances=("b",))


def test_the_bare_command_runs_the_default_agents_where_report_finds_them(
    scratch: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = scratch / "out" / "n50-seed0-medium"
    for agent in swebench.AGENTS:
        (out / agent).mkdir(parents=True)
        (out / agent / "predictions.jsonl").touch()
    with (
        patch.object(swebench, "_tasks", return_value=[_TASK]) as tasks,
        patch.object(swebench, "run", return_value={}) as execute,
        patch.object(swebench, "report", return_value="the report") as describe,
    ):
        mirrors = ["--mirrors", str(scratch / "m")]
        assert swebench.main(mirrors) == 0
        assert swebench.main(["--agent", "sagent-noskills", *mirrors]) == 0
        assert swebench.main(["report"]) == 0
    assert tasks.call_count == 2
    tasks.assert_called_with(
        "SWE-bench/SWE-bench_Verified",
        instances=[],
        sample=50,
        seed=0,
    )
    assert execute.call_count == 2
    first, second = execute.call_args_list
    assert first.args == (["sagent", "claude", "codex"], [_TASK])
    assert second.args == (["sagent-noskills"], [_TASK])
    configuration = cast(dict[str, object], first.kwargs)
    mirror = configuration["mirrors"]
    assert isinstance(mirror, Mirrors)
    assert mirror.root == scratch / "m"
    assert configuration == {
        "out": out,
        "mirrors": mirror,
        "work": scratch / "work" / "n50-seed0-medium",
        "model": None,
        "effort": "medium",
        "timeout_sec": 1800.0,
        "jobs": 2,
        "dataset": "SWE-bench/SWE-bench_Verified",
        "workers": 2,
        "runner": swebench._run,
    }
    assert describe.call_count == 3
    assert (
        describe.call_args_list
        == [
            ((out,), {"agents": list(swebench.AGENTS)}),
        ]
        * 3
    )
    assert capsys.readouterr().out == f"{out}\nthe report\n" * 3


def test_a_run_removes_new_images_while_system_exit_unwinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swebench, "_DOCKER_SOCKET", tmp_path)
    git = _Git()
    killed = SystemExit(128 + signal.SIGTERM)
    with (
        patch.object(swebench, "solve", return_value=0) as solve_tasks,
        patch.object(swebench, "_grade", side_effect=killed) as grade,
        pytest.raises(SystemExit) as exited,
    ):
        _ = _run_all(tmp_path, git, _imaged("a"), agents=["codex"])
    assert exited.value is killed
    assert solve_tasks.call_count == 1
    grade.assert_called_once_with(
        "codex",
        ["a"],
        out=tmp_path / "out",
        dataset="d",
        workers=2,
        run=git,
    )
    assert git.argvs() == [
        ["docker", "image", "inspect", "img-a"],
        ["docker", "rmi", "img-a"],
    ]


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


def test_a_run_passes_every_flag_on_and_says_where_it_runs(
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ran: list[dict[str, object]] = []
    configured: list[dict[str, object]] = []

    def recorded(
        agents: Sequence[Agent],
        tasks: Sequence[Task],
        **kwargs: object,
    ) -> int:
        ran.append({"agents": list(agents), "tasks": list(tasks), **kwargs})
        return 0

    def basic_config(**kwargs: object) -> None:
        configured.append(kwargs)

    monkeypatch.setattr(swebench, "run", recorded)

    def tasks(dataset: str, **flags: object) -> list[Task]:
        del dataset, flags
        return [_TASK]

    monkeypatch.setattr(swebench, "_tasks", tasks)
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    # An earlier main may have quieted these and set the handler already.
    for noisy in ("httpx", "huggingface_hub", "datasets"):
        monkeypatch.setattr(logging.getLogger(noisy), "level", logging.NOTSET)
    previous = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with caplog.at_level(logging.INFO, logger=swebench.__name__):
            assert (
                swebench.main(
                    [
                        "--agent=codex",
                        "--agent=sagent",
                        "--instance=x",
                        "--model=m/1",
                        "--timeout-sec=9",
                        "--jobs=3",
                        "--workers=4",
                        f"--mirrors={scratch / 'm'}",
                    ],
                )
                == 0
            )
        handler = signal.getsignal(signal.SIGTERM)
    finally:
        _ = signal.signal(signal.SIGTERM, previous)

    name = swebench.experiment(
        dataset="SWE-bench/SWE-bench_Verified",
        instances=["x"],
        sample=50,
        seed=0,
        effort="medium",
        model="m/1",
    )
    assert "m_1" in name
    (call,) = ran
    mirrors = call.pop("mirrors")
    assert isinstance(mirrors, Mirrors)
    assert mirrors.root == scratch / "m"
    assert call == {
        "agents": ["sagent", "codex"],
        "tasks": [_TASK],
        "out": scratch / "out" / name,
        "work": scratch / "work" / name,
        "model": "m/1",
        "effort": "medium",
        "timeout_sec": 9.0,
        "jobs": 3,
        "dataset": "SWE-bench/SWE-bench_Verified",
        "workers": 4,
        "runner": swebench._run,
    }
    assert handler is swebench._exit_on_term
    assert configured == [{"level": logging.INFO, "format": "%(asctime)s %(message)s"}]
    for noisy in ("httpx", "huggingface_hub", "datasets"):
        assert logging.getLogger(noisy).level == logging.WARNING
    assert caplog.messages == [f"experiment {scratch / 'out' / name}: sagent, codex"]


def test_help_says_what_the_driver_does(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        swebench.main(["--help"])

    out = capsys.readouterr().out
    assert out.split("\n\n")[1] == (
        "Run SWE-bench Verified with sagent, Claude Code, and Codex; grade and"
        " compare them."
    )
    assert "  run:    (the default) every agent at once" in out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swebench")
    swebench._add_arguments(parser)
    return parser


def test_each_flag_defaults_as_its_help_says() -> None:
    assert vars(_parser().parse_args([])) == {
        "command": "run",
        "agent": [],
        "dataset": "SWE-bench/SWE-bench_Verified",
        "instance": [],
        "sample": 50,
        "seed": 0,
        "model": None,
        "effort": "medium",
        "timeout_sec": 1800.0,
        "jobs": 2,
        "workers": 2,
        "out": None,
        "mirrors": Path("/opt/scratch/caches/swebench/repos"),
        "work": None,
    }


def test_each_flag_reads_its_own_type() -> None:
    flags = _parser().parse_args(
        [
            "report",
            "--agent=sagent",
            "--agent=sagent-noskills",
            "--dataset=d/x",
            "--instance=a",
            "--instance=b",
            "--sample=3",
            "--seed=4",
            "--model=m",
            "--effort=high",
            "--timeout-sec=2.5",
            "--jobs=5",
            "--workers=6",
            "--out=/o",
            "--mirrors=/m",
            "--work=/w",
        ],
    )

    assert vars(flags) == {
        "command": "report",
        "agent": ["sagent", "sagent-noskills"],
        "dataset": "d/x",
        "instance": ["a", "b"],
        "sample": 3,
        "seed": 4,
        "model": "m",
        "effort": "high",
        "timeout_sec": 2.5,
        "jobs": 5,
        "workers": 6,
        "out": Path("/o"),
        "mirrors": Path("/m"),
        "work": Path("/w"),
    }


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_each_effort_is_taken(effort: str) -> None:
    flags = cast(swebench.Flags, _parser().parse_args([f"--effort={effort}"]))
    assert flags.effort == effort


@pytest.mark.parametrize(
    "argv",
    [["bogus"], ["--agent=nobody"], ["--effort=max"], ["--sample=x"]],
)
def test_a_value_no_flag_takes_is_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)


def test_the_help_says_what_each_flag_does(monkeypatch: pytest.MonkeyPatch) -> None:
    # Wide enough that no help line wraps.
    monkeypatch.setenv("COLUMNS", "1000")
    shown = _parser().format_help()

    for said in (
        "--agent {sagent,claude,codex,sagent-noskills}",
        "{run,report}",
        "--instance ID",
        "--effort {low,medium,high}",
        (
            "Who solves; repeat for more (default: sagent, claude, codex). For report,"
            " whose results (default: every agent in the experiment)."
        ),
        "The Hugging Face dataset of tasks (default: SWE-bench Verified).",
        "A task to solve; repeat for more. Every task when none is given.",
        (
            "Solve this many tasks, each repository's share as in the dataset; 500 is"
            " all of Verified (default: 50)."
        ),
        "Which sample --sample draws (default: 0).",
        "Every agent's model; its default when omitted.",
        "Every agent's reasoning effort (default: medium).",
        "How long an agent may take on one task (default: 1800).",
        "Tasks each agent solves at once (default: 2).",
        "Tasks graded at once, and Docker images kept at once (default: 2).",
        (
            "The experiment's directory"
            " (default: /opt/scratch/artifacts/sagent/swebench/<named by the flags>)."
        ),
        "Where each repository is mirrored once.",
        (
            "Where tasks are checked out while solved"
            " (default: /opt/scratch/runs/swebench/<experiment>)."
        ),
    ):
        assert said in shown, said


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
