"""Tests for the environment lines: git state, platform, shell, and cutoff."""

from __future__ import annotations

from typing import TYPE_CHECKING

import platform
import subprocess
import sys

import pytest

from sagent.agent.environment import environment


if TYPE_CHECKING:
    from pathlib import Path


def test_a_directory_outside_any_repository_says_so(tmp_path: Path) -> None:
    lines = environment(tmp_path).splitlines()

    assert lines[0] == "Git repository: false"
    assert not any("worktree" in line for line in lines)


def test_a_repository_is_named_without_the_worktree_warning(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    lines = environment(repo).splitlines()

    assert lines[0] == "Git repository: true"
    assert not any("worktree" in line for line in lines)


def test_a_linked_worktree_is_told_to_stay_put(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    gitdir = repo / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/wt\n")
    checkout = tmp_path / "wt"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../repo/.git/worktrees/wt\n")

    lines = environment(checkout).splitlines()

    assert lines[0] == (
        "This working directory is a git worktree (not the primary checkout)."
        " Stay in this directory -- do not `cd` to the main repository."
    )
    assert "do not `cd` to the main repository" in lines[0]
    assert lines[1] == "Git repository: true"


def test_the_git_state_is_read_without_running_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / "sub").mkdir()

    def refuse(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert environment(repo / "sub").splitlines()[0] == "Git repository: true"


def test_a_submodule_is_a_repository_not_a_worktree(tmp_path: Path) -> None:
    gitdir = tmp_path / "super" / ".git" / "modules" / "lib"
    gitdir.mkdir(parents=True)
    _ = (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    checkout = tmp_path / "super" / "lib"
    checkout.mkdir()
    _ = (checkout / ".git").write_text("gitdir: ../.git/modules/lib\n")

    lines = environment(checkout).splitlines()

    assert lines[0] == "Git repository: true"
    assert not any("worktree" in line for line in lines)


@pytest.mark.parametrize(
    "dot_git",
    [
        "gitdir: ../nowhere\n",  # Names a gitdir that is gone.
        b"\xff\xfe",  # Not text.
    ],
)
def test_a_dot_git_file_naming_no_repository_is_none(
    tmp_path: Path,
    dot_git: str | bytes,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if isinstance(dot_git, str):
        _ = (checkout / ".git").write_text(dot_git)
    else:
        _ = (checkout / ".git").write_bytes(dot_git)

    assert environment(checkout).splitlines()[0] == "Git repository: false"


def test_a_dot_git_directory_without_a_head_is_passed_over(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    (outer / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    inner = outer / "inner"
    (inner / ".git").mkdir(parents=True)

    assert environment(inner).splitlines()[0] == "Git repository: true"


def test_the_platform_and_os_lines_name_this_machine(tmp_path: Path) -> None:
    lines = environment(tmp_path).splitlines()

    assert f"Platform: {sys.platform}" in lines
    assert f"OS: {platform.system()} {platform.release()}" in lines


@pytest.mark.parametrize(
    ("shell", "line"),
    [
        ("/usr/bin/bash", "Shell: bash"),
        ("/bin/zsh", "Shell: /bin/zsh"),
    ],
)
def test_the_shell_is_bash_by_name_or_else_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shell: str,
    line: str,
) -> None:
    monkeypatch.setenv("SHELL", shell)

    assert line in environment(tmp_path).splitlines()


def test_an_unset_shell_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SHELL", raising=False)

    assert "Shell: unknown" in environment(tmp_path).splitlines()


def test_windows_is_told_to_keep_unix_conventions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    assert (
        "Shell: bash (prefer Unix shell conventions -- /dev/null, forward slashes)"
        in environment(tmp_path).splitlines()
    )


def test_the_knowledge_cutoff_is_the_last_line_when_known(tmp_path: Path) -> None:
    assert environment(tmp_path, knowledge_cutoff="June 2026").splitlines()[-1] == (
        "Knowledge cutoff: June 2026."
    )
    assert "Knowledge cutoff" not in environment(tmp_path)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
