"""Tests for ``prompt``: system prompt assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import datetime
import os
import shutil
import subprocess

import pytest

from sagent.prompt import (
    SagentContext,
    _git_state,
    _shell_name,
    project_context,
    render_environment,
    static_prompt,
)
from sagent.types.capability import ModelCapability


if TYPE_CHECKING:
    from pathlib import Path


def _model(model_id: str) -> ModelCapability:
    return ModelCapability(model_id=model_id)


@pytest.fixture(autouse=True)
def stub_recipe_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace recipe + AGENTS.md + memory hooks with deterministic stubs.

    Without this, the sections read filesystem assets and walk the cwd's
    AGENTS.md tree -- both flaky and slow inside a test harness.
    """
    sections: dict[str, str] = {
        "base": "BASE\n",
        "env": (
            "cwd={cwd} git={is_git} plat={platform} shell={shell_line}"
            " os={os_version} id={model_id}{knowledge_cutoff_line}{worktree_line}"
            " now={now}"
        ),
    }

    def fake_recipe_dict(key: str) -> dict[str, str]:
        if key == "system_prompt":
            return sections
        return {}

    def fake_recipe_list(section: str, key: str) -> list[str]:
        del section, key
        return []

    def fake_read_asset(path: object) -> str:
        # Either a recipe-relative key or an absolute path-like.
        if str(path) == "BASE\n":
            return "static body {keep_recent}"
        return str(path)

    def fake_agents(*_args: object, **_kwargs: object) -> str:
        return "AGENTS_SECTION"

    def fake_memory(*_args: object, **_kwargs: object) -> str:
        return "MEMORY_SECTION"

    def not_git(cwd: str) -> tuple[bool, bool]:
        del cwd
        return False, False

    monkeypatch.setattr("sagent.prompt.recipe_dict", fake_recipe_dict)
    monkeypatch.setattr("sagent.prompt.recipe_list", fake_recipe_list)
    monkeypatch.setattr("sagent.prompt.read_asset", fake_read_asset)
    monkeypatch.setattr(
        "sagent.prompt.agents_md.build_section",
        fake_agents,
    )
    monkeypatch.setattr(
        "sagent.prompt.memory.build_system_section",
        fake_memory,
    )
    monkeypatch.setattr("sagent.prompt._git_state", not_git)


def test_static_prompt_is_the_recipe_base() -> None:
    assert static_prompt() == "static body {keep_recent}"


def test_project_context_joins_agents_md_then_memory() -> None:
    assert project_context("/p") == "AGENTS_SECTION\n\nMEMORY_SECTION"


def test_project_context_leaves_out_what_is_not_asked_for() -> None:
    assert project_context("/p", include_memory=False) == "AGENTS_SECTION"
    assert project_context("/p", include_agents_md=False) == "MEMORY_SECTION"


def test_sagent_context_renders_only_the_sections_it_includes() -> None:
    context = SagentContext(include_agents_md=False, include_environment=False)
    assert context.project("/p", additional_dirs=()) == "MEMORY_SECTION"
    assert context.environment(_model("m"), cwd="/p", context="") == ""
    assert "cwd=/p" in SagentContext().environment(_model("m"), cwd="/p", context="")


def test_environment_names_the_hour_so_it_changes_once_an_hour() -> None:
    at = datetime.datetime(2026, 10, 23, 14, 37, tzinfo=datetime.UTC)
    assert "now=Fri, 2026-10-23 14:00 UTC" in render_environment(
        _model("m"),
        cwd="/p",
        now=at,
    )


def test_environment_section_includes_the_model_id_verbatim() -> None:
    assert "opus-4.8+1m" in render_environment(
        _model("opus-4.8"),
        cwd="/p",
        context="+1m",
    )


def test_environment_reads_knowledge_cutoff_from_the_model_spec() -> None:
    model = ModelCapability(
        model_id="opus-5.5",
        knowledge_cutoff="June 2026",
    )
    assert "June 2026" in render_environment(model, cwd="/p")


def test_environment_omits_an_unspecified_knowledge_cutoff() -> None:
    assert "Knowledge cutoff" not in render_environment(_model("local"), cwd="/p")


def test_shell_name_recognizes_bash() -> None:
    assert _shell_name("/usr/bin/bash") == "bash"
    assert _shell_name("/usr/local/bin/fish") == "/usr/local/bin/fish"


def _repo(root: Path, gitdir: str | None = None) -> Path:
    """Make ``root`` a checkout, or with ``gitdir`` one whose ``.git`` file names it."""
    root.mkdir(parents=True)
    if gitdir is None:
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    else:
        (root / gitdir).mkdir(parents=True)
        (root / gitdir / "HEAD").write_text("ref: refs/heads/main\n")
        (root / ".git").write_text(f"gitdir: {gitdir}\n")
    (root / "src").mkdir()
    return root / "src"


@pytest.mark.parametrize(
    ("gitdir", "state"),
    [
        (None, (True, False)),
        ("main/.git/worktrees/wt", (True, True)),
        ("main/.git/modules/sub", (True, False)),
    ],
)
def test_git_state_reads_the_nearest_dot_git(
    tmp_path: Path,
    gitdir: str | None,
    state: tuple[bool, bool],
) -> None:
    """A linked worktree's ``.git`` names a gitdir under ``worktrees``; a submodule's does not."""
    assert _git_state(str(_repo(tmp_path / "r", gitdir))) == state


def test_git_state_passes_over_a_dot_git_that_is_no_repository(tmp_path: Path) -> None:
    """Git looks past an empty ``.git`` directory, as one left in ``/tmp``."""
    inner = _repo(tmp_path / "r", "main/.git/worktrees/wt") / "a"
    (inner / ".git").mkdir(parents=True)
    assert _git_state(str(inner)) == (True, True)


def test_git_state_agrees_with_git(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git")
    env = {
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    who = ["-c", "user.name=t", "-c", "user.email=t@t"]
    for args in (
        ["init", "-q", "main"],
        ["-C", "main", *who, "commit", "-q", "--allow-empty", "-m", "c"],
        ["-C", "main", "worktree", "add", "-q", "../wt"],
    ):
        _ = subprocess.run([git, *args], cwd=tmp_path, env=env, check=True)  # noqa: S603 -- fixed git arguments.
    (tmp_path / "main" / "sub").mkdir()
    for path, state in (
        (tmp_path / "main" / "sub", (True, False)),
        (tmp_path / "wt", (True, True)),
    ):
        assert _git_state(str(path)) == state


def test_git_state_runs_no_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It runs on the event loop every model round; a hung ``git`` stalled the agent."""

    def refuse(*_args: object, **_kw: object) -> None:
        raise AssertionError("spawned a process")

    monkeypatch.setattr("subprocess.run", refuse)
    monkeypatch.setattr("subprocess.Popen", refuse)
    assert _git_state(str(_repo(tmp_path / "r"))) == (True, False)


def test_static_prompt_reflects_recipe_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``set_recipe`` swap must be visible on the next ``static_prompt`` call."""
    current = {"base": "A\n"}

    def fake_recipe_dict(key: str) -> dict[str, str]:
        return current if key == "system_prompt" else {}

    def fake_read_asset(path: object) -> str:
        return f"BODY-{str(path).strip()}"

    monkeypatch.setattr("sagent.prompt.recipe_dict", fake_recipe_dict)
    monkeypatch.setattr("sagent.prompt.read_asset", fake_read_asset)
    assert static_prompt() == "BODY-A"
    current["base"] = "B\n"
    assert static_prompt() == "BODY-B"


@pytest.mark.parametrize(
    ("include_memory", "sections", "expect_memory"),
    [
        (True, None, True),
        (False, None, False),
        (True, ["memory"], True),
        (False, ["memory"], False),
        (True, ["static"], False),
        (False, ["static"], False),
    ],
)
def test_include_memory_vs_recipe_sections(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_memory: bool,
    sections: list[str] | None,
    expect_memory: bool,
) -> None:
    """`include_memory` is an AND-gate with the recipe ``sections`` list."""

    def fake_recipe_list(section: str, key: str) -> list[str]:
        del section, key
        return sections or []

    monkeypatch.setattr("sagent.prompt.recipe_list", fake_recipe_list)
    context = project_context("/p", include_memory=include_memory)
    assert ("MEMORY_SECTION" in context) is expect_memory


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
