"""Tests for ``prompt``: system prompt assembly."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import subprocess

import pytest

from sagent.prompt import (
    _is_git_repo,
    _is_git_worktree,
    _load_env_template,
    _load_static,
    _shell_name,
    build_system,
    build_system_dict,
    environment,
)


@pytest.fixture(autouse=True)
def _stub_recipe_and_helpers() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture consumed by name
    """Replace recipe + AGENTS.md + memory hooks with deterministic stubs.

    Without this, ``prompt.build_system`` reads filesystem assets and
    walks the cwd's AGENTS.md tree -- both flaky and slow inside a test
    harness.
    """
    sections: dict[str, str] = {
        "base": "BASE\n",
        "env": (
            "cwd={cwd} git={is_git} plat={platform} shell={shell_line}"
            " os={os_version} m={marketing} id={model_id} co={cutoff}{worktree_line}"
        ),
    }

    def fake_recipe_dict(key: str) -> dict[str, str]:
        if key == "system_prompt":
            return sections
        return {}

    def fake_recipe_list(_section: str, _key: str) -> list[str]:
        return []

    def fake_read_asset(path: object) -> str:
        # Either a recipe-relative key or an absolute path-like.
        if str(path) == "BASE\n":
            return "static body {keep_recent}"
        return str(path)

    patches = [
        patch(
            "sagent.prompt.recipe_dict",
            side_effect=fake_recipe_dict,
        ),
        patch(
            "sagent.prompt.recipe_list",
            side_effect=fake_recipe_list,
        ),
        patch(
            "sagent.prompt.read_asset",
            side_effect=fake_read_asset,
        ),
        patch(
            "sagent.prompt.agents_md.build_section",
            return_value="AGENTS_SECTION",
        ),
        patch(
            "sagent.prompt.memory.build_system_section",
            return_value="MEMORY_SECTION",
        ),
        patch(
            "sagent.prompt._is_git_repo",
            return_value=False,
        ),
        patch(
            "sagent.prompt._is_git_worktree",
            return_value=False,
        ),
    ]
    # Clear ``@cache`` decorators so stubs take effect.
    _load_static.cache_clear()
    _load_env_template.cache_clear()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        _load_static.cache_clear()
        _load_env_template.cache_clear()
        yield
    _load_static.cache_clear()
    _load_env_template.cache_clear()


def test_build_system_returns_concatenated_string() -> None:
    out = build_system("claude-sonnet-4-6", custom="custom note")
    assert "static body" in out
    assert "AGENTS_SECTION" in out
    assert "MEMORY_SECTION" in out
    assert "custom note" in out
    # Section joiner.
    assert "\n\n" in out


def test_build_system_omits_memory_when_disabled() -> None:
    out = build_system("claude-sonnet-4-6", include_memory=False)
    assert "MEMORY_SECTION" not in out


def test_build_system_dict_has_expected_keys() -> None:
    d = build_system_dict("claude-sonnet-4-6", custom="hi")
    assert "static" in d
    assert "environment" in d
    assert "agents_md" in d
    assert "memory" in d
    assert "user_instructions" in d


def test_build_system_dict_no_user_instructions_when_custom_empty() -> None:
    d = build_system_dict("claude-sonnet-4-6", custom="")
    assert "user_instructions" not in d


def test_environment_section_includes_model_metadata() -> None:
    out = environment("claude-sonnet-4-6")
    assert "claude-sonnet-4-6" in out
    assert "Claude Sonnet 4.6" in out
    assert "August 2025" in out


def test_environment_section_unknown_model_falls_back() -> None:
    out = environment("custom-model-xyz")
    assert "custom-model-xyz" in out
    assert "unknown" in out  # cutoff fallback


def test_shell_name_recognizes_bash_and_zsh() -> None:
    assert _shell_name("/usr/bin/bash") == "bash"
    assert _shell_name("/bin/zsh") == "zsh"
    assert _shell_name("/usr/local/bin/fish") == "/usr/local/bin/fish"


def test_is_git_repo_true_on_zero_return(monkeypatch: pytest.MonkeyPatch) -> None:
    _is_git_repo.cache_clear()

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert _is_git_repo("/tmp/has-git")  # noqa: S108 -- arbitrary cwd token


def test_is_git_repo_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    _is_git_repo.cache_clear()

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("not found")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_repo("/tmp/no-git")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _is_git_worktree.cache_clear()

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=128, stdout="", stderr=""
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/fail-worktree")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    _is_git_worktree.cache_clear()

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("missing git")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/oserror")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_on_unexpected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _is_git_worktree.cache_clear()

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="only-one-line", stderr=""
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/single-line")  # noqa: S108 -- arbitrary cwd token


def test_build_system_dict_environment_is_lazy() -> None:
    d = build_system_dict("claude-opus-4-7")
    env_section = d["environment"]
    # Environment is a callable so it re-evaluates per call.
    assert not isinstance(env_section, str)
    assert "claude-opus-4-7" in env_section()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
