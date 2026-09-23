"""Tests for ``prompt``: system prompt assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import subprocess

import pytest

from sagent.prompt import (
    _is_git_repo,
    _is_git_worktree,
    _load_static,
    _shell_name,
    build_system,
    build_system_dict,
    environment,
)
from sagent.types.capability import ModelCapability


if TYPE_CHECKING:
    from collections.abc import Iterator


def _model(model_id: str) -> ModelCapability:
    return ModelCapability(model_id=model_id)


@pytest.fixture(autouse=True)
def stub_recipe_and_helpers() -> Iterator[None]:
    """Replace recipe + AGENTS.md + memory hooks with deterministic stubs.

    Without this, ``prompt.build_system`` reads filesystem assets and
    walks the cwd's AGENTS.md tree -- both flaky and slow inside a test
    harness.
    """
    sections: dict[str, str] = {
        "base": "BASE\n",
        "env": (
            "cwd={cwd} git={is_git} plat={platform} shell={shell_line}"
            " os={os_version} id={model_id}{knowledge_cutoff_line}{worktree_line}"
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
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        yield


def test_build_system_returns_concatenated_string() -> None:
    out = build_system(_model("sonnet-4.6"), custom="custom note")
    assert "static body" in out
    assert "AGENTS_SECTION" in out
    assert "MEMORY_SECTION" in out
    assert "custom note" in out
    # Section joiner.
    assert "\n\n" in out


def test_build_system_omits_memory_when_disabled() -> None:
    out = build_system(_model("sonnet-4.6"), include_memory=False)
    assert "MEMORY_SECTION" not in out


def test_build_system_dict_has_expected_keys() -> None:
    d = build_system_dict(_model("sonnet-4.6"), custom="hi")
    assert "static" in d
    assert "environment" in d
    assert "agents_md" in d
    assert "memory" in d
    assert "user_instructions" in d


def test_build_system_dict_no_user_instructions_when_custom_empty() -> None:
    d = build_system_dict(_model("sonnet-4.6"), custom="")
    assert "user_instructions" not in d


def test_environment_section_includes_the_model_id_verbatim() -> None:
    assert "opus-4.8+1m" in environment(_model("opus-4.8"), context="+1m")


def test_environment_reads_knowledge_cutoff_from_the_model_spec() -> None:
    model = ModelCapability(
        model_id="opus-5.5",
        knowledge_cutoff="June 2026",
    )
    assert "June 2026" in environment(model)


def test_environment_omits_an_unspecified_knowledge_cutoff() -> None:
    assert "Knowledge cutoff" not in environment(_model("local"))


def test_shell_name_recognizes_bash() -> None:
    assert _shell_name("/usr/bin/bash") == "bash"
    assert _shell_name("/usr/local/bin/fish") == "/usr/local/bin/fish"


def test_is_git_repo_true_on_zero_return(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert _is_git_repo("/tmp/has-git")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_is_git_repo_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("not found")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_repo("/tmp/no-git")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_is_git_repo_reflects_runtime_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache used to freeze the first answer; ``git init`` must flip the result."""
    returncode = [128]

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode[0],
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_repo("/tmp/flips")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.
    returncode[0] = 0
    assert _is_git_repo("/tmp/flips")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_is_git_worktree_false_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/fail-worktree")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_is_git_worktree_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("missing git")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/oserror")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_is_git_worktree_false_on_unexpected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="only-one-line",
            stderr="",
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/single-line")  # noqa: S108 -- Cwd token handed to a stubbed `subprocess.run`, never opened.


def test_load_static_reflects_recipe_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``set_recipe`` swap must be visible on the next ``_load_static`` call."""
    current = {"base": "A\n"}

    def fake_recipe_dict(key: str) -> dict[str, str]:
        return current if key == "system_prompt" else {}

    def fake_read_asset(path: object) -> str:
        return f"BODY-{str(path).strip()}"

    monkeypatch.setattr("sagent.prompt.recipe_dict", fake_recipe_dict)
    monkeypatch.setattr("sagent.prompt.read_asset", fake_read_asset)
    assert _load_static() == "BODY-A"
    current["base"] = "B\n"
    assert _load_static() == "BODY-B"


def test_build_system_dict_environment_is_lazy() -> None:
    d = build_system_dict(_model("opus-4.7"))
    env_section = d["environment"]
    # Environment is a callable so it re-evaluates per call.
    assert "opus-4.7" in env_section()


def test_build_system_dict_all_values_callable() -> None:
    """All sections expose the same shape (callable) so callers don't branch."""
    d = build_system_dict(_model("opus-4.7"), custom="ci")
    assert all(callable(v) for v in d.values())


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
    d = build_system_dict(_model("opus-4.7"), include_memory=include_memory)
    assert ("memory" in d) is expect_memory


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
