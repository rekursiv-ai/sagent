"""Tests for ``prompt``: system prompt assembly."""

from __future__ import annotations

from collections.abc import Iterator
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


def test_environment_section_strips_context_tag() -> None:
    # The default model id carries a ``+1m`` context tag. The cutoff lookup
    # must canonicalize it like every other consumer (``_strip_context_tag``),
    # else the default session renders "Knowledge cutoff: unknown".
    out = environment("claude-opus-4-8+1m")
    assert "Claude Opus 4.8" in out
    assert "January 2026" in out


def test_environment_section_haiku_utility_model_has_cutoff() -> None:
    # The utility model's runtime id is ``claude-haiku-4-5``; the metadata
    # table must key on that, not a dated suffix, or the lookup misses and
    # the cutoff renders "unknown".
    out = environment("claude-haiku-4-5")
    assert "Claude Haiku 4.5" in out
    assert "February 2025" in out


def test_environment_section_fable_has_marketing_name() -> None:
    out = environment("claude-fable-5+1m")
    assert "Claude Fable 5" in out


def test_environment_section_sonnet_5_has_marketing_name() -> None:
    out = environment("claude-sonnet-5+1m")
    assert "Claude Sonnet 5" in out


def test_shell_name_recognizes_bash_and_zsh() -> None:
    assert _shell_name("/usr/bin/bash") == "bash"
    assert _shell_name("/bin/zsh") == "zsh"
    assert _shell_name("/usr/local/bin/fish") == "/usr/local/bin/fish"


def test_is_git_repo_true_on_zero_return(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert _is_git_repo("/tmp/has-git")  # noqa: S108 -- arbitrary cwd token


def test_is_git_repo_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("not found")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_repo("/tmp/no-git")  # noqa: S108 -- arbitrary cwd token


def test_is_git_repo_reflects_runtime_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache used to freeze the first answer; ``git init`` must flip the result."""
    returncode = [128]

    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=returncode[0], stdout="", stderr=""
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_repo("/tmp/flips")  # noqa: S108 -- arbitrary cwd token
    returncode[0] = 0
    assert _is_git_repo("/tmp/flips")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=128, stdout="", stderr=""
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/fail-worktree")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        raise OSError("missing git")

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/oserror")  # noqa: S108 -- arbitrary cwd token


def test_is_git_worktree_false_on_unexpected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="only-one-line", stderr=""
        )

    monkeypatch.setattr("sagent.prompt.subprocess.run", fake_run)
    assert not _is_git_worktree("/tmp/single-line")  # noqa: S108 -- arbitrary cwd token


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
    d = build_system_dict("claude-opus-4-7")
    env_section = d["environment"]
    # Environment is a callable so it re-evaluates per call.
    assert "claude-opus-4-7" in env_section()


def test_build_system_dict_all_values_callable() -> None:
    """All sections expose the same shape (callable) so callers don't branch."""
    d = build_system_dict("claude-opus-4-7", custom="ci")
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

    def fake_recipe_list(_section: str, _key: str) -> list[str]:
        return sections or []

    monkeypatch.setattr("sagent.prompt.recipe_list", fake_recipe_list)
    d = build_system_dict("claude-opus-4-7", include_memory=include_memory)
    assert ("memory" in d) is expect_memory


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
