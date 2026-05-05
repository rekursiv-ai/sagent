"""Tests for sagent.prompt."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from sagent.prompt import (
    _is_git_repo,
    _load_static,
    build_system,
    build_system_dict,
    environment,
)


class TestEnvironment:
    def test_includes_model_id(self) -> None:
        assert "test-model" in environment("test-model")

    def test_includes_cwd(self) -> None:
        assert "working directory" in environment("m").lower()

    def test_includes_platform(self) -> None:
        assert "Platform" in environment("m")

    def test_includes_shell(self) -> None:
        assert "Shell" in environment("m")


class TestStaticPrompts:
    def test_has_security_warning(self) -> None:
        text = _load_static()
        assert "penetration testing" in text

    def test_has_doing_tasks(self) -> None:
        text = _load_static()
        assert "# Doing tasks" in text

    def test_has_tools(self) -> None:
        text = _load_static()
        assert "Using your tools" in text

    def test_has_text_output_section(self) -> None:
        text = _load_static()
        assert "# Text output" in text

    def test_has_function_result_clearing(self) -> None:
        text = _load_static()
        assert "Function Result Clearing" in text
        assert "{keep_recent}" not in text  # placeholder substituted

    def test_has_misconception_overlay(self) -> None:
        text = _load_static()
        assert "collaborative" in text

    def test_has_false_claims_mitigation(self) -> None:
        text = _load_static()
        assert "truthful status" in text


class TestBuildSystem:
    def test_includes_environment(self) -> None:
        assert "test-model" in build_system("test-model")

    def test_custom_instructions(self) -> None:
        result = build_system("m", custom="Focus on Python.")
        assert "Focus on Python." in result
        assert "# User instructions" in result

    def test_no_custom(self) -> None:
        assert "# User instructions" not in build_system("m")


class TestBuildSystemDict:
    def test_returns_dict(self) -> None:
        d = build_system_dict("m")
        assert "static" in d
        assert "environment" in d

    def test_environment_is_callable(self) -> None:
        d = build_system_dict("test-model")
        env_fn = d["environment"]
        assert callable(env_fn)
        fn = cast(Callable[[], str], env_fn)  # pyright: ignore[reportUnnecessaryCast] -- ty needs cast; pyright narrows from callable()
        env_text = fn()
        assert isinstance(env_text, str)
        assert "test-model" in env_text

    def test_custom_instructions(self) -> None:
        d = build_system_dict("m", custom="Do X.")
        assert "user_instructions" in d
        val = d["user_instructions"]
        assert isinstance(val, str)
        assert "Do X." in val
        assert "# User instructions" in val

    def test_no_custom(self) -> None:
        d = build_system_dict("m")
        assert "user_instructions" not in d

    def test_advisor_section_not_in_scaffold(self) -> None:
        # Advisor now contributes via ``Advisor.prompt_section()``, not
        # via the core scaffold. ``build_system_dict`` stays feature-agnostic.
        d = build_system_dict("m")
        assert "advisor" not in d

    def test_can_disable_memory_section(self) -> None:
        d = build_system_dict("m", include_memory=False)
        assert "memory" not in d


class TestIsGitRepoError:
    def test_oserror_returns_false(self) -> None:
        _is_git_repo.cache_clear()
        with patch(
            "sagent.prompt.subprocess.run",
            side_effect=OSError("no git"),
        ):
            assert not _is_git_repo("/nonexistent/dir")
        _is_git_repo.cache_clear()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
