"""Tests for the Hypothesis budgets every test run loads."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from hypothesis import HealthCheck, configuration

import hypothesis
import pytest

from sagent.lib.testing.hypothesis_profiles import (
    configure_hypothesis,
    hypothesis_profile_name,
)
from sagent.lib.userdirs import cache_dir


if TYPE_CHECKING:
    from collections.abc import Iterator


class _Profile(NamedTuple):
    """Typed settings from a Hypothesis profile."""

    max_examples: int
    derandomize: bool
    deadline: object
    print_blob: bool
    suppressed: tuple[HealthCheck, ...]


def _profile(name: str) -> _Profile:
    registered = hypothesis.settings.get_profile(name)
    return _Profile(
        max_examples=registered.max_examples,
        derandomize=registered.derandomize,
        deadline=registered.deadline,
        print_blob=registered.print_blob,
        suppressed=tuple(registered.suppress_health_check),
    )


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure afresh under ``dev``, then restore the run's own home and profile."""
    home = configuration.storage_directory().home_directory
    loaded = hypothesis_profile_name()
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "dev")
    configure_hypothesis()
    try:
        yield
    finally:
        configuration.set_hypothesis_home_dir(str(home))
        hypothesis.settings.load_profile(loaded)


@pytest.mark.usefixtures("configured")
class TestConfigureHypothesis:
    def test_the_default_profile_is_deterministic(self) -> None:
        profile = _profile("dev")

        assert (profile.derandomize, profile.deadline) == (True, None)
        assert hypothesis.settings().max_examples == profile.max_examples

    def test_ci_searches_harder_and_prints_a_reproduction_blob(self) -> None:
        assert (_profile("dev").max_examples, _profile("ci").max_examples) == (20, 200)
        assert (_profile("dev").print_blob, _profile("ci").print_blob) == (False, True)

    def test_no_profile_times_a_test_or_its_input_generation(self) -> None:
        for name in ("dev", "ci"):
            profile = _profile(name)
            assert (profile.deadline, profile.suppressed) == (
                None,
                (HealthCheck.too_slow,),
            ), name

    def test_the_example_database_lives_in_the_users_cache(self) -> None:
        assert configuration.storage_directory().home_directory == Path(
            cache_dir() / "rekursiv-ai" / "hypothesis",
        )


class TestHypothesisProfileName:
    def test_the_loaded_profile_follows_the_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("HYPOTHESIS_PROFILE", raising=False)
        monkeypatch.setenv("CI", "1")
        assert hypothesis_profile_name() == "ci"

        monkeypatch.delenv("CI", raising=False)
        assert hypothesis_profile_name() == "dev"

    def test_an_explicit_override_beats_the_ci_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CI", "1")
        monkeypatch.setenv("HYPOTHESIS_PROFILE", "dev")

        assert hypothesis_profile_name() == "dev"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
