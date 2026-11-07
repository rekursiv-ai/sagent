"""The Hypothesis budgets every test run loads, in the monorepo and in each export.

A root conftest calls :func:`configure_hypothesis` once at import, before any test
module draws an example, so the monorepo and every exported package search the
same way: deterministic, with no timing checks a shared runner can trip.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, configuration

import hypothesis

from sagent.lib.userdirs import cache_dir


__all__ = ["configure_hypothesis", "hypothesis_profile_name"]


# Never mutated, nor is configure_hypothesis: every root conftest calls both at
# import, so mutmut's forced-fail run raises while loading the conftest, which it
# rejects as a bad test command.
def hypothesis_profile_name() -> str:  # pragma: no mutate block
    """Return the explicit profile, or choose dev/ci from the environment."""
    return os.environ.get("HYPOTHESIS_PROFILE") or (
        "ci" if os.environ.get("CI", "") else "dev"
    )


def configure_hypothesis() -> None:  # pragma: no mutate block
    """Register the property-testing budgets and load the one for this run."""
    # Hypothesis defaults to .hypothesis/ in each checkout.
    configuration.set_hypothesis_home_dir(
        str(cache_dir() / "rekursiv-ai" / "hypothesis"),
    )
    # Timing checks flake under shared-runner contention: deadlines, and too_slow,
    # which a cold run's first text draw (building the charmap) trips under coverage.
    # Repeat inputs across runs so local and CI failures can be reproduced.
    hypothesis.settings.register_profile(
        "dev",
        deadline=None,
        derandomize=True,
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
    )
    hypothesis.settings.register_profile(
        "ci",
        deadline=None,
        derandomize=True,
        max_examples=200,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow],
    )
    hypothesis.settings.load_profile(hypothesis_profile_name())
