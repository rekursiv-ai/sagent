"""The whole-suite pytest tiers the pre-push hooks run.

Standard library only: CI derives its job matrix from this table with a bare
``pre-commit`` tool environment, before the project's dependencies exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True, kw_only=True)
class Tier:
    """One whole-suite pytest tier, as a pre-push hook runs it.

    Attributes:
      name: The hook id; also names the ``WORKERS_<NAME>`` override.
      markers: Markers a test must carry, every one, to belong to the tier.
      excluded: Markers a test must not carry.
      allow_empty: Pass when nothing is collected (exit 5): a tree may ship no
        test for this tier.
      ci_testmon: Select with testmon in GitHub Actions too, not only locally.
      requires: Executable that must be on PATH, else the tier passes; a
        capability, not a switch.
      extra: Further pytest arguments.

    """

    name: str
    markers: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    allow_empty: bool = False
    ci_testmon: bool = False
    requires: str | None = None
    extra: tuple[str, ...] = ()

    @property
    def marker(self) -> str:
        """Return the ``-m`` expression that selects this tier's tests."""
        return " and ".join([*self.markers, *(f"not {name}" for name in self.excluded)])

    def selects(self, names: set[str] | frozenset[str]) -> bool:
        """Return whether a test carrying markers ``names`` belongs to this tier."""
        return set(self.markers) <= names and not names & set(self.excluded)


# The hooks select by `marker` and the node-id plugin partitions by `selects`,
# both from this one table, so a tier cannot be run one way and split another.
TIERS: Final = {
    tier.name: tier
    for tier in (
        # Fails on an empty collection: the whole suite collecting nothing is a
        # broken tree, never an empty tier.
        Tier(
            name="pytest-global",
            excluded=(
                "ci_smoke",
                "cluster",
                "cuda",
                "integration",
                "performance",
                "slow",
            ),
            ci_testmon=True,
        ),
        Tier(
            name="pytest-slow-global",
            markers=("slow",),
            excluded=("cluster", "cuda", "integration", "performance"),
            allow_empty=True,
        ),
        Tier(
            name="pytest-ci-smoke-global",
            markers=("ci_smoke",),
            allow_empty=True,
            extra=("--durations=20",),
        ),
        # Needs a real Postgres: skips where `pg_config` is absent and RUNS
        # where it is present, so a provisioned pgvector is never left unused.
        Tier(
            name="pytest-integration-global",
            markers=("integration",),
            allow_empty=True,
            requires="pg_config",
            extra=("--dist=loadgroup",),
        ),
    )
}
