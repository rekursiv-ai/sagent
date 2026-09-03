"""Shared pytest resource-marker rollups and timeout budgets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, cast

import os

import pytest


class MarkedItem(Protocol):
    """The marker surface the rollup reads and writes on a collected test.

    Narrower than ``pytest.Item`` on purpose: this is the whole contract, so a
    caller holding anything marker-shaped satisfies it.
    """

    def iter_markers(self, name: str | None = ...) -> Iterator[pytest.Mark]: ...

    def get_closest_marker(self, name: str) -> pytest.Mark | None: ...

    def add_marker(
        self, marker: str | pytest.MarkDecorator, *, append: bool = ...
    ) -> None: ...


def resource_marker_family(
    marker: str,
    *,
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> str:
    """Return the selector family encoded by a resource marker prefix."""
    family, separator, _specific = marker.partition("_")
    assert separator
    assert family in resource_families
    return family


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Derive timeouts and skips from concrete resource markers."""
    apply_resource_markers(
        items,
        resource_markers=registered_resource_markers(config),
    )


def registered_resource_markers(
    config: pytest.Config,
    *,
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> tuple[str, ...]:
    """Return registered concrete resource markers from pytest config."""
    configured = cast(list[str], config.getini("markers"))
    marker_names = tuple(marker.partition(":")[0] for marker in configured)
    return tuple(
        marker
        for marker in marker_names
        if marker.partition("_")[0] in resource_families and "_" in marker
    )


def resource_marker_aliases(
    marker: str,
    *,
    aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("bench_compile_time", ("performance",)),
        ("bench_memory", ("performance",)),
        ("bench_statistical", ("performance",)),
        ("bench_throughput", ("performance",)),
        ("bench_wallclock", ("performance",)),
        ("browser_chrome", ("integration",)),
        ("browser_zendriver", ("integration",)),
        ("cli_bash", ("slow",)),
        ("cli_claude", ("integration", "real_llm")),
        ("cli_codex", ("integration", "real_llm")),
        ("cli_docker", ("integration",)),
        ("cli_git", ("integration",)),
        ("cli_node", ("integration",)),
        ("cli_precommit", ("ci_smoke",)),
        ("cli_python_subprocess", ("integration",)),
        ("cli_rsync", ("integration",)),
        ("cli_ssh", ("integration",)),
        ("cli_uv", ("ci_smoke",)),
        ("compute_distributed", ("slow",)),
        ("compute_jax_jit", ("slow",)),
        ("compute_large_fixture", ("slow",)),
        ("compute_torch_compile", ("slow",)),
        ("compute_training", ("slow",)),
        ("db_pglite", ("integration",)),
        ("db_pgvector", ("integration",)),
        ("db_postgres", ("integration",)),
        ("gpu_cuda_runtime", ("cuda",)),
        ("gpu_flash_attention", ("cuda",)),
        ("gpu_jax_cuda", ("cuda",)),
        ("gpu_nvidia", ("cuda",)),
        ("gpu_torch_cuda", ("cuda",)),
        ("gpu_torch_mps", ("integration",)),
        ("gpu_triton", ("cuda",)),
        ("network_anthropic", ("integration",)),
        ("network_duckduckgo", ("integration",)),
        ("network_gemini", ("integration",)),
        ("network_google_search", ("integration",)),
        ("network_github", ("integration",)),
        ("network_huggingface", ("integration",)),
        ("network_kaggle", ("integration",)),
        ("network_localhost", ("integration",)),
        ("network_openai", ("integration",)),
        ("network_openml", ("integration",)),
        ("network_openreview", ("integration",)),
        ("network_pypi", ("integration",)),
        ("network_searxng", ("integration",)),
        ("network_shadeform", ("cluster",)),
        ("network_slack", ("integration",)),
        ("network_together", ("cluster",)),
        ("network_wandb", ("integration",)),
    ),
) -> tuple[str, ...]:
    """Return legacy selector marks for a concrete resource marker."""
    for candidate, candidate_aliases in aliases:
        if marker == candidate:
            return candidate_aliases
    # UsageError, not ValueError: this is reached from a collection hook, and
    # pytest renders anything else as an INTERNALERROR traceback that buries the
    # marker name.
    raise pytest.UsageError(f"Unknown resource marker: {marker}")


def resource_marker_timeout(
    marker: str,
    *,
    category_timeouts: tuple[tuple[str, int], ...] = (
        ("bench", 600),
        ("browser", 180),
        ("cli", 120),
        ("compute", 300),
        ("db", 120),
        ("gpu", 300),
        ("network", 180),
    ),
    specific_timeouts: tuple[tuple[str, int], ...] = (
        ("bench_throughput", 600),
        ("cli_claude", 1800),
        ("cli_codex", 1800),
        ("cli_docker", 300),
        ("cli_precommit", 300),
        # A spawned interpreter re-imports the tree it collects, which the
        # 120s `cli` default does not cover: marker_tiers_test's collection
        # measured 69s on a developer box and timed out 108 times on a loaded
        # 2-vCPU CI runner. A timeout is a ceiling, not a schedule, so this
        # cannot slow a subprocess test that already finishes quickly.
        ("cli_python_subprocess", 300),
        ("cli_rsync", 180),
        ("cli_ssh", 300),
        ("cli_uv", 180),
        ("compute_large_fixture", 180),
        ("compute_torch_compile", 900),
        ("db_pglite", 180),
        ("network_shadeform", 300),
        ("network_together", 4800),
    ),
) -> int:
    """Return a marker's specific timeout, falling back to its category."""
    specific = dict(specific_timeouts)
    if marker in specific:
        return specific[marker]
    return dict(category_timeouts)[resource_marker_family(marker)]


def apply_resource_markers(
    items: Sequence[MarkedItem],
    *,
    resource_markers: tuple[str, ...],
    ci_skipped_marks: tuple[str, ...] = (
        "cluster",
        "cuda",
        "performance",
        "bench_compile_time",
        "bench_memory",
        "bench_statistical",
        "bench_throughput",
        "bench_wallclock",
        "gpu_cuda_runtime",
        "gpu_flash_attention",
        "gpu_jax_cuda",
        "gpu_nvidia",
        "gpu_torch_cuda",
        "gpu_triton",
        "network_shadeform",
        "network_together",
    ),
    live_llm_marks: tuple[str, ...] = ("cli_claude", "cli_codex", "real_llm"),
    live_llm_env_var: str = "RUN_REAL_LLM",
) -> None:
    """Apply virtual family markers, timeout budgets, and skip policy."""
    known_resources = set(resource_markers)
    for item in items:
        # One marker walk per item: ``get_closest_marker`` re-walks the whole
        # parent chain per name, so asking it per registered marker costs a walk
        # per marker per item on every repo-wide collection.
        existing = {marker.name for marker in item.iter_markers()}
        _fail_on_unknown_resource_markers(existing, resource_markers=known_resources)
        carried = existing & known_resources
        resource_timeouts = [resource_marker_timeout(marker) for marker in carried]
        resource_aliases = {
            alias for marker in carried for alias in resource_marker_aliases(marker)
        }
        # A root conftest and a package conftest both bind this hook, so an item
        # is walked once per binding. EVERY mark added below is therefore guarded
        # by what the item already carries -- one rule, rather than a per-branch
        # check that the next branch forgets.
        for alias in sorted(resource_aliases - existing):
            alias_marker = cast(pytest.MarkDecorator, getattr(pytest.mark, alias))
            item.add_marker(alias_marker)
        if resource_timeouts and "timeout" not in existing:
            item.add_marker(pytest.mark.timeout(max(resource_timeouts)))
        if "skip" not in existing:
            _apply_skip_policy(
                item,
                names=existing,
                live_llm_marks=live_llm_marks,
                live_llm_env_var=live_llm_env_var,
                ci_skipped_marks=ci_skipped_marks,
            )


def _apply_skip_policy(
    item: MarkedItem,
    *,
    names: set[str],
    live_llm_marks: tuple[str, ...],
    live_llm_env_var: str,
    ci_skipped_marks: tuple[str, ...],
) -> None:
    """Skip an item whose resource is unavailable in this environment."""
    for mark in live_llm_marks:
        if mark in names and not os.environ.get(live_llm_env_var):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"{mark} test skipped"
                        f" (set {live_llm_env_var}=1 to run live model CLIs)"
                    )
                )
            )
            return
    if os.environ.get("CI") and not os.environ.get("RUN_INTEGRATION"):
        for mark in ci_skipped_marks:
            if mark in names:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{mark} test skipped in CI"
                            " (no live credentials/services/devices;"
                            " set RUN_INTEGRATION=1 to opt in)"
                        )
                    )
                )
                return


def _fail_on_unknown_resource_markers(
    names: set[str],
    *,
    resource_markers: set[str],
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> None:
    """Fail collection when a resource-prefixed marker is not registered."""
    for name in names - resource_markers:
        family, separator, _specific = name.partition("_")
        if separator and family in resource_families:
            raise pytest.UsageError(f"Unknown resource marker: {name}")
