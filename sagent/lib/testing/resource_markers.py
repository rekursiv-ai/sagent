"""Shared pytest resource-marker rollups and timeout budgets."""

from __future__ import annotations

from typing import cast

import os

import pytest


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
        ("gpu_triton", ("cuda",)),
        ("network_anthropic", ("integration",)),
        ("network_gemini", ("integration",)),
        ("network_github", ("integration",)),
        ("network_huggingface", ("integration",)),
        ("network_kaggle", ("integration",)),
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
    raise ValueError(f"Unknown resource marker: {marker}")


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
        ("cli_claude", 300),
        ("cli_codex", 300),
        ("cli_docker", 300),
        ("cli_precommit", 300),
        ("cli_rsync", 180),
        ("cli_ssh", 300),
        ("cli_uv", 180),
        ("compute_large_fixture", 180),
        ("network_openml", 60),
        ("network_shadeform", 300),
        ("network_together", 300),
    ),
) -> int:
    """Return a marker's specific timeout, falling back to its category."""
    specific = dict(specific_timeouts)
    if marker in specific:
        return specific[marker]
    return dict(category_timeouts)[resource_marker_family(marker)]


def apply_resource_markers(
    items: list[pytest.Item],
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
    for item in items:
        _fail_on_unknown_resource_markers(item, resource_markers=resource_markers)
        resource_timeouts: list[int] = []
        resource_aliases: set[str] = set()
        for marker in resource_markers:
            if item.get_closest_marker(marker) is None:
                continue
            resource_timeouts.append(resource_marker_timeout(marker))
            resource_aliases.update(resource_marker_aliases(marker))
        if resource_aliases:
            for alias in sorted(resource_aliases):
                alias_marker = cast(pytest.MarkDecorator, getattr(pytest.mark, alias))
                item.add_marker(alias_marker)
        if resource_timeouts and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(max(resource_timeouts)))
        for mark in live_llm_marks:
            if item.get_closest_marker(mark) is not None and not os.environ.get(
                live_llm_env_var
            ):
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{mark} test skipped"
                            f" (set {live_llm_env_var}=1 to run live model CLIs)"
                        )
                    )
                )
                break
        if os.environ.get("CI") and not os.environ.get("RUN_INTEGRATION"):
            for mark in ci_skipped_marks:
                if item.get_closest_marker(mark) is not None:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"{mark} test skipped in CI"
                                " (no live credentials/services/devices;"
                                " set RUN_INTEGRATION=1 to opt in)"
                            )
                        )
                    )
                    break


def _fail_on_unknown_resource_markers(
    item: pytest.Item,
    *,
    resource_markers: tuple[str, ...],
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
    known = set(resource_markers)
    for marker in item.iter_markers():
        family, separator, _specific = marker.name.partition("_")
        if separator and family in resource_families and marker.name not in known:
            raise ValueError(f"Unknown resource marker: {marker.name}")
