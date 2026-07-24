# pytest fixtures are referenced by parameter name, not imported
"""Pytest configuration for sagent tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import os

import pytest

from sagent.agent.state import (
    agent_registry,
    fresh_default_tool_state,
)
from sagent.tools.agent_spawn import _persistent_tasks


# In the OSS export the flattened top-level ``sagent/types`` masks stdlib
# ``types``, crashing xdist workers.
os.environ.setdefault("PYTHONSAFEPATH", "1")


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Explicitly set the anyio backend to asyncio."""
    return str(request.param)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers.

    Registered here (not only in ``.export/pyproject.toml``) so they resolve in
    every context the conftest is exported into: the public export runs with
    ``filterwarnings = ["error"]``, which turns an unregistered-mark warning into
    a collection-time error even for a deselected ``integration`` test.
    """
    config.addinivalue_line(
        "markers",
        "real_sleep: don't patch asyncio.sleep for this test",
    )
    config.addinivalue_line(
        "markers",
        "real_llm: spawns a real model CLI/binary; runs in a low-parallelism gate",
    )


@pytest.fixture(autouse=True)
def _fast_sleep(request: pytest.FixtureRequest) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    """Replace asyncio.sleep with a no-op so retry tests run instantly.

    Opt out on tests that depend on real sleep semantics (timing,
    watchdogs, yield ordering) via ``@pytest.mark.real_sleep``.
    """
    node = cast(pytest.Item, request.node)
    if node.get_closest_marker("real_sleep") is not None:
        yield
        return

    async def _instant(_: float) -> None:
        pass

    with patch("asyncio.sleep", _instant):
        yield


@pytest.fixture(autouse=True)
def _isolate_default_tool_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    """Reset the fallback ``ToolState`` to a fresh instance per test.

    ``get_tool_state()`` returns the module-level fallback when no
    ``tool_state_context`` is active. Tests (and tools-under-test) that
    call it outside a context mutate the singleton; state then bleeds
    into unrelated tests as flaky failures.
    """
    with fresh_default_tool_state():
        yield


@pytest.fixture(autouse=True)
def _isolate_agent_registry() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction] -- pytest fixture used via decorator
    """Snapshot and restore the process-global agent registries per test.

    ``agent_registry`` and ``_persistent_tasks`` are module-level dicts that
    tools and tests mutate directly. Under xdist a single worker runs many
    test files in one process, so an entry one test forgets to pop (e.g. a
    spawned ``fix-tools`` agent) leaks into a later test's view and flips
    unrelated assertions. Restoring both dicts to their pre-test contents
    makes that leakage structurally impossible -- one rule for every test.
    """
    registry_snapshot = dict(agent_registry)
    tasks_snapshot = dict(_persistent_tasks)
    try:
        yield
    finally:
        agent_registry.clear()
        agent_registry.update(registry_snapshot)
        _persistent_tasks.clear()
        _persistent_tasks.update(tasks_snapshot)
