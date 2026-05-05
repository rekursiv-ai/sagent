# pytest fixtures are referenced by parameter name, not imported
"""Pytest configuration for sagent tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Explicitly set the anyio backend to asyncio."""
    return str(request.param)


def pytest_configure(config: pytest.Config) -> None:
    """Register the real_sleep marker."""
    config.addinivalue_line(
        "markers",
        "real_sleep: don't patch asyncio.sleep for this test",
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
