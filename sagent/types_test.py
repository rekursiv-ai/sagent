"""Tests for ``AsyncResource``: ``async with`` opens, then closes."""

from __future__ import annotations

from typing import assert_type, override

import contextlib

import pytest

from sagent.types import AsyncResource, Resource


class _Logged(AsyncResource):
    def __init__(self) -> None:
        self.log: list[str] = []

    @override
    async def aopen(self) -> None:
        self.log.append("open")

    @override
    async def aclose(self) -> None:
        self.log.append("close")


@pytest.mark.asyncio
async def test_async_with_opens_yields_itself_and_closes() -> None:
    resource = _Logged()
    async with resource as entered:
        assert entered is resource
        assert resource.log == ["open"]
    assert resource.log == ["open", "close"]


@pytest.mark.asyncio
async def test_a_failure_inside_still_closes_and_propagates() -> None:
    resource = _Logged()
    with pytest.raises(RuntimeError, match="inside"):
        async with resource:
            raise RuntimeError("inside")
    assert resource.log == ["open", "close"]


@pytest.mark.asyncio
async def test_the_base_acquires_and_releases_nothing() -> None:
    resource: Resource = AsyncResource()
    async with resource as entered:
        assert entered is resource
    await resource.aopen()
    await resource.aclose()


def test_the_stdlib_still_knows_it_for_an_async_context_manager() -> None:
    assert isinstance(_Logged(), contextlib.AbstractAsyncContextManager)


@pytest.mark.asyncio
async def test_a_stack_gives_back_the_resource_as_its_own_type() -> None:
    """The checkers read ``enter_async_context``'s result off the base class."""
    resource = _Logged()
    async with contextlib.AsyncExitStack() as stack:
        entered = await stack.enter_async_context(resource)
        assert_type(entered, _Logged)
        assert entered.log == ["open"]
    assert resource.log == ["open", "close"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
