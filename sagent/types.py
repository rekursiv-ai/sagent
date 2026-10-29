"""What everything else builds on: resources that open and close.

Typical usage example:

  class Pool(AsyncResource):
      async def aclose(self) -> None:
          ...  # Release what it holds.

  async with Pool() as pool:
      ...
"""

from __future__ import annotations

from typing import Protocol, Self


__all__ = [
    "AsyncResource",
    "Resource",
]


class Resource(Protocol):
    """Something opened before use and closed after, as by ``async with``."""

    async def aopen(self) -> None:
        """Acquire what this needs; ``async with`` calls it."""
        ...

    async def aclose(self) -> None:
        """Release what this holds; ``async with`` calls it."""
        ...

    async def __aenter__(self) -> Self:
        """Open by ``aopen``; return this."""
        ...

    async def __aexit__(self, *exc: object) -> None:
        """Close by ``aclose``, whatever ended the block."""
        ...


# Not a subclass of ``AbstractAsyncContextManager["AsyncResource", None]``: the
# checkers read ``enter_async_context``'s result off that parameter, so every
# resource entered on a stack came back typed as the base. The stdlib ABC still
# recognizes the class by its methods.
class AsyncResource:
    """A ``Resource`` to subclass: override ``aopen`` and ``aclose`` as needed."""

    async def aopen(self) -> None:
        """Acquire nothing; a subclass overrides this to acquire what it needs."""
        return

    async def aclose(self) -> None:
        """Release nothing; a subclass overrides this to release what it holds."""
        return

    async def __aenter__(self) -> Self:
        await self.aopen()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
