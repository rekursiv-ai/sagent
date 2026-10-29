"""Providers, which serve models, and what they share.

Typical usage example:

  async with ModelCatalog() as catalog:            # shared HTTP, processes, tools
      gpt = catalog["openai-sub"]["gpt"]           # looking up does no I/O
      catalog["openai-sub"].usage                  # the account's rate limits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sagent.types import Resource


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    import httpx2

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request, Response, TextChunk
    from sagent.session.types import AnyToolResult, Attachment, ToolCall
    from sagent.tools.types import Tool


__all__ = [
    "Login",
    "Model",
    "ModelCatalog",
    "ProcessPool",
    "Provider",
    "ProviderFactory",
    "ProviderSharedResources",
    "ToolMount",
    "ToolServer",
    "UsageWindow",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageWindow:
    """How much of one rate-limit window an account has used."""

    label: str
    """The window's span, as the vendor names it (``5h``, ``7d``)."""

    utilization: float | None = None
    """Fraction used, from 0 to 1; ``None`` when the vendor did not say."""

    resets_at: float | None = None
    """When the window starts over, in seconds since the epoch."""

    blocked: bool = False


class ProcessPool(Resource, Protocol):
    """Running processes, each started before it is asked for."""

    async def acquire(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
    ) -> Process:
        """Return a process running ``argv`` in ``env``, started ahead if one was."""
        ...

    async def release(self, process: Process) -> None:
        """Stop ``process``."""
        ...


class ToolMount(Protocol):
    """Tools offered to one CLI, whose calls wait for the caller's results."""

    @property
    def url(self) -> str:
        """Return the MCP endpoint the CLI connects to."""
        ...

    def offer(self, tools: Sequence[Tool]) -> None:
        """Offer ``tools``, replacing those offered before."""
        ...

    async def called(self) -> ToolCall:
        """Wait for the next call a CLI makes here, and return it as the model made it."""
        ...

    async def answer(self, call: ToolCall, result: AnyToolResult) -> None:
        """Complete the waiting call that ``call`` describes with ``result``."""
        ...

    def cancel(self) -> None:
        """Fail every waiting call."""
        ...


class ToolServer(Resource, Protocol):
    """One MCP server, shared by every CLI process."""

    async def mount(self) -> ToolMount:
        """Return a new endpoint with no tools offered yet."""
        ...

    async def unmount(self, mount: ToolMount) -> None:
        """Remove ``mount``, failing its waiting calls."""
        ...


class ProviderSharedResources(Resource, Protocol):
    """What every provider in a catalog shares; each opens on first use."""

    @property
    def http(self) -> httpx2.AsyncClient:
        """Return the HTTP client."""
        ...

    def process_pool(self, name: str) -> ProcessPool:
        """Return the pool of processes named ``name``."""
        ...

    @property
    def tool_server(self) -> ToolServer:
        """Return the MCP server."""
        ...


class ProviderFactory(Protocol):
    """Builds a provider, as a provider class's constructor does."""

    def __call__(
        self,
        *,
        name: str,
        resources: ProviderSharedResources,
    ) -> Provider:
        """Return a provider the catalog knows as ``name``, sharing ``resources``."""
        ...


class ModelCatalog(ProviderSharedResources, Protocol):
    """Every provider, by name, sharing one set of resources."""

    factories: Mapping[str, ProviderFactory]

    @property
    def account(self) -> str | None:
        """Return the account providers sign in as; ``None`` for each one's default."""
        ...

    def __getitem__(self, provider: str) -> Provider:
        """Return the provider named ``provider``, built on first use."""
        ...

    def name(self, provider: str) -> str:
        """Return the name of the provider ``provider`` spells, short or whole."""
        ...

    def model(self, name: str, *, current: str) -> Model:
        """Return ``[PROVIDER/]MODEL``: from the provider named, else whoever serves it."""
        ...

    def signed_in(self, provider: str, *, account: str | None) -> Provider:
        """Return ``provider`` signed in as ``account``, the catalog's own account kept."""
        ...

    def use_account(self, account: str | None) -> None:
        """Sign providers in as ``account`` from now on, offering the same names."""
        ...


class Provider(Resource, Protocol):
    """Serves a vendor's models, billed to one account."""

    @property
    def name(self) -> str:
        """Return the name the catalog knows this provider by."""
        ...

    @property
    def resources(self) -> ProviderSharedResources:
        """Return what this provider shares with others."""
        ...

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return the account's rate-limit windows, as last reported."""
        ...

    def __getitem__(self, model: str) -> Model:
        """Return the model ``model`` names; this does no I/O."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Yield the exact name of each model served."""
        ...


class Model(Resource, Protocol):
    """One model, as served by one provider."""

    @property
    def provider(self) -> Provider:
        """Return the provider serving this model."""
        ...

    @property
    def capability(self) -> ModelCapability:
        """Return every setting this model offers."""
        ...

    @property
    def settings(self) -> ModelSettings:
        """Return the settings chosen."""
        ...

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        """Choose ``settings``; raise unless the capability offers each choice."""
        ...

    async def count(self, request: Request) -> int:
        """Return how many tokens ``request`` would send, exact where the vendor says."""
        ...

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return about how many tokens ``attachment`` adds, by the vendor's formula."""
        ...

    def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply as it is written, then the finished ``Response``."""
        ...


@runtime_checkable
class Login(Protocol):
    """A provider that signs in to an account."""

    async def login(self, *, headless: bool = False) -> None:
        """Sign in, interactively; ``headless`` skips the browser and takes a pasted code."""
        ...
