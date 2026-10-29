"""What the catalog knows of a provider beyond how to build it.

Typical usage example:

  Listing.of(Anthropic)
  Listing.signing_in(AnthropicCLI, default=ClaudeLogin.default_path)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sagent.providers.types import ProviderFactory


if TYPE_CHECKING:
    from sagent.providers.types import Provider, ProviderSharedResources


__all__ = [
    "Listing",
]


class _Sectioned(Protocol):
    """A provider class: it builds a provider, under its catalog section's name."""

    @property
    def section(self) -> str:
        """Return the catalog section it serves."""
        ...

    def __call__(
        self,
        *,
        name: str,
        resources: ProviderSharedResources,
    ) -> Provider:
        """Return a provider the catalog knows as ``name``."""
        ...


class _SigningIn(Protocol):
    """A provider class built with the login file it signs in with."""

    @property
    def section(self) -> str:
        """Return the catalog section it serves."""
        ...

    def __call__(
        self,
        *,
        name: str,
        resources: ProviderSharedResources,
        credentials: Path | None = None,
    ) -> Provider:
        """Return a provider the catalog knows as ``name``, signed in with ``credentials``."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Listing:
    """A provider the catalog offers, under its section's name."""

    section: str
    """The name it is offered under."""

    factory: ProviderFactory
    """How it is built, signed in with its default login if it signs in."""

    default_login: Callable[[], Path] | None = None
    """Where its default login is kept; ``None`` for one that signs in as no one."""

    signed_in: Callable[[Path], ProviderFactory] | None = None
    """How it is built signed in with a given login file."""

    former_names: tuple[str, ...] = ()
    """What it was once called, which it still answers to."""

    preferred: bool = False
    """Whether a session starts on it, unless told otherwise."""

    @classmethod
    def of(cls, provider: _Sectioned) -> Listing:
        """Return the listing of ``provider``, which signs in as no one."""
        return cls(section=provider.section, factory=provider)

    @classmethod
    def signing_in(
        cls,
        provider: _SigningIn,
        *,
        default: Callable[[], Path],
        former_names: tuple[str, ...] = (),
        preferred: bool = False,
    ) -> Listing:
        """Return the listing of ``provider``, which signs in.

        Args:
          provider: The provider class, built with a ``credentials`` file.
          default: Returns where its default login is kept.
          former_names: What it was once called.
          preferred: Whether a session starts on it.

        Returns:
          listing: Built with its default login, or signed in with a file given.

        """
        return cls(
            section=provider.section,
            factory=provider,
            default_login=default,
            signed_in=lambda credentials: partial(provider, credentials=credentials),
            former_names=former_names,
            preferred=preferred,
        )
