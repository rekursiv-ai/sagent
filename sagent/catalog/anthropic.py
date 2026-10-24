"""Anthropic transport restrictions and Messages API usage accounting.

Load model facts with ``sagent.catalog.models("anthropic")``.
Transport capabilities narrow those rows with ``&``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

from treekle import from_plain

from sagent.types.capability import ModelCapability, ThinkingCapability
from sagent.types.cost import TokenCount


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "CACHE_TTL_SEC",
    "api",
    "cli",
    "subscription",
    "usage_tokens",
]


# No ``0``: these transports write a breakpoint on every request, so "do not
# cache" is not a selection this vendor offers.
CACHE_TTL_SEC: Final = frozenset({300.0, 3600.0})


def api() -> ModelCapability:
    """Return what the Messages API adds on top of a model row.

    One model, three ways in::

        from sagent.catalog import models

        models("anthropic")["opus-5.0"] & api()          # API key: full tiers, cache
        models("anthropic")["opus-5.0"] & subscription() # same wire, OAuth-billed
        models("anthropic")["opus-5.0"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        cache_ttl_sec=CACHE_TTL_SEC,
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
        # No ``flex``: the Messages API takes only ``auto`` / ``standard_only``
        # (https://platform.claude.com/docs/en/api/messages/create), which map to
        # ``auto`` / ``default`` here. ``priority`` is the fast-mode beta.
        service_tier=frozenset({"auto", "default", "priority"}),
    )


def cli() -> ModelCapability:
    """Return what the ``claude`` subprocess narrows a model row to.

    One model, three ways in::

        from sagent.catalog import models

        models("anthropic")["opus-5.0"] & api()          # API key: full tiers, cache
        models("anthropic")["opus-5.0"] & subscription() # same wire, OAuth-billed
        models("anthropic")["opus-5.0"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        from sagent.catalog import models

        models("anthropic")["opus-5.0"] & api()          # API key: full tiers, cache
        models("anthropic")["opus-5.0"] & subscription() # same wire, OAuth-billed
        models("anthropic")["opus-5.0"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The subscription transport's restrictions.

    """
    return replace(api(), account_auth=True)


def usage_tokens(usage: Mapping[str, object], *, cache_ttl_sec: float) -> TokenCount:
    """Read one Messages-API ``usage`` block into disjoint meters.

    ``cache_creation`` splits the write by lifetime. A block without it (an
    older transcript) is attributed wholly to ``cache_ttl_sec``, the lifetime
    the request asked for.

    Args:
      usage: The response's ``usage`` object, as a mapping.
      cache_ttl_sec: Lifetime the request's cache breakpoints asked for.

    Returns:
      tokens: The request's usage.

    """
    written = from_plain(usage.get("cache_creation_input_tokens"), int, default=0)
    split = from_plain(usage.get("cache_creation"), dict[str, object], default={})
    if split:
        written_1h = from_plain(split.get("ephemeral_1h_input_tokens"), int, default=0)
    else:
        written_1h = written if cache_ttl_sec >= 3600.0 else 0
    return TokenCount(
        request=from_plain(usage.get("input_tokens"), int, default=0),
        response=from_plain(usage.get("output_tokens"), int, default=0),
        cache_write=written - written_1h,
        cache_write_1h=written_1h,
        cache_read=from_plain(usage.get("cache_read_input_tokens"), int, default=0),
    )
