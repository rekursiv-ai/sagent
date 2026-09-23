"""Anthropic model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - Limits & pricing: https://docs.anthropic.com/en/docs/about-claude/models
  - Fast mode: https://docs.anthropic.com/en/docs/build-with-claude/fast-mode
  - Vision: https://platform.claude.com/docs/en/build-with-claude/vision
  - Request size: https://platform.claude.com/docs/en/api/overview

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )


__all__ = [
    "api",
    "cache_ttls",
    "cli",
    "models",
    "subscription",
]


def cache_ttls() -> frozenset[float]:
    """Return the two lifetimes ``cache_control`` spells, in seconds.

    No ``0``: these transports write a breakpoint on every request, so
    "do not cache" is not a selection this vendor offers.

    Returns:
      ttls: The two cache control lifetimes (300s and 3600s).

    """
    return frozenset({300.0, 3600.0})


# Thinking capability measured against the live API (Jun 2026). ``auto`` is
# ``thinking.type=adaptive`` (no budget); ``fixed`` is ``enabled`` +
# ``budget_tokens``. A model missing ``text`` reasons but returns a
# signed-but-empty block -- the plaintext is never delivered.
#
# Every row states ``none`` in each thinking set: a model that CAN think can
# also omit an explicit thinking request, and the axis is total, so omitting it
# would make the provider default unselectable.
def models() -> Mapping[str, ModelCapability]:
    """Return every Anthropic model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # The 5 generation, as a row: every model below is this one with its
    # differences replaced. ``_limits`` and ``_prices`` build the two composed
    # fields, which ``replace`` cannot reach into.
    five = ModelCapability(
        approx_chars_per_token=2.38,
        context=_limits(window=200_000, edge=2576),
        prices=_prices(request=10.0, response=50.0),
        thinking_effort={"none", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto"},
        thinking_output={"none", "redacted"},
    )
    # The 4-x generation differs from it in three ways at once: smaller images,
    # readable reasoning, and a fixed budget.
    four = replace(
        five,
        approx_chars_per_token=3.12,
        context=_limits(
            window=200_000,
            edge=1568,
            long_betas=frozenset({"context-1m-2025-08-07"}),
        ),
        prices=_prices(request=3.0, response=15.0),
        thinking_effort={"none"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
    )
    rows = (
        # Fable 5.1 and Mythos 5.1 bill cache hits at 0.025x base input, not
        # the 0.1x every other model uses -- $0.25/Mtok here, and the flat
        # 0.1x this catalog assumed charged 4x that.
        replace(
            five,
            model_id="fable-5.1",
            wire_model_id="claude-fable-5-1",
            knowledge_cutoff="June 2026",
            prices=_prices(request=10.0, response=50.0, cache_read_multiple=0.025),
        ),
        replace(
            five,
            model_id="fable-5",
            wire_model_id="claude-fable-5",
        ),
        replace(
            five,
            model_id="opus-5.5",
            wire_model_id="claude-opus-5-5",
            knowledge_cutoff="June 2026",
            prices=_prices(request=4.0, response=20.0, fast_multiple=2.0),
            service_tier={"auto", "default", "priority"},
        ),
        replace(
            five,
            model_id="opus-5",
            wire_model_id="claude-opus-5",
            knowledge_cutoff="May 2026",
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            service_tier={"auto", "default", "priority"},
        ),
        replace(
            five,
            model_id="opus-4.8",
            wire_model_id="claude-opus-4-8",
            knowledge_cutoff="January 2026",
            approx_chars_per_token=2.38,
            context=_limits(
                window=200_000,
                edge=2576,
                long_betas=frozenset({"context-1m-2025-08-07"}),
            ),
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            service_tier={"auto", "default", "priority"},
        ),
        # No fast price row on 4-7 or 4-6: 4-7 lost fast mode on 2026-07-24 and
        # the API now rejects ``speed="fast"``; 4-6 never shipped it and bills
        # standard, so a fast row would misprice both.
        replace(
            five,
            model_id="opus-4.7",
            wire_model_id="claude-opus-4-7",
            knowledge_cutoff="January 2026",
            approx_chars_per_token=2.38,
            context=_limits(
                window=200_000,
                edge=2576,
                long_betas=frozenset({"context-1m-2025-08-07"}),
            ),
            prices=_prices(request=5.0, response=25.0),
        ),
        replace(
            four,
            model_id="opus-4.6",
            wire_model_id="claude-opus-4-6",
            knowledge_cutoff="May 2025",
            prices=_prices(request=5.0, response=25.0),
            thinking_effort={"none", "low", "medium", "high", "max"},
        ),
        replace(
            four,
            model_id="opus-4.5",
            wire_model_id="claude-opus-4-5",
            knowledge_cutoff="May 2025",
            prices=_prices(request=5.0, response=25.0),
            thinking_effort={"none", "low", "medium", "high"},
            thinking_budget={"none", "fixed"},
        ),
        # $2/$10 was introductory pricing through 2026-08-31; the scheduled
        # rise to $3/$15 on 2026-09-01 was cancelled and this is now standard.
        replace(
            five,
            model_id="sonnet-5",
            wire_model_id="claude-sonnet-5",
            knowledge_cutoff="January 2026",
            prices=_prices(request=2.0, response=10.0),
        ),
        replace(
            four,
            model_id="sonnet-4.6",
            wire_model_id="claude-sonnet-4-6",
            knowledge_cutoff="August 2025",
            thinking_effort={"none", "low", "medium", "high", "max"},
        ),
        replace(
            four,
            model_id="sonnet-4.5",
            wire_model_id="claude-sonnet-4-5",
            knowledge_cutoff="January 2025",
            thinking_budget={"none", "fixed"},
        ),
        replace(
            four,
            model_id="haiku-4.5",
            wire_model_id="claude-haiku-4-5",
            knowledge_cutoff="February 2025",
            context=_limits(window=200_000, edge=1568, response=64_000, long=0),
            prices=_prices(request=1.0, response=5.0),
            thinking_budget={"none", "fixed"},
        ),
    )
    catalog = {row.model_id: row for row in rows}
    return MappingProxyType(
        {
            "default": catalog["fable-5.1"],
            "utility": catalog["haiku-4.5"],
            **catalog,
        },
    )


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts. ``&`` can only remove, so a row asserting ``False`` would
# pin every transport -- these declare it per transport instead.


def api() -> ModelCapability:
    """Return what the Messages API adds on top of a model row.

    One model, three ways in::

        models()["opus-5"] & api()          # API key: full tiers, cache
        models()["opus-5"] & subscription() # same wire, OAuth-billed
        models()["opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        cache_ttl_sec=cache_ttls(),
        manage_context_server_side={False, True},
        retries_internally=True,
        # No ``flex``: the Messages API takes only ``auto`` / ``standard_only``
        # (https://platform.claude.com/docs/en/api/messages/create), which map to
        # ``auto`` / ``default`` here. ``priority`` is the fast-mode beta.
        service_tier={"auto", "default", "priority"},
    )


def cli() -> ModelCapability:
    """Return what the ``claude`` subprocess narrows a model row to.

    One model, three ways in::

        models()["opus-5"] & api()          # API key: full tiers, cache
        models()["opus-5"] & subscription() # same wire, OAuth-billed
        models()["opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text"},
        manage_context_server_side={True},
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        models()["opus-5"] & api()          # API key: full tiers, cache
        models()["opus-5"] & subscription() # same wire, OAuth-billed
        models()["opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The subscription transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        cache_ttl_sec=cache_ttls(),
        manage_context_server_side={False, True},
        retries_internally=True,
        account_auth=True,
        # No ``flex``: the Messages API takes only ``auto`` / ``standard_only``
        # (https://platform.claude.com/docs/en/api/messages/create), which map to
        # ``auto`` / ``default`` here. ``priority`` is the fast-mode beta.
        service_tier={"auto", "default", "priority"},
    )


def _limits(
    *,
    window: int,
    edge: int,
    response: int = 128_000,
    long: int = 1_000_000,
    long_betas: frozenset[str] = frozenset(),
) -> Mapping[ContextTag, ModelLimits]:
    """Default to the maximum window; ``+200k`` selects the smaller cap."""
    limits = ModelLimits(
        max_request_tokens=window,
        max_response_tokens=response,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=edge,
        max_image_bytes=5 * 1024 * 1024,
    )
    context: dict[ContextTag, ModelLimits] = {
        "": replace(
            limits,
            max_request_tokens=long,
            request_betas=long_betas,
        )
        if long
        else limits,
    }
    if long:
        context["+200k"] = limits
    return MappingProxyType(context)


# Cache rates are multiples of the BASE INPUT price, so they ride every other modifier:
# "Prompt caching multipliers apply on top of fast mode pricing"
# (https://docs.anthropic.com/en/docs/about-claude/pricing). Leaving them unmultiplied
# on the fast row under-billed cached fast requests by half.
def _prices(
    *,
    request: float,
    response: float,
    fast_multiple: float = 0.0,
    cache_read_multiple: float = 0.1,
) -> PriceCatalog:
    """USD per million tokens, plus the priority row when the model has one."""
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=request * 1.25,
            cache_read=request * cache_read_multiple,
        ),
    }
    if fast_multiple:
        fast_request = request * fast_multiple
        rows[PriceCatalogProduct(service_tier="priority")] = TokenPrice(
            request=fast_request,
            response=response * fast_multiple,
            cache_write=fast_request * 1.25,
            cache_read=fast_request * cache_read_multiple,
        )
    return PriceCatalog(rows)
