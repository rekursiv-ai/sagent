"""Anthropic model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - Limits, cutoffs: https://docs.anthropic.com/en/docs/about-claude/models
  - Pricing: https://docs.anthropic.com/en/docs/about-claude/pricing
  - Fast mode: https://docs.anthropic.com/en/docs/build-with-claude/fast-mode
  - Vision: https://platform.claude.com/docs/en/build-with-claude/vision
  - Request size: https://platform.claude.com/docs/en/api/overview

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
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
    "CACHE_TTL_SEC",
    "api",
    "cli",
    "models",
    "subscription",
]


# No ``0``: these transports write a breakpoint on every request, so "do not
# cache" is not a selection this vendor offers.
CACHE_TTL_SEC: Final = frozenset({300.0, 3600.0})


# Thinking capability measured against the live API (Jun 2026). ``auto`` is
# ``thinking.type=adaptive`` (no budget); ``fixed`` is ``enabled`` +
# ``budget_tokens``. A model missing ``text`` reasons but returns a
# signed-but-empty block -- the plaintext is never delivered.
#
# Every row states ``none`` in each thinking set: a model that CAN think can
# also omit an explicit thinking request, and the axis is total, so omitting it
# would make the provider default unselectable.
#
# Knowledge cutoffs are the vendor's "reliable knowledge cutoff", verified
# against each model's overview page on 2026-09-24.
def models() -> Mapping[str, ModelCapability]:
    """Return every Anthropic model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    default = ModelCapability(
        model_id="opus-5.5",
        wire_model_id="claude-opus-5-5",
        knowledge_cutoff="June 2026",
        approx_chars_per_token=2.38,
        context=_limits(),
        # Opus 5.5 bills cache hits at 0.05x base input, not the usual 0.1x.
        prices=_prices(
            input_usd=4.0,
            output_usd=20.0,
            cache_read_usd=0.2,
            priority_input_usd=8.0,
            priority_output_usd=40.0,
        ),
        thinking=ThinkingCapability(
            effort={"none", "low", "medium", "high", "xhigh", "max"},
            budget={"none", "auto"},
            output={"none", "redacted"},
        ),
    )
    # The 4.6-and-earlier reasoning contract: readable thinking and a fixed
    # budget alongside adaptive.
    extended = replace(
        default.thinking,
        budget={"none", "auto", "fixed"},
        output={"none", "text", "redacted"},
    )
    rows = (
        # Fable 5.1 and Mythos 5.1 bill cache hits at 0.025x base input.
        replace(
            default,
            model_id="fable-5.1",
            wire_model_id="claude-fable-5-1",
            prices=_prices(input_usd=10.0, output_usd=50.0, cache_read_usd=0.25),
        ),
        replace(
            default,
            model_id="fable-5",
            wire_model_id="claude-fable-5",
            knowledge_cutoff="January 2026",
            prices=_prices(input_usd=10.0, output_usd=50.0),
        ),
        default,
        replace(
            default,
            model_id="opus-5",
            wire_model_id="claude-opus-5",
            knowledge_cutoff="May 2026",
            prices=_prices(
                input_usd=5.0,
                output_usd=25.0,
                priority_input_usd=10.0,
                priority_output_usd=50.0,
            ),
        ),
        replace(
            default,
            model_id="opus-4.8",
            wire_model_id="claude-opus-4-8",
            knowledge_cutoff="January 2026",
            context=_limits(beta="context-1m-2025-08-07"),
            prices=_prices(
                input_usd=5.0,
                output_usd=25.0,
                priority_input_usd=10.0,
                priority_output_usd=50.0,
            ),
        ),
        # No fast price row on 4-7 or 4-6: 4-7 lost fast mode on 2026-07-24 and
        # the API now rejects ``speed="fast"``; 4-6 never shipped it and bills
        # standard, so a fast row would misprice both.
        replace(
            default,
            model_id="opus-4.7",
            wire_model_id="claude-opus-4-7",
            knowledge_cutoff="January 2026",
            context=_limits(beta="context-1m-2025-08-07"),
            prices=_prices(input_usd=5.0, output_usd=25.0),
        ),
        replace(
            default,
            model_id="opus-4.6",
            wire_model_id="claude-opus-4-6",
            knowledge_cutoff="May 2025",
            approx_chars_per_token=3.12,
            context=_limits(beta="context-1m-2025-08-07", image_edge_px=1568),
            prices=_prices(input_usd=5.0, output_usd=25.0),
            thinking=replace(extended, effort={"none", "low", "medium", "high", "max"}),
        ),
        replace(
            default,
            model_id="opus-4.5",
            wire_model_id="claude-opus-4-5",
            knowledge_cutoff="May 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(input_usd=5.0, output_usd=25.0),
            thinking=replace(
                extended,
                effort={"none", "low", "medium", "high"},
                budget={"none", "fixed"},
            ),
        ),
        # $2/$10 was introductory pricing through 2026-08-31; the scheduled
        # rise to $3/$15 on 2026-09-01 was cancelled and this is now standard.
        replace(
            default,
            model_id="sonnet-5",
            wire_model_id="claude-sonnet-5",
            knowledge_cutoff="January 2026",
            prices=_prices(input_usd=2.0, output_usd=10.0),
        ),
        replace(
            default,
            model_id="sonnet-4.6",
            wire_model_id="claude-sonnet-4-6",
            knowledge_cutoff="August 2025",
            approx_chars_per_token=3.12,
            context=_limits(beta="context-1m-2025-08-07", image_edge_px=1568),
            prices=_prices(input_usd=3.0, output_usd=15.0),
            thinking=replace(extended, effort={"none", "low", "medium", "high", "max"}),
        ),
        replace(
            default,
            model_id="sonnet-4.5",
            wire_model_id="claude-sonnet-4-5",
            knowledge_cutoff="January 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(input_usd=3.0, output_usd=15.0),
            thinking=replace(extended, effort={"none"}, budget={"none", "fixed"}),
        ),
        replace(
            default,
            model_id="haiku-4.5",
            wire_model_id="claude-haiku-4-5",
            knowledge_cutoff="February 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(input_usd=1.0, output_usd=5.0),
            thinking=replace(extended, effort={"none"}, budget={"none", "fixed"}),
        ),
    )
    # A tier is offered exactly when it is priced, so the row states it once,
    # in ``prices``, and a fast row can never be added without being offered.
    rows = tuple(
        replace(
            row,
            service_tier={"auto", "default", *(p.service_tier for p in row.prices)},
        )
        for row in rows
    )
    # Rows run newest-first within each family, so the first match is the latest
    # and a new release moves its alias without an edit here.
    latest = {
        alias: next(row for row in rows if row.model_id.startswith(f"{family}-"))
        for alias, family in (
            ("best", "fable"),
            ("default", "opus"),
            ("utility", "sonnet"),
        )
    }
    return MappingProxyType(latest | {row.model_id: row for row in rows})


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
        thinking=ThinkingCapability(
            effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
            budget={"none", "auto", "fixed"},
            output={"none", "text", "redacted"},
        ),
        cache_ttl_sec=CACHE_TTL_SEC,
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
        thinking=ThinkingCapability(
            effort={"none"},
            budget={"none", "auto", "fixed"},
            output={"none", "text"},
        ),
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
    return replace(api(), account_auth=True)


def _limits(
    *,
    max_tokens: int = 1_000_000,
    beta: str = "",
    output_tokens: int = 128_000,
    image_edge_px: int = 2576,
) -> Mapping[ContextTag, ModelLimits]:
    """Serve ``max_tokens`` by default; ``+200k`` selects 200k when it is larger."""
    limits = ModelLimits(
        max_request_tokens=max_tokens,
        max_response_tokens=output_tokens,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=image_edge_px,
        max_image_bytes=5 * 1024 * 1024,
        request_betas=frozenset({beta} if beta else ()),
    )
    context: dict[ContextTag, ModelLimits] = {"": limits}
    if max_tokens > 200_000:
        context["+200k"] = replace(
            limits,
            max_request_tokens=200_000,
            request_betas=frozenset(),
        )
    return MappingProxyType(context)


# Cache rates are multiples of the BASE INPUT price, so they ride every other modifier:
# "Prompt caching multipliers apply on top of fast mode pricing"
# (https://docs.anthropic.com/en/docs/about-claude/pricing). Leaving them unmultiplied
# on the fast row under-billed cached fast requests by half.
def _prices(
    *,
    input_usd: float,
    output_usd: float,
    cache_read_usd: float | None = None,
    priority_input_usd: float | None = None,
    priority_output_usd: float | None = None,
) -> PriceCatalog:
    """USD per million tokens; cache read defaults to 0.1x input."""
    cache_read_multiple = 0.1 if cache_read_usd is None else cache_read_usd / input_usd
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=input_usd,
            response=output_usd,
            cache_write=input_usd * 1.25,
            cache_read=input_usd * cache_read_multiple,
        ),
    }
    if priority_input_usd is not None and priority_output_usd is not None:
        rows[PriceCatalogProduct(service_tier="priority")] = TokenPrice(
            request=priority_input_usd,
            response=priority_output_usd,
            cache_write=priority_input_usd * 1.25,
            cache_read=priority_input_usd * cache_read_multiple,
        )
    return PriceCatalog(rows)
