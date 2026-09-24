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

from sagent.lib.custom_json import DictCodec, IntCodec
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenCount,
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
    "usage_tokens",
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
# Knowledge cutoffs are the vendor's "reliable knowledge cutoff", and prices
# its published table (input, output, 5m/1h cache write, cache hit), both
# verified on 2026-09-24.
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
        prices=_prices(
            _card(
                request=4.0,
                response=20.0,
                cache_write=5.0,
                cache_write_1h=8.0,
                cache_read=0.2,
            ),
            fast_request=8.0,
            fast_response=40.0,
        ),
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "redacted"}),
        ),
    )
    # The 4.6-and-earlier reasoning contract: readable thinking and a fixed
    # budget alongside adaptive.
    extended = replace(
        default.thinking,
        budget=frozenset({"none", "auto", "fixed"}),
        output=frozenset({"none", "text", "redacted"}),
    )
    rows = (
        replace(
            default,
            model_id="fable-5.1",
            wire_model_id="claude-fable-5-1",
            prices=_prices(
                _card(
                    request=10.0,
                    response=50.0,
                    cache_write=12.5,
                    cache_write_1h=20.0,
                    cache_read=0.25,
                ),
            ),
        ),
        replace(
            default,
            model_id="fable-5",
            wire_model_id="claude-fable-5",
            knowledge_cutoff="January 2026",
            prices=_prices(
                _card(
                    request=10.0,
                    response=50.0,
                    cache_write=12.5,
                    cache_write_1h=20.0,
                    cache_read=1.0,
                ),
            ),
        ),
        default,
        replace(
            default,
            model_id="opus-5",
            wire_model_id="claude-opus-5",
            knowledge_cutoff="May 2026",
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
                fast_request=10.0,
                fast_response=50.0,
            ),
        ),
        replace(
            default,
            model_id="opus-4.8",
            wire_model_id="claude-opus-4-8",
            knowledge_cutoff="January 2026",
            context=_limits(beta="context-1m-2025-08-07"),
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
                fast_request=10.0,
                fast_response=50.0,
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
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
        ),
        replace(
            default,
            model_id="opus-4.6",
            wire_model_id="claude-opus-4-6",
            knowledge_cutoff="May 2025",
            approx_chars_per_token=3.12,
            context=_limits(beta="context-1m-2025-08-07", image_edge_px=1568),
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"none", "low", "medium", "high", "max"}),
            ),
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
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"none", "low", "medium", "high"}),
                budget=frozenset({"none", "fixed"}),
            ),
        ),
        # $2/$10 was introductory pricing through 2026-08-31; the scheduled
        # rise to $3/$15 on 2026-09-01 was cancelled and this is now standard.
        replace(
            default,
            model_id="sonnet-5",
            wire_model_id="claude-sonnet-5",
            knowledge_cutoff="January 2026",
            prices=_prices(
                _card(
                    request=2.0,
                    response=10.0,
                    cache_write=2.5,
                    cache_write_1h=4.0,
                    cache_read=0.2,
                ),
            ),
        ),
        replace(
            default,
            model_id="sonnet-4.6",
            wire_model_id="claude-sonnet-4-6",
            knowledge_cutoff="August 2025",
            approx_chars_per_token=3.12,
            context=_limits(beta="context-1m-2025-08-07", image_edge_px=1568),
            prices=_prices(
                _card(
                    request=3.0,
                    response=15.0,
                    cache_write=3.75,
                    cache_write_1h=6.0,
                    cache_read=0.3,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"none", "low", "medium", "high", "max"}),
            ),
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
            prices=_prices(
                _card(
                    request=3.0,
                    response=15.0,
                    cache_write=3.75,
                    cache_write_1h=6.0,
                    cache_read=0.3,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"none"}),
                budget=frozenset({"none", "fixed"}),
            ),
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
            prices=_prices(
                _card(
                    request=1.0,
                    response=5.0,
                    cache_write=1.25,
                    cache_write_1h=2.0,
                    cache_read=0.1,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"none"}),
                budget=frozenset({"none", "fixed"}),
            ),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
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

        models()["opus-5"] & api()          # API key: full tiers, cache
        models()["opus-5"] & subscription() # same wire, OAuth-billed
        models()["opus-5"] & cli()          # subprocess: no effort/tier

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

        models()["opus-5"] & api()          # API key: full tiers, cache
        models()["opus-5"] & subscription() # same wire, OAuth-billed
        models()["opus-5"] & cli()          # subprocess: no effort/tier

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
    written = IntCodec.coerce(usage.get("cache_creation_input_tokens"), 0)
    split = DictCodec.coerce(usage.get("cache_creation"))
    if split:
        written_1h = IntCodec.coerce(split.get("ephemeral_1h_input_tokens"), 0)
    else:
        written_1h = written if cache_ttl_sec >= 3600.0 else 0
    return TokenCount(
        request=IntCodec.coerce(usage.get("input_tokens"), 0),
        response=IntCodec.coerce(usage.get("output_tokens"), 0),
        cache_write=written - written_1h,
        cache_write_1h=written_1h,
        cache_read=IntCodec.coerce(usage.get("cache_read_input_tokens"), 0),
    )


def _card(
    *,
    request: float,
    response: float,
    cache_write: float,
    cache_write_1h: float,
    cache_read: float,
) -> TokenPrice:
    """Return one published price-table row: input, output, 5m/1h writes, hits."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=cache_write_1h,
        cache_read=cache_read,
    )


# Fast mode publishes only input and output; "prompt caching multipliers apply
# on top of fast mode pricing", so each cache rate keeps its ratio to input.
def _prices(
    standard: TokenPrice,
    *,
    fast_request: float = 0.0,
    fast_response: float = 0.0,
) -> PriceCatalog:
    """Return the standard card, plus the fast-mode card when the model has one."""
    cards = {PriceKey("auto"): standard}
    if fast_request:
        scale = fast_request / standard.request
        cards[PriceKey("priority")] = _card(
            request=fast_request,
            response=fast_response,
            cache_write=standard.cache_write * scale,
            cache_write_1h=standard.cache_write_1h * scale,
            cache_read=standard.cache_read * scale,
        )
    return PriceCatalog(cards)


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
