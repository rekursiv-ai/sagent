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

from collections.abc import (
    Mapping,
)
from dataclasses import replace
from types import MappingProxyType

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


__all__ = ["api", "cache_ttls", "chars_per_token", "cli", "models", "subscription"]


def cache_ttls() -> frozenset[float]:
    """The two lifetimes ``cache_control`` spells, in seconds.

    No ``0``: these transports write a breakpoint on every request, so
    "do not cache" is not a selection this vendor offers.
    """
    return frozenset({300.0, 3600.0})


def chars_per_token(model_id: str) -> float:
    """Return the divisor the Anthropic provider's ``token_estimate`` uses.

    Measured via ``messages.count_tokens`` against the live API on 347k
    chars of real session text (2026-08-22): two tokenizer generations,
    not three. sonnet-4-5 and haiku-4-5 report counts byte-identical to
    opus-4-6, so the 4.83 tier they used to carry did not exist and
    under-counted their tokens by 55%.

    Args:
      model_id: Base model id, without option tags.

    Returns:
      divisor: Characters per token for that model.

    """
    match model_id:
        case (
            "claude-fable-5-1"
            | "claude-fable-5"
            | "claude-opus-5"
            | "claude-opus-4-8"
            | "claude-opus-4-7"
            | "claude-sonnet-5"
        ):
            return 2.38
        case _:
            return 3.12


# Thinking capability measured against the live API (Jun 2026). ``auto`` is
# ``thinking.type=adaptive`` (no budget); ``fixed`` is ``enabled`` +
# ``budget_tokens``. A model missing ``text`` reasons but returns a
# signed-but-empty block -- the plaintext is never delivered.
#
# Every row states ``none`` in each thinking set: a model that CAN think can
# also be asked not to, and the axis is total, so omitting it would make
# "thinking disabled" unselectable.
def models() -> Mapping[str, ModelCapability]:
    """Return every Anthropic model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # The 5 generation, as a row: every model below is this one with its
    # differences replaced. ``_limits`` and ``_prices`` build the two composed
    # fields, which ``replace`` cannot reach into.
    five = ModelCapability(
        context=_limits(window=1_000_000, edge=2576),
        prices=_prices(request=10.0, response=50.0),
        thinking_effort={"none", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto"},
        thinking_output={"none", "redacted"},
    )
    # The 4-x generation differs from it in four ways at once: smaller images,
    # a 200k untagged window, readable reasoning, a fixed budget.
    four = replace(
        five,
        context=_limits(window=200_000, edge=1568),
        prices=_prices(request=3.0, response=15.0),
        thinking_effort={"none"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
    )
    rows = (
        replace(
            five,
            model_id="claude-fable-5-1",
        ),
        replace(
            five,
            model_id="claude-fable-5",
        ),
        replace(
            five,
            model_id="claude-opus-5",
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            service_tier={"auto", "default", "priority"},
        ),
        replace(
            five,
            model_id="claude-opus-4-8",
            context=_limits(window=200_000, edge=2576),
            prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
            service_tier={"auto", "default", "priority"},
        ),
        # No fast price row on 4-7 or 4-6: 4-7 lost fast mode on 2026-07-24 and
        # the API now rejects ``speed="fast"``; 4-6 never shipped it and bills
        # standard, so a fast row would misprice both.
        replace(
            five,
            model_id="claude-opus-4-7",
            context=_limits(window=200_000, edge=2576),
            prices=_prices(request=5.0, response=25.0),
        ),
        replace(
            four,
            model_id="claude-opus-4-6",
            prices=_prices(request=5.0, response=25.0),
            thinking_effort={"none", "low", "medium", "high", "max"},
        ),
        replace(
            four,
            model_id="claude-opus-4-5",
            prices=_prices(request=5.0, response=25.0),
            thinking_effort={"none", "low", "medium", "high"},
            thinking_budget={"none", "fixed"},
        ),
        replace(
            five,
            model_id="claude-sonnet-5",
            prices=_prices(request=3.0, response=15.0),
        ),
        replace(
            four,
            model_id="claude-sonnet-4-6",
            thinking_effort={"none", "low", "medium", "high", "max"},
        ),
        replace(
            four,
            model_id="claude-sonnet-4-5",
            thinking_budget={"none", "fixed"},
        ),
        replace(
            four,
            model_id="claude-haiku-4-5",
            context=_limits(window=200_000, edge=1568, response=64_000, long=0),
            prices=_prices(request=1.0, response=5.0),
            thinking_budget={"none", "fixed"},
        ),
    )
    return MappingProxyType({row.model_id: row for row in rows})


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts. ``&`` can only remove, so a row asserting ``False`` would
# pin every transport -- these declare it per transport instead.


def api() -> ModelCapability:
    """Return what the Messages API adds on top of a model row.

    One model, three ways in::

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

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

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

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

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

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
    *, window: int, edge: int, response: int = 128_000, long: int = 1_000_000
) -> Mapping[ContextTag, ModelLimits]:
    """Both context tags; ``long=0`` offers no ``+1m`` variant."""
    limits = ModelLimits(
        max_request_tokens=window,
        max_response_tokens=response,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=edge,
        max_image_bytes=5 * 1024 * 1024,
    )
    context: dict[ContextTag, ModelLimits] = {"": limits}
    if long:
        context["+1m"] = replace(limits, max_request_tokens=long)
    return MappingProxyType(context)


def _prices(
    *, request: float, response: float, fast_multiple: float = 0.0
) -> PriceCatalog:
    """USD per million tokens, plus the priority row when the model has one.

    Fast mode surcharges request/response only: Anthropic's fast-mode table
    lists no separate cache rates.
    """
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    }
    if fast_multiple:
        rows[PriceCatalogProduct(service_tier="priority")] = TokenPrice(
            request=request * fast_multiple,
            response=response * fast_multiple,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    return PriceCatalog(rows)
