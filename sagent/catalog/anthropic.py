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

from collections.abc import Mapping
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


__all__ = ["api", "chars_per_token", "cli", "models", "subscription"]


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
    return MappingProxyType(
        {
            "claude-fable-5-1": ModelCapability(
                model_id="claude-fable-5-1",
                context=_windowed(edge=2576, default=1_000_000),
                prices=_prices(request=10.0, response=50.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
            ),
            "claude-fable-5": ModelCapability(
                model_id="claude-fable-5",
                context=_windowed(edge=2576, default=1_000_000),
                prices=_prices(request=10.0, response=50.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
            ),
            "claude-opus-5": ModelCapability(
                model_id="claude-opus-5",
                context=_windowed(edge=2576, default=1_000_000),
                prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
                service_tier=frozenset({"auto", "default", "priority"}),
            ),
            "claude-opus-4-8": ModelCapability(
                model_id="claude-opus-4-8",
                context=_windowed(edge=2576),
                prices=_prices(request=5.0, response=25.0, fast_multiple=2.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
                service_tier=frozenset({"auto", "default", "priority"}),
            ),
            "claude-opus-4-7": ModelCapability(
                model_id="claude-opus-4-7",
                context=_windowed(edge=2576),
                # Fast mode was removed from 4-7 on 2026-07-24; the API now
                # rejects ``speed="fast"`` rather than serving it.
                prices=_prices(request=5.0, response=25.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
            ),
            "claude-opus-4-6": ModelCapability(
                model_id="claude-opus-4-6",
                context=_windowed(edge=1568),
                # 4-6 never shipped fast mode: a ``speed="fast"`` request runs at
                # standard speed and bills standard, reporting usage.speed
                # "standard". A fast price row would overstate the cost.
                prices=_prices(request=5.0, response=25.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset({"none", "low", "medium", "high", "max"}),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text", "redacted"}),
            ),
            "claude-opus-4-5": ModelCapability(
                model_id="claude-opus-4-5",
                context=_windowed(edge=1568),
                prices=_prices(request=5.0, response=25.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset({"none", "low", "medium", "high"}),
                thinking_budget=frozenset({"none", "fixed"}),
                thinking_output=frozenset({"none", "text", "redacted"}),
            ),
            "claude-sonnet-5": ModelCapability(
                model_id="claude-sonnet-5",
                context=_windowed(edge=2576, default=1_000_000),
                prices=_prices(request=3.0, response=15.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset(
                    {"none", "low", "medium", "high", "xhigh", "max"}
                ),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "redacted"}),
            ),
            "claude-sonnet-4-6": ModelCapability(
                model_id="claude-sonnet-4-6",
                context=_windowed(edge=1568),
                prices=_prices(request=3.0, response=15.0),
                cache_ttl_sec=3600.0,
                thinking_effort=frozenset({"none", "low", "medium", "high", "max"}),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text", "redacted"}),
            ),
            "claude-sonnet-4-5": ModelCapability(
                model_id="claude-sonnet-4-5",
                context=_windowed(edge=1568),
                prices=_prices(request=3.0, response=15.0),
                cache_ttl_sec=3600.0,
                thinking_budget=frozenset({"none", "fixed"}),
                thinking_output=frozenset({"none", "text", "redacted"}),
            ),
            "claude-haiku-4-5": ModelCapability(
                model_id="claude-haiku-4-5",
                context=MappingProxyType(
                    {"": _limits(request=200_000, response=64_000, edge=1568)}
                ),
                prices=_prices(request=1.0, response=5.0),
                cache_ttl_sec=3600.0,
                thinking_budget=frozenset({"none", "fixed"}),
                thinking_output=frozenset({"none", "text", "redacted"}),
            ),
        }
    )


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
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"}
        ),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        cache_ttl_sec=3600.0,
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

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking_effort=frozenset({"none"}),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text"}),
        manage_context_server_side=frozenset({True}),
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
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"}
        ),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        cache_ttl_sec=3600.0,
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
        account_auth=True,
        # No ``flex``: the Messages API takes only ``auto`` / ``standard_only``
        # (https://platform.claude.com/docs/en/api/messages/create), which map to
        # ``auto`` / ``default`` here. ``priority`` is the fast-mode beta.
        service_tier=frozenset({"auto", "default", "priority"}),
    )


def _limits(*, request: int, response: int = 128_000, edge: int) -> ModelLimits:
    """``edge`` is the native resolution above which the server downscales."""
    return ModelLimits(
        max_request_tokens=request,
        max_response_tokens=response,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=edge,
        max_image_bytes=5 * 1024 * 1024,
    )


def _windowed(*, edge: int, default: int = 200_000) -> Mapping[ContextTag, ModelLimits]:
    """Both context tags; the 5 generation passes ``default=1_000_000``."""
    return MappingProxyType(
        {
            "": _limits(request=default, edge=edge),
            "+1m": _limits(request=1_000_000, edge=edge),
        }
    )


def _prices(
    *,
    request: float,
    response: float,
    fast_multiple: float = 0.0,
) -> PriceCatalog:
    """USD per million tokens, plus the fast-tier surcharge row when priced."""
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    }
    if fast_multiple:
        # Fast mode surcharges request/response only: Anthropic's fast-mode
        # table lists no separate cache rates.
        rows[PriceCatalogProduct(service_tier="priority")] = TokenPrice(
            request=request * fast_multiple,
            response=response * fast_multiple,
            cache_write=request * 1.25,
            cache_read=request * 0.1,
        )
    return PriceCatalog(rows)
