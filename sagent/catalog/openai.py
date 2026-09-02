"""OpenAI model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits: https://developers.openai.com/api/docs/models/<model>
  - Pricing: https://developers.openai.com/api/docs/pricing
  - Images: https://developers.openai.com/api/docs/guides/images-vision

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`chat`, :func:`subscription`), which can only remove.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ServiceTier,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


__all__ = ["api", "chat", "models", "reasoning_effort", "subscription"]


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts declared on ``API`` / ``SUBSCRIPTION``, since ``&`` can only
# remove. Every reasoning model takes an auto budget and returns readable text.
# ``chars_per_token`` measured from SERVER-reported ``usage.input_tokens``
# on 347k chars of real session text (2026-08-22), differencing out the
# per-request envelope. 3.71 across the whole range -- the catalog's prior
# flat 4.0 was the dataclass default, never measured, and over-credited
# every budget by ~7%. tiktoken says 3.81 locally; the gap is request
# framing the local tokenizer never sees.
def models() -> Mapping[str, ModelCapability]:
    """Return every OpenAI model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    return MappingProxyType(
        {
            "gpt-5.6-sol": ModelCapability(
                model_id="gpt-5.6-sol",
                context=_windowed(
                    request=272_000, response=128_000, gpt56_images=True, long=1_050_000
                ),
                prices=_prices(
                    request=5.0,
                    response=30.0,
                    cache_write=6.25,
                    cache_read=0.5,
                    two_tier=True,
                ),
                service_tier=_tiers(),
                thinking_effort=_gpt56_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            # Absent from ``GET /v1/models`` yet serves ``POST /v1/responses``
            # normally. Listing is an entitlement view, not the model set, so a
            # row must be dropped only on ``model_not_found`` from a real call.
            "gpt-5.6": ModelCapability(
                model_id="gpt-5.6",
                context=_windowed(
                    request=272_000, response=128_000, gpt56_images=True, long=1_050_000
                ),
                prices=_prices(
                    request=5.0,
                    response=30.0,
                    cache_write=6.25,
                    cache_read=0.5,
                    two_tier=True,
                ),
                service_tier=_tiers(),
                thinking_effort=_gpt56_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.6-luna": ModelCapability(
                model_id="gpt-5.6-luna",
                context=_windowed(
                    request=272_000, response=128_000, gpt56_images=True, long=1_050_000
                ),
                prices=_prices(
                    request=1.0,
                    response=6.0,
                    cache_write=1.25,
                    cache_read=0.1,
                    two_tier=True,
                ),
                service_tier=_tiers(),
                thinking_effort=_gpt56_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.6-terra": ModelCapability(
                model_id="gpt-5.6-terra",
                context=_windowed(
                    request=272_000, response=128_000, gpt56_images=True, long=1_050_000
                ),
                prices=_prices(
                    request=2.5,
                    response=15.0,
                    cache_write=3.125,
                    cache_read=0.25,
                    two_tier=True,
                ),
                service_tier=_tiers(),
                thinking_effort=_gpt56_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.5": ModelCapability(
                model_id="gpt-5.5",
                context=_windowed(request=272_000, response=128_000, long=1_000_000),
                prices=_prices(
                    request=5.0, response=30.0, cache_read=0.5, two_tier=True
                ),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.5-pro": ModelCapability(
                model_id="gpt-5.5-pro",
                context=_windowed(request=272_000, response=128_000, long=1_050_000),
                prices=_prices(request=30.0, response=180.0, two_tier=True),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.4": ModelCapability(
                model_id="gpt-5.4",
                context=_windowed(request=272_000, response=128_000, long=1_050_000),
                prices=_prices(
                    request=2.5, response=15.0, cache_read=0.25, two_tier=True
                ),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.4-pro": ModelCapability(
                model_id="gpt-5.4-pro",
                context=_windowed(request=272_000, response=128_000, long=1_050_000),
                prices=_prices(request=30.0, response=180.0, two_tier=True),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.4-mini": ModelCapability(
                model_id="gpt-5.4-mini",
                context=_context(request=400_000, response=128_000),
                prices=_prices(request=0.75, response=4.5, cache_read=0.075),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.4-nano": ModelCapability(
                model_id="gpt-5.4-nano",
                context=_context(request=400_000, response=128_000),
                prices=_prices(request=0.2, response=1.25, cache_read=0.02),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.3-codex": ModelCapability(
                model_id="gpt-5.3-codex",
                context=_context(request=400_000, response=128_000),
                prices=_prices(request=1.75, response=14.0, cache_read=0.175),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.3-chat-latest": ModelCapability(
                model_id="gpt-5.3-chat-latest",
                context=_context(request=128_000, response=16_384),
                prices=_prices(request=1.75, response=14.0, cache_read=0.175),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-5.2": ModelCapability(
                model_id="gpt-5.2",
                context=_context(request=400_000, response=128_000),
                prices=_prices(request=1.75, response=14.0, cache_read=0.175),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "o1": ModelCapability(
                model_id="o1",
                context=_context(request=200_000, response=100_000),
                prices=_prices(request=15.0, response=60.0, cache_read=7.5),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "o3-mini": ModelCapability(
                model_id="o3-mini",
                context=_context(request=200_000, response=100_000),
                prices=_prices(request=1.1, response=4.4, cache_read=0.55),
                service_tier=_tiers(),
                thinking_effort=_legacy_efforts(),
                thinking_budget=frozenset({"none", "auto"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "gpt-4.1": ModelCapability(
                model_id="gpt-4.1",
                context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
                prices=_prices(request=2.0, response=8.0, cache_read=0.5),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4.1-mini": ModelCapability(
                model_id="gpt-4.1-mini",
                context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
                prices=_prices(request=0.4, response=1.6, cache_read=0.1),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4.1-nano": ModelCapability(
                model_id="gpt-4.1-nano",
                context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
                prices=_prices(request=0.1, response=0.4, cache_read=0.025),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4o": ModelCapability(
                model_id="gpt-4o",
                context=_context(request=128_000, response=16_384),
                prices=_prices(request=2.5, response=10.0, cache_read=1.25),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4o-mini": ModelCapability(
                model_id="gpt-4o-mini",
                context=_context(request=128_000, response=16_384),
                prices=_prices(request=0.15, response=0.6, cache_read=0.075),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4-turbo": ModelCapability(
                model_id="gpt-4-turbo",
                context=_context(request=128_000, response=4_096),
                prices=_prices(request=10.0, response=30.0),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
            "gpt-4": ModelCapability(
                model_id="gpt-4",
                context=_context(request=8_192, response=8_192),
                prices=_prices(request=30.0, response=60.0),
                service_tier=_tiers(),
                # No reasoning knob on this generation.
            ),
        }
    )


def reasoning_effort(
    effort: ThinkingEffort, *, model_id: str, chat: bool = False
) -> str:
    """Return the wire ``reasoning_effort`` an effort sends.

    Two ladders: GPT-5.6 takes ``none`` and ``xhigh`` natively, while
    everything earlier absorbs them into ``minimal`` and ``high``.

    Args:
      effort: Selected effort level.
      model_id: Base model id, which picks the ladder.
      chat: Whether the request goes to Chat Completions.

    Returns:
      wire: Value the request body carries.

    """
    if not model_id.startswith("gpt-5.6"):
        match effort:
            case "none" | "min":
                return "minimal"
            case "xhigh" | "max":
                return "high"
            case _:
                return effort
    match effort:
        case "none" | "min":
            return "none"
        case "max":
            return "xhigh" if chat else "max"
        case _:
            return effort


def api() -> ModelCapability:
    """Return what the Responses API adds on top of a model row.

    One model, three ways in::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth
        models()["gpt-5.6"] & chat()         # Chat Completions: same set

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"}
        ),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
    )


def subscription() -> ModelCapability:
    """Return what Codex billing narrows a model row to.

    One model, three ways in::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth
        models()["gpt-5.6"] & chat()         # Chat Completions: same set

    Returns:
      capability: The subscription transport's restrictions.

    """
    # ``auto`` stays selectable: ``/fast`` picks ``priority``, but an
    # unqualified request still sends the default, so dropping it would make
    # ``ModelSettings()`` invalid on every Codex model.
    return ModelCapability(
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"}
        ),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        service_tier=frozenset({"auto", "priority"}),
        account_auth=True,
    )


def chat() -> ModelCapability:
    """Return what Chat Completions narrows a model row to.

    Removes nothing: only ``max`` differs, and that is a value remap (see
    :func:`reasoning_effort`).

    One model, three ways in::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth
        models()["gpt-5.6"] & chat()         # Chat Completions: same set

    Returns:
      capability: The Chat transport's restrictions.

    """
    # Every axis stated: ``&`` can only remove, and a defaulted axis is the
    # narrow value, so an omitted one would strip the model's real capability.
    return ModelCapability(
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"}
        ),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
    )


# Prompts above 272K input tokens bill at 2x input / 1.5x output for the whole
# request on gpt-5.4 and later. The untagged id caps the window here; a ``+1m``
# id opts into the full window and thus into the surcharge.
_TWO_TIER: Final = 272_000


def _one(*, request: int, response: int, gpt56_images: bool = False) -> ModelLimits:
    """Pre-5.6 tiles ``detail:high`` from a 2048px square; 5.6 does not."""
    if gpt56_images:
        return ModelLimits(
            max_request_tokens=request,
            max_response_tokens=response,
            max_request_bytes=512 * 1024 * 1024,
        )
    return ModelLimits(
        max_request_tokens=request,
        max_response_tokens=response,
        max_request_bytes=20 * 1024 * 1024,
        max_image_edge_px=2048,
        max_image_bytes=20 * 1024 * 1024,
    )


def _context(
    *, request: int, response: int, gpt56_images: bool = False
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag, for a model with no ``+1m`` variant."""
    return MappingProxyType(
        {"": _one(request=request, response=response, gpt56_images=gpt56_images)}
    )


def _windowed(
    *, request: int, response: int, long: int, gpt56_images: bool = False
) -> Mapping[ContextTag, ModelLimits]:
    """Both context tags; ``+1m`` opts into the full window and its surcharge."""
    return MappingProxyType(
        {
            "": _one(request=request, response=response, gpt56_images=gpt56_images),
            "+1m": _one(request=long, response=response, gpt56_images=gpt56_images),
        }
    )


def _prices(
    *,
    request: float,
    response: float,
    cache_write: float = 0.0,
    cache_read: float = 0.0,
    two_tier: bool = False,
) -> PriceCatalog:
    """USD per million tokens, plus the >272K surcharge row when two-tier."""
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=cache_write,
            cache_read=cache_read,
        )
    }
    if two_tier:
        # The >272K surcharge applies the input multiplier to all three input
        # pools and the output multiplier to the whole response.
        rows[PriceCatalogProduct(min_request_tokens=_TWO_TIER)] = TokenPrice(
            request=request * 2.0,
            response=response * 1.5,
            cache_write=cache_write * 2.0,
            cache_read=cache_read * 2.0,
        )
    return PriceCatalog(rows)


def _tiers() -> frozenset[ServiceTier]:
    """Every tier OpenAI sells; a transport narrows it (Codex: priority only)."""
    return frozenset({"auto", "default", "flex", "priority"})


def _legacy_efforts() -> frozenset[ThinkingEffort]:
    """Every effort the pre-5.6 reasoning wire accepts."""
    return frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"})


def _gpt56_efforts() -> frozenset[ThinkingEffort]:
    """Every effort the GPT-5.6 reasoning wire accepts."""
    return frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"})
