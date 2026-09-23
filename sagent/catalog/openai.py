"""OpenAI model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits: https://developers.openai.com/api/docs/models/<model>
  - Pricing: https://developers.openai.com/api/docs/pricing
  - Images: https://developers.openai.com/api/docs/guides/images-vision

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`subscription`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "api",
    "compatible",
    "models",
    "reasoning_effort",
    "subscription",
    "subscription_models",
]


def compatible() -> ModelCapability:
    """Return the common Chat Completions transport capability.

    Returns:
      capability: Restrictions shared by OpenAI-compatible transports.

    """
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        service_tier={"auto"},
        manage_context_server_side={False},
    )


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts declared on ``API`` / ``SUBSCRIPTION``, since ``&`` can only
# remove. Every reasoning model takes an auto budget and returns readable text.
#
# GPT-5.6 prices verified against the vendor table on 2026-09-04. The family
# was repriced after this catalog first landed -- Sol $5/$30 -> $4/$20, Luna
# $1/$6 -> $0.20/$1.20 -- so the old rows over-billed by up to 5x. Sol's rate
# is promotional "at least through November 21, 2026" and needs rechecking then.
# ``approx_chars_per_token`` measured from SERVER-reported ``usage.input_tokens``
# on 347k chars of real session text (2026-08-22), differencing out the
# per-request envelope. 3.71 across the whole range -- the catalog's prior
# flat 4.0 was the dataclass default, never measured, and over-credited
# every budget by ~7%. tiktoken says 3.81 locally; the gap is request
# framing the local tokenizer never sees.
@cache
def models() -> Mapping[str, ModelCapability]:
    """Return every OpenAI model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # A GPT-5.6 row: every model below is this one with its window, price,
    # and reasoning axes replaced. 5.6 bills images by 32x32 patch and takes
    # a far larger request body than the generations before it.
    gpt56 = ModelCapability(
        approx_chars_per_token=3.71,
        knowledge_cutoff="February 16, 2026",
        context=_windowed(
            request=272_000,
            response=128_000,
            gpt56_images=True,
            long=1_050_000,
        ),
        prices=_prices(
            request=4.0,
            response=20.0,
            cache_write=5.0,
            cache_read=0.4,
            two_tier=True,
        ),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
        thinking_effort=frozenset(
            {"none", "min", "low", "medium", "high", "xhigh", "max"},
        ),
        thinking_budget={"none", "auto"},
        thinking_output={"none", "text"},
    )
    # Pre-5.6 tiles ``detail:high`` from a 2048px square and caps the body at
    # 20MB.
    legacy = replace(
        gpt56,
        knowledge_cutoff=None,
        thinking_effort={"none", "low", "medium", "high", "xhigh"},
        context=_windowed(request=272_000, response=128_000, long=1_050_000),
        prices=_prices(request=2.5, response=15.0, cache_read=0.25, two_tier=True),
    )
    # No reasoning knob on the 4-x generation.
    gpt4 = replace(
        legacy,
        thinking_effort={"none"},
        thinking_budget={"none"},
        thinking_output={"none"},
    )
    rows = (
        # Measured against the live API 2026-09-04: ``reasoning.effort``
        # takes low..max but rejects ``none`` and ``minimal``.
        # Every GPT-6 rule here was measured on this id alone, so each names
        # it exactly rather than the generation: a later GPT-6 whose contract
        # differs would otherwise inherit a limit nothing verified for it.
        replace(
            gpt56,
            model_id="astra-6",
            wire_model_id="gpt-6-astra",
            knowledge_cutoff="April 30, 2026",
            context=_windowed(
                request=272_000,
                response=128_000,
                long=1_050_000,
                gpt56_images=True,
            ),
            prices=_prices(
                request=10.0,
                response=50.0,
                cache_write=12.5,
                cache_read=1.0,
                two_tier=True,
            ),
            thinking_effort={"low", "medium", "high", "xhigh", "max"},
        ),
        replace(
            gpt56,
            model_id="sol-6",
            wire_model_id="gpt-6-sol",
            knowledge_cutoff="April 20, 2026",
            prices=_prices(
                request=2.0,
                response=10.0,
                cache_write=2.5,
                cache_read=0.2,
                two_tier=True,
            ),
        ),
        replace(
            gpt56,
            model_id="luna-6",
            wire_model_id="gpt-6-luna",
            knowledge_cutoff="May 18, 2026",
            prices=_prices(
                request=0.1,
                response=0.5,
                cache_write=0.125,
                cache_read=0.01,
                two_tier=True,
            ),
        ),
        replace(
            gpt56,
            model_id="sol-5.6",
            wire_model_id="gpt-5.6-sol",
        ),
        # Absent from ``GET /v1/models`` yet serves ``POST /v1/responses``
        # normally. Listing is an entitlement view, not the model set, so a
        # row must be dropped only on ``model_not_found`` from a real call.
        replace(
            gpt56,
            model_id="gpt-5.6",
        ),
        replace(
            gpt56,
            model_id="luna-5.6",
            wire_model_id="gpt-5.6-luna",
            prices=_prices(
                request=0.2,
                response=1.2,
                cache_write=0.25,
                cache_read=0.02,
                two_tier=True,
            ),
        ),
        replace(
            gpt56,
            model_id="terra-5.6",
            wire_model_id="gpt-5.6-terra",
            prices=_prices(
                request=2.0,
                response=12.0,
                cache_write=2.5,
                cache_read=0.2,
                two_tier=True,
            ),
        ),
        replace(
            legacy,
            model_id="gpt-5.5",
            context=_windowed(request=272_000, response=128_000, long=1_000_000),
            prices=_prices(request=5.0, response=30.0, cache_read=0.5, two_tier=True),
        ),
        replace(
            legacy,
            model_id="gpt-5.5-pro",
            thinking_effort={"medium", "high", "xhigh"},
            prices=_prices(request=30.0, response=180.0, two_tier=True),
        ),
        replace(legacy, model_id="gpt-5.4"),
        replace(
            legacy,
            model_id="gpt-5.4-pro",
            thinking_effort={"medium", "high", "xhigh"},
            prices=_prices(request=30.0, response=180.0, two_tier=True),
        ),
        replace(
            legacy,
            model_id="gpt-5.4-mini",
            context=_context(request=400_000, response=128_000),
            prices=_prices(request=0.75, response=4.5, cache_read=0.075),
        ),
        replace(
            legacy,
            model_id="gpt-5.4-nano",
            context=_context(request=400_000, response=128_000),
            prices=_prices(request=0.2, response=1.25, cache_read=0.02),
        ),
        replace(
            legacy,
            model_id="gpt-5.3-codex",
            context=_context(request=400_000, response=128_000),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
        ),
        # No `*-chat-latest` row. Those aliases are listed by `/v1/models` but
        # rejected by `/v1/responses` with `model_not_found` (verified for
        # gpt-5, gpt-5.2 and gpt-5.3 variants), and the Responses API is the
        # only one this provider speaks -- so a row for one is a model no
        # caller here can reach.
        replace(
            legacy,
            model_id="gpt-5.2",
            context=_context(request=400_000, response=128_000),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
        ),
        replace(
            legacy,
            model_id="o1",
            thinking_effort={"low", "medium", "high"},
            context=_context(request=200_000, response=100_000),
            prices=_prices(request=15.0, response=60.0, cache_read=7.5),
        ),
        replace(
            legacy,
            model_id="o3-mini",
            thinking_effort={"low", "medium", "high"},
            context=_context(request=200_000, response=100_000),
            prices=_prices(request=1.1, response=4.4, cache_read=0.55),
        ),
        replace(
            gpt4,
            model_id="gpt-4.1",
            context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
            prices=_prices(request=2.0, response=8.0, cache_read=0.5),
        ),
        replace(
            gpt4,
            model_id="gpt-4.1-mini",
            context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
            prices=_prices(request=0.4, response=1.6, cache_read=0.1),
        ),
        replace(
            gpt4,
            model_id="gpt-4.1-nano",
            context=_windowed(request=1_047_576, response=32_768, long=1_047_576),
            prices=_prices(request=0.1, response=0.4, cache_read=0.025),
        ),
        replace(
            gpt4,
            model_id="gpt-4o",
            context=_context(request=128_000, response=16_384),
            prices=_prices(request=2.5, response=10.0, cache_read=1.25),
        ),
        replace(
            gpt4,
            model_id="gpt-4o-mini",
            context=_context(request=128_000, response=16_384),
            prices=_prices(request=0.15, response=0.6, cache_read=0.075),
        ),
        replace(
            gpt4,
            model_id="gpt-4-turbo",
            context=_context(request=128_000, response=4_096),
            prices=_prices(request=10.0, response=30.0),
        ),
        replace(
            gpt4,
            model_id="gpt-4",
            context=_context(request=8_192, response=8_192),
            prices=_prices(request=30.0, response=60.0),
        ),
    )
    catalog = {row.model_id: row for row in rows}
    return MappingProxyType(
        {
            "default": catalog["astra-6"],
            "utility": catalog["luna-6"],
            **catalog,
        },
    )


def reasoning_effort(
    effort: ThinkingEffort,
    *,
    model_id: str,
) -> Literal["none", "low", "medium", "high", "xhigh", "max"]:
    """Map a selected effort to the Responses vocabulary.

    Args:
      effort: Selected catalog effort.
      model_id: Base model id.

    Returns:
      wire: Responses reasoning effort.

    """
    row = models().get(model_id)
    if row is None:
        row = next(
            (
                candidate
                for candidate in models().values()
                if candidate.wire_model_id == model_id
            ),
            None,
        )
    canonical_id = row.model_id if row is not None else model_id
    if effort == "min":
        return "none" if canonical_id.endswith("-5.6") else "low"
    if effort == "none" and canonical_id == "astra-6":
        return "low"
    return effort


def api() -> ModelCapability:
    """Return what the Responses API adds on top of a model row.

    One model, two authentication modes::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        service_tier={"auto", "default", "flex", "priority"},
    )


def subscription() -> ModelCapability:
    """Return what Codex billing narrows a model row to.

    One model, two authentication modes::

        models()["gpt-5.6"] & api()          # API key: all four tiers
        models()["gpt-5.6"] & subscription() # Codex: priority only, OAuth

    Returns:
      capability: The subscription transport's restrictions.

    """
    # ``auto`` stays selectable: ``/fast`` picks ``priority``, but an
    # unqualified request still sends the default, so dropping it would make
    # ``ModelSettings()`` invalid on every Codex model.
    return ModelCapability(
        thinking_effort={"none", "min", "low", "medium", "high", "xhigh", "max"},
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text", "redacted"},
        service_tier={"auto", "priority"},
        account_auth=True,
    )


@cache
def subscription_models() -> Mapping[str, ModelCapability]:
    """Return catalog rows clamped to the ChatGPT backend contract.

    Returns:
      models: Subscription-visible rows with only their usable context.

    """
    rows: dict[str, ModelCapability] = {}
    for name, capability in models().items():
        limits = capability.context[""]
        rows[name] = replace(
            capability,
            context=MappingProxyType(
                {
                    "": replace(
                        limits,
                        max_request_tokens=min(limits.max_request_tokens, 272_000),
                        max_response_tokens=min(limits.max_response_tokens, 32_000),
                    ),
                },
            ),
        )
    return MappingProxyType(rows)


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
    *,
    request: int,
    response: int,
    gpt56_images: bool = False,
) -> Mapping[ContextTag, ModelLimits]:
    """One untagged context for a model with no smaller-window selection."""
    return MappingProxyType(
        {"": _one(request=request, response=response, gpt56_images=gpt56_images)},
    )


def _windowed(
    *,
    request: int,
    response: int,
    long: int,
    gpt56_images: bool = False,
) -> Mapping[ContextTag, ModelLimits]:
    """Default to the full window; ``+272k`` selects the smaller cap."""
    context: dict[ContextTag, ModelLimits] = {
        "": _one(request=long, response=response, gpt56_images=gpt56_images),
    }
    if request < long:
        context["+272k"] = _one(
            request=request,
            response=response,
            gpt56_images=gpt56_images,
        )
    return MappingProxyType(context)


def _prices(
    *,
    request: float,
    response: float,
    cache_write: float = 0.0,
    cache_read: float = 0.0,
    two_tier: bool = False,
) -> PriceCatalog:
    """USD per million tokens, plus the >272K surcharge row when two-tier."""
    # Prompts above 272K input tokens bill at 2x input / 1.5x output for the
    # whole request on gpt-5.4 and later. The untagged id exposes the full
    # window; ``+272k`` keeps requests below the surcharge threshold.
    two_tier_floor = 272_000
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=cache_write,
            cache_read=cache_read,
        ),
    }
    if two_tier:
        # The >272K surcharge applies the input multiplier to all three input
        # pools and the output multiplier to the whole response.
        rows[PriceCatalogProduct(min_request_tokens=two_tier_floor)] = TokenPrice(
            request=request * 2.0,
            response=response * 1.5,
            cache_write=cache_write * 2.0,
            cache_read=cache_read * 2.0,
        )
    return PriceCatalog(rows)
