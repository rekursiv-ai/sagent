"""OpenAI model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - ModelLimits: https://developers.openai.com/api/docs/models/<model>
  - Pricing: https://developers.openai.com/api/docs/pricing
  - Images: https://developers.openai.com/api/docs/guides/images-vision

:func:`models` is the API-key view. :func:`subscription_models` projects the
subscription request and response limits; transport capabilities then narrow
either view with ``&``, which can only remove.
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
    ThinkingCapability,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    ServiceTier,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "api",
    "compatible",
    "models",
    "reasoning_effort",
    "served_tier",
    "subscription",
    "subscription_models",
]


def compatible() -> ModelCapability:
    """Return the common Chat Completions transport capability.

    Returns:
      capability: Restrictions shared by OpenAI-compatible transports.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto"}),
        manage_context_server_side=frozenset({False}),
    )


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts declared on ``API`` / ``SUBSCRIPTION``, since ``&`` can only
# remove. Every reasoning model takes an auto budget and returns readable text.
#
# Every card is the vendor's published table (standard, flex, and fast tabs),
# verified 2026-09-24; a model missing from a tab does not offer that tier.
# Sol 5.6's rate is promotional "at least through November 21, 2026" and needs
# rechecking then.
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
    # The reference row: every other model states only how it differs from it.
    # Every GPT-6 rule here was measured on this id alone, so a later GPT-6
    # whose contract differs does not inherit a limit nothing verified for it.
    default = ModelCapability(
        model_id="astra-6",
        wire_model_id="gpt-6-astra",
        knowledge_cutoff="April 30, 2026",
        approx_chars_per_token=3.71,
        context=_limits(),
        prices=_prices(
            {
                "auto": _card(
                    request=10.0,
                    response=50.0,
                    cache_write=12.5,
                    cache_read=1.0,
                ),
                "flex": _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_read=0.5,
                ),
                "priority": _card(
                    request=20.0,
                    response=100.0,
                    cache_write=25.0,
                    cache_read=2.0,
                ),
            },
            long_context=True,
        ),
        # Measured against the live API 2026-09-04: ``reasoning.effort`` takes
        # low..max but rejects ``none`` and ``minimal``.
        thinking=ThinkingCapability(
            effort=frozenset({"low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text"}),
        ),
    )
    # The rest of GPT-6 and all of GPT-5.6 also take ``none`` and ``min``.
    full = replace(
        default.thinking,
        effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
    )
    # Pre-5.6: no ``min``, no ``max``.
    legacy = replace(
        default.thinking,
        effort=frozenset({"none", "low", "medium", "high", "xhigh"}),
    )
    rows = (
        default,
        replace(
            default,
            model_id="sol-6",
            wire_model_id="gpt-6-sol",
            knowledge_cutoff="April 20, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "flex": _card(
                        request=1.0,
                        response=5.0,
                        cache_write=1.25,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="luna-6",
            wire_model_id="gpt-6-luna",
            knowledge_cutoff="May 18, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=0.1,
                        response=0.5,
                        cache_write=0.125,
                        cache_read=0.01,
                    ),
                    "flex": _card(
                        request=0.05,
                        response=0.25,
                        cache_write=0.0625,
                        cache_read=0.005,
                    ),
                    "priority": _card(
                        request=0.2,
                        response=1.0,
                        cache_write=0.25,
                        cache_read=0.02,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="sol-5.6",
            wire_model_id="gpt-5.6-sol",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                    "flex": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "priority": _card(
                        request=8.0,
                        response=40.0,
                        cache_write=10.0,
                        cache_read=0.8,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        # Absent from ``GET /v1/models`` yet serves ``POST /v1/responses``
        # normally. Listing is an entitlement view, not the model set, so a
        # row must be dropped only on ``model_not_found`` from a real call.
        replace(
            default,
            model_id="gpt-5.6",
            wire_model_id="gpt-5.6",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=4.0,
                        response=20.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                    "flex": _card(
                        request=2.0,
                        response=10.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "priority": _card(
                        request=8.0,
                        response=40.0,
                        cache_write=10.0,
                        cache_read=0.8,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="luna-5.6",
            wire_model_id="gpt-5.6-luna",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=0.2,
                        response=1.2,
                        cache_write=0.25,
                        cache_read=0.02,
                    ),
                    "flex": _card(
                        request=0.1,
                        response=0.6,
                        cache_write=0.125,
                        cache_read=0.01,
                    ),
                    "priority": _card(
                        request=0.4,
                        response=2.4,
                        cache_write=0.5,
                        cache_read=0.04,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="terra-5.6",
            wire_model_id="gpt-5.6-terra",
            knowledge_cutoff="February 16, 2026",
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=12.0,
                        cache_write=2.5,
                        cache_read=0.2,
                    ),
                    "flex": _card(
                        request=1.0,
                        response=6.0,
                        cache_write=1.25,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=4.0,
                        response=24.0,
                        cache_write=5.0,
                        cache_read=0.4,
                    ),
                },
                long_context=True,
            ),
            thinking=full,
        ),
        replace(
            default,
            model_id="gpt-5.5",
            wire_model_id="gpt-5.5",
            knowledge_cutoff=None,
            context=_limits(max_tokens=1_000_000, patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=5.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                    "flex": _card(
                        request=2.5,
                        response=15.0,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                    "priority": _card(
                        request=12.5,
                        response=75.0,
                        cache_write=0.0,
                        cache_read=1.25,
                    ),
                },
                long_context=True,
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-5.5-pro",
            wire_model_id="gpt-5.5-pro",
            knowledge_cutoff=None,
            context=_limits(patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=180.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                    "flex": _card(
                        request=15.0,
                        response=90.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
                long_context=True,
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"medium", "high", "xhigh"}),
            ),
        ),
        replace(
            default,
            model_id="gpt-5.4",
            wire_model_id="gpt-5.4",
            knowledge_cutoff=None,
            context=_limits(patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.5,
                        response=15.0,
                        cache_write=0.0,
                        cache_read=0.25,
                    ),
                    "flex": _card(
                        request=1.25,
                        response=7.5,
                        cache_write=0.0,
                        cache_read=0.13,
                    ),
                    "priority": _card(
                        request=5.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                },
                long_context=True,
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-5.4-pro",
            wire_model_id="gpt-5.4-pro",
            knowledge_cutoff=None,
            context=_limits(patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=180.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                    "flex": _card(
                        request=15.0,
                        response=90.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
                long_context=True,
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"medium", "high", "xhigh"}),
            ),
        ),
        replace(
            default,
            model_id="gpt-5.4-mini",
            wire_model_id="gpt-5.4-mini",
            knowledge_cutoff=None,
            context=_limits(max_tokens=400_000, windowed=False, patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.75,
                        response=4.5,
                        cache_write=0.0,
                        cache_read=0.075,
                    ),
                    "flex": _card(
                        request=0.375,
                        response=2.25,
                        cache_write=0.0,
                        cache_read=0.0375,
                    ),
                    "priority": _card(
                        request=1.5,
                        response=9.0,
                        cache_write=0.0,
                        cache_read=0.15,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-5.4-nano",
            wire_model_id="gpt-5.4-nano",
            knowledge_cutoff=None,
            context=_limits(max_tokens=400_000, windowed=False, patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.2,
                        response=1.25,
                        cache_write=0.0,
                        cache_read=0.02,
                    ),
                    "flex": _card(
                        request=0.1,
                        response=0.625,
                        cache_write=0.0,
                        cache_read=0.01,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="gpt-5.3-codex",
            wire_model_id="gpt-5.3-codex",
            knowledge_cutoff=None,
            context=_limits(max_tokens=400_000, windowed=False, patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.75,
                        response=14.0,
                        cache_write=0.0,
                        cache_read=0.175,
                    ),
                    "priority": _card(
                        request=3.5,
                        response=28.0,
                        cache_write=0.0,
                        cache_read=0.35,
                    ),
                },
            ),
            thinking=legacy,
        ),
        # No `*-chat-latest` row. Those aliases are listed by `/v1/models` but
        # rejected by `/v1/responses` with `model_not_found` (verified for
        # gpt-5, gpt-5.2 and gpt-5.3 variants), and the Responses API is the
        # only one this provider speaks -- so a row for one is a model no
        # caller here can reach.
        replace(
            default,
            model_id="gpt-5.2",
            wire_model_id="gpt-5.2",
            knowledge_cutoff=None,
            context=_limits(max_tokens=400_000, windowed=False, patch_images=False),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.75,
                        response=14.0,
                        cache_write=0.0,
                        cache_read=0.175,
                    ),
                    "flex": _card(
                        request=0.875,
                        response=7.0,
                        cache_write=0.0,
                        cache_read=0.0875,
                    ),
                    "priority": _card(
                        request=3.5,
                        response=28.0,
                        cache_write=0.0,
                        cache_read=0.35,
                    ),
                },
            ),
            thinking=legacy,
        ),
        replace(
            default,
            model_id="o1",
            wire_model_id="o1",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=200_000,
                output_tokens=100_000,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=15.0,
                        response=60.0,
                        cache_write=0.0,
                        cache_read=7.5,
                    ),
                },
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"low", "medium", "high"}),
            ),
        ),
        replace(
            default,
            model_id="o3-mini",
            wire_model_id="o3-mini",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=200_000,
                output_tokens=100_000,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=1.1,
                        response=4.4,
                        cache_write=0.0,
                        cache_read=0.55,
                    ),
                },
            ),
            thinking=replace(
                default.thinking,
                effort=frozenset({"low", "medium", "high"}),
            ),
        ),
        # No reasoning knob on the 4-x generation, so every thinking axis
        # offers only its off value.
        replace(
            default,
            model_id="gpt-4.1",
            wire_model_id="gpt-4.1",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.0,
                        response=8.0,
                        cache_write=0.0,
                        cache_read=0.5,
                    ),
                    "priority": _card(
                        request=3.5,
                        response=14.0,
                        cache_write=0.0,
                        cache_read=0.875,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4.1-mini",
            wire_model_id="gpt-4.1-mini",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.4,
                        response=1.6,
                        cache_write=0.0,
                        cache_read=0.1,
                    ),
                    "priority": _card(
                        request=0.7,
                        response=2.8,
                        cache_write=0.0,
                        cache_read=0.175,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4.1-nano",
            wire_model_id="gpt-4.1-nano",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=1_047_576,
                output_tokens=32_768,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.1,
                        response=0.4,
                        cache_write=0.0,
                        cache_read=0.025,
                    ),
                    "priority": _card(
                        request=0.2,
                        response=0.8,
                        cache_write=0.0,
                        cache_read=0.05,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4o",
            wire_model_id="gpt-4o",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=128_000,
                output_tokens=16_384,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=2.5,
                        response=10.0,
                        cache_write=0.0,
                        cache_read=1.25,
                    ),
                    "priority": _card(
                        request=4.25,
                        response=17.0,
                        cache_write=0.0,
                        cache_read=2.125,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4o-mini",
            wire_model_id="gpt-4o-mini",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=128_000,
                output_tokens=16_384,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=0.15,
                        response=0.6,
                        cache_write=0.0,
                        cache_read=0.075,
                    ),
                    "priority": _card(
                        request=0.25,
                        response=1.0,
                        cache_write=0.0,
                        cache_read=0.125,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4-turbo",
            wire_model_id="gpt-4-turbo",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=128_000,
                output_tokens=4_096,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=10.0,
                        response=30.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
        replace(
            default,
            model_id="gpt-4",
            wire_model_id="gpt-4",
            knowledge_cutoff=None,
            context=_limits(
                max_tokens=8_192,
                output_tokens=8_192,
                windowed=False,
                patch_images=False,
            ),
            prices=_prices(
                {
                    "auto": _card(
                        request=30.0,
                        response=60.0,
                        cache_write=0.0,
                        cache_read=0.0,
                    ),
                },
            ),
            thinking=ThinkingCapability(),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    # Rows run newest-first within each family, so the first match is the latest
    # and a new release moves its alias without an edit here.
    latest = {
        alias: next(row for row in rows if row.model_id.startswith(f"{family}-"))
        for alias, family in (
            ("default", "astra"),
            ("utility", "luna"),
        )
    }
    return MappingProxyType(latest | {row.model_id: row for row in rows})


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
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
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
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "priority"}),
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


def served_tier(reported: str | None) -> ServiceTier:
    """Map the ``service_tier`` a response reports to the tier it bills at.

    Args:
      reported: The response's ``service_tier``; absent means standard.

    Returns:
      tier: Catalog tier whose card priced the request.

    Raises:
      ValueError: A tier this catalog cannot price.

    """
    match reported:
        case None | "auto" | "default":
            return "auto"
        case "flex":
            return "flex"
        case "priority" | "fast":
            return "priority"
        case _:
            raise ValueError(f"unpriced OpenAI service_tier {reported!r}")


def _card(
    *,
    request: float,
    response: float,
    cache_write: float,
    cache_read: float,
) -> TokenPrice:
    """Return one published price-table row; OpenAI has one cache-write lifetime."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


# "Prompts with >272K input tokens are priced at 2x input and 1.5x output for
# the full request" -- every model page states it as this ratio, and the tables
# publish the long column only for GPT-6, where it matches.
def _prices(
    tiers: Mapping[ServiceTier, TokenPrice],
    *,
    long_context: bool = False,
) -> PriceCatalog:
    """Return each tier's published card, plus its >272K band when it has one."""
    cards = {PriceKey(tier): card for tier, card in tiers.items()}
    if long_context:
        cards |= {
            PriceKey(tier, 272_000): replace(
                card,
                request=card.request * 2.0,
                cache_write=card.cache_write * 2.0,
                cache_read=card.cache_read * 2.0,
                response=card.response * 1.5,
            )
            for tier, card in tiers.items()
        }
    return PriceCatalog(cards)


# GPT-5.6 and later bill images by 32x32 patch and take a far larger body;
# earlier models tile ``detail:high`` from a 2048px square under a 20MB cap.
def _limits(
    *,
    max_tokens: int = 1_050_000,
    output_tokens: int = 128_000,
    windowed: bool = True,
    patch_images: bool = True,
) -> Mapping[ContextTag, ModelLimits]:
    """Serve ``max_tokens`` by default; ``+272k`` selects the cap below the surcharge."""
    if patch_images:
        limits = ModelLimits(
            max_request_tokens=max_tokens,
            max_response_tokens=output_tokens,
            max_request_bytes=512 * 1024 * 1024,
        )
    else:
        limits = ModelLimits(
            max_request_tokens=max_tokens,
            max_response_tokens=output_tokens,
            max_request_bytes=20 * 1024 * 1024,
            max_image_edge_px=2048,
            max_image_bytes=20 * 1024 * 1024,
        )
    context: dict[ContextTag, ModelLimits] = {"": limits}
    if windowed:
        context["+272k"] = replace(limits, max_request_tokens=272_000)
    return MappingProxyType(context)
