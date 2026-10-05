"""DashScope/Qwen model catalog, expressed as ``ModelCapability`` rows.

Sources (international / Singapore deployment, list prices, 2026-10-03):
  https://www.alibabacloud.com/help/en/model-studio/model-pricing
  https://www.alibabacloud.com/help/en/model-studio/text-generation-model
  https://www.alibabacloud.com/help/en/model-studio/deep-thinking
  https://www.alibabacloud.com/help/en/model-studio/context-cache
  https://www.alibabacloud.com/help/en/model-studio/vision

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared by ``openai.compatible()``. Model Studio
publishes max output only in its console, so every row keeps the 64K cap
it has always shipped.

Thinking axes: ``budget`` ``none`` is ``enable_thinking=false``; ``auto``
thinks at the model's default depth; ``fixed`` caps it with
``thinking_budget``. Rows with ``effort_as_level`` (Qwen 3.8) take the
effort as ``reasoning_effort`` instead, and reject ``thinking_budget``
alongside it. Effort ``none`` sends no depth knob at all.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.catalog.table import ModelTable
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
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["models", "thinking_budget"]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinking_budget`` token cap an effort sends.

    Only rows without ``effort_as_level`` send it, and only under a
    ``fixed`` budget; ``0`` is the off value an effort of ``none`` maps to.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    """
    match effort:
        case "none":
            return "0"
        case "min":
            return "1024"
        case "low":
            return "4096"
        case "medium":
            return "8192"
        case "high":
            return "16384"
        case "xhigh":
            return "20480"
        case "max":
            return "24576"


def models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    hybrid = ThinkingCapability(
        effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
        budget=frozenset({"none", "auto", "fixed"}),
        output=frozenset({"none", "text"}),
    )
    # Qwen 3.8 accepts ``reasoning_effort`` low / medium / xhigh; the other
    # spellings it folds onto these, so offering them would only alias.
    levels = ThinkingCapability(
        effort=frozenset({"none", "low", "medium", "xhigh"}),
        budget=frozenset({"none", "auto"}),
        output=frozenset({"none", "text"}),
    )
    # Images ride as base64 data URIs, capped at 20 MB as a STRING; base64
    # inflates 4/3, so raw bytes above ~15 MB would be rejected.
    vision = _limits(image_bytes=14_000_000)
    rows = (
        ModelCapability(
            model_id="qwen-max-3.8",
            wire_model_id="qwen3.8-max",
            context=vision,
            # Cache-hit rate unpublished (console only).
            prices=_prices((0, 2.0, 6.0), cache_hit=1.0),
            thinking=levels,
            effort_as_level=True,
        ),
        ModelCapability(
            model_id="qwen-flash-3.8",
            wire_model_id="qwen3.8-flash",
            context=vision,
            # Cache-hit rate unpublished (console only).
            prices=_prices((0, 0.15, 0.47), cache_hit=1.0),
            thinking=levels,
            effort_as_level=True,
        ),
        ModelCapability(
            model_id="qwen-27b-3.8",
            wire_model_id="qwen3.8-27b",
            context=_limits(),
            prices=_prices((0, 0.5, 3.0), cache_hit=0.2),
            thinking=levels,
            effort_as_level=True,
        ),
        ModelCapability(
            model_id="qwen-2400b-a95b-3.8",
            wire_model_id="qwen3.8-2.4t-a95b",
            context=_limits(),
            # Cache-hit rate unpublished (console only).
            prices=_prices((0, 2.0, 6.0), cache_hit=1.0),
            # "Supports thinking mode only": no ``enable_thinking=false``.
            thinking=replace(levels, budget=frozenset({"auto"})),
            effort_as_level=True,
        ),
        ModelCapability(
            model_id="qwen-max-3.7",
            wire_model_id="qwen3.7-max",
            context=vision,
            prices=_prices((0, 2.5, 7.5), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-plus-3.7",
            wire_model_id="qwen3.7-plus",
            context=vision,
            prices=_prices((0, 0.4, 1.6), (256_000, 1.2, 4.8), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-flash-3.7",
            wire_model_id="qwen3.7-flash",
            context=vision,
            prices=_prices(
                (0, 0.03, 0.13),
                (32_000, 0.10, 0.40),
                (256_000, 0.20, 0.80),
                cache_hit=0.2,
            ),
            thinking=hybrid,
        ),
        # The qwen3.6 rows support only EXPLICIT cache in Singapore, which
        # needs a ``cache_control`` marker this transport never writes.
        ModelCapability(
            model_id="qwen-max-3.6",
            wire_model_id="qwen3.6-max-preview",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 1.3, 7.8), (128_000, 2.0, 12.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-plus-3.6",
            wire_model_id="qwen3.6-plus",
            context=vision,
            prices=_prices((0, 0.5, 3.0), (256_000, 2.0, 6.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-flash-3.6",
            wire_model_id="qwen3.6-flash",
            context=vision,
            prices=_prices((0, 0.25, 1.5), (256_000, 1.0, 4.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        # ``-instruct`` and ``-coder`` reject the toggle, so every thinking
        # axis offers only its off value; their open-source deployments carry
        # no implicit cache.
        ModelCapability(
            model_id="qwen-235b-a22b-instruct-3.0",
            wire_model_id="qwen3-235b-a22b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.23, 0.92), cache_hit=1.0),
        ),
        ModelCapability(
            model_id="qwen-235b-a22b-thinking-3.0",
            wire_model_id="qwen3-235b-a22b-thinking-2507",
            context=_limits(max_tokens=262_144, output_tokens=32_768),
            prices=_prices((0, 0.23, 2.3), cache_hit=1.0),
            # These ids reject ``enable_thinking=false``.
            thinking=replace(
                hybrid,
                effort=hybrid.effort - {"none"},
                budget=hybrid.budget - {"none"},
            ),
        ),
        ModelCapability(
            model_id="qwen-30b-a3b-instruct-3.0",
            wire_model_id="qwen3-30b-a3b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.2, 0.8), cache_hit=1.0),
        ),
        ModelCapability(
            model_id="qwen-32b-3.0",
            wire_model_id="qwen3-32b",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.16, 0.64), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-coder-480b-a35b-3.0",
            wire_model_id="qwen3-coder-480b-a35b-instruct",
            context=_limits(max_tokens=262_144),
            prices=_prices(
                (0, 1.5, 7.5),
                (32_000, 2.7, 13.5),
                (128_000, 4.5, 22.5),
                cache_hit=1.0,
            ),
        ),
        # The unversioned aliases are named by the generation deep-thinking
        # files them under ("Qwen3 ... Commercial version"). ``qwen-max`` is
        # absent: no page states its generation.
        ModelCapability(
            model_id="qwen-plus-3.0",
            wire_model_id="qwen-plus",
            context=_limits(output_tokens=32_768),
            # Non-thinking card; thinking output bills $4 / $12, which no
            # ``PriceKey`` axis can select.
            prices=_prices((0, 0.4, 1.2), (256_000, 1.2, 3.6), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-turbo-3.0",
            wire_model_id="qwen-turbo",
            context=_limits(output_tokens=32_768),
            # Non-thinking card; thinking output bills $0.5, which no
            # ``PriceKey`` axis can select.
            prices=_prices((0, 0.05, 0.2), cache_hit=0.2),
            thinking=hybrid,
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        # text-generation-model: "Start with qwen3.7-plus ... To cut costs,
        # switch to qwen3.8-flash".
        roles={"default": "qwen-plus-3.7", "utility": "qwen-flash-3.8"},
    )


# Implicit cache creation bills as plain input, and this transport never writes the
# explicit marker, so a reported write bills at the input rate.
def _prices(*bands: tuple[int, float, float], cache_hit: float) -> PriceCatalog:
    """Return the published bands; each bills a whole prompt larger than its floor."""
    return PriceCatalog(
        {
            PriceKey("auto", floor): TokenPrice(
                request=request,
                response=response,
                cache_write=request,
                cache_write_1h=0.0,
                cache_read=request * cache_hit,
            )
            for floor, request, response in bands
        },
    )


def _limits(
    *,
    max_tokens: int = 1_000_000,
    output_tokens: int = 65_536,
    image_bytes: int = 0,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: DashScope ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
                max_image_bytes=image_bytes,
            ),
        },
    )
