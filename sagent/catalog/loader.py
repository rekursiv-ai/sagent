"""Load typed provider catalogs from YAML without importing data-named code."""

from datetime import date, datetime
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Final

import math

from treekle import from_plain

import yaml

from sagent.catalog.custom_typings import (
    CapabilityProfile,
    CatalogDocument,
    ModelRow,
    ProviderName,
)
from sagent.catalog.table import EXACT_NAME, ModelTable
from sagent.types.capability import ContextTag, ModelCapability, ModelLimits
from sagent.types.cost import PriceCatalog, PriceKey, ServiceTier, TokenPrice


_THIS: Final = Path(__file__).resolve()


@cache
def models(provider: ProviderName) -> ModelTable:
    """Return a provider's typed catalog, cached after its first load."""
    from_plain(provider, ProviderName)
    return load_catalog(_THIS.with_name("assets") / f"{provider}.yaml")


def load_catalog(path: Path) -> ModelTable:
    """Read a provider document using the schema in ``custom_typings``.

    Profile values are defaults for model fields, not recursive patches.
    Unknown fields, duplicate keys, aliases, and invalid types are rejected,
    as are names other than ``family-major.minor``, families listed
    oldest-first, and tiers that price no base card.

    Args:
      path: Provider YAML file.

    Returns:
      table: Validated models and their role resolution table.

    """
    text = path.read_text()
    _validate_yaml(yaml.compose(text, Loader=yaml.SafeLoader), seen=set())
    document = from_plain(yaml.safe_load(text), CatalogDocument)
    for layer in (document.reference, *document.profiles.values()):
        _require_whole_thinking(layer)
    from_plain(document.reference, CapabilityProfile)
    for profile in document.profiles.values():
        from_plain(document.reference | profile, CapabilityProfile)
    rows = tuple(_model(raw, document=document) for raw in document.models)
    return ModelTable(rows=rows, roles=document.roles)


# Nested objects replace whole, so an omitted axis would become ``none`` rather than
# inherit from the layer below.
def _require_whole_thinking(layer: dict[str, object]) -> None:
    """Reject a ``thinking`` override that omits an axis."""
    thinking = layer.get("thinking")
    if isinstance(thinking, dict) and thinking.keys() != {"effort", "budget", "output"}:
        raise ValueError("Catalog thinking must state effort, budget, output")


def _model(raw: dict[str, object], *, document: CatalogDocument) -> ModelCapability:
    values = dict(raw)
    _require_whole_thinking(values)
    profile = from_plain(values.pop("profile", ""), str)
    if profile and profile not in document.profiles:
        raise ValueError(f"Unknown catalog profile {profile!r}")
    values = document.reference | document.profiles.get(profile, {}) | values
    row = from_plain(values, ModelRow)
    if EXACT_NAME.fullmatch(row.model_id) is None:
        raise ValueError(f"Model id {row.model_id!r} is not family-major.minor")
    context = _contexts(row)
    if (
        not row.prices
        or not row.thinking.effort
        or not row.thinking.budget
        or not row.thinking.output
        or row.approx_chars_per_token <= 0
        or not math.isfinite(row.approx_chars_per_token)
    ):
        raise ValueError(f"Unusable catalog model {row.model_id!r}")
    catalog = _prices(row)
    return ModelCapability(
        model_id=row.model_id,
        wire_model_id=row.wire_model_id,
        knowledge_cutoff=row.knowledge_cutoff,
        approx_chars_per_token=row.approx_chars_per_token,
        context=MappingProxyType(context),
        thinking=row.thinking,
        effort_as_level=row.effort_as_level,
        prices=catalog,
        service_tier=catalog.service_tiers,
    )


def _contexts(row: ModelRow) -> dict[ContextTag, ModelLimits]:
    if any(tag not in {"max", "min"} for tag in row.context):
        raise ValueError("Catalog contexts require max/min presets")
    if row.default_context not in row.context:
        raise ValueError(f"Model {row.model_id!r} requires a default context")
    context: dict[ContextTag, ModelLimits] = {}
    numeric_tags: dict[int, ContextTag] = {
        200_000: "+200k",
        272_000: "+272k",
        1_000_000: "+1m",
    }
    if (
        "max" in row.context
        and "min" in row.context
        and (
            row.context["max"].max_request_tokens
            < row.context["min"].max_request_tokens
        )
    ):
        raise ValueError(f"Invalid catalog context ordering in {row.model_id!r}")
    for preset, limits in row.context.items():
        if (
            min(
                limits.max_request_tokens,
                limits.max_response_tokens,
                limits.max_request_bytes,
                limits.max_image_edge_px,
                limits.max_image_bytes,
            )
            < 0
        ):
            raise ValueError(f"Invalid catalog context limits in {row.model_id!r}")
        if preset == row.default_context:
            context[""] = limits
        else:
            tag: ContextTag = "+max" if preset == "max" else "+min"
            context[numeric_tags.get(limits.max_request_tokens, tag)] = limits
    return context


def _prices(row: ModelRow) -> PriceCatalog:
    prices: dict[PriceKey, TokenPrice] = {}
    for card in row.prices:
        rates = (
            card.request,
            card.response,
            card.cache_write,
            card.cache_write_1h,
            card.cache_read,
        )
        if (
            card.tokens_per_unit <= 0
            or card.effective_prompt_tokens < 0
            or card.service_tier == "default"
            or isinstance(card.effective_date, datetime)
            or any(rate < 0 or not math.isfinite(rate) for rate in rates)
        ):
            raise ValueError(f"Invalid catalog price in {row.model_id!r}")
        key = PriceKey(
            card.service_tier,
            card.effective_prompt_tokens,
            date.fromisoformat(card.effective_date)
            if isinstance(card.effective_date, str)
            else card.effective_date,
        )
        if key in prices:
            raise ValueError(f"Duplicate price key {key!r} in {row.model_id!r}")
        prices[key] = TokenPrice(
            request=card.request,
            response=card.response,
            cache_write=card.cache_write,
            cache_write_1h=card.cache_write_1h,
            cache_read=card.cache_read,
            tokens_per_unit=card.tokens_per_unit,
        )
    # ``rate`` raises for a request no card covers; only an undated band-0 card
    # covers every prompt on every day.
    tiers: set[ServiceTier] = {key.service_tier for key in prices}
    for tier in tiers:
        if PriceKey(tier) not in prices:
            raise ValueError(
                f"Tier {tier!r} in {row.model_id!r} has no base card"
                " (no prompt band, no effective date)",
            )
    return PriceCatalog(prices)


def _validate_yaml(node: yaml.Node | None, *, seen: set[int]) -> None:
    if node is None:
        return
    if id(node) in seen:
        raise ValueError("YAML aliases are not supported in catalogs")
    seen.add(id(node))
    if isinstance(node, yaml.MappingNode):
        keys: set[str] = set()
        for key, value in node.value:
            if (
                not isinstance(key, yaml.ScalarNode)
                or key.tag != "tag:yaml.org,2002:str"
            ):
                raise ValueError("Catalog mapping keys must be strings")
            if key.value in keys:
                raise ValueError(f"Duplicate YAML key {key.value!r}")
            keys.add(key.value)
            _validate_yaml(value, seen=seen)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            _validate_yaml(value, seen=seen)
