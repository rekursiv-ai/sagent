"""Tests for ``types.providers``: model-id resolution."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingCapability,
)
from sagent.types.cost import PriceCatalog, PriceCatalogProduct, TokenPrice
from sagent.types.providers import (
    ModelCatalog,
    UnknownModelError,
    UnsupportedTagError,
)


def _opus() -> ModelCapability:
    return ModelCapability(
        model_id="opus-4.8",
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000, max_image_bytes=5_000_000),
                "+1m": ModelLimits(
                    max_request_tokens=1_000_000,
                    max_image_bytes=5_000_000,
                ),
            },
        ),
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(request=5.0),
                PriceCatalogProduct(service_tier="priority"): TokenPrice(request=15.0),
            },
        ),
        thinking=ThinkingCapability(effort={"none", "max"}),
        service_tier={"auto", "default", "priority"},
    )


def _cli() -> ModelCapability:
    return ModelCapability(
        thinking=ThinkingCapability(effort={"none"}, output={"none", "text"}),
        manage_context_server_side={True},
    )


def resolve(
    model_id: str,
    *,
    models: dict[str, ModelCapability],
    transport: ModelCapability,
) -> tuple[ModelCapability, ModelSettings]:
    """Resolve through the sole public catalog API."""
    return ModelCatalog(rows=models, transport=transport).resolve(model_id)


def test_resolve_returns_capability_and_settings_as_peers() -> None:
    capability, settings = resolve(
        "opus-4.8+1m",
        models={"opus-4.8": _opus()},
        transport=ModelCapability(),
    )
    assert settings.context == "+1m"
    assert capability.context.keys() == {"", "+1m"}


def test_resolve_keeps_the_whole_context_table() -> None:
    _, settings = resolve(
        "opus-4.8",
        models={"opus-4.8": _opus()},
        transport=ModelCapability(),
    )
    assert settings.limits.max_request_tokens == 200_000
    wide = ModelSettings(capability=settings.capability, context="+1m")
    assert wide.limits.max_request_tokens == 1_000_000


def test_resolve_meets_the_transport() -> None:
    capability, _ = resolve(
        "opus-4.8",
        models={"opus-4.8": _opus()},
        transport=_cli(),
    )
    assert capability.thinking.effort == frozenset({"none"})
    assert capability.manage_context_server_side == frozenset({True})


def test_resolve_never_grants_what_the_row_lacks() -> None:
    capability, _ = resolve(
        "opus-4.8",
        models={"opus-4.8": _opus()},
        transport=ModelCapability(
            thinking=ThinkingCapability(
                effort={"none", "min", "low", "medium", "high", "max"},
            ),
        ),
    )
    assert capability.thinking.effort == frozenset({"none", "max"})


def test_resolve_rejects_a_context_the_model_lacks() -> None:
    with pytest.raises(UnsupportedTagError, match="no \\+200k context"):
        _ = resolve(
            "opus-4.8+200k",
            models={"opus-4.8": _opus()},
            transport=ModelCapability(),
        )


@pytest.mark.parametrize(
    "model_id",
    ["opus-4.8+1m+200k", "opus-4.8+200k+1m"],
)
def test_resolve_rejects_conflicting_context_tags(model_id: str) -> None:
    with pytest.raises(UnsupportedTagError, match="conflicting contexts"):
        _ = resolve(
            model_id,
            models={"opus-4.8": _opus()},
            transport=ModelCapability(),
        )


def test_resolve_accepts_explicit_1m_when_the_default_is_already_1m() -> None:
    row = replace(
        _opus(),
        context=MappingProxyType(
            {
                "": ModelLimits(
                    max_request_tokens=1_000_000,
                    max_response_tokens=128_000,
                ),
            },
        ),
    )
    capability, settings = resolve(
        "opus-4.8+1m",
        models={row.model_id: row},
        transport=ModelCapability(),
    )
    assert settings.context == "+1m"
    assert settings.limits == capability.context[""]


def test_a_transport_cannot_remove_a_context_window() -> None:
    """Windows are the model's; a transport restricts knobs, not physics."""
    narrow = ModelCapability(context=MappingProxyType({"": ModelLimits()}))
    _, settings = resolve(
        "opus-4.8+1m",
        models={"opus-4.8": _opus()},
        transport=narrow,
    )
    assert settings.limits.max_request_tokens == 1_000_000


def test_resolve_names_the_known_models_on_a_miss() -> None:
    with pytest.raises(UnknownModelError, match=r"opus-4\.8"):
        _ = resolve(
            "nope",
            models={"opus-4.8": _opus()},
            transport=ModelCapability(),
        )


def test_resolve_follows_a_catalog_alias() -> None:
    row = _opus()
    capability, _ = resolve(
        "utility",
        models={"utility": row, row.model_id: row},
        transport=ModelCapability(),
    )
    assert capability.model_id == "opus-4.8"


def test_resolve_accepts_a_vendor_id_as_a_compatibility_alias() -> None:
    row = replace(
        _opus(),
        model_id="opus-4.8",
        wire_model_id="claude-opus-4-8",
    )
    capability, _ = resolve(
        "claude-opus-4-8",
        models={row.model_id: row},
        transport=ModelCapability(),
    )
    assert capability.model_id == "opus-4.8"
    assert capability.wire_model_id == "claude-opus-4-8"


def test_resolve_settings_carry_the_capability_they_were_narrowed_from() -> None:
    """Construction validates, so the settings could not exist otherwise."""
    capability, settings = resolve(
        "opus-4.8+1m",
        models={"opus-4.8": _opus()},
        transport=_cli(),
    )
    assert settings.capability == capability


def test_model_catalog_owns_ids_and_resolution() -> None:
    row = _opus()
    catalog = ModelCatalog(
        rows=MappingProxyType({row.model_id: row}),
        transport=_cli(),
    )
    assert tuple(catalog.rows) == ("opus-4.8",)
    capability, _ = catalog.resolve("opus-4.8")
    assert capability.model_id == "opus-4.8"


def test_model_catalog_snapshots_and_freezes_rows() -> None:
    source = {"opus-4.8": _opus()}
    catalog = ModelCatalog(rows=source, transport=_cli())
    source.clear()
    assert "opus-4.8" in catalog.rows
    assert type(catalog.rows) is MappingProxyType


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
