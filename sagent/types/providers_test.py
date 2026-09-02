"""Tests for ``types.providers``: model-id resolution."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.types.cost import PriceCatalog, PriceCatalogProduct, TokenPrice
from sagent.types.providers import (
    UnknownModelError,
    UnsupportedTagError,
    resolve,
)


def _opus() -> ModelCapability:
    return ModelCapability(
        model_id="claude-opus-4-8",
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000, max_image_bytes=5_000_000),
                "+1m": ModelLimits(
                    max_request_tokens=1_000_000, max_image_bytes=5_000_000
                ),
            }
        ),
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(request=5.0),
                PriceCatalogProduct(service_tier="priority"): TokenPrice(request=15.0),
            }
        ),
        thinking_effort=frozenset({"none", "max"}),
        service_tier=frozenset({"auto", "default", "priority"}),
    )


def _cli() -> ModelCapability:
    return ModelCapability(
        thinking_effort=frozenset({"none"}),
        thinking_output=frozenset({"none", "text"}),
        manage_context_server_side=frozenset({True}),
    )


def test_resolve_returns_capability_and_settings_as_peers() -> None:
    capability, settings = resolve(
        "claude-opus-4-8+1m",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=ModelCapability(),
    )
    assert isinstance(capability, ModelCapability)
    assert isinstance(settings, ModelSettings)
    assert settings.context == "+1m"
    assert capability.context.keys() == {"", "+1m"}


def test_resolve_keeps_the_whole_context_table() -> None:
    _, settings = resolve(
        "claude-opus-4-8",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=ModelCapability(),
    )
    assert settings.limits.max_request_tokens == 200_000
    wide = ModelSettings(capability=settings.capability, context="+1m")
    assert wide.limits.max_request_tokens == 1_000_000


def test_resolve_meets_the_transport() -> None:
    capability, _ = resolve(
        "claude-opus-4-8",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=_cli(),
    )
    assert capability.thinking_effort == frozenset({"none"})
    assert capability.manage_context_server_side == frozenset({True})


def test_resolve_never_grants_what_the_row_lacks() -> None:
    capability, _ = resolve(
        "claude-opus-4-8",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=ModelCapability(
            thinking_effort=frozenset({"none", "min", "low", "medium", "high", "max"})
        ),
    )
    assert capability.thinking_effort == frozenset({"none", "max"})


def test_resolve_rejects_a_context_the_model_lacks() -> None:
    with pytest.raises(UnsupportedTagError, match="no \\+200k context"):
        _ = resolve(
            "claude-opus-4-8+200k",
            models={"claude-opus-4-8": _opus()},
            roles={},
            transport=ModelCapability(),
        )


def test_a_transport_cannot_remove_a_context_window() -> None:
    """Windows are the model's; a transport restricts knobs, not physics."""
    narrow = ModelCapability(context=MappingProxyType({"": ModelLimits()}))
    _, settings = resolve(
        "claude-opus-4-8+1m",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=narrow,
    )
    assert settings.limits.max_request_tokens == 1_000_000


def test_resolve_names_the_known_models_on_a_miss() -> None:
    with pytest.raises(UnknownModelError, match="claude-opus-4-8"):
        _ = resolve(
            "nope",
            models={"claude-opus-4-8": _opus()},
            roles={},
            transport=ModelCapability(),
        )


def test_resolve_follows_a_role() -> None:
    _, settings = resolve(
        "utility",
        models={"claude-opus-4-8": _opus()},
        roles={"utility": "claude-opus-4-8+1m"},
        transport=ModelCapability(),
    )
    assert settings.context == "+1m"


def test_resolve_settings_carry_the_capability_they_were_narrowed_from() -> None:
    """Construction validates, so the settings could not exist otherwise."""
    capability, settings = resolve(
        "claude-opus-4-8+1m",
        models={"claude-opus-4-8": _opus()},
        roles={},
        transport=_cli(),
    )
    assert settings.capability == capability


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
