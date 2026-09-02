"""Tests for ``types.capability``: the meet, and validation on assignment."""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType
from typing import Final

import pytest

from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
)
from sagent.types.cost import PriceCatalog, PriceCatalogProduct, TokenPrice


def _row() -> ModelCapability:
    return ModelCapability(
        model_id="m",
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000),
                "+1m": ModelLimits(max_request_tokens=1_000_000),
            }
        ),
        prices=PriceCatalog({PriceCatalogProduct(): TokenPrice(request=5.0)}),
        thinking_effort=frozenset({"none", "low", "high"}),
        thinking_budget=frozenset({"none", "auto"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        service_tier=frozenset({"auto", "default", "priority"}),
        cache_ttl_sec=3600.0,
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
    )


# ---- the meet --------------------------------------------------------------


def test_every_field_is_default_constructible() -> None:
    assert ModelCapability().model_id == ""
    assert ModelSettings().context == ""


def test_a_bare_transport_keeps_the_axis_a_row_cannot_know() -> None:
    """A row does not know who rolls history, so it must not pin that axis.

    Defaulting it narrow made every CLI meet empty: the CLI offers
    ``manage_context_server_side={True}``, and ``{False} & {True}`` is a
    model nothing can be requested from.
    """
    met = _row() & ModelCapability()
    assert met.manage_context_server_side == _row().manage_context_server_side
    # The thinking and tier axes stay narrow: a row DOES know what it can
    # think and which tiers the vendor sells it at.
    assert met.thinking_effort == frozenset({"none"})
    assert met.service_tier == frozenset({"auto", "default"})
    assert met.cache_ttl_sec == 0.0


def test_the_meet_only_removes() -> None:
    transport = ModelCapability(
        thinking_effort=frozenset({"none", "low", "medium", "high", "max"}),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
        thinking_output=frozenset({"none", "text", "redacted"}),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
        cache_ttl_sec=7200.0,
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
    )
    met = _row() & transport
    assert met.thinking_effort == frozenset({"none", "low", "high"})
    assert met.service_tier == frozenset({"auto", "default", "priority"})


def test_the_meet_cannot_grant() -> None:
    met = ModelCapability(thinking_effort=frozenset({"none"})) & ModelCapability(
        thinking_effort=frozenset({"none", "max"})
    )
    assert met.thinking_effort == frozenset({"none"})


def test_the_meet_takes_the_lower_cache_bound() -> None:
    assert (_row() & ModelCapability(cache_ttl_sec=300.0)).cache_ttl_sec == 300.0
    assert (_row() & ModelCapability(cache_ttl_sec=7200.0)).cache_ttl_sec == 3600.0


def test_the_meet_passes_context_and_prices_through() -> None:
    met = _row() & ModelCapability()
    assert met.context.keys() == {"", "+1m"}
    assert met.prices == _row().prices


def test_the_transport_declares_retries_and_auth() -> None:
    """Not intersected: a catalog row cannot know who bills or retries it.

    Conjoining made a transport's ``account_auth=True`` unreachable, because
    every row leaves it at the ``False`` default.
    """
    row = _row()
    assert (row & ModelCapability(retries_internally=True)).retries_internally is True
    assert (row & ModelCapability()).retries_internally is False
    assert (row & ModelCapability(account_auth=True)).account_auth is True
    assert (row & ModelCapability()).account_auth is False


# ---- assignment validates --------------------------------------------------


def test_every_settings_axis_names_a_capability_field() -> None:
    """The membership check is total only if the names line up."""
    capability_names = {f.name for f in fields(ModelCapability)}
    for f in fields(ModelSettings):
        if f.name == "capability":
            continue
        assert f.name in capability_names, f.name


def test_a_valid_selection_constructs() -> None:
    ModelSettings(
        capability=_row(),
        context="+1m",
        thinking_effort="high",
        thinking_budget="auto",
        thinking_output="redacted",
        service_tier="priority",
        cache_ttl_sec=300.0,
        manage_context_server_side=True,
    )


def test_defaults_construct_against_a_default_capability() -> None:
    ModelSettings(capability=ModelCapability())


def test_assignment_rejects_what_the_model_withholds() -> None:
    """The whole point of the setter: a stale knob never reaches the wire."""
    settings = ModelSettings(capability=_row())
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        settings.thinking_effort = "medium"
    assert settings.thinking_effort == "none"


def test_construction_rejects_what_the_model_withholds() -> None:
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        ModelSettings(capability=_row(), thinking_effort="medium")


def test_an_unoffered_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="context='\\+200k'"):
        ModelSettings(capability=_row(), context="+200k")


def test_a_bound_is_checked_by_comparison_not_membership() -> None:
    """A duration is a quantity: any value up to the bound is selectable."""
    ModelSettings(capability=_row(), cache_ttl_sec=1.0)
    ModelSettings(capability=_row(), cache_ttl_sec=3600.0)
    with pytest.raises(ValueError, match="exceeds the most"):
        ModelSettings(capability=_row(), cache_ttl_sec=3600.1)


def test_a_boolean_axis_validates_by_membership() -> None:
    row = ModelCapability(manage_context_server_side=frozenset({False}))
    ModelSettings(capability=row, manage_context_server_side=False)
    with pytest.raises(ValueError, match="manage_context_server_side=True"):
        ModelSettings(capability=row, manage_context_server_side=True)


def test_the_error_names_the_model_and_what_it_offers() -> None:
    with pytest.raises(ValueError, match="not offered by m") as excinfo:
        ModelSettings(capability=_row(), service_tier="flex")
    assert "'priority'" in str(excinfo.value)


_OFF = ModelCapability(
    thinking_effort=frozenset({"none"}),
    thinking_budget=frozenset({"none"}),
    thinking_output=frozenset({"none"}),
    service_tier=frozenset({"auto"}),
    cache_ttl_sec=0.0,
    manage_context_server_side=frozenset({False}),
)

_UNOFFERED: Final = (
    ("context", "+200k"),
    ("thinking_effort", "medium"),
    ("thinking_budget", "fixed"),
    ("thinking_output", "text"),
    ("service_tier", "flex"),
    ("cache_ttl_sec", 1.0),
    ("manage_context_server_side", True),
)


@pytest.mark.parametrize(("name", "value"), _UNOFFERED, ids=[n for n, _ in _UNOFFERED])
def test_assignment_checks_every_axis(name: str, value: object) -> None:
    settings = ModelSettings(capability=_OFF)
    with pytest.raises(ValueError, match=name):
        setattr(settings, name, value)


def test_the_axis_cases_cover_every_settings_axis() -> None:
    """Without this, a new ``ModelSettings`` axis is untested but green."""
    axes = {f.name for f in fields(ModelSettings)} - {"capability"}
    assert {n for n, _ in _UNOFFERED} == axes


def test_the_capability_itself_is_assignable() -> None:
    """It is the authority, not a choice, so it cannot validate against itself."""
    settings = ModelSettings(capability=_OFF)
    settings.capability = _row()
    settings.thinking_effort = "high"


# ---- take / adopt ----------------------------------------------------------


def test_take_applies_what_is_offered() -> None:
    settings = ModelSettings(capability=_row())
    settings.take(thinking_effort="high", thinking_budget="auto")
    assert settings.thinking_effort == "high"
    assert settings.thinking_budget == "auto"


def test_take_drops_what_is_not_offered_instead_of_raising() -> None:
    """A swap or resume must not FAIL on a knob the new model withholds."""
    settings = ModelSettings(capability=_row())
    settings.take(thinking_effort="max", thinking_budget="auto")
    assert settings.thinking_effort == "none"
    assert settings.thinking_budget == "auto"


def test_take_rejects_a_name_that_is_not_an_axis() -> None:
    with pytest.raises(ValueError, match="not settings axes: nonsense"):
        ModelSettings().take(nonsense=1)


def test_adopt_carries_the_selection_across_a_swap() -> None:
    old = ModelSettings(
        capability=_row(), thinking_effort="high", service_tier="priority"
    )
    new = ModelSettings(capability=_row(), context="+1m")
    new.adopt(old)
    assert new.thinking_effort == "high"
    assert new.service_tier == "priority"


def test_adopt_keeps_the_incoming_context_tag() -> None:
    """The new model id already named its window; the swap must not undo it."""
    old = ModelSettings(capability=_row(), context="")
    new = ModelSettings(capability=_row(), context="+1m")
    new.adopt(old)
    assert new.context == "+1m"


def test_adopt_drops_a_knob_the_new_model_rejects() -> None:
    old = ModelSettings(capability=_row(), thinking_effort="high")
    new = ModelSettings(capability=_OFF)
    new.adopt(old)
    assert new.thinking_effort == "none"


# ---- narrowest -------------------------------------------------------------


def test_narrowest_is_the_unset_value_of_every_axis_that_offers_one() -> None:
    assert ModelSettings.narrowest(_row()) == ModelSettings(capability=_row())


def test_narrowest_carries_the_context_the_id_selected() -> None:
    assert ModelSettings.narrowest(_row(), context="+1m").context == "+1m"


def test_narrowest_carries_the_capability_it_narrowed() -> None:
    """Without it every later write would validate against nothing."""
    assert ModelSettings.narrowest(_row()).capability == _row()


def test_narrowest_climbs_an_axis_that_withholds_its_unset_value() -> None:
    """A default-constructed settings was unbuildable against these rows.

    Model Studio's ``-thinking`` ids reject "do not think"; a CLI transport
    requires server-side history. Both make the field default unselectable.
    """
    row = ModelCapability(
        thinking_effort=frozenset({"low", "high"}),
        manage_context_server_side=frozenset({True}),
    )
    settings = ModelSettings.narrowest(row)
    assert settings.thinking_effort == "low"
    assert settings.manage_context_server_side is True


@pytest.mark.parametrize(
    ("models", "transport"),
    [(_row(), ModelCapability()), (_row(), _row())],
    ids=["bare", "self"],
)
def test_narrowest_builds_against_what_it_narrowed(
    models: ModelCapability, transport: ModelCapability
) -> None:
    """Construction validates, so an unselectable narrowest would raise here."""
    capability = models & transport
    assert ModelSettings.narrowest(capability).capability == capability


def test_narrowest_rejects_an_empty_axis() -> None:
    """An empty meet is a model nothing can be requested from -- say so."""
    with pytest.raises(ValueError, match="thinking_effort offers nothing"):
        ModelSettings.narrowest(ModelCapability(thinking_effort=frozenset()))


# ---- limits ----------------------------------------------------------------


def test_limits_are_derived_not_cached() -> None:
    """A cached row let a spec claim one context and carry another's."""
    row = _row()
    assert ModelSettings(capability=row).limits.max_request_tokens == 200_000
    assert (
        ModelSettings(capability=row, context="+1m").limits.max_request_tokens
        == 1_000_000
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
