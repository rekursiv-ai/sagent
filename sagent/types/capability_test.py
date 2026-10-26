"""Tests for what a model offers, the meet with a transport, and settings validation."""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType
from typing import Final

import pytest

from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    PruningPolicy,
    ThinkingCapability,
)
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice


def _row() -> ModelCapability:
    return ModelCapability(
        model_id="m",
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000),
                "+1m": ModelLimits(max_request_tokens=1_000_000),
            },
        ),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=5.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "high"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "default", "priority"}),
        cache_ttl_sec=frozenset({300.0, 3600.0}),
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
    )


_OFF = ModelCapability(
    thinking=ThinkingCapability(
        effort=frozenset({"none"}),
        budget=frozenset({"none"}),
        output=frozenset({"none"}),
    ),
    service_tier=frozenset({"auto"}),
    cache_ttl_sec=frozenset({0.0}),
    manage_context_server_side=frozenset({False}),
)

_UNOFFERED: Final = (
    ("context", "+200k"),
    ("thinking_effort", "medium"),
    ("thinking_budget", "fixed"),
    ("thinking_output", "text"),
    ("service_tier", "flex"),
    ("cache_ttl_sec", 300.0),
    ("manage_context_server_side", True),
)


# ---- the meet --------------------------------------------------------------


def test_every_field_is_default_constructible() -> None:
    assert ModelCapability().model_id == ""
    assert ModelSettings().context == ""


def test_a_bare_transport_keeps_the_axis_a_row_cannot_know() -> None:
    """A row does not know who rolls history, so it must not pin that axis."""
    met = _row() & ModelCapability()
    assert met.manage_context_server_side == frozenset({False, True})
    assert met.thinking.effort == frozenset({"none"})
    assert met.thinking.budget == frozenset({"none"})
    assert met.thinking.output == frozenset({"none"})
    assert met.service_tier == frozenset({"auto", "default"})


def test_the_meet_only_removes() -> None:
    transport = ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "medium", "high", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
        manage_context_server_side=frozenset({True}),
    )
    met = _row() & transport
    assert met.thinking.effort == frozenset({"none", "low", "high"})
    assert met.thinking.budget == frozenset({"none", "auto"})
    assert met.thinking.output == frozenset({"none", "text"})
    assert met.service_tier == frozenset({"auto", "default", "priority"})
    assert met.manage_context_server_side == frozenset({True})


def test_the_transport_declares_the_cache_lifetimes() -> None:
    """Not intersected: only the transport knows if it writes a breakpoint."""
    offered = frozenset({300.0, 7200.0})
    assert (_row() & ModelCapability(cache_ttl_sec=offered)).cache_ttl_sec == offered
    assert (_row() & ModelCapability()).cache_ttl_sec == frozenset({0.0})


def test_the_meet_passes_model_facts_through() -> None:
    row = ModelCapability(
        model_id="m",
        knowledge_cutoff="2026-01",
        effort_as_level=True,
        context=_row().context,
        prices=_row().prices,
    )
    met = row & ModelCapability(model_id="t", effort_as_level=False)
    assert met.model_id == "m"
    assert met.knowledge_cutoff == "2026-01"
    assert met.effort_as_level is True
    assert met.context.keys() == {"", "+1m"}
    assert met.prices == _row().prices


def test_the_transport_declares_retries_and_auth() -> None:
    """A catalog row cannot know who bills or retries it."""
    row = _row()
    assert (row & ModelCapability(retries_internally=True)).retries_internally is True
    assert (row & ModelCapability()).retries_internally is False
    assert (row & ModelCapability(account_auth=True)).account_auth is True
    assert (row & ModelCapability()).account_auth is False


# ---- assignment validates --------------------------------------------------


def test_a_valid_selection_constructs() -> None:
    settings = ModelSettings(
        capability=_row(),
        context="+1m",
        thinking_effort="high",
        thinking_budget="auto",
        thinking_output="redacted",
        service_tier="priority",
        cache_ttl_sec=300.0,
        manage_context_server_side=True,
    )
    assert settings.thinking_output == "redacted"


def test_assignment_rejects_what_the_model_withholds() -> None:
    settings = ModelSettings.narrowest(_row())
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        settings.thinking_effort = "medium"
    assert settings.thinking_effort == "none"


def test_construction_rejects_what_the_model_withholds() -> None:
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, thinking_effort="medium")


def test_the_cache_axis_validates_by_membership() -> None:
    ModelSettings(capability=_row(), cache_ttl_sec=3600.0)
    with pytest.raises(ValueError, match="not offered by m"):
        ModelSettings(capability=_row(), cache_ttl_sec=1800.0)


def test_the_error_names_the_model_and_what_it_offers() -> None:
    with pytest.raises(ValueError, match="not offered by m") as excinfo:
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, service_tier="flex")
    assert "allowed: 'auto', 'default', 'priority'" in str(excinfo.value)


def test_an_unnamed_model_is_called_this_model() -> None:
    with pytest.raises(ValueError, match="not offered by this model"):
        ModelSettings.narrowest(_OFF).service_tier = "flex"


@pytest.mark.parametrize(("name", "value"), _UNOFFERED, ids=[n for n, _ in _UNOFFERED])
def test_assignment_checks_every_axis(name: str, value: object) -> None:
    settings = ModelSettings.narrowest(_OFF)
    with pytest.raises(ValueError, match=name):
        setattr(settings, name, value)


def test_the_axis_cases_cover_every_offered_axis() -> None:
    """Without this, a new ``ModelSettings`` axis is untested but green."""
    axes = {f.name for f in fields(ModelSettings)} - {"capability", "pruning"}
    assert {n for n, _ in _UNOFFERED} == axes


def test_any_model_takes_a_pruning_policy() -> None:
    """Pruning is the caller's policy, not something a model offers."""
    settings = ModelSettings.narrowest(_OFF)
    policy = PruningPolicy(trigger_tokens=100, target_tokens=50)
    settings.pruning = policy
    assert settings.pruning == policy
    settings.take(pruning=None)
    assert settings.pruning is None


def test_the_capability_itself_is_assignable() -> None:
    settings = ModelSettings.narrowest(_OFF)
    settings.capability = _row()
    settings.thinking_effort = "high"
    assert settings.thinking_effort == "high"


def test_capability_assignment_renarrows_context_and_all_axes() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    settings.thinking_effort = "high"
    settings.capability = _OFF
    assert settings.capability == _OFF
    assert settings.context == ""
    assert settings.limits == ModelLimits()
    assert settings.thinking_effort == "none"
    assert settings.cache_ttl_sec == 0.0


def test_capability_assignment_keeps_what_the_new_one_still_offers() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    settings.thinking_effort = "high"
    settings.service_tier = "priority"
    settings.capability = _row()
    assert settings.context == "+1m"
    assert settings.thinking_effort == "high"
    assert settings.service_tier == "priority"


def test_invalid_capability_assignment_leaves_settings_unchanged() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    with pytest.raises(ValueError, match="cache_ttl_sec offers nothing"):
        settings.capability = ModelCapability(cache_ttl_sec=frozenset())
    assert settings.capability == _row()
    assert settings.context == "+1m"


# ---- take / adopt ----------------------------------------------------------


def test_take_applies_what_is_offered() -> None:
    settings = ModelSettings.narrowest(_row())
    settings.take(thinking_effort="high", thinking_budget="auto")
    assert settings.thinking_effort == "high"
    assert settings.thinking_budget == "auto"


def test_take_falls_back_to_the_lowest_rung_instead_of_raising() -> None:
    settings = ModelSettings.narrowest(_row())
    settings.thinking_effort = "high"
    settings.take(thinking_effort="max", thinking_budget="auto")
    assert settings.thinking_effort == "none"
    assert settings.thinking_budget == "auto"


@pytest.mark.parametrize("name", ["nonsense", "capability"])
def test_take_rejects_a_name_that_is_not_an_axis(name: str) -> None:
    with pytest.raises(ValueError, match=f"not settings axes: {name}"):
        ModelSettings().take(**{name: 1})


def test_take_names_every_unknown_axis_sorted() -> None:
    with pytest.raises(ValueError, match=r"not settings axes: alpha, beta$"):
        ModelSettings().take(beta=1, alpha=2)


def test_take_keeps_the_context_when_asked_for_an_unoffered_one() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    settings.take(context="+200k")
    assert settings.context == "+1m"


def test_adopt_carries_the_selection_across_a_swap() -> None:
    old = ModelSettings(
        capability=_row(),
        cache_ttl_sec=3600.0,
        thinking_effort="high",
        service_tier="priority",
    )
    new = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+1m")
    new.adopt(old)
    assert new.thinking_effort == "high"
    assert new.service_tier == "priority"
    assert new.cache_ttl_sec == 3600.0


def test_adopt_keeps_the_incoming_context_tag() -> None:
    old = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="")
    new = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+1m")
    new.adopt(old)
    assert new.context == "+1m"


def test_adopt_drops_a_knob_the_new_model_rejects() -> None:
    old = ModelSettings(capability=_row(), cache_ttl_sec=300.0, thinking_effort="high")
    new = ModelSettings.narrowest(_OFF)
    new.adopt(old)
    assert new.thinking_effort == "none"


# ---- narrowest -------------------------------------------------------------


def test_narrowest_is_the_unset_value_of_every_axis_that_offers_one() -> None:
    assert ModelSettings.narrowest(_row()) == ModelSettings(
        capability=_row(),
        cache_ttl_sec=300.0,
    )


def test_narrowest_selects_the_shortest_cache_offered() -> None:
    assert ModelSettings.narrowest(_row()).cache_ttl_sec == 300.0
    assert ModelSettings.narrowest(ModelCapability()).cache_ttl_sec == 0.0


def test_narrowest_carries_the_context_and_capability() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    assert settings.context == "+1m"
    assert settings.capability == _row()


def test_narrowest_climbs_an_axis_that_withholds_its_unset_value() -> None:
    row = ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"low", "high"}),
            budget=frozenset({"fixed", "auto"}),
            output=frozenset({"redacted", "text"}),
        ),
        service_tier=frozenset({"priority", "flex"}),
        manage_context_server_side=frozenset({True}),
    )
    settings = ModelSettings.narrowest(row)
    assert settings.thinking_effort == "low"
    assert settings.thinking_budget == "auto"
    assert settings.thinking_output == "text"
    assert settings.service_tier == "flex"
    assert settings.manage_context_server_side is True


@pytest.mark.parametrize(
    ("axis", "capability"),
    [
        (
            "thinking_effort",
            ModelCapability(thinking=ThinkingCapability(effort=frozenset())),
        ),
        (
            "thinking_budget",
            ModelCapability(thinking=ThinkingCapability(budget=frozenset())),
        ),
        (
            "thinking_output",
            ModelCapability(thinking=ThinkingCapability(output=frozenset())),
        ),
        ("service_tier", ModelCapability(service_tier=frozenset())),
        ("cache_ttl_sec", ModelCapability(cache_ttl_sec=frozenset())),
        (
            "manage_context_server_side",
            ModelCapability(manage_context_server_side=frozenset()),
        ),
    ],
)
def test_narrowest_names_an_empty_axis(axis: str, capability: ModelCapability) -> None:
    with pytest.raises(ValueError, match=f"^{axis} offers nothing selectable"):
        ModelSettings.narrowest(capability)


def test_limits_follow_the_selected_context() -> None:
    row = _row()
    assert ModelSettings.narrowest(row).limits.max_request_tokens == 200_000
    settings = ModelSettings.narrowest(row, context="+1m")
    assert settings.limits.max_request_tokens == 1_000_000


def test_equal_capabilities_hash_equal_regardless_of_mapping_order() -> None:
    a = ModelCapability(
        context=MappingProxyType({"": ModelLimits(), "+1m": ModelLimits()}),
    )
    b = ModelCapability(
        context=MappingProxyType({"+1m": ModelLimits(), "": ModelLimits()}),
    )
    assert hash(a) == hash(b)
    assert hash(_row()) == hash(_row())
    assert hash(_row()) != hash(ModelCapability())


def test_the_meet_cannot_grant() -> None:
    met = ModelCapability(
        thinking=ThinkingCapability(effort=frozenset({"none"})),
    ) & ModelCapability(
        thinking=ThinkingCapability(effort=frozenset({"none", "max"})),
    )
    assert met.thinking.effort == frozenset({"none"})


def test_the_meet_passes_context_and_prices_through() -> None:
    met = _row() & ModelCapability()
    assert met.context.keys() == {"", "+1m"}
    assert met.prices == _row().prices


def test_the_meet_keeps_the_rows_effort_spelling() -> None:
    """A model fact: the transport can neither grant nor revoke it."""
    level_row = ModelCapability(effort_as_level=True)
    assert (level_row & ModelCapability()).effort_as_level is True
    assert (_row() & ModelCapability(effort_as_level=True)).effort_as_level is False


def test_defaults_construct_against_a_default_capability() -> None:
    ModelSettings(capability=ModelCapability())


def test_an_unoffered_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="context='\\+200k'"):
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+200k")


def test_a_boolean_axis_validates_by_membership() -> None:
    row = ModelCapability(manage_context_server_side=frozenset({False}))
    ModelSettings(capability=row, manage_context_server_side=False)
    with pytest.raises(ValueError, match="manage_context_server_side=True"):
        ModelSettings(capability=row, manage_context_server_side=True)


def test_take_drops_what_is_not_offered_instead_of_raising() -> None:
    """A swap or resume must not FAIL on a knob the new model withholds."""
    settings = ModelSettings.narrowest(_row())
    settings.take(thinking_effort="max", thinking_budget="auto")
    assert settings.thinking_effort == "none"
    assert settings.thinking_budget == "auto"


def test_narrowest_selects_the_cache_the_wire_already_ships() -> None:
    """``0.0`` claimed no caching while a ``5m`` breakpoint shipped anyway."""
    assert ModelSettings.narrowest(_row()).cache_ttl_sec == 300.0
    assert ModelSettings.narrowest(ModelCapability()).cache_ttl_sec == 0.0
    only_long = ModelCapability(cache_ttl_sec=frozenset({3600.0}))
    assert ModelSettings.narrowest(only_long).cache_ttl_sec == 3600.0


def test_narrowest_carries_the_context_the_id_selected() -> None:
    assert ModelSettings.narrowest(_row(), context="+1m").context == "+1m"


def test_narrowest_carries_the_capability_it_narrowed() -> None:
    """Without it every later write would validate against nothing."""
    assert ModelSettings.narrowest(_row()).capability == _row()


@pytest.mark.parametrize(
    ("models", "transport"),
    [(_row(), ModelCapability()), (_row(), _row())],
    ids=["bare", "self"],
)
def test_narrowest_builds_against_what_it_narrowed(
    models: ModelCapability,
    transport: ModelCapability,
) -> None:
    """Construction validates, so an unselectable narrowest would raise here."""
    capability = models & transport
    assert ModelSettings.narrowest(capability).capability == capability


def test_narrowest_rejects_an_empty_axis() -> None:
    """An empty meet is a model nothing can be requested from -- say so."""
    with pytest.raises(ValueError, match="thinking_effort offers nothing"):
        ModelSettings.narrowest(
            ModelCapability(thinking=ThinkingCapability(effort=frozenset())),
        )


def test_limits_are_derived_not_cached() -> None:
    """A cached row let a spec claim one context and carry another's."""
    row = _row()
    assert ModelSettings.narrowest(row).limits.max_request_tokens == 200_000
    assert (
        ModelSettings.narrowest(row, context="+1m").limits.max_request_tokens
        == 1_000_000
    )


def test_settings_over_an_empty_axis_cannot_be_constructed() -> None:
    """So ``take`` never meets a capability whose ``narrowest`` raises."""
    with pytest.raises(ValueError, match=r"cache_ttl_sec=0\.0 is not offered"):
        ModelSettings(capability=ModelCapability(cache_ttl_sec=frozenset()))


def test_take_rejects_capability_as_a_non_axis() -> None:
    with pytest.raises(ValueError, match="not settings axes: capability"):
        ModelSettings().take(capability=ModelCapability())


def test_equal_capabilities_are_hashable_independent_of_mapping_order() -> None:
    assert hash(ModelCapability()) == hash(ModelCapability())
    assert hash(_row()) == hash(_row())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
