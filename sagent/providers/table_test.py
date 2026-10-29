"""Tests for a provider's model table: lookup by name, and reading a catalog."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import itertools
import math
import re

import pytest
import yaml

from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.exceptions import UnknownModelError, UnsupportedTagError
from sagent.providers.table import ModelTable, ModelTags, chosen, spelled


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


_FREE = {
    "request": 0.0,
    "response": 0.0,
    "cache_write": 0.0,
    "cache_write_1h": 0.0,
    "cache_read": 0.0,
}
_THINKING = {"effort": ["none"], "budget": ["none"], "output": ["none"]}
_EVERY = {
    "effort": ["none", "low", "high"],
    "budget": ["none", "auto", "fixed"],
    "output": ["none"],
}
_WHOLE = "A catalog's thinking must state effort, budget, and output."
_ALIASES = "Catalogs may not use YAML aliases."


def _exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


def _row(
    model_id: str = "opus-4.8",
    *,
    wire_model_id: str | None = None,
    window: int = 1_000_000,
    bands: tuple[int, ...] = (),
) -> ModelCapability:
    edges: tuple[int | float, ...] = (0, *(band + 1 for band in bands), math.inf)
    prices = PriceCatalog(
        prices={
            PriceCondition(token_size_range=(int(lo), hi)): TokenPrice()
            for lo, hi in itertools.pairwise(edges)
        },
        window=window,
    )
    return ModelCapability(
        model_id=model_id,
        wire_model_id=(
            "claude-" + model_id.replace(".", "-")
            if wire_model_id is None
            else wire_model_id
        ),
        limits=ModelLimits(max_request_tokens=window),
        prices=prices,
        default=ModelSettings(context_size=prices["auto", "fullcontext"]),
    )


def _table() -> ModelTable:
    """Newest first within each family, as every vendor's catalog orders them."""
    return ModelTable(
        rows=(
            _row("opus-5.5"),
            _row("opus-5.0"),
            _row("opus-4.8"),
            _row("sonnet-5.5"),
            _row("sonnet-5.0"),
            _row("sonnet-4.6"),
        ),
        roles={"default": "opus", "utility": "sonnet-4.6"},
    )


# ---- lookup ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "model_id"),
    [
        ("sonnet", "sonnet-5.5"),
        ("sonnet-5", "sonnet-5.5"),
        ("sonnet-5.5", "sonnet-5.5"),
        ("sonnet-5.0", "sonnet-5.0"),
        ("sonnet-4", "sonnet-4.6"),
        ("claude-opus-4-8", "opus-4.8"),
        ("default", "opus-5.5"),
        ("utility", "sonnet-4.6"),
    ],
)
def test_a_key_is_an_exact_name_a_vendor_id_a_role_or_a_family(
    key: str,
    model_id: str,
) -> None:
    assert _table()[key].model_id == model_id
    assert key in _table()


def test_an_exact_name_outranks_a_role_of_the_same_spelling() -> None:
    table = ModelTable(rows=(_row("a-1.0"), _row("b-1.0")), roles={"a-1.0": "b-1.0"})
    assert table["a-1.0"] is table.rows[0]


def test_a_vendor_id_outranks_a_family_of_the_same_spelling() -> None:
    """A server reporting ``flash`` means that model, not the family's newest."""
    table = ModelTable(
        rows=(_row("flash-2.0"), _row("flash-1.0", wire_model_id="flash")),
    )
    assert table["flash"].model_id == "flash-1.0"
    assert table["flash-2"].model_id == "flash-2.0"


@pytest.mark.parametrize(
    "key",
    ["opu", "sonnet-5.5x", "sonnet-5.5.1", "sonnet-", "haiku"],
)
def test_a_family_name_stops_at_a_separator(key: str) -> None:
    table = _table()
    assert key not in table
    assert table.get(key) is None
    with pytest.raises(KeyError):
        table[key]


def test_a_family_never_reaches_a_longer_family_that_starts_with_it() -> None:
    table = ModelTable(
        rows=(_row("flash-lite-3.5"), _row("flash-3.8"), _row("flash-2.5")),
    )
    assert table["flash"].model_id == "flash-3.8"
    assert table["flash-2"].model_id == "flash-2.5"
    assert table["flash-lite"].model_id == "flash-lite-3.5"
    assert "flash-lite-2" not in table


def test_iterating_lists_exact_names_only() -> None:
    table = _table()
    assert list(table) == [
        "opus-5.5",
        "opus-5.0",
        "opus-4.8",
        "sonnet-5.5",
        "sonnet-5.0",
        "sonnet-4.6",
    ]
    assert len(table) == 6
    assert 5 not in table


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("sonnet-5.0", "sonnet-5.0"),
        ("claude-opus-4-8", "opus-4.8"),
        ("sonnet", None),
        ("sonnet-5", None),
        ("default", None),
    ],
)
def test_exact_answers_only_to_an_exact_name_or_a_vendor_id(
    name: str,
    model_id: str | None,
) -> None:
    """Billing an id a server reports at a family's newest would misprice it."""
    row = _table().exact(name)
    assert (None if row is None else row.model_id) == model_id


def test_an_unversioned_id_is_an_exact_name_only() -> None:
    """A local endpoint's ``local-model`` has no family to order or match."""
    table = ModelTable(rows=(_row("local-model"), _row("local-model-b")))
    assert table["local-model"].model_id == "local-model"
    assert table["local-model-b"].model_id == "local-model-b"
    assert "local" not in table


# ---- the table's invariants --------------------------------------------------


def test_best_and_everyday_fall_back_to_the_default() -> None:
    table = _table()
    assert table.roles["best"] == table.roles["everyday"] == "opus"
    declared = ModelTable(
        rows=(_row("opus-5.5"), _row("sonnet-5.5")),
        roles={"best": "opus", "default": "opus", "everyday": "sonnet"},
    )
    assert declared["everyday"].model_id == "sonnet-5.5"
    bare = ModelTable(rows=(_row(),))
    assert dict(bare.roles) == {}


def test_roles_are_a_frozen_copy() -> None:
    roles = {"default": "opus"}
    table = ModelTable(rows=(_row(),), roles=roles)
    roles["default"] = "nope"
    assert table.roles["default"] == "opus"
    assert isinstance(table.roles, MappingProxyType)


def test_a_role_naming_no_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Role 'default' names no model: 'haiku'"):
        ModelTable(rows=(_row(),), roles={"default": "haiku"})


def test_a_model_listed_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Model 'opus-4\.8' is listed twice"):
        ModelTable(rows=(_row(), _row()))


@pytest.mark.parametrize(
    "ids",
    [
        ("opus-4.8", "opus-5.0"),
        ("opus-5.0", "sonnet-5.5", "opus-5.5"),
        ("opus-5.9", "opus-5.10"),
    ],
)
def test_rows_listed_oldest_first_are_rejected(ids: tuple[str, ...]) -> None:
    """``opus`` would otherwise name the older row and misprice every request."""
    with pytest.raises(ValueError, match="newest first"):
        ModelTable(rows=tuple(_row(i) for i in ids))


def test_families_may_interleave() -> None:
    table = ModelTable(rows=(_row("a-2.0"), _row("b-1.0"), _row("a-1.0")))
    assert table["a"].model_id == "a-2.0"


# ---- resolve -----------------------------------------------------------------


def test_resolve_returns_the_row_and_settings_for_its_size() -> None:
    capability, settings = _table().resolve("opus-4.8+258k")
    assert capability.model_id == "opus-4.8"
    assert settings.context_size.tokens == 258_000
    _, untagged = _table().resolve("opus-4.8")
    assert untagged == capability.default


@pytest.mark.parametrize("model_id", ["opus-5+258K", "opus-5+258k", "utility+258k"])
def test_resolve_follows_roles_families_and_tags_in_any_case(model_id: str) -> None:
    table = ModelTable(
        rows=(_row("opus-5.5"), _row("sonnet-4.6")),
        roles={"utility": "sonnet"},
    )
    capability, settings = table.resolve(model_id)
    assert settings.context_size.tokens == 258_000
    assert capability.model_id == (
        "sonnet-4.6" if "utility" in model_id else "opus-5.5"
    )


def test_a_repeated_tag_is_one_tag() -> None:
    _, settings = _table().resolve("opus+1m+1M")
    assert settings.context_size.tokens == 1_000_000


@pytest.mark.parametrize(
    ("tag", "bands", "context"),
    [
        ("+smallcontext", (32_000, 256_000), 32_000),
        ("+midcontext", (32_000, 256_000), 256_000),
        ("+fullcontext", (32_000, 256_000), 1_000_000),
        ("+smallcontext", (272_000,), 272_000),
        ("+midcontext", (272_000,), 1_000_000),
        ("+smallcontext", (), 1_000_000),
        ("+SmallContext", (32_000, 256_000), 32_000),
        # A journal's model line from before the rename.
        ("+large", (32_000, 256_000), 1_000_000),
    ],
)
def test_the_size_names_read_the_price_bands(
    tag: str,
    bands: tuple[int, ...],
    context: int,
) -> None:
    row = _row(bands=bands)
    _, settings = ModelTable(rows=(row,)).resolve(f"opus-4.8{tag}")
    assert settings.context_size.tokens == context


@pytest.mark.parametrize(
    ("tag", "tokens"),
    [
        ("+200k", 200_000),
        ("+272k", 272_000),
        ("+1m", 1_000_000),
        ("+2m", 1_000_000),
        ("+258123", 258_123),
    ],
)
def test_a_size_tag_is_that_many_tokens_within_the_window(
    tag: str,
    tokens: int,
) -> None:
    _, settings = _table().resolve(f"opus-4.8{tag}")
    assert settings.context_size.tokens == tokens


@pytest.mark.parametrize(
    ("suffix", "named"),
    [
        ("+fullcontext+smallcontext", "+fullcontext, +smallcontext"),
        ("+fullcontext+1m", "+1m, +fullcontext"),
        ("+200k+smallcontext", "+200k, +smallcontext"),
        ("+1m+200k", "+1m, +200k"),
    ],
)
def test_conflicting_tags_are_rejected(suffix: str, named: str) -> None:
    message = f"Model 'opus-4.8{suffix}' selects conflicting contexts: {named}."
    with pytest.raises(UnsupportedTagError, match=_exactly(message)):
        _table().resolve(f"opus-4.8{suffix}")
    with pytest.raises(UnsupportedTagError, match=_exactly(message)):
        ModelTags.parse(f"opus-4.8{suffix}")


@pytest.mark.parametrize("tag", ["+high", "+fast", " budget=auto"])
def test_resolve_takes_a_name_and_its_size_alone(tag: str) -> None:
    """The effort, tier, and choices are the caller's to apply; dropped, they vanish."""
    with pytest.raises(UnknownModelError):
        _table().resolve(f"opus{tag}")


@pytest.mark.parametrize(
    ("line", "tags"),
    [
        ("opus", ModelTags(model="opus")),
        ("opus+1m+high", ModelTags(model="opus", context=1_000_000, effort="high")),
        ("opus+HIGH+1M", ModelTags(model="opus", context=1_000_000, effort="high")),
        ("opus+high+258k", ModelTags(model="opus", context=258_000, effort="high")),
        ("opus+medthink", ModelTags(model="opus", effort="medium")),
        ("opus+MediumThink", ModelTags(model="opus", effort="medium")),
        ("opus+fullcontext", ModelTags(model="opus", context="fullcontext")),
        ("opus+SmallContext", ModelTags(model="opus", context="smallcontext")),
        # Journals keep model lines written before the tags were renamed.
        (
            "opus+large+max",
            ModelTags(model="opus", context="fullcontext", effort="max"),
        ),
        (
            "opus+small+none",
            ModelTags(model="opus", context="smallcontext", effort="none"),
        ),
        (
            "opus+mid+medium",
            ModelTags(model="opus", context="midcontext", effort="medium"),
        ),
        ("opus+1m+1M", ModelTags(model="opus", context=1_000_000)),
        (
            "opus+fast+1m+high",
            ModelTags(
                model="opus",
                context=1_000_000,
                effort="high",
                service_tier="priority",
            ),
        ),
        ("opus+flex", ModelTags(model="opus", service_tier="flex")),
        ("Org/M+cuda+medium", ModelTags(model="Org/M+cuda", effort="medium")),
        ("Org/M+medium+cuda", ModelTags(model="Org/M+cuda", effort="medium")),
        (
            "work@vendor/opus-5.5+fullcontext",
            ModelTags(model="work@vendor/opus-5.5", context="fullcontext"),
        ),
        (
            "opus+high budget=auto  output=text",
            ModelTags(
                model="opus",
                effort="high",
                choices={"budget": "auto", "output": "text"},
            ),
        ),
    ],
)
def test_parse_lifts_every_tag_from_anywhere(line: str, tags: ModelTags) -> None:
    assert ModelTags.parse(line) == tags


@pytest.mark.parametrize(
    "word",
    ["fullctx", "context", "think", "0", "1.5m", "nothing"],
)
def test_a_word_of_no_kind_stays_in_the_name(word: str) -> None:
    assert ModelTags.parse(f"opus+{word}") == ModelTags(model=f"opus+{word}")


@pytest.mark.parametrize(
    "tags",
    [
        ModelTags(model="opus"),
        *(
            ModelTags(model="opus", context=context)
            for context in ("smallcontext", "midcontext", "fullcontext", 258_000)
        ),
        *(
            ModelTags(model="opus", effort=effort)
            for effort in ("none", "min", "low", "medium", "high", "xhigh", "max")
        ),
        *(
            ModelTags(model="opus", service_tier=tier)
            for tier in ("auto", "default", "flex", "priority", "ultrafast")
        ),
        ModelTags(
            model="work@vendor/opus-5.5",
            context=1_000_000,
            effort="low",
            service_tier="flex",
            choices={"budget": "auto", "cache_ttl_sec": "3600"},
        ),
    ],
    ids=str,
)
def test_a_line_parses_back_to_its_tags(tags: ModelTags) -> None:
    assert ModelTags.parse(str(tags)) == tags


@pytest.mark.parametrize(
    ("tags", "line"),
    [
        (
            ModelTags(
                model="anthropic/opus-5.5",
                context="fullcontext",
                effort="medium",
            ),
            "anthropic/opus-5.5+fullcontext+medthink",
        ),
        (
            ModelTags(model="opus", effort="none"),
            "opus+nothink",  # codespell:ignore nothink
        ),
        (
            ModelTags(model="opus", context=258_000, service_tier="priority"),
            "opus+258k+priority",
        ),
        (ModelTags(model="opus", context=258_123), "opus+258123"),
        (
            ModelTags(model="opus", choices={"budget": "auto", "output": "text"}),
            "opus budget=auto output=text",
        ),
    ],
)
def test_a_line_spells_each_tag_its_one_way(tags: ModelTags, line: str) -> None:
    assert str(tags) == line


def test_the_name_is_the_model_and_its_size() -> None:
    """What a provider looks a model up by: the effort and tier are not its to read."""
    tags = ModelTags.parse("work@vendor/opus+258k+high+fast budget=auto")
    assert tags.name == "work@vendor/opus+258k"
    assert ModelTags(model="opus").name == "opus"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("opus+high+low", "'opus+high+low' names two efforts: high, low"),
        ("opus+flex+fast", "'opus+flex+fast' names two service tiers: flex, priority"),
        ("opus auto", "'opus auto' names 'auto', not a CHOICE=VALUE"),
        (
            "opus budget=auto budget=none",
            "'opus budget=auto budget=none' names 'budget' twice",
        ),
        (
            "opus speed=1",
            (
                "speed is no choice; give budget, output, cache_ttl_sec,"
                " manage_context_server_side"
            ),
        ),
    ],
)
def test_parse_refuses_two_of_one_kind_and_unknown_choices(
    line: str,
    message: str,
) -> None:
    with pytest.raises(UnsupportedTagError, match=_exactly(message)):
        ModelTags.parse(line)


@pytest.mark.parametrize(
    ("value", "text"),
    [(True, "true"), (False, "false"), (300.0, "300"), (0.5, "0.5"), ("high", "high")],
)
def test_a_value_is_spelled_as_a_user_types_it(value: object, text: str) -> None:
    assert spelled(value) == text


def test_chosen_reads_each_field_a_setting_names() -> None:
    settings = replace(
        _row().default,
        thinking_effort="high",
        thinking_budget="auto",
        thinking_output="text",
        service_tier="priority",
        cache_ttl_sec=3600.0,
        manage_context_server_side=True,
    )
    fields = (
        "thinking_effort",
        "thinking_budget",
        "thinking_output",
        "service_tier",
        "cache_ttl_sec",
        "manage_context_server_side",
    )
    assert [chosen(settings, f) for f in fields] == [
        "high",
        "auto",
        "text",
        "priority",
        3600.0,
        True,
    ]


def test_settings_name_their_size_their_effort_and_only_choices_made() -> None:
    row = _row(bands=(272_000,))
    small = replace(row.default, context_size=row.prices["auto", "smallcontext"])
    counted = replace(row.default, context_size=row.prices["auto", 100_000])
    lines = [
        str(ModelTags.from_settings(row, row.default)),
        str(ModelTags.from_settings(row, small, provider="vendor")),
        str(ModelTags.from_settings(row, counted, provider="vendor", account="work")),
    ]
    assert lines == [
        "opus-4.8+fullcontext+nothink",  # codespell:ignore nothink
        "vendor/opus-4.8+smallcontext+nothink",  # codespell:ignore nothink
        "work@vendor/opus-4.8+100k+nothink",  # codespell:ignore nothink
    ]


def test_an_unknown_model_names_the_known_models_and_roles() -> None:
    with pytest.raises(UnknownModelError) as raised:
        _table().resolve("nope+1m")
    assert str(raised.value) == (
        "Unknown model 'nope+1m'; known models: opus-5.5, opus-5.0, opus-4.8,"
        " sonnet-5.5, sonnet-5.0, sonnet-4.6, default, utility, best, everyday."
    )


# ---- from_yaml ---------------------------------------------------------------


def _write(
    path: Path,
    *models: dict[str, object],
    vendor: Mapping[str, object] = MappingProxyType({}),
    extra: Mapping[str, object] = MappingProxyType({}),
) -> Path:
    """Write a catalog whose reference supplies one window and every axis."""
    reference = {
        "limits": {"max_request_tokens": 1000},
        "thinking": _THINKING,
    }
    served = {
        "models": {str(model["model_id"]): None for model in models},
        **vendor,
    }
    document = {
        "reference": reference,
        "models": list(models),
        "providers": {"vendor": served},
        **extra,
    }
    path.write_text(yaml.dump(document, Dumper=_Plain))
    return path


def _model(model_id: str = "example-1.0", **fields: object) -> dict[str, object]:
    return {"model_id": model_id, "prices": [dict(_FREE)]} | fields


def _load(path: Path, *, provider: str = "vendor") -> ModelTable:
    return ModelTable.from_yaml(path, provider=provider)


def test_from_yaml_layers_reference_profile_model_section_and_entry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vendor.yaml"
    path.write_text("""
roles: {default: example}
reference:
  approx_chars_per_token: 3.5
  limits: {max_request_tokens: 1_000}
  thinking: {effort: [none], budget: [none], output: [none]}
profiles:
  base:
    knowledge_cutoff: May 2025
    thinking: {effort: [none, high], budget: [none, auto], output: [none, text]}
models:
  - model_id: example-1.0
    wire_model_id: vendor-example
    profile: base
    effort_as_level: true
    thinking: {effort: [none, low, high], budget: [none, auto], output: [none, text]}
    prices:
      - {request: 1.0, response: 2.0, cache_write: 0.0, cache_write_1h: 0.0,
         cache_read: 0.1, tokens_per_unit: 1000}
      - {service_tier: priority, request: 2.0, response: 4.0, cache_write: 0.0,
         cache_write_1h: 0.0, cache_read: 0.2}
      - {service_tier: priority, effective_prompt_tokens: 100,
         effective_date: '2026-10-07', request: 3.0, response: 6.0,
         cache_write: 0.0, cache_write_1h: 0.0, cache_read: 0.3}
  - model_id: unserved-1.0
    prices:
      - {request: 0.0, response: 0.0, cache_write: 0.0, cache_write_1h: 0.0,
         cache_read: 0.0}
providers:
  vendor:
    retries_internally: true
    thinking: {effort: [none, low]}
    models:
      example-1.0: {account_auth: true}
""")
    table = _load(path)
    assert list(table) == [
        "example-1.0",
    ]  # A model the section does not list is not served.
    row = table["default"]
    assert row.wire_model_id == "vendor-example"
    assert row.approx_chars_per_token == 3.5
    assert row.knowledge_cutoff == "May 2025"
    assert row.limits.max_request_tokens == 1000
    # The section's ``thinking`` replaces only the axes it names.
    assert (row.thinking_effort, row.thinking_budget, row.thinking_output) == (
        {"none", "low"},
        {"none", "auto"},
        {"none", "text"},
    )
    assert row.effort_as_level
    assert row.retries_internally
    assert row.account_auth
    assert row.service_tier == {"auto", "default", "priority"}
    assert row.prices.window == 1000
    # A card applies to prompts past its threshold, up to the next card's.
    assert dict(row.prices.prices) == {
        PriceCondition(): TokenPrice(
            request=1.0,
            response=2.0,
            cache_read=0.1,
            tokens_per_unit=1000,
        ),
        PriceCondition(
            service_tier="priority",
            token_size_range=(0, 101),
        ): TokenPrice(request=2.0, response=4.0, cache_read=0.2),
        PriceCondition(
            service_tier="priority",
            token_size_range=(101, math.inf),
            effective_date=date(2026, 10, 7),
        ): TokenPrice(request=3.0, response=6.0, cache_read=0.3),
    }


@pytest.mark.parametrize(
    ("layers", "effort"),
    [
        ({}, "none"),
        ({"reference": "high"}, "high"),
        ({"reference": "high", "vendor": "low"}, "low"),
        ({"reference": "high", "model": "low"}, "low"),
        ({"reference": "high", "vendor": "high", "model": "low"}, "high"),
        ({"vendor": "high", "model": "high", "entry": "low"}, "low"),
    ],
    ids=["unset", "reference", "provider", "model", "provider-over-model", "entry"],
)
def test_the_default_effort_layers_like_any_field(
    tmp_path: Path,
    layers: dict[str, str],
    effort: str,
) -> None:
    def given(layer: str) -> dict[str, object]:
        return {"thinking_effort_default": layers[layer]} if layer in layers else {}

    entry = {"models": {"example-1.0": given("entry")}} if "entry" in layers else {}
    path = _write(
        tmp_path / "x.yaml",
        _model() | given("model"),
        vendor=given("vendor") | entry,
        extra={
            "reference": {
                "limits": {"max_request_tokens": 1000},
                "thinking": _EVERY,
                **given("reference"),
            },
        },
    )
    assert _load(path)["example"].default.thinking_effort == effort


def test_the_default_budget_layers_like_any_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "x.yaml",
        _model() | {"thinking_budget_default": "auto"},
        vendor={"models": {"example-1.0": {"thinking_budget_default": "fixed"}}},
        extra={
            "reference": {
                "limits": {"max_request_tokens": 1000},
                "thinking": _EVERY,
                "thinking_budget_default": "none",
            },
        },
    )
    assert _load(path)["example"].default.thinking_budget == "fixed"


@pytest.mark.parametrize(
    ("default", "effort"),
    [("medium", "low"), ("max", "high"), ("min", "low")],
    ids=["nearest-below", "nearest-below-top", "nearest-above"],
)
def test_a_default_effort_the_row_lacks_falls_to_its_nearest_level(
    tmp_path: Path,
    default: str,
    effort: str,
) -> None:
    path = _write(
        tmp_path / "x.yaml",
        _model(thinking=_EVERY, thinking_effort_default=default),
    )
    assert _load(path)["example"].default.thinking_effort == effort


@pytest.mark.parametrize(
    ("default", "context"),
    [(None, 1000), ("fullcontext", 1000), ("smallcontext", 400), (700, 700)],
    ids=["unset", "fullcontext", "smallcontext", "count"],
)
def test_a_row_names_the_size_a_bare_id_starts_at(
    tmp_path: Path,
    default: str | int | None,
    context: int,
) -> None:
    banded = [dict(_FREE), _FREE | {"effective_prompt_tokens": 400}]
    given: dict[str, object] = (
        {} if default is None else {"context_size_default": default}
    )
    path = _write(
        tmp_path / "x.yaml",
        _model(prices=banded) | given,
        extra={
            "reference": {
                "limits": {"max_request_tokens": 1000},
                "thinking": _THINKING,
            },
        },
    )
    _, settings = _load(path).resolve("example-1.0")
    assert settings.context_size.tokens == context
    _, tagged = _load(path).resolve("example-1.0+300")
    assert tagged.context_size.tokens == 300


def test_a_section_may_narrow_the_tiers_offered(tmp_path: Path) -> None:
    cards = [dict(_FREE), _FREE | {"service_tier": "priority"}]
    path = _write(tmp_path / "x.yaml", _model(prices=cards))
    assert _load(path)["example"].service_tier == {"auto", "default", "priority"}
    path = _write(
        tmp_path / "x.yaml",
        _model(prices=cards),
        vendor={"service_tier": ["auto"]},
    )
    assert _load(path)["example"].service_tier == {"auto"}


@pytest.mark.parametrize("quoted", [True, False])
def test_a_card_starts_on_its_effective_date(tmp_path: Path, quoted: bool) -> None:
    stamp = "'2026-10-07'" if quoted else "2026-10-07"
    path = tmp_path / "x.yaml"
    _write(path, _model())
    text = path.read_text().replace(
        "  prices:\n",
        f"  prices:\n  - {{effective_date: {stamp}, request: 1.0, response: 0.0,"
        " cache_write: 0.0, cache_write_1h: 0.0, cache_read: 0.0}\n",
        1,
    )
    path.write_text(text)
    prices = _load(path)["example"].prices
    assert prices["auto", 1, date(2026, 10, 6)].request == 0.0
    assert prices["auto", 1, date(2026, 10, 7)].request == 1.0


@pytest.mark.parametrize(
    ("models", "sections", "message"),
    [
        ([_model(profile="missing")], {}, "no profile 'missing'"),
        ([_model(typo=1)], {}, "typo"),
        ([_model("opus-5")], {}, "is not family-major.minor"),
        ([_model("Opus-5.0")], {}, "is not family-major.minor"),
        ([_model("opus-5.0.1")], {}, "is not family-major.minor"),
        ([_model(prices=[])], {}, "missing a price or setting"),
        ([_model(thinking=_THINKING | {"effort": []})], {}, "missing a price"),
        ([_model(thinking=_THINKING | {"budget": []})], {}, "missing a price"),
        ([_model(thinking=_THINKING | {"output": []})], {}, "missing a price"),
        ([_model(approx_chars_per_token=0.0)], {}, "missing a price"),
        ([_model(approx_chars_per_token=float("nan"))], {}, "missing a price"),
        ([_model(approx_chars_per_token=float("inf"))], {}, "missing a price"),
        ([_model(thinking={"effort": ["none"]})], {}, _exactly(_WHOLE)),
        (
            [_model()],
            {"profiles": {"p": {"thinking": {"effort": []}}}},
            _exactly(_WHOLE),
        ),
        ([_model()], {"profiles": {"p": {"typo": True}}}, "typo"),
        ([_model(context={"large": {}})], {}, "context"),
        ([_model(context_size_default="huge")], {}, "context_size_default"),
        ([_model(context_size_default=0)], {}, "context_size_default"),
        ([_model(cache_ttl_sec=[])], {}, "^cache_ttl_sec offers nothing selectable"),
        ([_model()], {"unknown": True}, "unknown"),
    ],
)
def test_a_catalog_that_breaks_a_rule_is_rejected(
    tmp_path: Path,
    models: list[dict[str, object]],
    sections: dict[str, object],
    message: str,
) -> None:
    path = _write(tmp_path / "x.yaml", *models, extra=sections)
    with pytest.raises(ValueError, match=message):
        _load(path)


@pytest.mark.parametrize(
    "limit",
    [
        "max_request_tokens",
        "max_response_tokens",
        "max_context_tokens",
        "max_request_bytes",
        "max_image_edge_px",
        "max_image_bytes",
    ],
)
def test_a_negative_limit_is_rejected(tmp_path: Path, limit: str) -> None:
    path = _write(tmp_path / "x.yaml", _model(limits={limit: -1}))
    with pytest.raises(ValueError, match="negative limit"):
        _load(path)


def test_a_window_states_its_input_output_and_shared_ceilings(tmp_path: Path) -> None:
    window = {
        "max_request_tokens": 272_000,
        "max_response_tokens": 128_000,
        "max_context_tokens": 400_000,
        "truncates_at_context": False,
    }
    path = _write(tmp_path / "x.yaml", _model(limits=window))
    assert _load(path)["example"].limits == ModelLimits(
        max_request_tokens=272_000,
        max_response_tokens=128_000,
        max_context_tokens=400_000,
        truncates_at_context=False,
    )


def test_an_unstated_window_behavior_stays_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path / "x.yaml", _model())
    limits = _load(path)["example"].limits
    assert limits.max_context_tokens == 0
    assert limits.truncates_at_context is None


def test_a_shared_window_smaller_than_its_input_is_rejected(tmp_path: Path) -> None:
    window = {"max_request_tokens": 272_000, "max_context_tokens": 200_000}
    path = _write(tmp_path / "x.yaml", _model(limits=window))
    with pytest.raises(ValueError, match="context smaller than its request"):
        _load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokens_per_unit", 0),
        ("effective_prompt_tokens", -1),
        ("request", -1.0),
        ("response", float("nan")),
        ("cache_write", float("inf")),
        ("cache_write_1h", -float("inf")),
        ("cache_read", -1.0),
        ("service_tier", "default"),
    ],
)
def test_an_invalid_price_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _write(tmp_path / "x.yaml", _model(prices=[_FREE | {field: value}]))
    with pytest.raises(ValueError, match="invalid price"):
        _load(path)


def test_a_price_dated_to_the_second_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "x.yaml", _model())
    path.write_text(
        path.read_text().replace(
            "cache_read: 0.0\n",
            "cache_read: 0.0\n    effective_date: 2026-10-07T00:00:00Z\n",
            1,
        ),
    )
    with pytest.raises(ValueError, match="invalid price"):
        _load(path)


@pytest.mark.parametrize(
    "cards",
    [
        [
            dict(_FREE),
            _FREE | {"service_tier": "priority", "effective_prompt_tokens": 1},
        ],
        [
            dict(_FREE),
            _FREE | {"service_tier": "priority", "effective_date": "2027-01-01"},
        ],
        [_FREE | {"effective_prompt_tokens": 100}],
    ],
)
def test_every_priced_tier_needs_a_base_card(
    tmp_path: Path,
    cards: list[dict[str, object]],
) -> None:
    """A tier priced only above a band or after a date leaves requests unpriceable."""
    path = _write(tmp_path / "x.yaml", _model(prices=cards))
    message = (
        f"Model 'example-1.0' has no base card for tier '{cards[-1].get('service_tier', 'auto')}'"
        " (no prompt band, no effective date)."
    )
    with pytest.raises(ValueError, match=_exactly(message)):
        _load(path)


def test_a_card_given_twice_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "x.yaml", _model(prices=[dict(_FREE), dict(_FREE)]))
    with pytest.raises(ValueError, match="prices PriceCondition"):
        _load(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("models: []\nmodels: []", "gives key 'models' twice"),
        ("models: []\n1: bad", _exactly("Catalog mapping keys must be strings.")),
        ("models: &rows []\nprofiles: {bad: {context: *rows}}", _exactly(_ALIASES)),
        ("", "CatalogDocument"),
        ("models: []", "no provider 'vendor'"),
        (
            "models: []\nproviders: {vendor: {models: {ghost-1.0: }}}",
            r"lists unknown models: \['ghost-1.0'\]",
        ),
    ],
)
def test_an_ill_formed_document_is_rejected(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        _load(path)


def test_a_section_naming_an_unknown_thinking_axis_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "x.yaml",
        _model(),
        vendor={"thinking": {"speed": ["fast"]}},
    )
    with pytest.raises(ValueError, match=r"Unknown thinking axes: \['speed'\]"):
        _load(path)


def test_an_alias_inside_a_list_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(
        "reference: {thinking: {effort: [none], budget: [none], output: [none]}}\n"
        "models:\n"
        "- {model_id: a-1.0, prices: [&card {request: 0, response: 0, cache_write: 0,"
        " cache_write_1h: 0, cache_read: 0}]}\n"
        "- {model_id: b-1.0, prices: [*card]}\n",
    )
    with pytest.raises(ValueError, match=_exactly(_ALIASES)):
        _load(path)


def test_a_section_listing_no_models_serves_none(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("models: []\nproviders: {vendor: {}}")
    assert len(_load(path)) == 0


def test_a_catalog_without_thinking_offers_none(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(
        "models:\n- {model_id: a-1.0, limits: {}, prices: [{request: 0,"
        " response: 0, cache_write: 0, cache_write_1h: 0, cache_read: 0}]}\n"
        "providers: {vendor: {models: {a-1.0: }}}\n",
    )
    row = _load(path)["a"]
    assert (row.thinking_effort, row.thinking_budget, row.thinking_output) == (
        {"none"},
        {"none"},
        {"none"},
    )


def test_a_section_narrows_each_thinking_axis_and_states_transport_facts(
    tmp_path: Path,
) -> None:
    thinking = {
        "effort": ["none", "high"],
        "budget": ["none", "auto"],
        "output": ["none", "text"],
    }
    vendor = {
        "thinking": {"budget": ["none"], "output": ["text"]},
        "cache_ttl_sec": [300.0, 3600.0],
        "manage_context_server_side": [True],
    }
    path = _write(tmp_path / "x.yaml", _model(thinking=thinking), vendor=vendor)
    row = _load(path)["example"]
    assert (row.thinking_effort, row.thinking_budget, row.thinking_output) == (
        {"none", "high"},
        {"none"},
        {"text"},
    )
    assert row.cache_ttl_sec == {300.0, 3600.0}
    assert row.manage_context_server_side == {True}


@pytest.mark.parametrize(("estimate", "unit"), [(0.5, 1), (1.0, 1)])
def test_small_positive_estimates_and_units_are_valid(
    tmp_path: Path,
    estimate: float,
    unit: int,
) -> None:
    card = _FREE | {"tokens_per_unit": unit}
    path = _write(
        tmp_path / "x.yaml",
        _model(approx_chars_per_token=estimate, prices=[card]),
    )
    row = _load(path)["example"]
    assert row.approx_chars_per_token == estimate
    assert row.prices["auto", 1, date.min].tokens_per_unit == unit


def test_smallcontext_is_the_top_of_the_cheapest_price_band(tmp_path: Path) -> None:
    """GPT's 1.05M window bills double past 272k: ``+smallcontext`` stays below that."""
    cheap, dear = (
        dict(_FREE),
        _FREE | {"request": 1.0, "effective_prompt_tokens": 272_000},
    )
    path = _write(
        tmp_path / "x.yaml",
        _model(limits={"max_request_tokens": 1_050_000}, prices=[cheap, dear]),
    )
    capability, settings = _load(path).resolve("example+smallcontext")
    assert settings.context_size.tokens == 272_000
    assert capability.limits.max_request_tokens == 1_050_000
    # Vendors publish "prompts > 272k": exactly 272k is still the base band.
    assert capability.prices["auto", 272_000, date.min].request == 0.0
    assert capability.prices["auto", 272_001, date.min].request == 1.0


@pytest.mark.parametrize("model_id", ["opus-x.5", "opus-5.x", "opus-5."])
def test_a_version_must_be_major_dot_minor_digits(
    tmp_path: Path,
    model_id: str,
) -> None:
    path = _write(tmp_path / "x.yaml", _model(model_id))
    message = f"Model id {model_id!r} is not family-major.minor."
    with pytest.raises(ValueError, match=_exactly(message)):
        _load(path)


def test_two_spellings_of_one_version_are_out_of_order() -> None:
    with pytest.raises(ValueError, match="newest first"):
        ModelTable(rows=(_row("a-1.0"), _row("a-1.00")))


def test_a_family_with_a_major_reaches_a_hyphenated_family() -> None:
    table = ModelTable(rows=(_row("flash-lite-3.5"), _row("flash-3.8")))
    assert table["flash-lite-3"].model_id == "flash-lite-3.5"


def test_a_section_s_roles_replace_the_document_s_it_names(tmp_path: Path) -> None:
    """A provider that serves no ``pro`` defaults to what it does serve."""
    path = _write(
        tmp_path / "x.yaml",
        _model("pro-1.0"),
        _model("flash-1.0"),
        vendor={"roles": {"default": "flash"}},
        extra={"roles": {"default": "pro", "utility": "flash"}},
    )
    table = _load(path)
    assert (table["default"].model_id, table["utility"].model_id) == (
        "flash-1.0",
        "flash-1.0",
    )
    assert table["best"].model_id == "flash-1.0"


@pytest.mark.parametrize("target", ["local-model", "claude-opus-4-8"])
def test_a_role_may_name_an_unversioned_id_or_a_vendor_id(target: str) -> None:
    table = ModelTable(rows=(_row(), _row("local-model")), roles={"default": target})
    assert table["default"] is table.exact(target)


class _Plain(yaml.SafeDumper):
    """Writes a value shared between two places twice, never as an alias."""

    @override
    def ignore_aliases(self, data: object) -> bool:
        del data
        return True


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
