"""Tests for ``catalog.table``: model lookup by id, role, family; resolution."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from sagent.catalog.table import (
    CONTEXT_TAGS,
    ModelCatalog,
    ModelTable,
    UnknownModelError,
    UnsupportedTagError,
    base_model_id,
    split_model_id,
)
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingCapability,
)
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice


def _opus(model_id: str = "opus-4.8") -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        wire_model_id="claude-" + model_id.replace(".", "-"),
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
                PriceKey("auto"): TokenPrice(
                    request=5.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
        thinking=ThinkingCapability(effort=frozenset({"none", "max"})),
        service_tier=frozenset({"auto", "default"}),
    )


def _cli() -> ModelCapability:
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
    )


def _table() -> ModelTable:
    """Newest-first within each family, as every vendor catalog orders rows."""
    return ModelTable(
        rows=(
            _opus("opus-5.5"),
            _opus("opus-5.0"),
            _opus("opus-4.8"),
            _opus("sonnet-5.5"),
            _opus("sonnet-5.0"),
            _opus("sonnet-4.6"),
        ),
        roles={"default": "opus", "utility": "sonnet-4.6"},
    )


def _resolve(
    model_id: str,
    *,
    rows: tuple[ModelCapability, ...] = (),
    transport: ModelCapability | None = None,
) -> tuple[ModelCapability, ModelSettings]:
    table = ModelTable(rows=rows or (_opus(),))
    return ModelCatalog(
        models=table,
        transport=transport or ModelCapability(),
    ).resolve(model_id)


# ---- ModelTable lookup -----------------------------------------------------


@pytest.mark.parametrize(
    ("key", "model_id"),
    [
        ("sonnet", "sonnet-5.5"),
        ("sonnet-5", "sonnet-5.5"),
        ("sonnet-5.5", "sonnet-5.5"),
        ("sonnet-5.0", "sonnet-5.0"),
        ("sonnet-4", "sonnet-4.6"),
        ("opus-5.0", "opus-5.0"),
        ("claude-opus-4-8", "opus-4.8"),
        ("default", "opus-5.5"),
        ("utility", "sonnet-4.6"),
    ],
)
def test_a_key_names_an_exact_name_a_wire_id_a_role_or_a_prefix(
    key: str,
    model_id: str,
) -> None:
    assert _table()[key].model_id == model_id


def test_an_exact_name_outranks_a_role_of_the_same_spelling() -> None:
    """The module's stated precedence: exact name, wire id, role, prefix."""
    table = ModelTable(rows=(_opus("a-1.0"), _opus("b-1.0")), roles={"a-1.0": "b-1.0"})
    assert table["a-1.0"].model_id == "a-1.0"
    assert table.get("a-1.0") is table.rows[0]


def test_context_tags_are_the_alias_members_without_the_default() -> None:
    assert CONTEXT_TAGS == ("+200k", "+272k", "+1m", "+max", "+min")


def test_a_prefix_stops_at_a_separator() -> None:
    assert "opu" not in _table()
    assert "sonnet-5.5x" not in _table()
    assert "sonnet-5.5.1" not in _table()


def test_a_wire_id_outranks_a_family_of_the_same_spelling() -> None:
    """A server reporting ``flash`` means that model, not the family's newest."""
    old = replace(_opus("flash-1.0"), wire_model_id="flash")
    table = ModelTable(rows=(_opus("flash-2.0"), old))
    assert table["flash"].model_id == "flash-1.0"
    assert table["flash-2"].model_id == "flash-2.0"


def test_a_family_never_reaches_a_longer_family_that_starts_with_it() -> None:
    table = ModelTable(
        rows=(
            _opus("flash-lite-3.5"),
            _opus("flash-3.8"),
            _opus("flash-2.5"),
        ),
    )
    assert table["flash"].model_id == "flash-3.8"
    assert table["flash-2"].model_id == "flash-2.5"
    assert table["flash-lite"].model_id == "flash-lite-3.5"
    assert "flash-lite-2" not in table


def test_iteration_lists_exact_names_only() -> None:
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
    assert "default" in table
    assert "sonnet-5" in table
    assert "sonnet-5" not in list(table)


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("sonnet-5.0", "sonnet-5.0"),
        ("claude-opus-4-8", "opus-4.8"),
        ("sonnet", None),
        ("sonnet-5", None),
        ("default", None),
        ("haiku", None),
    ],
)
def test_exact_matches_only_an_exact_name_or_a_wire_id(
    name: str,
    model_id: str | None,
) -> None:
    row = _table().exact(name)
    assert (row.model_id if row is not None else None) == model_id


def test_an_unknown_key_is_a_key_error() -> None:
    with pytest.raises(KeyError):
        _ = _table()["haiku"]
    assert _table().get("haiku") is None


def test_a_role_naming_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="role 'default'"):
        _ = ModelTable(rows=(_opus(),), roles={"default": "haiku"})


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"duplicate model id 'opus-4\.8'"):
        _ = ModelTable(rows=(_opus(), _opus()))


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
    with pytest.raises(ValueError, match="newest-first"):
        _ = ModelTable(rows=tuple(_opus(i) for i in ids))


def test_a_caller_supplied_unversioned_id_is_an_exact_name_only() -> None:
    """A local endpoint's ``local-model`` has no family to order or prefix."""
    table = ModelTable(rows=(_opus("local-model"), _opus("local-model-b")))
    assert table["local-model"].model_id == "local-model"
    assert table["local-model-b"].model_id == "local-model-b"


def test_best_and_everyday_fall_back_to_default_when_undeclared() -> None:
    """A vendor with nothing above or below its default still answers both."""
    table = _table()
    assert table["best"].model_id == "opus-5.5"
    assert table["everyday"].model_id == "opus-5.5"
    assert table.roles["best"] == table.roles["everyday"] == "opus"


def test_a_declared_best_or_everyday_outranks_the_fallback() -> None:
    table = ModelTable(
        rows=(_opus("opus-5.5"), _opus("sonnet-5.5")),
        roles={"best": "opus", "default": "opus", "everyday": "sonnet"},
    )
    assert table["everyday"].model_id == "sonnet-5.5"
    assert table["best"].model_id == "opus-5.5"


def test_without_a_default_nothing_falls_back() -> None:
    table = ModelTable(rows=(_opus(),))
    assert "best" not in table
    assert "everyday" not in table
    assert dict(table.roles) == {}


def test_roles_are_frozen() -> None:
    roles = {"default": "opus"}
    table = ModelTable(rows=(_opus(),), roles=roles)
    roles["default"] = "nope"
    assert table.roles["default"] == "opus"
    assert type(table.roles) is MappingProxyType


# ---- model-id tags ---------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "base", "tags"),
    [
        ("opus-4.8", "opus-4.8", frozenset[ContextTag]()),
        ("opus-4.8+1m", "opus-4.8", frozenset({"+1m"})),
        ("opus-4.8+200k", "opus-4.8", frozenset({"+200k"})),
        ("astra-6.0+272k", "astra-6.0", frozenset({"+272k"})),
        ("Opus-4.8+1M", "Opus-4.8", frozenset({"+1m"})),
        ("model+unknown", "model+unknown", frozenset[ContextTag]()),
        # ``+fast`` was a second spelling of ``service_tier="priority"``.
        ("opus-5+fast", "opus-5+fast", frozenset[ContextTag]()),
        ("Opus-4.7+200k+1M", "Opus-4.7", frozenset({"+200k", "+1m"})),
    ],
)
def test_split_model_id(model_id: str, base: str, tags: frozenset[ContextTag]) -> None:
    assert split_model_id(model_id) == (base, tags)
    assert base_model_id(model_id) == base


def test_context_tags_derive_from_the_literal() -> None:
    """A tag the type admits but the tuple omits would be unparseable."""
    assert set(CONTEXT_TAGS) == {"+1m", "+272k", "+200k", "+max", "+min"}


@pytest.mark.parametrize("alias", ["+max", "+min", "+MAX", "+MIN"])
def test_split_context_extreme_aliases(alias: str) -> None:
    assert split_model_id("opus-4.8" + alias) == (
        "opus-4.8",
        frozenset({alias.lower()}),
    )


@pytest.mark.parametrize(
    ("alias", "default_tokens", "tagged_tokens", "expected"),
    [
        ("+max", 200_000, 1_000_000, "+1m"),
        ("+min", 200_000, 1_000_000, ""),
        ("+max", 1_000_000, 200_000, ""),
        ("+min", 1_000_000, 200_000, "+1m"),
        ("+max", 200_000, 200_000, "+1m"),
        ("+min", 200_000, 200_000, "+1m"),
    ],
)
def test_extreme_alias_selects_existing_context_by_token_limit(
    alias: str,
    default_tokens: int,
    tagged_tokens: int,
    expected: ContextTag,
) -> None:
    row = replace(
        _opus(),
        context=MappingProxyType(
            {
                "+1m": ModelLimits(max_request_tokens=tagged_tokens),
                "": ModelLimits(max_request_tokens=default_tokens),
            },
        ),
    )
    capability, settings = _resolve("opus-4.8" + alias, rows=(row,))
    assert settings.context == expected
    assert settings.limits == row.context[expected]
    assert tuple(capability.context.items()) == tuple(row.context.items())


@pytest.mark.parametrize("alias", ["+max", "+min"])
def test_extreme_alias_selects_sole_context(alias: str) -> None:
    row = replace(_opus(), context=MappingProxyType({"": ModelLimits()}))
    capability, settings = _resolve("opus-4.8" + alias, rows=(row,))
    assert settings.context == ""
    assert tuple(capability.context.items()) == tuple(row.context.items())


@pytest.mark.parametrize("suffix", ["+max+min", "+max+1m", "+200k+min"])
def test_extreme_alias_conflicts_with_other_context_tags(suffix: str) -> None:
    with pytest.raises(UnsupportedTagError, match="conflicting contexts"):
        _resolve("opus-4.8" + suffix)


# ---- ModelCatalog.resolve --------------------------------------------------


def test_resolve_returns_capability_and_settings_as_peers() -> None:
    capability, settings = _resolve("opus-4.8+1m")
    assert settings.context == "+1m"
    assert capability.context.keys() == {"", "+1m"}
    assert settings.capability == capability


def test_resolve_keeps_the_whole_context_table() -> None:
    _, settings = _resolve("opus-4.8")
    assert settings.limits.max_request_tokens == 200_000
    wide = ModelSettings(capability=settings.capability, context="+1m")
    assert wide.limits.max_request_tokens == 1_000_000


def test_resolve_meets_the_transport() -> None:
    capability, _ = _resolve("opus-4.8", transport=_cli())
    assert capability.thinking.effort == frozenset({"none"})
    assert capability.manage_context_server_side == frozenset({True})


def test_resolve_never_grants_what_the_row_lacks() -> None:
    capability, _ = _resolve(
        "opus-4.8",
        transport=ModelCapability(
            thinking=ThinkingCapability(
                effort=frozenset({"none", "min", "low", "medium", "high", "max"}),
            ),
        ),
    )
    assert capability.thinking.effort == frozenset({"none", "max"})


def test_resolve_rejects_a_context_the_model_lacks_naming_those_it_offers() -> None:
    row = replace(
        _opus(),
        context=MappingProxyType(
            {**_opus().context, "+200k": ModelLimits(max_request_tokens=200_000)},
        ),
    )
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve("opus-4.8+272k", rows=(row,))
    assert str(raised.value) == (
        "Model 'opus-4.8+272k' has no +272k context; offers: +1m, +200k"
    )


def test_resolve_says_none_when_the_model_offers_no_tagged_context() -> None:
    row = replace(_opus(), context=MappingProxyType({"": ModelLimits()}))
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve("opus-4.8+200k", rows=(row,))
    assert str(raised.value) == (
        "Model 'opus-4.8+200k' has no +200k context; offers: (none)"
    )


@pytest.mark.parametrize("model_id", ["opus-4.8+1m+200k", "opus-4.8+200k+1m"])
def test_resolve_rejects_conflicting_context_tags(model_id: str) -> None:
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve(model_id)
    assert str(raised.value) == (
        f"Model {model_id!r} selects conflicting contexts: +1m, +200k"
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
    capability, settings = _resolve("opus-4.8+1m", rows=(row,))
    assert settings.context == "+1m"
    assert settings.limits == capability.context[""]
    untagged, _ = _resolve("opus-4.8", rows=(row,))
    assert untagged.context.keys() == {""}


def test_a_transport_cannot_remove_a_context_window() -> None:
    """Windows are the model's; a transport restricts knobs, not physics."""
    narrow = ModelCapability(context=MappingProxyType({"": ModelLimits()}))
    _, settings = _resolve("opus-4.8+1m", transport=narrow)
    assert settings.limits.max_request_tokens == 1_000_000


def test_resolve_names_the_known_models_and_roles_on_a_miss() -> None:
    catalog = ModelCatalog(models=_table(), transport=ModelCapability())
    with pytest.raises(UnknownModelError) as raised:
        _ = catalog.resolve("nope")
    assert str(raised.value) == (
        "Unknown model 'nope'. Known models: opus-5.5, opus-5.0, opus-4.8,"
        " sonnet-5.5, sonnet-5.0, sonnet-4.6, default, utility, best, everyday"
    )


def test_resolve_follows_a_role_and_a_family_with_a_tag() -> None:
    catalog = ModelCatalog(models=_table(), transport=ModelCapability())
    assert catalog.resolve("utility")[0].model_id == "sonnet-4.6"
    capability, settings = catalog.resolve("opus-5+1m")
    assert capability.model_id == "opus-5.5"
    assert settings.context == "+1m"


def test_resolve_accepts_a_vendor_id() -> None:
    capability, _ = _resolve("claude-opus-4-8")
    assert capability.model_id == "opus-4.8"
    assert capability.wire_model_id == "claude-opus-4-8"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
