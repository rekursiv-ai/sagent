"""Provider document loading and validation tests."""

from datetime import date
from pathlib import Path
from typing import Final, cast, get_args

import re

import pytest
import yaml

from sagent import catalog
from sagent.catalog.custom_typings import ProviderName
from sagent.catalog.loader import load_catalog
from sagent.catalog.table import ModelCatalog
from sagent.types.capability import ModelCapability, ThinkingCapability
from sagent.types.cost import PriceKey, TokenPrice


_REFERENCE: Final = """
reference:
  context:
    max: {max_request_tokens: 1_000_000, max_response_tokens: 128_000}
"""

_FREE: Final = {
    "request": 0.0,
    "response": 0.0,
    "cache_write": 0.0,
    "cache_write_1h": 0.0,
    "cache_read": 0.0,
}

_THINKING: Final = {"effort": ["none"], "budget": ["none"], "output": ["none"]}


def _write(path: Path, *models: dict[str, object], **sections: object) -> Path:
    """Write a document whose reference supplies one context and full thinking."""
    reference = {
        "context": {"max": {"max_request_tokens": 1000}},
        "thinking": _THINKING,
    }
    document = {"reference": reference, "models": list(models)} | sections
    path.write_text(yaml.safe_dump(document))
    return path


def _row(model_id: str, **fields: object) -> dict[str, object]:
    # A fresh card per row: a shared dict serializes as a YAML alias.
    return {"model_id": model_id, "prices": [dict(_FREE)]} | fields


@pytest.mark.parametrize(
    "ids",
    [("a-1.0", "a-2.0"), ("a-2.0", "a-10.0"), ("a-1.0", "a-1.10")],
)
def test_a_family_must_list_its_rows_newest_first(
    tmp_path: Path,
    ids: tuple[str, ...],
) -> None:
    """A role or prefix means the family's first row, so order is the meaning."""
    path = _write(tmp_path / "x.yaml", *(_row(i) for i in ids))
    with pytest.raises(ValueError, match="newest-first"):
        load_catalog(path)


def test_families_may_interleave(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "x.yaml",
        _row("a-2.0"),
        _row("b-1.0"),
        _row("a-1.0"),
        roles={"default": "a"},
    )
    assert load_catalog(path)["default"].model_id == "a-2.0"


@pytest.mark.parametrize(
    "model_id",
    ["opus-5", "Opus-5.0", "opus-5.0.1", "opus", ""],
)
def test_a_name_is_family_major_minor(tmp_path: Path, model_id: str) -> None:
    """``opus-5`` is the prefix for the newest ``opus-5.x``; no row may own it."""
    with pytest.raises(ValueError, match=r"family-major\.minor"):
        load_catalog(_write(tmp_path / "x.yaml", _row(model_id)))


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
def test_every_priced_tier_has_a_base_card(
    tmp_path: Path,
    cards: list[dict[str, object]],
) -> None:
    """A tier priced only above a band or after a date leaves requests unpriceable."""
    path = _write(tmp_path / "x.yaml", _row("a-1.0", prices=cards))
    with pytest.raises(ValueError, match="base card"):
        load_catalog(path)


@pytest.mark.parametrize("layer", ["reference", "profile", "model"])
def test_a_partial_thinking_override_is_rejected(tmp_path: Path, layer: str) -> None:
    """An omitted axis would silently become ``none``, not inherit."""
    partial = {"effort": ["none", "low"]}
    path = tmp_path / "x.yaml"
    match layer:
        case "reference":
            path.write_text(
                yaml.safe_dump(
                    {
                        "reference": {
                            "context": {"max": {"max_request_tokens": 1}},
                            "thinking": partial,
                        },
                        "models": [_row("a-1.0")],
                    },
                ),
            )
        case "profile":
            _write(
                path,
                _row("a-1.0", profile="p"),
                profiles={"p": {"thinking": partial}},
            )
        case _:
            _write(path, _row("a-1.0", thinking=partial))
    with pytest.raises(ValueError, match="thinking must state effort, budget, output"):
        load_catalog(path)


def test_profile_resolution(tmp_path: Path) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text("""
roles: {default: example}
reference:
  approx_chars_per_token: 3.5
  default_context: "max"
profiles:
  base:
    context:
      "max": {max_request_tokens: 1_000}
    thinking:
      effort: [none, high]
      budget: [none, auto]
      output: [none, text]
models:
  - model_id: example-1.0
    wire_model_id: vendor-example
    profile: base
    knowledge_cutoff: June 2026
    effort_as_level: true
    thinking: {effort: [none, low], budget: [none, auto], output: [none, text]}
    prices:
      - service_tier: auto
        tokens_per_unit: 1000
        request: 1.0
        response: 2.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.1
      - service_tier: priority
        request: 2.0
        response: 4.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.2
      - service_tier: priority
        effective_prompt_tokens: 100
        effective_date: "2026-10-07"
        request: 2.0
        response: 4.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.2
""")
    table = load_catalog(path)
    row = table["default"]
    assert row.model_id == "example-1.0"
    assert row.wire_model_id == "vendor-example"
    assert row.approx_chars_per_token == 3.5
    assert row.context[""].max_request_tokens == 1000
    assert row.thinking == ThinkingCapability(
        effort=frozenset({"none", "low"}),
        budget=frozenset({"none", "auto"}),
        output=frozenset({"none", "text"}),
    )
    assert row.knowledge_cutoff == "June 2026"
    assert row.effort_as_level is True
    assert row.service_tier == {"auto", "default", "priority"}
    assert dict(row.prices) == {
        PriceKey("auto"): TokenPrice(
            request=1.0,
            response=2.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.1,
            tokens_per_unit=1000,
        ),
        PriceKey("priority"): TokenPrice(
            request=2.0,
            response=4.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.2,
        ),
        PriceKey("priority", 100, date(2026, 10, 7)): TokenPrice(
            request=2.0,
            response=4.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.2,
        ),
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("models: []\nunknown: true", r"unknown field\(s\) \['unknown'\]"),
        (
            "models: [{model_id: example-1.0, profile: missing, prices: []}]",
            "Unknown catalog profile 'missing'",
        ),
        (
            "models: [{model_id: example-1.0, prices: [{service_tier: auto}]}]",
            "cannot build PriceRow",
        ),
        (
            "models: [{model_id: example-1.0, prices: [], typo: 1}]",
            r"unknown field\(s\) \['typo'\]",
        ),
        (
            (
                "models: [{model_id: example-1.0, prices: [],"
                " thinking: {effort: [bogus], budget: [none], output: [none]}}]"
            ),
            "bogus",
        ),
        ("models: []\nmodels: []", "Duplicate YAML key 'models'"),
        ("", "CatalogDocument"),
        ("models: []\nprofiles: {bad: {typo: true}}", r"unknown field\(s\) \['typo'\]"),
        ("models: []\n1: bad", "mapping keys must be strings"),
        ("models: &rows []\nprofiles: {bad: {context: *rows}}", "aliases"),
        (
            'models: [{model_id: example-1.0, prices: [], context: {"min": {}}}]',
            "requires a default context",
        ),
        (
            _REFERENCE
            + "models: [{model_id: example-1.0, prices: [{request: 0, response: 0,"
            " cache_write: 0, cache_write_1h: 0, cache_read: 0}]},"
            " {model_id: example-1.0, prices: [{request: 0, response: 0,"
            " cache_write: 0, cache_write_1h: 0, cache_read: 0}]}]",
            "duplicate model id 'example-1.0'",
        ),
    ],
)
def test_invalid_document(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_catalog(path)


@pytest.mark.parametrize("quoted", [True, False])
def test_explicit_prices_and_duplicate_keys(tmp_path: Path, quoted: bool) -> None:
    path = tmp_path / "provider.yaml"
    stamp = '"2026-10-07"' if quoted else "2026-10-07"
    card = f"""
      - service_tier: auto
        effective_date: {stamp}
        request: 1.0
        response: 2.0
        cache_write: 3.0
        cache_write_1h: 4.0
        cache_read: 5.0
"""
    base = """
      - request: 0.0
        response: 0.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.0
"""
    prefix = _REFERENCE + "models:\n  - model_id: example-1.0\n    prices:\n" + base
    path.write_text(prefix + card)
    row = load_catalog(path)["example"]
    assert dict(row.prices) == {
        PriceKey("auto"): TokenPrice(
            request=0.0,
            response=0.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.0,
        ),
        PriceKey("auto", effective_date=date(2026, 10, 7)): TokenPrice(
            request=1.0,
            response=2.0,
            cache_write=3.0,
            cache_write_1h=4.0,
            cache_read=5.0,
        ),
    }
    path.write_text(prefix + card + card)
    with pytest.raises(ValueError, match="Duplicate price key"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("profile: missing", "Unknown catalog profile 'missing'"),
        ('context: {"min": {}}', "requires a default context"),
    ],
)
def test_resolution_errors(tmp_path: Path, model: str, message: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(f"models:\n  - model_id: example-1.0\n    prices: []\n    {model}")
    with pytest.raises(ValueError, match=message):
        load_catalog(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokens_per_unit", "0"),
        ("effective_prompt_tokens", "-1"),
        ("request", "-1.0"),
        ("response", ".nan"),
        ("cache_write", ".inf"),
        ("cache_write_1h", "-.inf"),
        ("cache_read", "-1.0"),
        ("service_tier", "default"),
        ("effective_date", "2026-10-07T00:00:00Z"),
    ],
)
def test_invalid_price_domains(tmp_path: Path, field: str, value: str) -> None:
    path = tmp_path / "provider.yaml"
    fields = {
        "service_tier": "auto",
        "request": "1.0",
        "response": "2.0",
        "cache_write": "0.0",
        "cache_write_1h": "0.0",
        "cache_read": "0.1",
    }
    fields[field] = value
    card = "\n".join(f"        {name}: {raw}" for name, raw in fields.items())
    path.write_text(
        _REFERENCE
        + "models:\n  - model_id: example-1.0\n    prices:\n      -\n"
        + card,
    )
    with pytest.raises(ValueError, match="Invalid catalog price"):
        load_catalog(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prices": []},
        {"thinking": _THINKING | {"effort": []}},
        {"thinking": _THINKING | {"budget": []}},
        {"thinking": _THINKING | {"output": []}},
        {"approx_chars_per_token": 0.0},
        {"approx_chars_per_token": float("nan")},
        {"approx_chars_per_token": float("inf")},
    ],
)
def test_unusable_model(tmp_path: Path, overrides: dict[str, object]) -> None:
    path = tmp_path / "provider.yaml"
    model: dict[str, object] = {
        "model_id": "example-1.0",
        "prices": [
            {
                "request": 1.0,
                "response": 2.0,
                "cache_write": 0.0,
                "cache_write_1h": 0.0,
                "cache_read": 0.1,
            },
        ],
    }
    path.write_text(_REFERENCE + yaml.safe_dump({"models": [model | overrides]}))
    with pytest.raises(ValueError, match="Unusable catalog model"):
        load_catalog(path)


@pytest.mark.parametrize("estimate", [0.5, 1.0])
def test_positive_token_estimates(tmp_path: Path, estimate: float) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(f"""
reference:
  approx_chars_per_token: {estimate}
  default_context: "max"
  context:
    "max": {{max_request_tokens: 1_000_000}}
models:
  - model_id: example-1.0
    prices:
      - request: 0.0
        response: 0.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.0
""")
    assert load_catalog(path)["example"].approx_chars_per_token == estimate


@pytest.mark.parametrize("default", ["max", "min"])
def test_named_default_context(tmp_path: Path, default: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(f"""
reference:
  default_context: "max"
  context:
    "max": {{max_request_tokens: 1_000_000}}
    "min": {{max_request_tokens: 200_000}}
models:
  - model_id: example-1.0
    default_context: "{default}"
    prices:
      - request: 0.0
        response: 0.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.0
""")
    row = load_catalog(path)["example"]
    assert row.context[""].max_request_tokens == (
        1_000_000 if default == "max" else 200_000
    )
    other = "+200k" if default == "max" else "+1m"
    assert other in row.context
    assert len(row.context) == 2
    path.write_text(path.read_text().replace(f'"{default}"', '"unqualified"'))
    with pytest.raises(ValueError, match="max/min presets"):
        load_catalog(path)


@pytest.mark.parametrize("default", ["max", "min"])
@pytest.mark.parametrize("windows", [(922_000, 123_456), (0, 0)])
def test_context_presets_accept_arbitrary_windows(
    tmp_path: Path,
    default: str,
    windows: tuple[int, int],
) -> None:
    maximum, minimum = windows
    path = tmp_path / "provider.yaml"
    path.write_text(f"""
reference:
  default_context: {default}
  context:
    max: {{max_request_tokens: {maximum}}}
    min: {{max_request_tokens: {minimum}}}
models:
  - model_id: example-1.0
    prices:
      - request: 0.0
        response: 0.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.0
""")
    table = load_catalog(path)
    row = table["example"]
    assert row.context[""].max_request_tokens == (
        maximum if default == "max" else minimum
    )
    assert {limits.max_request_tokens for limits in row.context.values()} == set(
        windows,
    )
    assert len(row.context) == 2
    runtime = ModelCatalog(models=table, transport=ModelCapability())
    for tag, expected in (("+max", maximum), ("+min", minimum)):
        capability, settings = runtime.resolve(f"example{tag}")
        assert capability.context[settings.context].max_request_tokens == expected


@pytest.mark.parametrize(
    "context",
    [
        {"max": {field: -1}}
        for field in (
            "max_request_tokens",
            "max_response_tokens",
            "max_request_bytes",
            "max_image_edge_px",
            "max_image_bytes",
        )
    ]
    + [
        {
            "max": {"max_request_tokens": 200_000},
            "min": {"max_request_tokens": 1_000_000},
        },
    ],
)
def test_invalid_context_domains(
    tmp_path: Path,
    context: dict[str, dict[str, int]],
) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(
        yaml.safe_dump({"reference": {"context": context}})
        + """
models:
  - model_id: example-1.0
    prices:
      - request: 0.0
        response: 0.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.0
""",
    )
    with pytest.raises(ValueError, match="Invalid catalog context"):
        load_catalog(path)


@pytest.mark.parametrize("provider", get_args(ProviderName.__value__))
def test_provider_rates_are_written_as_published(provider: ProviderName) -> None:
    """Vendors publish at most 4 decimals; ``1.7999999999999998`` is arithmetic."""
    text = (Path(catalog.__file__).parent / "assets" / f"{provider}.yaml").read_text()
    floats = re.findall(r"\b\d+\.\d+\b", text)
    assert [x for x in floats if len(x.partition(".")[2]) > 4] == []


_ROW_KEY_ORDER: Final = (
    "model_id",
    "wire_model_id",
    "profile",
    "knowledge_cutoff",
    "approx_chars_per_token",
    "effort_as_level",
    "default_context",
    "context",
    "thinking",
    "prices",
)


def _asset(provider: ProviderName) -> dict[str, object]:
    path = Path(catalog.__file__).parent / "assets" / f"{provider}.yaml"
    return cast(dict[str, object], yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("provider", get_args(ProviderName.__value__))
def test_provider_rows_list_fields_in_one_order(provider: ProviderName) -> None:
    for row in cast(list[dict[str, object]], _asset(provider)["models"]):
        keys = list(row)
        assert keys == sorted(keys, key=_ROW_KEY_ORDER.index), row["model_id"]


@pytest.mark.parametrize("provider", get_args(ProviderName.__value__))
def test_profiles_are_shape_names_and_all_used(provider: ProviderName) -> None:
    """A model-named profile (``gemini-pro-3.1``) on a Flash row reads as a typo."""
    document = _asset(provider)
    rows = cast(list[dict[str, str]], document["models"])
    used = {row["profile"] for row in rows if "profile" in row}
    profiles = set(cast(dict[str, object], document.get("profiles", {})))
    assert profiles == used
    assert [name for name in profiles if not re.fullmatch(r"[a-z0-9_]+", name)] == []


@pytest.mark.parametrize("provider", get_args(ProviderName.__value__))
def test_provider_assets_load_and_lookup_is_cached(provider: ProviderName) -> None:
    table = catalog.models(provider)
    assert table
    assert catalog.models(provider) is table
    for role in ("best", "default", "everyday", "utility"):
        assert table[role].model_id in table
    for model_id, row in table.items():
        assert row.model_id == model_id
        assert row.prices
        assert row.service_tier == row.prices.service_tiers
        assert row.thinking.effort
        assert row.thinking.budget
        assert row.thinking.output
        assert "" in row.context
        for limits in row.context.values():
            assert limits.max_request_tokens > 0
            assert limits.max_response_tokens > 0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
