"""Provider document loading and validation tests."""

from datetime import date
from pathlib import Path
from typing import Final, get_args

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
    thinking: {effort: [none, low]}
    prices:
      - service_tier: auto
        tokens_per_unit: 1000
        request: 1.0
        response: 2.0
        cache_write: 0.0
        cache_write_1h: 0.0
        cache_read: 0.1
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
    assert row.thinking == ThinkingCapability(effort=frozenset({"none", "low"}))
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
        PriceKey("priority", 100, date(2026, 10, 7)): TokenPrice(
            request=2.0,
            response=4.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.2,
        ),
    }


@pytest.mark.parametrize(
    "text",
    [
        "models: []\nunknown: true",
        "models: [{model_id: example-1.0, profile: missing, prices: []}]",
        "models: [{model_id: example-1.0, prices: [{service_tier: auto}]}]",
        "models: [{model_id: example-1.0, prices: [], typo: 1}]",
        "models: [{model_id: example-1.0, prices: [], thinking: {effort: [bogus]}}]",
        "models: []\nmodels: []",
        "",
        "models: []\nprofiles: {bad: {typo: true}}",
        "models: []\n1: bad",
        "models: &rows []\nprofiles: {bad: {context: *rows}}",
        'models: [{model_id: example-1.0, prices: [], context: {"min": {}}}]',
        (
            "models: [{model_id: example-1.0, prices: []}, "
            "{model_id: example-1.0, prices: []}]"
        ),
    ],
)
def test_invalid_document(tmp_path: Path, text: str) -> None:
    path = tmp_path / "provider.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=r".+"):
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
    prefix = _REFERENCE + "models:\n  - model_id: example-1.0\n    prices:\n"
    path.write_text(prefix + card)
    row = load_catalog(path)["example"]
    assert dict(row.prices) == {
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
        {"thinking": {"effort": []}},
        {"thinking": {"budget": []}},
        {"thinking": {"output": []}},
        {"approx_chars_per_token": 0.0},
        {"approx_chars_per_token": float("nan")},
        {"approx_chars_per_token": float("inf")},
        {"model_id": ""},
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
