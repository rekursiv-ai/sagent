"""Tests for the Gemini catalog: published facts and the thinking wire."""

from __future__ import annotations

from datetime import date

import pytest

from sagent.catalog import google
from sagent.types.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)


def _wire(
    model_id: str,
    *,
    effort: ThinkingEffort = "none",
    budget: ThinkingBudget = "fixed",
) -> object:
    """Return the ``thinkingConfig`` one selection sends on ``model_id``."""
    row = google.models()[model_id]
    settings = ModelSettings(
        capability=row,
        thinking_effort=effort,
        thinking_budget=budget,
        thinking_output="text",
    )
    return google.thinking_config(row, settings)


@pytest.mark.parametrize(
    ("model_id", "wire_id"),
    [
        ("gemini-pro-3.1", "gemini-3.1-pro-preview"),
        ("gemini-pro-2.5", "gemini-2.5-pro"),
        ("gemini-flash-3.8", "gemini-3.8-flash"),
        ("gemini-flash-3.7", "gemini-3.7-flash"),
        ("gemini-flash-3.6", "gemini-3.6-flash"),
        ("gemini-flash-3.5", "gemini-3.5-flash"),
        ("gemini-flash-3.0", "gemini-3-flash-preview"),
        ("gemini-flash-2.5", "gemini-2.5-flash"),
        ("gemini-flash-lite-3.5", "gemini-3.5-flash-lite"),
        ("gemini-flash-lite-3.1", "gemini-3.1-flash-lite"),
        ("gemini-flash-lite-2.5", "gemini-2.5-flash-lite"),
    ],
)
def test_a_catalog_id_names_the_vendor_wire_id(model_id: str, wire_id: str) -> None:
    """The vendor id keeps resolving, to the same row."""
    rows = google.models()
    assert rows[model_id].wire_model_id == wire_id
    assert rows[wire_id].model_id == model_id


def test_every_row_carries_its_wire_id() -> None:
    assert all(row.wire_model_id for row in google.models().values())


@pytest.mark.parametrize(
    "model_id",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
)
def test_shut_down_models_are_gone(model_id: str) -> None:
    assert model_id not in google.models()


def test_roles_follow_googles_guidance_for_new_projects() -> None:
    rows = google.models()
    assert rows["default"].model_id == "gemini-pro-3.1"
    assert rows["utility"].model_id == "gemini-flash-lite-3.5"


@pytest.mark.parametrize(
    ("model_id", "standard"),
    [
        ("gemini-flash-3.8", (0.75, 3.75, 0.075)),
        ("gemini-flash-3.7", (0.75, 3.75, 0.075)),
        ("gemini-flash-3.6", (0.75, 3.75, 0.075)),
        ("gemini-flash-3.5", (1.5, 9.0, 0.15)),
        ("gemini-flash-lite-3.5", (0.3, 2.5, 0.03)),
        ("gemini-flash-lite-3.1", (0.25, 1.5, 0.025)),
        ("gemini-flash-3.0", (0.5, 3.0, 0.05)),
        ("gemini-flash-2.5", (0.3, 2.5, 0.03)),
        ("gemini-flash-lite-2.5", (0.1, 0.4, 0.01)),
    ],
)
def test_a_row_bills_the_published_2026_rate(
    model_id: str,
    standard: tuple[float, float, float],
) -> None:
    price = google.models()[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=1_000,
        at=date(2026, 12, 31),
    )
    assert (price.request, price.response, price.cache_read) == standard


@pytest.mark.parametrize(
    "model_id",
    ["gemini-flash-3.8", "gemini-flash-3.7", "gemini-flash-3.6"],
)
def test_the_introductory_flash_rate_doubles_on_2027_01_01(model_id: str) -> None:
    prices = google.models()[model_id].prices
    last = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 12, 31))
    first = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2027, 1, 1))
    assert (last.request, last.response, last.cache_read) == (0.75, 3.75, 0.075)
    assert (first.request, first.response, first.cache_read) == (1.5, 7.5, 0.15)


@pytest.mark.parametrize("model_id", list(google.models()))
def test_every_row_has_the_published_windows(model_id: str) -> None:
    limits = google.models()[model_id].context[""]
    assert limits.max_request_tokens == 1_048_576
    assert limits.max_response_tokens == 65_536
    assert limits.max_request_bytes == 100 * 1024 * 1024


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("gemini-pro-3.1", "January 2025"),
        ("gemini-flash-lite-3.1", "January 2025"),
        ("gemini-flash-3.0", "January 2025"),
        ("gemini-pro-2.5", "January 2025"),
        ("gemini-flash-2.5", "January 2025"),
        ("gemini-flash-lite-2.5", "January 2025"),
        ("gemini-flash-3.8", None),
        ("gemini-flash-3.7", None),
        ("gemini-flash-3.6", None),
        ("gemini-flash-3.5", None),
        ("gemini-flash-lite-3.5", None),
    ],
)
def test_knowledge_cutoff_is_stated_only_where_published(
    model_id: str,
    cutoff: str | None,
) -> None:
    assert google.models()[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("model_id", "efforts"),
    [
        ("gemini-flash-3.8", {"none", "low", "medium", "high"}),
        ("gemini-flash-3.7", {"none", "low", "medium", "high"}),
        ("gemini-pro-3.1", {"none", "low", "medium", "high"}),
        ("gemini-flash-3.6", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-3.5", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-lite-3.5", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-lite-3.1", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-3.0", {"none", "min", "low", "medium", "high"}),
    ],
)
def test_a_gemini_3_row_offers_only_the_levels_it_accepts(
    model_id: str,
    efforts: set[str],
) -> None:
    """A level the model rejects is a 400; no level past ``high`` exists."""
    row = google.models()[model_id]
    assert row.thinking.effort == efforts
    assert row.effort_as_level


@pytest.mark.parametrize(
    "model_id",
    ["gemini-pro-2.5", "gemini-flash-2.5", "gemini-flash-lite-2.5"],
)
def test_gemini_2_5_takes_a_budget_not_a_level(model_id: str) -> None:
    assert not google.models()[model_id].effort_as_level
    assert _wire(model_id, effort="max") == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }
    assert _wire(model_id, budget="auto") == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


@pytest.mark.parametrize(
    ("effort", "level"),
    [("min", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_gemini_3_sends_the_effort_as_a_level(
    effort: ThinkingEffort,
    level: str,
) -> None:
    assert _wire("gemini-flash-3.6", effort=effort) == {
        "includeThoughts": True,
        "thinkingLevel": level,
    }


@pytest.mark.parametrize("budget", ["auto", "fixed"])
def test_gemini_3_never_sends_a_token_budget(budget: ThinkingBudget) -> None:
    """``thinkingBudget`` is backward-compat only on Gemini 3."""
    sent = _wire("gemini-pro-3.1", effort="high", budget=budget)
    assert sent == {"includeThoughts": True, "thinkingLevel": "high"}


@pytest.mark.parametrize("model_id", ["gemini-flash-3.8", "gemini-pro-2.5"])
def test_effort_none_leaves_the_models_default_in_place(model_id: str) -> None:
    """No level and no cap: never an explicit "off" a model may reject."""
    assert _wire(model_id) == {"includeThoughts": True}


def test_budget_none_sends_no_thinking_config() -> None:
    assert _wire("gemini-flash-2.5", budget="none") is None


@pytest.mark.parametrize(
    "transport",
    [google.api(), google.cli(), google.subscription()],
    ids=["api", "cli", "subscription"],
)
def test_no_transport_offers_redacted_thinking(transport: ModelCapability) -> None:
    """Gemini has no server-side redaction to select."""
    assert "redacted" not in transport.thinking.output


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
