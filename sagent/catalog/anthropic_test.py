"""Anthropic catalog rows, checked against the published docs (2026-10-03)."""

from __future__ import annotations

import pytest

from sagent.catalog import anthropic
from sagent.types.cost import PriceKey, TokenPrice


def test_sonnet_5_5_bills_its_published_card() -> None:
    """https://platform.claude.com/docs/en/models/sonnet-5-5/overview."""
    assert anthropic.models()["sonnet-5.5"].prices[PriceKey("auto")] == TokenPrice(
        request=2.0,
        response=10.0,
        cache_write=2.5,
        cache_write_1h=4.0,
        cache_read=0.2,
    )


def test_sonnet_5_5_is_the_newest_sonnet() -> None:
    rows = anthropic.models()
    assert rows["sonnet-5.5"].wire_model_id == "claude-sonnet-5-5"
    assert rows["sonnet"].model_id == "sonnet-5.5"
    assert rows["utility"].model_id == "sonnet-5.5"
    limits = rows["sonnet-5.5"].context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == (
        1_000_000,
        128_000,
    )


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("sonnet-5.5", "June 2026"),
        ("mythos-5.1", "June 2026"),
        ("mythos-5.0", "January 2026"),
    ],
)
def test_new_rows_state_their_published_cutoff(model_id: str, cutoff: str) -> None:
    assert anthropic.models()[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("mythos", "fable", "wire"),
    [
        ("mythos-5.1", "fable-5.1", "claude-mythos-5-1"),
        ("mythos-5.0", "fable-5.0", "claude-mythos-5"),
    ],
)
def test_mythos_shares_fables_specs_and_prices(
    mythos: str,
    fable: str,
    wire: str,
) -> None:
    """Each Mythos overview page: "It shares Claude Fable ...'s specifications"."""
    rows = anthropic.models()
    assert rows[mythos].wire_model_id == wire
    assert rows[mythos].prices == rows[fable].prices
    assert rows[mythos].context == rows[fable].context
    assert rows[mythos].thinking == rows[fable].thinking
    assert "priority" not in rows[mythos].service_tier


def test_mythos_family_resolves_to_its_newest_row() -> None:
    assert anthropic.models()["mythos"].model_id == "mythos-5.1"


def test_no_row_needs_a_context_beta() -> None:
    """For every 1M model, "1M is the default: you don't need a beta header"."""
    for model_id, row in anthropic.models().items():
        for tag, limits in row.context.items():
            assert not limits.request_betas, (model_id, tag)


def test_image_cap_leaves_room_for_base64_under_10_mb() -> None:
    """The vision page caps 10 MB base64-encoded; the cap applies to raw bytes."""
    for row in anthropic.models().values():
        raw = row.context[""].max_image_bytes
        assert raw == 7_500_000
        assert -(-raw // 3) * 4 <= 10_000_000


@pytest.mark.parametrize(
    "model_id",
    [
        "fable-5.1",
        "fable-5.0",
        "mythos-5.1",
        "mythos-5.0",
        "opus-5.5",
        "opus-5.0",
        "opus-4.8",
        "opus-4.7",
        "sonnet-5.5",
        "sonnet-5.0",
    ],
)
def test_omitted_display_models_still_offer_summarized_text(model_id: str) -> None:
    """``display`` defaults to omitted here; ``summarized`` opts back in."""
    output = anthropic.models()[model_id].thinking.output
    assert output == frozenset({"none", "text", "redacted"})


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
