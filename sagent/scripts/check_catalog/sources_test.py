"""Tests for check_catalog's sources: each vendor check against small literal pages."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from importlib.resources import files
from typing import TYPE_CHECKING

import math

import pytest

from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)
from sagent.model.cost import PriceCatalog, PriceCondition, ServiceTier, TokenPrice
from sagent.providers import anthropic, google, openai
from sagent.providers.table import ModelTable
from sagent.scripts.check_catalog.sources import (
    Finding,
    NewPrice,
    NewRow,
    _claude_ids,
    _count,
    _day,
    _gemini_model_id,
    _gemini_rates,
    _month,
    _new_prices,
    _openai_model_id,
    _text,
    check_anthropic,
    check_google,
    check_openai,
    waivers,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("1M tokens", 1_000_000),
        ("1.05M", 1_050_000),
        ("128K tokens", 128_000),
        ("1,050,000", 1_050_000),
        (", 128K", None),
        ("... soon", None),
        ("many", None),
        (None, None),
    ],
)
def test_a_count_is_read_or_left_unstated_never_raised(
    text: str | None,
    count: int | None,
) -> None:
    assert _count(text) == count


@pytest.mark.parametrize(
    ("text", "month"),
    [
        ("Jun 2026", "2026-06"),
        ("March 2026", "2026-03"),
        ("Sept 2026", "2026-09"),
        ("Updated 2026", None),
        ("soon", None),
        (None, None),
    ],
)
def test_a_month_is_read_or_left_unstated_never_raised(
    text: str | None,
    month: str | None,
) -> None:
    assert _month(text) == month


@pytest.mark.parametrize(
    ("text", "day"),
    [
        ("Jun 01, 2026", "2026-06-01"),
        ("March 1, 2027", "2027-03-01"),
        ("Sept 3, 2026", "2026-09-03"),
        ("Foo 3, 2026", None),
        ("Feb 30, 2026", None),
        ("March 1st, 2027", None),
    ],
)
def test_a_day_is_read_or_left_unstated_never_raised(
    text: str,
    day: str | None,
) -> None:
    assert _day(text) == day


@pytest.mark.parametrize(
    ("value", "text"),
    [(2.0, "2"), (0.1, "0.1"), (1.2345678, "1.2345678"), (1e-05, "1e-05")],
)
def test_a_rate_is_compared_and_written_exactly_never_rounded(
    value: float,
    text: str,
) -> None:
    """Six significant digits made 1.2345678 a mismatch the writer then rounded."""
    assert _text(value) == text
    assert float(_text(value)) == value


def test_a_card_list_with_no_base_card_makes_no_row() -> None:
    """A long-prompt card alone leaves short prompts unpriced: the table refuses it."""
    meters = ("request", "response", "cache_write", "cache_write_1h", "cache_read")
    stated = {f"price.auto.long.{m}": 1.0 for m in meters}
    assert (
        _new_prices(stated, unbilled=(), long_tokens=200_000, inherit=False, scaled=())
        == ()
    )


def test_a_finding_is_ok_mismatch_waived_or_unstated() -> None:
    base = Finding(
        provider="p",
        model_id="m-1.0",
        field="f",
        catalog="a",
        source="a",
        url="u",
    )
    assert base.status == "ok"
    assert replace(base, source="b").status == "mismatch"
    assert replace(base, source="b", waiver="why").status == "waived"
    assert replace(base, source=None).status == "unstated"


def test_every_waiver_names_a_shipped_row_and_a_compared_fact() -> None:
    # Read at run time, not as parameters: a waiver must be checked as it is.
    packages = {"anthropic": anthropic, "openai": openai, "google": google}
    tables = {
        name: ModelTable.from_yaml(files(package) / f"{name}.yaml", section=name)
        for name, package in packages.items()
    }
    named = sorted(waivers())
    for provider, model_id, field in named:
        # A model left out is waived by its vendor's id; every other waiver
        # names a shipped row, or ``*`` for all of the provider's.
        if field == "catalogued":
            assert tables[provider].exact(model_id) is None, (provider, model_id)
        elif model_id != "*":
            assert tables[provider].exact(model_id) is not None, (provider, model_id)
    assert {(p, f) for p, m, f in named if m == "*"} == {
        ("anthropic", "max_image_bytes"),
    }
    assert {(p, m) for p, m, f in named if f == "catalogued"} == {
        ("anthropic", "Claude Mythos Preview"),
        ("google", "gemini-3-pro-preview"),
        ("google", "gemini-robotics-er-2-preview"),
    }
    efforts = {(p, m) for p, m, f in named if f == "thinking_effort_default"}
    assert efforts == {
        ("anthropic", "fable-5.2"),
        ("anthropic", "sonnet-5.5"),
        ("anthropic", "opus-4.7"),
        ("openai", "sol-6.2"),
        ("openai", "sol-6.1"),
        ("openai", "sol-5.6"),
        ("openai", "gpt-5.6"),
    }
    catalogued = sum(f == "catalogued" for _, _, f in named)
    assert len(named) == 1 + catalogued + 7 + 7
    budgets = {m for _, m, f in named if f == "thinking_budget_default"}
    assert budgets == {
        "opus-4.8",
        "opus-4.7",
        "opus-4.6",
        "opus-4.5",
        "sonnet-4.6",
        "sonnet-4.5",
        "haiku-4.5",
    }


# Anthropic.


def test_a_model_a_vendor_lists_and_the_catalog_lacks_is_a_mismatch() -> None:
    docs = "https://platform.claude.com/docs/en"
    assert _uncatalogued(_anthropic(_haiku(), _claude_page())) == {
        ("Claude Opus 4.6", f"{docs}/build-with-claude/thinking.md"),
    }
    gpt = "https://developers.openai.com/api/docs/pricing.md"
    # Audio, realtime, and dated snapshots are listed too, and are no text model.
    extra = (
        "".join(
            f"| {m} | $0.60 | $12.00 | - |\n"
            for m in (
                "gpt-4o-mini-tts",
                "gpt-audio",
                "gpt-image-2",
                "gpt-realtime-2",
                "gpt-transcribe",
                "gpt-live-1",
                "gpt-4o-diarize",
                "gpt-4o-translate",
                "gpt-4o-whisper",
                "gpt-4o-2024-08-06",
            )
        )
        + "| o3 | $2.00 | $8.00 | $0.50 |\n"
    )
    prices = _openai_prices().replace(
        "| gpt-4o | $2.50 | $10.00 | $1.25 |\n",
        (f"| gpt-4o | $2.50 | $10.00 | $1.25 |\n{extra}"),
    )
    assert _uncatalogued(_openai(_sol(), _openai_page(), prices)) == {
        ("gpt-4o", gpt),
        ("o3", gpt),
    }
    gemini = "https://aistudio.google.com/docs/thinking.md"
    # An image model reasons too, but answers no chat.
    image = "| gemini-3.1-flash-lite-image | On (minimal) | minimal, high |\n"
    thinking = _GEMINI_THINKING + image
    assert _uncatalogued(_google(_flash(), _gemini_page(), thinking=thinking)) == {
        ("gemini-3.8-flash", gemini),
        ("gemini-2.5-flash-lite", gemini),
    }


def test_every_row_s_model_counts_as_catalogued_not_only_the_last() -> None:
    docs = "https://developers.openai.com/api/docs"
    older = _sol(model_id="sol-5.0", wire="gpt-4o")
    pages = {
        f"{docs}/pricing.md": _openai_prices(),
        f"{docs}/guides/prompt-caching.md": _OPENAI_CACHING,
        f"{docs}/models/gpt-6.1-sol.md": _openai_page(),
        f"{docs}/models/gpt-4o.md": _openai_page().replace("gpt-6.1-sol", "gpt-4o"),
    }
    # The price table lists both rows' models.
    findings = check_openai(
        ModelTable(rows=(_sol(), older)),
        fetch=lambda url: pages.get(url, ""),
        today=date(2026, 11, 18),
    )
    assert _uncatalogued(findings) == set()


@pytest.mark.parametrize(
    ("provider", "model"),
    [("anthropic", "Claude Mythos Preview"), ("google", "gemini-3-pro-preview")],
)
def test_a_model_left_out_on_purpose_is_waived_with_its_reason(
    provider: str,
    model: str,
) -> None:
    if provider == "anthropic":
        thinking = (
            _CLAUDE_THINKING + f"| {model} | Adaptive thinking | Adaptive thinking"
            " | Extended thinking | 400 error | 400 error |\n"
        )
        findings = _anthropic(_haiku(), _claude_page(), thinking=thinking)
    else:
        thinking = _GEMINI_THINKING + f"| {model} | On (high) | low, high |\n"
        findings = _google(_flash(), _gemini_page(), thinking=thinking)
    (left,) = [f for f in findings if f.model_id == model]
    assert (left.provider, left.field, left.status, left.waiver) == (
        provider,
        "catalogued",
        "waived",
        waivers()[provider, model, "catalogued"],
    )


def test_an_unreachable_model_page_leaves_the_vendor_s_list_unchecked() -> None:
    """With a page's title unknown, its model cannot be told from one left out."""
    findings = _anthropic(_haiku(), "")
    absent = {f.model_id: f.status for f in findings if f.field == "catalogued"}
    assert absent == {"Claude Opus 4.6": "unstated", "Claude Haiku 5.5": "unstated"}


def test_a_claude_page_that_agrees_yields_every_fact_ok() -> None:
    findings = _catalogued(_anthropic(_haiku(), _claude_page()))
    url = "https://platform.claude.com/docs/en/models/haiku-5-5/overview.md"
    image_cap = waivers()["anthropic", "*", "max_image_bytes"]
    assert {(f.provider, f.model_id, f.url, f.waiver) for f in findings} == {
        ("anthropic", "haiku-5.5", url, ""),
        ("anthropic", "haiku-5.5", url, image_cap),
    }
    assert {f.status for f in findings} == {"ok"}
    assert _facts(findings) == _same(
        {
            "knowledge_cutoff": "2026-06",
            "max_request_tokens": "1000000",
            "max_response_tokens": "128000",
            "thinking_budget": "auto,none",
            "thinking_budget_default": "auto",
            "thinking_off_max_effort": "high",
            "thinking_effort": "high,low,max,medium,xhigh",
            "thinking_effort_default": "medium",
            "price.auto.request": "0.1",
            "price.auto.response": "0.5",
            "price.auto.cache_write": "0.125",
            "price.auto.cache_write_1h": "0.2",
            "price.auto.cache_read": "0.01",
            "price.auto.long.request": "0.5",
            "price.auto.long.response": "2.5",
            "price.auto.long.cache_write": "0.625",
            "price.auto.long.cache_write_1h": "1",
            "price.auto.long.cache_read": "0.05",
            "price.priority": "absent",
            "max_request_bytes": "134217728",
            "max_image_bytes": "7864320",
            "max_image_edge_px": "2576",
        },
    )


@pytest.mark.parametrize(
    ("title", "edge"),
    [
        ("Claude Opus 4.6", "1568"),
        ("Claude Opus 4.7", "2576"),
        ("Claude Fable 5", "2576"),
        ("Claude Mythos Preview", None),
    ],
)
def test_a_claude_image_edge_is_its_resolution_tier(
    title: str,
    edge: str | None,
) -> None:
    """The guide's tiers: Claude 4.7 and later models, and all other models."""
    page = _claude_page().replace("title: Claude Haiku 5.5", f"title: {title}")
    row = replace(_haiku(), limits=replace(_haiku().limits, max_image_edge_px=1568))
    assert _status(_anthropic(row, page), "max_image_edge_px")[:2] == ("1568", edge)


def test_claude_byte_limits_are_unstated_where_the_guides_say_none() -> None:
    findings = _anthropic(_haiku(), _claude_page(), vision="", overview="")
    for fact in ("max_request_bytes", "max_image_bytes", "max_image_edge_px"):
        assert _status(findings, fact)[1] is None


def test_a_claude_page_that_disagrees_names_the_fact() -> None:
    page = _claude_page().replace("Jun 2026", "May 2026").replace("1M tokens", "200K")
    bad = {
        f.field: (f.catalog, f.source)
        for f in _catalogued(_anthropic(_haiku(), page))
        if f.status == "mismatch"
    }
    assert bad == {
        "knowledge_cutoff": ("2026-06", "2026-05"),
        "max_request_tokens": ("1000000", "200000"),
    }


def test_thinking_off_offered_where_claude_refuses_it_is_a_mismatch() -> None:
    thinking = _CLAUDE_THINKING.replace(
        "Thinking off at `high` effort or below",
        "400 error",
    )
    findings = _anthropic(_haiku(), _claude_page(), thinking=thinking)
    assert _status(findings, "thinking_budget") == ("auto,none", "auto", "mismatch")
    # A cap on a row the guide never lets turn thinking off is stray.
    assert _status(findings, "thinking_off_max_effort") == ("high", "any", "mismatch")
    uncapped = replace(_haiku(), thinking_budget=frozenset({"auto"}))
    findings = _anthropic(
        replace(uncapped, thinking_off_max_effort=None),
        _claude_page(),
        thinking=thinking,
    )
    assert "thinking_off_max_effort" not in _facts(findings)


def test_thinking_off_is_capped_at_the_effort_the_guide_names() -> None:
    uncapped = replace(_haiku(), thinking_off_max_effort=None)
    findings = _anthropic(uncapped, _claude_page())
    assert _status(findings, "thinking_off_max_effort") == ("any", "high", "mismatch")
    thinking = _CLAUDE_THINKING.replace(
        "Thinking off at `high` effort or below",
        "Thinking off",
    )
    findings = _anthropic(uncapped, _claude_page(), thinking=thinking)
    assert _status(findings, "thinking_off_max_effort") == ("any", "any", "ok")
    findings = _anthropic(_haiku(), _claude_page(), thinking=thinking)
    assert _status(findings, "thinking_off_max_effort") == ("high", "any", "mismatch")


@pytest.mark.parametrize(
    ("cell", "unasked"),
    [
        ("Thinking off", "none"),
        ("Adaptive thinking", "auto"),
        ("Extended thinking", "fixed"),
        ("Something new", None),
    ],
)
def test_what_claude_does_with_no_thinking_field_is_the_default_budget(
    cell: str,
    unasked: str | None,
) -> None:
    thinking = _CLAUDE_THINKING.replace(
        "| Claude Haiku 5.5 | Adaptive thinking |",
        f"| Claude Haiku 5.5 | {cell} |",
    )
    findings = _anthropic(_haiku(), _claude_page(), thinking=thinking)
    assert _status(findings, "thinking_budget_default")[:2] == ("auto", unasked)


def test_a_manual_budget_claude_still_offers_fixed() -> None:
    row = replace(
        _haiku(model_id="opus-4.6", wire="claude-opus-4-6"),
        thinking_budget=frozenset({"none", "auto", "fixed"}),
    )
    page = _claude_page().replace("Claude Haiku 5.5", "Claude Opus 4.6")
    findings = _anthropic(row, page)
    assert _status(findings, "thinking_budget") == (
        "auto,fixed,none",
        "auto,fixed,none",
        "ok",
    )


def test_each_top_effort_level_goes_to_the_models_the_guide_names() -> None:
    row = _haiku(model_id="fable-5.2", wire="claude-fable-5-2")
    page = _claude_page().replace("Claude Haiku 5.5", "Claude Fable 5.2")
    findings = _anthropic(row, page)
    assert _status(findings, "thinking_effort")[1] == "high,low,max,medium,xhigh"
    row = _haiku(model_id="opus-4.6", wire="claude-opus-4-6")
    page = _claude_page().replace("Claude Haiku 5.5", "Claude Opus 4.6")
    assert _status(_anthropic(row, page), "thinking_effort")[1] == "high,low,max,medium"


def test_a_model_the_effort_guide_omits_takes_no_effort() -> None:
    row = _haiku(model_id="sonnet-4.5", wire="claude-sonnet-4-5", effort={"none"})
    page = _claude_page().replace("`medium`", "Not supported")
    findings = _anthropic(row, page)
    assert _status(findings, "thinking_effort") == ("none", "none", "ok")
    assert _status(findings, "thinking_effort_default") == (
        "medium",
        "none",
        "mismatch",
    )


def test_a_dated_snapshot_in_the_effort_guide_counts_as_the_model() -> None:
    row = _haiku(model_id="opus-4.5", wire="claude-opus-4-5", effort={"low", "medium"})
    findings = _anthropic(row, _claude_page().replace("Haiku 5.5", "Opus 4.5"))
    assert _status(findings, "thinking_effort")[1] == "high,low,medium"


def test_a_deliberate_default_is_waived_with_its_reason() -> None:
    row = _haiku(model_id="fable-5.2", wire="claude-fable-5-2")
    findings = _anthropic(row, _claude_page().replace("`medium`", "`high`"))
    waived = next(f for f in findings if f.field == "thinking_effort_default")
    assert waived.status == "waived"
    assert "Claude Code" in waived.waiver


def test_fast_mode_pricing_is_compared_by_each_name_a_row_lists() -> None:
    fast = (
        "| Claude Opus 5.5 | $8 USD / MTok | $40 USD / MTok |\n"
        "| Claude Opus 5 / Claude Haiku 5.5 | $0.20 USD / MTok | $1 USD / MTok |\n"
    )
    findings = _anthropic(_haiku(), _claude_page(), pricing=fast)
    assert _status(findings, "price.priority.request") == ("absent", "0.2", "mismatch")
    assert _status(findings, "price.priority.response") == ("absent", "1", "mismatch")
    priced = _haiku(fast=TokenPrice(request=0.2, response=1.0))
    findings = _anthropic(priced, _claude_page(), pricing=fast)
    assert _status(findings, "price.priority.request") == ("0.2", "0.2", "ok")
    findings = _anthropic(priced, _claude_page())
    assert _status(findings, "price.priority") == ("priced", "absent", "mismatch")


def test_a_guide_whose_layout_changed_leaves_facts_unstated() -> None:
    findings = _anthropic(
        _haiku(),
        "title: Claude Haiku 5.5\n| Max output | many |\n"
        "| Reliable knowledge cutoff | soon |\n| Default effort | high |\n",
        thinking="",
        effort="",
        vision="",
        overview="",
    )
    assert {f.field for f in findings if f.status != "unstated"} == {"price.priority"}


def test_a_page_without_a_default_effort_row_leaves_it_unstated() -> None:
    page = _claude_page().replace("| [Default effort](https://x) | `medium` |\n", "")
    findings = _anthropic(_haiku(), page)
    assert _status(findings, "thinking_effort_default") == ("medium", None, "unstated")


# OpenAI.


def test_an_openai_page_and_price_table_that_agree_yield_every_fact_ok() -> None:
    findings = _catalogued(_openai(_sol(), _openai_page(), _openai_prices()))
    url = "https://developers.openai.com/api/docs/models/gpt-6.1-sol.md"
    assert {(f.provider, f.model_id, f.url, f.waiver) for f in findings} == {
        ("openai", "sol-6.0", url, ""),
    }
    assert {f.status for f in findings} == {"ok"}
    # Ultrafast's long-band ``-`` cells are unpublished, so they are not compared;
    # its short ``-`` cache is a meter not discounted, so it is ``0``.
    assert _facts(findings) == _same(
        {
            "knowledge_cutoff": "2026-04-30",
            "context_tokens": "1050000",
            "max_request_tokens": "921000",
            "max_response_tokens": "128000",
            "thinking_effort": "high,low,max,medium,xhigh",
            "thinking_effort_default": "medium",
            "cache_ttl_sec": "1800.0",
            "price.auto.request": "2",
            "price.auto.response": "10",
            "price.auto.cache_read": "0.1",
            "price.auto.cache_write": "2.5",
            "price.auto.long.request": "4",
            "price.auto.long.response": "15",
            "price.auto.long.cache_read": "0.2",
            "price.auto.long.cache_write": "5",
            "price.flex.request": "1",
            "price.priority.request": "4",
            "price.ultrafast.request": "12",
            "price.ultrafast.response": "60",
            "price.ultrafast.cache_read": "0",
            "price.ultrafast@2027-03-01.request": "24",
        },
    )


@pytest.mark.parametrize("heading", ["Ultrafast", "Ultrafast promotion"])
def test_either_ultrafast_heading_prices_the_tier(heading: str) -> None:
    prices = _openai_prices(ultrafast=heading)
    findings = _openai(_sol(), _openai_page(), prices)
    assert _status(findings, "price.ultrafast.request") == ("12", "12", "ok")


def test_an_unpublished_tier_and_a_changed_rate_are_mismatches() -> None:
    prices = _openai_prices().replace("| $2.00 | $10.00 |", "| $2.50 | $10.00 |")
    bad = {
        f.field: (f.catalog, f.source)
        for f in _catalogued(_openai(_sol(), _openai_page(), prices))
        if f.status == "mismatch"
    }
    assert bad == {"price.auto.request": ("2", "2.5")}
    no_flex = _openai_prices().replace("### Flex pricing data", "### Flexible")
    findings = _catalogued(_openai(_sol(), _openai_page(), no_flex))
    assert {f.field for f in findings if f.status == "mismatch"} == {"price.flex"}
    assert _status(findings, "price.flex") == ("priced", "absent", "mismatch")


def test_a_price_row_that_does_not_fit_its_header_is_refused() -> None:
    prices = _openai_prices().replace("| gpt-4o | $2.50 |", "| gpt-4o | $9 | $2.50 |")
    with pytest.raises(ValueError, match="zip"):
        _openai(_sol(), _openai_page(), prices)


@pytest.mark.parametrize(
    ("wire", "ttl"),
    [
        ("gpt-6.1-sol", "1800.0"),
        ("gpt-5.6-luna", "1800.0"),
        ("gpt-5.4", "1800.0"),
        ("gpt-5.4-mini", "300.0"),
        ("gpt-4.1-mini", "300.0"),
        ("o3", "300.0"),
    ],
)
def test_a_cache_lasts_30_minutes_where_retention_is_extended_else_5(
    wire: str,
    ttl: str,
) -> None:
    """GPT-5.6 and later keep 30 minutes; earlier models only on the extended list."""
    row = replace(_sol(wire=wire), cache_ttl_sec=frozenset({1800.0}))
    page = _openai_page().replace("gpt-6.1-sol", wire)
    findings = _openai(row, page, _openai_prices())
    assert _status(findings, "cache_ttl_sec")[:2] == ("1800.0", ttl)
    unlisted = _openai(row, page, _openai_prices(), caching="")
    assert _status(unlisted, "cache_ttl_sec")[1] is None


def test_a_price_column_neither_read_nor_known_unread_is_refused() -> None:
    """A new meter must be read or named as unread; it is never skipped unseen."""
    twelve = (
        _openai_prices()
        .replace(
            "| Short context 30m write |",
            "| Short context 30m write | Short context 12h write |",
        )
        .replace("| $0.10 | $2.50 |", "| $0.10 | $2.50 | $5.00 |")
        .replace(
            "| Long context 30m write |",
            "| Long context 30m write | Long context 12h write |",
        )
        .replace("| $0.20 | $5.00 |", "| $0.20 | $5.00 | $10.00 |")
    )
    assert _status(
        _openai(_sol(), _openai_page(), twelve),
        "price.auto.request",
    ) == ("2", "2", "ok")
    unknown = twelve.replace("12h write", "1d write")
    with pytest.raises(ValueError, match="'Short context 1d write'"):
        _openai(_sol(), _openai_page(), unknown)


def test_a_price_cell_that_is_no_rate_leaves_its_meter_unstated() -> None:
    prices = _openai_prices().replace("| $2.00 | $10.00 |", "| Free | $10.00* |")
    findings = _openai(_sol(), _openai_page(), prices)
    facts = _facts(findings)
    assert "price.auto.request" not in facts
    assert "price.auto.response" not in facts
    assert facts["price.auto.cache_read"] == ("0.1", "0.1")


def test_a_list_date_that_is_no_date_never_starts() -> None:
    prices = _openai_prices().replace("March 1, 2027", "March 1st, 2027")
    findings = _openai(_sol(), _openai_page(), prices)
    assert _status(findings, "price.ultrafast@9999-12-31.request")[1] == "24"


def test_a_model_the_price_page_omits_has_every_tier_unpublished() -> None:
    page = _openai_page().replace("gpt-6.1-sol", "gpt-6.1-moon")
    findings = _openai(_sol(wire="gpt-6.1-moon"), page, _openai_prices())
    assert {f.field: f.source for f in findings if f.field.startswith("price")} == {
        "price.auto": "absent",
        "price.flex": "absent",
        "price.priority": "absent",
        "price.ultrafast": "absent",
    }


def test_an_alias_page_is_priced_under_the_model_it_names() -> None:
    row = _sol(model_id="gpt-6.1", wire="gpt-6.1")
    findings = _openai(row, _openai_page(), _openai_prices())
    assert _status(findings, "price.auto.request") == ("2", "2", "ok")


def test_a_codex_default_is_waived_with_its_reason() -> None:
    row = _sol(model_id="sol-6.1", default="low")
    findings = _openai(row, _openai_page(), _openai_prices())
    waived = next(f for f in findings if f.field == "thinking_effort_default")
    assert (waived.catalog, waived.source, waived.status) == ("low", "medium", "waived")
    assert "Codex" in waived.waiver


@pytest.mark.parametrize(
    ("sentence", "levels", "default"),
    [
        (
            "Reasoning.effort supports: none, low, medium (default), high and xhigh.",
            "high,low,medium,none,xhigh",
            "medium",
        ),
        (
            "GPT-5.2 Pro supports reasoning.effort: medium, high, xhigh.",
            "high,medium,xhigh",
            None,
        ),
        ("The API rejects requests that set reasoning effort to `none`.", None, None),
        (
            "Reasoning.effort supports: minimal (default), low, medium, and high.",
            "high,low,medium,min",
            "min",
        ),
    ],
)
def test_an_openai_page_states_its_effort_levels_in_any_phrasing(
    sentence: str,
    levels: str | None,
    default: str | None,
) -> None:
    page = _openai_page().replace(_OPENAI_EFFORT, sentence)
    findings = _openai(_sol(), page, _openai_prices())
    assert _status(findings, "thinking_effort")[1] == levels
    assert _status(findings, "thinking_effort_default")[1] == default


def test_a_model_without_reasoning_states_no_effort() -> None:
    row = _sol(effort={"none"}, default="none")
    page = _openai_page().replace("- Reasoning token support\n", "")
    findings = _openai(row, page, _openai_prices())
    assert _status(findings, "thinking_effort") == ("none", "none", "ok")
    assert _status(findings, "thinking_effort_default") == ("none", "none", "ok")


def test_only_a_1m_window_keeps_its_measured_reply_margin() -> None:
    row = _sol(request=272_000, context=400_000)
    page = (
        _openai_page()
        .replace("1,050,000 context window", "400,000 context window")
        .replace("922,000", "272,000")
    )
    findings = _openai(row, page, _openai_prices())
    assert _status(findings, "max_request_tokens") == ("272000", "272000", "ok")


def test_a_one_digit_day_is_a_cutoff_too() -> None:
    row = replace(_sol(), knowledge_cutoff="2026-04-03")
    page = _openai_page().replace("Apr 30, 2026", "Apr 3, 2026")
    findings = _openai(row, page, _openai_prices())
    assert _status(findings, "knowledge_cutoff") == ("2026-04-03", "2026-04-03", "ok")


def test_an_openai_page_without_limits_leaves_them_unstated() -> None:
    findings = _openai(_sol(), "Model ID: `gpt-6.1-sol`\n", _openai_prices())
    assert _status(findings, "max_request_tokens")[1:] == (None, "unstated")
    assert _status(findings, "knowledge_cutoff")[1:] == (None, "unstated")


def test_without_a_list_date_the_list_rate_never_starts() -> None:
    prices = _openai_prices().replace("starting March 1, 2027", "later")
    findings = _openai(_sol(), _openai_page(), prices)
    assert _status(findings, "price.ultrafast@9999-12-31.request")[1] == "24"


# Google.


def test_a_gemini_page_that_agrees_yields_every_fact_ok() -> None:
    findings = _catalogued(_google(_flash(), _gemini_page()))
    url = "https://aistudio.google.com/docs/models/gemini-3.6-flash.md"
    assert {(f.provider, f.model_id, f.url, f.waiver) for f in findings} == {
        ("google", "gemini-flash-3.6", url, ""),
    }
    assert {f.status for f in findings} == {"ok"}
    # Audio and storage rates are not per-token text rates, so they are skipped.
    assert _facts(findings) == _same(
        {
            "knowledge_cutoff": "2026-03",
            "max_request_tokens": "1048576",
            "max_context_tokens": "0",
            "max_response_tokens": "65536",
            "thinking_effort": "high,low,medium,min",
            "thinking_effort_default": "min",
            "price.auto.request": "0.75",
            "price.auto.response": "3.75",
            "price.auto.cache_read": "0.075",
            "price.auto.long.request": "1.5",
            # One spelling for a dated card, whichever vendor: the writer reads it.
            "price.auto@2027-01-01.request": "3",
        },
    )


def test_a_level_other_than_minimal_is_named_as_written() -> None:
    row = replace(
        _flash(model_id="gemini-flash-3.8", wire="gemini-3.8-flash"),
        default=ModelSettings(thinking_effort="medium"),
    )
    findings = _google(row, _gemini_page())
    assert _status(findings, "thinking_effort_default") == ("medium", "medium", "ok")


def test_a_rate_compares_to_the_card_in_force_the_day_before_it_changes() -> None:
    later = _google(_flash(), _gemini_page(), today=date(2027, 6, 1))
    assert _status(later, "price.auto.request") == ("0.75", "0.75", "ok")
    undated = _GEMINI_PRICES.replace(" $3.00 starting January 1, 2027.", "")
    findings = _google(_flash(), _gemini_page(), prices=undated, today=date(2027, 6, 1))
    assert _status(findings, "price.auto.request") == ("3", "0.75", "mismatch")


def test_only_the_models_a_section_names_take_its_rates() -> None:
    row = _flash(model_id="gemini-pro-3.1", wire="gemini-3.1-pro-preview")
    findings = _google(row, _gemini_page())
    assert _status(findings, "price.auto") == ("priced", None, "unstated")


@pytest.mark.parametrize(
    ("cell", "rates"),
    [
        (
            "$0.75 through 2026. $1.50 starting January 1, 2027.",
            [(0.75, "base"), (1.5, "2027-01-01")],
        ),
        (
            "$2.00, prompts <= 200k $4.00, prompts > 200k",
            [(2.0, "base"), (4.0, "long")],
        ),
        ("$1.00 / 1,000,000 tokens per hour (storage price) $0.075", [(0.075, "base")]),
        ("$0.50 (audio) $0.25 (text) $0.30", [(0.25, "base"), (0.3, "base")]),
        ("$... later $0.40", [(0.4, "base")]),
        ("$1.00 starting Someday 1, 2027 $0.30", [(0.3, "base")]),
    ],
)
def test_a_gemini_paid_cell_yields_its_text_token_rates(
    cell: str,
    rates: list[tuple[float, str]],
) -> None:
    assert _gemini_rates(cell) == rates


def test_a_shared_window_is_a_context_limit() -> None:
    row = _flash(context=1_048_576)
    page = _gemini_page().replace(
        "**Input token limit** 1,048,576",
        "**Context window (unified input and output token limit)** 1,048,576",
    )
    findings = _google(row, page)
    assert _status(findings, "max_request_tokens") == ("1048576", "1048576", "ok")
    assert _status(findings, "max_context_tokens") == ("1048576", "1048576", "ok")


def test_a_cutoff_comes_from_the_card_or_the_card_it_defers_to() -> None:
    page = _gemini_page().replace("Knowledge cutoff", "Latest update")
    card = "https://deepmind.google/models/model-cards/x/"
    pdf = "https://storage.googleapis.com/x.pdf"
    page += f"\n[Model card]({card})\n"
    direct = {
        card: "<p>Knowledge <b>cutoff</b> date for Gemini 3.6 Flash is March 2026</p>",
    }
    assert _status(_google(_flash(), page, extra=direct), "knowledge_cutoff")[1] == (
        "2026-03"
    )
    deferred = {
        card: f"<h3>Known Limitations</h3>\n<p>See <a href={pdf} rel=x>card</a>.</p>",
        pdf: "The knowledge\ncutoff date for Gemini 3 Pro was January 2025.",
    }
    assert _status(_google(_flash(), page, extra=deferred), "knowledge_cutoff")[1] == (
        "2025-01"
    )
    silent = {card: "<h3>Known Limitations</h3><p>None stated.</p>"}
    assert _status(_google(_flash(), page, extra=silent), "knowledge_cutoff")[1] is None
    bare = page.replace(f"[Model card]({card})", "")
    assert _status(_google(_flash(), bare), "knowledge_cutoff")[1] is None


def test_a_budget_row_is_checked_for_thinking_off_by_default() -> None:
    row = _flash(
        model_id="gemini-flash-lite-2.5",
        wire="gemini-2.5-flash-lite",
        as_level=False,
        budget="none",
    )
    findings = _google(row, _gemini_page())
    assert _status(findings, "thinking_off_by_default") == ("True", "True", "ok")


def test_an_unlisted_gemini_leaves_thinking_and_prices_unstated() -> None:
    row = _flash(model_id="gemini-flash-9.9", wire="gemini-9.9-flash")
    findings = _google(row, _gemini_page())
    assert _status(findings, "thinking_effort")[2] == "unstated"
    assert _status(findings, "price.auto") == ("priced", None, "unstated")


# Helpers.


_CLAUDE_THINKING = (
    "| Claude Opus 4.6 | Thinking off | Adaptive thinking"
    " | Extended thinking (deprecated) | 400 error | Thinking off |\n"
    "| Claude Haiku 5.5 | Adaptive thinking | Adaptive thinking | 400 error | 400 error"
    " | Thinking off at `high` effort or below |\n"
)

_CLAUDE_EFFORT = """\
---
  supportedModels:
    - claude-haiku-5-5
    - claude-fable-5-2
    - claude-opus-4-6
    - claude-opus-4-5-20251101
  supportedPlatforms:
    - claude-sonnet-4-5
---
| `max`    | Most. Available on Claude Fable 5.2, Claude Opus 4.6, and Claude Haiku 5.5. | x |
| `xhigh`  | More. Available on Claude Fable 5.2 and Claude Haiku 5.5. | x |
"""


def test_a_claude_model_the_catalog_lacks_comes_with_the_row_its_pages_state() -> None:
    """Opus 4.6 is in the thinking guide and not the table: its pages make its row."""
    docs = "https://platform.claude.com/docs/en"
    page = _claude_page().replace("Claude Haiku 5.5", "Claude Opus 4.6")
    fast = "| Claude Opus 4.6 | $10 USD / MTok | $50 USD / MTok |\n"
    findings = check_anthropic(
        ModelTable(rows=(_haiku(),)),
        fetch=lambda url: {
            f"{docs}/build-with-claude/thinking.md": _CLAUDE_THINKING,
            f"{docs}/build-with-claude/effort.md": _CLAUDE_EFFORT,
            f"{docs}/about-claude/pricing.md": fast,
            f"{docs}/models/haiku-5-5/overview.md": _claude_page(),
            f"{docs}/models/opus-4-6/overview.md": page,
        }.get(url, ""),
        today=date(2026, 11, 18),
    )
    (absent,) = [f for f in findings if f.field == "catalogued"]
    assert absent.row == NewRow(
        model_id="opus-4.6",
        wire_model_id="claude-opus-4-6",
        knowledge_cutoff="2026-06",
        limits={"max_request_tokens": 1_000_000, "max_response_tokens": 128_000},
        thinking_effort=("low", "medium", "high", "max"),
        thinking_effort_default="medium",
        thinking_budget=("none", "auto", "fixed"),
        thinking_budget_default="none",
        prices=(
            NewPrice(
                service_tier="auto",
                request=0.1,
                response=0.5,
                cache_write=0.125,
                cache_write_1h=0.2,
                cache_read=0.01,
            ),
            NewPrice(
                service_tier="auto",
                effective_prompt_tokens=100_000,
                request=0.5,
                response=2.5,
                cache_write=0.625,
                cache_write_1h=1.0,
                cache_read=0.05,
            ),
            # Fast mode states input and output; its cache rates scale with them.
            NewPrice(
                service_tier="priority",
                request=10.0,
                response=50.0,
                cache_write=12.5,
                cache_write_1h=20.0,
                cache_read=1.0,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("name", "ids"),
    [
        ("Claude Opus 5.6", ("claude-opus-5-6", "opus-5.6")),
        ("Claude Opus 5", ("claude-opus-5", "opus-5.0")),
        ("Claude Mythos Preview", None),
    ],
)
def test_a_claude_name_gives_its_api_id_and_catalog_name(
    name: str,
    ids: tuple[str, str] | None,
) -> None:
    assert _claude_ids(name) == ids


@pytest.mark.parametrize(
    ("wire", "model_id"),
    [
        ("gpt-6.3-sol", "sol-6.3"),
        ("gpt-5.6", "gpt-5.6"),
        ("gpt-5.4-mini", "gpt-mini-5.4"),
        ("gpt-5-pro", "gpt-pro-5.0"),
        ("gpt-4o", "gpt-omni-4.0"),
        ("gpt-4o-mini", "gpt-omni-mini-4.0"),
        ("o3", "o-3.0"),
        ("o3-pro", "o-pro-3.0"),
        ("chatgpt-4o-latest", None),
    ],
)
def test_an_openai_id_gives_its_catalog_name(wire: str, model_id: str | None) -> None:
    assert _openai_model_id(wire) == model_id


@pytest.mark.parametrize(
    ("wire", "model_id"),
    [
        ("gemini-5-krypton", "gemini-krypton-5.0"),
        ("gemini-3.9-flash-lite", "gemini-flash-lite-3.9"),
        ("gemini-3.1-pro-preview", "gemini-pro-3.1"),
        ("gemini-embedding-001", None),
    ],
)
def test_a_gemini_id_gives_its_catalog_name(wire: str, model_id: str | None) -> None:
    assert _gemini_model_id(wire) == model_id


def _claude_page() -> str:
    def rate(small: str, large: str) -> str:
        return (
            f"${small} / MTok for prompts up to 100,000 tokens;"
            f" ${large} / MTok for prompts over 100,000 tokens"
        )

    return f"""\
---
title: Claude Haiku 5.5
---
| Feature | Value |
| :-- | :-- |
| Input | {rate("0.10", "0.50")} |
| Output | {rate("0.50", "2.50")} |
| [5m cache write](https://x) | {rate("0.125", "0.625")} |
| [1h cache write](https://x) | {rate("0.20", "1")} |
| [Cache read](https://x) | {rate("0.01", "0.05")} |
| [Context window](https://x) | 1M tokens |
| Max output | 128K tokens |
| Max output | 64K tokens |
| [Default effort](https://x) | `medium` |
| Reliable knowledge cutoff | Jun 2026 |
"""


def _haiku(
    *,
    model_id: str = "haiku-5.5",
    wire: str = "claude-haiku-5-5",
    effort: set[ThinkingEffort] | None = None,
    fast: TokenPrice | None = None,
) -> ModelCapability:
    small = TokenPrice(
        request=0.1,
        response=0.5,
        cache_write=0.125,
        cache_write_1h=0.2,
        cache_read=0.01,
    )
    large = TokenPrice(
        request=0.5,
        response=2.5,
        cache_write=0.625,
        cache_write_1h=1.0,
        cache_read=0.05,
    )
    cards: dict[tuple[str, int, float], TokenPrice] = {
        ("auto", 0, 100_001): small,
        ("auto", 100_001, math.inf): large,
    }
    if fast is not None:
        cards["priority", 0, math.inf] = fast
    row = _capability(
        model_id=model_id,
        wire=wire,
        cutoff="2026-06",
        limits=ModelLimits(
            max_request_tokens=1_000_000,
            max_response_tokens=128_000,
            max_request_bytes=134_217_728,
            max_image_bytes=7_864_320,
            max_image_edge_px=2_576,
        ),
        effort=effort or {"low", "medium", "high", "xhigh", "max"},
        budget={"none", "auto"},
        default=ModelSettings(thinking_effort="medium", thinking_budget="auto"),
        cards=cards,
    )
    return replace(row, thinking_off_max_effort="high")


_CLAUDE_VISION = """\
* 10\u00a0MB (base64-encoded) on the Claude API and Claude Models on Google Cloud.
* 5 MB (base64-encoded) on Amazon Bedrock.

| Resolution tier | Models                      | Max long edge | Max visual tokens |
| --------------- | --------------------------- | ------------- | ----------------- |
| High-resolution | Claude 4.7 and later models | 2,576 px      | 4,784             |
| Standard        | All other models            | 1,568 px      | 1,568             |
"""

_CLAUDE_OVERVIEW = """\
| Endpoint | Maximum request size |
| Messages, Token Counting | 128 MB |
| Message Batches API (see [Batch processing](https://x)) | 256 MB |
"""


def _anthropic(
    row: ModelCapability,
    page: str,
    *,
    thinking: str = _CLAUDE_THINKING,
    effort: str = _CLAUDE_EFFORT,
    pricing: str = "",
    vision: str = _CLAUDE_VISION,
    overview: str = _CLAUDE_OVERVIEW,
) -> list[Finding]:
    docs = "https://platform.claude.com/docs/en"
    pages = {
        f"{docs}/build-with-claude/thinking.md": thinking,
        f"{docs}/build-with-claude/effort.md": effort,
        f"{docs}/about-claude/pricing.md": pricing,
        f"{docs}/build-with-claude/vision.md": vision,
        f"{docs}/api/overview.md": overview,
        f"{docs}/models/{row.wire_model_id.removeprefix('claude-')}/overview.md": page,
    }
    # As ``cli._fetch`` reads a page it cannot get: empty.
    return check_anthropic(
        ModelTable(rows=(row,)),
        fetch=lambda url: pages.get(url, ""),
        today=date(2026, 11, 18),
    )


_OPENAI_EFFORT = "`reasoning.effort` supports `low`, `medium` (default), `high`, `xhigh`, and\n`max`."


def _openai_page() -> str:
    return f"""\
# GPT-6.1 Sol
Model ID: `gpt-6.1-sol`
{_OPENAI_EFFORT}
## Model details
- 1,050,000 context window
- Maximum input tokens: 922,000
- 128,000 max output tokens
- Apr 30, 2026 knowledge cutoff
- Reasoning token support
"""


def _openai_prices(*, ultrafast: str = "Ultrafast") -> str:
    return f"""\
### Standard pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write | Long context input | Long context output | Long context cached input | Long context 30m write |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol (<272K context length) | $2.00 | $10.00 | $0.10 | $2.50 | $4.00 | $15.00 | $0.20 | $5.00 |

| Model | Input | Output | Cached input |
| --- | --- | --- | --- |
| gpt-4o | $2.50 | $10.00 | $1.25 |

Batch pricing data follows.

### Batch pricing data

| Model | Input |
| --- | --- |
| gpt-6.1-sol | $1.00 |

### Flex pricing data

| Model | Input |
| --- | --- |
| gpt-6.1-sol | $1.00 |

### Fast pricing data

| Model | Input |
| --- | --- |
| gpt-6.1-sol | $4.00 |

### {ultrafast} pricing data

| Model | Input | Output | Cached input | Long context input | Long context output |
| --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | $12.00 | $60.00 | - | - | - |

Ultrafast costs 6 times the Standard rate starting March 1, 2027 (Pacific time).

### Ultrafast list pricing data

| Model | Input |
| --- | --- |
| gpt-6.1-sol | $24.00 |

### Grouped Pricing Table data
"""


def _sol(
    *,
    model_id: str = "sol-6.0",
    wire: str = "gpt-6.1-sol",
    effort: set[ThinkingEffort] | None = None,
    default: ThinkingEffort = "medium",
    request: int = 921_000,
    context: int = 1_050_000,
) -> ModelCapability:
    row = _capability(
        model_id=model_id,
        wire=wire,
        cutoff="2026-04-30",
        limits=ModelLimits(
            max_request_tokens=request,
            max_context_tokens=context,
            max_response_tokens=128_000,
        ),
        effort=effort or {"low", "medium", "high", "xhigh", "max"},
        budget={"auto"},
        default=ModelSettings(thinking_effort=default),
        cards={
            ("auto", 0, 272_001): TokenPrice(
                request=2.0,
                response=10.0,
                cache_write=2.5,
                cache_read=0.1,
            ),
            ("auto", 272_001, math.inf): TokenPrice(
                request=4.0,
                response=15.0,
                cache_write=5.0,
                cache_read=0.2,
            ),
            ("flex", 0, math.inf): TokenPrice(request=1.0),
            ("priority", 0, math.inf): TokenPrice(request=4.0),
            ("ultrafast", 0, math.inf): TokenPrice(request=12.0, response=60.0),
            ("ultrafast@2027-03-01", 0, math.inf): TokenPrice(request=24.0),
        },
    )
    return replace(row, cache_ttl_sec=frozenset({1800.0}))


_OPENAI_CACHING = (
    "\\* Extended retention is supported by `gpt-5.5`, `gpt-5.4`, and `gpt-4.1`.\n"
)


def _openai(
    row: ModelCapability,
    page: str,
    prices: str,
    *,
    caching: str = _OPENAI_CACHING,
) -> list[Finding]:
    docs = "https://developers.openai.com/api/docs"
    pages = {
        f"{docs}/pricing.md": prices,
        f"{docs}/guides/prompt-caching.md": caching,
        f"{docs}/models/{row.wire_model_id}.md": page,
    }
    # As ``cli._fetch`` reads a page it cannot get: empty.
    return check_openai(
        ModelTable(rows=(row,)),
        fetch=lambda url: pages.get(url, ""),
        today=date(2026, 11, 18),
    )


def test_a_gemini_model_the_catalog_lacks_comes_with_the_row_its_pages_state() -> None:
    """3.6 Flash is in the thinking guide and not the table: its pages make its row."""
    row = _flash(model_id="gemini-flash-3.8", wire="gemini-3.8-flash")
    docs = "https://aistudio.google.com/docs"
    findings = _google(
        row,
        _gemini_page(),
        extra={
            f"{docs}/models/gemini-3.6-flash.md": _gemini_page(),
            f"{docs}/models/gemini-2.5-flash-lite.md": _gemini_page(),
        },
    )
    absent = {f.model_id: f.row for f in findings if f.field == "catalogued"}
    # A band's unstated meter is its base card's: Gemini states only what changes.
    base = NewPrice(
        service_tier="auto",
        request=0.75,
        response=3.75,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=0.075,
    )
    assert absent["gemini-3.6-flash"] == NewRow(
        model_id="gemini-flash-3.6",
        wire_model_id="gemini-3.6-flash",
        knowledge_cutoff="2026-03",
        limits={"max_request_tokens": 1_048_576, "max_response_tokens": 65_536},
        thinking_effort=("min", "low", "medium", "high"),
        thinking_effort_default="min",
        prices=(
            base,
            replace(base, effective_prompt_tokens=200_000, request=1.5),
            replace(base, effective_date="2027-01-01", request=3.0),
        ),
    )
    # Off by default, with no level: a budget model, which levels misdescribe.
    assert absent["gemini-2.5-flash-lite"] is None


def test_an_openai_model_the_catalog_lacks_comes_with_the_row_its_pages_state() -> None:
    """A tier priced in part is left out: unpriced, it is unoffered, and flagged."""
    docs = "https://developers.openai.com/api/docs"
    findings = check_openai(
        ModelTable(rows=(_sol(model_id="gpt-omni-4.0", wire="gpt-4o"),)),
        fetch=lambda url: {
            f"{docs}/pricing.md": _openai_prices(),
            f"{docs}/guides/prompt-caching.md": _OPENAI_CACHING,
            f"{docs}/models/gpt-6.1-sol.md": _openai_page(),
        }.get(url, ""),
        today=date(2026, 11, 18),
    )
    absent = {f.model_id: f.row for f in findings if f.field == "catalogued"}
    base = NewPrice(
        service_tier="auto",
        request=2.0,
        response=10.0,
        cache_write=2.5,
        cache_write_1h=0.0,
        cache_read=0.1,
    )
    assert absent["gpt-6.1-sol"] == NewRow(
        model_id="sol-6.1",
        wire_model_id="gpt-6.1-sol",
        knowledge_cutoff="2026-04-30",
        # The page's 922,000, less the 1,000 held back on 1.05M windows; the
        # window wider than the input is the context the two share.
        limits={
            "max_request_tokens": 921_000,
            "max_response_tokens": 128_000,
            "max_context_tokens": 1_050_000,
        },
        thinking_effort=("low", "medium", "high", "xhigh", "max"),
        thinking_effort_default="medium",
        prices=(
            base,
            NewPrice(
                service_tier="auto",
                effective_prompt_tokens=272_000,
                request=4.0,
                response=15.0,
                cache_write=5.0,
                cache_write_1h=0.0,
                cache_read=0.2,
            ),
        ),
    )


def test_a_model_whose_id_follows_no_naming_is_left_for_a_person() -> None:
    row = _flash(model_id="gemini-flash-3.8", wire="gemini-3.8-flash")
    thinking = _GEMINI_THINKING + "| gemini-embedding-001 | On (low) | low |\n"
    findings = _google(row, _gemini_page(), thinking=thinking)
    absent = {f.model_id: f.row for f in findings if f.field == "catalogued"}
    assert absent["gemini-embedding-001"] is None


def test_a_gemini_model_whose_page_cannot_be_read_is_left_for_a_person() -> None:
    row = _flash(model_id="gemini-flash-3.8", wire="gemini-3.8-flash")
    findings = _google(row, _gemini_page())
    absent = {f.model_id: f.row for f in findings if f.field == "catalogued"}
    assert absent["gemini-3.6-flash"] is None


def _gemini_page() -> str:
    return """\
# Gemini 3.6 Flash
  *     **Input token limit** 1,048,576

    **Output token limit** 65,536

  {% icon name="cognition_2" /%}Knowledge cutoff

    {% /section %}

  * March 2026
"""


_GEMINI_THINKING = """\
| gemini-3.6-flash | On (minimal) | minimal, low, medium, high |
| gemini-3.8-flash | On (medium) | min, low, medium, high |
| gemini-2.5-flash-lite | Off | low, medium, high |
"""

_GEMINI_PRICES = """\
Prices per 1M tokens.
## Gemini 3.6 Flash  {% id="gemini-3.6-flash" %}
  *[`gemini-3.6-flash`](/docs/models/gemini-3.6-flash)*
{% hr /%}
Text is priced the same as [`gemini-3.1-pro-preview`](/docs/x).
{% tab title="Batch" %}
  * Input price
  * Free of charge
  * $0.01
{% /tab %}
{% tab title="Standard" %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.75, prompts <= 200k tokens $1.50, prompts > 200k tokens
  $3.00 starting January 1, 2027.

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $3.75 (text) $9.00 (audio)

  ---

  * Context caching price
  * Not available

  ---

  * Context caching price
  * Free of charge
  * $0.075 $1.00 / 1,000,000 tokens per hour (storage price)
{% /tab %}
{% tab title="Flex" %}
  * Feature

  ---

  * Input price
  * Free of charge
  * $0.02
{% /tab %}
## Gemini 3.8 Flash  {% id="gemini-3.8-flash" %}
  *[`gemini-3.8-flash`](/docs/models/gemini-3.8-flash)*
{% hr /%}
"""


def _flash(
    *,
    model_id: str = "gemini-flash-3.6",
    wire: str = "gemini-3.6-flash",
    context: int = 0,
    as_level: bool = True,
    budget: ThinkingBudget = "auto",
) -> ModelCapability:
    row = _capability(
        model_id=model_id,
        wire=wire,
        cutoff="2026-03",
        limits=ModelLimits(
            max_request_tokens=1_048_576,
            max_context_tokens=context,
            max_response_tokens=65_536,
        ),
        effort={"min", "low", "medium", "high"},
        budget={"auto"},
        default=ModelSettings(thinking_effort="min", thinking_budget=budget),
        cards={
            ("auto", 0, 200_001): TokenPrice(
                request=0.75,
                response=3.75,
                cache_read=0.075,
            ),
            ("auto", 200_001, math.inf): TokenPrice(request=1.5),
            ("auto@2027-01-01", 0, math.inf): TokenPrice(request=3.0),
        },
    )
    return replace(row, effort_as_level=as_level)


def _google(
    row: ModelCapability,
    page: str,
    *,
    extra: Mapping[str, str] | None = None,
    prices: str = _GEMINI_PRICES,
    thinking: str = _GEMINI_THINKING,
    today: date = date(2026, 11, 18),
) -> list[Finding]:
    docs = "https://aistudio.google.com/docs"
    pages = {
        f"{docs}/thinking.md": thinking,
        f"{docs}/pricing.md": prices,
        f"{docs}/models/{row.wire_model_id}.md": page,
        **(extra or {}),
    }
    # As ``cli._fetch`` reads a page it cannot get: empty.
    return check_google(
        ModelTable(rows=(row,)),
        fetch=lambda url: pages.get(url, ""),
        today=today,
    )


def _capability(
    *,
    model_id: str,
    wire: str,
    cutoff: str,
    limits: ModelLimits,
    effort: set[ThinkingEffort],
    budget: set[ThinkingBudget],
    default: ModelSettings,
    cards: Mapping[tuple[str, int, float], TokenPrice],
) -> ModelCapability:
    """Return a row; a card's tier may carry ``@YYYY-MM-DD``, the date it starts."""
    prices: dict[PriceCondition, TokenPrice] = {}
    for (tier, floor, ceil), price in cards.items():
        name, _, starts = tier.partition("@")
        condition = PriceCondition(
            service_tier=_tier(name),
            token_size_range=(floor, ceil),
            effective_date=date.fromisoformat(starts) if starts else date.min,
        )
        prices[condition] = price
    return ModelCapability(
        model_id=model_id,
        wire_model_id=wire,
        knowledge_cutoff=cutoff,
        limits=limits,
        thinking_effort=frozenset(effort),
        thinking_budget=frozenset(budget),
        default=default,
        prices=PriceCatalog(prices=prices, window=limits.max_request_tokens),
    )


def _tier(name: str) -> ServiceTier:
    tiers: dict[str, ServiceTier] = {
        "auto": "auto",
        "flex": "flex",
        "priority": "priority",
        "ultrafast": "ultrafast",
    }
    return tiers[name]


def _status(findings: list[Finding], field: str) -> tuple[str, str | None, str]:
    found = next(f for f in findings if f.field == field)
    return found.catalog, found.source, found.status


def _uncatalogued(findings: list[Finding]) -> set[tuple[str, str]]:
    """Return each model a vendor lists and the catalog lacks, with where it is listed."""
    absent = {f for f in findings if f.field == "catalogued"}
    providers = {f.provider for f in findings if f.field != "catalogued"}
    assert {(f.provider, f.catalog, f.source, f.status) for f in absent} <= {
        (provider, "absent", "listed", "mismatch") for provider in providers
    }
    return {(f.model_id, f.url) for f in absent}


def _catalogued(findings: list[Finding]) -> list[Finding]:
    """Return the findings about the catalog's own rows."""
    return [f for f in findings if f.field != "catalogued"]


def _facts(findings: list[Finding]) -> dict[str, tuple[str, str | None]]:
    return {f.field: (f.catalog, f.source) for f in findings}


def _same(values: Mapping[str, str]) -> dict[str, tuple[str, str | None]]:
    return {field: (value, value) for field, value in values.items()}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
