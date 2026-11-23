"""Tests for writing a vendor's stated facts back into its catalog document."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from sagent.providers.table import ModelTable
from sagent.scripts.check_catalog.sources import Finding, NewPrice, NewRow
from sagent.scripts.check_catalog.write import write


if TYPE_CHECKING:
    from pathlib import Path


_DOCUMENT = """\
models:
# Implicit caching only.
- model_id: gemini-neon-4.0
  wire_model_id: gemini-4-neon
  knowledge_cutoff: '2026-03-01'
  limits:
    max_request_tokens: 1_048_576
    max_context_tokens: 1_114_112
    max_response_tokens: 65_536
  prices:
  - service_tier: auto
    request: 3.0  # The list rate.
    response: 15.0
    cache_read: 0.15
  - service_tier: auto
    effective_date: '2026-12-01'
    request: 4.0
    response: 20.0
  - service_tier: auto
    effective_prompt_tokens: 200_000
    request: 6.0
  - service_tier: ultrafast
    request: 9.0
  - service_tier: ultrafast
    effective_date: '2027-03-01'
    request: 18.0

- model_id: gemini-argon-4.0
  limits: &shared
    max_request_tokens: 1_000
  prices: &cards
  - service_tier: auto
    request: 2.0

- model_id: gemini-flash-3.8
  limits: *shared
  prices: *cards

providers:
  google:
    sends: []
"""


def _finding(model_id: str, field: str, source: str) -> Finding:
    return Finding(
        provider="google",
        model_id=model_id,
        field=field,
        catalog="old",
        source=source,
        url="https://u",
    )


def test_each_fact_a_row_states_is_rewritten_in_place(tmp_path: Path) -> None:
    """Only the stated value changes: comments, order, and spelling stay."""
    path = tmp_path / "google.yaml"
    path.write_text(_DOCUMENT)
    left = write(
        path,
        [
            _finding("gemini-neon-4.0", "price.auto.request", "3.5"),
            _finding("gemini-neon-4.0", "price.auto@2026-12-01.response", "18"),
            _finding("gemini-neon-4.0", "price.auto.long.request", "7"),
            _finding("gemini-neon-4.0", "price.ultrafast@2027-03-01.request", "24"),
            _finding("gemini-neon-4.0", "knowledge_cutoff", "2026-04-01"),
            _finding("gemini-neon-4.0", "max_request_tokens", "1000000"),
            _finding("gemini-neon-4.0", "max_context_tokens", "1200000"),
            _finding("gemini-neon-4.0", "max_response_tokens", "131072"),
        ],
    )
    assert left == []
    expected = (
        _DOCUMENT.replace("request: 3.0  #", "request: 3.5  #")
        .replace("response: 20.0", "response: 18.0")
        .replace("request: 6.0", "request: 7.0")
        .replace("request: 18.0", "request: 24.0")
        .replace("'2026-03-01'", "'2026-04-01'")
        .replace("max_request_tokens: 1_048_576", "max_request_tokens: 1_000_000")
        .replace("max_context_tokens: 1_114_112", "max_context_tokens: 1_200_000")
        .replace("max_response_tokens: 65_536", "max_response_tokens: 131_072")
    )
    assert path.read_text() == expected


def test_a_fact_the_row_does_not_state_itself_is_left_for_a_person(
    tmp_path: Path,
) -> None:
    """A shared or missing value: writing one row's copy would split or invent it.

    Argon shares its window and its prices, each with Flash.
    """
    path = tmp_path / "google.yaml"
    path.write_text(_DOCUMENT)
    unwritable = [
        _finding("gemini-argon-4.0", "max_request_tokens", "2000"),
        _finding("gemini-argon-4.0", "price.auto.request", "4"),
        _finding("gemini-argon-4.0", "price.priority.request", "4"),
        _finding("gemini-argon-4.0", "price.priority", "absent"),
        _finding("gemini-neon-4.0", "price.auto.weird.request", "7"),
        _finding("gemini-argon-4.0", "thinking_effort", "high,low"),
        _finding("gemini-krypton-5.0", "knowledge_cutoff", "2026-09-01"),
        _finding("gemini-5-krypton", "catalogued", "listed"),
    ]
    assert write(path, unwritable) == unwritable
    assert path.read_text() == _DOCUMENT


def test_a_value_the_catalog_would_refuse_is_left_and_nothing_is_saved(
    tmp_path: Path,
) -> None:
    """Every edit is loaded before it is kept: a bad one never reaches the file."""
    path = tmp_path / "google.yaml"
    path.write_text(_DOCUMENT)
    refused = _finding("gemini-neon-4.0", "knowledge_cutoff", "2026-13-01")
    assert write(path, [refused]) == [refused]
    assert path.read_text() == _DOCUMENT


def test_a_tiny_rate_is_written_as_a_float_the_catalog_reads_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "google.yaml"
    path.write_text(_DOCUMENT)
    assert (
        write(path, [_finding("gemini-neon-4.0", "price.auto.cache_read", "1e-05")])
        == []
    )
    neon = ModelTable.from_yaml(path, section="google")["gemini-neon-4.0"]
    assert neon.prices["auto", 1, date(2026, 1, 1)].cache_read == 1e-05


_CATALOG = """\
reference:
  thinking_effort_default: high
  limits:
    max_request_tokens: 1_048_576
    max_response_tokens: 65_536
    max_request_bytes: 104_857_600
    truncates_at_context: true
  thinking:
    effort: [min, low, medium, high]
    budget: [auto]
    output: [none, text]

profiles:
  levels_xhigh:
    thinking:
      effort: [low, medium, high, xhigh]
      budget: [auto]
      output: [none, text]

models:
# Shared window.
- model_id: gemini-neon-4.0
  wire_model_id: gemini-4-neon
  profile: levels_xhigh
  limits: &gemini_4
    max_request_tokens: 1_048_576
    max_context_tokens: 1_048_576
    max_response_tokens: 65_536
    max_request_bytes: 104_857_600
    truncates_at_context: true
  prices:
  - service_tier: auto
    request: 3.0
    response: 15.0
    cache_write: 0.0
    cache_write_1h: 0.0
    cache_read: 0.15

providers:
  google:
    sends: [thinking_effort]
"""


def _absent(row: NewRow | None, *, model_id: str = "listed-id") -> Finding:
    return Finding(
        provider="google",
        model_id=model_id,
        field="catalogued",
        catalog="absent",
        source="listed",
        url="https://u",
        row=row,
    )


_BASE = NewPrice(
    service_tier="auto",
    request=4.0,
    response=20.0,
    cache_write=0.0,
    cache_write_1h=0.0,
    cache_read=0.2,
)

_LATER = NewPrice(
    service_tier="auto",
    effective_date="2027-01-01",
    request=5.0,
    response=25.0,
    cache_write=0.0,
    cache_write_1h=0.0,
    cache_read=0.25,
)

_GEMINI_4 = {
    "max_request_tokens": 1_048_576,
    "max_context_tokens": 1_048_576,
    "max_response_tokens": 65_536,
}


def test_a_new_model_is_written_reusing_the_document_s_profile_and_window(
    tmp_path: Path,
) -> None:
    """Its levels name a profile and its window an anchor: nothing is restated."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-krypton-5.0",
        wire_model_id="gemini-5-krypton",
        knowledge_cutoff="2026-09-01",
        limits=_GEMINI_4,
        thinking_effort=("low", "medium", "high", "xhigh"),
        thinking_effort_default="high",
        prices=(_BASE, _LATER),
    )
    assert write(path, [_absent(row)]) == []
    # Below the row that sets the window: an alias must follow its anchor.
    added = """\
- model_id: gemini-krypton-5.0
  wire_model_id: gemini-5-krypton
  profile: levels_xhigh
  knowledge_cutoff: '2026-09-01'
  limits: *gemini_4
  prices:
  - service_tier: auto
    request: 4.0
    response: 20.0
    cache_write: 0.0
    cache_write_1h: 0.0
    cache_read: 0.2
  - service_tier: auto
    effective_date: '2027-01-01'
    request: 5.0
    response: 25.0
    cache_write: 0.0
    cache_write_1h: 0.0
    cache_read: 0.25

"""
    assert path.read_text() == _CATALOG.replace("providers:\n", f"{added}providers:\n")
    table = ModelTable.from_yaml(path, section="google")
    krypton = table["krypton"]
    assert krypton.thinking_effort == {"low", "medium", "high", "xhigh"}
    assert krypton.limits.max_context_tokens == 1_048_576


def test_a_new_model_whose_window_no_anchor_holds_states_it(tmp_path: Path) -> None:
    """Neon 4.0's anchored window is not Krypton's: Krypton's is written out."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-krypton-5.0",
        wire_model_id="gemini-5-krypton",
        knowledge_cutoff="2026-09-01",
        limits={"max_request_tokens": 2_097_152},
        thinking_effort=("low", "medium", "high", "xhigh"),
        thinking_effort_default="high",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    text = path.read_text()
    assert "*gemini_4" not in text
    assert "  limits:\n    max_request_tokens: 2_097_152\n" in text
    table = ModelTable.from_yaml(path, section="google")
    assert table["krypton"].limits.max_request_tokens == 2_097_152


def test_a_new_model_above_its_window_s_anchor_states_the_window(
    tmp_path: Path,
) -> None:
    """Newest first puts Neon 4.5 above the row that anchors the window it shares."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-neon-4.5",
        wire_model_id="gemini-4.5-neon",
        knowledge_cutoff="2026-09-01",
        limits=_GEMINI_4,
        thinking_effort=("low", "medium", "high", "xhigh"),
        thinking_effort_default="high",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    text = path.read_text()
    assert text.index("gemini-neon-4.5") < text.index("# Shared window.")
    assert "*gemini_4" not in text
    table = ModelTable.from_yaml(path, section="google")
    assert table["neon"].limits.max_context_tokens == 1_048_576


def test_a_new_model_of_a_known_family_goes_above_its_elders_stating_what_differs(
    tmp_path: Path,
) -> None:
    """No profile has its levels, no row its window: both are written out whole."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-neon-4.5",
        wire_model_id="gemini-4.5-neon",
        knowledge_cutoff="2026-09-01",
        limits={"max_request_tokens": 2_097_152, "max_response_tokens": 131_072},
        thinking_effort=("low", "high"),
        thinking_effort_default="low",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    added = """\
- model_id: gemini-neon-4.5
  wire_model_id: gemini-4.5-neon
  thinking:
    effort: [low, high]
    budget: [auto]
    output: [none, text]
  thinking_effort_default: low
  knowledge_cutoff: '2026-09-01'
  limits:
    max_request_tokens: 2_097_152
    max_response_tokens: 131_072
    max_request_bytes: 104_857_600
    truncates_at_context: true
  prices:
  - service_tier: auto
    request: 4.0
    response: 20.0
    cache_write: 0.0
    cache_write_1h: 0.0
    cache_read: 0.2

"""
    assert path.read_text() == _CATALOG.replace(
        "# Shared window.\n- model_id: gemini-neon-4.0",
        f"{added}# Shared window.\n- model_id: gemini-neon-4.0",
    )
    table = ModelTable.from_yaml(path, section="google")
    assert table["neon"].model_id == "gemini-neon-4.5"
    assert table["neon"].default.thinking_effort == "low"


_NEON_39 = NewRow(
    model_id="gemini-neon-3.9",
    wire_model_id="gemini-3.9-neon",
    knowledge_cutoff="2026-01-01",
    limits={},
    thinking_effort=("min", "low", "medium", "high"),
    thinking_effort_default="high",
    prices=(_BASE,),
)


def test_an_older_new_model_goes_below_its_family_s_newer_rows(tmp_path: Path) -> None:
    """A family lists newest first; above Neon 4.0, Neon 3.9 would be refused."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    assert write(path, [_absent(_NEON_39)]) == []
    text = path.read_text()
    assert text.index("gemini-neon-4.0") < text.index("gemini-neon-3.9")
    assert text.index("gemini-neon-3.9") < text.index("providers:")
    table = ModelTable.from_yaml(path, section="google")
    assert list(table) == ["gemini-neon-4.0", "gemini-neon-3.9"]


def test_a_new_row_never_lands_between_a_row_and_its_comment(tmp_path: Path) -> None:
    """The comment above Carbon is Carbon's: Neon 3.9 goes above it, not below."""
    carbon = """\
# Implicit caching only.
- model_id: gemini-carbon-4.0
  wire_model_id: gemini-4-carbon
  limits: *gemini_4
  prices:
  - service_tier: auto
    request: 2.0

providers:
"""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG.replace("providers:\n", carbon))
    assert write(path, [_absent(_NEON_39)]) == []
    text = path.read_text()
    assert text.index("gemini-neon-3.9") < text.index("# Implicit caching only.")
    assert "# Implicit caching only.\n- model_id: gemini-carbon-4.0" in text
    assert len(ModelTable.from_yaml(path, section="google")) == 3


_ANCHORED = """\
reference:
  limits:
    max_request_tokens: 1_000
  thinking:
    effort: [low, high]
    budget: [auto]
    output: [none]

models:
  # Sol 6.0 is the newest.
- model_id: sol-6.0
  wire_model_id: gpt-6-sol
  knowledge_cutoff: '2026-01-01'
  prices:
  - service_tier: auto
    request: 2.0

- &sol_5_6
  model_id: sol-5.6
  knowledge_cutoff: '2025-06-01'
  prices:
  - service_tier: auto
    request: 4.0
  wire_model_id: gpt-5.6-
    sol

# The alias routes to Sol 5.6.
- <<: *sol_5_6
  model_id: gpt-5.6
  wire_model_id: gpt-5.6

providers:
  openai:
    sends: []
"""


def test_an_anchored_row_and_a_merged_row_are_rows_too(tmp_path: Path) -> None:
    """``- &sol_5_6`` opens Sol 5.6; ``- <<: *sol_5_6`` opens one that states less.

    Sol 5.6 ends in an id over two lines: Sol 5.5 goes below both.
    """
    path = tmp_path / "openai.yaml"
    path.write_text(_ANCHORED)
    sol_55 = NewRow(
        model_id="sol-5.5",
        wire_model_id="gpt-5.5-sol",
        knowledge_cutoff="2025-01-01",
        limits={"max_request_tokens": 2_000},
        thinking_effort=("low", "high"),
        thinking_effort_default="low",
        prices=(_BASE,),
    )
    # Sol 5.6's card is GPT-5.6's too, by the merge: neither row's copy is written.
    shared = [
        _finding("sol-5.6", "price.auto.request", "4.5"),
        _finding("gpt-5.6", "price.auto.request", "5"),
    ]
    assert (
        write(
            path,
            [
                _finding("sol-6.0", "price.auto.request", "2.5"),
                *shared,
                _absent(sol_55),
            ],
        )
        == shared
    )
    text = path.read_text()
    assert "request: 2.5" in text
    assert "  wire_model_id: gpt-5.6-\n    sol\n\n- model_id: sol-5.5\n" in text
    assert text.index("sol-5.5") < text.index("# The alias routes to Sol 5.6.")
    table = ModelTable.from_yaml(path, section="openai")
    assert list(table) == ["sol-6.0", "sol-5.6", "sol-5.5", "gpt-5.6"]
    assert table["gpt-5.6"].prices["auto", 1, date(2026, 1, 1)].request == 4.0
    assert table["sol-5.5"].limits.max_request_tokens == 2_000


def test_the_newest_model_goes_above_the_comment_of_the_row_it_leads(
    tmp_path: Path,
) -> None:
    """The comment above Sol 6.0 is Sol 6.0's, indented or not."""
    path = tmp_path / "openai.yaml"
    path.write_text(_ANCHORED)
    sol_61 = NewRow(
        model_id="sol-6.1",
        wire_model_id="gpt-6.1-sol",
        knowledge_cutoff="2026-03-01",
        limits={},
        thinking_effort=("low", "high"),
        thinking_effort_default="low",
        prices=(_BASE,),
    )
    assert write(path, [_absent(sol_61)]) == []
    text = path.read_text()
    assert "models:\n- model_id: sol-6.1\n" in text
    assert (
        "    cache_read: 0.2\n\n  # Sol 6.0 is the newest.\n- model_id: sol-6.0\n"
        in text
    )
    assert list(ModelTable.from_yaml(path, section="openai"))[:2] == [
        "sol-6.1",
        "sol-6.0",
    ]


def test_a_limit_of_a_row_with_no_window_is_left(tmp_path: Path) -> None:
    """Sol 6.0 states no ``limits`` to rewrite."""
    path = tmp_path / "openai.yaml"
    path.write_text(_ANCHORED)
    unwritable = [
        replace(_finding("sol-6.0", "max_request_tokens", "2000"), provider="openai"),
    ]
    assert write(path, unwritable) == unwritable
    assert path.read_text() == _ANCHORED


_ROWS = """\
reference:
  thinking: {effort: [none], budget: [none], output: [none]}

# Every model, newest first.
models:
- model_id: neon-4.0
  prices: [{request: 1.0}]
- model_id: argon-4.0
  prices: [{request: 1.0}]
- {model_id: flash-3.8, prices: [{request: 1.0}]}

providers:
  api:
    sends: []
"""


def test_a_card_without_a_tier_is_the_auto_one(tmp_path: Path) -> None:
    """Every row is read for the one named, Flash's one line among them."""
    path = tmp_path / "google.yaml"
    path.write_text(_ROWS)
    finding = replace(_finding("neon-4.0", "price.auto.request", "2"), provider="api")
    assert write(path, [finding]) == []
    assert path.read_text() == _ROWS.replace(
        "  prices: [{request: 1.0}]\n- model_id: argon",
        "  prices: [{request: 2.0}]\n- model_id: argon",
    )


def test_a_new_model_above_a_row_without_a_comment_goes_right_above_it(
    tmp_path: Path,
) -> None:
    """Neon 4.0 has no comment: the one above ``models:`` is no row's."""
    path = tmp_path / "google.yaml"
    path.write_text(_ROWS)
    neon_45 = NewRow(
        model_id="neon-4.5",
        wire_model_id="neon-4.5",
        knowledge_cutoff="2026-01-01",
        limits={},
        thinking_effort=("none",),
        thinking_effort_default="none",
        prices=(_BASE,),
    )
    assert write(path, [_absent(neon_45, model_id="neon-4.5")]) == []
    text = path.read_text()
    assert "# Every model, newest first.\nmodels:\n- model_id: neon-4.5\n" in text
    assert "    cache_read: 0.2\n\n- model_id: neon-4.0\n" in text
    assert list(ModelTable.from_yaml(path, section="api"))[:2] == [
        "neon-4.5",
        "neon-4.0",
    ]


def test_a_model_that_does_not_think_fits_a_reference_stating_no_thinking(
    tmp_path: Path,
) -> None:
    """No ``thinking`` reads as every axis ``[none]``, as the table reads it."""
    path = tmp_path / "x.yaml"
    path.write_text("models: []\nproviders:\n  x:\n    sends: []\n")
    row = NewRow(
        model_id="plain-1.0",
        wire_model_id="plain-1",
        knowledge_cutoff="2026-01-01",
        limits={"max_request_tokens": 1_000},
        thinking_effort=("none",),
        thinking_effort_default="none",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    assert "thinking" not in path.read_text()
    (only,) = ModelTable.from_yaml(path, section="x").rows
    assert (only.thinking_effort, only.limits.max_request_tokens) == ({"none"}, 1_000)


def test_a_new_model_states_each_default_and_band_its_layer_does_not(
    tmp_path: Path,
) -> None:
    """Thinking off only up to ``high``, off unasked, rates past 200k tokens."""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-krypton-5.0",
        wire_model_id="gemini-5-krypton",
        knowledge_cutoff="2026-09-01",
        limits={},
        thinking_effort=("min", "low", "medium", "high"),
        thinking_effort_default="high",
        thinking_budget=("auto",),
        thinking_budget_default="auto",
        thinking_off_max_effort="high",
        prices=(_BASE, replace(_BASE, effective_prompt_tokens=200_000, request=8.0)),
    )
    assert write(path, [_absent(row)]) == []
    added = path.read_text().split("models:\n")[1]
    assert "  thinking_off_max_effort: high\n  knowledge_cutoff" in added
    # Its levels and budget modes are the reference's: no ``thinking`` is written.
    assert "  thinking:\n" not in added
    # The reference's own defaults: neither restated.
    assert "thinking_budget_default" not in added
    assert "thinking_effort_default" not in added
    assert "    effective_prompt_tokens: 200_000\n    request: 8.0\n" in added
    # Unstated, a row starts at the lowest mode offered, ``none``: ``auto`` is said.
    path.write_text(_CATALOG)
    write(path, [_absent(replace(row, thinking_budget=("none", "auto")))])
    assert "  thinking_budget_default: auto\n" in path.read_text()


def test_a_model_no_layer_takes_writes_its_budget_modes_out(tmp_path: Path) -> None:
    """Budget modes are part of the fit: a profile with its levels but not them is
    no fit, and the modes it states are written, not the reference's.
    """
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    row = NewRow(
        model_id="gemini-krypton-5.0",
        wire_model_id="gemini-5-krypton",
        knowledge_cutoff="2026-09-01",
        limits={},
        thinking_effort=("low", "medium", "high", "xhigh"),
        thinking_effort_default="high",
        thinking_budget=("none", "auto"),
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    text = path.read_text()
    assert "  thinking:\n    effort: [low, medium, high, xhigh]\n" in text
    assert "    budget: [none, auto]\n    output: [none, text]\n" in text
    assert "profile: levels_xhigh\n  knowledge_cutoff: '2026-09-01'" not in text


def test_a_default_the_reference_states_is_not_restated(tmp_path: Path) -> None:
    """The reference starts thinking off up to ``high``, and budgets at ``auto``."""
    path = tmp_path / "x.yaml"
    path.write_text(
        "reference:\n"
        "  thinking_off_max_effort: high\n"
        "  thinking_budget_default: auto\n"
        "  thinking:\n"
        "    effort: [min, low, medium, high]\n"
        "    budget: [none, auto]\n"
        "    output: [none]\n"
        "models: []\n"
        "providers:\n  x:\n    sends: []\n",
    )
    row = NewRow(
        model_id="plain-1.0",
        wire_model_id="plain-1",
        knowledge_cutoff="2026-01-01",
        limits={"max_request_tokens": 1_000},
        thinking_effort=("min", "low", "medium", "high"),
        thinking_effort_default="min",
        thinking_budget=("none", "auto"),
        thinking_budget_default="auto",
        thinking_off_max_effort="high",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    added = path.read_text().split("models:\n")[1]
    assert "thinking_off_max_effort" not in added
    assert "thinking_budget_default" not in added
    (only,) = ModelTable.from_yaml(path, section="x").rows
    assert only.default.thinking_budget == "auto"


def test_a_new_model_takes_the_first_profile_whose_window_holds_its_own(
    tmp_path: Path,
) -> None:
    """Both profiles have its levels; only the second has its 2M window."""
    profile = (
        "  levels_xhigh_2m:\n"
        "    limits:\n"
        "      max_request_tokens: 2_097_152\n"
        "      max_response_tokens: 65_536\n"
        "      max_request_bytes: 104_857_600\n"
        "      truncates_at_context: true\n"
        "    thinking:\n"
        "      effort: [low, medium, high, xhigh]\n"
        "      budget: [auto]\n"
        "      output: [none, text]\n"
        "\nmodels:\n"
    )
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG.replace("\nmodels:\n", profile, 1))
    row = NewRow(
        model_id="gemini-krypton-5.0",
        wire_model_id="gemini-5-krypton",
        knowledge_cutoff="2026-09-01",
        limits={"max_request_tokens": 2_097_152},
        thinking_effort=("low", "medium", "high", "xhigh"),
        thinking_effort_default="high",
        prices=(_BASE,),
    )
    assert write(path, [_absent(row)]) == []
    added = path.read_text().split("- model_id: gemini-krypton-5.0\n")[1]
    assert "  profile: levels_xhigh_2m\n" in added
    assert "limits" not in added
    table = ModelTable.from_yaml(path, section="google")
    assert table["krypton"].limits.max_request_tokens == 2_097_152


def test_a_window_anchored_below_the_new_row_is_stated_not_aliased(
    tmp_path: Path,
) -> None:
    """Neon 3.9 goes below Neon 4.0, above Carbon, whose window it shares."""
    carbon = """\
- model_id: gemini-carbon-4.0
  wire_model_id: gemini-4-carbon
  profile: levels_xhigh
  limits: &carbon
    max_request_tokens: 2_097_152
    max_response_tokens: 65_536
    max_request_bytes: 104_857_600
    truncates_at_context: true
  prices:
  - service_tier: auto
    request: 2.0

providers:
"""
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG.replace("providers:\n", carbon))
    row = replace(
        _NEON_39,
        limits={"max_request_tokens": 2_097_152},
        thinking_effort=("low", "medium", "high", "xhigh"),
    )
    assert write(path, [_absent(row)]) == []
    text = path.read_text()
    assert text.index("gemini-neon-3.9") < text.index("gemini-carbon-4.0")
    assert "*carbon" not in text
    table = ModelTable.from_yaml(path, section="google")
    assert table["gemini-neon-3.9"].limits.max_request_tokens == 2_097_152


def test_a_new_model_its_pages_state_too_little_of_is_left_for_a_person(
    tmp_path: Path,
) -> None:
    path = tmp_path / "google.yaml"
    path.write_text(_CATALOG)
    unstated = _absent(None)
    assert write(path, [unstated]) == [unstated]
    assert path.read_text() == _CATALOG


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
