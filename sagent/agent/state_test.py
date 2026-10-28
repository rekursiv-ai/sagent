"""Tests for ``agent.state``: the read cache's memory bound."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

from sagent.agent.state import (
    MAX_CACHED_TEXT_CHARS,
    ConstraintLedger,
    ToolState,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_a_large_read_caches_a_digest_not_the_text(tmp_path: Path) -> None:
    """Every read held the whole file in memory for the session."""
    f = tmp_path / "big.txt"
    text = "x" * (MAX_CACHED_TEXT_CHARS + 1)
    f.write_text(text)
    s = ToolState()
    s.mark_read(str(f), content=text, mtime=1.0)
    assert str(f.resolve()) not in s._content_cache
    assert s.check_stale(str(f)) is False, "an mtime-only bump is not a change"
    f.write_text(text[:-1] + "y")
    os.utime(f, (5.0, 5.0))
    assert s.check_stale(str(f)) is True
    diffs = s.consume_changed_files()
    assert "changed" in diffs[str(f)]
    assert s.consume_changed_files() == {}


def test_a_small_read_keeps_its_text_for_diffs(tmp_path: Path) -> None:
    f = tmp_path / "small.txt"
    f.write_text("alpha\n")
    s = ToolState()
    s.mark_read(str(f), content="alpha\n", mtime=1.0)
    f.write_text("beta\n")
    assert "+beta" in s.consume_changed_files()[str(f)]


def test_constraint_ledger_empty_renders_empty_string() -> None:
    ledger = ConstraintLedger()
    assert ledger.to_prompt_prefix() == ""


def test_constraint_ledger_with_constraints_and_boundaries() -> None:
    ledger = ConstraintLedger(
        user_constraints=("Never commit to main",),
        workspace_boundaries=("/home/g/rekursiv/sagent",),
    )
    xml = ledger.to_prompt_prefix()
    assert "<protected-constraint-ledger>" in xml
    assert "<constraint>Never commit to main</constraint>" in xml
    assert "<boundary>/home/g/rekursiv/sagent</boundary>" in xml


def test_constraint_ledger_with_inquiry_title_and_validation() -> None:
    ledger = ConstraintLedger(
        active_inquiry_id="inq_01",
        inquiry_title="Context Hygiene",
        inquiry_validation="Pass all tests",
        active_experiment_id="exp_01",
    )
    xml = ledger.to_prompt_prefix()
    assert '<active-inquiry id="inq_01">' in xml
    assert "<title>Context Hygiene</title>" in xml
    assert "<validation>Pass all tests</validation>" in xml
    assert "<active-experiment>exp_01</active-experiment>" in xml

    updated = ledger.with_constraint("No deletions")
    assert "No deletions" in updated.user_constraints
    assert "No deletions" in updated.to_prompt_prefix()


def test_constraint_ledger_escapes_xml_special_characters() -> None:
    ledger = ConstraintLedger(
        user_constraints=('Avoid <dir>/file.txt & "tmp"',),
        inquiry_title="Verify A & B < C",
    )
    xml = ledger.to_prompt_prefix()
    assert "Avoid &lt;dir&gt;/file.txt &amp;" in xml
    assert "&lt;" in xml
    assert "<dir>" not in xml


def test_constraint_ledger_renders_inquiry_history() -> None:
    ledger = ConstraintLedger(
        active_inquiry_id="inq_active",
        trackinizer_inquiry_ids=("inq_past_1", "inq_past_2", "inq_active"),
    )
    xml = ledger.to_prompt_prefix()
    assert "<active-inquiry>inq_active</active-inquiry>" in xml
    assert "<inquiry-history>" in xml
    assert "<inquiry-id>inq_past_1</inquiry-id>" in xml
    assert "<inquiry-id>inq_past_2</inquiry-id>" in xml


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
