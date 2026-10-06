"""Tests for ``agent.state``: the read cache's memory bound."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

from sagent.agent.state import MAX_CACHED_TEXT_CHARS, ToolState


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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
