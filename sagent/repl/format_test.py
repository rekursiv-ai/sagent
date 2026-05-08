"""Tests for repl.render."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from sagent.repl.format import (
    format_count,
    format_elapsed,
    print_user_bar,
    set_terminal_title,
)


class TestPrintUserBar:
    def test_narrow_terminal_falls_back(self) -> None:
        out = MagicMock()
        out.width = 2
        print_user_bar(out, "hello")
        out.print.assert_called()

    def test_non_integer_width_treated_as_zero(self) -> None:
        out = MagicMock()
        out.width = "not-a-number"
        print_user_bar(out, "hi")
        out.print.assert_called()


class TestSetTerminalTitle:
    def test_writes_osc_on_tty(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.format.sys.stderr", fake):
            set_terminal_title("hello")
        assert fake.getvalue() == "\x1b]0;hello\x07"

    def test_noop_when_not_tty(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: False  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.format.sys.stderr", fake):
            set_terminal_title("hello")
        assert fake.getvalue() == ""

    def test_collapses_newlines(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.format.sys.stderr", fake):
            set_terminal_title("line one\nline two")
        assert "\n" not in fake.getvalue()
        assert "line one line two" in fake.getvalue()

    def test_truncates_long(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.format.sys.stderr", fake):
            set_terminal_title("x" * 500)
        out = fake.getvalue()
        assert out.endswith("\x07")
        assert "…" in out
        payload = out[len("\x1b]0;") : -len("\x07")]
        assert len(payload) == 80


class TestFormatElapsed:
    def test_under_60s_integer(self) -> None:
        assert format_elapsed(0.3) == "0s"
        assert format_elapsed(2.5) == "2s"
        assert format_elapsed(12.0) == "12s"
        assert format_elapsed(59.9) == "59s"

    def test_rolls_at_60s(self) -> None:
        assert format_elapsed(60.0) == "1m 0s"
        assert format_elapsed(91.0) == "1m 31s"
        assert format_elapsed(3599.0) == "59m 59s"

    def test_rolls_at_60min(self) -> None:
        assert format_elapsed(3600.0) == "1h 0m"
        assert format_elapsed(3600.0 + 23 * 60.0) == "1h 23m"
        assert format_elapsed(2 * 3600.0 + 17 * 60.0) == "2h 17m"


class TestFormatCount:
    def test_sub_10k_verbatim(self) -> None:
        assert format_count(0) == "0"
        assert format_count(7) == "7"
        assert format_count(412) == "412"
        assert format_count(9999) == "9999"

    def test_thousands(self) -> None:
        assert format_count(10_000) == "10K"
        assert format_count(12_000) == "12K"
        assert format_count(412_000) == "412K"
        assert format_count(999_499) == "999K"

    def test_no_1000k_artifact_at_1m_boundary(self) -> None:
        """Regression: banker's-rounded ``f"{n/1000:.0f}K"`` produces
        ``"1000K"`` for ``n in [999_500, 999_999]``. The threshold must
        step to the M scale before that band.
        """
        assert format_count(999_500) == "1.0M"
        assert format_count(999_999) == "1.0M"

    def test_millions(self) -> None:
        assert format_count(1_000_000) == "1.0M"
        assert format_count(1_800_000) == "1.8M"
        assert format_count(241_000_000) == "241.0M"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
