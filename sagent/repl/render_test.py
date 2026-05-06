"""Tests for repl.render."""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from sagent.custom_types import TokenCount
from sagent.repl.render import (
    format_count,
    format_elapsed,
    print_user_bar,
    render_toolbar,
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


class TestRenderToolbar:
    def _agent(self, **kw: Any) -> Any:
        a = MagicMock()
        a.active = kw.get("active", False)
        a.request_start_time = kw.get("request_start_time", 0.0)
        a.total_active_elapsed_seconds = kw.get("total_active_elapsed_seconds", 0.0)
        a.total_tokens = kw.get("total_tokens", TokenCount())
        a.total_cost_usd = kw.get("total_cost_usd", 0.0)
        a.live_model_response_tokens = kw.get("live_model_response_tokens", 0)
        return a

    def test_empty_when_nothing_to_show(self) -> None:
        assert render_toolbar(self._agent()) == ""

    def test_idle_renders_cumulative_bracket(self) -> None:
        agent = self._agent(
            total_active_elapsed_seconds=50.0,
            total_tokens=TokenCount(
                input_tokens=18,
                output_tokens=3114,
                cache_creation_tokens=140_000,
                cache_read_tokens=1_800_000,
            ),
            total_cost_usd=0.98,
        )
        assert render_toolbar(agent) == "[50s 18↑ 3114↓ 140K↟ 1.8M↡ $0.98]"

    def test_idle_minutes_format(self) -> None:
        agent = self._agent(
            total_active_elapsed_seconds=83.0,
            total_tokens=TokenCount(input_tokens=67, output_tokens=8902),
            total_cost_usd=14.71,
        )
        assert render_toolbar(agent) == "[1m 23s 67↑ 8902↓ 0↟ 0↡ $14.71]"

    def test_idle_hours_format(self) -> None:
        agent = self._agent(
            total_active_elapsed_seconds=8220.0,
            total_tokens=TokenCount(
                input_tokens=3200,
                output_tokens=412_000,
                cache_creation_tokens=18_000_000,
                cache_read_tokens=241_000_000,
            ),
            total_cost_usd=487.12,
        )
        assert render_toolbar(agent) == ("[2h 17m 3200↑ 412K↓ 18.0M↟ 241.0M↡ $487.12]")

    def test_idle_zero_cost_still_renders_bracket(self) -> None:
        agent = self._agent(
            total_active_elapsed_seconds=1.0,
            total_tokens=TokenCount(input_tokens=1, output_tokens=2),
            total_cost_usd=0.0,
        )
        assert render_toolbar(agent) == "[1s 1↑ 2↓ 0↟ 0↡ $0.00]"

    def test_active_prefixes_spinner_and_ticks_elapsed(self) -> None:
        agent = self._agent(
            active=True,
            request_start_time=0.0,
            total_active_elapsed_seconds=0.0,
            total_tokens=TokenCount(input_tokens=10, output_tokens=20),
            total_cost_usd=0.05,
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 5.0
            agent.total_active_elapsed_seconds = 5.0
            s = render_toolbar(agent)
        assert s.endswith("[5s 10↑ 20↓ 0↟ 0↡ $0.05]")
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        assert s[0] in spinner
        assert s[1] == " "

    def test_active_with_no_completed_runs_renders(self) -> None:
        agent = self._agent(
            active=True,
            request_start_time=0.0,
            total_active_elapsed_seconds=2.0,
            total_tokens=TokenCount(),
            total_cost_usd=0.0,
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 2.0
            s = render_toolbar(agent)
        assert "[2s 0↑ 0↓ 0↟ 0↡ $0.00]" in s

    def test_active_includes_live_output_token_estimate(self) -> None:
        agent = self._agent(
            active=True,
            request_start_time=0.0,
            total_active_elapsed_seconds=10.0,
            total_tokens=TokenCount(input_tokens=100, output_tokens=500),
            total_cost_usd=0.10,
            live_model_response_tokens=250,
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 10.0
            s = render_toolbar(agent)
        assert "750↓" in s

    def test_idle_ignores_live_output_token_estimate(self) -> None:
        agent = self._agent(
            active=False,
            total_active_elapsed_seconds=10.0,
            total_tokens=TokenCount(input_tokens=100, output_tokens=500),
            total_cost_usd=0.10,
            live_model_response_tokens=250,
        )
        s = render_toolbar(agent)
        assert "500↓" in s
        assert "750" not in s


class TestSetTerminalTitle:
    def test_writes_osc_on_tty(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.render.sys.stderr", fake):
            set_terminal_title("hello")
        assert fake.getvalue() == "\x1b]0;hello\x07"

    def test_noop_when_not_tty(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: False  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.render.sys.stderr", fake):
            set_terminal_title("hello")
        assert fake.getvalue() == ""

    def test_collapses_newlines(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.render.sys.stderr", fake):
            set_terminal_title("line one\nline two")
        assert "\n" not in fake.getvalue()
        assert "line one line two" in fake.getvalue()

    def test_truncates_long(self) -> None:
        fake = StringIO()
        fake.isatty = lambda: True  # ty: ignore[invalid-assignment] -- test mock
        with patch("sagent.repl.render.sys.stderr", fake):
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
