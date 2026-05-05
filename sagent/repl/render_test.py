"""Tests for repl.render."""

from __future__ import annotations

from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from sagent.custom_types import TokenCount
from sagent.repl.render import (
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
        a.live_model_response_tokens = kw.get("live_model_response_tokens", 0)
        a.last_elapsed = kw.get("last_elapsed", 0.0)
        a.last_model_request_tokens = kw.get("last_model_request_tokens", 0)
        a.last_model_response_tokens = kw.get("last_model_response_tokens", 0)
        a.total_cost_usd = kw.get("total_cost_usd", 0.0)
        a.last_run_tokens = kw.get(
            "last_run_tokens",
            TokenCount(
                input_tokens=a.last_model_request_tokens,
                output_tokens=a.last_model_response_tokens,
            ),
        )
        a.last_run_cost_usd = kw.get("last_run_cost_usd", a.total_cost_usd)
        a.active_children = kw.get("active_children", {})
        return a

    def test_active_state_shows_spinner(self) -> None:
        agent = self._agent(
            active=True, request_start_time=0.0, live_model_response_tokens=500
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 10.0
            s = render_toolbar(agent)
        assert "10s" in s
        assert "500" in s

    def test_active_shows_live_cost_from_agent(self) -> None:
        agent = self._agent(
            active=True,
            request_start_time=0.0,
            live_model_response_tokens=0,
            total_cost_usd=0.0234,
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 5.0
            assert "$0.02" in render_toolbar(agent)

    def test_active_hides_cost_when_zero(self) -> None:
        agent = self._agent(
            active=True,
            request_start_time=0.0,
            live_model_response_tokens=0,
            total_cost_usd=0.0,
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 5.0
            assert "$" not in render_toolbar(agent)

    def test_active_rolls_to_minutes_past_60s(self) -> None:
        agent = self._agent(
            active=True, request_start_time=0.0, live_model_response_tokens=0
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 91.0
            assert "1 min 31 sec" in render_toolbar(agent)

    def test_active_rolls_to_hours_past_60min(self) -> None:
        agent = self._agent(
            active=True, request_start_time=0.0, live_model_response_tokens=0
        )
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.time.return_value = 3600.0 + 23 * 60.0
            assert "1 hr 23 min" in render_toolbar(agent)

    def test_idle_shows_last_request_stats(self) -> None:
        agent = self._agent(
            active=False,
            last_elapsed=2.5,
            last_model_request_tokens=1000,
            last_model_response_tokens=200,
            total_cost_usd=0.0123,
        )
        s = render_toolbar(agent)
        assert "2s" in s
        assert "1000" in s
        assert "200" in s
        assert "0.01" in s

    def test_idle_shows_last_run_subtree_stats(self) -> None:
        agent = self._agent(
            active=False,
            last_elapsed=2.5,
            last_model_request_tokens=1000,
            last_model_response_tokens=200,
            total_cost_usd=0.0123,
        )
        agent.last_run_tokens = TokenCount(input_tokens=1017, output_tokens=205)
        agent.last_run_cost_usd = 0.0199

        s = render_toolbar(agent)

        assert "1017" in s
        assert "205" in s
        assert "$0.02" in s

    def test_idle_hides_cost_when_zero(self) -> None:
        agent = self._agent(
            active=False,
            last_elapsed=1.0,
            last_model_request_tokens=1,
            last_model_response_tokens=2,
            total_cost_usd=0.0,
        )
        assert "$" not in render_toolbar(agent)

    def test_empty_when_nothing_to_show(self) -> None:
        assert render_toolbar(self._agent()) == ""


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
        assert format_elapsed(59.9) == "60s"

    def test_rolls_at_60s(self) -> None:
        assert format_elapsed(60.0) == "1 min 0 sec"
        assert format_elapsed(91.0) == "1 min 31 sec"
        assert format_elapsed(3599.0) == "59 min 59 sec"

    def test_rolls_at_60min(self) -> None:
        assert format_elapsed(3600.0) == "1 hr 0 min"
        assert format_elapsed(3600.0 + 23 * 60.0) == "1 hr 23 min"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
