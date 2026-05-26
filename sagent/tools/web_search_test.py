"""Tests for ``tools.web_search``: pluggable search backend tool."""

from __future__ import annotations

from unittest.mock import patch

import asyncio

from sagent.lib.web.search import CaptchaError, SearchResult
from sagent.tools.web_search import WebSearch, _build_query
from sagent.types.history import ToolResult


def test_websearch_metadata() -> None:
    t = WebSearch()
    assert t.name == "WebSearch"
    assert t.tool_id == "application/x-tool-websearch"


def test_description_resolves() -> None:
    t = WebSearch()
    desc = t.description
    assert isinstance(desc, str)
    assert desc != ""


def test_summary_short_query() -> None:
    t = WebSearch()
    assert t.summary({"query": "cats"}) == "WebSearch 'cats'"


def test_summary_long_query_truncates() -> None:
    t = WebSearch()
    q = "x" * 80
    out = t.summary({"query": q})
    assert out.endswith("...'")


def test_summary_result_returns_none() -> None:
    assert WebSearch().summary_result(ToolResult(call_id="", content="x")) is None


def test_prompt_empty() -> None:
    assert WebSearch().prompt() == ""


def test_build_query_bare() -> None:
    assert _build_query("ml", None, None) == "ml"


def test_build_query_allowed_appended() -> None:
    q = _build_query("ml", ["arxiv.org", "openai.com"], None)
    assert "site:arxiv.org" in q
    assert "site:openai.com" in q


def test_build_query_blocked_appended() -> None:
    q = _build_query("ml", None, ["x.com"])
    assert "-site:x.com" in q


def test_build_query_strips_whitespace() -> None:
    q = _build_query("ml", ["  arxiv.org  "], None)
    assert "site:arxiv.org" in q


def test_build_query_ignores_non_string_domains() -> None:
    q = _build_query("ml", [1, "good.com"], None)
    assert "site:good.com" in q
    assert "site:1" not in q


def test_build_query_ignores_non_list() -> None:
    q = _build_query("ml", "not-a-list", None)
    assert q == "ml"


def test_run_returns_formatted_results() -> None:
    hits = [
        SearchResult(url="https://x/1", title="One", snippet="snip1"),
        SearchResult(url="https://x/2", title="Two", snippet="snip2"),
    ]
    with patch("sagent.tools.web_search.search", return_value=hits):
        result = asyncio.run(WebSearch().run({"query": "cats"}))
    assert "[One](https://x/1)" in result.content
    assert "snip1" in result.content
    assert "[Two](https://x/2)" in result.content


def test_run_no_results() -> None:
    with patch("sagent.tools.web_search.search", return_value=[]):
        result = asyncio.run(WebSearch().run({"query": "nothing"}))
    assert result.content == "(no results)"


def test_run_caps_at_10_results() -> None:
    hits = [
        SearchResult(url=f"https://x/{i}", title=f"T{i}", snippet="s")
        for i in range(25)
    ]
    with patch("sagent.tools.web_search.search", return_value=hits):
        result = asyncio.run(WebSearch().run({"query": "many"}))
    assert "T0" in result.content
    assert "T9" in result.content
    assert "T10" not in result.content


def test_run_runtime_error_returns_tool_result_error() -> None:
    with patch(
        "sagent.tools.web_search.search",
        side_effect=RuntimeError("boom"),
    ):
        result = asyncio.run(WebSearch().run({"query": "x"}))
    assert result.is_error
    assert "boom" in result.content


def test_run_value_error_returns_tool_result_error() -> None:
    with patch(
        "sagent.tools.web_search.search",
        side_effect=ValueError("nope"),
    ):
        result = asyncio.run(WebSearch().run({"query": "x"}))
    assert result.is_error
    assert "nope" in result.content


def test_run_captcha_error_returns_tool_result_error() -> None:
    with patch(
        "sagent.tools.web_search.search",
        side_effect=CaptchaError("captcha"),
    ):
        result = asyncio.run(WebSearch().run({"query": "x"}))
    assert result.is_error
    assert "captcha" in result.content


def test_run_invalid_backend() -> None:
    result = asyncio.run(
        WebSearch().run({"query": "x", "backend": "invalid-engine"}),
    )
    assert result.is_error
    assert "Invalid backend" in result.content


def test_run_valid_backend_passes_through() -> None:
    captured: dict[str, object] = {}

    def fake_search(q: str, *, backend: object) -> list[SearchResult]:
        captured["q"] = q
        captured["backend"] = backend
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        _ = asyncio.run(WebSearch().run({"query": "x", "backend": "duckduckgo"}))
    assert captured["backend"] == "duckduckgo"


def test_run_with_allowed_and_blocked_domains() -> None:
    captured: dict[str, object] = {}

    def fake_search(q: str, *, backend: object) -> list[SearchResult]:
        captured["q"] = q
        del backend
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        _ = asyncio.run(
            WebSearch().run(
                {
                    "query": "ml",
                    "allowed_domains": ["arxiv.org"],
                    "blocked_domains": ["x.com"],
                }
            ),
        )
    q = captured["q"]
    assert isinstance(q, str)
    assert "site:arxiv.org" in q
    assert "-site:x.com" in q


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
