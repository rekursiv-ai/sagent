"""Tests for ``tools.web_search``: pluggable search backend tool."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import asyncio

from sagent.lib.web.search import (
    CaptchaError,
    ImageResult,
    MapResult,
    PaperResult,
    SearchResult,
    TorrentResult,
    VideoResult,
)
from sagent.tools.web_search import (
    WebSearch,
    _build_query,
    _format_result,
)
from sagent.types.runtime import ToolResult


def test_websearch_metadata() -> None:
    t = WebSearch()
    assert t.name == "WebSearch"
    assert t.tool_id == "application/x-tool-websearch"


def test_description_resolves() -> None:
    t = WebSearch()
    desc = t.description
    # Assert the actual resolved content, not just type/non-emptiness, so a
    # wrong asset or empty template is caught.
    assert "Web search" in desc
    assert "Sources:" in desc


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

    def fake_search(
        q: str, *, backend: object, categories: object = "general"
    ) -> list[SearchResult]:
        captured["q"] = q
        captured["backend"] = backend
        captured["categories"] = categories
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        _ = asyncio.run(WebSearch().run({"query": "x", "backend": "duckduckgo"}))
    assert captured["backend"] == "duckduckgo"
    assert captured["categories"] == "general"


def test_run_with_allowed_and_blocked_domains() -> None:
    captured: dict[str, object] = {}

    def fake_search(
        q: str, *, backend: object, categories: object = "general"
    ) -> list[SearchResult]:
        captured["q"] = q
        del backend, categories
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


def test_run_category_forces_searxng_backend() -> None:
    captured: dict[str, object] = {}

    def fake_search(
        q: str, *, backend: object, categories: object = "general"
    ) -> list[SearchResult]:
        del q
        captured["backend"] = backend
        captured["categories"] = categories
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        # Caller leaves backend at default; a non-general category forces searxng.
        _ = asyncio.run(WebSearch().run({"query": "x", "categories": "science"}))
    assert captured["backend"] == "searxng"
    assert captured["categories"] == "science"


def test_run_invalid_category_errors() -> None:
    result = asyncio.run(WebSearch().run({"query": "x", "categories": "bogus"}))
    assert result.is_error
    assert "Invalid category" in result.content


def test_format_paper_result() -> None:
    out = _format_result(
        PaperResult(
            url="https://doi.org/10.1/x",
            title="Attn",
            snippet="abstract",
            authors=("A", "B"),
            journal="NeurIPS",
            doi="10.1/x",
            published=datetime(2017, 6, 1),  # noqa: DTZ001 -- naive ok in test
            citations=42,
        )
    )
    assert "[Attn](https://doi.org/10.1/x)" in out
    assert "abstract" in out
    assert "doi:10.1/x" in out
    assert "cites:42" in out
    assert "2017" in out


def test_format_image_result() -> None:
    out = _format_result(
        ImageResult(
            url="https://p",
            title="Cat",
            snippet="",
            image_url="https://img",
            resolution="1x1",
        )
    )
    assert "https://img" in out
    assert "1x1" in out


def test_format_map_result() -> None:
    out = _format_result(
        MapResult(
            url="https://m",
            title="Tower",
            snippet="",
            latitude=48.8,
            longitude=2.3,
        )
    )
    assert "48.8,2.3" in out


def test_format_torrent_result() -> None:
    out = _format_result(
        TorrentResult(
            url="https://t",
            title="ISO",
            snippet="",
            magnet_url="magnet:?xt=1",
            seed=10,
            leech=2,
        )
    )
    assert "seed:10" in out
    assert "leech:2" in out
    assert "magnet:?xt=1" in out


def test_format_plain_search_result_has_no_detail() -> None:
    out = _format_result(SearchResult(url="https://w", title="W", snippet="s"))
    assert out == "[W](https://w)\ns"


def test_video_detail_precedes_media_dispatch() -> None:
    # VideoResult is-a MediaResult; the formatter must use the video branch.
    out = _format_result(
        VideoResult(url="https://v", title="V", snippet="", views="1M", author="C")
    )
    assert "1M views" in out
    assert "C" in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
