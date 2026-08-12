"""Tests for ``tools.web_search``: pluggable search backend tool."""

from __future__ import annotations

from unittest.mock import patch

import asyncio

from wesearch.search.custom_types import SearchResult
from wesearch.types.errors import GoogleSorryError, PuzzleChallengeError

import pytest

from sagent.tools.paper_search import PaperSearch
from sagent.tools.web_search import WebSearch, _build_query


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
    # The @property exists to substitute {{NOW}} each access; an unsubstituted
    # template token must never survive into the rendered description.
    assert "{{NOW}}" not in desc


def test_papersearch_description_resolves_now() -> None:
    # PaperSearch.description is also a @property; same {{NOW}} contract.
    assert "{{NOW}}" not in PaperSearch().description


def test_summary_short_query() -> None:
    t = WebSearch()
    assert t.summary({"query": "cats"}) == "WebSearch 'cats'"


def test_summary_keeps_long_query() -> None:
    t = WebSearch()
    q = "x" * 80
    assert t.summary({"query": q}) == f"WebSearch {q!r}"


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


def test_build_query_rejects_non_string_domains() -> None:
    # Raising, not skipping: dropping the bad member ran an UNRESTRICTED search
    # while the caller believed it was scoped -- failing open on the one
    # argument whose whole purpose is to restrict.
    with pytest.raises(ValueError, match="non-hostname"):
        _build_query("ml", [1, "good.com"], None)


def test_build_query_rejects_non_list() -> None:
    with pytest.raises(TypeError, match="must be a list of hostnames"):
        _build_query("ml", "not-a-list", None)


def test_build_query_rejects_operator_injection() -> None:
    # A domain value carrying an embedded operator must not splice into the
    # query and un-scope/contradict the filter.
    with pytest.raises(ValueError, match="non-hostname"):
        _build_query("ml", ["example.com -site:trusted.com"], None)


def test_build_query_rejects_whitespace_and_non_hostname() -> None:
    for bad in ("not a host", "no-dot"):
        with pytest.raises(ValueError, match="non-hostname"):
            _build_query("ml", [bad], None)
    with pytest.raises(ValueError, match="non-hostname"):
        _build_query("ml", None, ["a.b -c"])


def test_build_query_accepts_wildcard_and_port() -> None:
    q = _build_query("ml", ["*.arxiv.org", "host.com:8080"], None)
    assert "site:*.arxiv.org" in q
    assert "site:host.com:8080" in q


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


def test_run_returns_every_result() -> None:
    hits = [
        SearchResult(url=f"https://x/{i}", title=f"T{i}", snippet="s")
        for i in range(25)
    ]
    with patch("sagent.tools.web_search.search", return_value=hits):
        result = asyncio.run(WebSearch().run({"query": "many"}))
    # The backend's own ``limit`` governs; a silent [:10] slice dropped
    # results the caller had already paid to fetch.
    assert "T0" in result.content
    assert "T9" in result.content
    assert "T24" in result.content


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
        side_effect=PuzzleChallengeError("captcha"),
    ):
        result = asyncio.run(WebSearch().run({"query": "x"}))
    assert result.is_error
    # The tool surfaces the bot-flag's specific guidance, not a bare "HTTP 403".
    assert PuzzleChallengeError.guidance in result.content


def test_run_bot_flag_preserves_per_instance_reason() -> None:
    # F4: a per-instance reason (e.g. scholar cooldown "~Nh on this IP") must
    # NOT be discarded in favor of the static class guidance.
    with patch(
        "sagent.tools.web_search.search",
        side_effect=GoogleSorryError("cooling down ~5.0h on this IP"),
    ):
        result = asyncio.run(WebSearch().run({"query": "x"}))
    assert result.is_error
    assert "cooling down ~5.0h on this IP" in result.content
    assert GoogleSorryError.guidance in result.content


def test_run_rejects_non_string_query() -> None:
    """A schema-invalid query is a directive error, not a search for its repr."""
    with patch("sagent.tools.web_search.search") as mock_search:
        result = asyncio.run(WebSearch().run({"query": []}))
    assert result.is_error
    assert "Invalid query" in result.content
    mock_search.assert_not_called()


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


def test_run_explicit_transport_passes_through() -> None:
    captured: dict[str, object] = {}

    def fake_search(
        q: str,
        *,
        backend: object,
        categories: object,
        transport: object,
    ) -> list[SearchResult]:
        del q, backend, categories
        captured["transport"] = transport
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        result = asyncio.run(WebSearch().run({"query": "x", "transport": "stdlib"}))
    assert not result.is_error
    assert captured["transport"] == "stdlib"


def test_run_rejects_invalid_transport() -> None:
    result = asyncio.run(WebSearch().run({"query": "x", "transport": "requests"}))
    assert result.is_error
    assert "Invalid transport" in result.content


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


def test_run_leaves_backend_resolution_to_the_library() -> None:
    """An unnamed backend reaches ``search`` as ``None``, not a substituted name.

    The tool used to force ``searxng`` itself for a non-general category. Two
    consequences, both fixed by resolving in ``search``: the MCP server had no
    such override and raised on the same call in the public build, and an
    EXPLICIT ``backend`` was silently overwritten (see below).
    """
    captured: dict[str, object] = {}

    def fake_search(
        q: str, *, backend: object, categories: object = "general"
    ) -> list[SearchResult]:
        del q
        captured["backend"] = backend
        captured["categories"] = categories
        return []

    with patch("sagent.tools.web_search.search", side_effect=fake_search):
        _ = asyncio.run(WebSearch().run({"query": "x", "categories": "science"}))
    assert captured["backend"] is None
    assert captured["categories"] == "science"


def test_run_does_not_overwrite_an_explicit_backend() -> None:
    """A stated backend that cannot serve the category ERRORS, silently rewritten.

    ``backend="duckduckgo", categories="science"`` used to run against SearXNG
    instead -- the caller's choice replaced without a word.
    """
    result = asyncio.run(
        WebSearch().run(
            {"query": "x", "backend": "duckduckgo", "categories": "science"}
        )
    )
    assert result.is_error
    assert "only supported by the 'searxng' backend" in result.content


def test_run_invalid_category_errors() -> None:
    result = asyncio.run(WebSearch().run({"query": "x", "categories": "bogus"}))
    assert result.is_error
    # Named for the PARAMETER, which is plural; the spec derives the
    # message from the declared name rather than a hand-written string.
    assert "Invalid categories" in result.content


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
