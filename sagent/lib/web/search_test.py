"""Tests for sagent.lib.web.search."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import json
import os
import urllib.error

import bs4
import pytest

from sagent.lib.web.fetch import FetchError
from sagent.lib.web.search import (
    CaptchaError,
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchError,
    SearchResult,
    TorrentResult,
    VideoResult,
    _duckduckgo_check_captcha,
    _duckduckgo_extract_url,
    _duckduckgo_parse,
    _duckduckgo_quote_bangs,
    _searxng_url,
    _strip_scripts,
    duckduckgo,
    gsa_headers_for_query,
    search,
    searxng,
)


def _patch_fetch(
    return_value: bytes = b"{}",
    side_effect: Any = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if side_effect is not None:
        kwargs["side_effect"] = side_effect
    else:
        kwargs["return_value"] = return_value
    return patch("sagent.lib.web.search.fetch", **kwargs)


@contextmanager
def _patch_searxng_fetch(payload: dict[str, Any]) -> Generator[MagicMock]:
    body = json.dumps(payload).encode()
    with (
        patch.dict(
            os.environ,
            {"SEARXNG_URL": "https://search.example.test/"},
        ),
        _patch_fetch(return_value=body) as mock,
    ):
        yield mock


class TestStripScripts:
    def test_removes_script_tags(self) -> None:
        soup = bs4.BeautifulSoup(
            "<div><p>Keep</p><script>remove()</script></div>",
            "html.parser",
        )
        _strip_scripts(soup)
        assert "remove" not in soup.get_text()
        assert "Keep" in soup.get_text()


class TestSearchDispatch:
    def test_network_error_normalized(self) -> None:
        err = urllib.error.URLError(ConnectionResetError(104, "reset"))
        with (
            _patch_fetch(side_effect=err),
            pytest.raises(SearchError, match="duckduckgo"),
        ):
            search("cats", backend="duckduckgo")

    def test_fetch_error_normalized(self) -> None:
        # INF-027: a backend's FetchError (e.g. HTTP 503) must surface as a
        # SearchError, not escape raw to the caller.
        err = FetchError("https://duckduckgo.com", 503, {}, b"unavailable")
        with (
            _patch_fetch(side_effect=err),
            pytest.raises(SearchError, match="duckduckgo"),
        ):
            search("cats", backend="duckduckgo")


class TestGsaHeaders:
    def test_query_selects_stable_user_agent(self) -> None:
        with patch(
            "sagent.lib.web.search._get_gsa_useragents",
            return_value=("ua0", "ua1", "ua2"),
        ):
            assert gsa_headers_for_query("same") == gsa_headers_for_query("same")
            assert gsa_headers_for_query("same")["User-Agent"].endswith(" NSTNWV")


class TestSearchSearxng:
    def test_requires_env_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        with pytest.raises(RuntimeError, match="SEARXNG_URL"):
            _searxng_url()

    def test_parses_results(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://example.com/page",
                    "title": "Example Title",
                    "content": "Snippet text.",
                },
            ],
        }
        with _patch_searxng_fetch(payload):
            results = searxng("test query")
        assert results == [
            SearchResult(
                url="https://example.com/page",
                title="Example Title",
                snippet="Snippet text.",
            ),
        ]

    def test_num_results_caps(self) -> None:
        payload = {
            "results": [
                {
                    "url": f"https://e{i}.com",
                    "title": f"T{i}",
                    "content": "s",
                }
                for i in range(5)
            ],
        }
        with _patch_searxng_fetch(payload):
            results = searxng("test", num_results=2)
        assert len(results) == 2

    def test_missing_results_key(self) -> None:
        with _patch_searxng_fetch({}):
            assert searxng("test") == []

    def test_non_dict_json_body_yields_empty(self) -> None:
        # A well-formed but non-object JSON response must not raise
        # AttributeError on ``.get`` -- it yields an empty result.
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(return_value=b"[]"),
        ):
            assert searxng("test") == []

    def test_non_list_results_field_yields_empty(self) -> None:
        with _patch_searxng_fetch({"results": "oops"}):
            assert searxng("test") == []

    def test_missing_env_raises_search_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        with pytest.raises(SearchError, match="SEARXNG_URL"):
            searxng("test")

    def test_missing_fields_default_to_empty(self) -> None:
        with _patch_searxng_fetch({"results": [{}]}):
            results = searxng("test")
        assert results == [
            SearchResult(url="", title="", snippet=""),
        ]

    def test_url_includes_query(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("hello world")
        url = mock.call_args.args[0]
        assert "q=hello+world" in url
        assert "format=json" in url
        assert url.startswith("https://search.example.test/search?")

    def test_fetch_error_propagates(self) -> None:
        err = FetchError("https://x.com", 500, {}, b"")
        with (
            patch.dict(
                os.environ,
                {"SEARXNG_URL": "https://search.example.test/"},
            ),
            _patch_fetch(side_effect=err),
            pytest.raises(FetchError),
        ):
            searxng("test")

    def test_default_category_is_general(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q")
        assert "categories=general" in mock.call_args.args[0]

    def test_social_media_category_url_encodes_space(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            results = searxng("q", categories="social media")
        # SearXNG's tab value carries a space; it must be percent-encoded.
        assert "categories=social+media" in mock.call_args.args[0]
        assert results == []


class TestSearchSearxngScience:
    _PAPER_PAYLOAD: ClassVar[dict[str, Any]] = {
        "results": [
            {
                "template": "paper.html",
                "url": "https://doi.org/10.1/x",
                "title": "Attention Is All You Need",
                "content": "We propose the Transformer.",
                "authors": ["Vaswani", "Shazeer", 42],
                "journal": "NeurIPS",
                "doi": "10.1/x",
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
                "publishedDate": "2017-06-12T00:00:00",
                "tags": ["Computer Science"],
                "comments": "100,000 citations from the year 2017 to 2024",
            }
        ]
    }

    def test_science_returns_paper_results(self) -> None:
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            results = searxng("transformer", categories="science")
        assert results == [
            PaperResult(
                url="https://doi.org/10.1/x",
                title="Attention Is All You Need",
                snippet="We propose the Transformer.",
                authors=("Vaswani", "Shazeer"),
                journal="NeurIPS",
                doi="10.1/x",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                # SearXNG emits naive ISO timestamps; fromisoformat keeps them
                # naive, so the expected value is naive too.
                published=datetime(2017, 6, 12),  # noqa: DTZ001
                tags=("Computer Science",),
                citations=100_000,
            )
        ]

    def test_paper_result_is_a_search_result(self) -> None:
        # LSB: a web-result consumer reading url/title/snippet works unchanged.
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            (paper,) = searxng("transformer", categories="science")
        assert isinstance(paper, SearchResult)
        assert paper.url
        assert paper.title
        assert paper.snippet

    def test_science_category_sent(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q", categories="science")
        assert "categories=science" in mock.call_args.args[0]

    def test_sparse_paper_defaults_empty(self) -> None:
        with _patch_searxng_fetch({"results": [{"template": "paper.html"}]}):
            (paper,) = searxng("q", categories="science")
        assert paper == PaperResult(url="", title="", snippet="")
        assert paper.authors == ()
        assert paper.citations is None

    def test_malformed_published_date_is_none(self) -> None:
        payload = {"results": [{"publishedDate": "not-a-date"}]}
        with _patch_searxng_fetch(payload):
            (paper,) = searxng("q", categories="science")
        assert paper.published is None

    def test_unparseable_comments_yields_no_citations(self) -> None:
        payload = {"results": [{"comments": "many citations"}]}
        with _patch_searxng_fetch(payload):
            (paper,) = searxng("q", categories="science")
        assert paper.citations is None

    def test_search_science_passthrough_types_as_papers(self) -> None:
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            results = search("q", backend="searxng", categories="science")
        assert isinstance(results[0], PaperResult)

    def test_categories_rejected_for_non_searxng_backend(self) -> None:
        # A non-default category on a non-searxng backend is rejected at
        # runtime; the type system permits the call (HTML backends share the
        # web-result overload), so this guard is the only enforcement point.
        with pytest.raises(ValueError, match="only supported by the 'searxng'"):
            search("q", backend="duckduckgo", categories="science")


class TestSearchSearxngStructuredCategories:
    def test_images(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://page",
                    "title": "Cat",
                    "content": "a cat",
                    "img_src": "https://img/cat.png",
                    "thumbnail_src": "https://img/cat_t.png",
                    "resolution": "1920x1080",
                    "img_format": "png",
                    "source": "example.com",
                    "filesize": "1MB",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("cat", categories="images")
        assert r == ImageResult(
            url="https://page",
            title="Cat",
            snippet="a cat",
            image_url="https://img/cat.png",
            thumbnail_url="https://img/cat_t.png",
            resolution="1920x1080",
            img_format="png",
            source="example.com",
            filesize="1MB",
        )

    def test_video_extends_media(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://v",
                    "title": "Clip",
                    "content": "desc",
                    "length": "3:21",
                    "views": "1.2M",
                    "author": "Channel",
                    "iframe_src": "https://embed",
                    "thumbnail": "https://t",
                    "publishedDate": "2020-01-02T00:00:00",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("clip", categories="videos")
        assert isinstance(r, MediaResult)  # VideoResult is-a MediaResult
        assert r.length == "3:21"
        assert r.views == "1.2M"
        assert r.author == "Channel"
        assert r.iframe_url == "https://embed"
        assert r.published == datetime(2020, 1, 2)  # noqa: DTZ001 -- naive ISO

    def test_news_is_media_result(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://n",
                    "title": "Headline",
                    "content": "story",
                    "publishedDate": "2026-06-22T00:00:00",
                    "thumbnail": "https://t",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("news", categories="news")
        assert isinstance(r, MediaResult)
        assert not isinstance(r, VideoResult)
        assert r.published == datetime(2026, 6, 22)  # noqa: DTZ001 -- naive ISO

    def test_music_is_media_result(self) -> None:
        payload = {"results": [{"url": "https://s", "audio_src": "https://a"}]}
        with _patch_searxng_fetch(payload):
            (r,) = searxng("song", categories="music")
        assert isinstance(r, MediaResult)
        assert r.audio_url == "https://a"

    def test_map(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://m",
                    "title": "Eiffel Tower",
                    "content": "tower",
                    "latitude": 48.8584,
                    "longitude": 2.2945,
                    "address": {"road": "Champ de Mars", "country": "France", "x": 9},
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("eiffel", categories="map")
        assert isinstance(r, MapResult)
        assert r.latitude == 48.8584
        assert r.longitude == 2.2945
        # Non-string address values are dropped, not coerced.
        assert dict(r.address) == {"road": "Champ de Mars", "country": "France"}

    def test_map_missing_coords_are_none(self) -> None:
        with _patch_searxng_fetch({"results": [{"url": "https://m"}]}):
            (r,) = searxng("x", categories="map")
        assert isinstance(r, MapResult)
        assert r.latitude is None
        assert r.longitude is None
        assert dict(r.address) == {}

    def test_it_dispatches_by_template(self) -> None:
        payload = {
            "results": [
                {
                    "template": "packages.html",
                    "url": "https://pkg",
                    "title": "numpy",
                    "content": "arrays",
                    "package_name": "numpy",
                    "version": "2.0",
                    "license_name": "BSD",
                    "tags": ["python", 7],
                },
                {
                    "template": "code.html",
                    "url": "https://code",
                    "title": "main.py",
                    "content": "def main()",
                    "repository": "org/repo",
                    "filename": "main.py",
                    "code_language": "python",
                },
                {"template": "default.html", "url": "https://w", "title": "wiki"},
            ]
        }
        with _patch_searxng_fetch(payload):
            pkg, code, web = searxng("numpy", categories="it")
        assert isinstance(pkg, PackageResult)
        assert pkg.package_name == "numpy"
        assert pkg.version == "2.0"
        assert pkg.tags == ("python",)
        assert isinstance(code, CodeResult)
        assert code.repository == "org/repo"
        assert code.code_language == "python"
        assert type(web) is SearchResult

    def test_files_dispatches_by_template(self) -> None:
        payload = {
            "results": [
                {
                    "template": "torrent.html",
                    "url": "https://t",
                    "title": "ISO",
                    "content": "",
                    "magnetlink": "magnet:?xt=1",
                    "seed": 10,
                    "leech": 2,
                    "filesize": "700MB",
                },
                {
                    "template": "file.html",
                    "url": "https://f",
                    "title": "doc",
                    "abstract": "a document",
                    "filename": "doc.pdf",
                    "size": "1MB",
                    "mimetype": "application/pdf",
                },
            ]
        }
        with _patch_searxng_fetch(payload):
            torrent, file = searxng("iso", categories="files")
        assert isinstance(torrent, TorrentResult)
        assert torrent.magnet_url == "magnet:?xt=1"
        assert torrent.seed == 10
        assert torrent.leech == 2
        assert isinstance(file, FileResult)
        assert file.filename == "doc.pdf"
        # file.html abstract feeds snippet.
        assert file.snippet == "a document"

    def test_search_passthrough_images(self) -> None:
        payload = {"results": [{"url": "https://p", "img_src": "https://i"}]}
        with _patch_searxng_fetch(payload):
            results = search("q", backend="searxng", categories="images")
        assert isinstance(results[0], ImageResult)


_SINGLE_DDG = """
<html><body>
<div id="links">
  <div class="result results_links results_links_deep web-result">
    <h2 class="result__title">
      <a class="result__a" href="https://example.com/page">Example Title</a>
    </h2>
    <a class="result__snippet">This is the snippet text.</a>
  </div>
</div>
</body></html>
"""

_TWO_DDG = """
<html><body>
<div id="links">
  <div class="result web-result">
    <h2><a class="result__a" href="https://first.com">First</a></h2>
    <a class="result__snippet">First snippet.</a>
  </div>
  <div class="result web-result">
    <h2><a class="result__a" href="https://second.com">Second</a></h2>
    <a class="result__snippet">Second snippet.</a>
  </div>
</div>
</body></html>
"""

_NO_RESULTS_DDG = "<html><body><div>No results.</div></body></html>"

_SCRIPT_DDG = """
<html><body>
<div id="links">
  <div class="result web-result">
    <h2><a class="result__a" href="https://example.com">Title</a></h2>
    <a class="result__snippet">Snippet.<script>noise()</script></a>
  </div>
</div>
</body></html>
"""

_MISSING_LINK_DDG = """
<html><body>
<div id="links"><div class="result web-result"><span>No link here</span></div></div>
</body></html>
"""


class TestQuoteBangsDdg:
    def test_quotes_leading_bang_token(self) -> None:
        assert _duckduckgo_quote_bangs("!w python") == "'!w' python"

    def test_quotes_embedded_bang_token(self) -> None:
        assert _duckduckgo_quote_bangs("python !gh repo") == "python '!gh' repo"

    def test_preserves_normal_query(self) -> None:
        assert _duckduckgo_quote_bangs("python web search") == "python web search"


class TestCheckCaptchaDdg:
    def test_detects_challenge_form(self) -> None:
        with pytest.raises(CaptchaError):
            _duckduckgo_check_captcha(
                '<html><body><form id="challenge-form"></form></body></html>',
            )

    def test_normal_page(self) -> None:
        _duckduckgo_check_captcha("<html><body>ok</body></html>")


class TestExtractUrlDdg:
    def test_direct_url(self) -> None:
        assert (
            _duckduckgo_extract_url("https://example.com/page")
            == "https://example.com/page"
        )

    def test_protocol_relative_url(self) -> None:
        assert (
            _duckduckgo_extract_url("//example.com/page") == "https://example.com/page"
        )

    def test_rejects_non_http_url(self) -> None:
        assert _duckduckgo_extract_url("httpx://example.com/page") is None

    def test_wrapped_url(self) -> None:
        assert (
            _duckduckgo_extract_url(
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%2520b",
            )
            == "https://example.com/a%20b"
        )

    def test_protocol_relative_wrapped_url(self) -> None:
        assert (
            _duckduckgo_extract_url(
                "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage",
            )
            == "https://example.com/page"
        )

    def test_empty_url(self) -> None:
        assert _duckduckgo_extract_url("") is None


class TestParseDdg:
    def test_single(self) -> None:
        results = _duckduckgo_parse(_SINGLE_DDG, 10)
        assert len(results) == 1
        assert results[0].url == "https://example.com/page"
        assert results[0].title == "Example Title"
        assert "snippet text" in results[0].snippet

    def test_two(self) -> None:
        results = _duckduckgo_parse(_TWO_DDG, 10)
        assert [r.title for r in results] == ["First", "Second"]

    def test_max_results_caps(self) -> None:
        assert len(_duckduckgo_parse(_TWO_DDG, 1)) == 1

    def test_no_results(self) -> None:
        assert _duckduckgo_parse(_NO_RESULTS_DDG, 10) == []

    def test_scripts_stripped(self) -> None:
        results = _duckduckgo_parse(_SCRIPT_DDG, 10)
        assert len(results) == 1
        assert "noise" not in results[0].snippet

    def test_missing_link_skipped(self) -> None:
        assert _duckduckgo_parse(_MISSING_LINK_DDG, 10) == []

    def test_long_snippet_not_truncated(self) -> None:
        long = "x" * 500
        html = f"""
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://example.com">T</a></h2>
          <a class="result__snippet">{long}</a>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert len(results) == 1
        assert len(results[0].snippet) >= 500

    def test_no_snippet_ok(self) -> None:
        html = """
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://example.com">Title</a></h2>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert len(results) == 1
        assert results[0].snippet == ""

    def test_wrapped_result_url(self) -> None:
        html = """
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Wrapped</a></h2>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert results[0].url == "https://example.com/page"

    def test_ignores_ad_results(self) -> None:
        html = """
        <html><body>
        <div id="links">
          <div class="result result--ad result--ad--small">
            <h2><a class="result__a" href="https://ad.example.com">Ad</a></h2>
          </div>
          <div class="result web-result">
            <h2><a class="result__a" href="https://example.com">Organic</a></h2>
          </div>
        </div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert [r.title for r in results] == ["Organic"]

    def test_requires_links_container(self) -> None:
        html = """
        <html><body>
          <div class="result web-result">
            <h2><a class="result__a" href="https://example.com">Loose</a></h2>
          </div>
        </body></html>
        """
        assert _duckduckgo_parse(html, 10) == []


class TestSearchDuckduckgo:
    def test_delegates_to_parse(self) -> None:
        with _patch_fetch(return_value=_SINGLE_DDG.encode()):
            results = duckduckgo("test query")
        assert len(results) == 1
        assert results[0].title == "Example Title"

    def test_uses_normal_browser_request_contract(self) -> None:
        with (
            patch(
                "sagent.lib.web.search._get_gsa_useragents",
                return_value=("ddg-test-ua",),
            ),
            _patch_fetch(return_value=_NO_RESULTS_DDG.encode()) as mock,
        ):
            duckduckgo("test")
        assert mock.call_args.kwargs["method"] == "POST"
        assert mock.call_args.kwargs["data"] == {"q": "test", "b": "", "kl": "wt-wt"}
        assert mock.call_args.kwargs["headers"] == {
            "User-Agent": "ddg-test-ua NSTNWV",
            "Accept": "*/*",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Accept-Language": "all,all-ALL;q=0.7",
            "Referer": "https://html.duckduckgo.com/html/",
        }
        assert "cookies" not in mock.call_args.kwargs
        # INF-025: the GSA mobile UA must be sent verbatim, without fetch's
        # default desktop Chrome sec-ch-ua headers.
        assert mock.call_args.kwargs["raw_headers"] is True
        assert mock.call_args.kwargs["retries"] == 2

    def test_quotes_bangs_before_request(self) -> None:
        with _patch_fetch(
            return_value=_NO_RESULTS_DDG.encode(),
        ) as mock:
            duckduckgo("!w python")
        assert mock.call_args.kwargs["data"]["q"] == "'!w' python"

    def test_rejects_too_long_query(self) -> None:
        # INF-026: an over-length query must raise, not silently return [].
        with _patch_fetch() as mock, pytest.raises(SearchError, match="exceeds"):
            duckduckgo("x" * 500)
        mock.assert_not_called()

    def test_fetch_error_propagates(self) -> None:
        err = FetchError("https://x.com", 500, {}, b"")
        with (
            _patch_fetch(side_effect=err),
            pytest.raises(FetchError),
        ):
            duckduckgo("test")


class TestHeadersArg:
    def test_searxng_custom_headers(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q", headers={"User-Agent": "custom/1.0"})
        assert mock.call_args.kwargs["headers"] == {
            "User-Agent": "custom/1.0",
        }

    def test_ddg_custom_headers_merge_with_defaults(self) -> None:
        with _patch_fetch(
            return_value=_NO_RESULTS_DDG.encode(),
        ) as mock:
            duckduckgo("q", headers={"User-Agent": "x"})
        assert mock.call_args.kwargs["headers"] == {
            "User-Agent": "x",
            "Accept": "*/*",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Accept-Language": "all,all-ALL;q=0.7",
            "Referer": "https://html.duckduckgo.com/html/",
        }


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
