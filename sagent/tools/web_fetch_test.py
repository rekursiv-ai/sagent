"""Tests for ``tools.web_fetch``: URL fetcher with SSRF guard + cache."""

from __future__ import annotations

from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, patch
from urllib.parse import unquote, urlparse

import asyncio
import json
import socket
import sys

import pytest


# Install a stub for ``trafilatura`` BEFORE importing the production module
# so the lazy import never pays the real package's import cost (~150ms).
# The tests patch ``...trafilatura.extract`` explicitly when needed.
if "trafilatura" not in sys.modules:
    _stub = ModuleType("trafilatura")
    # Set extract via __dict__ to avoid basedpyright's "unknown attr" complaint
    # about ModuleType.
    _stub.__dict__["extract"] = MagicMock(return_value="")
    sys.modules["trafilatura"] = _stub

from sagent.lib.web.fetch import FetchError
from sagent.tools.lib.bash import parse_bash
from sagent.tools.web_fetch import (
    _ADAPTERS,
    _KIND_HTML,
    _KIND_MARKDOWN,
    _KIND_REDDIT,
    _KIND_REDDIT_LISTING,
    _KIND_RSS,
    WebFetch,
    _format_rss,
    _GoogleNewsAdapter,
    _match_http_fetch,
    _parse_rss_cluster,
    _RedditAdapter,
    _url_is_safe,
    _validated_host,
    _XAdapter,
)
from sagent.types.runtime import ToolResult


# socket.getaddrinfo returns the canonical 5-tuple
# (family, type, proto, canonname, sockaddr); only the IP inside
# sockaddr matters here. ``AddrInfo`` names the shape once so the
# tests can stop repeating it.
type AddrInfo = tuple[int, int, int, str, tuple[str, int]]


def _addrinfo(ip: str) -> list[AddrInfo]:
    """Build a ``socket.getaddrinfo``-shaped result for a single IP."""
    return [(0, 0, 0, "", (ip, 0))]


def test_webfetch_metadata() -> None:
    t = WebFetch()
    assert t.name == "WebFetch"
    assert t.tool_id == "application/x-tool-webfetch"


def test_summary_short_url() -> None:
    t = WebFetch()
    assert t.summary({"url": "https://example.com"}) == "WebFetch https://example.com"


def test_summary_long_url_truncates() -> None:
    t = WebFetch()
    long_url = "https://example.com/" + ("x" * 100)
    out = t.summary({"url": long_url})
    assert out.endswith("...")
    assert len(out) <= 100


def test_summary_result_none() -> None:
    assert WebFetch().summary_result(ToolResult(call_id="", content="x")) is None


def test_prompt_empty() -> None:
    assert WebFetch().prompt() == ""


def test_url_is_safe_bad_scheme() -> None:
    err = _url_is_safe("ftp://example.com")
    assert err is not None
    assert "Unsupported scheme" in err


def test_url_is_safe_no_host() -> None:
    err = _url_is_safe("https:///abc")
    assert err is not None
    assert "no host" in err


def test_url_is_safe_dns_failure() -> None:
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")):
        err = _url_is_safe("https://does-not-exist.invalid")
    assert err is not None
    assert "DNS" in err


def test_url_is_safe_localhost_rejected() -> None:
    with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
        err = _url_is_safe("https://localhost")
    assert err is not None
    assert "non-public" in err


def test_url_is_safe_public_passes() -> None:
    with patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")):
        err = _url_is_safe("https://example.com")
    assert err is None


def test_validated_host_rejects_rebound_private_ip() -> None:
    with (
        patch(
            "socket.getaddrinfo",
            side_effect=[_addrinfo("8.8.8.8"), _addrinfo("127.0.0.1")],
        ),
        pytest.raises(ValueError, match="non-public"),
    ):
        _validated_host("example.com")


def test_reddit_adapter_matches_canonical() -> None:
    assert _RedditAdapter().matches("https://reddit.com/r/x") is True


def test_reddit_adapter_matches_subdomain() -> None:
    adapter = _RedditAdapter()
    assert adapter.matches("https://www.reddit.com/r/x") is True
    assert adapter.matches("https://old.reddit.com/r/x") is True


def test_reddit_adapter_rejects_non_reddit() -> None:
    assert _RedditAdapter().matches("https://example.com") is False


def test_match_http_fetch_simple_curl() -> None:
    assert _match_http_fetch("curl", ("https://example.com",)) is not None


def test_match_http_fetch_simple_wget() -> None:
    assert _match_http_fetch("wget", ("https://example.com",)) is not None


def test_match_http_fetch_no_url_returns_none() -> None:
    assert _match_http_fetch("curl", ("-v",)) is None


def test_match_http_fetch_two_urls_returns_none() -> None:
    assert _match_http_fetch("curl", ("https://a", "https://b")) is None


def test_match_http_fetch_output_flag_bails() -> None:
    assert _match_http_fetch("curl", ("-o", "f.txt", "https://x")) is None


def test_bash_match_simple_curl() -> None:
    trees = parse_bash("curl https://example.com")
    assert trees is not None
    assert WebFetch().bash_match(trees) is not None


def test_bash_match_with_cd_prefix() -> None:
    trees = parse_bash("cd /tmp && curl https://example.com")
    assert trees is not None
    assert WebFetch().bash_match(trees) is not None


def test_bash_match_with_env_prefix_rejected() -> None:
    trees = parse_bash("FOO=1 curl https://example.com")
    assert trees is not None
    # env_prefix kills the match.
    assert WebFetch().bash_match(trees) is None


def test_bash_match_non_curl_command_rejected() -> None:
    trees = parse_bash("ls -la")
    assert trees is not None
    assert WebFetch().bash_match(trees) is None


def test_bash_match_pipeline_rejected() -> None:
    trees = parse_bash("curl https://x | grep y")
    assert trees is not None
    # Unwrap fails; bash_match returns None.
    assert WebFetch().bash_match(trees) is None


def test_run_rejects_bad_method() -> None:
    result = asyncio.run(
        WebFetch().run({"url": "https://x", "method": "PUT"}),
    )
    assert result.is_error
    assert "Unsupported method" in result.content


def test_run_post_rejects_both_json_and_form() -> None:
    result = asyncio.run(
        WebFetch().run(
            {
                "url": "https://x",
                "method": "POST",
                "json": {"a": 1},
                "form": {"b": "2"},
            }
        ),
    )
    assert result.is_error
    assert "mutually exclusive" in result.content


def test_run_fetch_error_returns_tool_result_error() -> None:
    with patch(
        "sagent.tools.web_fetch._fetch_body",
        side_effect=ValueError("bad url"),
    ):
        result = asyncio.run(WebFetch().run({"url": "https://x"}))
    assert result.is_error
    assert "Fetch failed" in result.content


def test_run_json_response_skips_extraction() -> None:
    """JSON-looking content is returned as-is (no trafilatura involvement)."""
    body = b'{"hello": "world"}'

    async def fake_fetch_body(
        raw_url: str,
        *,
        method: str,
        json_body: object,
        form_body: object,
    ) -> tuple[bytes, str]:
        del raw_url, method, json_body, form_body
        return body, _KIND_HTML

    with patch(
        "sagent.tools.web_fetch._fetch_body",
        side_effect=fake_fetch_body,
    ):
        result = asyncio.run(WebFetch().run({"url": "https://api/json"}))
    assert '"hello"' in result.content


def test_run_cache_hit_skips_second_fetch() -> None:
    html = b"<html><body>cached body</body></html>"

    async def fake_fetch_body(
        raw_url: str,
        *,
        method: str,
        json_body: object,
        form_body: object,
    ) -> tuple[bytes, str]:
        del raw_url, method, json_body, form_body
        return html, _KIND_HTML

    tool = WebFetch()
    with (
        patch(
            "sagent.tools.web_fetch._fetch_body",
            side_effect=fake_fetch_body,
        ) as mock_body,
        patch(
            "sagent.tools.web_fetch.trafilatura.extract",
            return_value="extracted",
        ),
    ):
        _ = asyncio.run(tool.run({"url": "https://example.com"}))
        _ = asyncio.run(tool.run({"url": "https://example.com"}))
    assert mock_body.call_count == 1


def test_run_post_json_passes_through() -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_body(
        raw_url: str,
        *,
        method: str,
        json_body: object,
        form_body: object,
    ) -> tuple[bytes, str]:
        captured["method"] = method
        captured["json_body"] = json_body
        captured["form_body"] = form_body
        del raw_url
        return b'{"ok": 1}', _KIND_HTML

    with patch(
        "sagent.tools.web_fetch._fetch_body",
        side_effect=fake_fetch_body,
    ):
        result = asyncio.run(
            WebFetch().run(
                {
                    "url": "https://api/x",
                    "method": "POST",
                    "json": {"q": "test"},
                }
            ),
        )
    assert captured["method"] == "POST"
    assert captured["json_body"] == {"q": "test"}
    assert "ok" in result.content


def test_run_post_form_passes_through() -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_body(
        raw_url: str,
        *,
        method: str,
        json_body: object,
        form_body: object,
    ) -> tuple[bytes, str]:
        captured["form_body"] = form_body
        del raw_url, method, json_body
        return b"ok", _KIND_HTML

    with patch(
        "sagent.tools.web_fetch._fetch_body",
        side_effect=fake_fetch_body,
    ):
        _ = asyncio.run(
            WebFetch().run(
                {
                    "url": "https://api/x",
                    "method": "POST",
                    "form": {"a": "b"},
                }
            ),
        )
    form_body = captured["form_body"]
    assert isinstance(form_body, dict)
    body = cast(dict[str, str], form_body)
    assert body["a"] == "b"


def test_run_handles_reddit_thread_format() -> None:
    """Reddit thread JSON triggers _format_reddit_json."""
    json_bytes = (
        b'[{"kind":"Listing","data":{"children":[{"data":{"title":"T","author":"u",'
        b'"score":10,"selftext":"body"}}]}},'
        b'{"kind":"Listing","data":{"children":['
        b'{"kind":"t1","data":{"author":"u2","score":5,"body":"hi"}}'
        b"]}}]"
    )

    async def fake_fetch_body(
        raw_url: str,
        *,
        method: str,
        json_body: object,
        form_body: object,
    ) -> tuple[bytes, str]:
        del raw_url, method, json_body, form_body
        return json_bytes, _KIND_REDDIT

    with patch(
        "sagent.tools.web_fetch._fetch_body",
        side_effect=fake_fetch_body,
    ):
        result = asyncio.run(
            WebFetch().run({"url": "https://reddit.com/r/foo/comments/abc"}),
        )
    assert "# T" in result.content
    assert "u/u2" in result.content


def test_run_fetch_error_oserror() -> None:
    err = FetchError(url="https://x", status=500, headers={}, body=b"boom")
    with patch("sagent.tools.web_fetch._fetch_body", side_effect=err):
        result = asyncio.run(WebFetch().run({"url": "https://x"}))
    assert result.is_error


def test_fetch_body_non_reddit_path() -> None:
    """Non-Reddit URLs take the simple ``_safe_fetch`` path."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        return_value=b"hello",
    ):
        body, kind = asyncio.run(
            _fetch_body(
                "https://example.com",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    assert body == b"hello"
    assert kind == _KIND_HTML


def test_fetch_body_reddit_thread_takes_json_path() -> None:
    """Reddit comments URL is rewritten to a ``.json`` endpoint."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b'[{"kind":"Listing","data":{"children":[]}}]'

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        body, kind = asyncio.run(
            _fetch_body(
                "https://reddit.com/r/foo/comments/abc",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    url = captured["url"]
    assert isinstance(url, str)
    assert url.endswith(".json")
    assert "www.reddit.com" in url
    assert kind == _KIND_REDDIT
    assert body.startswith(b"[")


def test_fetch_body_reddit_root_no_thread_pattern() -> None:
    """Reddit non-thread URL: simple GET, not JSON-mode."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        return_value=b"<html>regular page</html>",
    ):
        body, kind = asyncio.run(
            _fetch_body(
                "https://reddit.com/r/foo",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    assert kind == _KIND_HTML
    assert b"regular" in body


def test_fetch_body_reddit_listing_json_path() -> None:
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b'{"kind":"Listing","data":{"children":[]}}'

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        body, kind = asyncio.run(
            _fetch_body(
                "https://www.reddit.com/r/foo/new/.json?limit=25",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    assert kind == _KIND_REDDIT_LISTING
    assert body.startswith(b"{")
    assert captured["url"] == "https://www.reddit.com/r/foo/new/.json?limit=25"


def test_extract_text_reddit_listing_formats_posts() -> None:
    from sagent.tools.web_fetch import _extract_text  # noqa: PLC0415

    payload = json.dumps(
        {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "title": "First post",
                            "author": "alice",
                            "score": 12,
                            "num_comments": 3,
                            "created_utc": 1_779_731_536,
                            "permalink": "/r/foo/comments/abc/first/",
                            "url": "https://example.com/a",
                            "link_flair_text": "Discussion",
                            "selftext": "Body text for the listing item.",
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "title": "Second post",
                            "author": "bob",
                            "score": 1,
                            "num_comments": 0,
                            "created_utc": 1_779_731_600,
                            "permalink": "/r/foo/comments/def/second/",
                            "url": "https://www.reddit.com/r/foo/comments/def/second/",
                            "selftext": "",
                        },
                    },
                ]
            },
        }
    ).encode()
    out = asyncio.run(_extract_text(payload, kind=_KIND_REDDIT_LISTING, method="GET"))
    assert "# Reddit listing" in out
    assert "First post" in out
    assert "u/alice" in out
    assert "2026-05-25T17:52:16+00:00" in out
    assert "https://www.reddit.com/r/foo/comments/abc/first/" in out
    assert "Body text for the listing item." in out
    assert "Second post" in out
    assert '"kind"' not in out


def test_extract_text_html_path_uses_trafilatura() -> None:
    """``_extract_text`` for HTML goes through trafilatura.extract."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _extract_text,
    )

    with patch(
        "sagent.tools.web_fetch.trafilatura.extract",
        return_value="cleaned text",
    ):
        out = asyncio.run(
            _extract_text(
                b"<html><body><p>Hi</p></body></html>",
                kind=_KIND_HTML,
                method="GET",
            ),
        )
    assert out == "cleaned text"


def test_extract_text_html_fallback_when_extract_none() -> None:
    """When trafilatura returns nothing, the raw decoded content is returned."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _extract_text,
    )

    with patch(
        "sagent.tools.web_fetch.trafilatura.extract",
        return_value=None,
    ):
        out = asyncio.run(
            _extract_text(
                b"<html><body>raw fallback</body></html>",
                kind=_KIND_HTML,
                method="GET",
            ),
        )
    assert "raw fallback" in out


def test_extract_text_reddit_invalid_json_returns_truncated_content() -> None:
    """Reddit JSON parse failure falls back to raw content slice."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _extract_text,
    )

    out = asyncio.run(
        _extract_text(b"not json at all", kind=_KIND_REDDIT, method="GET"),
    )
    assert out == "not json at all"


def test_format_reddit_comments_handles_nested_replies() -> None:
    """Nested ``replies`` recursion produces an indented child line."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _format_reddit_json,
    )

    payload: list[object] = [
        {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Top",
                            "author": "u1",
                            "score": 5,
                            "selftext": "",
                        }
                    }
                ]
            },
        },
        {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "author": "p",
                            "score": 1,
                            "body": "parent\nline2",
                            "replies": {
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "author": "c",
                                                "score": 2,
                                                "body": "child",
                                            },
                                        }
                                    ]
                                }
                            },
                        },
                    }
                ]
            },
        },
    ]
    out = _format_reddit_json(payload)
    assert "**u/p**" in out
    assert "**u/c**" in out


def test_extract_text_markdown_kind_returns_as_is() -> None:
    """Markdown kind (reader-proxy output) skips trafilatura."""
    from sagent.tools.web_fetch import _extract_text  # noqa: PLC0415

    md = b"# Title\n\nReader proxy returned this verbatim.\n"
    # If trafilatura ran, it would strip the markdown structure; we use
    # a sentinel return value to assert it stayed untouched.
    with patch(
        "sagent.tools.web_fetch.trafilatura.extract",
        return_value="WRONG",
    ):
        out = asyncio.run(_extract_text(md, kind=_KIND_MARKDOWN, method="GET"))
    assert "# Title" in out
    assert "WRONG" not in out


def test_fetch_with_fallback_passthrough_on_success() -> None:
    """Initial rung success returns ``(body, _KIND_HTML)`` with no fallback."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    proxy = MagicMock(return_value=b"PROXY")
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            return_value=b"OK",
        ),
        patch(
            "sagent.tools.web_fetch._reader_proxy_fetch",
            proxy,
        ),
    ):
        body, kind = _fetch_with_fallback(
            "https://example.com",
            method="GET",
            json_body=None,
            form_body=None,
        )
    assert body == b"OK"
    assert kind == _KIND_HTML
    proxy.assert_not_called()


def test_fetch_with_fallback_403_falls_to_reader_proxy() -> None:
    """A 403 GET routes through ``_reader_proxy_fetch`` with kind=markdown."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    err = FetchError(url="https://x", status=403, headers={}, body=b"blocked")
    proxy = MagicMock(return_value=b"# extracted")
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            side_effect=err,
        ),
        patch(
            "sagent.tools.web_fetch._reader_proxy_fetch",
            proxy,
        ),
    ):
        body, kind = _fetch_with_fallback(
            "https://x",
            method="GET",
            json_body=None,
            form_body=None,
        )
    assert body == b"# extracted"
    assert kind == _KIND_MARKDOWN
    proxy.assert_called_once_with("https://x")


def test_fetch_with_fallback_429_and_503_also_trigger_fallback() -> None:
    """The bot-wall set is {403, 429, 503} — all three engage the ladder."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    for status in (429, 503):
        err = FetchError(url="https://x", status=status, headers={}, body=b"")
        proxy = MagicMock(return_value=b"# md")
        with (
            patch(
                "sagent.tools.web_fetch._safe_fetch",
                side_effect=err,
            ),
            patch(
                "sagent.tools.web_fetch._reader_proxy_fetch",
                proxy,
            ),
        ):
            body, kind = _fetch_with_fallback(
                "https://x",
                method="GET",
                json_body=None,
                form_body=None,
            )
        assert body == b"# md", f"status {status}: proxy body not returned"
        assert kind == _KIND_MARKDOWN, f"status {status}: kind not markdown"


def test_fetch_with_fallback_404_does_not_engage_ladder() -> None:
    """Non-bot-wall errors (404) surface immediately, no fallback."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    err = FetchError(url="https://x", status=404, headers={}, body=b"")
    proxy = MagicMock()
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            side_effect=err,
        ),
        patch(
            "sagent.tools.web_fetch._reader_proxy_fetch",
            proxy,
        ),
        pytest.raises(FetchError) as exc_info,
    ):
        _fetch_with_fallback(
            "https://x",
            method="GET",
            json_body=None,
            form_body=None,
        )
    assert exc_info.value.status == 404
    proxy.assert_not_called()


def test_fetch_with_fallback_post_403_does_not_engage_ladder() -> None:
    """POST 403 surfaces immediately — fallback is GET-only."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    err = FetchError(url="https://x", status=403, headers={}, body=b"")
    proxy = MagicMock()
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            side_effect=err,
        ),
        patch(
            "sagent.tools.web_fetch._reader_proxy_fetch",
            proxy,
        ),
        pytest.raises(FetchError) as exc_info,
    ):
        _fetch_with_fallback(
            "https://x",
            method="POST",
            json_body=None,
            form_body={"a": "1"},
        )
    assert exc_info.value.status == 403
    proxy.assert_not_called()


def test_fetch_with_fallback_all_rungs_fail_raises_original_error() -> None:
    """When proxy also fails, the original (rung-1) FetchError is raised."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    orig = FetchError(url="https://x", status=403, headers={}, body=b"orig")
    proxy_err = FetchError(
        url="https://r.jina.ai/...", status=500, headers={}, body=b""
    )
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            side_effect=orig,
        ),
        patch(
            "sagent.tools.web_fetch._reader_proxy_fetch",
            side_effect=proxy_err,
        ),
        pytest.raises(FetchError) as exc_info,
    ):
        _fetch_with_fallback(
            "https://x",
            method="GET",
            json_body=None,
            form_body=None,
        )
    # The rung-1 error is what surfaces; proxy error is chained as ``__cause__``.
    assert exc_info.value.status == 403
    assert exc_info.value.body == b"orig"
    assert isinstance(exc_info.value.__cause__, FetchError)
    assert exc_info.value.__cause__.status == 500


def test_reader_proxy_fetch_raises_on_soft_failure_sentinel() -> None:
    """Jina returns 200 with a Warning: line when its backend got 403'd.

    The proxy must detect that sentinel and raise FetchError so the
    ladder treats it as a fall-through, not as a successful fetch.
    """
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _reader_proxy_fetch,
    )

    soft_fail = (
        b"Title: example.org\n\n"
        b"Warning: Target URL returned error 403: Forbidden\n\n"
        b"Markdown Content:\n\n"
    )
    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            return_value=soft_fail,
        ),
        pytest.raises(FetchError) as exc_info,
    ):
        _reader_proxy_fetch("https://www.example.org/article")
    # 502 is the synthetic status used to signal proxy-level failure.
    assert exc_info.value.status == 502


def test_fetch_with_fallback_jina_soft_failure_surfaces_rung1_error() -> None:
    """Soft-failed proxy + 403 rung-1 surfaces the original rung-1 error."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _fetch_with_fallback,
    )

    orig = FetchError(url="https://x", status=403, headers={}, body=b"akamai")
    soft_fail = b"Warning: Target URL returned error 403\n"

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        if "r.jina.ai" in url:
            return soft_fail
        raise orig

    with (
        patch(
            "sagent.tools.web_fetch._safe_fetch",
            side_effect=fake_safe_fetch,
        ),
        pytest.raises(FetchError) as exc_info,
    ):
        _fetch_with_fallback(
            "https://x",
            method="GET",
            json_body=None,
            form_body=None,
        )
    assert exc_info.value.status == 403
    assert exc_info.value.body == b"akamai"


def test_reader_proxy_fetch_uses_jina_template_and_url_encodes() -> None:
    """Reader proxy targets r.jina.ai with the user URL as path data."""
    from sagent.tools.web_fetch import (  # noqa: PLC0415
        _reader_proxy_fetch,
    )

    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b"# markdown"

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        body = _reader_proxy_fetch(
            "https://www.example.org/2026/05/article?x=1&y=2#frag"
        )
    assert body == b"# markdown"
    url = captured["url"]
    assert isinstance(url, str)
    parsed_proxy = urlparse(url)
    assert parsed_proxy.scheme == "https"
    assert parsed_proxy.netloc == "r.jina.ai"
    encoded = parsed_proxy.path.removeprefix("/")
    target = urlparse(unquote(encoded))
    assert target.scheme == "https"
    assert target.netloc == "www.example.org"
    assert target.path == "/2026/05/article"
    assert target.query == "x=1&y=2"
    assert target.fragment == "frag"
    assert "%3F" in encoded
    assert "%26" in encoded
    assert "%23" in encoded
    assert "?" not in encoded
    assert "#" not in encoded


# Host adapter dispatch & per-host adapters.


def test_adapter_registry_contains_all_three_hosts() -> None:
    """Registry holds Reddit, Google News, and X adapters in that order."""
    assert len(_ADAPTERS) == 3
    assert isinstance(_ADAPTERS[0], _RedditAdapter)
    assert isinstance(_ADAPTERS[1], _GoogleNewsAdapter)
    assert isinstance(_ADAPTERS[2], _XAdapter)


def test_fetch_body_unmatched_url_uses_generic_ladder() -> None:
    """A URL with no matching adapter falls through to ``_fetch_with_fallback``."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    with patch(
        "sagent.tools.web_fetch._fetch_with_fallback",
        return_value=(b"generic", _KIND_HTML),
    ) as mock_ladder:
        body, kind = asyncio.run(
            _fetch_body(
                "https://example.com",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    mock_ladder.assert_called_once()
    assert body == b"generic"
    assert kind == _KIND_HTML


def test_fetch_body_post_bypasses_adapters() -> None:
    """POST requests skip adapter dispatch even on matching hosts."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    with patch(
        "sagent.tools.web_fetch._fetch_with_fallback",
        return_value=(b"posted", _KIND_HTML),
    ) as mock_ladder:
        asyncio.run(
            _fetch_body(
                "https://reddit.com/api/v1/x",
                method="POST",
                json_body={"a": 1},
                form_body=None,
            ),
        )
    mock_ladder.assert_called_once()


# Google News adapter.


def test_google_news_matches_only_exact_host() -> None:
    adapter = _GoogleNewsAdapter()
    assert adapter.matches("https://news.google.com/") is True
    assert adapter.matches("https://news.google.com/topstories") is True
    assert adapter.matches("https://google.com/news") is False
    assert adapter.matches("https://www.news.google.com/") is False


@pytest.mark.parametrize(
    ("input_url", "expected_path"),
    [
        ("https://news.google.com/", "/rss"),
        ("https://news.google.com", "/rss"),
        ("https://news.google.com/home", "/rss"),
        ("https://news.google.com/topstories", "/rss"),
        ("https://news.google.com/foryou", "/rss"),
        ("https://news.google.com/topstories/", "/rss"),
        ("https://news.google.com/search?q=foo", "/rss/search"),
    ],
)
def test_google_news_rewrites_front_page_paths_to_rss(
    input_url: str, expected_path: str
) -> None:
    """Front-page and search paths route to their RSS counterparts."""
    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b"<?xml version='1.0'?><rss><channel></channel></rss>"

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        body, kind = asyncio.run(_GoogleNewsAdapter().fetch(input_url))
    fetched = captured["url"]
    assert isinstance(fetched, str)
    from urllib.parse import urlparse  # noqa: PLC0415

    assert urlparse(fetched).path == expected_path
    assert kind == _KIND_RSS
    assert body.startswith(b"<?xml")


def test_google_news_preserves_query_string_on_rewrite() -> None:
    """Locale parameters (``hl``, ``gl``, ``ceid``) survive the rewrite."""
    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b"<?xml version='1.0'?><rss><channel></channel></rss>"

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        asyncio.run(
            _GoogleNewsAdapter().fetch(
                "https://news.google.com/topstories?hl=en-US&gl=US&ceid=US:en",
            ),
        )
    fetched = captured["url"]
    assert isinstance(fetched, str)
    assert "hl=en-US" in fetched
    assert "gl=US" in fetched


def test_google_news_already_rss_path_passes_through() -> None:
    """A URL already on ``/rss/...`` is fetched unchanged."""
    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b"<?xml version='1.0'?><rss><channel></channel></rss>"

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        _, kind = asyncio.run(
            _GoogleNewsAdapter().fetch("https://news.google.com/rss/search?q=foo"),
        )
    assert captured["url"] == "https://news.google.com/rss/search?q=foo"
    assert kind == _KIND_RSS


def test_google_news_article_url_not_rewritten() -> None:
    """Article-detail URLs fall through as HTML (no RSS equivalent)."""
    captured: dict[str, object] = {}

    def fake_safe_fetch(url: str, **_kw: object) -> bytes:
        captured["url"] = url
        return b"<html>article body</html>"

    article = "https://news.google.com/articles/CAIiE..."
    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        side_effect=fake_safe_fetch,
    ):
        body, kind = asyncio.run(_GoogleNewsAdapter().fetch(article))
    assert captured["url"] == article
    assert kind == _KIND_HTML
    assert b"article" in body


# RSS formatter.


def test_format_rss_renders_title_and_items() -> None:
    """Each ``<item>`` becomes a heading with link and meta line."""
    feed = (
        b"<?xml version='1.0'?>"
        b"<rss><channel>"
        b"<title>Top stories</title>"
        b"<item>"
        b"<title>Headline one</title>"
        b"<link>https://example.com/a</link>"
        b"<pubDate>Fri, 22 May 2026 00:00:00 GMT</pubDate>"
        b"<source url='https://nyt.example'>NYT</source>"
        b"</item>"
        b"</channel></rss>"
    )
    out = _format_rss(feed)
    assert "# Top stories" in out
    assert "## Headline one" in out
    assert "https://example.com/a" in out
    assert "NYT" in out
    assert "Fri, 22 May 2026" in out


def test_format_rss_expands_google_news_cluster() -> None:
    """Sibling stories embedded in a description's ``<ol>`` become bullets."""
    cluster = (
        "<ol>"
        '<li><a href="https://lead.example">Lead headline</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">NYT</font></li>'
        '<li><a href="https://sib.example/1">Sibling one</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">CNN</font></li>'
        '<li><a href="https://sib.example/2">Sibling two</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">BBC</font></li>'
        "</ol>"
    )
    feed = (
        "<?xml version='1.0'?>"
        "<rss><channel>"
        "<title>Top stories</title>"
        "<item>"
        "<title>Lead headline</title>"
        "<link>https://lead.example</link>"
        f"<description><![CDATA[{cluster}]]></description>"
        "</item>"
        "</channel></rss>"
    ).encode()
    out = _format_rss(feed)
    # Lead is shown once at the top; siblings as bullets.
    assert out.count("Lead headline") == 1
    assert "- [Sibling one](https://sib.example/1) -- CNN" in out
    assert "- [Sibling two](https://sib.example/2) -- BBC" in out


def test_format_rss_invalid_xml_returns_raw_decoded() -> None:
    """Malformed XML degrades gracefully to a decoded byte slice."""
    out = _format_rss(b"not xml at all")
    assert "not xml" in out


def test_format_rss_rejects_entity_expansion_payload() -> None:
    """Defusedxml blocks billion-laughs / XXE payloads at parse time."""
    payload = (
        b"<?xml version='1.0'?>"
        b"<!DOCTYPE lolz ["
        b"<!ENTITY lol 'lol'>"
        b"<!ENTITY lol2 '&lol;&lol;&lol;&lol;&lol;'>"
        b"]>"
        b"<rss><channel><title>&lol2;</title></channel></rss>"
    )
    out = _format_rss(payload)
    # The parse fails (defusedxml refuses entity definitions); we
    # fall through to the decoded-bytes branch. The "lol" entity
    # text must not appear expanded.
    assert "lollollol" not in out


def test_format_rss_empty_channel_no_items() -> None:
    """A feed with no items still emits the title heading."""
    out = _format_rss(
        b"<?xml version='1.0'?><rss><channel><title>Empty feed</title></channel></rss>",
    )
    assert "# Empty feed" in out


def test_parse_rss_cluster_unescapes_entities() -> None:
    """HTML entities in titles and sources are decoded."""
    fragment = (
        '<li><a href="https://example.com/x">It&#8217;s here</a>'
        "&nbsp;&nbsp;<font>The &amp; Co.</font></li>"
    )
    entries = _parse_rss_cluster(fragment)
    assert len(entries) == 1
    title, link, source = entries[0]
    assert title == "It\u2019s here"
    assert link == "https://example.com/x"
    assert source == "The & Co."


def test_parse_rss_cluster_optional_source() -> None:
    """Missing ``<font>`` source yields an empty string, not a crash."""
    fragment = '<li><a href="https://example.com/y">Bare title</a></li>'
    entries = _parse_rss_cluster(fragment)
    assert entries == [("Bare title", "https://example.com/y", "")]


# X (Twitter) adapter.


def test_x_adapter_matches_x_and_twitter_hosts() -> None:
    adapter = _XAdapter()
    assert adapter.matches("https://x.com/user/status/123") is True
    assert adapter.matches("https://twitter.com/user/status/123") is True
    assert adapter.matches("https://mobile.twitter.com/user") is True
    assert adapter.matches("https://example.com/x.com") is False


def test_x_adapter_routes_through_reader_proxy() -> None:
    """X fetches always flow through the Jina reader proxy as markdown."""
    with patch(
        "sagent.tools.web_fetch._reader_proxy_fetch",
        return_value=b"# tweet content",
    ) as mock_proxy:
        body, kind = asyncio.run(
            _XAdapter().fetch("https://x.com/user/status/123"),
        )
    mock_proxy.assert_called_once_with("https://x.com/user/status/123")
    assert body == b"# tweet content"
    assert kind == _KIND_MARKDOWN


# Integration: dispatch through WebFetch.run end-to-end.


def test_run_google_news_routes_to_rss_extraction() -> None:
    """End-to-end: news.google.com home URL → RSS feed → markdown output."""
    feed = (
        b"<?xml version='1.0'?>"
        b"<rss><channel>"
        b"<title>Top stories - Google News</title>"
        b"<item><title>Lead</title><link>https://x</link></item>"
        b"</channel></rss>"
    )
    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        return_value=feed,
    ):
        result = asyncio.run(WebFetch().run({"url": "https://news.google.com/"}))
    assert "# Top stories - Google News" in result.content
    assert "## Lead" in result.content


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
