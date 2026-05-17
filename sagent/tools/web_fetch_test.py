"""Tests for ``tools.web_fetch``: URL fetcher with SSRF guard + cache."""

from __future__ import annotations

from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, patch

import asyncio
import socket
import sys


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
    WebFetch,
    _is_reddit_url,
    _match_http_fetch,
    _url_is_safe,
)
from sagent.types.history import ToolResult


def test_webfetch_metadata() -> None:
    t = WebFetch()
    assert t.name == "WebFetch"
    assert t.tool_id == "application/x-tool-webfetch"
    assert t.supports_microcompaction is True


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
    with patch(
        "socket.getaddrinfo",
        return_value=[(0, 0, 0, "", ("127.0.0.1", 0))],
    ):
        err = _url_is_safe("https://localhost")
    assert err is not None
    assert "non-public" in err


def test_url_is_safe_public_passes() -> None:
    with patch(
        "socket.getaddrinfo",
        return_value=[(0, 0, 0, "", ("8.8.8.8", 0))],
    ):
        err = _url_is_safe("https://example.com")
    assert err is None


def test_is_reddit_url_canonical() -> None:
    assert _is_reddit_url("https://reddit.com/r/x") is True


def test_is_reddit_url_subdomain() -> None:
    assert _is_reddit_url("https://www.reddit.com/r/x") is True
    assert _is_reddit_url("https://old.reddit.com/r/x") is True


def test_is_reddit_url_non_reddit() -> None:
    assert _is_reddit_url("https://example.com") is False


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
    ) -> tuple[bytes, bool]:
        del raw_url, method, json_body, form_body
        return body, False

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
    ) -> tuple[bytes, bool]:
        del raw_url, method, json_body, form_body
        return html, False

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
    ) -> tuple[bytes, bool]:
        captured["method"] = method
        captured["json_body"] = json_body
        captured["form_body"] = form_body
        del raw_url
        return b'{"ok": 1}', False

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
    ) -> tuple[bytes, bool]:
        captured["form_body"] = form_body
        del raw_url, method, json_body
        return b"ok", False

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
    ) -> tuple[bytes, bool]:
        del raw_url, method, json_body, form_body
        return json_bytes, True

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
        body, is_reddit = asyncio.run(
            _fetch_body(
                "https://example.com",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    assert body == b"hello"
    assert is_reddit is False


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
        body, is_reddit = asyncio.run(
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
    assert is_reddit is True
    assert body.startswith(b"[")


def test_fetch_body_reddit_root_no_thread_pattern() -> None:
    """Reddit non-thread URL: simple GET, not JSON-mode."""
    from sagent.tools.web_fetch import _fetch_body  # noqa: PLC0415

    with patch(
        "sagent.tools.web_fetch._safe_fetch",
        return_value=b"<html>regular page</html>",
    ):
        body, is_reddit = asyncio.run(
            _fetch_body(
                "https://reddit.com/r/foo",
                method="GET",
                json_body=None,
                form_body=None,
            ),
        )
    assert is_reddit is False
    assert b"regular" in body


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
                reddit_thread=False,
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
                reddit_thread=False,
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
        _extract_text(b"not json at all", reddit_thread=True, method="GET"),
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
