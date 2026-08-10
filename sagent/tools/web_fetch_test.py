"""Tests for ``tools.web_fetch``: URL fetcher tool, SSRF guard, cache."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import asyncio
import socket

from wesearch.errors import CloudflareChallengeError, FetchError
from wesearch.fetch import Policy
from wesearch.web import WebFetchResult

import pytest

from sagent.tools.lib.bash import parse_bash
from sagent.tools.web_fetch import (
    WebFetch,
    _match_http_fetch,
    _request_bodies,
)


# socket.getaddrinfo returns the canonical 5-tuple
# (family, type, proto, canonname, sockaddr); only the IP inside
# sockaddr matters here. ``AddrInfo`` names the shape once so the
# tests can stop repeating it.
type AddrInfo = tuple[int, int, int, str, tuple[str, int]]


def _addrinfo(ip: str) -> list[AddrInfo]:
    """Build a ``socket.getaddrinfo``-shaped result for a single IP."""
    fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(fam, 0, 0, "", (ip, 0))]


def _result(text: str, *, kind: str = "html") -> WebFetchResult:
    """A ``WebFetchResult`` stub for patching ``fetch_web`` in ``run`` tests."""
    return WebFetchResult(text=text, url="https://x", kind=kind, truncated=False)


def test_webfetch_metadata() -> None:
    t = WebFetch()
    assert t.name == "WebFetch"
    assert t.tool_id == "application/x-tool-webfetch"


def test_summary_short_url() -> None:
    t = WebFetch()
    assert t.summary({"url": "https://example.com"}) == "WebFetch https://example.com"


def test_summary_keeps_long_url() -> None:
    t = WebFetch()
    long_url = "https://example.com/" + ("x" * 100)
    assert t.summary({"url": long_url}) == f"WebFetch {long_url}"


def test_prompt_empty() -> None:
    assert WebFetch().prompt() == ""


def test_run_leaves_the_agent_url_untrusted() -> None:
    """``run`` states a transport and nothing else; SSRF safety is the default.

    The tool used to pass an SSRF resolver explicitly, which is what cost it
    every browser transport -- the library rejected that resolver on the browser
    legs. Holding no policy beyond the transport is the point: this wrapper is
    thin, and the safe behavior is wesearch's default.
    """
    captured: dict[str, object] = {}

    def fake_fetch_web(url: str, **kwargs: object) -> WebFetchResult:
        del url
        captured["policy"] = kwargs.get("policy")
        return _result("{}")

    with patch(
        "sagent.tools.web_fetch.fetch_web",
        side_effect=fake_fetch_web,
    ):
        result = asyncio.run(WebFetch().run({"url": "https://example.com"}))
    assert not result.is_error
    policy = captured["policy"]
    assert isinstance(policy, Policy)
    assert policy.trust == "untrusted"


def test_run_appends_truncation_notice_when_body_exceeds_limit() -> None:
    """An over-limit page is cut WITH the truncation notice, not silently.

    Drives the real ``fetch_web`` (only the network layer is mocked) so the
    regression is exercised end-to-end: ``fetch_web`` must not pre-cap the text
    such that sagent's ``truncate`` sees an at-limit string and appends nothing.
    """
    from wesearch import web  # noqa: PLC0415 -- patch the network below fetch_web.

    from sagent.tools.core import (  # noqa: PLC0415 -- test-local constant.
        TOOL_RESULT_MAX_CHARS,
    )

    over = TOOL_RESULT_MAX_CHARS + 500
    with (
        patch.object(
            web, "fetch_with_reader_fallback", return_value=(b"z" * over, False)
        ),
        patch.object(web, "_extract_text", return_value="z" * over),
        patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")),
    ):
        result = asyncio.run(WebFetch().run({"url": "https://example.com"}))
    assert not result.is_error
    assert "(truncated, 500 chars omitted)" in result.content


def test_match_http_fetch_simple_curl() -> None:
    assert _match_http_fetch(("https://example.com",)) is not None


def test_match_http_fetch_simple_wget() -> None:
    assert _match_http_fetch(("https://example.com",)) is not None


def test_match_http_fetch_no_url_returns_none() -> None:
    assert _match_http_fetch(("-v",)) is None


def test_match_http_fetch_two_urls_returns_none() -> None:
    assert _match_http_fetch(("https://a", "https://b")) is None


def test_match_http_fetch_output_flag_bails() -> None:
    assert _match_http_fetch(("-o", "f.txt", "https://x")) is None


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
        "sagent.tools.web_fetch.fetch_web",
        side_effect=ValueError("bad url"),
    ):
        result = asyncio.run(WebFetch().run({"url": "https://x"}))
    assert result.is_error
    assert "Fetch failed" in result.content


def test_run_explicit_transport_passes_through() -> None:
    captured: dict[str, object] = {}

    def fake_fetch_web(url: str, **kwargs: object) -> WebFetchResult:
        del url
        policy = kwargs.get("policy")
        captured["transport"] = getattr(policy, "transport", None)
        return _result("{}")

    with patch(
        "sagent.tools.web_fetch.fetch_web",
        side_effect=fake_fetch_web,
    ):
        result = asyncio.run(
            WebFetch().run({"url": "https://example.com", "transport": "zendriver"})
        )
    assert not result.is_error
    assert captured["transport"] == "zendriver"


def test_run_rejects_invalid_transport() -> None:
    result = asyncio.run(
        WebFetch().run({"url": "https://example.com", "transport": "requests"})
    )
    assert result.is_error
    assert "Invalid transport" in result.content


def test_run_returns_extracted_text() -> None:
    """The tool wraps ``fetch_web``'s text into the ToolResult content."""
    with patch(
        "sagent.tools.web_fetch.fetch_web",
        return_value=_result('{"hello": "world"}'),
    ):
        result = asyncio.run(WebFetch().run({"url": "https://api/json"}))
    assert '"hello"' in result.content


def test_run_cloudflare_challenge_yields_specific_guidance_not_bare_403() -> None:
    # fetch() classifies the block at the boundary and raises a
    # CloudflareChallengeError (is-a FetchError). The tool's ERROR path must
    # render its SPECIFIC actionable guidance, not the generic "Fetch failed:
    # HTTP 403" that the plain FetchError path produces.
    err = CloudflareChallengeError(
        url="https://x.com",
        status=403,
        headers={"server": "cloudflare", "cf-ray": "a1"},
        body=b"<title>Just a moment...</title>",
    )
    with patch("sagent.tools.web_fetch.fetch_web", side_effect=err):
        result = asyncio.run(WebFetch().run({"url": "https://x.com"}))
    assert result.is_error
    assert "cloudflare" in result.content.lower()
    assert CloudflareChallengeError.guidance in result.content
    assert "HTTP 403" not in result.content


def test_run_fetch_error_oserror() -> None:
    err = FetchError(url="https://x", status=500, headers={}, body=b"boom")
    with patch("sagent.tools.web_fetch.fetch_web", side_effect=err):
        result = asyncio.run(WebFetch().run({"url": "https://x"}))
    assert result.is_error


def test_run_cache_hit_skips_second_fetch() -> None:
    with patch(
        "sagent.tools.web_fetch.fetch_web",
        return_value=_result("extracted"),
    ) as mock_web:
        tool = WebFetch()
        _ = asyncio.run(tool.run({"url": "https://example.com"}))
        _ = asyncio.run(tool.run({"url": "https://example.com"}))
    assert mock_web.call_count == 1


def test_run_cache_separates_transports() -> None:
    with patch(
        "sagent.tools.web_fetch.fetch_web",
        return_value=_result("{}"),
    ) as mock_web:
        tool = WebFetch()
        _ = asyncio.run(tool.run({"url": "https://example.com"}))
        _ = asyncio.run(
            tool.run({"url": "https://example.com", "transport": "zendriver"})
        )
    assert mock_web.call_count == 2


def test_run_post_json_passes_through() -> None:
    captured: dict[str, object] = {}

    def fake_fetch_web(url: str, **kwargs: object) -> WebFetchResult:
        del url
        captured["method"] = kwargs.get("method")
        captured["json_body"] = kwargs.get("json_body")
        captured["form_body"] = kwargs.get("form_body")
        return _result('{"ok": 1}')

    with patch(
        "sagent.tools.web_fetch.fetch_web",
        side_effect=fake_fetch_web,
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

    def fake_fetch_web(url: str, **kwargs: object) -> WebFetchResult:
        del url
        captured["form_body"] = kwargs.get("form_body")
        return _result("ok")

    with patch(
        "sagent.tools.web_fetch.fetch_web",
        side_effect=fake_fetch_web,
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


@pytest.mark.parametrize("bad_form", [[], "stringly", 42])
def test_post_form_non_object_rejected(bad_form: object) -> None:
    """``form`` schema declares ``type: object``; reject non-mappings.

    Pre-fix, ``form=[]`` propagated to ``.items()`` and raised
    ``AttributeError`` outside the tool envelope (escaping ``run``'s
    ``except ValueError``). Reject at parse time so the directive
    error surfaces as a normal tool error.
    """
    with pytest.raises(ValueError, match="form"):
        _request_bodies("POST", {"form": bad_form})


@pytest.mark.parametrize("body_key", ["json", "form"])
def test_get_body_rejected(body_key: str) -> None:
    with pytest.raises(ValueError, match="POST"):
        _request_bodies("GET", {body_key: {"key": "value"}})


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
