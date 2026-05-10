"""WebFetch tool: fetch a URL (GET or POST) and extract its content."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from urllib.parse import urlparse

import asyncio
import ipaddress
import json
import re
import socket

import cachetools

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, JSONValue, json_freeze, json_unfreeze
from sagent.lib.lazy_import import lazy_import
from sagent.lib.message import get_directive
from sagent.lib.web.fetch import FetchError, ValidatedHost, fetch
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix


trafilatura = lazy_import("trafilatura")


class WebFetch:
    """Fetch a web page and extract its main content as clean text."""

    name: str = "WebFetch"
    tool_id: str = "application/x-tool-webfetch"
    description: str = load_tool_description("WebFetch")
    supports_microcompaction: bool = True
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": (
                        "HTTP method. Defaults to GET. Use POST to call"
                        " JSON or form APIs."
                    ),
                },
                "json": {
                    "description": (
                        "JSON-serializable body for POST requests. Sets"
                        " Content-Type: application/json. Mutually"
                        " exclusive with 'form'."
                    ),
                },
                "form": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "Form fields for POST requests. Sets Content-Type:"
                        " application/x-www-form-urlencoded. Mutually"
                        " exclusive with 'json'."
                    ),
                },
            },
            "required": ["url"],
        }
    )

    def __init__(self) -> None:
        self._cache = cachetools.TTLCache[str, str](maxsize=128, ttl=15 * 60)

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``curl URL`` / ``wget URL``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Bails on output redirection (``-o``/``-O``/``--output``),
        ``--data-binary @file`` style file uploads, and any non-http(s)
        URL -- those are cases WebFetch can't cleanly replace. Pipelines
        and stdout redirects are already filtered by ``unwrap_cd_prefix``.

        Args:
          trees: Parsed shell AST nodes.

        Returns:
          nudge: Suggested WebFetch invocation, or ``None`` if no match.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        _cwd, cmd = unwrapped
        if cmd.exe not in {"curl", "wget"} or cmd.env_prefix:
            return None
        return _match_http_fetch(cmd.exe, cmd.args)

    def summary(self, msg: Message) -> str:
        """Return a short display label for this invocation.

        Args:
          msg: Directive message.

        Returns:
          label: Human-readable summary string.

        """
        directive = get_directive(msg)
        url = str(directive.get("url", ""))
        if len(url) > 60:
            url = url[:57] + "..."
        return f"WebFetch {url}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Fetch the URL, extract main content, and return as text.

        Args:
          msg: Directive message containing the target URL and optional
            ``method``/``json``/``form`` for POST requests.

        Returns:
          result: Extracted page text or an error message.

        """
        directive = get_directive(msg)
        raw_url = str(directive.get("url", ""))
        method = str(directive.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            return TextMessage(
                f"Unsupported method {method!r}; only GET and POST allowed.",
                "text/x-error",
            )

        try:
            json_body, form_body = _request_bodies(method, directive)
        except ValueError as e:
            return TextMessage(str(e), "text/x-error")

        cache_key = raw_url if method == "GET" else None
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return TextMessage(cached, "text/plain")

        try:
            body, reddit_thread = await _fetch_body(
                raw_url,
                method=method,
                json_body=json_body,
                form_body=form_body,
            )
        except (FetchError, ValueError, OSError) as e:
            return TextMessage(f"Fetch failed: {e}", "text/x-error")

        text = await _extract_text(
            body,
            reddit_thread=reddit_thread,
            method=method,
        )
        truncated = truncate(text, TOOL_RESULT_MAX_CHARS)
        if cache_key is not None:
            self._cache[cache_key] = truncated
        return TextMessage(truncated, "text/plain")


def _request_bodies(
    method: str,
    directive: JSON,
) -> tuple[JSONValue, dict[str, str] | None]:
    """Return POST request bodies from a tool directive."""
    if method != "POST":
        return None, None
    raw_json = directive.get("json")
    raw_form = directive.get("form")
    if raw_json is not None and raw_form is not None:
        raise ValueError("'json' and 'form' are mutually exclusive.")
    if raw_json is not None:
        return json_unfreeze(raw_json), None
    if raw_form is None:
        return None, None
    return None, {
        str(k): str(v) for k, v in cast(dict[str, Any], json_unfreeze(raw_form)).items()
    }


async def _extract_text(body: bytes, *, reddit_thread: bool, method: str) -> str:
    """Extract tool result text from a response body."""
    content = body.decode("utf-8", errors="replace")
    if reddit_thread:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content[:TOOL_RESULT_MAX_CHARS]
        return _format_reddit_json(data)
    if method == "POST" or content.lstrip().startswith(("{", "[")):
        return content[:TOOL_RESULT_MAX_CHARS]
    extracted = await asyncio.to_thread(
        trafilatura.extract,
        content,
        include_links=True,
        include_tables=True,
    )
    return extracted or content[:TOOL_RESULT_MAX_CHARS]


async def _fetch_body(
    raw_url: str,
    *,
    method: str,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
) -> tuple[bytes, bool]:
    """Fetch a URL and identify Reddit thread JSON responses.

    Returns:
      body: Raw response bytes.
      is_reddit_thread_json: Whether ``body`` came from Reddit's thread JSON
        endpoint and should be formatted as Reddit comments instead of extracted
        as generic text or HTML.

    """
    is_reddit_get = method == "GET" and _is_reddit_url(raw_url)

    if not is_reddit_get:
        body = await asyncio.to_thread(
            _safe_fetch,
            raw_url,
            method=method,
            json_body=json_body,
            form_body=form_body,
        )
        return body, False

    # Reddit thread pages are easier and more stable through Reddit's JSON view.
    if re.search(r"^https?://(?:\w+\.)?reddit\.com/r/\w+/comments/\w+", raw_url):
        url = re.sub(
            r"^(https?://)(?:\w+\.)?reddit\.com/",
            r"\1www.reddit.com/",
            raw_url.rstrip("/"),
        )
        if not url.endswith(".json"):
            url += ".json"
        body = await asyncio.to_thread(_safe_fetch, url)
        return body, True

    try:
        body = await asyncio.to_thread(_safe_fetch, raw_url)
    except FetchError as e:
        if not _is_reddit_verification_page(e.body):
            raise
        return await _fetch_old_reddit(raw_url), False

    # Old Reddit is only a fallback when canonical Reddit serves JS verification.
    if _is_reddit_verification_page(body):
        return await _fetch_old_reddit(raw_url), False
    return body, False


def _safe_fetch(
    url: str,
    *,
    method: str = "GET",
    json_body: JSONValue = None,
    form_body: dict[str, str] | None = None,
) -> bytes:
    """Fetch with SSRF check on the initial URL and every redirect."""
    _check_ssrf(url)
    return fetch(
        url,
        method=method,
        json=json_body,
        data=form_body,
        on_redirect=_check_ssrf,
        validated_hosts=_validated_host,
        timeout_sec=15,
    )


def _check_ssrf(url: str) -> None:
    """Raise if ``url`` resolves to a non-public address."""
    err = _url_is_safe(url)
    if err is not None:
        raise ValueError(err)


def _validated_host(netloc: str) -> ValidatedHost:
    """Return a host/IP pair after SSRF validation."""
    parsed = urlparse(f"//{netloc}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host.")
    err = _url_is_safe(f"http://{netloc}")
    if err is not None:
        raise ValueError(err)
    infos = socket.getaddrinfo(host, None)
    return ValidatedHost(host=netloc, ip=str(infos[0][4][0]))


def _url_is_safe(url: str) -> str | None:
    """Return an error string if ``url`` is unsafe to fetch, else None."""
    try:
        parsed = urlparse(url)
    except ValueError as e:
        return f"Invalid URL: {e}"
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported scheme {parsed.scheme!r}; only http(s) allowed."
    host = parsed.hostname
    if not host:
        return "URL has no host."
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError) as e:
        return f"DNS resolution failed for {host!r}: {e}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return f"Refusing to fetch {host!r} (resolves to non-public address {ip})."
    return None


async def _fetch_old_reddit(raw_url: str) -> bytes:
    """Fetch a Reddit HTML page through the legacy host."""
    url = re.sub(
        r"^(https?://)(?:www\.)?reddit\.com/",
        r"\1old.reddit.com/",
        raw_url,
    )
    try:
        return await asyncio.to_thread(_safe_fetch, url)
    except (FetchError, ValueError, OSError) as e:
        raise ValueError(
            "Reddit returned a JavaScript verification page for "
            f"{raw_url}; old Reddit fallback failed: {e}"
        ) from e


def _is_reddit_url(raw_url: str) -> bool:
    """True for ``reddit.com`` and any subdomain (``old``, ``np``, ``new``).

    Matches subdomains so legacy thread URLs still take the JSON /
    verification-fallback path. ``urlparse(...).hostname`` is already
    lowercased.
    """
    hostname = urlparse(raw_url).hostname or ""
    return hostname == "reddit.com" or hostname.endswith(".reddit.com")


def _is_reddit_verification_page(content: bytes) -> bool:
    """Return whether Reddit returned its JavaScript verification page."""
    text = content.decode("utf-8", errors="replace")
    return "Reddit - Please wait for verification" in text and "js_challenge" in text


def _format_reddit_json(data: list[Any] | dict[str, Any]) -> str:
    """Extract readable text from Reddit's JSON API response."""
    listings = data if isinstance(data, list) else [data]
    lines: list[str] = []
    if listings and listings[0].get("kind") == "Listing":
        posts = listings[0].get("data", {}).get("children", [])
        if posts:
            post = posts[0].get("data", {})
            lines.append(f"# {post.get('title', '')}")
            lines.append(
                f"by u/{post.get('author', '[deleted]')} "
                f"({post.get('score', 0)} points)\n"
            )
            if post.get("selftext", ""):
                lines.append(str(post.get("selftext", "")))
                lines.append("")
    if len(listings) > 1 and listings[1].get("kind") == "Listing":
        comments = listings[1].get("data", {}).get("children", [])
        _format_reddit_comments(comments, lines, depth=0)
    return "\n".join(lines)


def _format_reddit_comments(children: list[Any], lines: list[str], depth: int) -> None:
    """Append formatted Reddit comment JSON to ``lines``."""
    indent = "  " * depth
    for child in children:
        if child.get("kind") != "t1":
            continue
        comment = child.get("data", {})
        lines.append(
            f"{indent}**u/{comment.get('author', '[deleted]')}** "
            f"({comment.get('score', 0)} pts):"
        )
        lines.extend(
            f"{indent}  {body_line}"
            for body_line in comment.get("body", "").splitlines()
        )
        lines.append("")
        replies = comment.get("replies")
        if isinstance(replies, dict):
            replies_dict = cast(dict[str, Any], replies)
            reply_children = replies_dict.get("data", {}).get("children", [])
            _format_reddit_comments(reply_children, lines, depth + 1)


_NUDGE = "curl/wget via Bash is a bad UX. Use the WebFetch tool."
_HTTP_FETCH_BAIL_FLAGS: frozenset[str] = frozenset(
    {
        "-o",
        "-O",
        "--output",
        "--output-document",
        "--data-binary",
        "--upload-file",
        "-T",
        "-F",
        "--form",
    }
)


def _match_http_fetch(exe: str, args: tuple[str, ...]) -> str | None:
    """Return a nudge when a shell command is a simple HTTP fetch."""
    del exe
    url_count = 0
    for arg in args:
        if arg in _HTTP_FETCH_BAIL_FLAGS:
            return None
        if arg.startswith(("http://", "https://")):
            url_count += 1
    if url_count != 1:
        return None
    return _NUDGE
