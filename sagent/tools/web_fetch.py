"""WebFetch tool: fetch a URL and extract its main content."""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import urlparse

import asyncio
import ipaddress
import json
import re
import socket

import cachetools

from sagent.custom_types import Message, TextMessage
from sagent.lib.json import JSON, json_freeze
from sagent.lib.lazy_import import lazy_import
from sagent.lib.message import get_directive
from sagent.lib.web.fetch import FetchError, ValidatedHost, fetch
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)


trafilatura = lazy_import("trafilatura")

_WEBFETCH_CACHE = cachetools.TTLCache[str, str](
    maxsize=128,
    ttl=15 * 60,
)


_REDDIT_RE = re.compile(r"^https?://(?:\w+\.)?reddit\.com/r/(\w+)/comments/(\w+)")


def _is_reddit_url(url: str) -> bool:
    return _REDDIT_RE.search(url) is not None


def _to_reddit_json_url(url: str) -> str:
    """Rewrite a Reddit thread URL to its .json endpoint."""
    url = re.sub(r"^(https?://)(?:\w+\.)?reddit\.com/", r"\1www.reddit.com/", url)
    url = url.rstrip("/")
    if not url.endswith(".json"):
        url += ".json"
    return url


def _format_reddit_json(data: list[Any] | dict[str, Any]) -> str:
    """Extract readable text from Reddit's JSON API response."""
    listings: list[Any] = data if isinstance(data, list) else [data]

    lines: list[str] = []

    if listings and listings[0].get("kind") == "Listing":
        posts: list[Any] = listings[0].get("data", {}).get("children", [])
        if posts:
            post: dict[str, Any] = posts[0].get("data", {})
            title: str = post.get("title", "")
            selftext: str = post.get("selftext", "")
            author: str = post.get("author", "[deleted]")
            score: int = post.get("score", 0)
            lines.append(f"# {title}")
            lines.append(f"by u/{author} ({score} points)\n")
            if selftext:
                lines.append(selftext)
                lines.append("")

    if len(listings) > 1 and listings[1].get("kind") == "Listing":
        comments: list[Any] = listings[1].get("data", {}).get("children", [])
        _format_comments(comments, lines, depth=0)

    return "\n".join(lines)


def _format_comments(children: list[Any], lines: list[str], depth: int) -> None:
    indent = "  " * depth
    for child in children:
        if child.get("kind") != "t1":
            continue
        c: dict[str, Any] = child.get("data", {})
        author: str = c.get("author", "[deleted]")
        score: int = c.get("score", 0)
        body: str = c.get("body", "")
        lines.append(f"{indent}**u/{author}** ({score} pts):")
        lines.extend(f"{indent}  {body_line}" for body_line in body.splitlines())
        lines.append("")
        replies: Any = c.get("replies")
        if isinstance(replies, dict):
            replies_dict = cast(dict[str, Any], replies)
            reply_children: list[Any] = replies_dict.get("data", {}).get("children", [])
            _format_comments(reply_children, lines, depth + 1)


def _normalize_url(url: str) -> str:
    """Rewrite known-problematic URLs to fetchable equivalents."""
    return re.sub(r"^(https?://)(?:\w+\.)?reddit\.com/", r"\1old.reddit.com/", url)


def _check_ssrf(url: str) -> None:
    """Raise if ``url`` resolves to a non-public address."""
    err = _url_is_safe(url)
    if err is not None:
        raise ValueError(err)


def _validated_host(netloc: str) -> ValidatedHost:
    parsed = urlparse(f"//{netloc}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host.")
    err = _url_is_safe(f"http://{netloc}")
    if err is not None:
        raise ValueError(err)
    infos = socket.getaddrinfo(host, None)
    ip = str(infos[0][4][0])
    return ValidatedHost(host=netloc, ip=ip)


def _url_is_safe(url: str) -> str | None:
    """Return an error string if ``url`` is unsafe to fetch, else None.

    Rejects non-HTTP schemes and any hostname that resolves to a
    loopback / link-local / private address. Prevents SSRF when the
    tool is exposed via a hosted surface (e.g. Slack bot) where the
    model's URL argument is effectively attacker-controlled.
    """
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


def _safe_fetch(url: str) -> bytes:
    """Fetch with SSRF check on the initial URL and every redirect."""
    _check_ssrf(url)
    return fetch(
        url,
        on_redirect=_check_ssrf,
        validated_hosts=_validated_host,
        timeout_sec=15,
    )


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
            },
            "required": ["url"],
        }
    )

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

    def prompt(self) -> str:
        """Return supplemental system-prompt text.

        Returns:
          prompt: Empty string; this tool adds no prompt.

        """
        return ""

    async def run(self, msg: Message) -> Message:
        """Fetch the URL, extract main content, and return as text.

        Args:
          msg: Directive message containing the target URL.

        Returns:
          result: Extracted page text or an error message.

        """
        directive = get_directive(msg)
        raw_url = str(directive.get("url", ""))
        reddit = _is_reddit_url(raw_url)
        url = _to_reddit_json_url(raw_url) if reddit else _normalize_url(raw_url)
        cache_key = raw_url
        cached = _WEBFETCH_CACHE.get(cache_key)
        if cached is not None:
            return TextMessage(cached, "text/plain")
        try:
            body = await asyncio.to_thread(_safe_fetch, url)
        except (FetchError, ValueError, OSError) as e:
            return TextMessage(f"Fetch failed: {e}", "text/x-error")
        content = body.decode("utf-8", errors="replace")
        if reddit:
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                text = content[:TOOL_RESULT_MAX_CHARS]
            else:
                text = _format_reddit_json(data)
        else:
            extracted = await asyncio.to_thread(
                trafilatura.extract,
                content,
                include_links=True,
                include_tables=True,
            )
            text = extracted or content[:TOOL_RESULT_MAX_CHARS]
        truncated = truncate(text, TOOL_RESULT_MAX_CHARS)
        _WEBFETCH_CACHE[cache_key] = truncated
        return TextMessage(truncated, "text/plain")
