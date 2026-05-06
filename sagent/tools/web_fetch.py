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

_WEBFETCH_CACHE = cachetools.TTLCache[str, str](
    maxsize=128,
    ttl=15 * 60,
)

_NUDGE = "curl/wget via Bash is a bad UX. Use the WebFetch tool."


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


def _safe_fetch(
    url: str,
    *,
    method: str = "GET",
    json_body: JSONValue = None,
    form_body: dict[str, str] | None = None,
) -> bytes:
    """Fetch with SSRF check on the initial URL and every redirect.

    Args:
      url: Fully-qualified URL to fetch.
      method: HTTP method. Restricted to GET or POST.
      json_body: JSON-serializable body (POST only). Mutually exclusive
        with ``form_body``.
      form_body: Form fields encoded as application/x-www-form-urlencoded
        (POST only). Mutually exclusive with ``json_body``.

    Returns:
      Response body bytes.

    """
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

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``curl URL`` / ``wget URL``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Bails on output redirection (``-o``/``-O``/``--output``),
        ``--data-binary @file`` style file uploads, and any non-http(s)
        URL — those are cases WebFetch can't cleanly replace. Pipelines
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
        json_body: JSONValue = None
        form_body: dict[str, str] | None = None
        if method == "POST":
            raw_json = directive.get("json")
            raw_form = directive.get("form")
            if raw_json is not None and raw_form is not None:
                return TextMessage(
                    "'json' and 'form' are mutually exclusive.",
                    "text/x-error",
                )
            if raw_json is not None:
                json_body = json_unfreeze(raw_json)
            elif raw_form is not None:
                form_body = {
                    str(k): str(v)
                    for k, v in cast(dict[str, Any], json_unfreeze(raw_form)).items()
                }

        reddit = _is_reddit_url(raw_url) and method == "GET"
        url = _to_reddit_json_url(raw_url) if reddit else _normalize_url(raw_url)
        # Cache GETs only; POSTs are non-idempotent.
        cache_key = raw_url if method == "GET" else None
        if cache_key is not None:
            cached = _WEBFETCH_CACHE.get(cache_key)
            if cached is not None:
                return TextMessage(cached, "text/plain")
        try:
            body = await asyncio.to_thread(
                _safe_fetch,
                url,
                method=method,
                json_body=json_body,
                form_body=form_body,
            )
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
        elif method == "POST" or content.lstrip().startswith(("{", "[")):
            # POST responses and JSON-shaped bodies pass through verbatim;
            # trafilatura would mangle them.
            text = content[:TOOL_RESULT_MAX_CHARS]
        else:
            extracted = await asyncio.to_thread(
                trafilatura.extract,
                content,
                include_links=True,
                include_tables=True,
            )
            text = extracted or content[:TOOL_RESULT_MAX_CHARS]
        truncated = truncate(text, TOOL_RESULT_MAX_CHARS)
        if cache_key is not None:
            _WEBFETCH_CACHE[cache_key] = truncated
        return TextMessage(truncated, "text/plain")


# curl/wget flags that take a value but mean "save to disk", "upload a
# local file", or otherwise change the request shape in ways WebFetch
# can't mirror. Encountering one bails the matcher.
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
    # Shape: ``<exe> [flags] URL`` with exactly one http(s):// URL and
    # no flag from the bail set. Bare ``-`` and unknown long flags are
    # tolerated; flags Wget/curl share that imply a file sink are not.
    del exe  # Hint is fixed; the LLM rederives the URL from its own command.
    url_count = 0
    i = 0
    while i < len(args):
        a = args[i]
        if a in _HTTP_FETCH_BAIL_FLAGS:
            return None
        if a.startswith(("http://", "https://")):
            url_count += 1
        i += 1
    if url_count != 1:
        return None
    return _NUDGE
