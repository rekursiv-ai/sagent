"""WebFetch: a URL's content as text, by GET or POST.

Text fetched by GET is kept for 15 minutes, keyed by URL, transport, and
extractor; a POST may have effects, so it is never replayed.

Typical usage example:

  result = await WebFetch(session=FetchSession()).call(context, {"url": "https://peps.python.org/pep-0008/"})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlsplit

import asyncio
import ipaddress
import time

from treekle import PlainTree, mutable
from wesearch.fetch import Extractor, FetchSession, PolicyParams, Transport
from wesearch.mcp.types import FetchBodyParamsSchema, FetchMethod
from wesearch.types.errors import BotDetectionError, FetchError
from wesearch.types.params import NO_BODY
from wesearch.web import fetch_web

import cachetools

from sagent.session.types import UncategorizedToolResult, WebFetchResult
from sagent.tools.command_analyzer import Invocation, replaceable
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from sagent.tools.types import ToolContext


__all__ = [
    "WebFetch",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WebFetch:
    """Fetch a URL and return its main content as text."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        url: Annotated[str, FetchBodyParamsSchema.url.description]
        method: Annotated[
            FetchMethod,
            FetchBodyParamsSchema.method.description,
        ] = "GET"
        transport: Annotated[
            Transport | None,
            FetchBodyParamsSchema.transport.description,
        ] = None
        extractor: Annotated[
            Extractor | None,
            FetchBodyParamsSchema.extractor.description,
        ] = None
        json: Annotated[PlainTree, FetchBodyParamsSchema.json.description] = None
        form: Annotated[
            Mapping[str, str] | None,
            FetchBodyParamsSchema.form.description,
        ] = None

    name = "WebFetch"
    description = (files(__package__) / "assets" / "webfetch.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    display: ToolDisplay = field(default=ToolDisplay())

    session: FetchSession = field(repr=False)
    """Connections, browser, and egress every fetch reuses; whoever made it closes it."""

    policy: PolicyParams = field(default_factory=PolicyParams)
    """How to fetch when a call does not say."""

    _cache: cachetools.TTLCache[tuple[str, str, str], str] = field(
        default_factory=lambda: cachetools.TTLCache[tuple[str, str, str], str](
            maxsize=128,
            ttl=15 * 60,
        ),
        init=False,
        repr=False,
    )

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``WebFetch <url>``."""
        return f"WebFetch {arguments.get('url', '')}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: fetches are independent."""
        del context, arguments
        return None

    def from_shell(
        self,
        invocation: Invocation,
    ) -> Mapping[str, PlainTree] | str | None:
        """Return the WebFetch call for a plain ``curl URL`` or ``wget -qO- URL``.

        Any other flag -- a header, a body, an output file -- a word bash
        expands, or a private host, such as ``localhost``, keeps the command a
        shell command.

        Args:
          invocation: One simple command, with its pipeline.

        Returns:
          arguments: The WebFetch call for the same page, or ``None``.

        """
        # A word bash expands first (``$P``) names a URL only the shell knows.
        if invocation.expands or not replaceable(
            invocation,
            exes=frozenset({"curl", "wget"}),
        ):
            return None
        urls = [a for a in invocation.args if a.startswith(("http://", "https://"))]
        flags = [a for a in invocation.args if a not in urls]
        quiet = {"-s", "-S", "-L", "-f", "-sS", "-sL", "-fsSL", "-sSL", "-fsS", "-fL"}
        quiet |= {"--silent", "--show-error", "--location", "--fail"}
        if invocation.exe == "wget":
            # Bare ``wget`` saves to a file; only ``-O -`` prints the page.
            plain = flags in (
                ["-qO-"],
                ["-qO", "-"],
                ["-q", "-O", "-"],
                ["-O", "-"],
                ["-O-"],
            )
        else:
            plain = set(flags) <= quiet
        if len(urls) != 1 or not plain:
            return None
        host = urlsplit(urls[0]).hostname or ""
        try:
            private = not ipaddress.ip_address(host).is_global
        except ValueError:
            private = host == "localhost" or host.endswith((".local", ".internal"))
        return None if private else {"url": urls[0]}

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> WebFetchResult | UncategorizedToolResult:
        """Fetch ``url`` and extract its text.

        A POST carries ``json`` or ``form``, never both; a GET carries neither.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``WebFetch.Arguments``.

        Returns:
          result: The page's text, or why it could not be fetched.

        """
        # A model writes "get" as readily as "GET".
        method = arguments.get("method")
        if isinstance(method, str):
            arguments = {**arguments, "method": method.upper()}
        a = parse(arguments, WebFetch.Arguments)
        has_body = a.json is not None or a.form is not None
        if a.method == "GET" and has_body:
            return _error("'json' and 'form' require method='POST'.")
        if a.json is not None and a.form is not None:
            return _error("'json' and 'form' are mutually exclusive.")
        policy = replace(
            self.policy,
            transport=a.transport or self.policy.transport,
            extractor=a.extractor or self.policy.extractor,
        )
        key = (a.url, policy.transport, policy.extractor)
        started = time.monotonic()
        text = self._cache.get(key) if a.method == "GET" else None
        if text is None:
            try:
                fetched = await asyncio.to_thread(
                    fetch_web,
                    a.url,
                    method=a.method,
                    json_body=NO_BODY if a.json is None else mutable(a.json),
                    form_body=None if a.form is None else dict(a.form),
                    policy=policy,
                    session=self.session,
                )
            except BotDetectionError as e:
                # Which block it was decides the remedy: Cloudflare, a
                # puzzle, or Google's /sorry page.
                return _error(e.explain(a.url))
            except (FetchError, ValueError, OSError) as e:
                return _error(f"Fetch failed: {e}")
            text = fetched.text
            if a.method == "GET":
                self._cache[key] = text
        lines = text.splitlines(keepends=True)
        note = f"\n... ({len(lines):,} more lines omitted)"
        body, kept = context.fit(lines, reserve=len(note))
        if kept < len(lines):
            body += f"\n... ({len(lines) - kept:,} more lines omitted)"
        return WebFetchResult(
            call_id="",
            url=a.url,
            content=body,
            duration_sec=time.monotonic() - started,
            size=len(text),
        )


def _error(message: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=message, is_error=True)
