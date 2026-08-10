"""WebFetch tool: fetch a URL (GET or POST) and extract its content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal, cast, get_args

import asyncio

from wesearch.errors import BotDetectionError, FetchError
from wesearch.fetch import Policy, Transport
from wesearch.web import fetch_web

import cachetools

from sagent.lib.custom_json import JSON, JSONValue, json_freeze, json_unfreeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import Node, walk_commands
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


# The only HTTP methods this tool supports; enforced at the directive boundary.
HttpMethod = Literal["GET", "POST"]


class WebFetch:
    """Fetch a web page and extract its main content as clean text."""

    name: str = "WebFetch"
    tool_id: str = "application/x-tool-webfetch"
    clearable_results: bool = True
    description: str = load_tool_description("WebFetch")
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
                "transport": {
                    "type": "string",
                    "enum": get_args(Transport),
                    "description": (
                        "Retrieval path. 'auto' tries curl and escalates to "
                        "Zendriver when a site bot-blocks it. Set an explicit "
                        "transport to stress a path."
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

    output: Annotated[Toggle, CLI_SETTABLE] = "off"
    """Whether the result body renders in the pane."""

    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    """Leading body rows kept."""

    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    """Trailing body rows kept, after a ``⋯ N lines ⋯`` marker."""

    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    """Cell width cap; ``0`` uses the pane width."""

    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"
    """``wrap`` continues an over-wide line, ``chop`` marks the cut."""

    def __init__(self) -> None:
        self._cache = cachetools.TTLCache[tuple[Transport, str], str](
            maxsize=128, ttl=15 * 60
        )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``curl URL`` / ``wget URL``.

        Policy only: :func:`walk_commands` supplies every simple command
        with its context, so a leading ``cd``, an enclosing loop, and a
        sequence all reach this matcher without it re-deriving AST shape.
        Bails on output redirection (``-o``/``-O``/``--output``),
        ``--data-binary @file`` style file uploads, and any non-http(s)
        URL -- those are cases WebFetch can't cleanly replace. A piped
        fetch is feeding another program rather than being read, which
        WebFetch does not replace either.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the WebFetch tool, or ``None``.

        """
        for inv in walk_commands(trees):
            if inv.env_prefix or inv.captures_stdout or inv.piped_into:
                continue
            if inv.exe not in {"curl", "wget"}:
                continue
            hint = _match_http_fetch(inv.exe, inv.args)
            if hint is not None:
                return hint
        return None

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Directive carrying ``url``.

        Returns:
          label: ``WebFetch <url>`` line shown before invocation.

        """
        return f"WebFetch {args.get('url', '')}"

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for WebFetch.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Fetch the URL, extract main content, and return as text.

        GET responses are cached per URL and transport for 15 minutes. A URL
        whose server-side extraction path changes during the TTL window (e.g. a
        Reddit page that starts returning a different shape and falls into a
        different adapter) continues to serve the previously-extracted body
        until the entry expires. Switching transport bypasses that cache entry.

        Args:
          args: Directive with ``url`` and optional ``method`` / ``json``
              / ``form`` keys.

        Returns:
          result: Extracted text body, or a fetch/extraction error.

        """
        raw_url = str(args.get("url", ""))
        raw_method = str(args.get("method", "GET")).upper()
        if raw_method not in ("GET", "POST"):
            return ToolResult(
                call_id="",
                content=(
                    f"Unsupported method {raw_method!r}; only GET and POST allowed."
                ),
                is_error=True,
            )
        # The guard above admits only "GET"/"POST", narrowing raw_method to
        # the HttpMethod literal.
        method: HttpMethod = raw_method
        raw_transport = args.get("transport", "auto")
        if not isinstance(raw_transport, str) or raw_transport not in get_args(
            Transport
        ):
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid transport {raw_transport!r}."
                    f" Valid: {', '.join(get_args(Transport))}."
                ),
                is_error=True,
            )
        transport = cast(Transport, raw_transport)

        try:
            json_body, form_body = _request_bodies(method, args)
        except ValueError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

        cache_key = (transport, raw_url) if method == "GET" else None
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # This TTL also hides a local edit to wesearch.web: the
                # tool replays pre-change text, so a working extraction fix
                # reads as a failed one. Verify such a change via fetch_web in
                # a fresh process, not by re-running this tool.
                return ToolResult(call_id="", content=cached)

        try:
            # No max_chars: fetch_web returns the full text, and truncate() below
            # applies sagent's presentation cap WITH the truncation notice. Passing
            # the cap here would pre-cut the text to exactly the limit, so truncate
            # would see an at-limit string and append no notice.
            result = await asyncio.to_thread(
                fetch_web,
                raw_url,
                method=method,
                json_body=json_body,
                form_body=form_body,
                policy=Policy(transport=transport),
            )
        except BotDetectionError as e:
            # fetch() classified the block at the boundary: surface the SPECIFIC
            # kind (Cloudflare vs puzzle vs Google /sorry), each with its own
            # actionable guidance, rather than a generic "HTTP 403".
            return ToolResult(call_id="", content=e.explain(raw_url), is_error=True)
        except (FetchError, ValueError, OSError) as e:
            return ToolResult(call_id="", content=f"Fetch failed: {e}", is_error=True)

        content = truncate(result.text, TOOL_RESULT_MAX_CHARS)
        if cache_key is not None:
            self._cache[cache_key] = content
        return ToolResult(call_id="", content=content)


def _request_bodies(
    method: HttpMethod,
    args: Mapping[str, object],
) -> tuple[JSONValue, dict[str, str] | None]:
    """Return POST request bodies from a tool directive."""
    raw_json = args.get("json")
    raw_form = args.get("form")
    if method != "POST":
        if raw_json is not None or raw_form is not None:
            raise ValueError("'json' and 'form' require method='POST'.")
        return None, None
    if raw_json is not None and raw_form is not None:
        raise ValueError("'json' and 'form' are mutually exclusive.")
    if raw_json is not None:
        return json_unfreeze(raw_json), None
    if raw_form is None:
        return None, None
    unfrozen_form = json_unfreeze(raw_form)
    # Schema declares ``form`` as an object, but LLM-supplied
    # directives can violate the schema (``form=[]`` slipped through
    # historically). Reject anything not a mapping so the downstream
    # ``.items()`` call can't AttributeError out of the tool envelope.
    # ``ValueError`` (not ``TypeError``) so it joins the existing
    # caller-side ``except ValueError`` envelope in ``WebFetch.run``.
    if not isinstance(unfrozen_form, dict):
        raise ValueError(  # noqa: TRY004 -- caller catches ValueError uniformly.
            f"'form' must be an object of string fields, got {type(unfrozen_form).__name__}."
        )
    return None, {
        str(k): str(v) for k, v in cast(dict[str, Any], unfrozen_form).items()
    }


_NUDGE: Final = "curl/wget via Bash is a bad UX. Use the WebFetch tool."
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
