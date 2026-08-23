"""WebFetch tool: fetch a URL (GET or POST) and extract its content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Final, cast, get_args

import asyncio

from wesearch.fetch import Extractor, PolicyParams, Transport
from wesearch.fetch.custom_types import FetchBodyParamsSchema, HttpMethod
from wesearch.types.errors import BotDetectionError, FetchError
from wesearch.web import fetch_web

import cachetools

from sagent.lib.custom_json import JSONValue, json_freeze, json_unfreeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.lib.bash import Node, walk_commands
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class WebFetch:
    """Fetch a web page and extract its main content as clean text."""

    name = "WebFetch"
    tool_id = "application/x-tool-webfetch"
    clearable_results = True
    description = load_tool_description("WebFetch")
    directive_schema = json_freeze(FetchBodyParamsSchema.json_schema())

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

    # The cache is per-instance MUTABLE state on a frozen record: the
    # freeze is about configuration, not about the response cache, and
    # ``/tool`` rebuilds the instance so a default_factory keeps each
    # swap from inheriting a stale cache.
    _cache: cachetools.TTLCache[tuple[Transport, Extractor, str], str] = field(
        default_factory=lambda: cachetools.TTLCache[
            tuple[Transport, Extractor, str], str
        ](maxsize=128, ttl=15 * 60),
        repr=False,
        compare=False,
    )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``curl URL`` / ``wget URL``.

        PolicyParams only: :func:`walk_commands` supplies every simple command
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

        GET responses are cached per URL, transport, and extractor for 15
        minutes. A URL whose server-side extraction path changes during the TTL
        window (e.g. a Reddit page that starts returning a different shape and
        falls into a different adapter) continues to serve the
        previously-extracted body until the entry expires. Switching transport
        or extractor bypasses that cache entry.

        Args:
          args: Directive with ``url`` and optional ``method`` / ``transport``
              / ``extractor`` / ``json`` / ``form`` keys.

        Returns:
          result: Extracted text body, or a fetch/extraction error.

        """
        # Case-folded before coercion, not by it: a model writes "get" as
        # readily as "GET", and the spec's job is to say what the accepted
        # values ARE, not which spellings of them a directive may use.
        directive = dict(args)
        if isinstance(directive.get("method"), str):
            directive["method"] = str(directive["method"]).upper()
            if directive["method"] not in get_args(HttpMethod):
                # Named separately from the generic coercion error: a rejected
                # VERB wants the reason ("this tool does not do PUT") ahead of
                # the list of accepted values.
                return ToolResult(
                    call_id="",
                    content=(
                        f"Unsupported method {directive['method']!r};"
                        " only GET and POST allowed."
                    ),
                    is_error=True,
                )
        try:
            # One call replaces a per-parameter validation ladder that restated
            # every name, type, and default the schema above already declared.
            params = FetchBodyParamsSchema.coerce(directive)
        except ValueError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
        raw_url = str(params["url"])
        method = cast(HttpMethod, params["method"])
        transport = cast(Transport, params["transport"])
        extractor = cast(Extractor, params["extractor"])

        try:
            json_body, form_body = _request_bodies(method, args)
        except ValueError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

        # The extractor is part of the key: two extractors of one URL are two
        # different results, and omitting it would replay the first one's text
        # for the second.
        cache_key = (transport, extractor, raw_url) if method == "GET" else None
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
                policy=PolicyParams(transport=transport, extractor=extractor),
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
    # Values checked, not stringified: str() turned {"x": []} into the literal
    # field "x=[]" and {"x": {"a": 1}} into "x={'a': 1}" -- a request the caller
    # never described, sent to a real endpoint, while the schema promised
    # strings. cast() to a value-typed dict so the check narrows rather than
    # asserting.
    form: dict[str, str] = {}
    for key, value in cast(dict[str, object], unfrozen_form).items():
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004 -- caller catches ValueError uniformly.
                f"'form' field {key!r} must be a string, got {type(value).__name__}."
            )
        form[str(key)] = value
    return None, form


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
        # MEASURED: ``curl -K opts.conf URL`` with ``output = "out.html"``
        # in the file wrote out.html. The flag that writes is IN THE FILE,
        # so no argv scan can ever see it -- the option source itself has
        # to be the bail.
        "-K",
        "--config",
        # ``-d @file`` reads the body from disk; WebFetch's ``json``/
        # ``form`` are inline values only.
        "-d",
        "--data",
        # Saves every URL to a file, as ``-O`` does for one.
        "--remote-name-all",
        "-J",
        "--remote-header-name",
    }
)

# Utilities that write to disk BY DEFAULT, so the bare invocation is the
# one that needs a bail. MEASURED: ``wget https://example.com`` wrote
# index.html into the cwd. Only an explicit "send it to stdout" spelling
# makes wget the shape WebFetch replaces.
_WRITES_BY_DEFAULT: frozenset[str] = frozenset({"wget"})


def _match_http_fetch(exe: str, args: tuple[str, ...]) -> str | None:
    """Return a nudge when a shell command is a simple HTTP fetch.

    A fetch that writes a file or uploads one is not something WebFetch
    can do, so those forms stay with Bash. Exact-string matching missed
    every spelling but the separated one: ``--output=x``, the bundled
    ``-sO``, and ``--output-document=x`` all still nudged.

    Two axes, because a flag denylist alone answers neither: an option
    FILE can carry the write flag where argv never shows it, and ``wget``
    writes with no flag at all.
    """
    # ``-O -`` is wget's "write the body to stdout", so the output flag
    # is exactly what makes this shape replaceable. Asked FIRST, because
    # the generic scan below denies ``-O`` on sight.
    streams = _streams_to_stdout(args)
    url_count = 0
    for arg in args:
        if not streams and _writes_a_file(arg):
            return None
        if arg.startswith(("http://", "https://")):
            url_count += 1
    if url_count != 1:
        return None
    # ``wget`` saves to disk with no flag at all, so a bare invocation is
    # the one that needs the bail rather than the one that earns a nudge.
    if exe in _WRITES_BY_DEFAULT and not streams:
        return None
    return _NUDGE


def _streams_to_stdout(args: tuple[str, ...]) -> bool:
    """Whether ``wget`` was told to write the body to stdout.

    ``-O -`` (and its bundled ``-qO-``) is the only form that does; every
    other invocation saves a file, so the polarity is the reverse of the
    ``-O`` denial that applies to ``curl``.
    """
    for i, a in enumerate(args):
        if a in ("-O", "--output-document") and i + 1 < len(args):
            return args[i + 1] == "-"
        if a.startswith("--output-document="):
            return a.partition("=")[2] == "-"
        if a.startswith("-") and not a.startswith("--") and a.endswith("O-"):
            return True
    return False


def _writes_a_file(arg: str) -> bool:
    """Whether one argument makes the fetch write or upload a file."""
    if arg in _HTTP_FETCH_BAIL_FLAGS:
        return True
    # ``--output=out.html`` -- the value is attached, so the flag never
    # appears as its own token.
    if arg.startswith("--") and arg.partition("=")[0] in _HTTP_FETCH_BAIL_FLAGS:
        return True
    # ``-sO`` / ``-oout`` -- short flags bundle, so the output flag is a
    # letter inside the cluster rather than the whole token.
    if arg.startswith("-") and not arg.startswith("--"):
        return any(letter in arg[1:] for letter in _HTTP_FETCH_BAIL_LETTERS)
    return False


# Short flags whose presence anywhere in a bundle means the fetch writes
# or uploads a file: ``-o``/``-O`` output, ``-T`` upload, ``-F`` form.
_HTTP_FETCH_BAIL_LETTERS: frozenset[str] = frozenset({"o", "O", "T", "F"})
