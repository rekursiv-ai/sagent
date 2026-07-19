"""Shared contracts and HTTP policy for fetch transports."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import gzip
import io
import zlib

import brotli
import zstandard

from sagent.lib.web.chrome_headers import chrome_client_hints
from sagent.lib.web.errors import FetchError


__all__ = [
    "Observer",
    "ValidatedHost",
    "ValidatedHosts",
    "apply_redirect",
    "bracket_ipv6",
    "decompress",
    "decompress_error_body",
    "default_port",
    "host_header",
    "join_headers",
    "redirect_target",
    "rewrite_origin",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedHost:
    host: str
    ip: str


ValidatedHosts = Callable[[str], ValidatedHost]

# Internal per-hop response sink: (status, response headers, responding URL).
# The URL lets a sink scope what it learns (cookies, hints) to the hop's origin,
# so a cross-origin redirect never mis-attributes the target's Set-Cookie to the
# source. The public ``RequestParams.on_response`` stays (status, headers).
Observer = Callable[[int, dict[str, str], str], None]
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _origin(url: str) -> str:
    """The scheme://host[:port] origin of a URL (the Accept-CH opt-in key)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def decompress(body: bytes, encoding: str) -> bytes:
    """Decompress a response body per Content-Encoding; raise ValueError if bad.

    ``Content-Encoding`` may chain several codings (RFC 9110 SS 8.4.1, e.g.
    ``gzip, br``); they are applied left-to-right on encode, so decode
    right-to-left. Each token is one coding.
    """
    for enc in reversed([tok.strip().lower() for tok in encoding.split(",")]):
        body = _decompress_one(body, enc)
    return body


def _decompress_one(body: bytes, enc: str) -> bytes:
    """Decompress ``body`` under a SINGLE Content-Encoding token."""
    if enc in ("", "identity"):
        return body
    try:
        if enc == "gzip":
            return gzip.decompress(body)
        if enc == "deflate":
            # RFC 7230 says zlib-wrapped, but some servers emit RAW DEFLATE (no
            # header); browsers retry with a negative window. Try zlib first,
            # fall back to raw so a header-less stream still decodes.
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
        if enc == "br":
            return brotli.decompress(body)
        if enc == "zstd":
            # stream_reader handles frames without an embedded size,
            # which `.decompress()` rejects. Servers (e.g. Cloudflare)
            # commonly emit such frames.
            return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)).read()
    except (OSError, zlib.error, brotli.error, zstandard.ZstdError) as e:
        raise ValueError(f"Decompression failed ({enc}): {e}") from None
    raise ValueError(f"Unknown Content-Encoding: {enc!r}")


def decompress_error_body(body: bytes, headers: dict[str, str]) -> bytes:
    """Decompress an ERROR response body best-effort, never raising."""
    # Error pages are compressed like any success body (Cloudflare serves its
    # challenge pages zstd/br), so a raw FetchError.body is undecodable garbage
    # and a caller cannot tell a challenge from a genuine 404. This must NOT
    # raise: an undecodable body must still surface the original HTTP error, so
    # a decompression failure returns the raw bytes rather than mask the status.
    encoding = headers.get("content-encoding", "identity")
    try:
        return decompress(body, encoding)
    except ValueError:
        return body


def join_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lowercase header pairs, folding duplicates into one value.

    Duplicates join with ``", "`` per RFC 9110 SS 5.3 -- EXCEPT ``set-cookie``,
    which that RFC explicitly exempts (a cookie value may itself contain ``", "``,
    so folding then re-splitting mis-parses it). Multiple ``set-cookie`` headers
    join with a newline instead -- a byte that never appears in a header value --
    so :func:`sagent.lib.web.profile.parse_set_cookie` can split them back exactly.
    """
    out: dict[str, str] = {}
    for k, v in pairs:
        key = k.lower()
        if key not in out:
            out[key] = v
        elif key == "set-cookie":
            out[key] = f"{out[key]}\n{v}"
        else:
            out[key] = f"{out[key]}, {v}"
    return out


def redirect_target(current_url: str, status: int, headers: dict[str, str]) -> str:
    """Resolve a redirect ``Location`` against *current_url*; raise if absent."""
    location = headers.get("location")
    if not location:
        raise FetchError(
            current_url, status, headers, b"Redirect with no Location header"
        )
    # RFC 3986 relative resolution: handles absolute, scheme-relative, and
    # path-relative Locations without corrupting the host.
    return urljoin(current_url, location)


def default_port(scheme: str) -> int:
    """Return the default TCP port for an HTTP scheme."""
    return 443 if scheme == "https" else 80


def _netloc(hostname: str, port: int | None) -> str:
    """Recombine a hostname and optional port into a netloc."""
    return f"{hostname}:{port}" if port is not None else hostname


def host_header(host: str, port: int | None, scheme: str) -> str:
    """The ``Host`` header value: bare host, plus a non-default port."""
    # RFC 9110 requires the port in Host only when it is not the scheme default;
    # a real browser omits :443/:80. The ONE place this rule lives, so the
    # initial hop and the cross-host-redirect rebuild cannot disagree.
    if port is not None and port != default_port(scheme):
        return _netloc(host, port)
    return host


def bracket_ipv6(host: str) -> str:
    """Wrap an IPv6 literal in brackets; pass hostnames and IPv4 through."""
    # http.client splits host on the last ':' for a port, misparsing a bare
    # "2606:4700::6810:7c60" as host+port. Bracketing avoids that heuristic.
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def rewrite_origin(headers: dict[str, str], target_url: str) -> dict[str, str]:
    """Return ``headers`` with ``Origin`` reset to ``target_url``'s origin."""
    # A cross-origin redirect must NOT leak the source Origin to the new host (a
    # real browser sets it to the new origin, never the old). A GET carries no
    # Origin and passes through. Field names are case-insensitive, so a
    # caller-supplied "origin" in any case is matched and rewritten in place.
    origin_key = next((k for k in headers if k.lower() == "origin"), None)
    if origin_key is None:
        return headers
    parsed = urlparse(target_url)
    scheme = parsed.scheme or "https"
    # Bracket a v6 host: an unbracketed IPv6 literal is not a valid Origin
    # (the colons collide with the scheme/port delimiters).
    netloc = _netloc(bracket_ipv6(parsed.hostname or ""), parsed.port)
    return {**headers, origin_key: f"{scheme}://{netloc}"}


def apply_redirect(
    current_url: str,
    headers: dict[str, str],
    method: str,
    body: bytes | None,
    status: int,
    redirect_url: str,
) -> tuple[dict[str, str], str, bytes | None]:
    """Compute the (headers, method, body) for the next hop of a redirect.

    The ONE place the per-hop transform lives, called by every transport's
    redirect loop so the rules cannot drift, mirroring what a real browser does:

    - Rewrite ``Origin`` to the new target.
    - On 301/302/303 of a non-GET, convert to a bodyless GET dropping
      Content-Type (browsers downgrade all three; only 307/308 preserve the
      method -- that is why 307/308 exist).
    - On a CROSS-ORIGIN hop, drop every origin-bound header (``Cookie`` and the
      extended client hints), since those belong to the source origin and must
      not leak to the target. Same-origin hops keep them.

    Casing-insensitive throughout.
    """
    headers = rewrite_origin(headers, redirect_url)
    if status in (301, 302, 303) and method != "GET":
        method = "GET"
        body = None
        headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    if _origin(current_url) != _origin(redirect_url):
        headers = {k: v for k, v in headers.items() if k.lower() not in _ORIGIN_BOUND}
    return headers, method, body


# Headers scoped to the origin that set/opted-into them; dropped on a
# cross-origin redirect so the source origin's Cookie and extended client hints
# never leak to the target (a real browser scopes both per origin).
_ORIGIN_BOUND: frozenset[str] = frozenset(
    {"cookie"} | {name.lower() for name in chrome_client_hints(major=1)}
)
