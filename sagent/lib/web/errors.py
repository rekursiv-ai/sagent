"""HTTP fetch errors and automated-access detection taxonomy for ``sagent.lib.web``.

This is the lowest layer of ``sagent.lib.web`` -- it imports nothing from the
package, so every other module (notably :mod:`sagent.lib.web.fetch`) can depend
on it without a cycle.

:class:`FetchError` is the base for every non-success HTTP outcome. A fetch can
also be stopped when a site treats the request as automated and serves a
challenge or block page instead of content. Those responses form a flat
hierarchy under :class:`BotDetectionError`, itself a ``FetchError`` subclass: it
carries the same ``url``/``status``/``headers``/``body`` as any fetch failure,
so every existing ``except FetchError`` catches it, PLUS ``guidance`` that says
how to proceed. One class per kind of response we can identify from the body:

- :class:`PuzzleChallengeError` -- a solve-a-puzzle page (reCAPTCHA/hCaptcha) or
  an interactive challenge form. Cleared by a human/solver completing a puzzle.
- :class:`CloudflareChallengeError` -- Cloudflare's managed challenge (``cf_chl``
  / ``challenge-platform`` / ``cf-ray``). Detectable as Cloudflare, but its mode
  (checkbox / puzzle / invisible-JS) is decided client-side in the served JS, so
  the mechanism is not further splittable from the body.
- :class:`GoogleSorryError` -- Google's ``/sorry`` interstitial: a refusal
  served when Google treats the traffic as automated, not a solve-this challenge.
- :class:`GoogleJavascriptRequiredError` -- Google's ``enablejs`` shell: an
  HTTP-200 page with no results that meta-refreshes to ``/httpservice/retry/
  enablejs``, served when Google enforces its JavaScript requirement on a
  server-side (no-JS) request. Distinct from ``/sorry`` -- not an IP refusal but
  a "run the page JS" wall.

The base :class:`BotDetectionError` is raised directly when a response is clearly
such a block but the kind is not determinable (e.g. a generic "security check
required" page).

:func:`classify_bot_detection` returns the most specific class the body matches
(or ``None`` for ordinary content); :func:`classify_http_error` uses it at the
fetch boundary to raise the matching :class:`BotDetectionError` subclass.
"""

from __future__ import annotations


__all__ = [
    "BotDetectionError",
    "CloudflareChallengeError",
    "FetchError",
    "GoogleJavascriptRequiredError",
    "GoogleSorryError",
    "PuzzleChallengeError",
    "classify_bot_detection",
    "classify_http_error",
]


class FetchError(Exception):
    """HTTP request returned a non-success status code.

    Attributes:
      url: Requested URL.
      status: HTTP status code.
      headers: Response headers.
      body: Response body bytes, decompressed (for debugging / classifying
        error pages). An access block is classified at the fetch boundary and
        raised as a :class:`BotDetectionError` subclass, so callers discriminate
        on the exception TYPE, not by re-inspecting this body.

    """

    def __init__(
        self,
        url: str,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        self.url = url
        self.status = status
        self.headers = headers
        self.body = body
        # Status 0 is the "no HTTP response" sentinel (timeout / TLS / connect
        # failure), not a real HTTP status -- render it as a connection failure
        # and surface the reason the body carries rather than a bogus "HTTP 0".
        if status == 0:
            reason = body.decode("utf-8", "replace").strip() or "connection failed"
            super().__init__(f"connection failed: {url}: {reason}")
        else:
            super().__init__(f"HTTP {status}: {url}")


class BotDetectionError(FetchError):
    """A site served an automated-access challenge or block instead of content.

    Raised directly when the kind of block is not determinable from the response;
    subclasses name a kind the body identifies.

    A :class:`FetchError` subclass, so every ``except FetchError`` catches it and
    inherits the ``url``/``status``/``headers``/``body`` attributes. It ADDS
    ``guidance``: a specific, actionable one-liner on WHAT it is and HOW to
    proceed, so a consumer rendering the error tells a caller something useful,
    not a generic "blocked". :meth:`explain` composes it with the offending URL.

    Two construction modes, because such a block surfaces at two kinds of site
    with different information in hand:

    - Boundary (:func:`sagent.lib.web.fetch.fetch`) has the full HTTP quadruple:
      ``CloudflareChallengeError(url, status, headers, body)``.
    - A scraper that spots a challenge mid-parse has only a reason string, not a
      live HTTP response: ``PuzzleChallengeError("DuckDuckGo challenge form")``.
      It must NOT fabricate a status/headers or stuff the message into ``body``
      (``body`` is response bytes that :func:`classify_bot_detection` reads).

    So the HTTP context is optional; when omitted, ``status`` is 0, ``headers``
    empty, ``body`` empty, and the message is the given reason.
    """

    guidance = (
        "The site served an automated-access block. Retry later or from a "
        "different IP; a full browser session may be required."
    )

    def __init__(
        self,
        reason: str = "",
        *,
        url: str = "",
        status: int = 0,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.url = url
        self.status = status
        self.headers = headers or {}
        self.body = body
        # Skip FetchError.__init__: this class owns its message (the scraper's
        # reason, or the guidance when none is given), not FetchError's
        # HTTP-status string.
        Exception.__init__(self, reason or self.guidance)

    @classmethod
    def explain(cls, url: str) -> str:
        """A user-facing message: what happened, the specific guidance, the URL.

        Args:
          url: The URL whose fetch was blocked.

        """
        return f"Fetch blocked: {url} -- {cls.guidance}"


class PuzzleChallengeError(BotDetectionError):
    """A solve-a-puzzle wall (reCAPTCHA/hCaptcha) or interactive challenge form."""

    guidance = (
        "The site served a solve-a-puzzle CAPTCHA (reCAPTCHA/hCaptcha or a "
        "challenge form). It needs a human, a CAPTCHA solver, or an "
        "authenticated browser session -- an automated fetch cannot clear it."
    )


class CloudflareChallengeError(BotDetectionError):
    """Cloudflare's managed challenge (``cf_chl`` / ``challenge-platform``)."""

    guidance = (
        "Cloudflare served a managed challenge (Turnstile / 'Just a moment'). "
        "It clears by running the page JS in a real browser engine or from a "
        "cleaner IP -- rotate the egress IP or retry later."
    )


class GoogleSorryError(BotDetectionError):
    """Google's ``/sorry`` interstitial: a refusal, not a solve-this challenge."""

    guidance = (
        "Google served its /sorry page after classifying this IP's traffic as "
        "automated. Rotate the egress IP; it typically clears on its own within "
        "a few hours to ~a day."
    )


class GoogleJavascriptRequiredError(BotDetectionError):
    """Google's ``enablejs`` shell: an HTTP-200 no-results page requiring JS."""

    guidance = (
        "Google served its enablejs shell (an HTTP-200 page with no results that "
        "meta-refreshes to /httpservice/retry/enablejs). Since Jan 2025 Google "
        "enforces JavaScript on Search, so a plain server-side request gets no "
        "results -- render the page in a real browser engine or route through a "
        "backend that does."
    )


def _is_cloudflare_front(headers: dict[str, str]) -> bool:
    """Whether *headers* show a Cloudflare front (not a proxied origin)."""
    # A plain origin error relayed through a CDN keeps the ORIGIN's server name,
    # so requiring a CF front avoids misclassifying it.
    lower_headers = {k.lower(): v.lower() for k, v in headers.items()}
    return (
        "cloudflare" in lower_headers.get("server", "")
        or "cf-ray" in lower_headers
        or "cf-mitigated" in lower_headers
    )


def _text(content: str | bytes) -> str:
    """Lowercase decoded text for marker matching."""
    decoded = (
        content.decode("utf-8", "replace") if isinstance(content, bytes) else content
    )
    return decoded.lower()


def classify_bot_detection(
    content: str | bytes,
    *,
    on_success_body: bool = False,
    puzzle_page: tuple[str, ...] = ("challenge-form", "gs_captcha_f"),
    puzzle_widget: tuple[str, ...] = (
        "g-recaptcha",
        "recaptcha",
        "h-captcha",
        "hcaptcha",
        "data-sitekey",
    ),
    cloudflare: tuple[str, ...] = (
        "cf_chl",
        "cf-challenge",
        "just a moment",
        "cf-turnstile",
        "turnstile",
    ),
    # Cloudflare's ambient "JavaScript Detections" beacon
    # (``/cdn-cgi/challenge-platform/scripts/jsd/main.js``) is injected into
    # EVERY proxied page, served 200s included -- so it corroborates a block only
    # on an ERROR body, never on a success body (mirrors ``puzzle_widget``).
    cloudflare_ambient: tuple[str, ...] = ("challenge-platform",),
    # ``google.com/sorry`` (not a bare ``/sorry/``): the host qualifier avoids
    # misclassifying a benign page that merely links to its own apology path.
    google_sorry: tuple[str, ...] = ("google.com/sorry",),
    # The enablejs shell meta-refreshes to this exact retry path; the full path
    # (not a bare ``enablejs``) avoids misclassifying a page that merely mentions
    # enabling JavaScript.
    google_enablejs: tuple[str, ...] = ("/httpservice/retry/enablejs",),
    generic: tuple[str, ...] = (
        "checking your browser",
        "attention required",
        "security check required",
        "unusual activity from your network",
    ),
) -> type[BotDetectionError] | None:
    """Return the block class ``content`` matches, or ``None`` if it is not one.

    Returns the most specific class the body matches (puzzle, Cloudflare, or
    Google /sorry), else the root :class:`BotDetectionError` when the response is
    clearly a block but the kind is indeterminate. ``None`` for ordinary content.

    The marker groups are load-bearing keyword defaults on this function (NOT
    module state) so the one function that consumes them owns them, and a caller
    can retune any group.

    Args:
      content: The response body.
      on_success_body: Set when classifying a body that arrived with a SUCCESS
        (HTTP 200) status, e.g. a reader-proxy 200. In that mode an embedded
        CAPTCHA WIDGET is NOT a block (a legitimate login/contact page hosts
        one), so only ``puzzle_page`` signatures count. On an error response
        leave it False (a widget in a 403 body IS the block).
      puzzle_page: Whole-page challenge signatures (the page IS a challenge:
        DuckDuckGo ``challenge-form``, Scholar ``gs_captcha_f``).
      puzzle_widget: Embeddable CAPTCHA widgets a legit page hosts; a block only
        on an ERROR body (see ``on_success_body``).
      cloudflare: Cloudflare managed-challenge signatures (the interstitial page
        itself; always a block).
      cloudflare_ambient: Cloudflare's telemetry beacon, injected into normal
        served pages too; a block only on an ERROR body (see ``on_success_body``).
      google_sorry: Google ``/sorry`` refusal.
      google_enablejs: Google ``enablejs`` JavaScript-required shell.
      generic: The response is clearly a block but the kind is indeterminate.

    """
    text = _text(content)
    # A whole-page puzzle challenge (the page IS a challenge form) is
    # unambiguous, so it wins first. Then Cloudflare -- BEFORE the puzzle
    # WIDGET markers, because real Turnstile markup carries ``data-sitekey``
    # (a widget marker) and must classify as Cloudflare (run-JS remedy), not a
    # solve-a-CAPTCHA puzzle. Google /sorry sits between as its own refusal.
    if any(m in text for m in puzzle_page):
        return PuzzleChallengeError
    if any(m in text for m in google_sorry):
        return GoogleSorryError
    if any(m in text for m in google_enablejs):
        return GoogleJavascriptRequiredError
    if any(m in text for m in cloudflare):
        return CloudflareChallengeError
    if not on_success_body and any(m in text for m in cloudflare_ambient):
        return CloudflareChallengeError
    if not on_success_body and any(m in text for m in puzzle_widget):
        return PuzzleChallengeError
    if any(m in text for m in generic):
        return BotDetectionError
    return None


def classify_http_error(
    url: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
    *,
    mitigation_statuses: tuple[int, ...] = (403, 429, 503),
) -> FetchError:
    """Build the most specific :class:`FetchError` a 4xx/5xx response proves.

    Classifies ONCE at the fetch boundary so every consumer's ``except
    FetchError`` sees the right subclass without re-deriving it: a body marker
    (or a Cloudflare front on a mitigation status) yields the matching
    :class:`BotDetectionError` subclass; anything else is a plain
    :class:`FetchError`. Use only for real HTTP-status failures -- a status-0
    connection failure is never a challenge, so those sites construct
    :class:`FetchError` directly.

    ``mitigation_statuses`` (403/429/503) is a load-bearing keyword default: a
    Cloudflare front on one of these is a challenge regardless of body prose (the
    JS challenge and the wordier block page share these codes but differ in text).

    Args:
      url: The requested URL.
      status: The HTTP status code of the failing response.
      headers: The response headers (consulted for a Cloudflare front).
      body: The decompressed response body (scanned for challenge markers).
      mitigation_statuses: Statuses on which a Cloudflare front alone (no body
        marker) is treated as a challenge.

    Returns:
      error: A :class:`BotDetectionError` subclass when a block is identified,
        else a plain :class:`FetchError`.

    """
    cls = classify_bot_detection(body)
    if cls is None and status in mitigation_statuses and _is_cloudflare_front(headers):
        cls = CloudflareChallengeError
    if cls is None:
        return FetchError(url, status, headers, body)
    # BotDetectionError takes the HTTP context by keyword (its first positional is
    # the optional scraper reason string).
    return cls(url=url, status=status, headers=headers, body=body)
