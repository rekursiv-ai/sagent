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
- :class:`CloudflareChallengeError` -- Cloudflare's managed challenge, detected by
  its challenge-only DOM markup (``#cf-challenge-running`` / ``#turnstile-wrapper``
  / ``cf-turnstile`` / the ``/cdn-cgi/challenge-platform/`` script) or a block
  ``<title>`` (``Just a moment...``). Its mode (checkbox / puzzle / invisible-JS)
  is decided client-side in the served JS, so it is not further splittable.
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

Its Cloudflare challenge markers (the challenge-only DOM selectors and the block
``<title>`` values) are ported from FlareSolverr's ``CHALLENGE_SELECTORS`` /
``CHALLENGE_TITLES`` / ``ACCESS_DENIED_TITLES`` -- the same STRUCTURAL model (a
challenge is challenge-only markup or a block title, never a keyword in body
prose), which is why a real page that merely mentions "turnstile"/"just a
moment" is not misclassified.

    FlareSolverr -- MIT License, Copyright (c) 2025 Diego Heras (ngosang).
    https://github.com/FlareSolverr/FlareSolverr
    src/flaresolverr_service.py @ 237faf1730e7de4d126532a75b9ac16bd5f7539b
"""

from __future__ import annotations

import re


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


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL)


def _page_title(text: str) -> str | None:
    """The stripped text of the first ``<title>`` in ``text`` (already lowered).

    Title matching (not body-substring) is how a block ``<title>`` is told from a
    real page that merely mentions the phrase -- FlareSolverr checks ``<title>``
    exactly for the same reason.
    """
    match = _TITLE_RE.search(text)
    return match.group(1).strip() if match is not None else None


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
    # STRUCTURAL Cloudflare challenge markers: id/class/name tokens the challenge
    # interstitial injects, which never appear in ordinary page prose. Ported
    # from FlareSolverr's CHALLENGE_SELECTORS (MIT; see module note). Matched as
    # substrings because each is a distinctive HTML attribute value -- the bare
    # word "turnstile" is DELIBERATELY absent (it appears in real content, e.g. a
    # README about Turnstile); only the widget markup ``cf-turnstile`` /
    # ``cf-turnstile-response`` / ``turnstile-wrapper`` counts.
    cloudflare: tuple[str, ...] = (
        "cf_chl",
        "cf-challenge",
        "cf-challenge-running",
        "cf-please-wait",
        "challenge-spinner",
        "trk_jschal_js",
        "turnstile-wrapper",
        "cf-turnstile",
        "cf-turnstile-response",
    ),
    # Cloudflare's ambient "JavaScript Detections" beacon
    # (``/cdn-cgi/challenge-platform/scripts/jsd/main.js``) is injected into EVERY
    # proxied page, 200s included, so it corroborates a block ONLY on an error
    # body -- never a success body (a real article carries it too).
    cloudflare_ambient: tuple[str, ...] = ("/cdn-cgi/challenge-platform/",),
    # TITLE markers: matched against the page <title> ONLY (not body prose), per
    # FlareSolverr's CHALLENGE_TITLES/ACCESS_DENIED_TITLES. A real page whose body
    # merely contains "just a moment" or "attention required" is not a challenge;
    # the interstitial's whole <title> IS one of these.
    cloudflare_titles: tuple[str, ...] = (
        "just a moment...",
        "attention required! | cloudflare",
        "access denied",
    ),
    # ``google.com/sorry`` (not a bare ``/sorry/``): the host qualifier avoids
    # misclassifying a benign page that merely links to its own apology path.
    google_sorry: tuple[str, ...] = ("google.com/sorry",),
    # The enablejs shell meta-refreshes to this exact retry path; the full path
    # (not a bare ``enablejs``) avoids misclassifying a page that merely mentions
    # enabling JavaScript.
    google_enablejs: tuple[str, ...] = ("/httpservice/retry/enablejs",),
) -> type[BotDetectionError] | None:
    """Return the block class ``content`` matches, or ``None`` if it is not one.

    Returns the most specific class the body matches (puzzle, Cloudflare, or
    Google /sorry), or ``None`` for ordinary content. A prose-only block with no
    structural marker is not classifiable from the body alone -- the fetch
    boundary catches it by error status + Cloudflare-front headers instead (see
    :func:`classify_http_error`).

    The marker groups are load-bearing keyword defaults on this function (NOT
    module state) so the one function that consumes them owns them, and a caller
    can retune any group.

    Detection mirrors FlareSolverr's STRUCTURAL model (MIT-licensed; ported --
    see the module-level note): a challenge is identified by challenge-only DOM
    markup (id/class/name tokens the interstitial injects) or by the page
    ``<title>`` being a known block title -- NOT by a challenge-related WORD
    appearing anywhere in body prose. This is why a real 405 KB page whose
    content merely mentions "turnstile" or "just a moment" is ordinary content:
    it carries no challenge DOM markup and its ``<title>`` is its own.

    Args:
      content: The response body.
      on_success_body: Set when classifying a body that arrived with a SUCCESS
        (HTTP 200) status, e.g. a reader-proxy 200. In that mode an embedded
        CAPTCHA WIDGET is NOT a block (a legitimate login/contact page hosts
        one), so only whole-page/structural signatures count. On an error
        response leave it False (a widget in a 403 body IS the block).
      puzzle_page: Whole-page challenge signatures (the page IS a challenge:
        DuckDuckGo ``challenge-form``, Scholar ``gs_captcha_f``).
      puzzle_widget: Embeddable CAPTCHA widgets a legit page hosts; a block only
        on an ERROR body (see ``on_success_body``).
      cloudflare: Structural Cloudflare challenge markers (challenge-only DOM
        id/class/name tokens); always a block, on any status.
      cloudflare_titles: Whole-page block ``<title>`` values (matched against the
        page title only, never body prose).
      cloudflare_ambient: Cloudflare's JS-Detections beacon path, present on
        normal pages too; a block only on an ERROR body (see ``on_success_body``).
      google_sorry: Google ``/sorry`` refusal.
      google_enablejs: Google ``enablejs`` JavaScript-required shell.

    """
    text = _text(content)
    title = _page_title(text)
    # Google's own refusals are host/path-qualified (unambiguous), so they win
    # first on any status. Then the page <title>: a whole-page block title is
    # sound on ANY status -- a real page's title is its own, and a reader-proxy
    # 200 of a challenge still carries "Just a moment...".
    if any(m in text for m in google_sorry):
        return GoogleSorryError
    if any(m in text for m in google_enablejs):
        return GoogleJavascriptRequiredError
    if title is not None and any(title.startswith(t) for t in cloudflare_titles):
        return CloudflareChallengeError
    # A whole-page challenge FORM (the page IS the challenge: DuckDuckGo's
    # ``challenge-form``, Scholar's ``gs_captcha_f``) is a specific form id, not a
    # token a page documents in passing -- sound on any status.
    if any(m in text for m in puzzle_page):
        return PuzzleChallengeError
    # Everything below matches challenge MARKUP as a substring, which a page that
    # merely DOCUMENTS the markup (a wiki about Cloudflare selectors, a README)
    # also contains. On a SUCCESS body that ambiguity is unacceptable -- a real
    # challenge arrives as an error status (403/429/503) or as the title above,
    # so these fire only on an error body (``on_success_body`` False). This
    # mirrors FlareSolverr, which queries the live DOM (a doc page has no such
    # ELEMENT); lacking a DOM here, restricting to error bodies is the sound
    # approximation.
    if not on_success_body:
        # Cloudflare BEFORE the puzzle widget: real Turnstile markup carries
        # ``data-sitekey`` (a widget marker) but its remedy is run-JS/rotate-IP,
        # not solve-a-CAPTCHA, so it must classify as Cloudflare.
        if any(m in text for m in cloudflare):
            return CloudflareChallengeError
        if any(m in text for m in cloudflare_ambient):
            return CloudflareChallengeError
        if any(m in text for m in puzzle_widget):
            return PuzzleChallengeError
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
