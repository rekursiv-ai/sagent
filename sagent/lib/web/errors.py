"""HTTP fetch errors and automated-access error taxonomy."""

from __future__ import annotations

import shlex


__all__ = [
    "BotDetectionError",
    "CloudflareChallengeError",
    "FetchError",
    "GoogleJavascriptRequiredError",
    "GoogleSorryError",
    "PuzzleChallengeError",
]


class FetchError(Exception):
    """HTTP request returned a non-success status code.

    Attributes:
      url: Requested URL.
      status: HTTP status code, or zero when no response was received.
      headers: Response headers.
      body: Response body bytes.

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
        if status == 0:
            reason = body.decode("utf-8", "replace").strip() or "connection failed"
            super().__init__(f"connection failed: {url}: {reason}")
        else:
            super().__init__(f"HTTP {status}: {url}")


class BotDetectionError(FetchError):
    """A site served an automated-access challenge instead of content."""

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
        message = reason or self.guidance
        if url:
            message = f"{message} {self.recovery(url)}"
        Exception.__init__(self, message)

    @classmethod
    def recovery(cls, url: str) -> str:
        """Return the interactive-browser recovery instruction for ``url``."""
        return (
            f"Run `fetch-zendriver {shlex.quote(url)}`, solve the challenge, "
            "then close Chrome to clear this domain's cooldown."
        )

    @classmethod
    def explain(cls, url: str) -> str:
        """Return actionable guidance for a blocked URL.

        Args:
          url: URL whose fetch was blocked.

        Returns:
          message: User-facing explanation.

        """
        return f"Fetch blocked: {url} -- {cls.guidance} {cls.recovery(url)}"


class PuzzleChallengeError(BotDetectionError):
    """A solve-a-puzzle CAPTCHA or interactive challenge form."""

    guidance = (
        "The site served a solve-a-puzzle CAPTCHA (reCAPTCHA/hCaptcha or a "
        "challenge form). It needs a human, a CAPTCHA solver, or an "
        "authenticated browser session -- an automated fetch cannot clear it."
    )


class CloudflareChallengeError(BotDetectionError):
    """Cloudflare served a managed challenge."""

    guidance = (
        "Cloudflare served a managed challenge (Turnstile / 'Just a moment'). "
        "It clears by running the page JS in a real browser engine or from a "
        "cleaner IP -- rotate the egress IP or retry later."
    )


class GoogleSorryError(BotDetectionError):
    """Google served its automated-traffic refusal page."""

    guidance = (
        "Google served its /sorry page after classifying this IP's traffic as "
        "automated. Rotate the egress IP; it typically clears on its own within "
        "a few hours to ~a day."
    )


class GoogleJavascriptRequiredError(BotDetectionError):
    """Google Search served its JavaScript-required shell."""

    guidance = (
        "Google served its enablejs shell (an HTTP-200 page with no results that "
        "meta-refreshes to /httpservice/retry/enablejs). Since Jan 2025 Google "
        "enforces JavaScript on Search, so a plain server-side request gets no "
        "results -- render the page in a real browser engine or route through a "
        "backend that does."
    )
