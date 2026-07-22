"""Tests for ``sagent.lib.web.errors``."""

from __future__ import annotations

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    GoogleJavascriptRequiredError,
    GoogleSorryError,
    PuzzleChallengeError,
)


class TestHierarchy:
    def test_leaves_subclass_root(self) -> None:
        assert issubclass(PuzzleChallengeError, BotDetectionError)
        assert issubclass(CloudflareChallengeError, BotDetectionError)
        assert issubclass(GoogleSorryError, BotDetectionError)
        assert issubclass(GoogleJavascriptRequiredError, BotDetectionError)

    def test_leaves_are_distinct(self) -> None:
        assert not issubclass(PuzzleChallengeError, CloudflareChallengeError)
        assert not issubclass(CloudflareChallengeError, GoogleSorryError)
        assert not issubclass(GoogleJavascriptRequiredError, GoogleSorryError)
        assert not issubclass(GoogleSorryError, GoogleJavascriptRequiredError)

    def test_bot_detection_is_fetch_error(self) -> None:
        assert issubclass(BotDetectionError, FetchError)


class TestBotDetectionErrorConstruction:
    def test_http_context_by_keyword(self) -> None:
        err = CloudflareChallengeError(
            url="https://x.com/doc",
            status=403,
            headers={"server": "cloudflare", "cf-ray": "a1"},
            body=b"<title>Just a moment...</title>",
        )
        assert err.url == "https://x.com/doc"
        assert err.status == 403
        assert err.headers == {"server": "cloudflare", "cf-ray": "a1"}
        assert err.body == b"<title>Just a moment...</title>"
        assert "fetch-zendriver https://x.com/doc" in str(err)
        assert "close Chrome" in str(err)

    def test_reason_only(self) -> None:
        err = PuzzleChallengeError("DuckDuckGo returned a challenge form.")
        assert str(err) == "DuckDuckGo returned a challenge form."
        assert err.status == 0
        assert err.headers == {}
        assert err.body == b""

    def test_reasonless_uses_guidance(self) -> None:
        assert str(PuzzleChallengeError()) == PuzzleChallengeError.guidance


class TestGuidance:
    def test_each_class_has_distinct_guidance(self) -> None:
        guidances = {
            PuzzleChallengeError.guidance,
            CloudflareChallengeError.guidance,
            GoogleSorryError.guidance,
            GoogleJavascriptRequiredError.guidance,
            BotDetectionError.guidance,
        }
        assert len(guidances) == 5
        assert "captcha" in PuzzleChallengeError.guidance.lower()
        assert "cloudflare" in CloudflareChallengeError.guidance.lower()
        assert "google" in GoogleSorryError.guidance.lower()
        assert "javascript" in GoogleJavascriptRequiredError.guidance.lower()

    def test_explain_includes_url_and_guidance(self) -> None:
        message = CloudflareChallengeError.explain("https://x.com/doc")
        assert message.startswith("Fetch blocked: https://x.com/doc")
        assert CloudflareChallengeError.guidance in message


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
