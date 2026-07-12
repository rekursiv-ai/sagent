"""Tests for sagent.lib.web.errors bot-flag taxonomy and classifier."""

from __future__ import annotations

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    GoogleJavascriptRequiredError,
    GoogleSorryError,
    PuzzleChallengeError,
    classify_bot_detection,
    classify_http_error,
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
        # enablejs (JS-required wall) is not /sorry (IP refusal): distinct kinds.
        assert not issubclass(GoogleJavascriptRequiredError, GoogleSorryError)
        assert not issubclass(GoogleSorryError, GoogleJavascriptRequiredError)

    def test_bot_flagged_is_a_fetch_error(self) -> None:
        # The core of the refactor: a BotDetectionError is-a FetchError, so every
        # existing ``except FetchError`` catches it and inherits its attributes.
        assert issubclass(BotDetectionError, FetchError)
        assert issubclass(CloudflareChallengeError, FetchError)
        assert issubclass(PuzzleChallengeError, FetchError)


class TestBotDetectionErrorConstruction:
    def test_boundary_mode_carries_http_context_by_keyword(self) -> None:
        # The fetch() boundary builds it with the full HTTP quadruple by keyword.
        err = CloudflareChallengeError(
            url="https://x.com/doc",
            status=403,
            headers={"server": "cloudflare", "cf-ray": "a1"},
            body=b"<title>Just a moment...</title>",
        )
        assert isinstance(err, FetchError)  # every except FetchError catches it
        assert err.url == "https://x.com/doc"
        assert err.status == 403
        assert err.headers == {"server": "cloudflare", "cf-ray": "a1"}
        assert err.body == b"<title>Just a moment...</title>"
        assert "cloudflare" in err.guidance.lower()
        assert err.explain("https://x.com/doc").startswith(
            "Fetch blocked: https://x.com/doc"
        )

    def test_scraper_mode_takes_a_reason_string_only(self) -> None:
        # A scraper that spots a challenge mid-parse has only a reason -- it must
        # NOT be forced to fabricate status/headers or stuff the message into
        # body (body is response bytes classify_bot_detection reads).
        err = PuzzleChallengeError("DuckDuckGo returned a challenge form.")
        assert isinstance(err, FetchError)
        assert str(err) == "DuckDuckGo returned a challenge form."
        assert err.status == 0
        assert err.headers == {}
        assert err.body == b""

    def test_reasonless_construction_falls_back_to_guidance(self) -> None:
        assert str(PuzzleChallengeError()) == PuzzleChallengeError.guidance


class TestClassifyHttpError:
    def test_cloudflare_body_yields_cloudflare_subclass(self) -> None:
        err = classify_http_error(
            "https://x.com",
            403,
            {"server": "cloudflare", "cf-ray": "a1"},
            b'<div class="challenge-platform"></div>',
        )
        assert isinstance(err, CloudflareChallengeError)
        assert err.status == 403

    def test_recaptcha_body_yields_puzzle_subclass(self) -> None:
        err = classify_http_error(
            "https://x.com", 403, {}, b'<div class="g-recaptcha"></div>'
        )
        assert isinstance(err, PuzzleChallengeError)

    def test_cf_front_mitigation_status_without_body_marker_is_cloudflare(self) -> None:
        # A CF block page with NO challenge-JS marker but a CF front + 403 is
        # still a mitigation -> CloudflareChallengeError (mirrors is_challenge).
        err = classify_http_error(
            "https://x.com",
            403,
            {"server": "cloudflare", "cf-ray": "a1"},
            b"<title>Temporarily Unavailable</title>",
        )
        assert isinstance(err, CloudflareChallengeError)

    def test_genuine_404_is_plain_fetch_error(self) -> None:
        err = classify_http_error(
            "https://x.com", 404, {"server": "nginx"}, b"<h1>Not Found</h1>"
        )
        assert type(err) is FetchError
        assert not isinstance(err, BotDetectionError)

    def test_non_cf_403_without_markers_is_plain_fetch_error(self) -> None:
        # A 403 that is NOT CF-fronted and carries no body marker is a plain
        # origin refusal, not a bot flag -- must not be misclassified.
        err = classify_http_error("https://x.com", 403, {"server": "nginx"}, b"denied")
        assert type(err) is FetchError


class TestClassifyBotFlag:
    def test_recaptcha_is_puzzle(self) -> None:
        body = '<div class="g-recaptcha" data-sitekey="x"></div>'
        assert classify_bot_detection(body) is PuzzleChallengeError

    def test_hcaptcha_is_puzzle(self) -> None:
        assert classify_bot_detection("<div class=h-captcha>") is PuzzleChallengeError

    def test_ddg_challenge_form_is_puzzle(self) -> None:
        assert (
            classify_bot_detection("<form id='challenge-form'>") is PuzzleChallengeError
        )

    def test_scholar_captcha_form_is_puzzle(self) -> None:
        assert (
            classify_bot_detection("<form id='gs_captcha_f'>") is PuzzleChallengeError
        )

    def test_cloudflare_managed_challenge(self) -> None:
        # Verified live on ResearchGate's 403 (cf_chl / challenge-platform, no
        # captcha markers). Detectable as Cloudflare; mode is client-side.
        body = (
            b"<html><head><title>Just a moment...</title></head>"
            b"<body><script>window._cf_chl_opt={};</script>"
            b'<div class="challenge-platform"></div></body></html>'
        )
        assert classify_bot_detection(body) is CloudflareChallengeError

    def test_turnstile_is_cloudflare(self) -> None:
        assert classify_bot_detection("<div class=cf-turnstile></div>") is (
            CloudflareChallengeError
        )

    def test_turnstile_with_data_sitekey_is_cloudflare_not_puzzle(self) -> None:
        # REV2061-004: real Turnstile markup carries data-sitekey (a puzzle-
        # widget marker). Cloudflare must win over the generic puzzle-widget so
        # the guidance is the CF remedy (run JS / rotate IP), not "solve a
        # CAPTCHA". Requires classifying Cloudflare before the widget markers.
        body = '<div class="cf-turnstile" data-sitekey="0x4AAA"></div>'
        assert classify_bot_detection(body) is CloudflareChallengeError

    def test_marker_groups_are_retunable_via_kwargs(self) -> None:
        # The marker groups are load-bearing keyword defaults (not module
        # globals): a caller can retune any group. Prove the contract -- a body
        # matching no default marker is classified once a custom marker is added.
        assert classify_bot_detection("widgetguard-v2 active") is None
        assert (
            classify_bot_detection(
                "widgetguard-v2 active", cloudflare=("widgetguard-v2",)
            )
            is CloudflareChallengeError
        )

    def test_google_sorry_is_sorry(self) -> None:
        # A genuine Google /sorry reference carries the google.com host.
        assert (
            classify_bot_detection("redirected to https://www.google.com/sorry/index")
            is GoogleSorryError
        )

    def test_enablejs_shell_is_javascript_required(self) -> None:
        # Google's HTTP-200 enablejs shell meta-refreshes to this exact path.
        body = (
            '<noscript><meta http-equiv="refresh" '
            'content="0;url=/httpservice/retry/enablejs?sei=abc"></noscript>'
        )
        assert classify_bot_detection(body) is GoogleJavascriptRequiredError

    def test_enablejs_is_not_google_sorry(self) -> None:
        # The JS-required wall must NOT be misread as an IP /sorry ban -- the
        # remedy differs (render JS vs rotate IP).
        body = "<meta content='0;url=/httpservice/retry/enablejs'>"
        assert classify_bot_detection(body) is not GoogleSorryError

    def test_benign_enablejs_mention_is_not_flagged(self) -> None:
        # A page that merely tells a user to enable JavaScript, without the
        # retry-service path, is ordinary content.
        assert classify_bot_detection("Please enable JavaScript to continue.") is None

    def test_benign_sorry_path_is_not_google_sorry(self) -> None:
        # O-SPEC-1: a bare "/sorry/" substring is too weak -- a benign page that
        # links to its own apology page (example.com/sorry/) must NOT be
        # misclassified as Google's bot interstitial.
        body = "<html>Read our apology at example.com/sorry/ for the outage.</html>"
        assert classify_bot_detection(body) is None

    def test_generic_block_is_root(self) -> None:
        # A wall with no kind-specific marker -> the root, never a wrong leaf.
        assert classify_bot_detection("Security check required. Ray ID: abc") is (
            BotDetectionError
        )

    def test_real_content_is_none(self) -> None:
        assert classify_bot_detection("<h1>Build isolation</h1> uv builds ...") is None

    def test_empty_is_none(self) -> None:
        assert classify_bot_detection("") is None

    def test_accepts_bytes_and_str(self) -> None:
        assert classify_bot_detection(b"g-recaptcha") is PuzzleChallengeError
        assert classify_bot_detection("g-recaptcha") is PuzzleChallengeError


class TestGuidance:
    def test_each_class_has_distinct_actionable_guidance(self) -> None:
        # The point of the taxonomy for UX: each kind carries its OWN specific,
        # actionable guidance -- not one generic string. A caller that renders
        # the error must be able to tell a solvable puzzle from an IP-gated
        # Cloudflare block from a Google /sorry ban.
        guidances = {
            PuzzleChallengeError.guidance,
            CloudflareChallengeError.guidance,
            GoogleSorryError.guidance,
            GoogleJavascriptRequiredError.guidance,
            BotDetectionError.guidance,
        }
        assert len(guidances) == 5  # all distinct
        # each names its mechanism/remedy
        assert "captcha" in PuzzleChallengeError.guidance.lower()
        assert "cloudflare" in CloudflareChallengeError.guidance.lower()
        assert "google" in GoogleSorryError.guidance.lower()
        assert "javascript" in GoogleJavascriptRequiredError.guidance.lower()

    def test_explain_includes_url_and_guidance(self) -> None:
        msg = CloudflareChallengeError.explain("https://x.com/doc")
        assert msg.startswith("Fetch blocked: https://x.com/doc")
        assert CloudflareChallengeError.guidance in msg

    def test_explain_is_classmethod_usable_on_the_type(self) -> None:
        # classify_bot_detection returns a TYPE; explain must work on it directly.
        cls = classify_bot_detection("g-recaptcha")
        assert cls is not None
        # explain() renders "Fetch blocked: <url> -- <guidance>"; assert the URL
        # lands in the rendered slot (exact prefix, not a substring membership).
        assert cls.explain("https://y.com").startswith("Fetch blocked: https://y.com")


class TestOnSuccessBodyMode:
    def test_embedded_widget_not_flagged_on_success_body(self) -> None:
        # DRV-1: an embedded reCAPTCHA widget on a 200 page is not a block.
        body = '<div class="g-recaptcha" data-sitekey="x"></div>'
        assert classify_bot_detection(body, on_success_body=True) is None
        # But on an error body it IS the block (default mode).
        assert classify_bot_detection(body) is PuzzleChallengeError

    def test_whole_page_challenge_flagged_even_on_success_body(self) -> None:
        assert (
            classify_bot_detection("<form id='challenge-form'>", on_success_body=True)
            is PuzzleChallengeError
        )

    def test_cloudflare_still_flagged_on_success_body(self) -> None:
        assert (
            classify_bot_detection("just a moment", on_success_body=True)
            is CloudflareChallengeError
        )

    def test_challenge_platform_beacon_not_flagged_on_success_body(self) -> None:
        # A real Cloudflare-fronted 200 page carries the ambient JS-Detections
        # beacon (/cdn-cgi/challenge-platform/scripts/jsd/main.js) in its markup;
        # it rides EVERY proxied page, so on a success body it is NOT a block.
        # (thecompassforsbc.org served its article this way.)
        body = (
            b"<!doctype html><html><body>real article"
            b"<script>a.src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'"
            b"</script></body></html>"
        )
        assert classify_bot_detection(body, on_success_body=True) is None
        # But the same beacon on an ERROR body still corroborates the block.
        assert classify_bot_detection(body) is CloudflareChallengeError


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
