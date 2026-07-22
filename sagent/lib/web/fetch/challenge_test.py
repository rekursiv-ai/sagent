"""Tests for cross-site challenge detection."""

from __future__ import annotations

import pytest

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)
from sagent.lib.web.fetch.challenge import classify_challenge, classify_http_error


@pytest.mark.parametrize("marker", ["g-recaptcha", "h-captcha", "data-sitekey"])
def test_error_body_captcha_widget_is_puzzle(marker: str) -> None:
    assert classify_challenge(f'<div class="{marker}"></div>') is PuzzleChallengeError


def test_success_body_captcha_widget_is_content() -> None:
    body = '<form><div class="g-recaptcha" data-sitekey="x"></div></form>'
    assert classify_challenge(body, on_success_body=True) is None


def test_cloudflare_challenge_markup() -> None:
    body = '<script>window._cf_chl_opt={};</script><div class="challenge-platform">'
    assert classify_challenge(body) is CloudflareChallengeError


def test_turnstile_wins_over_generic_widget() -> None:
    body = '<div class="cf-turnstile" data-sitekey="0x4AAA"></div>'
    assert classify_challenge(body) is CloudflareChallengeError


@pytest.mark.parametrize(
    "markup",
    [
        '<div id="cf-challenge-running"></div>',
        '<div id="challenge-spinner"></div>',
        '<div id="cf-please-wait"></div>',
        '<div id="turnstile-wrapper"></div>',
        "<input name='cf-turnstile-response' value='x'>",
    ],
)
def test_cloudflare_structural_markup_on_success(markup: str) -> None:
    assert classify_challenge(markup, on_success_body=True) is CloudflareChallengeError


def test_cloudflare_title_on_success() -> None:
    assert (
        classify_challenge("<title>Just a moment...</title>", on_success_body=True)
        is CloudflareChallengeError
    )


def test_title_starting_with_challenge_phrase_is_content() -> None:
    body = "<title>Access Denied: A History of Firewalls</title>"
    assert classify_challenge(body, on_success_body=True) is None


def test_challenge_words_in_prose_are_content() -> None:
    body = (
        "<html><head><title>Blog</title></head><body>"
        "Please wait just a moment while the turnstile loads.</body></html>"
    )
    assert classify_challenge(body, on_success_body=True) is None


def test_page_documenting_cloudflare_selectors_is_content() -> None:
    body = (
        "<html><head><title>FlareSolverr | DeepWiki</title></head><body>"
        "Detection checks '#cf-challenge-running' and '#turnstile-wrapper'."
        "</body></html>"
    )
    assert classify_challenge(body, on_success_body=True) is None


def test_cloudflare_ambient_beacon_only_counts_on_error() -> None:
    body = "<script src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'></script>"
    assert classify_challenge(body, on_success_body=True) is None
    assert classify_challenge(body) is CloudflareChallengeError


def test_marker_groups_are_retunable() -> None:
    assert classify_challenge("<div class='widgetguard-v2'></div>") is None
    assert (
        classify_challenge(
            "<div class='widgetguard-v2'></div>",
            cloudflare=("widgetguard-v2",),
        )
        is CloudflareChallengeError
    )


def test_provider_markers_are_not_shared_challenges() -> None:
    for body in (
        "<form id='challenge-form'></form>",
        "<form id='gs_captcha_f'></form>",
        "https://www.google.com/sorry/index",
        "<meta content='0;url=/httpservice/retry/enablejs'>",
    ):
        assert classify_challenge(body) is None


def test_http_error_uses_body_challenge() -> None:
    error = classify_http_error(
        "https://x.com", 403, {}, b'<div class="g-recaptcha"></div>'
    )
    assert isinstance(error, PuzzleChallengeError)


def test_http_error_uses_cloudflare_front_on_mitigation_status() -> None:
    error = classify_http_error(
        "https://x.com",
        403,
        {"server": "cloudflare", "cf-ray": "a1"},
        b"Temporarily unavailable",
    )
    assert isinstance(error, CloudflareChallengeError)


def test_http_error_preserves_plain_failure() -> None:
    error = classify_http_error("https://x.com", 404, {"server": "nginx"}, b"Not found")
    assert type(error) is FetchError
    assert not isinstance(error, BotDetectionError)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
