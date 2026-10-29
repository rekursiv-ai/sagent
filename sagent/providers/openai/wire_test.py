"""Tests for what the OpenAI wires share: events, user turns, errors, limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import base64
import io
import time
import uuid

from PIL import Image

import pytest

from sagent.model.capability import ModelLimits
from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
)
from sagent.model.types import Request
from sagent.providers.openai import wire
from sagent.providers.types import UsageWindow
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SystemMessage,
    TokenUsage,
    TurnContext,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> float:
    """Freeze the clock at a known instant."""
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    return 1_000.0


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


# ---- events ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_are_each_data_line_until_done() -> None:
    lines = _lines(
        "event: response.created",
        'data: {"type": "a"}',
        "",
        ": keep-alive",
        "data: not json",
        "data: [1, 2]",
        'data:{"type": "b"}',
        "data: [DONE]",
        'data: {"type": "after"}',
    )
    assert [event async for event in wire.events(lines)] == [
        {"type": "a"},
        {"type": "b"},
    ]
    # A body that ends without ``[DONE]`` ends the events too.
    assert [event async for event in wire.events(_lines('data: {"type": "c"}'))] == [
        {"type": "c"},
    ]


# ---- the user side -----------------------------------------------------------


def test_flatten_replaces_each_clear_with_its_summary_and_history() -> None:
    inner = ContextClear(summary="older", history=(UserMessage(content="a"),))
    records = [
        ContextClear(summary="so far", history=(inner, AssistantMessage(content="b"))),
        ContextClear(history=(UserMessage(content="c"),)),
        UserMessage(content="d"),
    ]
    assert list(wire.flatten(records)) == [
        UserMessage(content="so far"),
        UserMessage(content="older"),
        UserMessage(content="a"),
        AssistantMessage(content="b"),
        UserMessage(content="c"),
        UserMessage(content="d"),
    ]


def test_user_content_is_what_each_record_says_on_the_users_side() -> None:
    image = Attachment(mime_descriptor="image/png", data=b"x")
    assert wire.user_content(UserMessage(content="hi", attachments=(image,))) == (
        "hi",
        (image,),
    )
    assert wire.user_content(UserMessage()) == ("", ())
    assert wire.user_content(
        AgentToAgentMessage(content="done", sender="worker", attachments=(image,)),
    ) == ("[from worker]: done", (image,))
    assert wire.user_content(AgentToAgentMessage(content="anon")) == ("anon", ())
    assert wire.user_content(SystemMessage(content="note")) == (
        "<system-reminder>\nnote\n</system-reminder>",
        (),
    )
    assert wire.user_content(TurnContext(cwd="/w")) == (
        "<system-reminder>\nWorking directory: /w\n</system-reminder>",
        (),
    )
    for unsaid in (
        SystemMessage(content=""),
        TurnContext(),
        AssistantMessage(content="mine"),
        TokenUsage(),
    ):
        assert wire.user_content(unsaid) is None


def test_image_url_is_a_data_url_of_the_image_shrunk_to_the_limits() -> None:
    small = _png(4, 2)
    url = wire.image_url(
        Attachment(mime_descriptor="image/png", data=small),
        limits=ModelLimits(),
    )
    assert url == f"data:image/png;base64,{base64.b64encode(small).decode()}"
    shrunk = wire.image_url(
        Attachment(mime_descriptor="image/png", data=_png(40, 20)),
        limits=ModelLimits(max_image_edge_px=10),
    )
    assert shrunk is not None
    head, _, data = shrunk.partition(",")
    assert head == "data:image/webp;base64"
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (10, 5)
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF")
    assert wire.image_url(pdf, limits=ModelLimits()) is None


def test_cache_key_is_one_uuid_per_conversation() -> None:
    opening = Request(system=("Be terse.",), records=(UserMessage(content="fix it"),))
    later = Request(
        system=("Be terse.",),
        records=(
            TurnContext(cwd="/w"),
            UserMessage(content="fix it"),
            AssistantMessage(content="done"),
            UserMessage(content="again"),
        ),
    )
    key = wire.cache_key(opening)
    assert uuid.UUID(key).version == 4
    # The first user-side record opens the conversation, a reminder included.
    assert wire.cache_key(later) != key
    assert (
        wire.cache_key(
            Request(system=("Be terse.",), records=(*opening.records, *later.records)),
        )
        == key
    )
    assert (
        wire.cache_key(Request(system=("Be brief.",), records=opening.records)) != key
    )


# ---- errors ------------------------------------------------------------------


def _error(message: str, *, code: str = "", kind: str = "") -> dict[str, object]:
    return {"error": {"message": message, "code": code, "type": kind}}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (413, "Payload Too Large", RequestTooLargeError),
        # The status alone says it, whatever the text.
        (413, "nope", RequestTooLargeError),
        (400, _error("Request exceeds the maximum request size"), RequestTooLargeError),
        # Some servers send 413 for the token window, which compaction relieves.
        (413, _error("maximum context length is 8192 tokens"), ContextOverflowError),
        (400, _error("x", code="context_length_exceeded"), ContextOverflowError),
        (400, _error("x", code="string_above_max_length"), ContextOverflowError),
        (400, _error("Input exceeds the maximum context"), ContextOverflowError),
        (400, _error("The context window was exceeded"), ContextOverflowError),
        (400, _error("Model context exceeded"), ContextOverflowError),
        (400, _error("x", code="content_policy_violation"), RefusedError),
        (400, _error("x", code="content_filter"), RefusedError),
        (
            400,
            _error("Your prompt was flagged as violating", code="invalid_prompt"),
            RefusedError,
        ),
        (400, _error("Against usage policy", code="invalid_prompt"), RefusedError),
        # An invalid prompt is refused only when it says why.
        (400, _error("Malformed", code="invalid_prompt"), RuntimeError),
        (0, _error("flagged", code="other"), RuntimeError),
        (429, _error("Quota", code="insufficient_quota"), BillingError),
        (400, _error("x", code="billing_hard_limit_reached"), BillingError),
        (400, _error("Plan", kind="usage_not_included"), BillingError),
        (401, _error("Incorrect API key"), AuthError),
        (403, _error("Forbidden"), AuthError),
        (400, _error("x", code="invalid_api_key"), AuthError),
        (0, _error("x", kind="authentication_error"), AuthError),
        (0, _error("x", kind="permission_error"), AuthError),
        (429, _error("Slow down"), RateLimitedError),
        (0, {"type": "error", "code": "rate_limit_exceeded"}, RateLimitedError),
        (0, _error("x", code="rate_limit"), RateLimitedError),
        (0, _error("x", kind="rate_limit_error"), RateLimitedError),
        (0, _error("Plan spent", kind="usage_limit_reached"), RateLimitedError),
        (500, _error("Oops"), OverloadedError),
        (408, "timeout", OverloadedError),
        (409, "conflict", OverloadedError),
        (410, "gone", RuntimeError),
        (0, _error("x", code="server_error"), OverloadedError),
        (0, _error("x", kind="api_error"), OverloadedError),
        (0, _error("x", kind="overloaded_error"), OverloadedError),
        (0, _error("x", code="server_is_overloaded"), OverloadedError),
        (0, _error("Something broke. You can retry."), OverloadedError),
    ],
)
def test_an_error_is_the_most_specific_exception_that_fits(
    status: int,
    body: object,
    expected: type[Exception],
) -> None:
    assert type(wire.to_exception(status, body=body)) is expected


# Each server's own spelling; one missing reached the runtime raw, retried each time.
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("REQUEST_TOO_LARGE", RequestTooLargeError),
        ("Above the maximum request size", RequestTooLargeError),
        ("Too large a number of bytes", RequestTooLargeError),
        ("Payload too large", RequestTooLargeError),
        ("Request entity too large", RequestTooLargeError),
        ("context_length_exceeded", ContextOverflowError),
        ("This model's maximum context length is 8192", ContextOverflowError),
        ("Input too large for the model", ContextOverflowError),
        ("Input too long", ContextOverflowError),
        ("string_above_max_length", ContextOverflowError),
        ("Prompt is too long", ContextOverflowError),
        ("Prompt too long", ContextOverflowError),
        ("input_too_long", ContextOverflowError),
        ("The context window overflowed", ContextOverflowError),
        ("Past the context window maximum", ContextOverflowError),
        ("Model context exceeded", ContextOverflowError),
    ],
)
def test_every_spelling_of_a_limit_names_its_kind(
    message: str,
    expected: type[Exception],
) -> None:
    assert type(wire.to_exception(400, body=_error(message))) is expected


@pytest.mark.parametrize(
    ("status", "message", "code"),
    [
        (413, "request entity too large", ""),
        (400, "context window was exceeded", ""),
        (400, "flagged", "content_filter"),
        (400, "quota", "insufficient_quota"),
        (401, "bad key", ""),
        (429, "slow down", ""),
        (503, "busy", ""),
    ],
)
def test_an_error_carries_the_server_s_message(
    status: int,
    message: str,
    code: str,
) -> None:
    error = wire.to_exception(status, body=_error(message, code=code))
    assert str(error) == message


def test_an_unrecognized_error_names_the_status_code_and_message() -> None:
    error = wire.to_exception(400, body=_error("Bad schema", code="invalid_value"))
    assert type(error) is RuntimeError
    assert str(error) == "The server replied 400 invalid_value: Bad schema"
    assert (
        str(wire.to_exception(404, body="<html>")) == "The server replied 404 : <html>"
    )
    # ``context window`` alone is a schema complaint, not an overflow.
    error = wire.to_exception(400, body=_error("'context window' field is unknown"))
    assert type(error) is RuntimeError


@pytest.mark.parametrize(
    ("headers", "fields", "wait_sec"),
    [
        ({"retry-after-ms": "1500"}, {}, 1.5),
        ({"Retry-After": "7"}, {}, 7.0),
        ({"retry-after": "Thu, 01 Jan 1970 00:16:50 GMT"}, {}, 10.0),
        ({"retry-after": "soon"}, {}, 0.0),
        ({}, {"resets_in_seconds": 90}, 90.0),
        ({}, {"resets_at": 1_060}, 60.0),
        (
            {"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s"},
            {},
            360.0,
        ),
        ({"x-ratelimit-reset-requests": "2s"}, {}, 2.0),
        ({"retry-after": "999999"}, {}, 691_200.0),
        ({}, {"resets_in_seconds": 259_200}, 259_200.0),
        ({}, {"resets_at": 10}, 0.0),
        ({}, {}, 0.0),
    ],
)
def test_a_rate_limit_carries_the_advised_wait(
    now: float,
    headers: dict[str, str],
    fields: dict[str, object],
    wait_sec: float,
) -> None:
    del now
    body = {"error": {"message": "slow", "type": "usage_limit_reached", **fields}}
    error = wire.to_exception(429, body=body, headers=headers)
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_sec == wait_sec


@pytest.mark.parametrize(
    ("used", "limit"),
    [
        ({"primary": "100", "secondary": "12.5"}, "5-hour"),
        ({"primary": "40", "secondary": "100"}, "weekly"),
        ({"primary": "40", "secondary": "60"}, ""),
    ],
)
def test_a_chatgpt_logins_spent_window_is_named(
    used: dict[str, str],
    limit: str,
) -> None:
    headers = {
        "x-codex-primary-window-minutes": "300",
        "x-codex-primary-used-percent": used["primary"],
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-used-percent": used["secondary"],
    }
    body = {"error": {"message": "limit", "type": "usage_limit_reached"}}
    error = wire.to_exception(429, body=body, headers=headers)
    assert isinstance(error, RateLimitedError)
    assert error.limit == limit


def test_an_overload_carries_the_advised_wait() -> None:
    error = wire.to_exception(503, body="busy", headers={"retry-after": "3"})
    assert isinstance(error, OverloadedError)
    assert error.retry_after_sec == 3.0


# ---- rate limits -------------------------------------------------------------


def test_rate_limits_are_an_api_keys_request_and_token_windows(now: float) -> None:
    headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "25",
        "x-ratelimit-reset-requests": "1m30s",
        "X-RateLimit-Limit-Tokens": "1000",
        "X-RateLimit-Remaining-Tokens": "0",
        "X-RateLimit-Reset-Tokens": "500ms",
    }
    assert wire.rate_limits(headers) == (
        UsageWindow(label="requests", utilization=0.75, resets_at=now + 90.0),
        UsageWindow(label="tokens", utilization=1.0, resets_at=now + 0.5, blocked=True),
    )
    # A window reported without a limit has no utilization to compute.
    assert wire.rate_limits({"x-ratelimit-reset-tokens": "2"}) == (
        UsageWindow(label="tokens", resets_at=now + 2.0),
    )
    assert wire.rate_limits({"x-ratelimit-reset-tokens": "1h1d"}) == (
        UsageWindow(label="tokens", resets_at=now + 90_000.0),
    )
    assert wire.rate_limits({"x-ratelimit-reset-tokens": "never"}) == ()
    assert wire.rate_limits({}) == ()


def test_rate_limits_are_a_chatgpt_logins_windows_by_span(now: float) -> None:
    headers = {
        "x-codex-primary-window-minutes": "10080",
        "x-codex-primary-used-percent": "100",
        "x-codex-primary-reset-at": "5000",
        "x-codex-secondary-window-minutes": "300",
        "x-codex-secondary-used-percent": "12.5",
        "x-codex-secondary-reset-after-seconds": "60",
    }
    assert wire.rate_limits(headers) == (
        UsageWindow(label="7d", utilization=1.0, resets_at=5000.0, blocked=True),
        UsageWindow(label="5h", utilization=0.125, resets_at=now + 60.0),
    )
    odd = {
        "x-codex-primary-window-minutes": "90",
        "x-codex-secondary-window-minutes": "0",
    }
    assert wire.rate_limits(odd) == (UsageWindow(label="90m"),)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
