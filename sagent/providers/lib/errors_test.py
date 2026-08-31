"""Tests for ``providers.lib.errors``: shared provider-boundary helpers."""

from __future__ import annotations

import json

import httpx2
import pytest

from sagent.providers.lib.errors import (
    PER_ITEM_STRING_CAP_BODY,
    StreamingResponseNotReadError,
    error_status_code,
    find_response_not_read,
    is_context_overflow_text,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.types.model import RequestTooLargeError


def test_context_overflow_survives_a_non_string_error_code() -> None:
    """A malformed vendor body must classify, not crash.

    ``error.code`` is whatever the vendor sent. Testing membership of an
    unhashable value against a ``frozenset`` raises ``TypeError``, and
    this runs at the provider's exception boundary -- so a single
    malformed body would replace the real error with a crash inside the
    handler meant to classify it.
    """
    for body in (
        '{"error":{"code":{}}}',
        '{"error":{"code":[]}}',
        '{"error":{"code":1}}',
    ):
        assert is_context_overflow_text(body) is False, body


def test_find_response_not_read_searches_both_chain_branches() -> None:
    """``__cause__`` and ``__context__`` are a tree, not a list.

    The docstring promises both are walked. Following ``cause or
    context`` takes only one branch, so a ``ResponseNotRead`` reachable
    through the context of an exception that also has a cause is missed
    -- and the SDK produces exactly that shape when a handler raises
    while formatting a hidden body.
    """
    root = RuntimeError("outer")
    root.__cause__ = ValueError("decoy branch")
    target = httpx2.ResponseNotRead()
    root.__context__ = target
    assert find_response_not_read(root) is target


def test_find_response_not_read_direct() -> None:
    err = httpx2.ResponseNotRead()
    assert find_response_not_read(err) is err


def test_find_response_not_read_via_cause() -> None:
    inner = httpx2.ResponseNotRead()
    outer = RuntimeError("formatting failed")
    outer.__cause__ = inner
    assert find_response_not_read(outer) is inner


def test_find_response_not_read_via_context() -> None:
    inner = httpx2.ResponseNotRead()
    outer = RuntimeError("formatting failed")
    outer.__context__ = inner
    assert find_response_not_read(outer) is inner


def test_find_response_not_read_absent_returns_none() -> None:
    assert find_response_not_read(RuntimeError("unrelated")) is None


def test_find_response_not_read_handles_cyclic_chain() -> None:
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__context__ = a
    assert find_response_not_read(a) is None


def test_streaming_response_not_read_error_chains_its_raise_cause() -> None:
    """``raise ... from`` sets the cause; the class must not fight it.

    A constructor that also assigned ``__cause__`` always lost to the
    ``from`` clause every call site uses, so the kwarg was dead weight.
    """
    original = httpx2.ResponseNotRead()

    def _raise() -> None:
        raise StreamingResponseNotReadError(provider_name="Anthropic") from original

    with pytest.raises(StreamingResponseNotReadError) as raised:
        _raise()
    assert raised.value.__cause__ is original
    assert "Anthropic" in str(raised.value)


def test_is_request_too_large_status_413() -> None:
    """A bare 413 defaults to byte-limit when the body is not token-context."""
    assert is_request_too_large(413, "anything") is True


def test_is_request_too_large_body_phrases() -> None:
    """Status-less variants are caught by body phrase."""
    for body in (
        '{"error": {"type": "request_too_large"}}',
        "The maximum request size is 32 MB",
        "Request exceeds the maximum allowed number of bytes",
        "413 Payload Too Large",
        "Request Entity Too Large",
    ):
        assert is_request_too_large(None, body) is True, body


def test_is_request_too_large_false_for_token_overflow() -> None:
    """Token context-window overflow is a different condition."""
    assert is_request_too_large(400, "prompt is too long") is False
    assert is_request_too_large(400, "input too large for model context") is False
    assert is_request_too_large(400, "maximum context length exceeded") is False


def test_is_request_too_large_false_for_unrelated() -> None:
    assert is_request_too_large(None, "nope") is False
    assert is_request_too_large(500, "internal error") is False


def test_ambiguous_byte_prefix_does_not_force_byte_classification() -> None:
    """A body whose only "byte" signal is an ambiguous prefix stays token.

    "request exceeds the maximum context length" and "Request too large: ...
    context window" are TOKEN overflows (a larger window relieves them). The
    classifier no longer treats the loose prefixes "request exceeds the
    maximum" / "request too large" as byte signals -- only unambiguous
    request-size phrases qualify -- so the token context phrase decides.
    """
    assert (
        is_request_too_large(
            413, "request exceeds the maximum context length for this model"
        )
        is False
    )
    assert (
        is_request_too_large(
            400, "Request too large: prompt exceeds the model context window"
        )
        is False
    )


def test_unambiguous_byte_phrase_wins_over_co_occurring_context_phrase() -> None:
    """An unambiguous byte phrase classifies as byte even with a context phrase.

    A 413 "request entity too large" whose remediation prose mentions the
    context window is still a BYTE overflow -- it must route to byte-overflow
    recovery, not a ``/model`` larger-window suggestion the byte ceiling
    ignores.
    """
    assert (
        is_request_too_large(
            413, "request entity too large; please reduce context window usage"
        )
        is True
    )


def test_context_overflow_text_false_positive_tools_schema_validation() -> None:
    """Tool-schema validation errors mention 'model context' benignly."""
    msg = "Provider rejected: 'model context' field missing in tools schema"
    assert is_context_overflow_text(msg) is False


def test_context_overflow_text_structured_body_canonical_code() -> None:
    """``error.code == 'context_length_exceeded'`` is the canonical signal."""
    body = json.dumps(
        {
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 128000 tokens.",
            }
        }
    )
    assert is_context_overflow_text(body) is True


def test_context_overflow_text_structured_body_unrelated_code() -> None:
    """Structured error with unrelated code must not classify as overflow."""
    body = json.dumps(
        {
            "error": {
                "code": "invalid_request_error",
                "message": "tools[0].function: 'model context' field missing",
            }
        }
    )
    assert is_context_overflow_text(body) is False


@pytest.mark.parametrize(
    "body",
    [
        # OpenAI.
        "context_length_exceeded",
        "This model's maximum context length is 128000 tokens",
        "input too large",
        # Anthropic.
        "prompt is too long: 250000 tokens > 200000 maximum",
        '{"type":"error","error":{"type":"invalid_request_error","message":"too_long"}}',
        # Google.
        "The request exceeds the maximum number of tokens",
        "Input too long for the model",
    ],
)
def test_every_vendor_spelling_reaches_one_classifier(body: str) -> None:
    """Each provider's overflow spelling must classify for ALL providers.

    Five providers each kept a private phrase list, so a body naming an
    overflow the way only one vendor spells it propagated raw from the
    other four -- the shape that wedged session ``190b6baec7ed``. The
    union is deliberate: a missed overflow is fatal, a false positive
    costs one wasted compaction.
    """
    assert is_context_overflow_text(body) is True


def test_per_item_string_cap_is_not_the_byte_limit() -> None:
    """OpenAI's per-item string cap is token-side, not the wire ceiling.

    Verbatim body from session ``190b6baec7ed``, which wedged. This half of
    the classification is already correct -- the provider's
    ``is_context_overflow`` is what must also recognise it, so the
    compactor's shrink-and-retry path runs instead of the raw 400
    propagating. See ``providers/openai/sub_test.py``.
    """
    assert is_request_too_large(400, PER_ITEM_STRING_CAP_BODY) is False
    raise_if_request_too_large(400, PER_ITEM_STRING_CAP_BODY)  # must not raise


def test_raise_if_request_too_large_raises_on_413() -> None:
    with pytest.raises(RequestTooLargeError):
        raise_if_request_too_large(413, "Request exceeds the maximum size")


def test_raise_if_request_too_large_noop_on_token_overflow() -> None:
    raise_if_request_too_large(400, "prompt is too long")  # must not raise


class _StatusError(Exception):
    """Error carrying a typed ``status_code`` attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__("x")
        self.status_code = status_code


def test_error_status_code_from_status_attr() -> None:
    assert error_status_code(_StatusError(413)) == 413


def test_error_status_code_from_response_attr() -> None:
    request = httpx2.Request("POST", "https://example.test")
    err = httpx2.HTTPStatusError(
        "x", request=request, response=httpx2.Response(413, request=request)
    )
    assert error_status_code(err) == 413


def test_error_status_code_absent_returns_none() -> None:
    assert error_status_code(RuntimeError("no status")) is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
