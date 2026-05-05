"""Tests for lib.descriptors."""

from __future__ import annotations

from typing import cast

from sagent.custom_types import (
    BytesDescriptor,
    BytesMessage,
    Message,
    MultipartDescriptor,
    MultipartMessage,
    TextDescriptor,
    TextMessage,
)
from sagent.lib.descriptors import (
    collect_binary,
    collect_text,
    flat_text,
    has_error,
    is_binary,
    is_image,
    is_multipart,
    is_user_message,
    strip_binary,
)


def _msg(descriptor: str, content: str | bytes | tuple[Message, ...]) -> Message:
    if isinstance(content, str):
        return TextMessage(content, cast(TextDescriptor, descriptor))
    if isinstance(content, tuple):
        return MultipartMessage(content, cast(MultipartDescriptor, descriptor))
    return BytesMessage(content, cast(BytesDescriptor, descriptor))


def _multi(descriptor: str, *children: Message) -> Message:
    return _msg(descriptor, children)


# -- Classification helpers --------------------------------------------------


def test_is_multipart() -> None:
    assert is_multipart("multipart/mixed")
    assert not is_multipart("text/plain")


def test_is_image() -> None:
    assert is_image("image/png")
    assert not is_image("application/pdf")


def test_is_binary() -> None:
    assert is_binary("image/png")
    assert is_binary("application/pdf")
    assert not is_binary("text/plain")


def test_is_user_message() -> None:
    assert is_user_message("text/x-user-message")
    assert is_user_message("multipart/x-user-message")
    assert not is_user_message("text/plain")


# -- has_error ---------------------------------------------------------------


def test_has_error_direct() -> None:
    assert has_error(_msg("text/x-error", "boom"))


def test_has_error_nested() -> None:
    msg = _multi(
        "multipart/mixed",
        _msg("text/plain", "ok"),
        _msg("text/x-error", "boom"),
    )
    assert has_error(msg)


def test_has_error_absent() -> None:
    assert not has_error(_msg("text/plain", "ok"))


# -- collect_binary ----------------------------------------------------------


def test_collect_binary_leaf() -> None:
    img = _msg("image/png", b"\x89PNG")
    assert collect_binary(img) == [img]


def test_collect_binary_nested() -> None:
    img = _msg("image/png", b"\x89PNG")
    pdf = _msg("application/pdf", b"%PDF")
    msg = _multi(
        "multipart/mixed",
        _msg("text/plain", "caption"),
        img,
        pdf,
    )
    assert collect_binary(msg) == [img, pdf]


def test_collect_binary_text_returns_empty() -> None:
    assert collect_binary(_msg("text/plain", "hi")) == []


# -- strip_binary ------------------------------------------------------------


def test_strip_binary_removes_leaf() -> None:
    assert strip_binary(_msg("image/png", b"\x89PNG")) is None


def test_strip_binary_keeps_text() -> None:
    txt = _msg("text/plain", "hi")
    assert strip_binary(txt) is txt


def test_strip_binary_filters_multipart() -> None:
    txt = _msg("text/plain", "caption")
    msg = _multi(
        "multipart/mixed",
        txt,
        _msg("image/png", b"\x89PNG"),
    )
    result = strip_binary(msg)
    assert result is not None
    assert result.content == (txt,)


def test_strip_binary_all_binary_returns_none() -> None:
    msg = _multi(
        "multipart/mixed",
        _msg("image/png", b"\x89PNG"),
        _msg("application/pdf", b"%PDF"),
    )
    assert strip_binary(msg) is None


# -- flat_text ---------------------------------------------------------------


def test_flat_text_text_descriptor() -> None:
    assert flat_text(_msg("text/plain", "hello")) == "hello"


def test_flat_text_multipart() -> None:
    msg = _multi(
        "multipart/x-model-message",
        _msg("text/x-thinking", "hmm"),
        _msg("text/plain", "answer"),
    )
    assert flat_text(msg) == "answer"


def test_flat_text_include_errors() -> None:
    msg = _multi(
        "multipart/x-tool-result",
        _msg("text/x-error", "fail"),
    )
    assert flat_text(msg) == ""
    assert flat_text(msg, include_errors=True) == "fail"


def test_flat_text_non_text_returns_empty() -> None:
    assert flat_text(_msg("application/json", "{}")) == ""


# -- collect_text ------------------------------------------------------------


def test_collect_text_plain() -> None:
    assert collect_text(_msg("text/plain", "hi")) == "hi"


def test_collect_text_nested() -> None:
    msg = _multi(
        "multipart/mixed",
        _msg("text/plain", "a"),
        _msg("image/png", b"\x89PNG"),
        _multi(
            "multipart/x-tool-result",
            _msg("text/plain", "b"),
        ),
    )
    assert collect_text(msg) == "a\nb"


def test_collect_text_non_text_returns_empty() -> None:
    assert collect_text(_msg("application/json", "{}")) == ""


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
