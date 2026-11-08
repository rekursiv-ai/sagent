"""Tests for fold: shrinking one oversized text to a token budget.

The summarizer is hostile the ways live models were: replies cut off at the
cap, empty replies, prompts the provider rejects as too long, and summaries
that never shrink. A marker per chunk proves each chunk reached the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import Mock

import math
import re

import pytest

from sagent.agent import fold as agent_fold
from sagent.agent.fold import FoldError, fold
from sagent.model.capability import ModelCapability
from sagent.model.cost import TokenCost
from sagent.model.exceptions import (
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
)
from sagent.model.scripted import ScriptedModel, ScriptedProvider, call, reply
from sagent.model.types import TextChunk
from sagent.session.types import AssistantMessage, UserMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sagent.model.types import Request, Response


_MARKER = re.compile(r"M\d{4}")
_HEAD = "HEAD-INSTRUCTION: answer in exactly this format.\n"
_TAIL = "\nTAIL-LINE: the last hunk."


@dataclass(slots=True, kw_only=True)
class _Summarize:
    """Answer each request with the markers it carried.

    Attributes:
      pad_share: Filler added to every summary, as a share of its request, to
        force a second level.
      truncate_calls: Calls answered as cut off at the cap.
      empty_calls: Calls answered with no text.
      overflow_over_chars: Requests longer than this raise ``ContextOverflowError``.
      echo: Answer with the request itself, so nothing ever shrinks.

    """

    pad_share: float = 0.0
    truncate_calls: frozenset[int] = frozenset()
    empty_calls: frozenset[int] = frozenset()
    overflow_over_chars: int = 0
    echo: bool = False
    calls: int = field(default=0)

    def __call__(self, request: Request) -> Response:
        n = self.calls
        self.calls += 1
        body = _prompt(request)
        if self.overflow_over_chars and len(body) > self.overflow_over_chars:
            raise ContextOverflowError("too long")
        if self.echo:
            return reply(body)
        if n in self.empty_calls:
            return reply(" \n\t")
        listing = "summary of: " + " ".join(_MARKER.findall(body))
        if n in self.truncate_calls:
            return reply(listing[: len(listing) // 2], stop_reason="max_tokens")
        return reply(listing + "\n" + "p" * int(len(body) * self.pad_share))


def _model(summarize: _Summarize) -> tuple[ScriptedProvider, ScriptedModel]:
    provider = ScriptedProvider(replies=[summarize] * 64)
    return provider, provider["scripted"]


def _prompt(request: Request) -> str:
    (record,) = request.records
    assert isinstance(record, UserMessage)
    return record.content or ""


def _tokens(text: str) -> int:
    return len(text) // 4


def _document(chunks: int = 12, chunk_chars: int = 200) -> tuple[str, set[str]]:
    """Return a head, ``chunks`` marked lines, and a tail; and the markers."""
    markers = [f"M{i:04d}" for i in range(chunks)]
    middle = "".join(f"{m} {'d' * (chunk_chars - 6)}\n" for m in markers)
    return _HEAD + middle + _TAIL, set(markers)


async def _fold(model: ScriptedModel, text: str) -> str:
    return await fold(text, model=model, max_tokens=200, call_tokens=400)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [1, 2])
async def test_a_text_that_fits_is_returned_untouched(max_tokens: int) -> None:
    provider = ScriptedProvider()

    assert (
        await fold(
            "short",
            model=provider["scripted"],
            max_tokens=max_tokens,
            call_tokens=2,
        )
        == "short"
    )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_the_fold_fits_its_budget_and_keeps_head_and_tail_verbatim() -> None:
    provider = ScriptedProvider(replies=[_costly] * 8)
    text, markers = _document()
    spent: list[float] = []

    out = await fold(
        text,
        model=provider["scripted"],
        max_tokens=200,
        call_tokens=400,
        spent=spent.append,
    )

    assert _tokens(out) <= 200
    assert out.startswith(_HEAD)
    assert out.endswith(_TAIL)
    assert set(_MARKER.findall(out)) == markers
    assert "[summary of the omitted middle of this message]\n" in out
    assert "\n[end of summary]" in out
    assert provider.requests
    assert spent == [0.01] * len(provider.requests)
    for request in provider.requests:
        prompt = _prompt(request)
        assert _tokens(prompt) <= 400
        content = prompt.split("<content>\n", 1)[1].removesuffix("\n</content>\n")
        words = max(1, math.floor(max(1, _tokens(content) // 4) * 0.75))
        assert prompt.startswith(
            f"Summarize the content between the <content> tags in at most {words} words.\n",
        )


def _costly(request: Request) -> Response:
    """Answer as ``_Summarize`` does, at a cent a call."""
    return replace(_Summarize()(request), spend=TokenCost(response=0.01))


@pytest.mark.asyncio
async def test_a_single_line_larger_than_one_call_is_cut_mid_line() -> None:
    provider, model = _model(_Summarize())
    markers = [f"M{i:04d}" for i in range(12)]
    text = _HEAD + "".join(f"{m} {'d' * 194} " for m in markers) + _TAIL

    out = await _fold(model, text)

    assert provider.requests
    assert all(_tokens(_prompt(r)) <= 400 for r in provider.requests)
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_summaries_that_still_overflow_are_folded_again() -> None:
    summarize = _Summarize(pad_share=0.3)
    provider, model = _model(summarize)
    text, markers = _document()

    out = await _fold(model, text)

    assert _tokens(out) <= 200
    assert set(_MARKER.findall(out)) == markers
    assert any("summary of:" in _prompt(r) for r in provider.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("truncate_calls", "empty_calls"),
    [(frozenset({0}), frozenset[int]()), (frozenset[int](), frozenset({0}))],
    ids=["cut-off", "empty"],
)
async def test_a_failed_summary_is_asked_again(
    truncate_calls: frozenset[int],
    empty_calls: frozenset[int],
) -> None:
    provider, model = _model(
        _Summarize(truncate_calls=truncate_calls, empty_calls=empty_calls),
    )
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == markers
    assert _prompt(provider.requests[0]) == _prompt(provider.requests[1])


@pytest.mark.asyncio
async def test_a_chunk_the_provider_rejects_as_too_long_is_halved() -> None:
    provider, model = _model(_Summarize(overflow_over_chars=1_000))
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == markers
    assert any(len(_prompt(r)) > 1_000 for r in provider.requests)
    assert any(len(_prompt(r)) <= 1_000 for r in provider.requests)


@pytest.mark.asyncio
async def test_a_transient_failure_counts_as_a_failed_attempt() -> None:
    summarize = _Summarize()
    provider = ScriptedProvider(replies=[OverloadedError("busy"), *[summarize] * 8])
    text, markers = _document(chunks=8)

    out = await _fold(provider["scripted"], text)

    assert set(_MARKER.findall(out)) == markers


@pytest.mark.asyncio
async def test_attempts_are_honored_at_every_halving() -> None:
    provider, model = _model(_Summarize(empty_calls=frozenset({0, 1})))
    text, markers = _document(chunks=8)

    out = await fold(text, model=model, max_tokens=200, call_tokens=400, attempts=1)

    assert set(_MARKER.findall(out)) == markers
    first, second, third = (
        len(_MARKER.findall(_prompt(r))) for r in provider.requests[:3]
    )
    assert first == 2 * second == 4 * third


@pytest.mark.asyncio
async def test_each_call_carries_the_prompt_and_its_word_budget() -> None:
    provider, model = _model(_Summarize())
    text, _ = _document(chunks=8)
    text = text.replace("M0004 ", "M0004 {{words}} ", 1)

    _ = await fold(
        text,
        model=model,
        max_tokens=200,
        call_tokens=400,
        prompt="Say it in {{words}} words. <content>{{content}}</content>",
        shrink=2,
        words_per_token=0.5,
        tag=partial(replace, session="s-1"),
    )

    assert provider.requests
    for request in provider.requests:
        assert (request.purpose, request.session) == ("compaction", "s-1")
        prompt = _prompt(request)
        content = prompt.split("<content>", 1)[1].removesuffix("</content>")
        words = math.floor(_tokens(content) // 2 * 0.5)
        assert prompt.startswith(f"Say it in {words} words. <content>")
    assert any("M0004 {{words}}" in _prompt(r) for r in provider.requests)


@pytest.mark.asyncio
async def test_summaries_that_never_shrink_raise() -> None:
    _, model = _model(_Summarize(echo=True))
    text, _ = _document(chunks=8)

    with pytest.raises(FoldError, match="could not be summarized or split"):
        _ = await _fold(model, text)


@pytest.mark.asyncio
async def test_every_attempt_coming_back_empty_raises() -> None:
    _, model = _model(_Summarize(empty_calls=frozenset(range(64))))
    text, _ = _document(chunks=8)

    with pytest.raises(FoldError, match="could not be summarized or split"):
        _ = await _fold(model, text)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [1, 2])
async def test_a_budget_the_head_and_tail_fill_raises_before_any_call(
    max_tokens: int,
) -> None:
    provider = ScriptedProvider(replies=[reply("x")])

    with pytest.raises(FoldError, match="no room"):
        _ = await fold(
            "abcdefghijkl",
            model=provider["scripted"],
            max_tokens=max_tokens,
            call_tokens=100,
            opener="",
            closer="",
            prompt="{{content}}",
        )

    assert provider.requests == []


@pytest.mark.asyncio
async def test_a_summary_that_fills_the_remaining_budget_needs_no_second_call() -> None:
    provider = ScriptedProvider(replies=[reply("summary!")])

    out = await fold(
        "abcdefghijklmnopqrstuvwx",
        model=provider["scripted"],
        max_tokens=4,
        call_tokens=100,
        opener="",
        closer="",
        prompt="{{content}}",
    )

    assert _tokens(out) <= 4
    assert out.startswith("abcd")
    assert out.endswith("uvwx")
    assert "summary!" in out
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (("alpha", "beta"), "HEAD\n\nalpha\nbeta\n\nTAIL\n"),
        (("", "a", "b", "c"), "HEAD\n\na\nb\nc\n\nTAIL\n"),
    ],
    ids=["chunks", "halved-chunk"],
)
async def test_joined_summaries_add_no_words_to_the_model_replies(
    responses: tuple[str, ...],
    expected: str,
) -> None:
    provider = ScriptedProvider(replies=[reply(text) for text in responses])

    out = await fold(
        "HEAD\naaaa\nbbbb\ncccc\ndddd\nTAIL\n",
        model=provider["scripted"],
        max_tokens=6,
        call_tokens=4,
        opener="",
        closer="",
        prompt="{{content}}",
        attempts=1,
    )

    assert _tokens(out) <= 6
    assert out == expected
    assert len(provider.requests) == len(responses)


@pytest.mark.asyncio
async def test_a_head_line_longer_than_the_edge_is_cut_mid_line() -> None:
    _, model = _model(_Summarize())
    markers = [f"M{i:04d}" for i in range(12)]
    text = "x" * 1_000 + "\n" + "".join(f"{m} {'d' * 194}\n" for m in markers)

    out = await _fold(model, text)

    assert _tokens(out) <= 200
    assert out.startswith("x" * 100)
    assert "x" * 101 not in out
    assert set(_MARKER.findall(out)) == set(markers)


def _shorter(request: Request) -> Response:
    """Answer with the content less one character."""
    return reply("b" * (len(_prompt(request)) - 1))


@pytest.mark.asyncio
async def test_a_level_that_does_not_shrink_in_tokens_raises() -> None:
    # Each summary is a token shorter than its chunk, but at 3.5 characters per
    # token the newline joining two of them gives that token back.
    provider = ScriptedProvider(
        replies=[_shorter] * 10,
        capability=ModelCapability(model_id="scripted", approx_chars_per_token=3.5),
    )
    text = "HHHH\n" + "aaa\naaa\n" + "\nTTTTT"

    with pytest.raises(FoldError, match="did not shrink"):
        _ = await fold(
            text,
            model=provider["scripted"],
            max_tokens=4,
            call_tokens=2,
            opener="",
            closer="",
            prompt="{{content}}",
        )

    assert [_prompt(r) for r in provider.requests] == ["aaa\n", "aaa\n"]


@pytest.mark.parametrize(
    ("text", "chars_per_token", "limit", "expected"),
    [
        ("ab\ncd\n", 4.0, 100, "ab\ncd\n"),
        ("", 4.0, 1, ""),
        ("abc\nnext", 1.5, 1, "ab"),
    ],
)
def test_an_edge_preserves_whole_lines_or_cuts_the_first_line(
    text: str,
    chars_per_token: float,
    limit: int,
    expected: str,
) -> None:
    model = ScriptedProvider(
        capability=ModelCapability(
            model_id="scripted",
            approx_chars_per_token=chars_per_token,
        ),
    )["scripted"]

    assert agent_fold._edge(text, model=model, limit=limit) == expected


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("", 10, []),
        ("aaa\nbbbbbbb\nccc\ndddd\n", 3, ["aaa\nbbbbbbb\n", "ccc\ndddd\n"]),
        ("aaa\nbbbbbbb\nccc\ndddd\n", 0, ["aaa\n", "bbbbbbb\n", "ccc\n", "dddd\n"]),
    ],
)
def test_chunk_boundaries_preserve_every_character(
    text: str,
    limit: int,
    expected: list[str],
) -> None:
    assert (
        agent_fold._chunks(text, model=ScriptedProvider()["scripted"], limit=limit)
        == expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abcde", ("ab", "cde")),
        ("a\nlonger\n", ("a\n", "longer\n")),
        ("a\nb\nc\nd\n", ("a\nb\n", "c\nd\n")),
    ],
)
def test_halving_preserves_lines_and_every_character(
    text: str,
    expected: tuple[str, str],
) -> None:
    assert agent_fold._halve(text) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokens", "shrink", "words_per_token", "words", "responses"),
    [
        (7, 2, 0.75, 2, ("a",)),
        (3, 2, 1.5, 1, ("bbbbbbbbbbbb", "a")),
        (1, 4, 1.75, 1, ("a",)),
    ],
)
async def test_a_filled_call_at_its_budget_uses_the_requested_word_share(
    tokens: int,
    shrink: int,
    words_per_token: float,
    words: int,
    responses: tuple[str, ...],
) -> None:
    provider = ScriptedProvider(
        replies=[*[reply(text) for text in responses], *[reply("a")] * 20],
    )
    content = "c" * (4 * tokens)
    invocation = agent_fold._Call(
        model=provider["scripted"],
        call_tokens=tokens,
        shrink=shrink,
        attempts=len(responses),
        words_per_token=words_per_token,
        spent=lambda _: None,
        tag=lambda request: request,
        prompt="{{words}}:{{content}}",
    )

    assert await agent_fold._summarize(content, call=invocation) == "a"
    assert [_prompt(request) for request in provider.requests] == [
        f"{words}:{content}",
    ] * len(responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [False, True], ids=["unfinished", "multipart"])
async def test_an_attempt_uses_only_finished_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
    finished: bool,
) -> None:
    response = replace(
        reply("", spend=TokenCost(response=0.01)),
        records=(
            AssistantMessage(content=" alpha"),
            call("Read"),
            AssistantMessage(content=None),
            AssistantMessage(content="beta "),
        ),
    )

    async def stream(request: Request) -> AsyncIterator[TextChunk | Response]:
        del request
        yield TextChunk(text="partial")
        if finished:
            yield response

    model = ScriptedProvider()["scripted"]
    monkeypatch.setattr(model, "stream", stream)
    spent = Mock()
    invocation = agent_fold._Call(
        model=model,
        call_tokens=10,
        shrink=4,
        attempts=1,
        words_per_token=0.75,
        spent=spent,
        tag=lambda request: request,
        prompt="{{content}}",
    )

    assert await agent_fold._attempt("content", call=invocation) == (
        "alphabeta" if finished else ""
    )
    if finished:
        spent.assert_called_once_with(0.01)
    else:
        spent.assert_not_called()


@pytest.mark.asyncio
async def test_a_rate_limited_model_is_asked_no_more_than_its_attempts() -> None:
    provider = ScriptedProvider(
        replies=[RateLimitedError("slow down", retry_after_sec=0.0)] * 3,
    )
    text, _ = _document(chunks=4)

    with pytest.raises(RateLimitedError):
        _ = await fold(
            text,
            model=provider["scripted"],
            max_tokens=200,
            call_tokens=400,
        )

    assert len(provider.requests) == 3


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
