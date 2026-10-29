"""Tests for fold: shrinking one oversized text to a token budget.

The summarizer is hostile the ways live models were: replies cut off at the
cap, empty replies, prompts the provider rejects as too long, and summaries
that never shrink. A marker per chunk proves each chunk reached the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

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
from sagent.model.scripted import ScriptedModel, ScriptedProvider, reply
from sagent.session.types import UserMessage


if TYPE_CHECKING:
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
    provider = ScriptedProvider(replies=[summarize] * 2_000)
    return provider, provider["scripted"]


def _prompt(request: Request) -> str:
    (record,) = request.records
    assert isinstance(record, UserMessage)
    return record.content or ""


def _tokens(text: str) -> int:
    return len(text) // 4


def _document(chunks: int = 200, chunk_chars: int = 1_000) -> tuple[str, set[str]]:
    """Return a head, ``chunks`` marked lines, and a tail; and the markers."""
    markers = [f"M{i:04d}" for i in range(chunks)]
    middle = "".join(f"{m} {'d' * (chunk_chars - 6)}\n" for m in markers)
    return _HEAD + middle + _TAIL, set(markers)


async def _fold(model: ScriptedModel, text: str) -> str:
    return await fold(text, model=model, max_tokens=6_000, call_tokens=6_000)


@pytest.mark.asyncio
async def test_a_text_that_fits_is_returned_untouched() -> None:
    provider, model = _model(_Summarize())

    assert await _fold(model, "short") == "short"
    assert provider.requests == []


@pytest.mark.asyncio
async def test_the_fold_fits_its_budget_and_keeps_head_and_tail_verbatim() -> None:
    provider, model = _model(_Summarize())
    text, markers = _document()

    out = await _fold(model, text)

    assert _tokens(out) <= 6_000
    assert out.startswith(_HEAD)
    assert out.endswith(_TAIL)
    assert set(_MARKER.findall(out)) == markers
    assert "[summary of the omitted middle of this message]\n" in out
    assert "\n[end of summary]" in out
    assert provider.requests


def _costly(request: Request) -> Response:
    """Answer as ``_Summarize`` does, at a cent a call."""
    return replace(_Summarize()(request), spend=TokenCost(response=0.01))


@pytest.mark.asyncio
async def test_each_summary_call_reports_what_it_cost() -> None:
    provider = ScriptedProvider(replies=[_costly] * 2_000)
    text, _ = _document()
    spent: list[float] = []

    _ = await fold(
        text,
        model=provider["scripted"],
        max_tokens=6_000,
        call_tokens=6_000,
        spent=spent.append,
    )

    assert spent == [0.01] * len(provider.requests)


@pytest.mark.asyncio
async def test_no_call_is_larger_than_the_per_call_budget() -> None:
    provider, model = _model(_Summarize())
    text, _ = _document()

    _ = await _fold(model, text)

    assert all(_tokens(_prompt(r)) <= 6_000 for r in provider.requests)


@pytest.mark.asyncio
async def test_a_single_line_larger_than_one_call_is_cut_mid_line() -> None:
    provider, model = _model(_Summarize())
    markers = [f"M{i:04d}" for i in range(40)]
    text = _HEAD + "".join(f"{m} {'d' * 994} " for m in markers) + _TAIL

    out = await _fold(model, text)

    assert all(_tokens(_prompt(r)) <= 6_000 for r in provider.requests)
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_summaries_that_still_overflow_are_folded_again() -> None:
    summarize = _Summarize(pad_share=0.3)
    provider, model = _model(summarize)
    text, markers = _document()

    out = await _fold(model, text)

    assert _tokens(out) <= 6_000
    assert set(_MARKER.findall(out)) == markers
    assert any("summary of:" in _prompt(r) for r in provider.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summarize",
    [
        _Summarize(truncate_calls=frozenset({0})),
        _Summarize(empty_calls=frozenset({0})),
    ],
    ids=["cut-off", "empty"],
)
async def test_a_failed_summary_is_asked_again(summarize: _Summarize) -> None:
    provider, model = _model(summarize)
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == markers
    assert _prompt(provider.requests[0]) == _prompt(provider.requests[1])


@pytest.mark.asyncio
async def test_a_chunk_the_provider_rejects_as_too_long_is_halved() -> None:
    provider, model = _model(_Summarize(overflow_over_chars=8_000))
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == markers
    assert any(len(_prompt(r)) <= 8_000 for r in provider.requests)


@pytest.mark.asyncio
async def test_a_transient_failure_counts_as_a_failed_attempt() -> None:
    summarize = _Summarize()
    provider = ScriptedProvider(replies=[OverloadedError("busy"), *[summarize] * 100])
    text, markers = _document(chunks=40)

    out = await _fold(provider["scripted"], text)

    assert set(_MARKER.findall(out)) == markers


@pytest.mark.asyncio
async def test_attempts_are_honored_at_every_halving() -> None:
    provider, model = _model(_Summarize(empty_calls=frozenset({0, 1})))
    text, markers = _document(chunks=40)

    out = await fold(text, model=model, max_tokens=6_000, call_tokens=6_000, attempts=1)

    assert set(_MARKER.findall(out)) == markers
    first, second, third = (
        len(_MARKER.findall(_prompt(r))) for r in provider.requests[:3]
    )
    assert first == 2 * second == 4 * third


@pytest.mark.asyncio
async def test_each_call_carries_the_prompt_and_its_word_budget() -> None:
    provider, model = _model(_Summarize())
    text, _ = _document(chunks=40)
    text = text.replace("M0020 ", "M0020 {{words}} ", 1)

    _ = await fold(
        text,
        model=model,
        max_tokens=6_000,
        call_tokens=6_000,
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
    assert any("M0020 {{words}}" in _prompt(r) for r in provider.requests)


@pytest.mark.asyncio
async def test_summaries_that_never_shrink_raise() -> None:
    _, model = _model(_Summarize(echo=True))
    text, _ = _document(chunks=30)

    with pytest.raises(FoldError, match="could not be summarized or split"):
        _ = await _fold(model, text)


@pytest.mark.asyncio
async def test_every_attempt_coming_back_empty_raises() -> None:
    _, model = _model(_Summarize(empty_calls=frozenset(range(2_000))))
    text, _ = _document(chunks=30)

    with pytest.raises(FoldError, match="could not be summarized or split"):
        _ = await _fold(model, text)


@pytest.mark.asyncio
async def test_a_budget_the_head_and_tail_fill_raises_before_any_call() -> None:
    provider, model = _model(_Summarize())
    text, _ = _document()

    with pytest.raises(FoldError, match="no room"):
        _ = await fold(text, model=model, max_tokens=1, call_tokens=6_000)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_a_head_line_longer_than_the_edge_is_cut_mid_line() -> None:
    _, model = _model(_Summarize())
    markers = [f"M{i:04d}" for i in range(40)]
    text = "x" * 20_000 + "\n" + "".join(f"{m} {'d' * 994}\n" for m in markers)

    out = await _fold(model, text)

    assert _tokens(out) <= 6_000
    assert out.startswith("x" * 100)
    assert "x" * 3_001 not in out
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


def test_an_edge_wider_than_the_text_takes_all_of_it() -> None:
    model = ScriptedProvider()["scripted"]

    assert agent_fold._edge("ab\ncd\n", model=model, limit=100) == "ab\ncd\n"
    assert agent_fold._edge("", model=model, limit=1) == ""


def test_no_text_is_no_chunks() -> None:
    assert agent_fold._chunks("", model=ScriptedProvider()["scripted"], limit=10) == []


@pytest.mark.asyncio
async def test_a_rate_limited_model_is_asked_no_more_than_its_attempts() -> None:
    provider = ScriptedProvider(
        replies=[RateLimitedError("slow down", retry_after_sec=0.0)] * 1_000,
    )
    text, _ = _document(chunks=4)

    with pytest.raises(RateLimitedError):
        _ = await fold(
            text,
            model=provider["scripted"],
            max_tokens=200,
            call_tokens=6_000,
        )

    assert len(provider.requests) == 3


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
