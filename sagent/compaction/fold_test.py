"""Tests for ``compaction.fold``: shrinking one oversized message to a budget.

The fakes are hostile in the ways live models were: replies cut off at the
cap, empty replies, prompts the provider rejects as too long, and summaries
that never shrink. A marker per chunk proves each chunk reached the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import logging
import re
import subprocess
import sys

import pytest

from sagent.compaction import fold
from sagent.compaction.fold import FoldError, fold_text
from sagent.testing import MockModelCaps
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
)
from sagent.types.runtime import AssistantMessage, UserMessage


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.types.runtime import RuntimeEvent


@dataclass(slots=True, kw_only=True)
class _FoldModel(MockModelCaps):
    """Summarize by listing the markers a request carried.

    Attributes:
      pad_chars: Filler added to every summary, to force a second level.
      truncate_calls: Call indices answered as cut off at the cap.
      empty_calls: Call indices answered with no text.
      overflow_over_tokens: Requests above this size raise ``PromptTooLongError``.
      echo: Answer with the request's own content, so nothing ever shrinks.

    """

    model_id: str = "fold-model"
    max_request_tokens: int = 200_000
    pad_chars: int = 0
    truncate_calls: frozenset[int] = frozenset()
    empty_calls: frozenset[int] = frozenset()
    overflow_over_tokens: int = 0
    echo: bool = False
    requests: list[ModelRequest] = field(default_factory=list)

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del publish
        call = len(self.requests)
        self.requests.append(request)
        if self.overflow_over_tokens and (
            self.approx_request_tokens(request) > self.overflow_over_tokens
        ):
            raise PromptTooLongError("too long")
        body = _request_text(request)
        if self.echo:
            return _reply(body)
        if call in self.empty_calls:
            return _reply(" \n\t")
        listing = "summary of: " + " ".join(_MARKER.findall(body))
        if call in self.truncate_calls:
            return _reply(listing[: len(listing) // 2], stop_reason="max_tokens")
        return _reply(listing + "\n" + "p" * self.pad_chars)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)


_MARKER = re.compile(r"M\d{4}")
_HEAD = "HEAD-INSTRUCTION: answer in exactly this format.\n"
_TAIL = "\nTAIL-LINE: the last hunk."


def _reply(text: str, *, stop_reason: str = "model_finished") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text=text), stop_reason=stop_reason)


def _request_text(request: ModelRequest) -> str:
    return "\n".join(m.text for m in request.messages if isinstance(m, UserMessage))


def _document(chunks: int = 200, chunk_chars: int = 1_000) -> tuple[str, list[str]]:
    """Return a head, ``chunks`` marked chunks, and a tail; plus the markers."""
    markers = [f"M{i:04d}" for i in range(chunks)]
    middle = "".join(f"{m} {'d' * (chunk_chars - 6)}\n" for m in markers)
    return _HEAD + middle + _TAIL, markers


async def _fold(model: _FoldModel, text: str) -> str:
    return await fold_text(text, model=model, max_tokens=6_000, call_tokens=6_000)


@pytest.mark.asyncio
async def test_fold_leaves_a_text_that_fits_untouched() -> None:
    model = _FoldModel()

    assert await _fold(model, "short") == "short"
    assert model.requests == []


@pytest.mark.asyncio
async def test_fold_fits_the_budget_and_keeps_head_and_tail_verbatim() -> None:
    """The head carries the instruction the final answer must follow (R4)."""
    model = _FoldModel()
    text, markers = _document()

    out = await _fold(model, text)

    assert model.approx_text_tokens(out) <= 6_000
    assert out.startswith(_HEAD)
    assert out.endswith(_TAIL)
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_never_sends_a_call_over_the_per_call_budget() -> None:
    model = _FoldModel()
    text, _ = _document()

    _ = await _fold(model, text)

    assert model.requests
    assert all(model.approx_request_tokens(r) <= 6_000 for r in model.requests)


@pytest.mark.asyncio
async def test_fold_halves_a_chunk_whose_prompt_would_overflow_the_call() -> None:
    """Measured tokens are not additive: a chunk under half the call can still
    tokenize past it once wrapped in the prompt, so the call itself is checked.
    """

    @dataclass(slots=True, kw_only=True)
    class _PromptHeavyModel(_FoldModel):
        @override
        def approx_request_tokens(self, request: ModelRequest) -> int:
            return 2 * super(_PromptHeavyModel, self).approx_request_tokens(request)

    model = _PromptHeavyModel()
    text, markers = _document()

    out = await _fold(model, text)

    assert all(model.approx_request_tokens(r) <= 6_000 for r in model.requests)
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_splits_a_single_line_larger_than_one_call() -> None:
    """Minified JSON or a lockfile: no newline to cut at, so cut mid-line."""
    model = _FoldModel()
    markers = [f"M{i:04d}" for i in range(40)]
    line = "".join(f"{m} {'d' * 994} " for m in markers)
    text = f"{_HEAD}{line}{_TAIL}"

    out = await _fold(model, text)

    assert all(model.approx_request_tokens(r) <= 6_000 for r in model.requests)
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_caps_every_summary_below_its_input() -> None:
    """Live: without a sent cap, summaries outgrew their input estimate."""
    model = _FoldModel()
    text, _ = _document()

    _ = await _fold(model, text)

    for request in model.requests:
        cap = request.max_response_tokens
        assert cap is not None
        assert cap < model.approx_text_tokens(_request_text(request))


@pytest.mark.asyncio
async def test_fold_folds_again_when_the_summaries_still_overflow() -> None:
    model = _FoldModel(pad_chars=4_000)
    text, markers = _document()

    out = await _fold(model, text)

    assert model.approx_text_tokens(out) <= 6_000
    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_retries_a_summary_cut_off_at_the_cap() -> None:
    """Live: ``stop_reason=max_tokens`` arrives as success with half a summary."""
    model = _FoldModel(truncate_calls=frozenset({0}))
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_retries_an_empty_summary() -> None:
    model = _FoldModel(empty_calls=frozenset({0}))
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_splits_a_chunk_the_provider_rejects_as_too_long() -> None:
    """Estimates run low; a rejected chunk is halved, never dropped."""
    model = _FoldModel(overflow_over_tokens=2_000)
    text, markers = _document()

    out = await _fold(model, text)

    assert set(_MARKER.findall(out)) == set(markers)


@pytest.mark.asyncio
async def test_fold_sends_each_attempt_once_and_logs_what_recovered(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fold owns retries (re-ask, then halve); the transport must not add its own.

    A transport retry would multiply every failed summary's sends behind the fold's
    back, and a persistent one would wait out a rate limit the fold never sees.
    """
    seen: list[tuple[int, bool]] = []

    async def send(
        model: _FoldModel,
        request: ModelRequest,
        *,
        max_attempts: int,
        persistent_retry: bool,
        publish_recoverable: Callable[[str], None],
    ) -> ModelResponse:
        seen.append((max_attempts, persistent_retry))
        publish_recoverable("transient blip")
        return await model.stream(request)

    monkeypatch.setattr(fold, "send_with_retry", send)
    caplog.set_level(logging.INFO, logger=fold.__name__)
    text, markers = _document()

    out = await _fold(_FoldModel(), text)

    assert set(_MARKER.findall(out)) == set(markers)
    assert seen
    assert set(seen) == {(1, False)}
    assert "fold recoverable: transient blip" in caplog.messages


@pytest.mark.asyncio
async def test_fold_honors_its_attempts_at_every_halving() -> None:
    """A tuned knob that only the first call sees is not a knob."""
    # The first two sends fail: with ``attempts=1`` each one halves its chunk at
    # once, so the third send is a quarter-chunk; the default would retry first.
    model = _FoldModel(empty_calls=frozenset({0, 1}))
    text, markers = _document(chunks=40)

    out = await fold_text(
        text,
        model=model,
        max_tokens=6_000,
        call_tokens=6_000,
        attempts=1,
    )

    assert set(_MARKER.findall(out)) == set(markers)
    first, second, third = (
        len(_MARKER.findall(_request_text(r))) for r in model.requests[:3]
    )
    assert first == 2 * second == 4 * third


@pytest.mark.asyncio
async def test_fold_sends_its_prompt_and_word_budget() -> None:
    """Every summary call carries the caller's prompt, filled with its word budget."""
    model = _FoldModel()
    text, _ = _document(chunks=40)

    _ = await fold_text(
        text,
        model=model,
        max_tokens=6_000,
        call_tokens=6_000,
        prompt="Say it in {{words}} words. <content>{{content}}</content>",
        words_per_token=0.5,
    )

    assert model.requests
    for request in model.requests:
        cap = request.max_response_tokens
        assert cap is not None
        assert _request_text(request).startswith(
            f"Say it in {cap // 2} words. <content>",
        )


def test_fold_imports_on_its_own() -> None:
    """Imported first, ``fold`` reached ``agent.agent`` mid-load through ``retry``."""
    done = subprocess.run(
        [sys.executable, "-c", "import sagent.compaction.fold"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr


@pytest.mark.asyncio
async def test_fold_raises_when_summaries_never_shrink() -> None:
    model = _FoldModel(echo=True)
    text, _ = _document()

    with pytest.raises(FoldError):
        _ = await _fold(model, text)


@pytest.mark.asyncio
async def test_fold_raises_when_every_attempt_comes_back_empty() -> None:
    model = _FoldModel(empty_calls=frozenset(range(1_000)))
    text, _ = _document()

    with pytest.raises(FoldError):
        _ = await _fold(model, text)


@pytest.mark.asyncio
async def test_fold_raises_when_head_and_tail_alone_leave_no_room() -> None:
    """A budget of one token cannot hold any head, tail, and summary at all."""
    model = _FoldModel()
    text, _ = _document()

    with pytest.raises(FoldError, match="no room"):
        _ = await fold_text(text, model=model, max_tokens=1, call_tokens=6_000)

    assert model.requests == []


@pytest.mark.asyncio
async def test_fold_cuts_a_head_line_longer_than_the_edge() -> None:
    """One long first line used to become the whole head and leave no room."""
    model = _FoldModel()
    markers = [f"M{i:04d}" for i in range(40)]
    text = "x" * 20_000 + "\n" + "".join(f"{m} {'d' * 994}\n" for m in markers)

    out = await _fold(model, text)

    assert model.approx_text_tokens(out) <= 6_000
    assert out.startswith("x" * 100)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
