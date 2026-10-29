"""Tests for what a model is asked: a bare request offers everything and says nothing."""

from __future__ import annotations

import uuid

import pytest

from sagent.model.capability import ModelLimits
from sagent.model.types import Request, TextChunk


def test_a_bare_request_is_empty_and_allows_every_tool() -> None:
    request = Request()
    assert (request.system, tuple(request.records), tuple(request.tools)) == (
        (),
        (),
        (),
    )
    assert request.allowed is None
    assert request.max_response_tokens == 0


@pytest.mark.parametrize(
    ("limit", "cap", "reply"),
    [
        (128_000, 0, 128_000),
        (128_000, 20_000, 20_000),
        (128_000, 200_000, 128_000),
        (0, 20_000, 20_000),
        (0, 0, 0),
    ],
)
def test_a_reply_holds_the_least_of_the_models_limit_and_the_requests(
    limit: int,
    cap: int,
    reply: int,
) -> None:
    request = Request(max_response_tokens=cap)
    limits = ModelLimits(max_response_tokens=limit)
    assert request.response_tokens(limits) == reply


def test_a_prompt_and_each_call_of_it_have_ids_that_a_resume_derives_again() -> None:
    asked = Request(agent="a-1", turn=2, step=0)
    again = Request(agent="a-1", turn=2, step=1)
    assert asked.prompt_id == again.prompt_id
    assert asked.call_id != again.call_id
    assert asked.call_id == Request(agent="a-1", turn=2, step=0).call_id
    ids = {
        Request(agent="a-1", turn=3).prompt_id,
        Request(agent="a-2", turn=2).prompt_id,
        asked.prompt_id,
    }
    assert len(ids) == 3
    assert all(str(uuid.UUID(each)) == each for each in (*ids, asked.call_id))


def test_a_chunk_is_reply_text_unless_marked_thinking() -> None:
    assert TextChunk(text="hi").thinking is False
    assert TextChunk(text="hm", thinking=True).thinking is True


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
