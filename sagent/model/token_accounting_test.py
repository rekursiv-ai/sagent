"""Regression tests for request-bound token measurements and journal resume."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import sys

import pytest

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.model.capability import ModelCapability
from sagent.model.cost import TokenCount
from sagent.model.scripted import ScriptedProvider, reply
from sagent.model.token_accounting import TokenAccounting
from sagent.model.types import Request, Response
from sagent.session.jsonl import JsonlJournal, read_record, record_line
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    ReadResult,
    SessionRecord,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    TurnContext,
    UserMessage,
)
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path


def test_prompt_counts_include_all_cache_pools_and_not_output() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    request = Request(system=("system",), records=(UserMessage(content="hi"),))
    accounting.observe(
        accounting.capture(request, model=model),
        tokens=TokenCount(
            request=10,
            cache_read=20,
            cache_write=30,
            cache_write_1h=40,
            response=900,
        ),
    )
    assert accounting.estimate_request(request, model=model) == 100


def test_appended_group_is_measured_whole_and_reused_after_prefix_removed() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    first = Request(records=(UserMessage(content="prefix"),))
    block = (AssistantMessage(content="answer"), UserMessage(content="next"))
    accounting.observe(
        accounting.capture(first, model=model),
        tokens=TokenCount(request=100),
    )
    second = replace(first, records=(*first.records, *block))
    measured = accounting.observe(
        accounting.capture(second, model=model),
        tokens=TokenCount(request=157),
    )
    assert measured.block_tokens == 57
    assert accounting.estimate_block(block, model=model) == 57
    assert accounting.estimate_block(block[:1], model=model) != 57
    assert (
        accounting.estimate_block(
            (replace(block[0], content="changed"), block[1]),
            model=model,
        )
        != 57
    )
    assert (
        accounting.estimate_request(
            replace(second, records=(UserMessage(content="edited"), *block)),
            model=model,
        )
        != 157
    )


@pytest.mark.parametrize("change", ["system", "tools", "settings", "prefix"])
def test_changed_framing_or_prefix_never_produces_a_delta(change: str) -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    first = Request(records=(UserMessage(content="prefix"),))
    accounting.observe(
        accounting.capture(first, model=model),
        tokens=TokenCount(request=100),
    )
    second = replace(
        first,
        records=(*first.records, AssistantMessage(content="answer")),
    )
    if change == "system":
        second = replace(second, system=("new system",))
    elif change == "tools":
        second = replace(second, tools=(Read(),))
    elif change == "settings":
        model.settings = replace(model.settings, manage_context_server_side=True)
    else:
        second = replace(
            second,
            records=(UserMessage(content="different"), *second.records[1:]),
        )
    measured = accounting.observe(
        accounting.capture(second, model=model),
        tokens=TokenCount(request=157),
    )
    assert measured.block_tokens is None


def test_tokenizer_matches_win_over_newer_other_tokenizer_measurements() -> None:
    model = ScriptedProvider(
        name="one",
        capability=ModelCapability(model_id="m", tokenizer="shared"),
    )["m"]
    other = ScriptedProvider(
        name="two",
        capability=ModelCapability(model_id="n", tokenizer="other"),
    )["n"]
    shared = ScriptedProvider(
        name="three",
        capability=ModelCapability(model_id="o", tokenizer="shared"),
    )["o"]
    accounting = TokenAccounting()
    request = Request(records=(UserMessage(content="tiny"),))
    accounting.observe(
        accounting.capture(request, model=model),
        tokens=TokenCount(request=800),
    )
    assert accounting.estimate_request(request, model=other) == 800
    accounting.observe(
        accounting.capture(request, model=other),
        tokens=TokenCount(request=900),
    )
    assert accounting.estimate_request(request, model=shared) == 800
    assert accounting.estimate_request(request, model=other) == 900


def test_snapshot_freezes_settings_and_records_before_streaming() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    records = [UserMessage(content="before")]
    request = Request(records=records)
    snapshot = accounting.capture(request, model=model)
    records.append(UserMessage(content="during"))
    model.settings = replace(model.settings, manage_context_server_side=True)
    accounting.observe(snapshot, tokens=TokenCount(request=987))
    model.settings = replace(model.settings, manage_context_server_side=False)
    assert accounting.estimate_request(Request(records=records[:1]), model=model) == 987
    assert accounting.estimate_request(request, model=model) > 987


def test_measurement_round_trip_restores_delta_without_whole_history() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    request = Request(records=(UserMessage(content="x" * 20_000),))
    measured = accounting.observe(
        accounting.capture(request, model=model),
        tokens=TokenCount(request=700),
    )
    line = record_line(TokenUsage(measurement=measured))
    assert len(line) < 2_000
    usage = read_record(line)
    assert isinstance(usage, TokenUsage)
    assert usage.measurement == measured
    restored = TokenAccounting()
    restored.restore(measured)
    block = (AssistantMessage(content="new"), UserMessage(content="again"))
    next_request = replace(request, records=(*request.records, *block))
    next_measurement = restored.observe(
        restored.capture(next_request, model=model),
        tokens=TokenCount(request=755),
    )
    assert next_measurement.block_tokens == 55
    assert restored.estimate_block(block, model=model) == 55
    assert read_record(
        b'{"py/object":"TokenUsage","info":{"request":42}}',
    ) == TokenUsage(info={"request": 42})


@pytest.mark.asyncio
async def test_runtime_persists_measurements_and_resumes_across_windows(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        replies=[reply("answer", tokens=TokenCount(request=654))],
    )
    config = AgentsMdConfig(system_dir=tmp_path / "system", user_dir=tmp_path / "user")
    async with (
        JsonlJournal(tmp_path / "session.jsonl") as journal,
        Agent(
            model=provider["scripted"],
            journal=journal,
            agents_md=config,
            system_prompt="S",
            environment=False,
            tool_context=ToolContext(cwd=tmp_path),
        ) as agent,
    ):
        await agent.send("question")
        request = provider.requests[0]
        assert (
            agent.runtime.token_accounting.estimate_request(
                request,
                model=agent.model,
            )
            == 654
        )
        await agent.session.clear()
    async with (
        JsonlJournal(tmp_path / "session.jsonl") as journal,
        Agent(
            model=provider["scripted"],
            journal=journal,
            agents_md=config,
            system_prompt="S",
            environment=False,
            tool_context=ToolContext(cwd=tmp_path),
        ) as resumed,
    ):
        assert (
            resumed.runtime.token_accounting.estimate_request(
                request,
                model=resumed.model,
            )
            == 654
        )


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (ReadResult(call_id="a", content="x" * 40), 10),
        (TurnContext(model="abc"), 2),
        (ToolCall(name="tool", call_id="a", arguments={"x": 1}), 2),
        (Thinking(content="x" * 40), 10),
        (Thinking(summary="x" * 40), 10),
        (Thinking(encrypted="x" * 40), 10),
        (TokenUsage(), 0),
        (
            UserMessage(
                content="",
                attachments=(Attachment(mime_descriptor="image/test", data=b"x" * 40),),
            ),
            10,
        ),
    ],
)
def test_unmeasured_content_uses_characters_and_attachment_estimates(
    record: SessionRecord,
    expected: int,
) -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    assert not accounting.has_measurements
    assert accounting.estimate_block((record,), model=model) == expected
    assert accounting.estimate_block((reply("x" * 40),), model=model) == 10
    request = Request(system=("s" * 40,), records=(record,), tools=(Read(),))
    assert accounting.estimate_request(request, model=model) > 10 + expected


def test_zero_usage_never_replaces_known_measurements_or_bridges_a_delta() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    request = Request(records=(UserMessage(content="first"),))
    snapshot = accounting.capture(request, model=model)
    accounting.observe(snapshot, tokens=TokenCount())
    assert not accounting.has_measurements
    measured = accounting.observe(snapshot, tokens=TokenCount(request=70))
    assert measured.tokenizer == "scripted/scripted"
    assert accounting.has_measurements
    accounting.observe(snapshot, tokens=TokenCount())
    assert accounting.estimate_request(request, model=model) == 70
    grown = replace(
        request,
        records=(*request.records, AssistantMessage(content="more")),
    )
    result = accounting.observe(
        accounting.capture(grown, model=model),
        tokens=TokenCount(request=90),
    )
    assert result.block_tokens is None


def test_caps_change_without_changing_prompt_framing_and_negative_deltas_are_unknown() -> (
    None
):
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    first = Request(records=(UserMessage(content="first"),), max_response_tokens=100)
    accounting.observe(
        accounting.capture(first, model=model),
        tokens=TokenCount(request=100),
    )
    second = replace(
        first,
        records=(*first.records, AssistantMessage(content="answer")),
        max_response_tokens=20,
    )
    result = accounting.observe(
        accounting.capture(second, model=model),
        tokens=TokenCount(request=120),
    )
    assert result.block_tokens == 20
    third = replace(second, records=(*second.records, UserMessage(content="next")))
    result = accounting.observe(
        accounting.capture(third, model=model),
        tokens=TokenCount(request=80),
    )
    assert result.block_tokens is None


def test_measured_groups_prefer_matching_tokenizer_then_other_actual_counts() -> None:
    accounting = TokenAccounting()
    first = Request(records=(UserMessage(content="prefix"),))
    block = (AssistantMessage(content="answer"), UserMessage(content="next"))
    second = replace(first, records=(*first.records, *block))
    models = [ScriptedProvider(name=name)["scripted"] for name in ("a", "b", "c")]
    for model, delta in zip(models[:2], (57, 67), strict=True):
        accounting.observe(
            accounting.capture(first, model=model),
            tokens=TokenCount(request=100),
        )
        measured = accounting.observe(
            accounting.capture(second, model=model),
            tokens=TokenCount(request=100 + delta),
        )
        assert measured.tokenizer == f"{model.provider.name}/scripted"
    assert accounting.estimate_block(block, model=models[0]) == 57
    assert accounting.estimate_block(block, model=models[1]) == 67
    assert accounting.estimate_block(block, model=models[2]) == 67
    assert accounting.estimate_block((*block, *block), model=models[0]) == 114


@pytest.mark.asyncio
async def test_runtime_captures_request_before_stream_changes_context(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider()
    async with Agent(
        model=provider["scripted"],
        system_prompt="before",
        environment=False,
        agents_md=AgentsMdConfig(
            system_dir=tmp_path / "system",
            user_dir=tmp_path / "user",
        ),
        tool_context=ToolContext(cwd=tmp_path),
    ) as agent:

        def during(request: Request) -> Response:
            agent.runtime.system = ("changed",)
            agent.runtime.append_history(SystemMessage(content="unsent"))
            return reply(str(len(request.records)), tokens=TokenCount(request=777))

        provider.replies.append(during)
        await agent.send("question")
        actual = provider.requests[0]
        accounting = agent.runtime.token_accounting
        assert accounting.estimate_request(actual, model=agent.model) == 777
        assert (
            accounting.estimate_request(agent.runtime.request(), model=agent.model)
            != 777
        )
        usage = next(r for r in agent.session.history() if isinstance(r, TokenUsage))
        assert usage.measurement is not None
        assert usage.measurement.records == len(actual.records)


@pytest.mark.asyncio
async def test_old_journal_usage_restores_response_meters(tmp_path: Path) -> None:
    journal = RecordingJournal()
    journal.append(
        UserMessage(content="question"),
        AssistantMessage(content="answer"),
        TokenUsage(info={"request": 11, "cache_read": 22, "response": 33}),
    )
    async with Agent(
        model=ScriptedProvider()["scripted"],
        journal=journal,
        environment=False,
        agents_md=AgentsMdConfig(
            system_dir=tmp_path / "system",
            user_dir=tmp_path / "user",
        ),
        tool_context=ToolContext(cwd=tmp_path),
    ) as agent:
        response = next(
            r for r in agent.runtime.context().messages if isinstance(r, Response)
        )
        assert response.tokens == TokenCount(request=11, cache_read=22, response=33)
        assert not agent.runtime.token_accounting.has_measurements


def test_a_capture_hashes_as_a_fresh_one_does_and_a_changed_record_anew() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    prefix, said, later = (
        UserMessage(content="prefix"),
        AssistantMessage(content="a"),
        UserMessage(content="next"),
    )
    first = Request(records=(prefix, said))
    grown = replace(first, records=(prefix, said, later))
    changed = replace(grown, records=(prefix, replace(said, content="b"), later))

    snapshots = [
        accounting.capture(each, model=model) for each in (first, grown, changed)
    ]

    fresh = [
        TokenAccounting().capture(each, model=model) for each in (first, grown, changed)
    ]
    assert snapshots == fresh
    assert snapshots[2].hashes[1] != snapshots[1].hashes[1]
    assert snapshots[2].hashes[::2] == snapshots[1].hashes[::2]


def test_accounting_lets_go_of_records_no_later_capture_holds() -> None:
    model = ScriptedProvider()["scripted"]
    accounting = TokenAccounting()
    old = UserMessage(content="old")
    held = sys.getrefcount(old)

    _ = accounting.capture(Request(records=(old,)), model=model)
    kept = sys.getrefcount(old)
    for number in range(3):
        later = Request(records=(UserMessage(content=f"new {number}"),))
        _ = accounting.capture(later, model=model)

    assert kept > held
    assert sys.getrefcount(old) == held


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
