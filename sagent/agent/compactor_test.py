"""Tests for the summary compactor: when it is due, what it sends, what it keeps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import asyncio
import logging
import math
import threading
import time

from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent, spent
from sagent.agent.compactor import (
    SummaryCompactor,
    _request_tokens,
    halve,
    summary_tokens,
)
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, TokenCost, TokenCount
from sagent.model.exceptions import (
    ContextOverflowError,
    OverloadedError,
    RateLimitedError,
)
from sagent.model.scripted import (
    Reply,
    ScriptedModel,
    ScriptedProvider,
    call,
    reply,
)
from sagent.model.types import Request, Response, TextChunk
from sagent.runtime import types
from sagent.session.jsonl import JsonlJournal
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.read import Read
from sagent.tools.skill import Skill
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from sagent.model.capability import ThinkingEffort
    from sagent.runtime.tape import ContextSplice
    from sagent.runtime.types import ModelContextEvent
    from sagent.session.types import SessionRecord


_SUMMARY_TEXT = "<analysis>scratch notes</analysis>\n<summary>the gist</summary>"
_SUMMARY = reply(_SUMMARY_TEXT)


def _capability(
    *,
    window: int = 200_000,
    response: int = 8_000,
    context: int = 0,
    efforts: frozenset[ThinkingEffort] = frozenset({"none"}),
) -> ModelCapability:
    prices = PriceCatalog(window=window)
    return ModelCapability(
        model_id="scripted",
        limits=ModelLimits(
            max_request_tokens=window,
            max_response_tokens=response,
            max_context_tokens=context,
        ),
        prices=prices,
        thinking_effort=efforts,
        default=ModelSettings(context_size=prices["fullcontext"]),
    )


def _agent(
    tmp_path: Path,
    *replies: Reply,
    capability: ModelCapability | None = None,
    system_prompt: str = "",
) -> tuple[ScriptedProvider, Agent]:
    provider = ScriptedProvider(replies=replies, capability=capability)
    agent = Agent(
        model=provider["scripted"],
        tools=[Read()],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt=system_prompt,
        compactor=SummaryCompactor,
    )
    return provider, agent


async def _compact(
    compactor: SummaryCompactor,
    agent: Agent,
    history: Sequence[ModelContextEvent],
    instructions: str | None = None,
) -> ContextSplice:
    return await compactor.compact(
        agent.runtime.tape,
        history,
        agent.model,
        agent.runtime.mint_ref,
        instructions,
    )


def _user(text: str, *attachments: Attachment) -> UserMessage:
    return UserMessage(content=text, attachments=attachments)


def _turn(text: str = "", *calls: ToolCall) -> Response:
    return reply(text, calls=calls)


def _result(call_id: str, text: str = "ok") -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id=call_id, content=text)


def _instruction(request: Request) -> str:
    """Return the summary instruction: the request's last record."""
    last = request.records[-1]
    assert isinstance(last, UserMessage)
    return last.content or ""


def _sent(request: Request) -> list[SessionRecord]:
    """Return what the request asked to have summarized."""
    return list(request.records[:-1])


def _texts(records: Sequence[SessionRecord]) -> list[str]:
    out: list[str] = []
    for r in records:
        if isinstance(r, ToolResult):
            out.append(r.text())
        elif isinstance(r, (UserMessage, AssistantMessage, AgentToAgentMessage)):
            out.append(r.content or "")
    return out


def _records(entries: Sequence[ModelContextEvent]) -> list[SessionRecord]:
    """Return ``entries`` as records, each model turn as its own."""
    return [
        r for e in entries for r in (e.records if isinstance(e, Response) else (e,))
    ]


def _lead(splice: ContextSplice) -> str:
    """Return the text of the payload's first user message."""
    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    return first.content or ""


_HISTORY: tuple[ModelContextEvent, ...] = (
    _user("fix the bug"),
    _turn("reading", call("Read", "c1", file_path="a.py")),
    _result("c1", "def f(): ..."),
    _turn("fixed it"),
)


@pytest.mark.parametrize(
    ("body", "share"),
    [
        (1_000, 0.10),
        (32_000, 0.10),
        (62_500, 0.10),
        (128_000, 0.0836),
        (500_000, 0.0595),
        (1_000_000, 0.050),
        (2_000_000, 0.050),
        (10_000_000, 0.050),
    ],
)
def test_the_summary_share_falls_from_10_percent_to_a_5_percent_floor(
    body: int,
    share: float,
) -> None:
    assert summary_tokens(body) / body == pytest.approx(share, abs=5e-4)


def test_the_summary_share_takes_its_cap_floor_and_power() -> None:
    assert summary_tokens(1_000, max_share=0.2) / 1_000 == pytest.approx(0.2)
    assert summary_tokens(2e6, min_share=0.04) / 2e6 == pytest.approx(0.04)
    assert summary_tokens(5e5, power=0.5) / 5e5 == pytest.approx(0.05 * 2**0.5)
    assert summary_tokens(4e6, max_context=4e6) / 4e6 == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_share": 0.0}, "min_share"),
        ({"min_share": math.nan}, "min_share"),
        ({"max_share": 0.04}, "max_share"),
        ({"max_share": math.nan}, "max_share"),
        ({"power": 0.0}, "power"),
        ({"power": 1.5}, "power"),
        ({"power": math.nan}, "power"),
    ],
    ids=[
        "no-floor",
        "nan-floor",
        "cap-below-floor",
        "nan-cap",
        "no-power",
        "power-past-1",
        "nan-power",
    ],
)
def test_out_of_domain_shares_and_powers_are_refused(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = summary_tokens(1e5, **kwargs)


@pytest.mark.parametrize("power", [0.25, 1.0])
def test_a_body_and_its_summary_rise_together(power: float) -> None:
    """The trigger's bisection needs ``B + S(B)`` never to fall as ``B`` grows."""
    held = [b + summary_tokens(b, power=power) for b in range(0, 2_000_001, 5_000)]

    assert held == sorted(held)


def test_an_empty_body_needs_no_summary() -> None:
    assert summary_tokens(0) == 0.0


def test_too_few_attempts_are_refused(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)

    with pytest.raises(ValueError, match="max_attempts"):
        _ = SummaryCompactor(agent, max_attempts=0)


def test_the_trigger_is_the_largest_body_whose_summary_fits(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=1_000_000))
    compactor = SummaryCompactor(agent)
    room = 1_000_000 - compactor._prompt_tokens()
    trigger = int(compactor.trigger_tokens())

    assert trigger + summary_tokens(trigger) <= room
    assert trigger + 1 + summary_tokens(trigger + 1) > room


def test_a_1m_window_compacts_at_95_percent_and_holds_5_percent_for_the_summary(
    tmp_path: Path,
) -> None:
    trigger = _trigger(tmp_path, window=1_000_000)

    assert summary_tokens(trigger) > 48_000
    assert 0.95 * 1_000_000 < trigger < 0.96 * 1_000_000


def _trigger(
    tmp_path: Path,
    *,
    window: int,
    response: int = 0,
    context: int = 0,
) -> float:
    capability = _capability(window=window, response=response, context=context)
    _, agent = _agent(tmp_path, capability=capability)
    return SummaryCompactor(agent).trigger_tokens()


def test_a_compaction_is_due_once_the_body_reaches_the_trigger(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=100_000))
    compactor = SummaryCompactor(agent)
    # Unbilled, the request holds the agent's tools beside the message.
    body = math.ceil(compactor.trigger_tokens()) - 1 - compactor._tool_tokens()
    agent.session.append(_user("x" * 4 * body))
    assert not compactor.should_run()

    agent.session.append(_user("x" * 8))

    assert compactor.should_run()


def test_the_reply_limit_holds_back_none_of_the_window(tmp_path: Path) -> None:
    """A reply gets the room left; the vendor's ceiling on it reserves nothing."""
    assert _trigger(tmp_path, window=1_000_000, response=128_000) == _trigger(
        tmp_path,
        window=1_000_000,
    )


def test_a_shared_window_holds_back_the_summarys_room(tmp_path: Path) -> None:
    """With output apart, input alone bounds it; shared, the summary needs room."""
    shared = _trigger(tmp_path, window=272_000)
    apart = _trigger(tmp_path, window=272_000, context=1_050_000)

    assert apart - shared == pytest.approx(summary_tokens(shared), abs=2)


def test_with_output_apart_the_input_limit_alone_bounds_the_trigger(
    tmp_path: Path,
) -> None:
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=272_000, context=1_050_000),
    )
    compactor = SummaryCompactor(agent)
    fixed = compactor._system_tokens() + compactor._prompt_tokens()

    assert compactor.trigger_tokens() + fixed == 272_000


def test_the_system_prompt_comes_out_of_the_trigger(tmp_path: Path) -> None:
    capability = _capability(window=100_000)
    _, plain = _agent(tmp_path, capability=capability)
    _, prompted = _agent(tmp_path, capability=capability, system_prompt="s" * 4_000)
    plain_held, prompted_held = (
        trigger + summary_tokens(trigger)
        for trigger in (
            SummaryCompactor(agent).trigger_tokens() for agent in (plain, prompted)
        )
    )
    system = SummaryCompactor(prompted)._system_tokens()

    assert plain_held - prompted_held == pytest.approx(system, abs=2)


def test_the_models_chosen_size_is_a_smaller_window(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=1_000_000))
    agent.model.settings = replace(
        agent.model.settings,
        context_size=agent.model.capability.prices[272_000],
    )

    assert SummaryCompactor(agent).trigger_tokens() == _trigger(
        tmp_path,
        window=272_000,
    )


def test_a_one_token_size_is_a_size(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        context_size=agent.model.capability.prices[1],
    )
    assert SummaryCompactor(agent).trigger_tokens() == 0


@pytest.mark.parametrize(
    ("window", "context", "truncates", "room"),
    [
        (100_000, 0, None, 95_000 - 1_000 - 10),
        (100_000, 0, False, 95_000 - 1_000 - 10),
        (100_000, 400_000, None, 380_000 - 1_000 - 10),
        (100_000, 0, True, 0),
        (0, 0, None, 0),
        (1, 0, None, 1),
        (1_000, 0, None, 1),
    ],
    ids=[
        "unknown-rejects",
        "rejects",
        "shared-window",
        "truncates",
        "no-window",
        "one-token-window",
        "full",
    ],
)
def test_the_reply_may_hold_what_the_window_leaves(
    tmp_path: Path,
    window: int,
    context: int,
    truncates: bool | None,
    room: int,
) -> None:
    """A vendor that stops the reply at the window needs no cap; one that rejects does."""
    limits = ModelLimits(
        max_request_tokens=window,
        max_response_tokens=8_000,
        max_context_tokens=context,
        truncates_at_context=truncates,
    )
    capability = ModelCapability(
        model_id="scripted",
        limits=limits,
    )
    _, agent = _agent(tmp_path, capability=capability)
    compactor = SummaryCompactor(agent)
    agent.session.append(_user("hi"))
    _ = agent.runtime.append_history(reply("ok", tokens=TokenCount(request=1_000)))
    _ = agent.runtime.append_history(_user("y" * 40))

    assert compactor.reply_room() == room


def test_reply_budget_leaves_room_for_token_estimation_error(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=262_144, response=262_144),
    )
    agent.runtime.tools.clear()
    agent.runtime.system = ()
    agent.session.append(_user("x" * (7_554 * 4)))

    room = SummaryCompactor(agent).reply_room()

    assert 7_690 + room <= 262_144
    assert room == 249_036 - 7_554


@pytest.mark.parametrize(("image_bytes", "due"), [(20, True), (18, False)])
def test_images_near_the_vendors_byte_ceiling_make_compaction_due(
    tmp_path: Path,
    image_bytes: int,
    due: bool,
) -> None:
    """Base64 sends 20 bytes as 26.7, past 80% of 32; 18 becomes 24, short."""
    capability = replace(
        _capability(window=1_000_000),
        limits=ModelLimits(
            max_request_tokens=1_000_000,
            max_response_tokens=8_000,
            max_request_bytes=32,
        ),
    )
    _, agent = _agent(tmp_path, capability=capability)
    image = Attachment(mime_descriptor="image/png", data=b"\0" * image_bytes)
    agent.session.append(_user("", image))
    _ = agent.runtime.append_history(reply("", tokens=TokenCount(request=2_000)))
    compactor = SummaryCompactor(agent)

    assert compactor._held() < compactor.trigger_tokens()
    assert compactor.should_run() is due


def test_a_window_of_unknown_size_is_never_due(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=0, response=0))
    agent.session.append(_user("x"))

    assert not SummaryCompactor(agent).should_run()


def _tooled(tmp_path: Path, *, tools: Sequence[_Wait]) -> SummaryCompactor:
    """Return the compactor of an agent offering ``tools``, its context one message."""
    provider = ScriptedProvider(capability=_capability())
    agent = Agent(
        model=provider["scripted"],
        tools=tools,
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
        compactor=SummaryCompactor,
    )
    agent.session.append(_user("hi"))
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    return compactor


def test_an_unbilled_context_counts_the_tool_schemas_it_sends(tmp_path: Path) -> None:
    """A resumed or new session has no bill yet; its estimate must hold the tools."""
    plain = _tooled(tmp_path, tools=())
    tooled = _tooled(tmp_path, tools=(_Wait(description="d" * 40),))

    assert tooled._held() - plain._held() >= 40 // 4


def test_a_billed_context_counts_the_tool_schemas_once(tmp_path: Path) -> None:
    """The bill holds the tools already, so the estimate adds none."""
    plain = _tooled(tmp_path, tools=())
    tooled = _tooled(tmp_path, tools=(_Wait(description="d" * 40),))
    for compactor in (plain, tooled):
        _ = compactor._agent.runtime.append_history(
            reply("ok", tokens=TokenCount(request=5_000)),
        )

    assert tooled._held() == plain._held()


def test_the_last_replys_billed_size_counts_with_what_came_after(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=100_000))
    compactor = SummaryCompactor(agent)
    trigger = math.ceil(compactor.trigger_tokens())
    agent.session.append(_user("hi"))
    _ = agent.runtime.append_history(
        reply("ok", tokens=TokenCount(request=trigger - 10)),
    )
    _ = agent.runtime.append_history(_user("y" * 36))
    assert not compactor.should_run()

    _ = agent.runtime.append_history(_user("y" * 4))

    assert compactor.should_run()


def test_a_billed_size_counts_the_system_prompt_once(tmp_path: Path) -> None:
    """The billed request holds the system prompt, which the trigger sets aside."""
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=100_000),
        system_prompt="s" * 400,
    )
    compactor = SummaryCompactor(agent)
    system = compactor._system_tokens()
    below = system + int(compactor.trigger_tokens()) - 1
    assert system > 0
    agent.session.append(_user("hi"))
    _ = agent.runtime.append_history(reply("ok", tokens=TokenCount(request=below)))
    assert not compactor.should_run()


@pytest.mark.asyncio
async def test_what_a_summary_costs_the_agent_spends(tmp_path: Path) -> None:
    """Charged as a child's calls are: counted, told to its observers, journaled."""
    summary = reply(
        _SUMMARY_TEXT,
        tokens=TokenCount(request=100, cache_read=40, response=7),
        spend=TokenCost(response=0.25),
    )
    journal = RecordingJournal()
    provider = ScriptedProvider(replies=[summary], capability=_capability())
    agent = Agent(
        model=provider["scripted"],
        journal=journal,
        tool_context=ToolContext(cwd=tmp_path),
        compactor=SummaryCompactor,
    )
    told: list[types.RuntimeEvent] = []
    agent.runtime.observers.append(told.append)
    _ = await _compact(SummaryCompactor(agent), agent, [_user("a"), reply("b")])

    assert agent.spend_usd == 0.25
    assert [e for e in told if isinstance(e, types.Charged)] == [
        types.Charged(usd=0.25, by=""),
    ]
    assert spent(journal) == 0.25
    (usage,) = (
        record for record in journal.history() if isinstance(record, TokenUsage)
    )
    assert usage.measurement is not None
    assert usage.measurement.prompt == summary.tokens.prompt == 140
    assert usage.measurement.records == len(provider.requests[0].records) == 3
    snapshot = agent.runtime.token_accounting.capture(
        provider.requests[0],
        model=agent.model,
    )
    assert usage.measurement.frame == snapshot.frame
    assert usage.measurement.tokenizer == snapshot.tokenizer


@pytest.mark.parametrize(
    ("context", "tokens"),
    [
        ((), 0),
        ((_user("x" * 40), reply("ok", tokens=TokenCount(request=440))), 440),
        ((reply("ok", tokens=TokenCount(request=440)), _user("y" * 40)), 450),
        ((reply("ok", tokens=TokenCount(request=1)),), 1),
    ],
    ids=["empty", "billed-last", "billed-then-more", "one-token"],
)
def test_the_request_size_starts_from_the_newest_billed_reply(
    context: tuple[ModelContextEvent, ...],
    tokens: int,
) -> None:
    assert _request_tokens(ScriptedProvider()["scripted"], context) == tokens


def _said(attached: tuple[Attachment, ...]) -> ModelContextEvent:
    return _user("y" * 40, *attached)


def _returned(attached: tuple[Attachment, ...]) -> ModelContextEvent:
    return UncategorizedToolResult(call_id="c", content="y" * 40, attachments=attached)


@pytest.mark.parametrize("carrier", [_said, _returned], ids=["user", "result"])
def test_an_attachment_since_the_bill_counts_what_the_model_says(
    carrier: Callable[[tuple[Attachment, ...]], ModelContextEvent],
) -> None:
    model = ScriptedProvider()["scripted"]
    image = Attachment(mime_descriptor="image/png", data=b"x" * 400)
    billed = reply("ok", tokens=TokenCount(request=10))

    bare = _request_tokens(model, (billed, carrier(())))
    attached = _request_tokens(model, (billed, carrier((image,))))

    assert attached - bare == model.attachment_tokens(image) > 0


@pytest.mark.asyncio
async def test_the_summary_replaces_the_history_without_its_analysis(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    compactor = SummaryCompactor(agent)

    with caplog.at_level(logging.INFO, logger="sagent.agent.compactor"):
        splice = await _compact(compactor, agent, _HISTORY)

    assert splice.strategy == "summary"
    assert splice.fallback_reason == ""
    assert splice.preserved_tail_count == 0
    assert len(splice.payload) == 1
    lead = _lead(splice)
    assert lead.startswith("Another language model was working on this task")
    assert "Summary:\nthe gist\nResume work immediately without preamble." in lead
    assert "scratch notes" not in lead
    assert "compacted 4 entries, kept 0" in caplog.messages
    (request,) = provider.requests
    original = agent.runtime.request()
    assert request.system == original.system
    assert request.tools == original.tools
    assert _texts(_sent(request)) == _texts(_records(_HISTORY))
    assert _instruction(request).startswith("CRITICAL: Produce plain text exclusively.")
    assert _instruction(request).rstrip().endswith("causing task failure.")


@pytest.mark.asyncio
async def test_a_proactive_continuation_tells_the_model_to_carry_on(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)

    splice = await _compact(SummaryCompactor(agent, proactive=True), agent, _HISTORY)

    assert "Summary:\nthe gist\n\nContinue working autonomously." in _lead(splice)
    assert "Resume work immediately" not in _lead(splice)


@pytest.mark.asyncio
async def test_a_summary_quoting_a_template_slot_stays_literal(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        reply("<summary>use {{resume}} and {{recent}}</summary>"),
    )

    splice = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert "use {{resume}} and {{recent}}" in _lead(splice)


@pytest.mark.asyncio
async def test_a_custom_prompt_replaces_the_default_body(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)

    _ = await _compact(SummaryCompactor(agent, prompt="CUSTOM BODY"), agent, _HISTORY)

    instruction = _instruction(provider.requests[0])
    assert "CUSTOM BODY" in instruction
    assert "Generate a comprehensive recap" not in instruction


@pytest.mark.asyncio
async def test_a_custom_partial_prompt_is_used_when_a_tail_is_kept(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    compactor = SummaryCompactor(agent, partial_prompt="PARTIAL BODY", keep_recent=1)

    _ = await _compact(compactor, agent, (*_HISTORY, _user("next")))

    assert "PARTIAL BODY" in _instruction(provider.requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        reply("I summarized nothing."),
        reply("<summary>   </summary>"),
        reply("<analysis>still thinking", stop_reason="max_tokens"),
        reply(""),
        reply("I will not.", stop_reason="refusal"),
    ],
    ids=["no-tag", "empty-tag", "cut-off-before-it", "empty", "refused"],
)
async def test_a_reply_that_is_no_summary_preserves_history(
    tmp_path: Path,
    answer: Response,
) -> None:
    provider, agent = _agent(tmp_path, answer, answer, answer)
    _hold(agent, _HISTORY)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="summary failed after 3 attempts"):
        _ = await _live(agent, SummaryCompactor(agent))

    assert len(provider.requests) == 3
    assert all(r.records == provider.requests[0].records for r in provider.requests)
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_a_finished_summary_missing_its_closing_tag_is_kept(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, reply("<summary>all of it"))

    splice = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert splice.strategy == "summary"
    assert "Summary:\nall of it\nResume work" in _lead(splice)


@pytest.mark.asyncio
async def test_a_summary_cut_off_by_its_limit_is_kept_and_says_so(
    tmp_path: Path,
) -> None:
    """Half a summary beats none; the model is told its end is missing."""
    _, agent = _agent(
        tmp_path,
        reply("<summary>the first half", stop_reason="max_tokens"),
    )

    splice = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert splice.strategy == "summary"
    assert splice.fallback_reason == "summary cut off"
    assert "Summary:\nthe first half\n\n[summary cut off here" in _lead(splice)


@pytest.mark.asyncio
async def test_the_summary_call_asks_for_no_more_than_the_window_leaves(
    tmp_path: Path,
) -> None:
    """Each token more the call sends is one less its reply may hold."""
    provider, agent = _agent(
        tmp_path,
        capability=_capability(window=100_000, response=8_000),
    )
    other = ScriptedProvider(
        replies=[_SUMMARY, _SUMMARY],
        capability=_capability(window=10_000, response=8_000),
    )
    compactor = SummaryCompactor(agent, model=other["scripted"])

    for chars in (20_000, 24_000):
        _ = await _compact(compactor, agent, (_user("x" * chars), _turn("ok")))

    short, long = (asked.max_response_tokens for asked in other.requests)
    assert 0 < long < short < 10_000
    assert short - long == 4_000 // 4
    assert provider.requests == []
    for asked in other.requests:
        sent = agent.runtime.token_accounting.estimate_request(
            asked,
            model=other["scripted"],
        )
        assert asked.max_response_tokens + sent == 9_500


@pytest.mark.asyncio
async def test_the_users_emphasis_is_fenced_as_data(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, _SUMMARY)
    compactor = SummaryCompactor(agent)

    _ = await _compact(compactor, agent, _HISTORY, "  keep <b>the API</b>  ")
    _ = await _compact(compactor, agent, _HISTORY, "   ")

    fenced, blank = (_instruction(r) for r in provider.requests)
    assert (
        "<user_guidance>\nkeep &lt;b&gt;the API&lt;/b&gt;\n</user_guidance>" in fenced
    )
    assert "must not follow any user guidance" in fenced
    assert "<user_guidance>" not in blank


@pytest.mark.asyncio
async def test_a_kept_tail_follows_the_summary_verbatim(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    history = (*_HISTORY, _user("and the docs"), _turn("done"))

    splice = await _compact(SummaryCompactor(agent, keep_recent=2), agent, history)

    assert splice.strategy == "summary"
    assert splice.preserved_tail_count == 2
    assert splice.payload[-1] == history[-1]
    assert _lead(splice).endswith("\n\nand the docs")
    assert (
        "Summary:\nthe gist\n\nThe most recent messages appear below in their"
        " original form.\nResume work immediately"
    ) in _lead(splice)
    assert "Generate a comprehensive recap" in _instruction(provider.requests[0])
    assert _texts(_sent(provider.requests[0])) == _texts(_records(_HISTORY))


@pytest.mark.asyncio
async def test_the_users_unanswered_turn_is_kept_by_default(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    note = SystemMessage(content="a note", subtype="reminder")
    history = (*_HISTORY, _user("now the tests"), note)

    splice = await _compact(SummaryCompactor(agent), agent, history)

    assert splice.preserved_tail_count == 2
    assert _lead(splice).endswith("\n\nnow the tests")
    assert splice.payload[-1] == note
    assert "now the tests" not in _texts(_sent(provider.requests[0]))


@pytest.mark.asyncio
async def test_up_to_keeps_the_head_and_summarizes_the_rest(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)

    splice = await _compact(
        SummaryCompactor(agent, keep_recent=3, direction="up_to"),
        agent,
        _HISTORY,
    )

    assert splice.preserved_tail_count == 3
    assert splice.payload[:3] == _HISTORY[:3]
    last = splice.payload[-1]
    assert isinstance(last, UserMessage)
    assert "earlier messages appear above in their original form" in (
        last.content or ""
    )
    assert "the EARLIER messages are preserved verbatim" in _instruction(
        provider.requests[0],
    )
    assert "The first 4 entries are already retained." in _instruction(
        provider.requests[0],
    )


@pytest.mark.asyncio
async def test_the_kept_tail_never_cuts_a_call_from_its_result(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    history = (
        _user("go"),
        _turn("", call("Read", "c1")),
        _result("c1"),
        _turn("", call("Read", "c2")),
        _result("c2"),
        _user("next"),
    )

    splice = await _compact(SummaryCompactor(agent, keep_recent=2), agent, history)

    assert splice.preserved_tail_count == 3
    assert splice.payload[1:] == history[3:]


@pytest.mark.asyncio
async def test_a_tail_an_open_call_spans_is_summarized_with_the_rest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    history = (_turn("", call("Read", "c1")), _result("c1"))

    with caplog.at_level(logging.WARNING):
        splice = await _compact(SummaryCompactor(agent, keep_recent=1), agent, history)

    assert splice.preserved_tail_count == 0
    assert (
        "compaction keep_recent=1 kept no tail: an unanswered tool call spans the"
        " boundary; summarizing everything."
    ) in caplog.messages
    sent = _sent(provider.requests[0])
    assert sent == _records(history)


@pytest.mark.asyncio
async def test_attachments_are_sent_unchanged_and_kept_in_the_tail(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    image = Attachment(mime_descriptor="image/png", data=b"png")
    pdf = Attachment(mime_descriptor="application/pdf", data=b"pdf")
    other = Attachment(mime_descriptor="audio/wav", data=b"wav")
    shot = UncategorizedToolResult(
        call_id="c1",
        content="shot",
        attachments=(image,),
        is_error=True,
        stub="pending",
    )
    history = (
        _user("look", image, pdf),
        _turn("", call("Read", "c1")),
        shot,
        _user("", other),
        _turn("seen"),
        _user("again", image),
    )

    splice = await _compact(SummaryCompactor(agent), agent, history)

    sent = _sent(provider.requests[0])
    assert _texts(sent) == ["look", "shot", "", "seen"]
    assert sent == _records(history[:-1])
    assert any(r.attachments for r in sent if isinstance(r, (UserMessage, ToolResult)))
    marked = next(r for r in sent if isinstance(r, ToolResult))
    assert (marked.call_id, marked.is_error, marked.stub) == ("c1", True, "pending")
    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    assert first.attachments == (image,)


@pytest.mark.asyncio
async def test_the_summary_request_preserves_notes_peers_and_skill_bodies(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    history = (
        _user("go"),
        TurnContext(cwd=str(tmp_path)),
        _turn("", call("Read", "c1"), call("Read", "c2")),
        _result("c1", "first result"),
        _result("c2", "second result"),
        AgentToAgentMessage(content="status", sender="reviewer"),
        _turn("", call("Skill", "s1")),
        AgentToAgentMessage(content="peer note", sender="reviewer"),
        _result("s1", "the whole skill body"),
        SystemMessage(content="a reminder"),
        _turn("done"),
    )

    _ = await _compact(SummaryCompactor(agent), agent, history)

    assert _sent(provider.requests[0]) == _records(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["from", "up_to"])
async def test_an_overflow_keeps_results_whole_and_preserves_the_unselected_suffix(
    tmp_path: Path,
    direction: Literal["from", "up_to"],
) -> None:
    provider, agent = _agent(tmp_path, ContextOverflowError("too long"), _SUMMARY)
    history = (
        _user("go"),
        _turn("", call("Read", "c1")),
        _result("c1", "r" * 20_000),
        _turn("ok"),
    )

    splice = await _compact(
        SummaryCompactor(agent, direction=direction),
        agent,
        history,
    )

    first, second = provider.requests
    assert _sent(first) == _records(history)
    assert _sent(second) == _records(history[:-1])
    assert first.system == second.system == agent.runtime.system
    assert first.tools == second.tools != ()
    assert splice.payload[-1] == history[-1]
    assert splice.payload.count(history[-1]) == 1


@pytest.mark.asyncio
async def test_an_overflow_with_small_results_retains_the_oldest_prefix(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, ContextOverflowError("too long"), _SUMMARY)
    history = tuple(
        e for i in range(5) for e in (_user(f"ask {i}"), _turn(f"answer {i}"))
    )

    _ = await _compact(SummaryCompactor(agent, keep_recent=0), agent, history)

    first, second = (_texts(_sent(r)) for r in provider.requests)
    assert first[0] == "ask 0"
    assert second == first[:-1]


@pytest.mark.asyncio
async def test_an_overflowing_indivisible_pair_is_not_discarded(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, ContextOverflowError("too long"))
    history = (_turn("", call("Read", "c1")), _result("c1"))

    with pytest.raises(ContextOverflowError, match="no nonempty prefix"):
        _ = await _compact(SummaryCompactor(agent), agent, history)

    assert len(provider.requests) == 1
    assert _sent(provider.requests[0]) == _records(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", [1, 2, None], ids=["one", "two", "default"])
async def test_a_summary_that_overflows_every_attempt_errors_with_bounded_calls(
    tmp_path: Path,
    attempts: int | None,
) -> None:
    provider, agent = _agent(tmp_path, *[ContextOverflowError("too long")] * 5)
    history = (
        _user("go"),
        _turn("", call("Read", "c1")),
        _result("c1", "r" * 20_000),
        _turn("ok"),
    )
    compactor = (
        SummaryCompactor(agent)
        if attempts is None
        else SummaryCompactor(agent, max_attempts=attempts)
    )

    expected = 3 if attempts is None else attempts
    with pytest.raises(
        ContextOverflowError,
        match=f"summary failed after {expected} attempts",
    ):
        _ = await _compact(compactor, agent, history)

    assert len(provider.requests) == expected
    first = _texts(_sent(provider.requests[0]))[1]
    assert first == "r" * 20_000


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, agent = _agent(
        tmp_path,
        OverloadedError("busy", retry_after_sec=0.0),
        _SUMMARY,
    )

    with caplog.at_level(logging.WARNING, logger="sagent.agent.compactor"):
        splice = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert splice.strategy == "summary"
    assert len(provider.requests) == 2
    assert caplog.messages == ["summary call failed (busy); waiting 0.0s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", [1, 3])
async def test_repeated_transient_failures_sleep_only_between_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempts: int,
) -> None:
    waited: list[float] = []

    async def sleep(delay: float) -> None:
        waited.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    provider, agent = _agent(
        tmp_path,
        *[OverloadedError("busy", retry_after_sec=0.25)] * attempts,
    )

    with pytest.raises(
        ContextOverflowError,
        match=f"summary failed after {attempts} attempts",
    ):
        _ = await _compact(
            SummaryCompactor(agent, max_attempts=attempts),
            agent,
            _HISTORY,
        )

    assert len(provider.requests) == attempts
    assert waited == [0.25] * (attempts - 1)


@pytest.mark.asyncio
async def test_any_other_failure_propagates(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, ValueError("bad request"))

    with pytest.raises(ValueError, match="bad request"):
        _ = await _compact(SummaryCompactor(agent), agent, _HISTORY)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["from", "up_to"])
async def test_nothing_to_summarize_keeps_the_tail_and_calls_no_model(
    tmp_path: Path,
    direction: Literal["from", "up_to"],
) -> None:
    provider, agent = _agent(tmp_path)

    splice = await _compact(
        SummaryCompactor(agent, direction=direction),
        agent,
        [_user("only this")],
    )

    assert splice.payload == (_user("only this"),)
    assert splice.strategy == "summary_fallback"
    assert splice.fallback_reason == "nothing to summarize"
    assert splice.preserved_tail_count == 1
    assert provider.requests == []


@pytest.mark.asyncio
async def test_the_token_counts_are_the_models_estimate(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        _SUMMARY,
        capability=ModelCapability(model_id="scripted", approx_chars_per_token=2.0),
    )
    history = [_user("x" * 400), _turn("y" * 200)]

    splice = await _compact(SummaryCompactor(agent), agent, history)

    assert splice.token_before == 300
    assert splice.token_after == len(_lead(splice)) // 2


@pytest.mark.asyncio
async def test_the_size_before_is_the_last_bill_plus_what_followed_it(
    tmp_path: Path,
) -> None:
    """Only what no bill has counted yet is estimated."""
    _, agent = _agent(tmp_path, _SUMMARY)
    billed = reply("ok", tokens=TokenCount(request=750_000))
    history = [_user("go"), billed, _user("y" * 40)]

    splice = await _compact(SummaryCompactor(agent), agent, history)

    assert splice.token_before == 750_000 + 40 // 4


@pytest.mark.asyncio
async def test_another_model_can_write_the_summary(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path)
    other = ScriptedProvider(replies=[_SUMMARY])

    splice = await _compact(
        SummaryCompactor(agent, model=other["scripted"]),
        agent,
        _HISTORY,
    )

    assert splice.strategy == "summary"
    assert provider.requests == []
    assert len(other.requests) == 1


class _Wordless(ScriptedModel):
    """A model whose stream breaks its contract: words, and then no reply."""

    @override
    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        del request
        yield TextChunk(text="thinking aloud")


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_a_reply_is_an_error(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path)
    provider = ScriptedProvider()
    wordless = _Wordless(
        provider=provider,
        capability=provider.capability,
        settings=provider.capability.default,
    )

    with pytest.raises(
        RuntimeError,
        match=r"^The model ended its stream without a reply\.$",
    ):
        _ = await _compact(SummaryCompactor(agent, model=wordless), agent, _HISTORY)


@pytest.mark.asyncio
async def test_a_kept_head_that_is_all_there_is_leads_with_the_notes(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path)
    _start_a_job(agent)
    only = _turn("all there is")

    splice = await _compact(
        SummaryCompactor(agent, keep_recent=5, direction="up_to"),
        agent,
        [only],
    )

    assert splice.fallback_reason == "nothing to summarize"
    assert provider.requests == []
    assert splice.payload[1:] == (only,)
    assert _lead(splice).startswith("Active background tasks")


def _skilled(
    tmp_path: Path,
    *bodies: tuple[str, str],
    window: int = 20_000,
) -> Agent:
    """Return an agent whose Skill tool finds each ``(name, body)`` in ``tmp_path``."""
    for name, body in bodies:
        home = tmp_path / ".sagent" / "skills" / name
        home.mkdir(parents=True)
        (home / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\n{body}\n",
            encoding="utf-8",
        )
    provider = ScriptedProvider(
        replies=[_SUMMARY, _SUMMARY],
        capability=_capability(window=window),
    )
    return Agent(
        model=provider["scripted"],
        tools=[Skill(user_dir=tmp_path / "user")],
        tool_context=ToolContext(cwd=tmp_path),
        compactor=SummaryCompactor,
    )


def _loaded(name: str, call_id: str) -> list[ModelContextEvent]:
    """Return a window that loaded skill ``name``."""
    return [
        _user("go"),
        _turn("", call("Skill", call_id, skill=name)),
        _result(call_id, f"<skill name='{name}' source='project'>\nsteps\n</skill>"),
        _turn("done"),
    ]


@pytest.mark.asyncio
async def test_a_skill_loaded_before_compaction_is_restored_through_each_one(
    tmp_path: Path,
) -> None:
    agent = _skilled(tmp_path, ("tdd", "Write the test first."), ("old", "Old."))
    history = [*_loaded("old", "c0"), *_loaded("tdd", "c1")]

    first = await _compact(SummaryCompactor(agent), agent, history)
    second = await _compact(
        SummaryCompactor(agent),
        agent,
        [*first.payload, _turn("ok")],
    )

    for lead in (_lead(first), _lead(second)):
        assert (
            "Skills loaded before compaction, restored:\n\n"
            "<skill name='tdd' source='project'>\nWrite the test first.\n</skill>"
            in lead
        )
        assert lead.index("name='tdd'") < lead.index("name='old'")


@pytest.mark.asyncio
async def test_a_skill_block_the_user_pasted_is_no_skill_loaded(tmp_path: Path) -> None:
    """A handoff quoting a restored skill names it; the agent never loaded it."""
    agent = _skilled(tmp_path, ("tdd", "Write the test first."))
    pasted = _user(
        "Last session's summary:\n"
        "<skill name='tdd' source='project'>\nWrite the test first.\n</skill>",
    )

    lead = _lead(await _compact(SummaryCompactor(agent), agent, [pasted, _turn("ok")]))

    assert "Skills loaded before compaction" not in lead


@pytest.mark.asyncio
async def test_a_skill_too_long_to_restore_is_named_not_cut(tmp_path: Path) -> None:
    agent = _skilled(tmp_path, ("big", "x" * 40_000), ("small", "Short."))
    history = [*_loaded("big", "c0"), *_loaded("small", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "Short." in lead
    assert "x" * 100 not in lead
    assert (
        "</skill>\n\nSkills loaded before compaction but too long to restore;"
        " re-load with Skill: big."
    ) in lead


@pytest.mark.asyncio
async def test_a_skill_deleted_since_it_loaded_is_not_restored(tmp_path: Path) -> None:
    agent = _skilled(tmp_path, ("tdd", "Write the test first."))
    history = [*_loaded("gone", "c0"), *_loaded("tdd", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "name='tdd'" in lead
    assert "gone" not in lead


@pytest.mark.asyncio
async def test_a_skill_that_failed_to_load_is_not_restored(tmp_path: Path) -> None:
    agent = _skilled(tmp_path, ("tdd", "Write the test first."))
    history = [
        _user("go"),
        _turn("", call("Skill", "c1", skill="tdd")),
        UncategorizedToolResult(call_id="c1", content="no", is_error=True),
        _turn("done"),
    ]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "<skill" not in lead


@pytest.mark.asyncio
async def test_the_notes_join_the_first_user_message_past_a_kept_turn(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    _start_a_job(agent)
    head = _turn("kept head")

    splice = await _compact(
        SummaryCompactor(agent, keep_recent=1, direction="up_to"),
        agent,
        [head, _user("summarized"), _turn("this too")],
    )

    assert splice.payload[0] == head
    last = splice.payload[1]
    assert isinstance(last, UserMessage)
    assert "Summary:\nthe gist" in (last.content or "")
    assert (last.content or "").endswith("- [w1] Wait: running (0s since started)")


def _start_a_job(agent: Agent) -> None:
    """Have ``agent``'s runtime hold one call still running."""
    agent.runtime._started["w1"] = (ToolCall(call_id="w1", name="Wait"), time.time())


@dataclass(slots=True, kw_only=True)
class _EffortSeen:
    """Answer with a summary, noting the effort the model was set to."""

    agent: Agent
    answer: Reply = _SUMMARY
    seen: list[str] = field(default_factory=list)

    def __call__(self, request: Request) -> Response:
        del request
        self.seen.append(self.agent.model.settings.thinking_effort)
        if isinstance(self.answer, BaseException):
            raise self.answer
        assert isinstance(self.answer, Response)
        return self.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effort", "during"),
    [("low", "low"), ("min", "high"), ("max", "high")],
    ids=["lowered", "not-offered", "not-lower"],
)
async def test_the_summary_runs_at_a_lower_effort_then_restores_it(
    tmp_path: Path,
    effort: ThinkingEffort,
    during: str,
) -> None:
    provider = ScriptedProvider(
        capability=_capability(efforts=frozenset({"none", "low", "high", "max"})),
    )
    agent = Agent(
        model=provider["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
    )
    agent.model.settings = replace(agent.model.settings, thinking_effort="high")
    seen = _EffortSeen(agent=agent)
    provider.replies.append(seen)

    _ = await _compact(SummaryCompactor(agent, effort=effort), agent, _HISTORY)

    assert seen.seen == [during]
    assert agent.model.settings.thinking_effort == "high"


@pytest.mark.asyncio
async def test_the_effort_is_restored_after_a_failure(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        capability=_capability(efforts=frozenset({"none", "high"})),
    )
    agent = Agent(
        model=provider["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
    )
    agent.model.settings = replace(agent.model.settings, thinking_effort="high")
    seen = _EffortSeen(agent=agent, answer=ValueError("bad"))
    provider.replies.append(seen)

    with pytest.raises(ValueError, match="bad"):
        _ = await _compact(SummaryCompactor(agent, effort="none"), agent, _HISTORY)

    assert seen.seen == ["none"]
    assert agent.model.settings.thinking_effort == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second", "kept"),
    [
        (
            reply("<summary>the gist, with the file paths</summary>"),
            "the gist, with the file paths",
        ),
        (reply("IDENTICAL"), "the gist"),
        (reply("no tags at all"), "the gist"),
        (reply("<summary>   </summary>"), "the gist"),
        (reply("<summary>better but cut", stop_reason="max_tokens"), "the gist"),
        (ValueError("verifier down"), "the gist"),
        (ContextOverflowError("too long"), "the gist"),
        (reply("<summary>refused</summary>", stop_reason="refusal"), "the gist"),
        (reply("<summary>tool call</summary>", calls=[call("Read", "c9")]), "the gist"),
    ],
    ids=[
        "improved",
        "identical",
        "unparseable",
        "empty",
        "cut-off",
        "failed",
        "overflowed",
        "refused",
        "tool-call",
    ],
)
async def test_a_verified_summary_is_replaced_only_by_a_better_one(
    tmp_path: Path,
    second: Reply,
    kept: str,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, second)

    splice = await _compact(
        SummaryCompactor(agent, verify_summary=True),
        agent,
        _HISTORY,
    )

    assert f"Summary:\n{kept}" in _lead(splice)
    assert len(provider.requests) == 2
    first, verification = provider.requests
    assert _sent(verification) == _sent(first)
    assert verification.system == first.system
    assert verification.tools == first.tools
    probe = _instruction(verification)
    assert f"<previous_summary>\n{_SUMMARY_TEXT}\n</previous_summary>" in probe


@pytest.mark.asyncio
async def test_an_unverified_summary_is_one_call(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, _SUMMARY)

    _ = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert len(provider.requests) == 1


# The model reads again, in one turn, what it still needs: Write refuses a file not
# read since the compaction, and Edit's exact old_string fails on one not seen.
@pytest.mark.asyncio
async def test_a_compaction_brings_back_no_file_the_model_read(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    read = tmp_path / "encode.rs"
    _ = read.write_text("fn encode() {}\n")
    agent.runtime.tool_context.files.saw(read)

    splice = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    lead = _lead(splice)
    assert "fn encode()" not in lead
    assert str(read) not in lead
    assert splice.token_after < 1_000


@pytest.mark.asyncio
async def test_thinking_counts_toward_the_token_estimate(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    thought = Response(
        records=(Thinking(content="t" * 400), AssistantMessage(content="ok")),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )

    splice = await _compact(SummaryCompactor(agent), agent, [_user("go"), thought])

    assert splice.token_before == (len("go") + 400 + len("ok")) // 4


@dataclass(slots=True, kw_only=True)
class _Answer:
    """Answer summary requests with ``summary`` and record their cost."""

    summary: str = "the gist"
    usd: float = 0.0

    def __call__(self, request: Request) -> Response:
        del request
        return reply(
            f"<summary>{self.summary}</summary>",
            spend=TokenCost(response=self.usd),
        )


_TAIL: tuple[ModelContextEvent, ...] = (
    _user("a" * 2_800),
    _turn("b" * 2_800),
    _user("c" * 2_800),
    _turn("d" * 2_800),
)


def _unreserved(agent: Agent, *, keep_recent: int = 0) -> SummaryCompactor:
    """Return a compactor that holds no room back for its summary or its prompt."""
    return SummaryCompactor(
        agent,
        keep_recent=keep_recent,
        prompt="",
        partial_prompt="",
        summary_tokens=lambda _: 0.0,
    )


def _tiny(
    tmp_path: Path,
    answer: _Answer,
    *,
    system_prompt: str = "",
    window: int = 2_000,
    response: int = 100,
) -> tuple[ScriptedProvider, Agent]:
    return _agent(
        tmp_path,
        *[answer] * 50,
        capability=_capability(window=window, response=response),
        system_prompt=system_prompt,
    )


@pytest.mark.asyncio
async def test_a_kept_message_larger_than_the_window_errors_without_folding(
    tmp_path: Path,
) -> None:
    provider, agent = _tiny(tmp_path, _Answer())
    huge = _user("HEAD line\n" + "".join(f"line {i} {'d' * 90}\n" for i in range(200)))
    _hold(agent, (*_HISTORY, huge))
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError):
        _ = await _live(agent, _unreserved(agent))

    assert agent.runtime.context().messages == before
    assert all(huge not in request.records for request in provider.requests)
    assert len(provider.requests) <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [10, 1])
async def test_a_frame_leaving_no_nonempty_prefix_errors_without_sending(
    tmp_path: Path,
    target: int,
) -> None:
    _, sizing = _agent(tmp_path, capability=_capability(window=10_000, response=100))
    fixed = _unreserved(sizing)._system_tokens() + _unreserved(sizing)._prompt_tokens()
    provider, agent = _agent(
        tmp_path,
        capability=_capability(window=fixed + target, response=100),
    )
    huge = _user("".join(f"line {i} {'d' * 90}\n" for i in range(200)))
    _hold(agent, (*_HISTORY, huge))
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="no nonempty prefix"):
        _ = await _live(agent, _unreserved(agent))

    assert provider.requests == []
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_a_tail_too_large_together_errors_without_summarizing_it(
    tmp_path: Path,
) -> None:
    provider, agent = _tiny(tmp_path, _Answer())
    _hold(agent, (*_HISTORY, *_TAIL))
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError):
        _ = await _live(agent, _unreserved(agent, keep_recent=4))

    assert agent.runtime.context().messages == before
    assert all(
        record not in request.records
        for record in _records(_TAIL)
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_a_completed_pass_that_grows_errors_without_replacing_history(
    tmp_path: Path,
) -> None:
    bloated = reply(f"<summary>{'s' * 8_000}</summary>")
    provider, agent = _agent(
        tmp_path,
        bloated,
        capability=_capability(window=2_000, response=100),
    )
    _hold(agent, _HISTORY)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="did not shrink"):
        _ = await _live(agent, _unreserved(agent))

    assert len(provider.requests) == 1
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_a_system_prompt_that_fills_the_window_overflows(tmp_path: Path) -> None:
    _, agent = _tiny(tmp_path, _Answer(), system_prompt="p" * 8_000)

    with pytest.raises(ContextOverflowError, match="no nonempty prefix"):
        _ = await _compact(_unreserved(agent), agent, _HISTORY)


@pytest.mark.asyncio
async def test_run_compacts_at_the_next_safe_point_and_forgets_read_files(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, reply("resumed"))
    seen = tmp_path / "seen.txt"
    _ = seen.write_text("x")
    async with agent:
        agent.session.append(_user("fix the bug"), AssistantMessage(content="fixed"))
        agent.runtime.tool_context.files.saw(seen)

        await agent.session.compactor.run("keep the plan")

        first = next(iter(agent.session))
        assert isinstance(first, UserMessage)
        assert "Summary:\nthe gist" in (first.content or "")
        assert seen not in agent.runtime.tool_context.files
    assert "keep the plan" in _instruction(provider.requests[0])


@pytest.mark.asyncio
async def test_a_compaction_asked_for_is_manual_and_marks_the_next_call(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, reply("one"), reply("two"))
    async with agent:
        agent.session.append(_user("fix the bug"), AssistantMessage(content="fixed"))
        await agent.session.compactor.run()
        # Looking at the next request sends nothing, so counts no call.
        _ = agent.runtime.request()
        _ = agent.runtime.request()
        assert await agent.send("go on") == "two"
    compaction, first, second = provider.requests
    assert [r.purpose for r in provider.requests] == ["compaction", "main", "main"]
    assert [r.compaction for r in provider.requests] == ["manual", "manual", ""]
    assert [r.window for r in provider.requests] == [0, 1, 1]
    assert compaction.session == first.session == second.session != ""
    assert first.turn < second.turn
    # Calls count within a turn; the turn sent records when it began.
    assert [r.step for r in provider.requests] == [0, 1, 0]
    assert second.turn_started_ms > 0


@pytest.mark.asyncio
async def test_a_resumed_agent_goes_on_counting_its_turns_and_compactions(
    tmp_path: Path,
) -> None:
    """A resume that counted from zero would reuse an earlier turn's ids."""
    path = tmp_path / "s" / "session.jsonl"
    async with JsonlJournal(path) as journal:
        first = ScriptedProvider(replies=[reply("one"), _SUMMARY])
        agent = Agent(
            model=first["scripted"],
            tool_context=ToolContext(cwd=tmp_path),
            journal=journal,
            compactor=SummaryCompactor,
        )
        async with agent:
            assert await agent.send("go") == "one"
            await agent.session.compactor.run()
    async with JsonlJournal(path) as journal:
        again = ScriptedProvider(replies=[reply("two")])
        agent = Agent(
            model=again["scripted"],
            tool_context=ToolContext(cwd=tmp_path),
            journal=journal,
            compactor=SummaryCompactor,
        )
        async with agent:
            assert await agent.send("go on") == "two"
    before, after = first.requests[0], again.requests[0]
    assert after.turn == before.turn + 1
    assert after.prompt_id != before.prompt_id
    assert after.window == 1


@pytest.mark.asyncio
async def test_run_without_instructions_asks_for_no_emphasis(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    async with agent:
        agent.session.append(_user("fix the bug"), AssistantMessage(content="fixed"))

        await agent.session.compactor.run()

    assert "<user_guidance>" not in _instruction(provider.requests[0])


@pytest.mark.asyncio
async def test_run_raises_what_the_compaction_raised(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, ValueError("no summary today"))
    async with agent:
        agent.session.append(_user("fix the bug"), AssistantMessage(content="fixed"))

        with pytest.raises(ValueError, match="no summary today"):
            await agent.session.compactor.run()


@pytest.mark.asyncio
async def test_a_due_compaction_runs_before_the_model_call(tmp_path: Path) -> None:
    provider, agent = _agent(
        tmp_path,
        _SUMMARY,
        reply("answered"),
        capability=_capability(window=4_000, response=100),
    )
    async with agent:
        for letter in "xyz":
            agent.session.append(
                _user(letter * 3_000),
                AssistantMessage(content="noted"),
            )

        assert await agent.send("go on") == "answered"

    compaction, answer = provider.requests
    assert (compaction.purpose, answer.purpose) == ("compaction", "main")
    assert (compaction.compaction, answer.compaction) == ("auto", "auto")
    assert (compaction.window, answer.window) == (0, 1)
    # The summary is asked as the agent's own next request, so it reads its cache.
    assert compaction.system == answer.system
    assert compaction.records[0] == _user("x" * 3_000)
    assert _instruction(compaction).startswith("CRITICAL: Produce plain text")
    first = answer.records[0]
    assert isinstance(first, UserMessage)
    assert "Summary:\nthe gist" in (first.content or "")
    assert "x" * 3_000 not in _texts(answer.records)


@pytest.mark.asyncio
async def test_a_reply_the_summary_keeps_does_not_make_it_due_again(
    tmp_path: Path,
) -> None:
    """A kept reply was billed for the history the summary replaced."""
    provider = ScriptedProvider(
        replies=(_SUMMARY, reply("answered")),
        capability=_capability(window=4_000, response=100),
    )
    agent = Agent(
        model=provider["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
        compactor=partial(SummaryCompactor, keep_recent=2),
    )
    async with agent:
        _hold(
            agent,
            [
                _user("x" * 3_000),
                reply("first"),
                _user("y" * 3_000),
                reply("second"),
                _user("z" * 3_000),
                reply("noted", tokens=TokenCount(request=3_900)),
            ],
        )

        assert await agent.send("go on") == "answered"
        assert not agent.session.compactor.should_run()

    assert len(provider.requests) == 2


def _hold(agent: Agent, history: Sequence[ModelContextEvent]) -> None:
    """Put ``history`` in the agent's runtime, as its context."""
    for entry in history:
        agent.runtime.append_history(entry)


async def _live(agent: Agent, compactor: SummaryCompactor) -> ContextSplice:
    """Compact what the agent's runtime holds, as the runtime would."""
    return await _compact(compactor, agent, agent.runtime.context().messages)


@pytest.mark.asyncio
async def test_a_summary_of_the_live_context_is_its_next_request_and_more(
    tmp_path: Path,
) -> None:
    """Prefix caching hits only when tools, system, and records start the same."""
    provider, agent = _agent(tmp_path, _SUMMARY, system_prompt="S")
    _hold(agent, _HISTORY)
    expected = agent.runtime.request()

    splice = await _live(agent, SummaryCompactor(agent))

    (asked,) = provider.requests
    assert splice.strategy == "summary"
    assert (asked.system, asked.tools, asked.allowed) == (
        expected.system,
        expected.tools,
        expected.allowed,
    )
    assert asked.records[:-1] == expected.records
    assert _instruction(asked).startswith("CRITICAL: Produce plain text")


@pytest.mark.parametrize(
    "first",
    [
        reply(calls=[call("Read", "c9", file_path="a.py")]),
        reply("<summary>invalid with tools</summary>", calls=[call("Read", "c9")]),
        ContextOverflowError("too long"),
        RateLimitedError("busy", retry_after_sec=0.0),
    ],
    ids=["calls-a-tool", "summary-and-tool", "overflows", "rate-limited"],
)
@pytest.mark.asyncio
async def test_a_live_summary_retry_keeps_the_original_frame(
    tmp_path: Path,
    first: Reply,
) -> None:
    provider, agent = _agent(tmp_path, first, _SUMMARY, system_prompt="S")
    _hold(agent, _HISTORY)
    original = agent.runtime.request()

    splice = await _live(agent, SummaryCompactor(agent))

    initial, retried = provider.requests
    assert initial.purpose == retried.purpose == "compaction"
    assert initial.tools == retried.tools == original.tools
    assert initial.system == retried.system == original.system
    assert initial.allowed == retried.allowed == original.allowed
    assert _instruction(initial) == _instruction(retried)
    if isinstance(first, ContextOverflowError):
        assert _sent(retried) == _sent(initial)[:-1]
    else:
        assert retried.records == initial.records
    assert splice.strategy == "summary"
    assert "invalid with tools" not in _lead(splice)


@pytest.mark.parametrize(
    "options",
    [{"effort": "low"}, {"direction": "up_to", "keep_recent": 1}, {"model": "own"}],
    ids=["own-effort", "head-kept", "own-model"],
)
@pytest.mark.asyncio
async def test_a_custom_model_effort_or_direction_keeps_the_original_frame(
    tmp_path: Path,
    options: Mapping[str, object],
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, _SUMMARY, system_prompt="S")
    _hold(agent, _HISTORY)
    own = ScriptedProvider(replies=[_SUMMARY])
    compactor = SummaryCompactor(
        agent,
        effort="low" if "effort" in options else None,
        direction="up_to" if "direction" in options else "from",
        keep_recent=1 if "keep_recent" in options else 0,
        model=own["scripted"] if "model" in options else None,
    )

    _ = await _live(agent, compactor)

    (asked,) = provider.requests or own.requests
    original = agent.runtime.request()
    assert asked.tools == original.tools
    assert asked.system == original.system
    assert asked.allowed == original.allowed
    assert _sent(asked) == original.records


@dataclass(slots=True, kw_only=True)
class _Wait:
    """A tool that runs until the test ends."""

    name: str = "Wait"
    description: str = "Wait."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        self.started.set()
        await asyncio.Event().wait()
        return UncategorizedToolResult(call_id="", content="never")


@pytest.mark.asyncio
async def test_a_job_still_running_is_named_after_the_summary(tmp_path: Path) -> None:
    wait = _Wait()
    provider = ScriptedProvider(
        replies=[reply(calls=[call("Wait", "w1")]), _SUMMARY, reply("resumed")],
    )
    agent = Agent(
        model=provider["scripted"],
        tools=[wait],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
        compactor=SummaryCompactor,
    )
    async with agent:
        agent.runtime.push(UserMessage(content="wait for it"))
        await wait.started.wait()

        await agent.session.compactor.run()

        first = next(iter(agent.session))
    assert isinstance(first, UserMessage)
    assert (
        "Active background tasks (re-surfaced post-compaction):\n- [w1] Wait: detached ("
        in (first.content or "")
    )


@pytest.mark.asyncio
async def test_the_prompts_read_as_utf8_whatever_the_locale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latin1 = _Latin1()
    monkeypatch.setattr(Path, "read_text", latin1)
    provider, agent = _agent(tmp_path, _SUMMARY)

    _ = await _compact(SummaryCompactor(agent), agent, _HISTORY)

    assert "your sole response \u2014 the task will fail" in _instruction(
        provider.requests[0],
    )


@dataclass(slots=True, kw_only=True)
class _Latin1:
    """Read text as a latin-1 locale would, and note the encodings asked for."""

    asked: dict[str, str | None] = field(default_factory=dict)

    def __get__(self, path: Path, owner: type[Path] | None = None) -> partial[str]:
        """Bind to the path read, as the method it stands in for would."""
        del owner
        return partial(self, path)

    def __call__(
        self,
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        del errors, newline
        self.asked[path.name] = encoding
        return path.read_bytes().decode(encoding or "latin-1")


@pytest.mark.parametrize(
    ("sizes", "budget", "k"),
    [
        ([1.0, 2.0, 3.0], 6.0, 0),  # E = 6 ≤ B: fits.
        ([1.0, 2.0, 3.0], 5.0, 2),  # 1 + 2 ≥ 3 = E/2.
        ([1.0, 1.0, 4.0], 5.0, 3),  # 1 + 1 < 3 ≤ 6.
        ([4.0, 1.0, 1.0], 5.0, 1),  # 4 ≥ 3.
        ([1.0, 1.0, 1.0, 1.0], 3.0, 2),  # 1 + 1 ≥ 2.
        ([], -1.0, 0),  # `m` = 0: nothing to cut.
    ],
)
def test_halving_follows_its_spec(sizes: list[float], budget: float, k: int) -> None:
    assert halve(sizes, budget=budget) == k


_AUDIT = reply("<analysis>notes</analysis><summary>the audit</summary>")


def _auditing(
    tmp_path: Path,
    *replies: Reply,
    journal: RecordingJournal | None = None,
    window: int = 200_000,
) -> tuple[ScriptedProvider, Agent, SummaryCompactor]:
    provider = ScriptedProvider(
        replies=replies,
        capability=_capability(window=window, response=100),
    )
    agent = Agent(
        model=provider["scripted"],
        tools=[Read()],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="S",
        journal=journal,
        compactor=partial(SummaryCompactor, style="session-audit"),
    )
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    return provider, agent, compactor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime", "marker"),
    [
        ("image/png", "[image]"),
        ("application/pdf", "[document]"),
        ("audio/wav", "[attachment]"),
    ],
)
async def test_an_audit_reads_every_window_and_says_who_said_what(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mime: str,
    marker: str,
) -> None:
    """Earlier windows come from the journal; their summaries and results do not."""
    journal = RecordingJournal()
    journal.append(
        ContextClear(system_prompt="S"),
        UserMessage(content="first ask"),
        ToolCall(call_id="c0", name="Read", arguments={"file_path": "a.py"}),
        UncategorizedToolResult(call_id="c0", content="the file body"),
        Thinking(content="hmm"),
        AssistantMessage(content="read it"),
        ContextClear(
            summary="an older summary",
            history=(UserMessage(content="kept"),),
        ),
        AgentToAgentMessage(content="peer says", sender="Agent_0", recipient="Agent"),
        UserMessage(
            content="",
            attachments=(Attachment(mime_descriptor=mime, data=b"x"),),
        ),
        SystemMessage(content="a note"),
        AssistantMessage(content=""),
    )
    provider, agent, compactor = _auditing(tmp_path, _AUDIT, journal=journal)
    _hold(agent, (_user("now this"),))

    with caplog.at_level(logging.INFO, logger="sagent.agent.compactor"):
        splice = await _compact(
            compactor,
            agent,
            agent.runtime.context().messages,
            "keep X",
        )

    assert "audited 6 session lines, kept 1" in caplog.messages
    assert splice.fallback_reason == ""
    (asked,) = provider.requests
    assert asked.tools == ()
    assert asked.system[0].startswith("Your role is to condense")
    lines = _instruction(asked).split("<session>\n")[1].removesuffix("\n</session>")
    assert lines.splitlines() == [
        "L1 USER: first ask",
        'L2 TOOL Read {"file_path": "a.py"}',
        "L3 AGENT: read it",
        "L4 Agent_0 to Agent: peer says",
        f"L5 USER: {marker}",
        "L6 USER: now this",
    ]
    assert "<user_guidance>\nkeep X\n</user_guidance>" in _instruction(asked)
    assert splice.strategy == "session_audit"
    assert "Summary:\nthe audit" in _lead(splice)
    # The message not yet answered is kept, to be answered.
    assert splice.preserved_tail_count == 1
    assert _lead(splice).endswith("\n\nnow this")


class _Threaded(RecordingJournal):
    """A journal that notes whether each read of its history ran on the loop."""

    def __init__(self) -> None:
        super().__init__()
        self.on_loop: list[bool] = []

    @override
    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        records = super().history(start)

        def read() -> Iterator[SessionRecord]:
            self.on_loop.append(threading.current_thread() is threading.main_thread())
            yield from records

        return read()


@pytest.mark.asyncio
async def test_an_audit_reads_the_journal_off_the_loop(tmp_path: Path) -> None:
    """A long session's journal takes seconds to read; the screen must not wait."""
    journal = _Threaded()
    journal.append(ContextClear(system_prompt="S"), UserMessage(content="first ask"))
    provider, agent, compactor = _auditing(tmp_path, _AUDIT, journal=journal)
    _hold(agent, (_user("now this"),))

    _ = await _compact(compactor, agent, agent.runtime.context().messages)

    assert "L1 USER: first ask" in _instruction(provider.requests[0])
    assert journal.on_loop == [False]


@pytest.mark.asyncio
async def test_without_a_journal_an_audit_reads_the_window_it_sees(
    tmp_path: Path,
) -> None:
    provider, agent, compactor = _auditing(tmp_path, _AUDIT)
    _hold(agent, _HISTORY)

    splice = await _compact(compactor, agent, agent.runtime.context().messages)

    assert "L1 USER: fix the bug" in _instruction(provider.requests[0])
    assert splice.strategy == "session_audit"


@pytest.mark.asyncio
async def test_an_audit_too_long_for_one_request_summarizes_its_older_half(
    tmp_path: Path,
) -> None:
    provider, agent, compactor = _auditing(
        tmp_path,
        reply("<summary>the older half</summary>"),
        _AUDIT,
        window=14_000,
    )
    _hold(agent, (_user("a" * 30_000), _turn("ok"), _user("b" * 30_000)))

    splice = await _compact(compactor, agent, agent.runtime.context().messages)

    digest, audit = provider.requests
    assert "a" * 30_000 in _instruction(digest)
    assert "b" * 30_000 not in _instruction(digest)
    assert "[earlier lines, summarized] Summary:\nthe older half" in _instruction(audit)
    assert "a" * 30_000 not in _instruction(audit)
    assert splice.strategy == "session_audit"


@pytest.mark.parametrize(
    "replies",
    [
        (ContextOverflowError("too long"),),
        (RateLimitedError("busy"),),
        (reply("no summary here"),),
        (reply("<summary>x</summary>", stop_reason="refusal"),),
        (reply("<summary>x</summary>", calls=[call("Read", "c9")]),),
    ],
    ids=["overflows", "rate-limited", "no-summary", "refused", "tool-call"],
)
@pytest.mark.asyncio
async def test_an_audit_that_cannot_be_had_preserves_history(
    tmp_path: Path,
    replies: tuple[Reply, ...],
) -> None:
    provider, agent, compactor = _auditing(tmp_path, *replies)
    _hold(agent, _HISTORY)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="the session audit failed"):
        _ = await _live(agent, compactor)

    assert agent.runtime.context().messages == before
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "digest",
    [
        reply("no summary here"),
        reply("<summary>" + "z" * 80_000 + "</summary>"),
        reply("<summary>x</summary>", stop_reason="refusal"),
        reply("<summary>x</summary>", calls=[call("Read", "c9")]),
    ],
    ids=["no-summary", "no-shorter", "refused", "tool-call"],
)
@pytest.mark.asyncio
async def test_an_audit_that_will_not_shrink_preserves_history(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    digest: Reply,
) -> None:
    provider, agent, compactor = _auditing(tmp_path, digest, _AUDIT, window=14_000)
    _hold(agent, (_user("a" * 30_000), _turn("ok"), _user("b" * 30_000)))

    before = agent.runtime.context().messages
    with (
        caplog.at_level(logging.WARNING, logger="sagent.agent.compactor"),
        pytest.raises(ContextOverflowError, match="the session audit failed"),
    ):
        _ = await _live(agent, compactor)

    assert len(provider.requests) == 1
    assert agent.runtime.context().messages == before
    assert "the session audit could not be cut to fit" in caplog.messages


@pytest.mark.asyncio
async def test_the_style_can_be_switched_on_a_compactor(tmp_path: Path) -> None:
    provider, agent = _agent(tmp_path, _AUDIT)
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    assert compactor.style == "summary"
    compactor.style = "session-audit"
    _hold(agent, _HISTORY)

    splice = await _compact(compactor, agent, agent.runtime.context().messages)

    assert splice.strategy == "session_audit"
    assert "<session>" in _instruction(provider.requests[0])


def test_the_share_may_be_one_fixed_value() -> None:
    assert summary_tokens(5e5, max_share=0.05) == pytest.approx(0.05 * 5e5)


def test_a_body_the_size_of_the_context_holds_exactly_the_floor() -> None:
    assert summary_tokens(1e6) == 50_000.0


@pytest.mark.parametrize(
    ("window", "context"),
    [(1, 0), (10, 1_000_000)],
    ids=["one-token-window", "input-full-output-free"],
)
def test_a_window_the_prompts_fill_has_no_room_for_a_body(
    tmp_path: Path,
    window: int,
    context: int,
) -> None:
    assert _trigger(tmp_path, window=window, context=context) == 0


def test_the_trigger_holds_room_for_the_prompt_a_kept_part_asks(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, capability=_capability(window=100_000))

    def trigger(**options: int | str) -> float:
        return SummaryCompactor(
            agent,
            prompt="",
            partial_prompt="p" * 4_000,
            summary_tokens=lambda _: 0.0,
            keep_recent=int(options.get("keep_recent", 0)),
            direction="up_to" if options.get("direction") == "up_to" else "from",
        ).trigger_tokens()

    full, tail, head = (
        trigger(),
        trigger(keep_recent=1),
        trigger(
            keep_recent=1,
            direction="up_to",
        ),
    )

    assert full - tail == 4_000 // 4
    assert head < tail


@pytest.mark.parametrize(
    ("ceiling", "said", "due"),
    [(1, 1, True), (10_000, 8_000, True), (10_000, 7_999, False)],
    ids=["one-byte", "at-the-share", "below-it"],
)
def test_the_system_prompt_and_words_count_toward_the_byte_ceiling(
    tmp_path: Path,
    ceiling: int,
    said: int,
    due: bool,
) -> None:
    """``said`` is the request's characters, the system prompt's included."""
    capability = replace(
        _capability(window=1_000_000),
        limits=ModelLimits(
            max_request_tokens=1_000_000,
            max_response_tokens=8_000,
            max_request_bytes=ceiling,
        ),
    )
    _, agent = _agent(tmp_path, capability=capability, system_prompt="s" * 400)
    system = sum(map(len, agent.runtime.system))
    assert system >= 400
    agent.session.append(_user("x" * max(1, said - system)))

    assert SummaryCompactor(agent).should_run() is due


def test_measured_reply_room_counts_the_prompt_without_response_tokens(
    tmp_path: Path,
) -> None:
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=100_000),
        system_prompt="s" * 4_000,
    )
    _hold(agent, (_user("hi"),))
    accounting = agent.runtime.token_accounting
    snapshot = accounting.capture(
        agent.runtime.request(max_response_tokens=0),
        model=agent.model,
    )
    _ = accounting.observe(snapshot, tokens=TokenCount(request=5_000, response=700))

    assert SummaryCompactor(agent).reply_room() == 95_000 - 5_000


def test_the_reply_room_counts_the_system_prompt_once(tmp_path: Path) -> None:
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=100_000),
        system_prompt="s" * 4_000,
    )
    compactor = SummaryCompactor(agent)
    agent.session.append(_user("hi"))
    _ = agent.runtime.append_history(reply("ok", tokens=TokenCount(request=5_000)))
    _ = agent.runtime.append_history(_user("y" * 40))

    assert compactor.reply_room() == 95_000 - 5_000 - 40 // 4


def test_a_bill_of_the_system_prompt_alone_holds_nothing_past_it(
    tmp_path: Path,
) -> None:
    _, agent = _agent(
        tmp_path,
        capability=_capability(window=100_000),
        system_prompt="s" * 400,
    )
    compactor = SummaryCompactor(agent)
    system = compactor._system_tokens()
    agent.session.append(_user("hi"))
    _ = agent.runtime.append_history(reply("ok", tokens=TokenCount(request=system)))

    assert compactor._held() == 0


def test_an_unbilled_context_counts_each_tools_schema(tmp_path: Path) -> None:
    plain = _tooled(tmp_path, tools=())
    schema: dict[str, PlainTree] = {
        "type": "object",
        "properties": {"x": {"description": "d" * 40}},
    }
    tooled = _tooled(tmp_path, tools=(_Wait(schema=schema),))

    assert tooled._held() - plain._held() >= 40 // 4


@pytest.mark.asyncio
async def test_a_wordless_message_takes_the_jobs_note_alone(tmp_path: Path) -> None:
    """It gains no ``None`` before the note, and names no skills."""
    _, agent = _agent(tmp_path)
    _start_a_job(agent)
    image = Attachment(mime_descriptor="image/png", data=b"x" * 400)

    splice = await _compact(
        SummaryCompactor(agent),
        agent,
        [UserMessage(attachments=(image,))],
    )

    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    assert (first.content or "").startswith("Active background tasks")
    assert first.extra == {}
    assert splice.token_before == agent.model.attachment_tokens(image)


@pytest.mark.asyncio
async def test_each_running_job_is_named_on_its_own_line(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    _start_a_job(agent)
    agent.runtime._started["w2"] = (ToolCall(call_id="w2", name="Wait"), time.time())

    lead = _lead(await _compact(SummaryCompactor(agent), agent, _HISTORY))

    assert (
        "- [w1] Wait: running (0s since started)\n"
        "- [w2] Wait: running (0s since started)"
    ) in lead


@pytest.mark.asyncio
async def test_a_kept_call_still_running_is_paired_outside_the_window(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    history = (*_HISTORY, _user("wait"), _turn("", call("Wait", "w1")))

    splice = await _compact(SummaryCompactor(agent, keep_recent=1), agent, history)

    assert splice.payload[-1] == history[-1]
    assert splice.paired_externally == frozenset({"w1"})


@pytest.mark.asyncio
async def test_a_turn_before_the_users_message_is_summarized(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)

    splice = await _compact(
        SummaryCompactor(agent),
        agent,
        (_turn("earlier"), _user("now")),
    )

    assert splice.strategy == "summary"
    assert splice.preserved_tail_count == 1
    assert _lead(splice).endswith("\n\nnow")


@pytest.mark.asyncio
async def test_a_kept_head_also_preserves_the_unanswered_tail(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    note = SystemMessage(content="a note")
    image = Attachment(mime_descriptor="image/png", data=b"png")
    history = (*_HISTORY, _user("now", image), note)

    splice = await _compact(
        SummaryCompactor(agent, keep_recent=1, direction="up_to"),
        agent,
        history,
    )

    assert splice.preserved_tail_count == 3
    assert splice.payload[-1] == note
    assert "now" not in _texts(_sent(provider.requests[0]))
    assert note not in _sent(provider.requests[0])
    assert _lead(splice).endswith("\n\nnow")
    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    assert first.attachments == (image,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        reply("I will not.", stop_reason="refusal"),
        reply("no tags"),
        ContextOverflowError("too long"),
    ],
    ids=["refused", "no-summary", "every-round-dropped"],
)
async def test_a_failed_summary_preserves_the_history_and_kept_tail(
    tmp_path: Path,
    answer: Reply,
) -> None:
    provider, agent = _agent(tmp_path, *[answer] * 3)
    _hold(agent, (*_HISTORY, _user("next")))
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError):
        _ = await _live(agent, SummaryCompactor(agent, keep_recent=1))

    assert agent.runtime.context().messages == before
    assert len(provider.requests) == 3
    assert all("next" not in _texts(_sent(r)) for r in provider.requests)


@pytest.mark.asyncio
async def test_a_failed_summary_preserves_the_history_and_kept_head(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(
        tmp_path,
        *[reply("I will not.", stop_reason="refusal")] * 3,
    )
    history = (_user("head"), _turn("kept"), _user("more"), _turn("done"))
    _hold(agent, history)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="summary failed after 3 attempts"):
        _ = await _live(
            agent,
            SummaryCompactor(agent, keep_recent=2, direction="up_to"),
        )

    assert agent.runtime.context().messages == before
    assert len(provider.requests) == 3
    assert all(_sent(r) == _records(history) for r in provider.requests)


@pytest.mark.asyncio
async def test_the_prompt_body_ends_where_its_additions_begin(tmp_path: Path) -> None:
    """Emphasis is escaped only where it could close its fence; quotes stay."""
    provider, agent = _agent(tmp_path, _SUMMARY, _SUMMARY)

    _ = await _compact(
        SummaryCompactor(agent, prompt="CUSTOM\n"),
        agent,
        _HISTORY,
        "keep \"x\" & 'y' <b>",
    )
    _ = await _compact(
        SummaryCompactor(
            agent,
            partial_prompt="PARTIAL\n",
            keep_recent=1,
            direction="up_to",
        ),
        agent,
        _HISTORY,
    )

    guided, head = (_instruction(r) for r in provider.requests)
    assert "CUSTOM\n\nUser-supplied compaction emphasis follows." in guided
    assert "<user_guidance>\nkeep \"x\" &amp; 'y' &lt;b&gt;\n</user_guidance>" in guided
    assert "PARTIAL\n\nNote: in this compaction the EARLIER messages" in head


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "kept"),
    [
        (
            "<analysis>draft <summary>old</summary></analysis><summary>real</summary>",
            "Summary:\nreal\nResume",
        ),
        (
            "<summary>the <analysis>aside</analysis>gist</summary>",
            "Summary:\nthe gist\n",
        ),
        ("<summary>one\n\n\n\ntwo</summary>", "Summary:\none\n\ntwo\nResume"),
    ],
    ids=["draft-in-analysis", "analysis-in-summary", "blank-lines"],
)
async def test_the_summary_is_read_past_its_analysis_and_blank_lines(
    tmp_path: Path,
    answer: str,
    kept: str,
) -> None:
    _, agent = _agent(tmp_path, reply(answer))

    assert kept in _lead(await _compact(SummaryCompactor(agent), agent, _HISTORY))


@pytest.mark.asyncio
async def test_a_reply_with_no_summary_says_how_long_it_was(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, agent = _agent(tmp_path, reply("I summarized nothing."))

    with (
        caplog.at_level(logging.WARNING, logger="sagent.agent.compactor"),
        pytest.raises(ContextOverflowError, match="summary failed after 1 attempts"),
    ):
        _ = await _compact(SummaryCompactor(agent, max_attempts=1), agent, _HISTORY)

    assert caplog.messages == [
        "compactor output has no usable <summary> (21 chars)",
    ]


@pytest.mark.asyncio
async def test_a_summary_said_in_parts_is_read_whole(tmp_path: Path) -> None:
    answer = Response(
        records=(
            AssistantMessage(content="<summary>the "),
            AssistantMessage(content=None),
            AssistantMessage(content="gist</summary>"),
        ),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )
    _, agent = _agent(tmp_path, answer)

    lead = _lead(await _compact(SummaryCompactor(agent), agent, _HISTORY))

    assert "Summary:\nthe gist\n" in lead


@pytest.mark.asyncio
async def test_a_calls_arguments_and_sealed_thinking_count_as_the_model_reads_them(
    tmp_path: Path,
) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)
    turn = Response(
        records=(
            Thinking(encrypted="sealed"),
            call("Read", "c1", file_path="p" * 400),
        ),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="tool_use",
    )
    arguments = '{"file_path": "' + "p" * 400 + '"}'

    splice = await _compact(
        SummaryCompactor(agent),
        agent,
        [_user("go"), turn, _result("c1")],
    )

    assert splice.token_before == (len("go") + len(arguments) + len("ok")) // 4


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["ab", "r" * 100, "r" * 20_000])
async def test_error_results_are_sent_whole_regardless_of_length(
    tmp_path: Path,
    result: str,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    returned = UncategorizedToolResult(call_id="c1", content=result, is_error=True)
    history = (_user("go"), _turn("", call("Read", "c1")), returned, _turn("ok"))

    _ = await _compact(SummaryCompactor(agent), agent, history)

    assert _sent(provider.requests[0])[2] == returned


@pytest.mark.asyncio
async def test_an_overflow_shortens_to_the_previous_safe_tool_boundary(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, ContextOverflowError("too long"), _SUMMARY)
    history = (
        *_HISTORY,
        _user("last ask"),
        _turn("", call("Read", "c2"), call("Read", "c3")),
        _result("c2", "first"),
        _result("c3", "second"),
    )

    splice = await _compact(SummaryCompactor(agent), agent, history)

    first, retry = provider.requests
    assert _sent(first) == _records(history)
    assert _sent(retry) == _records(history[:-3])
    assert splice.payload[-3:] == history[-3:]
    assert splice.paired_externally == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second", "logged"),
    [
        (reply("IDENTICAL"), []),
        (reply("identical"), []),
        (reply(""), []),
        (
            reply("no tags at all"),
            ["verifier returned no usable summary; keeping the original"],
        ),
        (
            reply("<summary>better but cut", stop_reason="max_tokens"),
            ["verifier's summary was cut off; keeping the original"],
        ),
        (
            ValueError("verifier down"),
            ["summary verification failed; using the original: verifier down"],
        ),
        (
            ContextOverflowError("too long"),
            ["summary verification failed; using the original: too long"],
        ),
    ],
    ids=[
        "identical",
        "identical-lowercase",
        "empty",
        "unparseable",
        "cut-off",
        "failed",
        "overflowed",
    ],
)
async def test_a_verification_says_why_it_kept_the_original(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    second: Reply,
    logged: list[str],
) -> None:
    _, agent = _agent(tmp_path, _SUMMARY, second)

    with caplog.at_level(logging.WARNING, logger="sagent.agent.compactor"):
        splice = await _compact(
            SummaryCompactor(agent, verify_summary=True),
            agent,
            _HISTORY,
        )

    assert "Summary:\nthe gist\n" in _lead(splice)
    assert [m for m in caplog.messages if "verif" in m] == logged


@pytest.mark.asyncio
async def test_a_live_summary_asks_for_no_more_than_the_window_leaves(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, system_prompt="S" * 400)
    _hold(agent, _HISTORY)
    compactor = SummaryCompactor(agent)

    _ = await _live(agent, compactor)

    (asked,) = provider.requests
    assert asked.tools
    held = agent.runtime.token_accounting.estimate_request(asked, model=agent.model)
    assert asked.max_response_tokens + held == 190_000


@pytest.mark.asyncio
async def test_restored_skills_with_no_message_to_join_ride_on_their_own(
    tmp_path: Path,
) -> None:
    """The names ride on the note, so the next compaction restores them again."""
    agent = _skilled(tmp_path, ("tdd", "Write the test first."))
    head = await _compact(
        SummaryCompactor(agent, keep_recent=5, direction="up_to"),
        agent,
        _loaded("tdd", "c1")[1:],
    )
    assert head.fallback_reason == "nothing to summarize"

    again = await _compact(
        SummaryCompactor(agent),
        agent,
        [head.payload[0], _turn("ok")],
    )

    assert "<skill name='tdd' source='project'>" in _lead(again)


@pytest.mark.asyncio
async def test_a_newer_skill_since_deleted_leaves_the_older_restored(
    tmp_path: Path,
) -> None:
    agent = _skilled(tmp_path, ("tdd", "Write the test first."))
    history = [*_loaded("tdd", "c0"), *_loaded("gone", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "name='tdd'" in lead


@pytest.mark.asyncio
async def test_a_newer_skill_too_long_leaves_the_older_restored(
    tmp_path: Path,
) -> None:
    agent = _skilled(tmp_path, ("big", "x" * 40_000), ("small", "Short."))
    history = [*_loaded("small", "c0"), *_loaded("big", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "Short." in lead


@pytest.mark.asyncio
async def test_skills_restored_share_one_quarter_of_the_window(
    tmp_path: Path,
) -> None:
    """Each fits alone; together they pass the quarter, so the older is named."""
    agent = _skilled(tmp_path, ("old", "o" * 8_000), ("new", "n" * 8_000))
    history = [*_loaded("old", "c0"), *_loaded("new", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "name='new'" in lead
    assert "name='old'" not in lead
    assert lead.endswith("re-load with Skill: old.")


@pytest.mark.asyncio
async def test_a_skill_exactly_the_quarter_is_restored(tmp_path: Path) -> None:
    window = next(
        w
        for w in range(20_000, 20_008)
        if SummaryCompactor(_skilled(tmp_path, window=w)).trigger_tokens() % 4 == 0
    )
    room = 0.25 * SummaryCompactor(_skilled(tmp_path, window=window)).trigger_tokens()
    frame = len("<skill name='edge' source='project'>\n\n</skill>")
    agent = _skilled(
        tmp_path / "edge",
        ("edge", "e" * (4 * int(room) - frame)),
        window=window,
    )
    assert 0.25 * SummaryCompactor(agent).trigger_tokens() == room

    lead = _lead(await _compact(SummaryCompactor(agent), agent, _loaded("edge", "c0")))

    assert "name='edge'" in lead


@pytest.mark.asyncio
async def test_skills_none_of_which_fit_are_named_with_no_restored_header(
    tmp_path: Path,
) -> None:
    agent = _skilled(tmp_path, ("big", "x" * 40_000), ("huge", "y" * 40_000))
    history = [*_loaded("big", "c0"), *_loaded("huge", "c1")]

    lead = _lead(await _compact(SummaryCompactor(agent), agent, history))

    assert "restored:" not in lead
    assert lead.endswith("re-load with Skill: huge, big.")


@pytest.mark.asyncio
async def test_without_the_skill_tool_no_skill_is_restored(tmp_path: Path) -> None:
    _, agent = _agent(tmp_path, _SUMMARY)

    lead = _lead(await _compact(SummaryCompactor(agent), agent, _loaded("tdd", "c0")))

    assert "<skill" not in lead


@pytest.mark.asyncio
async def test_a_payload_exactly_the_target_is_left_whole(tmp_path: Path) -> None:
    provider, agent = _tiny(tmp_path, _Answer(summary="q"))
    compactor = _unreserved(agent)
    probe = await _compact(compactor, agent, _HISTORY)
    request = replace(
        agent.runtime.request(max_response_tokens=0),
        records=_records(probe.payload),
    )
    fixed = agent.runtime.token_accounting.estimate_request(request, model=agent.model)
    extra = 2_000 - fixed - compactor._prompt_tokens()
    provider.replies.clear()
    provider.replies.append(_Answer(summary="q" * (1 + extra * 4)))
    provider.requests.clear()

    splice = await _compact(compactor, agent, _HISTORY)

    request = replace(request, records=_records(splice.payload))
    held = agent.runtime.token_accounting.estimate_request(request, model=agent.model)
    assert held + compactor._prompt_tokens() == 2_000
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["from", "up_to"])
async def test_a_pending_message_over_half_the_window_is_kept_whole(
    tmp_path: Path,
    direction: Literal["from", "up_to"],
) -> None:
    provider, agent = _tiny(tmp_path, _Answer(), window=4_000)
    image = Attachment(mime_descriptor="image/png", data=b"png")
    pending = _user("pending:" + "p" * 10_000, image)

    splice = await _compact(
        SummaryCompactor(agent, direction=direction),
        agent,
        (*_HISTORY, pending),
    )

    assert _lead(splice).endswith(pending.content or "")
    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    assert first.attachments == pending.attachments
    assert len(provider.requests) == 1
    assert _sent(provider.requests[0]) == _records(_HISTORY)


@pytest.mark.asyncio
async def test_every_prefix_call_reserves_reply_room_and_charges_the_agent(
    tmp_path: Path,
) -> None:
    provider, agent = _tiny(tmp_path, _Answer(usd=0.01), window=4_000)
    history = tuple(
        entry
        for i in range(5)
        for entry in (_user(f"ask {i}" + "u" * 4_000), _turn("a" * 4_000))
    )

    _ = await _compact(_unreserved(agent), agent, history)

    assert len(provider.requests) > 1
    for request in provider.requests:
        held = agent.runtime.token_accounting.estimate_request(
            request,
            model=agent.model,
        )
        assert held + request.max_response_tokens == 3_800
        assert request.session == agent.runtime.request().session != ""
        assert request.system == agent.runtime.system
        assert request.tools == agent.runtime.request().tools
    assert agent.spend_usd == pytest.approx(0.01 * len(provider.requests))


@pytest.mark.asyncio
@pytest.mark.parametrize("chars", range(40_000, 60_000, 400))
async def test_an_audits_request_and_its_summary_fit_the_window(
    tmp_path: Path,
    chars: int,
) -> None:
    """As the trigger requires: ``B + s + p ≤ R`` and ``B + s + p + S(B) ≤ C``."""
    window = 14_000
    provider, agent, compactor = _auditing(
        tmp_path,
        *[_Answer(summary="s")] * 4,
        window=window,
    )
    _hold(agent, (_user("a" * chars), _turn("ok")))

    _ = await _compact(compactor, agent, agent.runtime.context().messages)

    audit = _instruction(provider.requests[-1])
    body, _, session = audit.rpartition("\n\n<session>\n")
    lines = session.removesuffix("\n</session>").splitlines()
    held = sum(map(len, lines)) / 4
    fixed = (len(provider.requests[-1].system[0]) + len(body)) / 4
    assert held + fixed <= window
    assert held + fixed + summary_tokens(held) <= window


@pytest.mark.asyncio
async def test_an_audits_digest_reads_the_older_lines_as_a_summary_is_asked(
    tmp_path: Path,
) -> None:
    provider, agent, compactor = _auditing(
        tmp_path,
        reply("<summary>the older half</summary>"),
        _AUDIT,
        window=14_000,
    )
    _hold(agent, (_user("a" * 15_000), _turn("x" * 15_000), _user("b" * 30_000)))

    _ = await _compact(compactor, agent, agent.runtime.context().messages)

    digest = _instruction(provider.requests[0])
    assert digest.startswith(
        f"L1 USER: {'a' * 15_000}\nL2 AGENT: {'x' * 15_000}\n\n"
        "CRITICAL: Produce plain text exclusively.",
    )
    assert digest.endswith("causing task failure.\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "logged"),
    [
        (ContextOverflowError("too long"), "too long"),
        (RateLimitedError("busy"), "busy"),
    ],
    ids=["overflows", "rate-limited"],
)
async def test_an_audit_request_that_fails_says_why(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
    logged: str,
) -> None:
    _, agent, compactor = _auditing(tmp_path, failure)
    _hold(agent, (*_HISTORY, _user("now this")))

    before = agent.runtime.context().messages
    with (
        caplog.at_level(logging.WARNING, logger="sagent.agent.compactor"),
        pytest.raises(ContextOverflowError, match="the session audit failed"),
    ):
        _ = await _live(agent, compactor)

    assert f"the session audit request failed: {logged}" in caplog.messages
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_a_finished_audit_missing_its_closing_tag_is_kept_whole(
    tmp_path: Path,
) -> None:
    """Only a reply its limit stopped is marked cut off."""
    _, agent, compactor = _auditing(tmp_path, reply("<summary>all of the audit"))
    _hold(agent, _HISTORY)

    splice = await _compact(compactor, agent, agent.runtime.context().messages)

    assert splice.strategy == "session_audit"
    assert "Summary:\nall of the audit\nResume work" in _lead(splice)


@pytest.mark.asyncio
async def test_a_proactive_audit_names_the_kept_tail_and_carries_on(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(replies=[_AUDIT], capability=_capability())
    agent = Agent(
        model=provider["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="S",
    )
    compactor = SummaryCompactor(agent, style="session-audit", proactive=True)
    _hold(agent, (*_HISTORY, _user("now this")))

    splice = await _compact(compactor, agent, agent.runtime.context().messages)

    assert (
        "Summary:\nthe audit\n\nThe most recent messages appear below in their"
        " original form.\n\nContinue working autonomously."
    ) in _lead(splice)


@pytest.mark.asyncio
async def test_an_audit_line_holds_200_characters_of_arguments_and_no_silent_peer(
    tmp_path: Path,
) -> None:
    journal = RecordingJournal()
    journal.append(
        ContextClear(system_prompt="S"),
        UserMessage(content="go"),
        ToolCall(call_id="c0", name="Read", arguments={"file_path": "p" * 300}),
        AgentToAgentMessage(content=None, sender="Agent_0", recipient="Agent"),
    )
    provider, agent, compactor = _auditing(tmp_path, _AUDIT, journal=journal)
    _hold(agent, (_user("now"),))

    _ = await _compact(compactor, agent, agent.runtime.context().messages)

    session = _instruction(provider.requests[0]).split("<session>\n")[1]
    arguments = ('{"file_path": "' + "p" * 300 + '"}')[:200]
    assert session.removesuffix("\n</session>").splitlines() == [
        "L1 USER: go",
        f"L2 TOOL Read {arguments}",
        "L3 USER: now",
    ]


@pytest.mark.asyncio
async def test_overflow_shortens_the_suffix_without_discarding_any_block(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, ContextOverflowError("too long"), _SUMMARY)
    history = tuple(
        entry for i in range(4) for entry in (_user(f"ask {i}"), _turn(f"answer {i}"))
    )
    _hold(agent, history)
    original = agent.runtime.request()

    splice = await _live(agent, SummaryCompactor(agent))

    first, retry = provider.requests
    assert first.system == retry.system == original.system
    assert first.tools == retry.tools == original.tools
    assert first.records[:-1] == original.records
    assert retry.records[:-1] == original.records[: len(retry.records) - 1]
    assert 0 < len(retry.records) < len(first.records)
    retained = _texts(_records(splice.payload))
    assert "answer 3" in retained
    assert "Summary:\nthe gist" in "\n".join(retained)


@pytest.mark.asyncio
async def test_prefix_summary_keeps_the_tool_result_and_attachment_unchanged(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    image = Attachment(mime_descriptor="image/png", data=b"image")
    history = (
        _user("go", image),
        _turn("", call("Read", "c1")),
        _result("c1", "body" * 10_000),
        _turn("done"),
    )
    _hold(agent, history)
    original = agent.runtime.request()

    _ = await _live(agent, SummaryCompactor(agent))

    assert provider.requests[0].records[:-1] == original.records
    result = next(r for r in original.records if isinstance(r, ToolResult))
    assert result.text() == "body" * 10_000
    assert original.records[0] == _user("go", image)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [reply("no summary"), ContextOverflowError("full")])
async def test_failed_compaction_leaves_history_intact(
    tmp_path: Path,
    answer: Reply,
) -> None:
    _, agent = _agent(tmp_path, *[answer] * 3)
    _hold(agent, _HISTORY)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError):
        _ = await _live(agent, SummaryCompactor(agent))

    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_no_nonempty_prefix_that_fits_errors_without_a_model_call(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, capability=_capability(window=2_000))
    agent.runtime.system = ()
    agent.runtime.tools.clear()
    history = (_user("x" * 100_000), _turn("done"))

    with pytest.raises(ContextOverflowError, match="prefix"):
        _ = await _compact(_unreserved(agent), agent, history)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_completed_prefix_summaries_keep_their_separator_and_order(
    tmp_path: Path,
) -> None:
    originals = tuple(
        entry
        for i in range(2)
        for entry in (_user(f"U{i}:" + "u" * 5_997), _turn(f"A{i}:" + "a" * 5_997))
    )
    first = "D1:" + "s" * 3_997
    second = "D2:" + "t" * 3_997
    provider, agent = _agent(
        tmp_path,
        reply(f"<summary>{first}</summary>"),
        reply(f"<summary>{second}</summary>"),
        capability=_capability(window=4_000),
    )
    agent.runtime.system = ()
    agent.runtime.tools.clear()

    splice = await _compact(_unreserved(agent), agent, originals)

    assert len(provider.requests) == 2
    assert _sent(provider.requests[0]) == _records(originals[:2])
    assert _sent(provider.requests[1])[1:] == _records(originals[2:3])
    assert f"Summary:\n{first}\n\nSummary:\n{second}" in _lead(splice)
    assert splice.payload[-1] == originals[-1]


@pytest.mark.asyncio
async def test_original_prefixes_are_consumed_before_summaries_are_revisited(
    tmp_path: Path,
) -> None:
    originals = tuple(
        entry
        for i in range(3)
        for entry in (_user(f"U{i}:" + "u" * 5_997), _turn(f"A{i}:" + "a" * 5_997))
    )
    pending = _user("PENDING:" + "p" * 9_992)
    provider, agent = _agent(
        tmp_path,
        reply("<summary>D1:" + "s" * 7_197 + "</summary>"),
        reply("<summary>D2:" + "s" * 7_197 + "</summary>"),
        reply("<summary>D3:" + "s" * 7_197 + "</summary>"),
        reply("<summary>FINAL</summary>"),
        capability=_capability(window=6_000),
    )
    agent.runtime.system = ()
    agent.runtime.tools.clear()
    compactor = _unreserved(agent)

    splice = await _compact(compactor, agent, (*originals, pending))

    assert len(provider.requests) == 4
    consumed: list[str] = []
    for index, request in enumerate(provider.requests[:3]):
        sent = _texts(_sent(request))
        consumed.extend(s for s in sent if s.startswith(("U", "A")))
        if index:
            assert sent[:index] == [
                f"Summary:\nD{i}:" + "s" * 7_197 for i in range(1, index + 1)
            ]
            assert (
                f"The first {index} entries are already retained."
                " Use them only as context; summarize only the newly selected"
                " entries after them. Do not repeat the retained summaries."
            ) in _instruction(request)
    assert consumed == _texts(_records(originals))
    assert all(
        record in [r for request in provider.requests[:3] for r in _sent(request)]
        for record in _records(originals)
    )
    last = "\n".join(_texts(_sent(provider.requests[-1])))
    assert all(f"D{i}:" in last for i in (1, 2, 3))
    assert "U0:" not in last
    assert all(pending not in request.records for request in provider.requests)
    assert "Summary:\nFINAL" in _lead(splice)
    assert _lead(splice).endswith(pending.content or "")


@pytest.mark.asyncio
async def test_audit_replacement_fitting_never_summarizes_pending_input(
    tmp_path: Path,
) -> None:
    provider, agent, compactor = _auditing(
        tmp_path,
        reply("<summary>" + "s" * 12_000 + "</summary>"),
        _SUMMARY,
        window=6_000,
    )
    image = Attachment(mime_descriptor="image/png", data=b"png")
    pending = _user("PENDING:" + "p" * 11_992, image)
    _hold(agent, (*_HISTORY, pending))

    splice = await _live(agent, compactor)

    assert len(provider.requests) == 2
    assert "PENDING:" not in "\n".join(_texts(_sent(provider.requests[1])))
    assert provider.requests[1].system == agent.runtime.system
    assert provider.requests[1].tools == agent.runtime.request().tools
    assert _lead(splice).endswith(pending.content or "")
    assert "Summary:\nthe gist" in _lead(splice)
    first = splice.payload[0]
    assert isinstance(first, UserMessage)
    assert first.attachments == pending.attachments
    assert splice.preserved_tail_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["from", "up_to"])
async def test_only_oversized_pending_input_errors_without_sending(
    tmp_path: Path,
    direction: str,
) -> None:
    provider, agent = _agent(tmp_path, capability=_capability(window=2_000))
    pending = _user("p" * 12_000)
    _hold(agent, (pending,))
    before = agent.runtime.context().messages
    compactor = SummaryCompactor(
        agent,
        direction="up_to" if direction == "up_to" else "from",
    )

    with pytest.raises(ContextOverflowError, match="no nonempty prefix"):
        _ = await _live(agent, compactor)

    assert provider.requests == []
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
@pytest.mark.parametrize("reserve", [0, 1_000])
async def test_the_largest_fitting_prefix_reserves_summary_output(
    tmp_path: Path,
    reserve: int,
) -> None:
    provider, agent = _agent(
        tmp_path,
        *[_SUMMARY] * 10,
        capability=_capability(window=4_000),
    )
    history = tuple(
        entry
        for i in range(3)
        for entry in (_user(f"U{i}:" + "u" * 2_997), _turn(f"A{i}:" + "a" * 2_997))
    )
    compactor = SummaryCompactor(agent, summary_tokens=lambda _: float(reserve))

    _ = await _compact(compactor, agent, history)

    first = provider.requests[0]
    selected = len(_sent(first))
    assert 0 < selected < len(history)
    assert _sent(first) == _records(history[:selected])
    larger = replace(
        first,
        records=[*_records(history[: selected + 1]), first.records[-1]],
    )
    accounting = agent.runtime.token_accounting
    assert accounting.estimate_request(first, model=agent.model) + reserve <= 4_000
    assert accounting.estimate_request(larger, model=agent.model) + reserve > 4_000
    assert all(
        accounting.estimate_request(r, model=agent.model) + reserve <= 4_000
        for r in provider.requests
    )


@pytest.mark.asyncio
async def test_failure_after_a_successful_prefix_leaves_all_history_intact(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(
        tmp_path,
        _SUMMARY,
        *[reply("invalid")] * 3,
        capability=_capability(window=4_000),
    )
    history = tuple(
        entry
        for i in range(5)
        for entry in (_user(f"ask {i}" + "u" * 4_000), _turn("a" * 4_000))
    )
    _hold(agent, history)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="summary failed after 3 attempts"):
        _ = await _live(agent, _unreserved(agent))

    assert len(provider.requests) == 4
    assert all(r.records == provider.requests[1].records for r in provider.requests[1:])
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_an_oversized_kept_head_is_never_summarized_or_dropped(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, capability=_capability(window=4_000))
    history = (_user("h" * 20_000), _turn("kept"), *_HISTORY)
    _hold(agent, history)
    before = agent.runtime.context().messages

    with pytest.raises(ContextOverflowError, match="no nonempty prefix"):
        _ = await _live(
            agent,
            SummaryCompactor(agent, keep_recent=2, direction="up_to"),
        )

    assert provider.requests == []
    assert agent.runtime.context().messages == before


@pytest.mark.asyncio
async def test_adjacent_original_users_preserve_the_live_request_prefix(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY)
    history = (_user("first"), _user("second"), _turn("done"))
    _hold(agent, history)
    original = agent.runtime.request()

    _ = await _live(agent, SummaryCompactor(agent))

    assert provider.requests[0].records[:-1] == original.records


@pytest.mark.asyncio
async def test_verification_preserves_the_retained_head_summary_boundary(
    tmp_path: Path,
) -> None:
    provider, agent = _agent(tmp_path, _SUMMARY, reply("IDENTICAL"))

    splice = await _compact(
        SummaryCompactor(agent, keep_recent=3, direction="up_to", verify_summary=True),
        agent,
        _HISTORY,
    )

    assert splice.payload[:3] == _HISTORY[:3]
    assert len(provider.requests) == 2
    assert all(_sent(r) == _records(_HISTORY) for r in provider.requests)
    for request in provider.requests:
        assert (
            "The first 4 entries are already retained."
            " Use them only as context; summarize only the newly selected"
            " entries after them. Do not repeat the retained summaries."
        ) in _instruction(request)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
