"""Tests for the pruner: old tool results dropped once that costs less than keeping them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import math
import re
import time

import pytest

from sagent.agent.agent import Agent, NoShrinker
from sagent.agent.compaction import pruner as pruner_module
from sagent.agent.compaction.compactor import SummaryCompactor
from sagent.agent.compaction.pruner import (
    Pruner,
    Saving,
    client_policy,
    cut,
    eligible,
    expired,
    idle,
    left_in,
    limit_tokens,
    prices,
    reads,
    requests_left,
    saving,
    sizes,
    superseded,
)
from sagent.journal.types import (
    AssistantMessage,
    BashResult,
    Cancelled,
    EditResult,
    Failed,
    LineSpan,
    Outcome,
    ReadResult,
    Succeeded,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UncategorizedToolResult,
    UserMessage,
    WriteResult,
)
from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    PruningPolicy,
)
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.scripted import ScriptedProvider, reply
from sagent.model.types import Response
from sagent.runtime import types
from sagent.runtime.tape import (
    ContextSplice,
    full_tape_mask,
    unpaired_call_ids,
)
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence
    from pathlib import Path

    from sagent.agent.types import Factory, Shrinker
    from sagent.journal.types import SessionRecord
    from sagent.runtime.types import ModelContextEvent
    from sagent.tools.types import Tool


_STUB = "[pruned: old tool output removed to keep the context small]"

# Anthropic's shape: a read is a tenth of input, a five-minute write a quarter more.
# A rewritten token costs ``(3.75 - 0.3) / 0.3 = 11.5`` cached reads.
_CACHED = TokenPrice(
    request=3.0,
    response=15.0,
    cache_write=3.75,
    cache_write_1h=6.0,
    cache_read=0.3,
)

# No cache discount: a rewrite costs nothing more than the read it replaces.
_UNCACHED = TokenPrice(
    request=3.0,
    response=15.0,
    cache_write=0.0,
    cache_write_1h=0.0,
    cache_read=3.0,
)


def _agent(
    tmp_path: Path,
    *,
    price: TokenPrice | None = _CACHED,
    window: int = 200_000,
    compactor: Factory[Shrinker] = NoShrinker,
    tools: Sequence[Tool] = (),
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Agent, Pruner, ScriptedProvider]:
    prices = PriceCatalog(
        prices={} if price is None else {PriceCondition(): price},
        window=window,
    )
    provider = ScriptedProvider(
        capability=ModelCapability(
            model_id="priced",
            limits=ModelLimits(
                max_request_tokens=window,
                max_response_tokens=min(8_000, window // 4),
            ),
            prices=prices,
            cache_ttl_sec=frozenset({0.0, 300.0, 3600.0}),
            default=ModelSettings(
                cache_ttl_sec=300.0,
                context_size=prices["fullcontext"],
            ),
        ),
    )
    agent = Agent(
        model=provider["priced"],
        tools=tools,
        compactor=compactor,
        pruner=partial(Pruner, clock=clock),
        tool_context=ToolContext(cwd=tmp_path),
        system_prompt="",
    )
    pruner = agent.session.pruner
    assert isinstance(pruner, Pruner)
    # Every result expires at once, so a test of the rent rule sees all of them.
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0, expiry=0.0),
    )
    return agent, pruner, provider


# Each call is one request; ``requests`` adds that many short answered turns.
def _turns(*results: str, tail: str = "done", requests: int = 0) -> list[SessionRecord]:
    """Return a turn that calls ``Read`` once per result, then says ``tail``."""
    records: list[SessionRecord] = [UserMessage(content="go")]
    for i, content in enumerate(results):
        records.append(ToolCall(call_id=f"c{i}", name="Read"))
        records.append(
            ToolMessage(
                call_id=f"c{i}",
                state="done",
                stdout=(content,),
                result=UncategorizedToolResult(),
            ),
        )
    records.append(AssistantMessage(content=tail))
    for i in range(requests):
        records.append(UserMessage(content=f"q{i}"))
        records.append(AssistantMessage(content=f"a{i}"))
    return records


def _read(
    call_id: str,
    path: str,
    *ranges: tuple[int | None, int | None],
) -> list[SessionRecord]:
    """Return a Read of ``ranges`` of ``path``, 1000 tokens long; all of it when none."""
    return [
        ToolCall(call_id=call_id, name="Read", arguments={"file_path": path}),
        ToolMessage(
            call_id=call_id,
            state="done",
            stdout=("x" * 4000, *(LineSpan(start=s, count=n) for s, n in ranges)),
            result=ReadResult(path=path),
        ),
    ]


def _unnamed(call_id: str, tool: str) -> list[SessionRecord]:
    """Return a ``tool`` call with no arguments and its result with no path or command."""
    tool_result = ReadResult(exit_code=1) if tool == "Read" else BashResult(exit_code=1)
    return [
        ToolCall(call_id=call_id, name=tool),
        ToolMessage(call_id=call_id, state="done", stderr=("no",), result=tool_result),
    ]


def _shell(call_id: str, command: str) -> list[SessionRecord]:
    return [
        ToolCall(call_id=call_id, name="Bash", arguments={"command": command}),
        ToolMessage(
            call_id=call_id,
            state="done",
            stdout=("o" * 4000,),
            result=BashResult(command=("bash", "-c", command)),
        ),
    ]


def _ids(context: Sequence[ModelContextEvent], indices: Collection[int]) -> set[str]:
    return {
        entry.call_id for i in indices if isinstance(entry := context[i], ToolMessage)
    }


def _results(agent: Agent) -> dict[str, str]:
    return {r.call_id: r.text() for r in agent.session if isinstance(r, ToolMessage)}


def _prunes(agent: Agent) -> list[ContextSplice]:
    return [
        r
        for r in agent.runtime.tape
        if isinstance(r, ContextSplice) and r.strategy == "prune"
    ]


def _notes(agent: Agent) -> list[str]:
    return [
        r.content or ""
        for r in agent.runtime.tape.context().messages
        if isinstance(r, SystemMessage) and r.subtype == "prune"
    ]


def _noted(agent: Agent) -> set[str]:
    """Return the calls whose outputs the last prune note listed; none with no note."""
    notes = _notes(agent)
    if not notes:
        return set()
    found = cast(list[str], re.findall(r"^- #(\d+) ", notes[-1], re.MULTILINE))
    listed = [int(n) for n in found]
    return _ids(agent.runtime.tape.context().messages, listed)


def test_a_drop_that_costs_more_than_it_saves_waits(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # One request has read 1000 tokens, with 1000 kept after it to rewrite:
    # 1 x 985 cached reads paid is less than 11.5 x 1015.
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    assert not pruner.should_run()
    assert not pruner.prune()
    assert (_prunes(agent), _notes(agent)) == ([], [])


def test_a_drop_is_noted_once_enough_requests_will_read_it(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Twenty-one requests have read 10,000 tokens of result; dropping it
    # rewrites only the few dozen tokens of short turns after it.
    agent.session.append(*_turns("x" * 40_000, requests=20))

    assert pruner.should_run()
    _ = pruner.prune()

    assert _noted(agent) == {"c0"}


def test_a_note_waits_until_the_reads_paid_cover_the_rewrite(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Twenty requests so far forecast twenty more, but one request has read the
    # result: 1 x 985 cached reads paid is less than 11.5 x 1015 to rewrite.
    agent.session.append(*_turns(requests=20))
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    assert not pruner.should_run()

    # A second request: 2 x 985 still short; the twelfth: 12 x 985 > 11.5 x 1026.
    agent.session.append(*_turns(requests=1))
    assert not pruner.should_run()
    agent.session.append(*_turns(requests=10))
    _ = pruner.prune()
    assert _noted(agent) == {"c0"}


def test_a_result_the_model_has_not_read_is_neither_noted_nor_dropped(
    tmp_path: Path,
) -> None:
    uncached = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=3.0,
    )
    agent, pruner, _ = _agent(tmp_path, price=uncached)
    # The cut at the read result would take the unread one after it along.
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="read", name="Read"),
        ToolMessage(
            call_id="read",
            state="done",
            stdout=("x" * 4000,),
            result=UncategorizedToolResult(),
        ),
        AssistantMessage(content="ok"),
        UserMessage(content="next"),
        ToolCall(call_id="unread", name="Read"),
        ToolMessage(
            call_id="unread",
            state="done",
            stdout=("y" * 4000,),
            result=UncategorizedToolResult(),
        ),
        UserMessage(content="also"),
    )

    _ = pruner.prune()
    assert _noted(agent) == {"read"}


def test_an_unread_result_never_rides_along_though_superseded(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=_UNCACHED)
    for entry in (
        UserMessage(content="go"),
        reply(calls=[ToolCall(call_id="old", name="Bash")]),
        _result("old"),
        reply(
            calls=[
                ToolCall(call_id="unread", name="Read"),
                ToolCall(call_id="again", name="Read"),
            ],
        ),
        ToolMessage(
            call_id="unread",
            state="done",
            stdout=("x" * 400,),
            result=ReadResult(path=_P),
        ),
        ToolMessage(
            call_id="again",
            state="done",
            stdout=("x" * 400,),
            result=ReadResult(path=_P),
        ),
    ):
        _ = agent.runtime.tape.append_history(entry)
    agent.runtime.queued["old"] = "spent"

    _ = pruner.prune()

    assert _results(agent) == {"old": _STUB, "unread": "x" * 400, "again": "x" * 400}


_P = "/home/me/repo/agent/x.py"


@pytest.mark.parametrize(
    ("records", "gone"),
    [
        ([*_read("a", _P), *_read("b", _P)], {"a"}),
        ([*_read("a", _P, (1, 10)), *_read("b", _P, (1, 20))], {"a"}),
        ([*_read("a", _P, (1, 10)), *_read("b", _P, (5, 10))], set[str]()),
        ([*_read("a", _P), *_read("b", _P, (1, 20))], set[str]()),
        ([*_read("a", _P, (5, 10)), *_read("b", _P)], {"a"}),
        # Two later reads cover it between them; neither covers the other.
        (
            [
                *_read("a", _P, (1, 10)),
                *_read("b", _P, (1, 5)),
                *_read("c", _P, (6, 5)),
            ],
            {"a"},
        ),
        ([*_read("a", _P, (3, None)), *_read("b", _P, (1, 10))], set[str]()),
        ([*_read("a", _P, (3, 4)), *_read("b", _P, (1, None))], {"a"}),
        # A read with no start starts at line 1.
        ([*_read("a", _P), *_read("b", _P, (2, None))], set[str]()),
        ([*_read("a", _P, (None, 10)), *_read("b", _P, (2, 9))], set[str]()),
        ([*_read("a", _P, (None, 10)), *_read("b", _P, (1, 10))], {"a"}),
        # Covered already, a later span past a gap changes nothing.
        (
            [
                *_read("a", _P, (1, 10)),
                *_read("b", _P, (1, 10)),
                *_read("c", _P, (50, 5)),
            ],
            {"a"},
        ),
        # An earlier read covers nothing; another file covers nothing.
        ([*_read("a", _P, (1, 20)), *_read("b", _P, (1, 10))], set[str]()),
        ([*_read("a", _P), *_read("b", "/repo/agent/y.py")], set[str]()),
        ([*_shell("a", "ls"), *_shell("b", "ls")], {"a"}),
        # A read that failed names no file, and a shell call may name no command:
        # neither is superseded, nor stands in the way of what is.
        ([*_read("a", _P), *_unnamed("x", "Read"), *_read("b", _P)], {"a"}),
        ([*_shell("a", "ls"), *_unnamed("x", "Bash"), *_shell("b", "ls")], {"a"}),
        ([*_unnamed("x", "Read"), *_unnamed("y", "Read")], set[str]()),
        ([*_unnamed("x", "Bash"), *_unnamed("y", "Bash")], set[str]()),
        ([*_shell("a", "ls"), *_shell("b", "ls -a")], set[str]()),
    ],
)
def test_a_result_is_superseded_once_all_it_showed_is_shown_again(
    records: list[SessionRecord],
    gone: set[str],
) -> None:
    context: list[ModelContextEvent] = [
        reply(calls=[r]) if isinstance(r, ToolCall) else r
        for r in records
        if isinstance(r, ToolCall | ToolMessage)
    ]

    assert _ids(context, superseded(context)) == gone


def test_a_result_is_idle_since_its_file_was_last_named() -> None:
    context = [
        reply(calls=[ToolCall(call_id="a", name="Read", arguments={"file_path": _P})]),
        ToolMessage(call_id="a", state="done", result=ReadResult(path=_P)),
        reply(calls=[ToolCall(call_id="g", name="Grep", arguments={"path": _P})]),
        ToolMessage(
            call_id="g",
            state="done",
            stdout=("hit",),
            result=UncategorizedToolResult(),
        ),
        reply(
            calls=[
                ToolCall(
                    call_id="s",
                    name="Bash",
                    arguments={"command": "sed -n 1p agent/x.py"},
                ),
            ],
        ),
        ToolMessage(
            call_id="s",
            state="done",
            stdout=("line",),
            result=UncategorizedToolResult(),
        ),
        reply(
            calls=[
                ToolCall(
                    call_id="e",
                    name="Edit",
                    arguments={"file_path": "/repo/y.py"},
                ),
            ],
        ),
        ToolMessage(
            call_id="e",
            state="done",
            result=EditResult(path="/repo/y.py"),
        ),
        reply("done"),
    ]
    at = {e.call_id: i for i, e in enumerate(context) if isinstance(e, ToolMessage)}
    k = reads(context)

    quiet = idle(context)

    # Grep named the file, then Bash by its last two parts: idle since Bash.
    assert quiet[at["a"]] == k[at["s"]] < k[at["g"]] < k[at["a"]]
    # A result with no file, or whose file nothing named again: since it came.
    assert quiet[at["g"]] == k[at["g"]]
    assert quiet[at["e"]] == k[at["e"]] == 1


@pytest.mark.parametrize(
    ("quiet", "gone", "prices", "expiry", "old"),
    [
        # H = 100, B = 1·230 + 2·100 = 430: expires once idle for 4.3 requests.
        ([0, 5, 0, 4, 0], set[int](), (1.0, 2.0), 1.0, [1]),
        ([0, 5, 0, 4, 0], set[int](), (1.0, 2.0), 0.5, [1, 3]),
        ([0, 5, 0, 0, 0], {3}, (1.0, 2.0), 1.0, [1, 3]),  # Superseded: B = 0.
        ([0, 0, 0, 0, 0], set[int](), (1.0, 2.0), 0.0, [1, 3]),
        ([0, 9, 0, 9, 0], {3}, None, 1.0, [3]),  # Unpriced: superseded only.
        ([0, 9, 0, 9, 0], set[int](), None, 1.0, []),
    ],
)
def test_expiry_follows_its_spec(
    *,
    quiet: list[int],
    gone: set[int],
    prices: tuple[float, float] | None,
    expiry: float,
    old: list[int],
) -> None:
    assert (
        expired(
            [10.0, 100.0, 10.0, 100.0, 10.0],
            [1, 3],
            idle=quiet,
            superseded=gone,
            stub=0.0,
            prices=prices,
            expiry=expiry,
        )
        == old
    )


def test_a_result_s_rent_is_what_it_holds_beyond_its_stub() -> None:
    # H = 4·1·(100 − 10) = 360 < B = 1·230 + 2·100 = 430: not yet.
    assert (
        expired(
            [10.0, 100.0, 10.0, 100.0, 10.0],
            [1],
            idle=[0, 4, 0, 0, 0],
            superseded=set[int](),
            stub=10.0,
            prices=(1.0, 2.0),
            expiry=1.0,
        )
        == []
    )


def _read_turn(
    path: str,
    *,
    before: int,
    named: str,
    after: int,
) -> list[SessionRecord]:
    """Return a read of ``path``, ``before`` requests, a Grep of ``named``, ``after`` more."""
    return [
        UserMessage(content="go"),
        *_read("c0", path),
        AssistantMessage(content="read"),
        *_turns(requests=before),
        UserMessage(content="look"),
        ToolCall(call_id="g", name="Grep", arguments={"pattern": "x", "path": named}),
        ToolMessage(
            call_id="g",
            state="done",
            stdout=("hit",),
            result=UncategorizedToolResult(),
        ),
        AssistantMessage(content="ok"),
        *_turns(requests=after),
    ]


def test_a_file_result_naming_no_file_is_idle_since_it_came() -> None:
    context = [
        reply(calls=[ToolCall(call_id="e", name="Edit")]),
        ToolMessage(call_id="e", state="done", result=EditResult()),
        reply(
            calls=[ToolCall(call_id="g", name="Grep", arguments={"pattern": "None"})],
        ),
        ToolMessage(
            call_id="g",
            state="done",
            stdout=("hit",),
            result=UncategorizedToolResult(),
        ),
        reply("done"),
    ]
    at = 1

    assert idle(context)[at] == reads(context)[at]


def test_a_file_named_only_before_its_result_is_idle_since_it_came() -> None:
    context = [
        reply(calls=[ToolCall(call_id="g", name="Grep", arguments={"path": _P})]),
        ToolMessage(
            call_id="g",
            state="done",
            stdout=("hit",),
            result=UncategorizedToolResult(),
        ),
        reply(
            "editing",
            calls=[ToolCall(call_id="e", name="Edit", arguments={"file_path": _P})],
        ),
        ToolMessage(call_id="e", state="done", result=EditResult(path=_P)),
        reply("done"),
    ]
    at = {e.call_id: i for i, e in enumerate(context) if isinstance(e, ToolMessage)}

    assert idle(context)[at["e"]] == reads(context)[at["e"]] < reads(context)[at["g"]]


def test_a_file_result_after_one_naming_no_file_is_still_followed() -> None:
    context = [
        reply(
            calls=[
                ToolCall(call_id="e", name="Edit"),
                ToolCall(call_id="a", name="Read", arguments={"file_path": _P}),
            ],
        ),
        ToolMessage(call_id="e", state="done", result=WriteResult()),
        ToolMessage(call_id="a", state="done", result=ReadResult(path=_P)),
        reply(calls=[ToolCall(call_id="g", name="Grep", arguments={"path": _P})]),
        ToolMessage(
            call_id="g",
            state="done",
            stdout=("hit",),
            result=UncategorizedToolResult(),
        ),
        reply("done"),
    ]
    at = {e.call_id: i for i, e in enumerate(context) if isinstance(e, ToolMessage)}

    assert idle(context)[at["a"]] == reads(context)[at["g"]] < reads(context)[at["a"]]


@pytest.mark.parametrize("server", [False, True])
def test_with_no_policy_to_prune_by_nothing_is_planned(
    tmp_path: Path,
    *,
    server: bool,
) -> None:
    context = [
        ToolMessage(
            call_id="a",
            state="done",
            stdout=("x" * 64,),
            result=UncategorizedToolResult(),
        ),
        reply("done"),
    ]
    agent = Agent(
        model=ScriptedProvider(
            capability=ModelCapability(limits=ModelLimits(max_request_tokens=16)),
        )["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0) if server else None,
        manage_context_server_side=server,
    )
    for entry in context:
        _ = agent.runtime.tape.append_history(entry)
    before = list(agent.runtime.tape)
    pruner = Pruner(agent)

    assert not pruner.should_run()
    assert not pruner.prune()
    assert list(agent.runtime.tape) == before


def test_a_file_named_deep_in_frozen_arguments_is_in_use() -> None:
    context = [
        reply(
            calls=[
                ToolCall(call_id="a", name="Read", arguments={"file_path": _P}),
                ToolCall(call_id="numeric", name="Read", arguments={"file_path": "2"}),
            ],
        ),
        ToolMessage(call_id="a", state="done", result=ReadResult(path=_P)),
        ToolMessage(
            call_id="numeric",
            state="done",
            result=ReadResult(path="2"),
        ),
        reply("before"),
        reply(
            calls=[
                ToolCall(
                    call_id="m",
                    name="MultiEdit",
                    arguments={"edits": [{"metadata": [2], "files": [_P]}]},
                ),
            ],
        ),
        reply("after"),
    ]

    assert idle(context) == [3, 1, 3, 2, 1, 0]


@pytest.mark.parametrize(("named", "kept"), [(_P, True), ("/repo/other.py", False)])
def test_a_file_in_use_is_kept_though_its_rent_has_paid(
    tmp_path: Path,
    named: str,
    kept: bool,
) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0),
    )
    # H = 0.3·985 a request; B = 0.3·~1050 + 3.75·1000: expires 14 requests unused.
    agent.session.append(*_read_turn(_P, before=15, named=named, after=5))

    _ = pruner.prune()
    assert (_noted(agent) == {"c0"}) is not kept


def test_a_superseded_read_is_noted_while_the_newer_stays(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0),
    )
    agent.session.append(
        UserMessage(content="go"),
        *_read("c0", _P),
        *_turns(requests=14),
        UserMessage(content="again"),
        *_read("c1", _P),
        AssistantMessage(content="read"),
        *_turns(requests=2),
    )

    _ = pruner.prune()
    assert _noted(agent) == {"c0"}


def test_a_cut_its_reads_have_not_paid_for_waits(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Dropping both rewrites the 10,000-token answer between them, more than
    # 3 + 1 reads of 9985 paid; dropping the second alone rewrites almost nothing.
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="early", name="Read"),
        ToolMessage(
            call_id="early",
            state="done",
            stdout=("e" * 40_000,),
            result=UncategorizedToolResult(),
        ),
        AssistantMessage(content="a" * 40_000),
        UserMessage(content="more"),
        ToolCall(call_id="late", name="Read"),
        ToolMessage(
            call_id="late",
            state="done",
            stdout=("l" * 40_000,),
            result=UncategorizedToolResult(),
        ),
        AssistantMessage(content="done"),
    )

    _ = pruner.prune()

    assert _noted(agent) == {"late"}


def test_an_expired_cache_makes_any_drop_pay(tmp_path: Path) -> None:
    now = [1000.0]
    agent, pruner, _ = _agent(tmp_path, clock=lambda: now[0])
    agent.session.append(*_turns("x" * 4000, "y" * 4000, tail="t" * 4000))
    agent.runtime.publish(
        types.ModelResponseCompleted(outcome=Succeeded(value=reply("ok"))),
    )

    now[0] += 300.0
    assert not pruner.should_run()  # Still warm at its lifetime.
    now[0] += 1.0

    _ = pruner.prune()
    assert _noted(agent) == {"c0", "c1"}
    agent.runtime.queued["c0"] = "spent"
    _ = pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": "y" * 4000}


@pytest.mark.parametrize(
    "outcome",
    [Cancelled(), Failed(exception=RuntimeError("down"))],
    ids=["cancelled", "failed"],
)
def test_a_call_that_brought_no_reply_warms_no_cache(
    tmp_path: Path,
    outcome: Outcome[Response],
) -> None:
    now = [1000.0]
    agent, pruner, _ = _agent(tmp_path, clock=lambda: now[0])
    agent.session.append(*_turns("x" * 4000, "y" * 4000, tail="t" * 4000))
    agent.runtime.publish(
        types.ModelResponseCompleted(outcome=Succeeded(value=reply("ok"))),
    )
    now[0] += 301.0

    agent.runtime.publish(types.ModelResponseCompleted(outcome=outcome))

    assert pruner.should_run()


def test_without_a_cache_discount_every_drop_is_noted(tmp_path: Path) -> None:
    uncached = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=3.0,
    )
    agent, pruner, _ = _agent(tmp_path, price=uncached)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    _ = pruner.prune()
    assert _noted(agent) == {"c0"}


def test_a_free_cache_write_still_rewrites_at_the_input_price(tmp_path: Path) -> None:
    # OpenAI's shape: no write premium, reads at a tenth. A rewrite costs plain
    # input, so the drop that does not pay on Anthropic's prices does not here.
    free_write = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=0.3,
    )
    agent, pruner, _ = _agent(tmp_path, price=free_write)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    assert not pruner.should_run()


@pytest.mark.parametrize(
    "price",
    [
        None,
        TokenPrice(
            request=0.0,
            response=0.0,
            cache_write=0.0,
            cache_write_1h=0.0,
            cache_read=0.0,
        ),
    ],
)
def test_an_unpriced_or_free_model_is_never_pruned_for_cost(
    tmp_path: Path,
    price: TokenPrice | None,
) -> None:
    agent, pruner, _ = _agent(tmp_path, price=price)
    agent.session.append(*_turns("x" * 40_000, requests=20))

    assert not pruner.prune()
    assert _notes(agent) == []


def test_an_hour_long_cache_prices_its_rewrite_at_the_hour_rate(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Twelve requests have read 985 tokens, 1026 kept after: at the five-minute
    # rate a rewritten token costs 11.5 reads and the drop pays (11,823 > 11,802);
    # at the hour rate it costs 19, and it does not.
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000, requests=11))
    assert pruner.should_run()

    agent.model.settings = replace(agent.model.settings, cache_ttl_sec=3600.0)

    assert not pruner.should_run()


@pytest.mark.asyncio
async def test_a_forced_prune_forgets_the_files_read_and_runs_once(
    tmp_path: Path,
) -> None:
    agent, pruner, _ = _agent(tmp_path)
    seen = tmp_path / "seen.txt"
    _ = seen.write_text("x")
    async with agent:
        agent.session.append(*_turns("x" * 400, "y" * 400))
        agent.runtime.tool_context.files.saw(seen)

        await pruner.run()
        tape = len(agent.runtime.tape)
        await pruner.run()

        assert len(agent.runtime.tape) == tape
    assert _results(agent) == {"c0": _STUB, "c1": _STUB}
    assert len(_prunes(agent)) == 1
    assert seen not in agent.runtime.tool_context.files


@pytest.mark.asyncio
async def test_the_runtime_notes_before_a_model_call_what_a_drop_would_pay_for(
    tmp_path: Path,
) -> None:
    uncached = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=3.0,
    )
    agent, _, provider = _agent(tmp_path, price=uncached)
    provider.replies.append(reply("noted"))
    async with agent:
        agent.session.append(*_turns("x" * 4000))

        assert await agent.send("next") == "noted"

    (request,) = provider.requests
    sent = [r for r in request.records if isinstance(r, ToolMessage)]
    assert [r.text() for r in sent] == [f"{'x' * 4000}\n[1,000 tok]"]
    assert [r.subtype for r in request.records if isinstance(r, SystemMessage)] == [
        "prune",
    ]


@pytest.mark.parametrize(
    ("requests", "prompt", "first", "left"),
    [
        (0, 100.0, 0.0, 0.0),  # No request yet.
        (1, 200.0, 100.0, 1.0),  # One request: one more.
        (4, 100.0, 100.0, 4.0),  # No growth: n.
        (10, 999.0, 994.0, 2.0),  # Half a token a request: room for 2 more.
        (10, 500.0, 100.0, 10.0),  # Room for 12.5 more: n.
        (10, 900.0, 100.0, 1.25),  # Room for 1.25 more.
        (10, 1100.0, 100.0, 0.0),  # Already past the trigger.
    ],
)
def test_requests_left_is_the_requests_so_far_capped_by_the_room_left(
    requests: int,
    prompt: float,
    first: float,
    left: float,
) -> None:
    assert (
        requests_left(requests=requests, prompt=prompt, first=first, limit=1000.0)
        == left
    )


def test_a_contexts_first_request_is_what_came_before_its_first_reply() -> None:
    """Growth is from the first request on: 100 tokens in one request, room for 1."""
    context = [UserMessage(content="a"), reply("b"), UserMessage(content="c")]
    assert left_in(context, [100.0, 0.0, 100.0], limit=300.0) == 1.0


@pytest.mark.parametrize(
    ("sizes", "reads", "prices", "left", "limit", "cold", "drop"),
    [
        ([1, 100, 1, 100, 1], [1] * 5, None, 2.0, 1e3, False, []),  # Unpriced.
        ([1, 100, 1, 100, 1], [0] * 5, (1.0, 2.0), 2.0, 1e3, True, [1, 3]),  # Cold.
        # At 3: 1·99 > 1·2. At 1: 1·99 + 1·99 > 1·4. The earliest is taken.
        ([1, 100, 1, 100, 1], [1] * 5, (1.0, 2.0), 2.0, 1e3, False, [1, 3]),
        # At 3: 1·99 < 2·51. At 1: 2·99 + 1·99 > 2·53; old rent carries the new.
        ([1, 100, 1, 100, 50], [3, 2, 2, 1, 0], (1.0, 3.0), 9.0, 1e6, False, [1, 3]),
        # At 3: 1·99 < 2·51. At 1: 1·99 + 1·99 < 2·252. Not yet paid.
        ([1, 100, 200, 100, 50], [2, 1, 1, 1, 0], (1.0, 3.0), 9.0, 1e6, False, []),
        # At 3: 2·99 > 2·51. At 1: 1·99 + 2·99 < 2·252. Skip the long middle.
        ([1, 100, 200, 100, 50], [3, 1, 2, 2, 0], (1.0, 3.0), 9.0, 1e6, False, [3]),
        # Paid, but compaction comes first: 0.5·99 < 2·51 and 0.5·198 < 2·53.
        ([1, 100, 1, 100, 50], [9] * 5, (1.0, 3.0), 0.5, 1e6, False, []),
        ([1, 100, 1, 900, 1], [0] * 5, None, 2.0, 1e3, False, [1, 3]),  # 1003 → 5.
        ([1, 100, 999, 1, 1], [0] * 5, None, 2.0, 1e3, False, []),  # 1102 → 1003.
        ([1, 100, 1, 100, 1], [0] * 5, None, 2.0, 5.0, False, []),  # 203 → 5: at L.
        ([1, 100, 1, 100, 1], [0] * 5, None, 2.0, 203.0, False, [1, 3]),  # At L.
        ([500, 100, 1, 100, 1], [0] * 5, None, 2.0, 600.0, False, [1, 3]),  # 702 → 504.
        # At 3: H = 1·10 = K = 1·10, not past it. At 1: 10 < 12.
        ([1, 100, 1, 11, 9], [0, 0, 0, 1, 0], (1.0, 2.0), 2.0, 1e6, False, []),
        # At 1 the rent carried from 3 pays: 10 + 1·3 > 12.
        ([1, 2, 1, 11, 9], [0, 3, 0, 1, 0], (1.0, 2.0), 2.0, 1e6, False, [1, 3]),
        # At 3: N·r·D = 1·10 = K, not past it. At 1: 20 < 111.
        ([1, 100, 100, 11, 9], [0, 0, 0, 2, 0], (1.0, 2.0), 1.0, 1e6, False, []),
    ],
)
def test_the_cut_follows_its_spec(
    *,
    sizes: list[float],
    reads: list[int],
    prices: tuple[float, float] | None,
    left: float,
    limit: float,
    cold: bool,
    drop: list[int],
) -> None:
    assert (
        cut(
            sizes,
            [1, 3],
            reads=reads,
            stub=1.0,
            prices=prices,
            left=left,
            limit=limit,
            cold=cold,
        )
        == drop
    )


@pytest.mark.parametrize(
    ("chosen", "prices", "expected"),
    [
        # D = 99 + 99; T = 202 − 198; S = 2·1·198 − 1·4; k = 1·4 / (1·198).
        (
            [1, 3],
            (1.0, 2.0),
            Saving(saved=198, rewritten=4, priced=(392, 4 / 198)),
        ),
        # The cache breaks at the first chosen: all after it is written again.
        ([3], (1.0, 2.0), Saving(saved=99, rewritten=2, priced=(196, 2 / 99))),
        # S = 2·2·198 − 1·4; k = 1·4 / (2·198).
        (
            [1, 3],
            (2.0, 3.0),
            Saving(saved=198, rewritten=4, priced=(788, 4 / 396)),
        ),
        ([1, 3], None, Saving(saved=198, rewritten=4, priced=None)),
        # A stub as big as its result saves nothing and never pays.
        (
            [0],
            (1.0, 2.0),
            Saving(saved=0, rewritten=203, priced=(-203, math.inf)),
        ),
    ],
)
def test_a_saving_follows_its_spec(
    chosen: list[int],
    prices: tuple[float, float] | None,
    expected: Saving,
) -> None:
    sizes = [1.0, 100.0, 1.0, 100.0, 1.0]
    assert saving(sizes, chosen, stub=1.0, prices=prices, left=2.0) == expected


def test_a_saving_of_under_a_token_still_pays_back_in_time() -> None:
    # D = 0.5; T = 101 − 0.5; k = 1·100.5 / (1·0.5).
    assert saving(
        [1.0, 100.0, 1.0, 100.0, 1.0],
        [3],
        stub=99.5,
        prices=(1.0, 2.0),
        left=2.0,
    ) == Saving(saved=0.5, rewritten=100.5, priced=(-99.5, 201.0))


def test_a_saving_needs_a_result_to_drop() -> None:
    with pytest.raises(ValueError, match=r"^Choose at least one result to drop\.$"):
        _ = saving([1.0], [], stub=0.0, prices=None, left=1.0)


def test_nothing_eligible_is_nothing_to_cut() -> None:
    assert (
        cut(
            [1.0],
            [],
            reads=[1],
            stub=0.0,
            prices=(1.0, 1.0),
            left=9.0,
            limit=0.0,
            cold=True,
        )
        == []
    )


def test_a_small_window_leaves_fewer_requests_to_pay_a_drop_back(
    tmp_path: Path,
) -> None:
    records = _turns("x" * 4000, tail="t" * 4000, requests=11)
    wide, wide_pruner, _ = _agent(tmp_path)
    wide.session.append(*records)
    # 4,000 tokens, 1,000 for the reply: compaction at 3,000, about six requests away.
    small, small_pruner, _ = _agent(tmp_path, window=4000)
    small.session.append(*records)

    assert wide_pruner.should_run()
    assert not small_pruner.should_run()


def test_a_drop_too_small_to_avert_compaction_is_left_to_it(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    agent.session.append(*_turns("x" * 400, tail="t" * 14_000))

    assert not pruner.prune()


def test_the_compactor_s_own_trigger_is_the_limit(tmp_path: Path) -> None:
    records = _turns("x" * 9600, "y" * 800)
    # 4,000 tokens: SummaryCompactor fires below the summary call's prompt and
    # room, under the 3,000 the window less the reply leaves.
    compacting, compacting_pruner, _ = _agent(
        tmp_path,
        price=None,
        window=4000,
        compactor=SummaryCompactor,
    )
    compacting.session.append(*records)
    plain, plain_pruner, _ = _agent(tmp_path, price=None, window=4000)
    plain.session.append(*records)
    held = sum(sizes(compacting.model, compacting.runtime.tape.context().messages))
    assert limit_tokens(compacting) < held < limit_tokens(plain)

    assert compacting_pruner.should_run()
    assert not plain_pruner.should_run()


@pytest.mark.parametrize(("window", "limit"), [(0, math.inf), (1, 1.0)])
def test_a_model_with_no_window_never_compacts(
    tmp_path: Path,
    window: int,
    limit: float,
) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=window)
    agent.session.append(*_turns("x" * 40_000, requests=20))

    assert limit_tokens(agent) == limit
    assert not pruner.should_run()


class _Evening:
    """Stands in for ``datetime``: 6 October in the evening here, 7 October in UTC."""

    @classmethod
    def now(cls, tz: object = None) -> datetime:
        if tz is UTC:
            return datetime(2026, 10, 7, 1, tzinfo=UTC)
        return datetime(2026, 10, 6, 18)  # noqa: DTZ001 -- the local wall clock.


def test_prices_are_read_off_the_card_of_the_utc_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pruner_module, "datetime", _Evening)
    halved = replace(_CACHED, cache_read=0.15)
    provider = ScriptedProvider(
        capability=ModelCapability(
            model_id="dated",
            prices=PriceCatalog(
                prices={
                    PriceCondition(): _CACHED,
                    PriceCondition(effective_date=date(2026, 10, 7)): halved,
                },
            ),
        ),
    )

    assert prices(provider["dated"], prompt=0) == (0.15 / 1e6, 3.75 / 1e6)


def test_a_result_at_the_trigger_is_noted_unexpired(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    # Results expire only once superseded, which none is.
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0),
    )
    records = _turns("x" * 8000)
    agent.session.append(*records)
    # Padded to the 3,000-token trigger exactly.
    held = sum(sizes(agent.model, agent.runtime.tape.context().messages))
    agent.session.append(UserMessage(content="p" * int((3000 - held) * 4)))
    assert sum(sizes(agent.model, agent.runtime.tape.context().messages)) == 3000

    _ = pruner.prune()
    assert _noted(agent) == {"c0"}


def test_a_model_with_no_cache_has_none_to_go_cold(tmp_path: Path) -> None:
    now = [1000.0]
    agent, pruner, _ = _agent(tmp_path, clock=lambda: now[0])
    agent.model.settings = replace(agent.model.settings, cache_ttl_sec=0.0)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))
    agent.runtime.publish(
        types.ModelResponseCompleted(outcome=Succeeded(value=reply("ok"))),
    )
    now[0] += 1.0

    assert not pruner.should_run()


@pytest.mark.parametrize("server", [False, True], ids=["client", "server"])
def test_a_given_policy_overrides_the_models(tmp_path: Path, *, server: bool) -> None:
    agent, _, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=None,
        manage_context_server_side=server,
    )
    agent.session.append(*_turns("x" * 40_000, requests=20))
    given = Pruner(agent, policy=PruningPolicy(keep_recent=0, expiry=0.0))

    assert not Pruner(agent).should_run()
    assert given.should_run()
    _ = given.prune()
    assert _noted(agent) == {"c0"}


@dataclass(slots=True, kw_only=True)
class _Asked:
    """A pruner over a real agent whose runtime ends each prune with ``outcome``."""

    pruner: Pruner
    asked: list[types.SessionPrune]


def _asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Outcome[None],
    *context: ModelContextEvent,
    keep_recent: int = 0,
) -> _Asked:
    agent = Agent(
        model=ScriptedProvider()["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
    )
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=keep_recent),
    )
    for entry in context:
        _ = agent.runtime.tape.append_history(entry)
    asked: list[types.SessionPrune] = []

    async def ask(request: types.SessionPrune) -> Outcome[None]:
        asked.append(request)
        return outcome

    monkeypatch.setattr(agent.runtime, "ask", ask)
    return _Asked(pruner=Pruner(agent), asked=asked)


@pytest.mark.asyncio
async def test_a_pruner_takes_no_instructions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _asked(tmp_path, monkeypatch, Succeeded(value=None))

    with pytest.raises(ValueError, match=r"^A pruner takes no instructions\.$"):
        await runtime.pruner.run("keep the plan")

    assert runtime.asked == []


@pytest.mark.asyncio
async def test_a_forced_prune_asks_the_runtime_for_all_the_policy_lets_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _asked(
        tmp_path,
        monkeypatch,
        Succeeded(value=None),
        reply(calls=[ToolCall(call_id="c0", name="Read")]),
        ToolMessage(call_id="c0", state="done", stdout=("x" * 400,)),
        ToolMessage(call_id="c9", state="pending", stdout=("wake\n" + "y" * 400,)),
        reply("done"),
    )

    await runtime.pruner.run()

    assert runtime.asked == [
        types.SessionPrune(
            call_ids=("c0", "c9"),
            reason="every output the policy lets go",
        ),
    ]


@pytest.mark.asyncio
async def test_a_forced_prune_names_no_call_with_an_output_the_policy_keeps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call run apart has several outputs, and a prune names the call: all of them."""
    runtime = _asked(
        tmp_path,
        monkeypatch,
        Succeeded(value=None),
        reply(calls=[ToolCall(call_id="c9", name="Bash")]),
        ToolMessage(call_id="c9", state="pending", shown="[detached]"),
        reply("waiting"),
        ToolMessage(call_id="c9", state="pending", stdout=("news\n" + "y" * 400,)),
        reply("still waiting"),
        ToolMessage(call_id="c9", state="done", stdout=("z" * 400,)),
        reply("done"),
        keep_recent=1,
    )

    await runtime.pruner.run()

    assert runtime.asked == [
        types.SessionPrune(call_ids=(), reason="every output the policy lets go"),
    ]


def _result(call_id: str, text: str = "x" * 400) -> ToolMessage:
    return ToolMessage(call_id=call_id, state="done", stdout=(text,))


def _wake(call_id: str, text: str = "x" * 400) -> ToolMessage:
    return ToolMessage(
        call_id=call_id,
        state="pending",
        stdout=(f"[talk {call_id} still running after 15s]\n{text}",),
    )


@pytest.mark.parametrize(
    ("policy", "server", "client"),
    [
        (None, False, None),
        (PruningPolicy(keep_recent=1), False, PruningPolicy(keep_recent=1)),
        (PruningPolicy(keep_recent=1), True, None),
    ],
    ids=["no-policy", "the-client-prunes", "the-server-prunes"],
)
def test_the_client_prunes_by_the_policy_unless_the_server_does(
    policy: PruningPolicy | None,
    server: bool,
    client: PruningPolicy | None,
) -> None:
    settings = ModelSettings(pruning=policy, manage_context_server_side=server)
    assert client_policy(settings) == client


def test_an_entry_is_read_by_each_reply_after_it() -> None:
    context = [
        UserMessage(content="go"),
        reply(calls=[ToolCall(call_id="c0", name="Read")]),
        _result("c0"),
        reply("done"),
        UserMessage(content="more"),
    ]
    assert reads(context) == [2, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("policy", "kept"),
    [
        (None, {"c0", "c1", "c2", "c3"}),
        (PruningPolicy(keep_recent=0), set[str]()),
        (PruningPolicy(keep_recent=1), {"c3"}),
        (PruningPolicy(keep_recent=9), {"c0", "c1", "c2", "c3"}),
        (PruningPolicy(keep_recent=0, keep_tools=frozenset({"Memory"})), {"c1"}),
    ],
    ids=["no-policy", "none-kept", "the-newest", "more-than-there-are", "a-kept-tool"],
)
def test_the_policy_says_which_results_may_go(
    policy: PruningPolicy | None,
    kept: set[str],
) -> None:
    context = [
        reply(
            calls=[
                ToolCall(call_id=f"c{i}", name="Memory" if i == 1 else "Read")
                for i in range(4)
            ],
        ),
        *(_result(f"c{i}", str(i) * 2) for i in range(4)),
        reply("done"),
    ]
    assert eligible(context, policy, stub="_") == [
        i + 1 for i in range(4) if f"c{i}" not in kept
    ]


def test_an_output_no_reply_has_followed_may_not_go() -> None:
    context = [
        reply(calls=[ToolCall(call_id="c0", name="Read")]),
        _result("c0"),
        reply(calls=[ToolCall(call_id="c1", name="Read")]),
        _result("c1"),
    ]
    assert eligible(context, PruningPolicy(keep_recent=0), stub="_") == [1]


def test_an_output_a_drop_would_not_shrink_may_not_go() -> None:
    context = [
        reply(calls=[ToolCall(call_id="c0", name="Read")]),
        _result("c0", "tiny"),
        ToolMessage(call_id="c1", state="pending", shown=types.Stubs().detached),
        _result("c2"),
        reply("done"),
    ]
    assert eligible(context, PruningPolicy(keep_recent=0), stub="[gone]") == [3]


def test_a_long_call_s_news_may_go_and_is_no_result_to_keep() -> None:
    """``keep_recent`` counts tool results; a wake after one does not push it out."""
    context = [
        reply(calls=[ToolCall(call_id="c0", name="Read")]),
        _result("c0"),
        reply("read it"),
        _wake("c9"),
        SystemMessage(content="x" * 400, subtype="reminder"),
        reply("waiting"),
        _wake("c9"),
        reply("still"),
    ]
    assert eligible(context, PruningPolicy(keep_recent=1), stub="[gone]") == [3, 6]


@pytest.mark.asyncio
async def test_a_forced_prune_that_fails_raises_what_stopped_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _asked(
        tmp_path,
        monkeypatch,
        Failed(exception=ValueError("no such entry")),
    )

    with pytest.raises(ValueError, match=r"^no such entry$"):
        await runtime.pruner.run()


@pytest.mark.asyncio
async def test_a_forced_prune_the_runtime_quit_before_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _asked(tmp_path, monkeypatch, Cancelled())

    with pytest.raises(asyncio.CancelledError):
        await runtime.pruner.run()


def test_a_drop_that_pays_is_noted_for_the_model_not_taken(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.session.append(*_turns("x" * 40_000, requests=20))

    # A note while there is room puts nothing off.
    assert not pruner.prune()

    assert _prunes(agent) == []
    assert _results(agent) == {"c0": "x" * 40_000}
    # Keep: 0.3e-6 x (10,000 - 14.75); reread: 0.3e-6 x P + 3.75e-6 x 10,000.
    assert _notes(agent) == [
        (
            "Prunable (keep: $ per request; reread: $ at current context):\n"
            "- #2 Read: $0.003; $0.041\n"
            "Pruning rewrites the cache once ($0.00014); pruned results must be reread"
            " to attend to.\n"
            "Prune only what you won't need, with AgentSession prune_queue, only now or"
            " when the user asks. Cost never justifies partial reads or dropping rules"
            " you follow."
        ),
    ]


def test_a_note_s_rent_is_what_a_result_holds_beyond_its_stub(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=_UNCACHED)
    agent.session.append(*_turns("x" * 400, tail="t" * 4000))

    _ = pruner.prune()
    # 3e-6 x (100 - 14.75); with the stub added it would read $0.00034.
    assert _notes(agent)[0].splitlines()[1] == "- #2 Read: $0.00026; $0.0036"


def test_a_note_names_news_whose_call_is_gone_as_such(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=_UNCACHED)
    for entry in (
        UserMessage(content="go"),
        _wake("c9"),
        reply("done"),
    ):
        _ = agent.runtime.tape.append_history(entry)

    _ = pruner.prune()
    assert (
        _notes(agent)[0]
        .splitlines()[1]
        .startswith(
            "- #1 output of a call no longer shown: $",
        )
    )


@pytest.mark.parametrize("price", [_CACHED, None], ids=["priced", "unpriced"])
def test_each_result_sent_says_its_size_and_nothing_a_price_change_moves(
    tmp_path: Path,
    price: TokenPrice | None,
) -> None:
    agent, _, _ = _agent(tmp_path, price=price)
    agent.session.append(*_turns("x" * 4000, "tiny"))

    sent = [r for r in agent.runtime.request().records if isinstance(r, ToolMessage)]

    assert [r.text() for r in sent] == [f"{'x' * 4000}\n[1,000 tok]", "tiny"]


def test_a_note_names_each_result_by_its_tool_s_summary(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, tools=[Read()])
    agent.session.append(
        UserMessage(content="go"),
        *_read("c0", _P),
        *_turns(requests=20),
    )

    _ = pruner.prune()
    assert f"- #2 Read {_P}: $" in _notes(agent)[0]


def test_a_result_noted_and_left_is_not_noted_again_until_it_expires_again(
    tmp_path: Path,
) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0),
    )
    # Unused 21 requests; it expires after 14 unused.
    agent.session.append(*_read_turn(_P, before=15, named="/repo/other.py", after=5))
    _ = pruner.prune()
    assert _noted(agent) == {"c0"}

    agent.session.append(*_turns(requests=10))
    _ = pruner.prune()
    assert len(_notes(agent)) == 1
    # A pruner that starts over, as on a resume, reads what was listed off the tape.
    fresh = Pruner(agent)
    _ = fresh.prune()
    assert len(_notes(agent)) == 1
    agent.session.append(*_turns(requests=10))

    _ = fresh.prune()
    assert len(_notes(agent)) == 2


def test_a_listing_counts_from_where_its_note_stands_after_a_compaction(
    tmp_path: Path,
) -> None:
    """A compaction that keeps a listed result and its note moves both together."""
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0),
    )
    agent.session.append(*_turns(requests=30))
    agent.session.append(*_read_turn(_P, before=15, named="/repo/other.py", after=5))
    _ = pruner.prune()
    assert _noted(agent) == {"c0"}
    _compact_before(agent, call_id="c0")

    agent.session.append(*_turns(requests=20))
    _ = pruner.prune()

    assert len(_notes(agent)) == 2


def test_a_queue_drops_in_one_splice_once_it_pays(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0, expiry=math.inf),
    )
    agent.session.append(*_turns("x" * 4000, "y" * 4000, tail="t" * 4000))

    agent.runtime.queued["c0"] = "spent"
    assert not pruner.prune()
    agent.session.append(*_turns(requests=1))
    agent.runtime.queued["c1"] = "spent too"
    assert not pruner.prune()
    assert _prunes(agent) == []
    agent.session.append(*_turns(requests=12))

    # A drop asks nothing of the model, so it puts no compaction off.
    assert not pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": _STUB}
    assert len(_prunes(agent)) == 1
    assert agent.runtime.queued == {}


def test_a_queue_that_does_not_pay_yet_reports_its_shortfall(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))
    agent.runtime.queued["c0"] = "spent"

    chosen, worth, due = pruner.pending()

    assert chosen == [2]
    assert worth is not None
    assert worth.priced is not None
    assert worth.priced[0] < 0
    assert not due


def test_nothing_queued_is_nothing_pending(tmp_path: Path) -> None:
    _, pruner, _ = _agent(tmp_path)
    assert pruner.pending() == ([], None, False)


def test_superseded_results_after_the_queue_ride_along(tmp_path: Path) -> None:
    uncached = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=3.0,
    )
    agent, pruner, _ = _agent(tmp_path, price=uncached)
    agent.session.append(
        UserMessage(content="go"),
        *_shell("early", "make"),
        *_read("a", _P),
        *_read("b", _P),
        AssistantMessage(content="read"),
    )
    agent.runtime.queued["early"] = "spent"

    _ = pruner.prune()

    assert _results(agent) == {"early": _STUB, "a": _STUB, "b": "x" * 4000}


def test_an_unpriced_queue_drops_at_once(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))
    agent.runtime.queued["c0"] = "spent"

    _ = pruner.prune()
    assert _results(agent) == {"c0": _STUB}


def test_a_note_while_there_is_room_puts_no_due_compaction_off(
    tmp_path: Path,
) -> None:
    """Full is what the compactor says; the pruner's own count does not decide it."""
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    agent.session.append(*_turns("x" * 12_000, "y" * 400))
    agent.runtime.compact_due = lambda: False

    assert not pruner.prune()
    assert "The context is full" not in _notes(agent)[0]


def test_a_full_context_notes_once_then_leaves_it_to_compaction(
    tmp_path: Path,
) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    # 3,100 tokens against a 3,000 trigger; the two results are most of it.
    agent.session.append(*_turns("x" * 12_000, "y" * 400))
    agent.runtime.compact_due = lambda: True

    assert pruner.prune()
    assert _notes(agent) == [
        (
            "Prunable (tokens):\n"
            "- #2 Read: 3,000 tok\n"
            "- #4 Read: 100 tok\n"
            "The context is full: without pruning, it compacts next.\n"
            "Pruned results must be reread to attend to.\n"
            "Prune only what you won't need, with AgentSession prune_queue, only now or"
            " when the user asks. Cost never justifies partial reads or dropping rules"
            " you follow."
        ),
    ]
    assert not pruner.prune()
    # Once its note is compacted away, a result may be offered again.
    _compact_before(agent, call_id="c0", keep_note=False)
    assert pruner.prune()

    agent.runtime.queued.update({"c0": "spent", "c1": "spent"})
    assert not pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": _STUB}


@pytest.mark.asyncio
async def test_the_runtime_waits_one_round_to_compact_after_a_note(
    tmp_path: Path,
) -> None:
    agent, _, provider = _agent(
        tmp_path,
        price=None,
        window=4000,
        compactor=SummaryCompactor,
    )
    provider.replies.extend(
        [reply("noted"), reply("<summary>gist</summary>"), reply("done")],
    )
    async with agent:
        agent.session.append(*_turns("x" * 12_000, "y" * 400))

        assert await agent.send("next") == "noted"
        assert await agent.send("again") == "done"

    first, second, _ = provider.requests
    assert any(
        isinstance(r, SystemMessage) and r.subtype == "prune" for r in first.records
    )
    assert second.purpose == "compaction"


def _compact_before(agent: Agent, *, call_id: str, keep_note: bool = True) -> None:
    """Splice the context down to ``call_id``'s call on, as a compaction keeping a tail does."""
    runtime = agent.runtime
    context = runtime.tape.context().messages
    start = next(
        i
        for i, e in enumerate(context)
        if isinstance(e, Response)
        and any(isinstance(r, ToolCall) and r.call_id == call_id for r in e.records)
    )
    tail = tuple(
        e
        for e in context[start:]
        if keep_note or not (isinstance(e, SystemMessage) and e.subtype == "prune")
    )
    replacement = (UserMessage(content="summary"), *tail)
    _ = runtime.tape.append_splice(
        mask=full_tape_mask(runtime.tape),
        replacement=replacement,
        strategy="summary",
        paired_externally=unpaired_call_ids(replacement),
        discards_content=True,
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
