"""Tests for the pruner: old tool results dropped once that costs less than keeping them."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import math
import time

import pytest

from sagent.agent.agent import Agent, NoShrinker
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.pruner import (
    Pruner,
    Saving,
    cut,
    drop,
    expired,
    idle,
    limit_tokens,
    reads,
    requests_left,
    saving,
    sizes,
    superseded,
)
from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    PruningPolicy,
)
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.scripted import ScriptedProvider, reply
from sagent.runtime import types
from sagent.runtime.tape import ContextSplice
from sagent.runtime.types import Markers
from sagent.session.types import (
    AssistantMessage,
    FileEditResult,
    FileReadResult,
    ShellCommandResult,
    ToolCall,
    ToolResult,
    UncategorizedToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from sagent.agent.types import Factory, Shrinker
    from sagent.runtime.types import ModelContextEvent
    from sagent.session.types import SessionRecord


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


def _agent(
    tmp_path: Path,
    *,
    price: TokenPrice | None = _CACHED,
    stub: str | None = None,
    window: int = 200_000,
    compactor: Factory[Shrinker] = NoShrinker,
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
            cache_ttl_sec=frozenset({300.0, 3600.0}),
            default=ModelSettings(
                cache_ttl_sec=300.0,
                context_size=prices["auto", "fullcontext"],
            ),
        ),
    )
    agent = Agent(
        model=provider["priced"],
        compactor=compactor,
        pruner=Pruner if stub is None else partial(Pruner, stub=stub),
        cwd=tmp_path,
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
        records.append(UncategorizedToolResult(call_id=f"c{i}", content=content))
    records.append(AssistantMessage(content=tail))
    for i in range(requests):
        records.append(UserMessage(content=f"q{i}"))
        records.append(AssistantMessage(content=f"a{i}"))
    return records


def _read(
    call_id: str,
    path: str,
    *ranges: tuple[int, int | None],
) -> list[SessionRecord]:
    """Return a Read of ``ranges`` of ``path``, 1000 tokens long; all of it when none."""
    return [
        ToolCall(call_id=call_id, name="Read", arguments={"file_path": path}),
        FileReadResult(call_id=call_id, path=path, content="x" * 4000, ranges=ranges),
    ]


def _shell(call_id: str, command: str) -> list[SessionRecord]:
    return [
        ToolCall(call_id=call_id, name="Bash", arguments={"command": command}),
        ShellCommandResult(
            call_id=call_id,
            command=("bash", "-c", command),
            stdout="o" * 4000,
        ),
    ]


def _context(tmp_path: Path, *records: SessionRecord) -> Sequence[ModelContextEvent]:
    """Return the context of a session that said ``records``, one reply each call."""
    agent, _, _ = _agent(tmp_path)
    agent.session.append(UserMessage(content="go"), *records)
    agent.session.append(AssistantMessage(content="done"))
    return agent.runtime.context().messages


def _ids(context: Sequence[ModelContextEvent], indices: Collection[int]) -> set[str]:
    return {
        entry.call_id for i in indices if isinstance(entry := context[i], ToolResult)
    }


def _results(agent: Agent) -> dict[str, str]:
    return {r.call_id: r.text() for r in agent.session if isinstance(r, ToolResult)}


def _prunes(agent: Agent) -> list[ContextSplice]:
    return [
        r
        for r in agent.runtime.tape
        if isinstance(r, ContextSplice) and r.strategy == "prune"
    ]


def test_a_drop_that_costs_more_than_it_saves_waits(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # One request has read 1000 tokens, with 1000 kept after it to rewrite:
    # 1 x 985 cached reads paid is less than 11.5 x 1015.
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    assert not pruner.should_run()
    assert not pruner.prune()
    assert _prunes(agent) == []


def test_a_drop_pays_once_enough_requests_will_read_it(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Twenty-one requests have read 10,000 tokens of result; dropping it
    # rewrites only the few dozen tokens of short turns after it.
    agent.session.append(*_turns("x" * 40_000, requests=20))

    assert pruner.should_run()
    assert pruner.prune()

    assert _results(agent) == {"c0": _STUB}
    assert len(_prunes(agent)) == 1


def test_a_drop_waits_until_the_reads_paid_cover_the_rewrite(tmp_path: Path) -> None:
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
    assert pruner.prune()
    assert _results(agent) == {"c0": _STUB}


def test_a_result_the_model_has_not_read_is_never_dropped(tmp_path: Path) -> None:
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
        UncategorizedToolResult(call_id="read", content="x" * 4000),
        AssistantMessage(content="ok"),
        UserMessage(content="next"),
        ToolCall(call_id="unread", name="Read"),
        UncategorizedToolResult(call_id="unread", content="y" * 4000),
        UserMessage(content="also"),
    )

    assert pruner.prune()
    assert _results(agent) == {"read": _STUB, "unread": "y" * 4000}


_P = "/repo/agent/x.py"


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
        # An earlier read covers nothing; another file covers nothing.
        ([*_read("a", _P, (1, 20)), *_read("b", _P, (1, 10))], set[str]()),
        ([*_read("a", _P), *_read("b", "/repo/agent/y.py")], set[str]()),
        ([*_shell("a", "ls"), *_shell("b", "ls")], {"a"}),
        ([*_shell("a", "ls"), *_shell("b", "ls -a")], set[str]()),
    ],
)
def test_a_result_is_superseded_once_all_it_showed_is_shown_again(
    tmp_path: Path,
    records: list[SessionRecord],
    gone: set[str],
) -> None:
    context = _context(tmp_path, *records)

    assert _ids(context, superseded(context)) == gone


def test_a_result_is_idle_since_its_file_was_last_named(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        *_read("a", _P),
        *_turns(requests=4),
        ToolCall(call_id="g", name="Grep", arguments={"pattern": "def", "path": _P}),
        UncategorizedToolResult(call_id="g", content="hit"),
        *_turns(requests=2),
        ToolCall(
            call_id="s",
            name="Bash",
            arguments={"command": "sed -n 1p agent/x.py"},
        ),
        UncategorizedToolResult(call_id="s", content="line"),
        *_turns(requests=3),
        ToolCall(call_id="e", name="Edit", arguments={"file_path": "/repo/y.py"}),
        FileEditResult(call_id="e", path="/repo/y.py"),
        *_turns(requests=1),
    )
    at = {e.call_id: i for i, e in enumerate(context) if isinstance(e, ToolResult)}
    k = reads(context)

    quiet = idle(context)

    # Grep named the file, then Bash by its last two parts: idle since Bash.
    assert quiet[at["a"]] == k[at["s"]] < k[at["g"]] < k[at["a"]]
    # A result with no file, or whose file nothing named again: since it came.
    assert quiet[at["g"]] == k[at["g"]]
    assert quiet[at["e"]] == k[at["e"]] == 3


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
        UncategorizedToolResult(call_id="g", content="hit"),
        AssistantMessage(content="ok"),
        *_turns(requests=after),
    ]


def test_a_file_result_naming_no_file_is_idle_since_it_came(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        ToolCall(call_id="e", name="Edit"),
        FileEditResult(call_id="e", path=None),
        *_turns(requests=2),
        ToolCall(call_id="g", name="Grep", arguments={"pattern": "None"}),
        UncategorizedToolResult(call_id="g", content="hit"),
        *_turns(requests=1),
    )
    (at,) = (i for i, e in enumerate(context) if isinstance(e, FileEditResult))

    assert idle(context)[at] == reads(context)[at]


@pytest.mark.parametrize("server", [False, True])
def test_with_no_policy_to_prune_by_nothing_is_planned(
    tmp_path: Path,
    server: bool,
) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=PruningPolicy(keep_recent=0) if server else None,
        manage_context_server_side=server,
    )
    agent.session.append(*_turns("x" * 40_000, requests=50))

    assert not pruner.should_run()
    assert not pruner.prune()


def test_a_file_named_deep_in_frozen_arguments_is_in_use(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        *_read("a", _P),
        *_turns(requests=4),
        ToolCall(
            call_id="m",
            name="MultiEdit",
            arguments={"edits": [{"files": [_P]}]},
        ),
        UncategorizedToolResult(call_id="m", content="done"),
        *_turns(requests=1),
    )
    at = {e.call_id: i for i, e in enumerate(context) if isinstance(e, ToolResult)}

    assert idle(context)[at["a"]] == reads(context)[at["m"]]


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

    assert pruner.prune() is not kept
    assert (_results(agent)["c0"] == _STUB) is not kept


def test_a_superseded_read_goes_while_the_newer_stays(tmp_path: Path) -> None:
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

    assert pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": "x" * 4000}


def test_a_cut_its_reads_have_not_paid_for_waits(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path)
    # Dropping both rewrites the 10,000-token answer between them, more than
    # 3 + 1 reads of 9985 paid; dropping the second alone rewrites almost nothing.
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="early", name="Read"),
        UncategorizedToolResult(call_id="early", content="e" * 40_000),
        AssistantMessage(content="a" * 40_000),
        UserMessage(content="more"),
        ToolCall(call_id="late", name="Read"),
        UncategorizedToolResult(call_id="late", content="l" * 40_000),
        AssistantMessage(content="done"),
    )

    assert pruner.prune()

    assert _results(agent) == {"early": "e" * 40_000, "late": _STUB}


def test_an_expired_cache_makes_any_drop_pay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, pruner, _ = _agent(tmp_path)
    agent.session.append(*_turns("x" * 4000, "y" * 4000, tail="t" * 4000))
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    agent.runtime.publish(types.ModelResponseComplete(response=reply("ok")))

    now[0] += 300.0
    assert not pruner.should_run()  # Still warm at its lifetime.
    now[0] += 1.0

    assert pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": _STUB}


def test_without_a_cache_discount_every_drop_pays(tmp_path: Path) -> None:
    uncached = TokenPrice(
        request=3.0,
        response=15.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=3.0,
    )
    agent, pruner, _ = _agent(tmp_path, price=uncached)
    agent.session.append(*_turns("x" * 4000, tail="t" * 4000))

    assert pruner.prune()
    assert _results(agent) == {"c0": _STUB}


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
@pytest.mark.parametrize(
    ("policy", "server", "kept"),
    [
        (None, False, {"c0", "c1", "c2", "c3"}),
        (PruningPolicy(keep_recent=0), True, {"c0", "c1", "c2", "c3"}),
        (PruningPolicy(keep_recent=1), False, {"c3"}),
        (PruningPolicy(keep_recent=9), False, {"c0", "c1", "c2", "c3"}),
        (PruningPolicy(keep_recent=0, keep_tools=frozenset({"Memory"})), False, {"c1"}),
    ],
)
async def test_the_policy_says_which_results_may_go(
    tmp_path: Path,
    policy: PruningPolicy | None,
    server: bool,
    kept: set[str],
) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None)
    agent.model.settings = replace(
        agent.model.settings,
        pruning=policy,
        manage_context_server_side=server,
    )
    records = _turns("0" * 400, "1" * 400, "2" * 400, "3" * 400)
    records[3] = ToolCall(call_id="c1", name="Memory")
    async with agent:
        agent.session.append(*records)

        # ``run`` drops every result allowed, unpriced or not.
        await pruner.run()

    assert {cid for cid, text in _results(agent).items() if text != _STUB} == kept


@pytest.mark.asyncio
async def test_a_small_result_or_a_stub_is_never_dropped(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, stub="[gone]")
    pending = UncategorizedToolResult(
        call_id="c1",
        content=Markers().detached,
        stub="pending",
    )
    async with agent:
        agent.session.append(
            UserMessage(content="go"),
            ToolCall(call_id="c0", name="Read"),
            UncategorizedToolResult(call_id="c0", content="tiny"),
            ToolCall(call_id="c1", name="Bash"),
            pending,
            ToolCall(call_id="c2", name="Read"),
            UncategorizedToolResult(call_id="c2", content="x" * 400, is_error=True),
            AssistantMessage(content="started"),
        )

        await pruner.run()

    results = {r.call_id: r for r in agent.session if isinstance(r, ToolResult)}
    assert results["c0"].text() == "tiny"
    assert results["c1"] == pending
    assert (results["c2"].text(), results["c2"].is_error) == ("[gone]", True)


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
async def test_the_runtime_prunes_before_a_model_call_that_it_pays_for(
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
    sent = [r for r in request.records if isinstance(r, ToolResult)]
    assert [r.text() for r in sent] == [_STUB]


@pytest.mark.parametrize(
    ("requests", "prompt", "first", "left"),
    [
        (0, 100.0, 0.0, 0.0),  # No request yet.
        (4, 100.0, 100.0, 4.0),  # No growth: n.
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


def test_a_saving_needs_a_result_to_drop() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _ = saving([1.0], [], stub=0.0, prices=None, left=1.0)


def test_only_a_tool_result_may_be_dropped(tmp_path: Path) -> None:
    agent, _, _ = _agent(tmp_path)
    agent.session.append(*_turns("x" * 400))
    with pytest.raises(ValueError, match="Entry 0 is no tool result"):
        drop(agent.runtime, [0], stub="gone")
    drop(agent.runtime, [], stub="gone")
    drop(agent.runtime, [2], stub="gone")
    assert _results(agent) == {"c0": "gone"}


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


def test_a_drop_that_averts_compaction_is_taken_unpriced(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    # 3,100 tokens against a 3,000 trigger; the two results are most of it.
    agent.session.append(*_turns("x" * 12_000, "y" * 400))

    assert pruner.prune()
    assert _results(agent) == {"c0": _STUB, "c1": _STUB}


def test_a_drop_too_small_to_avert_compaction_is_left_to_it(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=4000)
    agent.session.append(*_turns("x" * 400, tail="t" * 14_000))

    assert not pruner.prune()


def test_the_compactor_s_own_trigger_is_the_limit(tmp_path: Path) -> None:
    records = _turns("x" * 9200, "y" * 800)
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
    held = sum(sizes(compacting.model, compacting.runtime.context().messages))
    assert limit_tokens(compacting) < held < limit_tokens(plain)

    assert compacting_pruner.should_run()
    assert not plain_pruner.should_run()


def test_a_model_with_no_window_never_compacts(tmp_path: Path) -> None:
    agent, pruner, _ = _agent(tmp_path, price=None, window=0)
    agent.session.append(*_turns("x" * 40_000, requests=20))

    assert not pruner.should_run()


@pytest.mark.asyncio
async def test_a_pruner_takes_no_instructions(tmp_path: Path) -> None:
    _, pruner, _ = _agent(tmp_path)

    with pytest.raises(ValueError, match="no instructions"):
        await pruner.run("keep the plan")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
