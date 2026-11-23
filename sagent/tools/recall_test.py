"""Tests for Recall: what it asks the server, what it writes, and when it wakes.

The server is never reached: the fetch is replaced where the tool imports it, and
answers from a script. How a woken model is told belongs to the runtime's tests.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, override

from treekle import json_dumps
from wesearch.fetch import FetchSession, RequestParams
from wesearch.types.errors import FetchError

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.journal.jsonl import JsonlJournal
from sagent.journal.projects import child_journal
from sagent.journal.types import RecallResult
from sagent.model.capability import ModelCapability
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.tools import recall
from sagent.tools.recall import Recall
from sagent.tools.types import Stream, ToolContext, settled


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree

    from sagent.journal.types import ToolMessage


def test_a_call_is_summarized_by_its_name_and_runs_beside_anything(
    tmp_path: Path,
) -> None:
    team, _agent = _team(tmp_path)
    tool = Recall(team=team)
    assert tool.summary({}) == "Recall"
    assert tool.serialize_key(ToolContext(cwd=tmp_path), {}) is None


@pytest.mark.asyncio
async def test_each_session_is_written_once_and_each_search_that_found_one_wakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, agent = _team(tmp_path)
    asked = _answers(
        monkeypatch,
        _found(_hit("a", idx=3), _hit("a", idx=5), _hit("b", title="")),
        _found(_hit("a", idx=7), _hit("b")),
        _found(_hit("c", snippet="one\n  two")),
    )
    slept: list[float] = []
    events: list[object] = []
    monkeypatch.setattr(recall, "Stream", partial(_Logged, events))
    monkeypatch.setattr(agent.runtime, "wake", events.append)
    tool = Recall(
        team=team,
        url="https://trax.example",
        key="k",
        session="00000000-0000-0000-0000-000000000009",
        interval_sec=7.0,
        sleep=partial(_sleep, slept),
    )
    fetch_session = FetchSession()
    streaming_result = tool.call(
        ToolContext(cwd=tmp_path, label="Agent", fetch_session=fetch_session),
        {},
    )
    # It says it never ends of itself, so its agent idles while it runs.
    assert streaming_result.runs_until_stopped

    result = await settled(streaming_result)

    # Each ends its line: the model reads what a call wrote joined, with nothing between.
    lines = (
        "session a p0#3 (title a): snippet a\n",
        "session b p0#0 (untitled): snippet b\n",
        "session c p0#0 (title c): one two\n",
    )
    assert result.stdout == lines
    # Each search that found a session wakes once, after it wrote them: the
    # second found none it had not named.
    calling = streaming_result.result
    assert events == [lines[0], lines[1], calling, lines[2], calling]
    assert slept == [7.0, 7.0, 7.0]
    assert [(url, session) for url, _, session in asked] == [
        ("https://trax.example/api/web/search_sessions", fetch_session),
    ] * 4
    request = asked[0][1]
    assert request.content.params == {
        "near": "00000000-0000-0000-0000-000000000009",
        "mode": "vector",
        "limit": 5,
    }
    assert request.content.headers == {"Authorization": "Bearer k"}
    assert request.policy.allow_private_hosts
    # The script ran out: the server's failure is the call's.
    assert (result.state, result.stderr) == (
        "failed",
        ("FetchError: HTTP 503: https://trax.example/api/web/search_sessions",),
    )


@pytest.mark.asyncio
async def test_unset_settings_are_read_from_the_environment_and_no_key_sends_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, _agent = _team(tmp_path)
    monkeypatch.setenv("TRACKINIZER_URL", "https://trax.example/")
    monkeypatch.setenv("TRAX_SESSION", "00000000-0000-0000-0000-000000000009")
    monkeypatch.delenv("TRACKINIZER_TOKEN", raising=False)
    monkeypatch.delenv("TRACKINIZER_TOKEN_FILE", raising=False)
    asked = _answers(monkeypatch)

    result = await _called(Recall(team=team), tmp_path, {"limit": 2})

    ((url, request, _session),) = asked
    assert url == "https://trax.example/api/web/search_sessions"
    assert request.content.params == {
        "near": "00000000-0000-0000-0000-000000000009",
        "mode": "vector",
        "limit": 2,
    }
    assert request.content.headers == {}
    assert result.state == "failed"


@pytest.mark.asyncio
async def test_the_environment_s_key_goes_to_the_environment_s_server_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, _agent = _team(tmp_path)
    monkeypatch.setenv("TRACKINIZER_URL", "https://other.example")
    monkeypatch.setenv("TRAX_SESSION", "other")
    monkeypatch.setenv("TRACKINIZER_TOKEN", "from-env")
    asked = _answers(monkeypatch)

    for tool in (
        Recall(team=team),
        Recall(team=team, key="given"),
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
            key="given",
        ),
    ):
        _ = await _called(tool, tmp_path, {})

    other = "https://other.example/api/web/search_sessions"
    assert [
        (url, request.content.params["near"], request.content.headers)
        for url, request, _ in asked
    ] == [
        (other, "other", {"Authorization": "Bearer from-env"}),
        (other, "other", {"Authorization": "Bearer given"}),
        # A server the environment did not name is never sent the environment's key.
        (
            "https://trax.example/api/web/search_sessions",
            "00000000-0000-0000-0000-000000000009",
            {},
        ),
        (
            "https://trax.example/api/web/search_sessions",
            "00000000-0000-0000-0000-000000000009",
            {"Authorization": "Bearer given"},
        ),
    ]


@pytest.mark.asyncio
async def test_the_key_is_the_token_else_what_the_token_file_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, _agent = _team(tmp_path)
    monkeypatch.setenv("TRACKINIZER_URL", "https://trax.example")
    monkeypatch.setenv("TRAX_SESSION", "00000000-0000-0000-0000-000000000009")
    held = tmp_path / "key"
    _ = held.write_text(" from-file\n")
    asked = _answers(monkeypatch)

    monkeypatch.delenv("TRACKINIZER_TOKEN", raising=False)
    monkeypatch.setenv("TRACKINIZER_TOKEN_FILE", str(held))
    _ = await _called(Recall(team=team), tmp_path, {})
    # With the token set the file is not read: this one is not there.
    monkeypatch.setenv("TRACKINIZER_TOKEN", "from-env")
    monkeypatch.setenv("TRACKINIZER_TOKEN_FILE", str(tmp_path / "missing"))
    _ = await _called(Recall(team=team), tmp_path, {})

    assert [request.content.headers for _, request, _ in asked] == [
        {"Authorization": "Bearer from-file"},
        {"Authorization": "Bearer from-env"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token", "held", "refusal"),
    [
        ("", None, "TRACKINIZER_TOKEN_FILE {path}: No such file or directory"),
        (
            "",
            b"two words",
            "TRACKINIZER_TOKEN_FILE {path} must hold printable ASCII with no whitespace",
        ),
        (
            "",
            "calf\u00e9".encode(),
            "TRACKINIZER_TOKEN_FILE {path} must hold printable ASCII with no whitespace",
        ),
        # No text at all: read as text, its bytes would be quoted in the failure.
        (
            "",
            b"\xffkey",
            "TRACKINIZER_TOKEN_FILE {path} must hold printable ASCII with no whitespace",
        ),
        (
            "two words",
            None,
            "TRACKINIZER_TOKEN must be printable ASCII with no whitespace",
        ),
        (
            "calf\u00e9",
            None,
            "TRACKINIZER_TOKEN must be printable ASCII with no whitespace",
        ),
    ],
    ids=[
        "no-file",
        "file-with-a-space",
        "file-not-ascii",
        "file-not-text",
        "token-with-a-space",
        "token-not-ascii",
    ],
)
async def test_a_key_that_cannot_be_read_or_sent_is_refused_without_its_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    held: bytes | None,
    refusal: str,
) -> None:
    team, _agent = _team(tmp_path)
    monkeypatch.setenv("TRACKINIZER_URL", "https://trax.example")
    monkeypatch.setenv("TRAX_SESSION", "00000000-0000-0000-0000-000000000009")
    monkeypatch.setenv("TRACKINIZER_TOKEN", token)
    path = tmp_path / "key"
    monkeypatch.setenv("TRACKINIZER_TOKEN_FILE", str(path))
    if held is not None:
        _ = path.write_bytes(held)
    asked = _answers(monkeypatch, _found(_hit("a")))

    result = await _called(Recall(team=team), tmp_path, {})

    assert (result.state, result.stderr) == (
        "failed",
        (f"ValueError: {refusal.format(path=path)}",),
    )
    assert asked == []


@pytest.mark.asyncio
async def test_a_watcher_hears_of_sessions_like_the_one_the_setting_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, agent = _team(tmp_path)
    watcher = _spawned(team, agent, label="watcher")
    asked = _answers(monkeypatch, _found(_hit("a")))
    woken: list[str] = []
    monkeypatch.setattr(agent.runtime, "wake", partial(_woke, woken, "Agent"))
    monkeypatch.setattr(watcher.runtime, "wake", partial(_woke, woken, "watcher"))
    tool = Recall(
        team=team,
        url="https://trax.example",
        session="00000000-0000-0000-0000-000000000009",
        sleep=partial(_sleep, []),
    )

    _ = await _called(tool, tmp_path, {"agent": "Agent"}, label="watcher")

    # The session is the one named, which the setting stores; the caller is woken.
    assert [request.content.params for _, request, _ in asked] == [
        {"near": "00000000-0000-0000-0000-000000000009", "mode": "vector", "limit": 5},
    ] * 2
    assert woken == ["watcher"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller", "arguments"),
    [("watcher", {}), ("Agent", {"agent": "watcher"})],
    ids=["itself", "named"],
)
async def test_a_spawned_agent_s_session_is_named_under_the_set_one_by_its_own_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller: str,
    arguments: dict[str, PlainTree],
) -> None:
    team, agent = _team(tmp_path)
    watcher = _spawned(team, agent, label="watcher")
    path = watcher.session.journal.path
    assert path is not None
    asked = _answers(monkeypatch)

    _ = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        arguments,
        label=caller,
    )

    # A spawned agent's id is its spawner's folder, then its own.
    assert [request.content.params for _, request, _ in asked] == [
        {
            "near": "00000000-0000-0000-0000-000000000009",
            "spawned": f"{'ab' * 6}/{path.parent.name}",
            "mode": "vector",
            "limit": 5,
        },
    ]


@pytest.mark.asyncio
async def test_an_agent_that_keeps_no_journal_has_no_stored_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, _agent = _team(tmp_path)
    unkept = Agent(
        model=_agent.model,
        tool_context=ToolContext(cwd=tmp_path),
        environment=False,
    )
    _ = team.join(unkept, label="unkept")
    asked = _answers(monkeypatch, _found(_hit("a")))

    result = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        {"agent": "unkept"},
    )

    assert (result.state, result.stderr) == (
        "failed",
        ("ValueError: unkept keeps no journal, so no session of it is stored.",),
    )
    assert asked == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "session"),
    [("", "00000000-0000-0000-0000-000000000009"), ("https://trax.example", "")],
    ids=["no-server", "no-session"],
)
async def test_a_call_with_no_server_or_no_session_is_refused_and_asks_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    session: str,
) -> None:
    team, _agent = _team(tmp_path)
    monkeypatch.delenv("TRACKINIZER_URL", raising=False)
    monkeypatch.delenv("TRAX_SESSION", raising=False)
    asked = _answers(monkeypatch, _found(_hit("a")))

    result = await _called(Recall(team=team, url=url, session=session), tmp_path, {})

    refusal = (
        "ValueError: Recall needs a server and a stored session: set "
        "tool.Recall.url and tool.Recall.session, or TRACKINIZER_URL and "
        "TRAX_SESSION."
    )
    assert (result.state, result.stderr) == ("failed", (refusal,))
    assert asked == []


@pytest.mark.asyncio
async def test_a_caller_in_no_team_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, _agent = _team(tmp_path)
    asked = _answers(monkeypatch, _found(_hit("a")))

    result = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        {},
        label="Stranger",
    )

    assert (result.state, result.stderr) == (
        "failed",
        ("ValueError: Recall needs a calling agent in the team.",),
    )
    assert asked == []


@pytest.mark.asyncio
async def test_a_server_that_searched_no_vectors_ends_the_call_and_wakes_no_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, agent = _team(tmp_path)
    asked = _answers(monkeypatch, _found(_hit("a"), semantic=False))
    woken: list[object] = []
    monkeypatch.setattr(agent.runtime, "wake", woken.append)

    result = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        {},
    )

    assert (result.state, result.stdout, result.stderr) == (
        "done",
        (),
        ("The server searched no vectors: it embeds no sessions, or not yet.",),
    )
    assert result.result == RecallResult(exit_code=1)
    assert len(asked) == 1
    assert woken == []


@pytest.mark.asyncio
async def test_an_answer_of_another_shape_fails_the_call_naming_where(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team, agent = _team(tmp_path)
    misshapen = json_dumps({"hits": "none", "semantic": True, "degraded": False})
    _ = _answers(monkeypatch, misshapen.encode())
    woken: list[object] = []
    monkeypatch.setattr(agent.runtime, "wake", woken.append)

    result = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        {},
    )

    assert result.state == "failed"
    assert result.stderr[0].startswith("ReadError: ")
    assert "['hits']" in result.stderr[0]
    assert (result.stdout, woken) == ((), [])


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 51])
async def test_a_limit_out_of_bounds_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    team, _agent = _team(tmp_path)
    asked = _answers(monkeypatch, _found(_hit("a")))

    result = await _called(
        Recall(
            team=team,
            url="https://trax.example",
            session="00000000-0000-0000-0000-000000000009",
        ),
        tmp_path,
        {"limit": limit},
    )

    assert result.state == "failed"
    assert "'limit' must be at" in result.stderr[0]
    assert asked == []


class _Logged(Stream[str]):
    """A stream that notes each line written, in ``events``, beside what else is."""

    def __init__(self, events: list[object]) -> None:
        super().__init__()
        self.events = events

    @override
    def write(self, item: str) -> None:
        self.events.append(item)
        super().write(item)


def _team(tmp_path: Path) -> tuple[Team, Agent]:
    catalog = ModelCatalog(
        factories={
            "scripted": partial(
                ScriptedProvider,
                capability=ModelCapability(model_id="scripted"),
            ),
        },
    )
    team = Team(catalog=catalog)
    # A session's own folder is named by twelve hex digits.
    journal = tmp_path / ("ab" * 6) / "session.jsonl"
    journal.parent.mkdir()
    journal.touch()
    agent = Agent(
        model=catalog["scripted"]["scripted"],
        tool_context=ToolContext(cwd=tmp_path),
        journal=JsonlJournal(journal),
        environment=False,
    )
    _ = team.join(agent)
    return team, agent


def _spawned(team: Team, spawner: Agent, *, label: str) -> Agent:
    """Join an agent whose journal is a child of ``spawner``'s, as a spawn makes it."""
    above = spawner.session.journal.path
    assert above is not None
    agent = Agent(
        model=spawner.model,
        tool_context=spawner.runtime.tool_context,
        journal=JsonlJournal(child_journal(above)),
        environment=False,
    )
    _ = team.join(agent, label=label)
    return agent


async def _called(
    tool: Recall,
    tmp_path: Path,
    arguments: dict[str, PlainTree],
    *,
    label: str = "Agent",
) -> ToolMessage[RecallResult, str, str]:
    return await settled(tool.call(ToolContext(cwd=tmp_path, label=label), arguments))


def _hit(
    name: str,
    *,
    idx: int = 0,
    title: str | None = None,
    snippet: str | None = None,
) -> dict[str, PlainTree]:
    """Return one hit, shaped as the server's ``search_sessions`` answers."""
    return {
        "session_id": name,
        "part": 0,
        "idx": idx,
        "field": "content",
        "chunk": 0,
        "score": 0.5,
        "source": "semantic",
        "snippet": f"snippet {name}" if snippet is None else snippet,
        "title": f"title {name}" if title is None else title,
    }


def _found(*hits: dict[str, PlainTree], semantic: bool = True) -> bytes:
    return json_dumps(
        {"hits": list(hits), "semantic": semantic, "degraded": not semantic},
    ).encode()


# Past the script the server is down, which ends a call that would never end.
def _answers(
    monkeypatch: pytest.MonkeyPatch,
    *bodies: bytes,
) -> list[tuple[str, RequestParams, FetchSession | None]]:
    """Answer each fetch the tool makes from ``bodies``; return what it asked."""
    asked: list[tuple[str, RequestParams, FetchSession | None]] = []
    script = list(bodies)

    def fetch(
        url: str,
        *,
        request: RequestParams,
        session: FetchSession | None = None,
    ) -> bytes:
        asked.append((url, request, session))
        if not script:
            raise FetchError(url, 503, {}, b"")
        return script.pop(0)

    monkeypatch.setattr(recall, "fetch", fetch)
    return asked


def _woke(woken: list[str], label: str, task: object) -> None:
    """Note in ``woken`` that ``label``'s runtime was asked to wake its model."""
    del task
    woken.append(label)


async def _sleep(slept: list[float], delay_sec: float) -> None:
    """Wait no time; note in ``slept`` how long was asked for."""
    slept.append(delay_sec)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
