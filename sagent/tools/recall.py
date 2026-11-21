"""Recall: stored sessions like one session, told to the model as each is found.

A trackinizer server stores the sessions and embeds what each says. The tool
asks it, by plain HTTP, which other sessions' records lie nearest what one
stored session last said (``GET /api/web/search_sessions?near=``), and asks
again each ``interval_sec``. It writes each session it has not named before,
then wakes the caller's model (``Runtime.wake``), so the model hears of one
without polling. A call ends only when it is stopped or the server fails.

The session is the caller's own, or that of the agent a call names. The tool
is set to the id the process's own session is stored under, its first
agent's; each agent that session spawned is found under it by the id its
journal's folders spell. So a watcher names the agent it watches, and hears
of sessions like that agent's.

Typical usage example:

  tool = Recall(team=team, url="https://trax.example", session=str(session_id))
  streaming_result = tool.call(context, {})
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict

import asyncio
import os
import re

from treekle import json_loads
from wesearch.fetch import ContentParams, PolicyParams, RequestParams, fetch

from sagent.agent.team import Team
from sagent.session.projects import session_id, spawner_journal
from sagent.session.types import RecallResult
from sagent.tools.agent_tool import target_agent
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Arg,
    Stream,
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.agent.agent import Agent
    from sagent.tools.types import ToolContext


__all__ = [
    "Recall",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Recall:
    """Tell the model of stored sessions like one session, as each is found."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        agent: Annotated[
            str,
            "Agent label or relation whose session to search near; omit for yours.",
        ] = ""
        limit: Annotated[
            int,
            "How many records each search returns, nearest first.",
            Arg(minimum=1, maximum=50),
        ] = 5

    name = "Recall"
    description = (files(__package__) / "assets" / "recall.md").read_text(
        encoding="utf-8",
    )
    schema = json_schema(Arguments)

    team: Team = field(repr=False)
    """Where the caller's runtime is found, to wake its model."""

    url: str = ""
    """The trackinizer server; ``""`` reads ``$TRACKINIZER_URL``."""

    key: str = field(default="", repr=False)
    """The server's API key, for the ``url`` it is given with: a change of that
    setting drops it. ``""`` sends the environment's key to the environment's
    server (``$TRACKINIZER_TOKEN``, else what ``$TRACKINIZER_TOKEN_FILE``
    holds), and no key to a ``url`` given here."""

    session: str = ""
    """The id the process's own session is stored under, its first agent's;
    ``""`` reads ``$TRAX_SESSION``."""

    interval_sec: float = 60.0
    """How long a search waits after the last: the server embeds new records
    about as often."""

    sleep: Callable[[float], Awaitable[None]] = field(
        default=asyncio.sleep,
        repr=False,
    )

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Recall``."""
        del arguments
        return "Recall"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``None``: a search changes nothing."""
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[RecallResult, str, str]:
        """Search near the session until stopped, waking the caller for each new one.

        Args:
          context: The calling agent's state; that agent's model is woken.
          arguments: What the model passed, as ``Recall.Arguments``.

        Returns:
          streaming_result: Each session found, one line as it is found. Its
            result comes only if the server searched no vectors.

        Raises:
          ValueError: No server or no session is set, the environment's key
            cannot be read or sent, the caller is in no team, the agent named
            is not one agent or keeps no journal, or the limit is out of bounds.
          FetchError: The server failed or could not be reached.

        """
        stdout: Stream[str] = Stream()
        return whole_output(
            self._run(context, arguments, stdout),
            stdout=stdout,
            runs_until_stopped=True,
        )

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
        stdout: Stream[str],
    ) -> ToolOutput[RecallResult, str, str]:
        a = parse(arguments, Recall.Arguments)
        url = self.url or os.environ.get("TRACKINIZER_URL") or ""
        session = self.session or os.environ.get("TRAX_SESSION") or ""
        if not url or not session:
            raise ValueError(
                "Recall needs a server and a stored session: set tool.Recall.url and "
                "tool.Recall.session, or TRACKINIZER_URL and TRAX_SESSION.",
            )
        # The environment's key is for the environment's server: sent to a server
        # named here, it would hand that server another's key.
        key = self.key or ("" if self.url else _environment_key())
        near = target_agent(
            self.team,
            context=context,
            name=self.name,
            agent=a.agent,
            reading=True,
        )
        spawned = _spawned_id(near, label=a.agent or context.label)
        # The caller is in the team: ``target_agent`` refused one that is not.
        runtime = self.team[context.label].runtime
        request = RequestParams(
            content=ContentParams(
                params={
                    "near": session,
                    **({"spawned": spawned} if spawned else {}),
                    "mode": "vector",
                    "limit": a.limit,
                },
                headers={"Authorization": f"Bearer {key}"} if key else {},
            ),
            # The server is one its operator runs, often on a private network.
            policy=PolicyParams(allow_private_hosts=True),
        )
        calling = asyncio.current_task()
        named: set[str] = set()
        while True:
            body = await asyncio.to_thread(
                fetch,
                f"{url.removesuffix('/')}/api/web/search_sessions",
                request=request,
                session=context.fetch_session,
            )
            found = json_loads(body.decode(), _Found)
            if not found["semantic"]:
                return ToolOutput(
                    stderr=(
                        "The server searched no vectors: it embeds no sessions, or not yet.",
                    ),
                    result=RecallResult(exit_code=1),
                )
            known = len(named)
            for hit in found["hits"]:
                if hit["session_id"] not in named:
                    named.add(hit["session_id"])
                    stdout.write(
                        f"session {hit['session_id']} p{hit['part']}#{hit['idx']} "
                        f"({hit['title'] or 'untitled'}): "
                        f"{' '.join(hit['snippet'].split())}\n",
                    )
            # Written first: the runtime reads what the call wrote when it tells
            # the model.
            if len(named) > known:
                runtime.wake(calling)
            await self.sleep(self.interval_sec)


class _Hit(TypedDict):
    """One record the server found, as ``search_sessions`` answers."""

    session_id: str
    part: int
    idx: int
    field: str
    chunk: int
    score: float
    source: str
    snippet: str
    title: str


class _Found(TypedDict):
    """What ``search_sessions`` answers; ``semantic`` is whether vectors were searched."""

    hits: list[_Hit]
    semantic: bool
    degraded: bool


# The server stores each agent a session spawned as a session of its own, under the
# id ``session_id`` spells from its journal's folders, and finds it below its
# spawner's by that id.
def _spawned_id(agent: Agent, *, label: str) -> str:
    """Return the id ``agent``'s session is found by under the process's own; ``""`` for that one."""
    journal = agent.session.journal.path
    if journal is None:
        raise ValueError(f"{label} keeps no journal, so no session of it is stored.")
    return "" if spawner_journal(journal) is None else session_id(journal)


# The same variables, order and refusals as ``trax``, trackinizer's CLI, reads its key
# by (its ``connection._env_key``), which sagent does not import: one environment must
# name one key to both. A key is refused here with none of its text:
# sent, a newline or a non-ASCII character in it fails with an error that quotes it.
def _environment_key() -> str:
    """Return ``$TRACKINIZER_TOKEN``, else the key in ``$TRACKINIZER_TOKEN_FILE``."""
    key = os.environ.get("TRACKINIZER_TOKEN") or ""
    refusal = "TRACKINIZER_TOKEN must be printable ASCII with no whitespace"
    if not key and (path := os.environ.get("TRACKINIZER_TOKEN_FILE") or ""):
        try:
            held = Path(path).read_bytes()
        except OSError as error:
            raise ValueError(
                f"TRACKINIZER_TOKEN_FILE {path}: {error.strerror}",
            ) from None
        # What is no text reads as a character no key holds, so it is refused below
        # as any other is: read as text, the failure would quote its bytes.
        key = held.decode(errors="replace").strip()
        refusal = f"TRACKINIZER_TOKEN_FILE {path} must hold printable ASCII with no whitespace"
    if key and re.fullmatch(r"[!-~]+", key) is None:
        raise ValueError(refusal)
    return key
