"""Gemini through ``agy`` in headless stream-json mode, billed to the login it holds.

``agy`` is only a way to reach the model; it runs no tool of its own. Each process
runs a custom main agent whose instructions are our system prompt and which has
none of ``agy``'s tools, in a home of its own that names our MCP server as its one
tool server. Each turn is a ``user`` line on stdin; its words arrive as
``step_update`` lines, and a ``result`` line ends it. A tool call reaches our MCP
server, which holds the call, and ``stream`` returns it as a ``ToolCall``. The
next ``stream`` hands the caller's result to the held call and the same turn
continues. Anything else -- a new system prompt, new tools, another model or
effort, a rewritten history -- starts a fresh ``agy``, which reads the
conversation as text.

``agy`` keeps its own login, in the system keyring, and signs in with
``agy`` itself: run it once. It takes text alone, so an image is named, not sent.

Typical usage example:

  async with GoogleCLI() as google:
      flash = google["gemini-flash"]
      async for event in flash.stream(request):   # The same events as the API.
          ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import asyncio
import hashlib
import os
import re
import shutil
import tempfile

from treekle import ReadError, from_plain, json_dumps, json_loads, mutable

from sagent.model.cost import TokenCount
from sagent.model.exceptions import AuthError, DisconnectedError, RateLimitedError
from sagent.model.types import Response, TextChunk
from sagent.providers.google.api import models
from sagent.providers.resources import SharedResources
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
    SessionRecord,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request
    from sagent.providers.types import (
        ProviderSharedResources,
        ToolMount,
        UsageWindow,
    )
    from sagent.tools.types import Tool


__all__ = [
    "GoogleCLI",
    "GoogleCLIModel",
]


class GoogleCLI(AsyncResource):
    """Gemini models through ``agy``, billed to its login.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.

    """

    section: ClassVar[str] = "google-cli"
    """The catalog section whose models and settings this provider serves."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        # Stopped with the provider: a catalog closes providers before the pools
        # their processes run in.
        self._started: set[GoogleCLIModel] = set()

    @property
    def name(self) -> str:
        """Return the name the catalog knows this provider by."""
        return self._name

    @property
    def resources(self) -> ProviderSharedResources:
        """Return what this provider shares with others."""
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return ``()``: ``agy`` does not report the account's rate limits."""
        return ()

    def __getitem__(self, model: str) -> GoogleCLIModel:
        capability, settings = models(self.section).resolve(model)
        return GoogleCLIModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(models(self.section))

    @override
    async def aclose(self) -> None:
        for model in tuple(self._started):
            await model.aclose()
        if self._owns_resources:
            await self._resources.aclose()

    def started(self, model: GoogleCLIModel) -> None:
        """Remember ``model``, which started an ``agy`` this provider stops on closing."""
        self._started.add(model)


class GoogleCLIModel(AsyncResource):
    """One Gemini model, reached through one running ``agy`` at a time."""

    respawn_turns: ClassVar[int] = 100
    """Turns one ``agy`` serves before a fresh one replaces it."""

    respawn_fraction: ClassVar[float] = 0.5
    """Share of the window a conversation may fill before ``agy`` is replaced."""

    quiet_sec: ClassVar[float] = 600.0
    """How long ``agy`` may say nothing before the turn counts as dropped."""

    def __init__(
        self,
        *,
        provider: GoogleCLI,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._mount: ToolMount | None = None
        self._agy: _Agy | None = None

    @property
    def provider(self) -> GoogleCLI:
        """Return the provider serving this model."""
        return self._provider

    @property
    def capability(self) -> ModelCapability:
        """Return every setting this model offers."""
        return self._capability

    @property
    def settings(self) -> ModelSettings:
        """Return the settings chosen."""
        return self._settings

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        """Choose ``settings``; raise unless the capability offers each choice."""
        self._capability.check(settings)
        self._settings = settings

    @override
    async def aclose(self) -> None:
        await self._stop()
        mount, self._mount = self._mount, None
        if mount is not None:
            await self.provider.resources.tool_server.unmount(mount)

    async def count(self, request: Request) -> int:
        """Return an estimate of the tokens ``request`` would send; ``agy`` cannot count.

        Args:
          request: What the model would be asked.

        Returns:
          tokens: The text ``agy`` is given -- the prompt, the tools, and the
            conversation as a transcript -- over the model's characters per token.

        """
        text = "\n\n".join(
            (
                *request.system,
                *_described(request.tools),
                *_transcript(request.records),
            ),
        )
        return int(len(text) / self.capability.approx_chars_per_token)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return the tokens of the line naming ``attachment``: ``agy`` sends text alone."""
        return int(len(_named(attachment)) / self.capability.approx_chars_per_token)

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply as it is written, then the finished ``Response``.

        Args:
          request: What the model is asked.

        Yields:
          event: Each ``TextChunk`` as the model writes it, then one ``Response``.

        """
        records = tuple(request.records)
        mount = await self._mounted()
        mount.offer(request.tools)
        # A process serves one system prompt, one tool set, one model and effort.
        started_with = (*request.system, _digest(request.tools), self._model_id())
        turn = _Turn()
        try:
            agy = self._agy
            unseen = (
                agy.unseen(records, started_with=started_with)
                if agy is not None
                else None
            )
            # Read before answering, which empties it: answered calls continue
            # the turn in flight, and a new prompt would start another.
            continuing = agy is not None and bool(agy.held)
            if (
                agy is None
                or unseen is None
                or (not continuing and self._worn(agy))
                or (continuing and not await agy.answer(unseen, mount=mount))
            ):
                agy = await self._start(request, started_with=started_with)
            elif not continuing:
                await agy.prompt(_said_text(unseen))
            async for chunk in agy.read(turn, mount=mount):
                yield chunk
            response = turn.response(capability=self.capability)
        except BaseException:
            # The process is mid-turn, so its conversation no longer matches.
            await self._stop()
            raise
        agy.saw(records, response=response, context_tokens=await self.count(request))
        yield response

    def _model_id(self) -> str:
        """Return the id ``agy`` takes: the model's, suffixed by its effort."""
        model = self.capability.wire_model_id or self.capability.model_id
        return f"{model}-{self.settings.thinking_effort}"

    async def _mounted(self) -> ToolMount:
        if self._mount is None:
            self._mount = await self.provider.resources.tool_server.mount()
        return self._mount

    async def _start(
        self,
        request: Request,
        *,
        started_with: tuple[str, ...],
    ) -> _Agy:
        """Start a fresh ``agy`` in a home of its own, and tell it the conversation."""
        await self._stop()
        mount = await self._mounted()
        home = Path(tempfile.mkdtemp(prefix="sagent-agy-"))
        await asyncio.to_thread(
            _furnish,
            home,
            system=request.system,
            tools=request.tools,
            url=mount.url,
        )
        # Started in the workdir: ``agy`` reads rules and agents from where it
        # starts, and the caller's directory holds the caller's.
        argv = [
            "/bin/sh",
            "-c",
            'cd "$0" && exec "$@"',
            str(home / "workdir"),
            "agy",
            "--agent",
            "sagent",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--model",
            self._model_id(),
        ]
        # An API key would send ``agy`` to the Gemini API, not the plan.
        drop = {"GEMINI_API_KEY", "GOOGLE_API_KEY"}
        env = {key: value for key, value in os.environ.items() if key not in drop}
        env |= {"HOME": str(home), "USERPROFILE": str(home)}
        process = await self.provider.resources.process_pool("agy").acquire(
            argv,
            env=env,
        )
        self._agy = _Agy(
            process,
            home=home,
            started_with=started_with,
            quiet_sec=self.quiet_sec,
        )
        self.provider.started(self)
        await self._agy.prompt("\n\n".join(_transcript(request.records)))
        return self._agy

    async def _stop(self) -> None:
        agy, self._agy = self._agy, None
        if self._mount is not None:
            self._mount.cancel()
        if agy is not None:
            agy.close()
            await self.provider.resources.process_pool("agy").release(agy.process)
            await asyncio.to_thread(shutil.rmtree, agy.home, ignore_errors=True)

    # A long-lived ``agy`` accretes state of its own (caches, a context it
    # compresses at will); a fresh one between turns, told the conversation,
    # starts clean before its window fills.
    def _worn(self, agy: _Agy) -> bool:
        """Whether ``agy`` has served enough turns, or context, to retire."""
        window = self.capability.limits.max_request_tokens
        return (
            agy.seen.turns >= self.respawn_turns
            or agy.seen.context_tokens > window * self.respawn_fraction
        )


class _Agy:
    """One running ``agy``, its home, and the conversation it has seen."""

    def __init__(
        self,
        process: Process,
        *,
        home: Path,
        started_with: tuple[str, ...],
        quiet_sec: float,
    ) -> None:
        self.process = process
        self.home = home
        self._quiet_sec = quiet_sec
        self.started_with = started_with
        self._seen: _Seen | None = None
        self._line: asyncio.Task[bytes] | None = None
        self._call: asyncio.Task[ToolCall] | None = None
        # What the stream in flight has cost so far. A step reports its own call
        # before its stream ends; the compactor reads a reply's count as the
        # context's size, so a turn's calls summed would overstate it.
        self._spent = TokenCount()
        # Whether ``agy`` said it is calling our tools and the call has not yet
        # reached our server: only then is our server waited on beside its output.
        self._expecting = False

    @property
    def seen(self) -> _Seen:
        """Return what this process has heard and said, as of its last reply."""
        # A process that never replied is stopped, so none is asked this.
        assert isinstance(self._seen, _Seen)
        return self._seen

    @property
    def held(self) -> tuple[ToolCall, ...]:
        """Return the calls its last reply made, which wait for results."""
        return self.seen.held

    def unseen(
        self,
        records: Sequence[SessionRecord],
        *,
        started_with: tuple[str, ...],
    ) -> Sequence[SessionRecord] | None:
        """Return what ``records`` adds for this process; ``None`` if it cannot take it.

        It takes the results of the calls it holds, or, holding none, new
        user messages. Anything else needs a process that hears it all anew.

        Args:
          records: The whole conversation the next turn answers.
          started_with: What a process for that turn would be started with.

        Returns:
          unseen: The records this process has not seen, or ``None``.

        """
        if (
            started_with != self.started_with
            or tuple(records[: len(self.seen.records)]) != self.seen.records
        ):
            return None
        # Usage and settings records are ours alone; the model never sees them.
        unseen = [
            record
            for record in records[len(self.seen.records) :]
            if isinstance(
                record,
                (
                    UserMessage,
                    AgentToAgentMessage,
                    SystemMessage,
                    TurnContext,
                    AssistantMessage,
                    Thinking,
                    ToolCall,
                    ToolResult,
                    ContextClear,
                ),
            )
        ]
        if self.held:
            answered = {
                record.call_id for record in unseen if isinstance(record, ToolResult)
            }
            # A note that arrives with the results waits for the next user line.
            results = [
                r for r in unseen if not isinstance(r, (SystemMessage, TurnContext))
            ]
            exact = len(answered) == len(results) and answered == {
                c.call_id for c in self.held
            }
            return unseen if exact else None
        if unseen and all(
            isinstance(
                record,
                (UserMessage, AgentToAgentMessage, SystemMessage, TurnContext),
            )
            for record in unseen
        ):
            return unseen
        return None

    async def answer(
        self,
        results: Sequence[SessionRecord],
        *,
        mount: ToolMount,
    ) -> bool:
        """Hand each result to the call it answers.

        Args:
          results: The results of the calls this process holds.
          mount: Where those calls wait.

        Returns:
          answered: ``False`` when a call is no longer waiting.

        """
        calls = {call.call_id: call for call in self.held}
        try:
            for result in results:
                if isinstance(result, ToolResult):
                    await mount.answer(calls[result.call_id], result)
        except TimeoutError:
            return False
        return True

    async def prompt(self, text: str) -> None:
        """Start a turn.

        Args:
          text: What the user line says.

        """
        stdin = self.process.stdin
        # The pool starts every process with pipes.
        assert isinstance(stdin, asyncio.StreamWriter)
        line = {"event": "user", "message": {"content": text}}
        stdin.write(json_dumps(line).encode() + b"\n")
        await stdin.drain()

    async def read(self, turn: _Turn, *, mount: ToolMount) -> AsyncIterator[TextChunk]:
        """Collect ``turn`` until it ends or the model calls our tools.

        Args:
          turn: Collects the reply.
          mount: Where the model's calls of our tools arrive.

        Yields:
          chunk: Each piece of text, as the model writes it.

        Raises:
          DisconnectedError: ``agy`` exited, or went quiet too long.

        """
        while True:
            message = await self._next(mount)
            if isinstance(message, ToolCall):
                # It now waits on the held call, which the caller runs.
                turn.calls.append(message)
                turn.tokens, self._spent = self._spent, TokenCount()
                return
            event = message.get("event")
            if event == "result":
                result = _object(message.get("result"))
                if result.get("status") != "SUCCESS":
                    raise _failure(result)
                turn.tokens, self._spent = self._spent, TokenCount()
                return
            if event == "step_update":
                chunk = self._step(_object(message.get("step_update")), turn=turn)
                if chunk is not None:
                    yield chunk

    def saw(
        self,
        records: Sequence[SessionRecord],
        *,
        response: Response,
        context_tokens: int,
    ) -> None:
        """Record that this process heard ``records`` and replied ``response``.

        Args:
          records: The conversation it was asked to answer.
          response: Its reply, which may stop at calls.
          context_tokens: About how many tokens that conversation fills.

        """
        held = tuple(r for r in response.records if isinstance(r, ToolCall))
        turns = self._seen.turns if self._seen is not None else 0
        self._seen = _Seen(
            records=(*records, *response.records),
            held=held,
            # A turn that stops at calls goes on in the next stream.
            turns=turns if held else turns + 1,
            context_tokens=context_tokens,
        )

    def close(self) -> None:
        """Stop waiting on the process and on our server."""
        for task in (self._line, self._call):
            if task is not None:
                task.cancel()

    async def _next(self, mount: ToolMount) -> dict[str, object] | ToolCall:
        """Return the next line from ``agy``, or a call that reached our server."""
        # A call is taken only once ``agy`` has said it is making one, so every
        # word the model wrote before the call is read first.
        stdout = self.process.stdout
        assert isinstance(stdout, asyncio.StreamReader)
        # Kept across calls: a read cancelled midway could drop part of a line.
        if self._line is None:
            self._line = asyncio.ensure_future(stdout.readline())
        waiting: set[asyncio.Future[bytes] | asyncio.Future[ToolCall]] = {self._line}
        if self._expecting:
            if self._call is None:
                self._call = asyncio.ensure_future(mount.called())
            waiting.add(self._call)
        # A model that goes quiet this long has dropped the turn.
        done, _ = await asyncio.wait(
            waiting,
            timeout=self._quiet_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise DisconnectedError(f"agy went quiet for {self._quiet_sec:g} seconds")
        if self._line not in done and self._call is not None:
            call, self._call = self._call.result(), None
            self._expecting = False
            return call
        raw, self._line = self._line.result(), None
        if not raw:
            raise DisconnectedError("agy exited before its answer")
        # A line that is no JSON, as a warning, says nothing to us.
        try:
            return json_loads(raw, dict[str, object])
        except ValueError:
            return {}

    def _step(self, step: Mapping[str, object], *, turn: _Turn) -> TextChunk | None:
        """Take one ``step_update``; return the text it adds, if any."""
        usage = _object(step.get("usage"))
        if usage:
            self._spent += TokenCount(
                request=from_plain(usage.get("input_tokens"), int, default=0),
                response=from_plain(usage.get("output_tokens"), int, default=0),
                cache_read=from_plain(usage.get("cache_read_tokens"), int, default=0),
            )
        kind = step.get("step_type")
        # Only our tools are offered: a call of one is a call our server will hold.
        if kind == "tool" and step.get("state") == "ACTIVE":
            self._expecting = True
        text = str(step.get("text_delta") or "")
        if kind != "agent_response" or not text:
            return None
        turn.text.append(text)
        return TextChunk(text=text)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Seen:
    """What one ``agy`` has heard and said, as of its last reply."""

    records: tuple[SessionRecord, ...]
    held: tuple[ToolCall, ...]
    """The calls the last reply made, which wait for results."""

    turns: int
    """Turns it finished."""

    context_tokens: int
    """About how many tokens the conversation it holds fills."""


@dataclass(slots=True, kw_only=True)
class _Turn:
    """One stream's share of a turn: its words, and its tokens once it ends."""

    text: list[str] = field(default_factory=list[str])
    calls: list[ToolCall] = field(default_factory=list[ToolCall])
    tokens: TokenCount = field(default_factory=TokenCount)
    """This stream's steps', held at a call or ending the turn."""

    def response(self, *, capability: ModelCapability) -> Response:
        """Return this stream's reply, costed.

        Args:
          capability: The model replying, whose prices cost the reply.

        Returns:
          response: The text and calls, and their tokens and cost.

        """
        records: list[AssistantMessage | ToolCall] = []
        if self.text:
            records.append(AssistantMessage(content="".join(self.text)))
        records.extend(self.calls)
        return Response(
            records=tuple(records),
            tokens=self.tokens,
            spend=capability.prices[
                "auto",
                self.tokens.prompt,
                datetime.now(UTC).date(),
            ]
            * self.tokens,
            stop_reason="tool_use" if self.calls else "end_turn",
        )


def _furnish(
    home: Path,
    *,
    system: Sequence[str],
    tools: Sequence[Tool],
    url: str,
) -> None:
    """Lay out a home for ``agy``: our agent, our tool server, and its settings."""
    (home / "workdir").mkdir(mode=0o700)
    config = home / ".gemini" / "config"
    agent = config / "agents" / "sagent"
    agent.mkdir(parents=True, mode=0o700)
    described = _described(tools)
    preface = (
        "# Tools\n\nEach tool is on the MCP server `sagent`; call it through"
        " call_mcp_tool, with arguments as its input schema says. A relative path"
        " resolves against sagent's working directory, not the workspace agy names."
    )
    instructions = "\n\n".join(
        part for part in (*system, preface if tools else "", *described) if part
    )
    # ``tools: []`` gives the agent none of ``agy``'s own; its body is the prompt.
    (agent / "agent.md").write_text(
        "---\nname: sagent\ndescription: Answers as sagent asks, with sagent's tools.\n"
        f"mainAgent: true\ntools: []\n---\n{instructions}\n",
    )
    (config / "mcp_config.json").write_text(
        json_dumps({"mcpServers": {"sagent": {"serverUrl": url}}}),
    )
    settings = home / ".gemini" / "antigravity-cli"
    settings.mkdir(mode=0o700)
    # Our tools ask for no approval; the caller already decided.
    (settings / "settings.json").write_text(
        json_dumps(
            {
                "permissions": {"allow": ["mcp(sagent/*)"]},
                "showTips": False,
                "showFeedbackSurvey": False,
            },
        ),
    )


# ``agy`` names each MCP tool to the model and keeps its schema in a file,
# so without this the model guesses arguments or spends a call reading it.
def _described(tools: Sequence[Tool]) -> list[str]:
    """Return each tool's section of the agent's instructions."""
    return [
        f"## {tool.name}\n\n{tool.description}\n\n"
        "Input schema: "
        + json_dumps(mutable(tool.schema), separators=(",", ":"), ensure_ascii=False)
        for tool in tools
    ]


def _said_text(records: Sequence[SessionRecord]) -> str:
    """Return the prompt that says the new user messages to a running ``agy``."""
    return "\n\n".join(line for record in records if (line := _said(record)))


def _transcript(records: Sequence[SessionRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        match record:
            case ContextClear(summary=str() as summary, history=history):
                lines.append(f"Summary of the conversation so far:\n{summary}")
                lines.extend(_transcript(history))
            case ContextClear(history=history):
                lines.extend(_transcript(history))
            case UserMessage(content=str() as text) if text:
                lines.append(f"User: {_with_attachments(text, record.attachments)}")
            case AgentToAgentMessage(content=str() as text, sender=str() as sender) if (
                text
            ):
                lines.append(f"Agent {sender}: {text}")
            case AgentToAgentMessage(content=str() as text) if text:
                lines.append(f"Agent: {text}")
            case SystemMessage() | TurnContext():
                if note := _said(record):
                    lines.append(note)
            case AssistantMessage(content=str() as text) if text:
                lines.append(f"Assistant: {text}")
            case ToolCall(call_id=call_id, name=name, arguments=arguments):
                # The id pairs the call with the result line that names it.
                lines.append(
                    f"Assistant called {name} as {call_id}: {json_dumps(mutable(arguments))}",
                )
            case ToolResult():
                lines.append(f"{record.call_id} returned: {record.text()}")
            case _:
                continue
    return lines


def _said(record: SessionRecord) -> str:
    """Return what a new prompt carries for ``record``: words, or a reminder."""
    if isinstance(record, UserMessage):
        return _with_attachments(record.content or "", record.attachments)
    if isinstance(record, AgentToAgentMessage):
        body = record.content or ""
        return f"[from {record.sender}]: {body}" if record.sender is not None else body
    if isinstance(record, SystemMessage) and record.content is not None:
        return f"<system-reminder>\n{record.content}\n</system-reminder>"
    if isinstance(record, TurnContext) and record.text():
        return f"<system-reminder>\n{record.text()}\n</system-reminder>"
    return ""


def _with_attachments(text: str, attachments: Sequence[Attachment]) -> str:
    """Return ``text``, each attachment named after it: ``agy`` takes text alone."""
    named = [_named(a) for a in attachments]
    return "\n".join([text, *named]) if named else text


def _named(attachment: Attachment) -> str:
    """Return the line that stands for ``attachment`` in what ``agy`` is sent."""
    return f"[an {attachment.mime_descriptor} attachment, which agy cannot send]"


def _digest(tools: Sequence[Tool]) -> str:
    offered = [(tool.name, tool.description, mutable(tool.schema)) for tool in tools]
    return hashlib.sha256(json_dumps(offered).encode()).hexdigest()


def _failure(result: Mapping[str, object]) -> Exception:
    """Return the exception a failed ``result`` from ``agy`` means."""
    error = str(result.get("error") or result.get("status") or "agy failed")
    lowered = error.casefold()
    if "authenticat" in lowered or "sign in" in lowered:
        return AuthError(f"{error}; run agy once to sign in")
    if "resource_exhausted" in lowered or "quota" in lowered or "429" in lowered:
        return RateLimitedError(error, retry_after_sec=_reset_sec(lowered))
    return RuntimeError(f"agy: {error}")


# The server's words, which agy passes on: "Resets in 2h39m52s." from agy 1.2
# (antigravity-cli#1018), "Your quota will reset after 155h19m17s." before it.
def _reset_sec(error: str) -> float:
    """Return the wait a spent quota's message names; ``0`` when it names none."""
    found = re.search(
        r"reset(?:s in| after) ((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)",
        error,
    )
    if found is None or not found[1]:
        return 0.0
    parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", found[1])
    hours, minutes, seconds = (
        (float(p or 0) for p in parts.groups()) if parts else (0, 0, 0)
    )
    return min(hours * 3_600 + minutes * 60 + seconds, 691_200.0)


def _object(value: object) -> dict[str, object]:
    try:
        return from_plain(value, dict[str, object])
    except ReadError:
        return {}
