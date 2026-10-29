"""Claude through the ``claude`` CLI, billed to whatever login it holds.

The CLI is only a way to reach the model; it runs no tool of its own. A tool
call reaches it through our MCP server, which holds the call, and ``stream``
returns it as a ``ToolCall``. The next ``stream`` hands the caller's results to
the held calls and the model continues. Anything else -- a new system prompt,
new tools, a rewritten history -- starts a fresh ``claude``, which reads the
conversation as text.

Typical usage example:

  async with AnthropicCLI() as anthropic:
      opus = anthropic["opus"]
      async for event in opus.stream(request):   # The same events as the API.
          ...

  work = AnthropicCLI(credentials=Path("~/.claude/.credentials-work.json"))
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import asyncio
import hashlib
import os
import shutil
import tempfile

from treekle import from_plain, json_dumps, json_loads, mutable

from sagent.model import image
from sagent.model.exceptions import ContextOverflowError, DisconnectedError
from sagent.providers.anthropic import wire
from sagent.providers.anthropic.api import models
from sagent.providers.anthropic.login import ClaudeLogin
from sagent.providers.resources import SharedResources
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    ContextClear,
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
    from collections.abc import AsyncIterator, Iterator, Sequence

    from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.types import (
        ProviderSharedResources,
        ToolMount,
        UsageWindow,
    )
    from sagent.session.types import SessionRecord
    from sagent.tools.types import Tool


__all__ = [
    "AnthropicCLI",
    "AnthropicCLIModel",
]


class AnthropicCLI(AsyncResource):
    """Claude models through the ``claude`` CLI, billed to its login.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      credentials: A login file; ``None`` uses the login ``claude`` keeps.

    """

    section: ClassVar[str] = "anthropic-cli"
    """The catalog section whose models and settings this provider serves."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        credentials: Path | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self._credentials = credentials
        self._login = ClaudeLogin(credentials, resources=self._resources)

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
        """Return ``()``: the CLI does not report the account's rate limits."""
        return ()

    def __getitem__(self, model: str) -> AnthropicCLIModel:
        capability, settings = models(self.section).resolve(model)
        return AnthropicCLIModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(models(self.section))

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    async def login(self, *, headless: bool = False) -> None:
        """Sign in through the browser, or with a code pasted from it."""
        await self._login.sign_in(headless=headless)

    async def environment(self) -> dict[str, str]:
        """Return the environment a ``claude`` process starts in.

        Returns:
          env: This process's environment, without any variable that points
            the CLI at another account, and with its telemetry turned off.

        """
        # Any of these points the CLI at another account or endpoint than the
        # login it holds.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))
            and key != "AWS_BEARER_TOKEN_BEDROCK"
        }
        env |= {
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_AUTO_COMPACT": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            # A held call waits for the caller's tool, which may run for hours.
            "MCP_TOOL_TIMEOUT": str(24 * 3600 * 1000),
        }
        # ``claude`` keeps one login of its own; a named one arrives as a token.
        if self._credentials is not None:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = await self._login.token()
        return env


class AnthropicCLIModel(AsyncResource):
    """One Claude model, reached through one running ``claude`` at a time."""

    def __init__(
        self,
        *,
        provider: AnthropicCLI,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._mount: ToolMount | None = None
        self._claude: _Claude | None = None
        self._files: Path | None = None

    @property
    def provider(self) -> AnthropicCLI:
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
        files, self._files = self._files, None
        if files is not None:
            await asyncio.to_thread(shutil.rmtree, files)

    async def count(self, request: Request) -> int:
        """Return an estimate of the tokens ``request`` would send; the CLI cannot count.

        Args:
          request: What the model would be asked.

        Returns:
          tokens: The API request's length, its attachments left out, over the
            model's characters per token, plus each attachment's tokens.

        """
        attachments = [
            attachment
            for record in request.records
            if isinstance(record, (UserMessage, AgentToAgentMessage, ToolResult))
            for attachment in record.attachments
        ]
        bare = [
            replace(record, attachments=())
            if isinstance(record, (UserMessage, AgentToAgentMessage, ToolResult))
            else record
            for record in request.records
        ]
        system = wire.system_blocks(request.system, settings=self.settings)
        body, _ = wire.encode(
            replace(request, records=bare),
            capability=self.capability,
            settings=self.settings,
            system=system,
        )
        return int(
            len(json_dumps(body)) / self.capability.approx_chars_per_token,
        ) + sum(self.attachment_tokens(attachment) for attachment in attachments)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return about how many tokens ``attachment`` adds, by Anthropic's formula."""
        return wire.attachment_tokens(attachment, limits=self.capability.limits)

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
        env = await self.provider.environment()
        # A process serves one system prompt, one tool set, one login.
        started_with = (*request.system, _digest(request.tools), json_dumps(env))
        try:
            claude = self._claude
            unseen = (
                claude.unseen(records, started_with=started_with)
                if claude is not None
                else None
            )
            # Read before ``answer`` clears it: answered calls continue the
            # turn claude is in, and a user line written now would start another.
            held = claude.held if claude is not None else ()
            if (
                claude is None
                or unseen is None
                or (
                    held
                    and not await claude.answer(
                        unseen,
                        mount=mount,
                        limits=self.capability.limits,
                    )
                )
            ):
                claude = await self._start(request, started_with=started_with, env=env)
            elif not held:
                await claude.say(
                    _content(
                        (*claude.waiting, *unseen),
                        replay=False,
                        limits=self.capability.limits,
                    ),
                )
                claude.waiting = ()
            reader = wire.StreamReader(
                capability=self.capability,
                settings=self.settings,
            )
            async for chunk in claude.read(reader):
                yield chunk
            response = reader.response()
        except BaseException:
            # The process is mid-turn, so its conversation no longer matches.
            await self._stop()
            raise
        # Our tools reach the model under the MCP server's name.
        response = replace(
            response,
            records=tuple(
                replace(record, name=record.name.removeprefix("mcp__sagent__"))
                if isinstance(record, ToolCall)
                else record
                for record in response.records
            ),
        )
        claude.saw(records, response=response)
        yield response

    async def _mounted(self) -> ToolMount:
        if self._mount is None:
            self._mount = await self.provider.resources.tool_server.mount()
        return self._mount

    async def _start(
        self,
        request: Request,
        *,
        started_with: tuple[str, ...],
        env: dict[str, str],
    ) -> _Claude:
        """Start a fresh ``claude`` and tell it the conversation so far."""
        await self._stop()
        mount = await self._mounted()
        # Any local user can read a process's arguments, so what is private or
        # long reaches claude in files of our own: the mount's URL is the only
        # key to its tools, and a long system prompt would overflow argv.
        argv = [
            "claude",
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--model",
            self.capability.wire_model_id or self.capability.model_id,
            "--system-prompt-file",
            str(self._written("system", "\n\n".join(request.system))),
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            str(
                self._written(
                    "mcp",
                    json_dumps(
                        {"mcpServers": {"sagent": {"type": "http", "url": mount.url}}},
                    ),
                ),
            ),
            # Its own tools off; ours arrive over MCP.
            "--tools",
            "",
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
        ]
        process = await self.provider.resources.process_pool("claude").acquire(
            argv,
            env=env,
        )
        self._claude = _Claude(process, started_with=started_with)
        await self._claude.say(
            _content(request.records, replay=True, limits=self.capability.limits),
        )
        return self._claude

    async def _stop(self) -> None:
        claude, self._claude = self._claude, None
        if self._mount is not None:
            self._mount.cancel()
        if claude is not None:
            await self.provider.resources.process_pool("claude").release(claude.process)

    # Named by content, the next ``claude`` started with the same text has the same
    # command line, and so takes the spare started ahead for it.
    def _written(self, kind: str, text: str) -> Path:
        """Return a file only we can read holding ``text``, named by its content."""
        if self._files is None:
            self._files = Path(tempfile.mkdtemp(prefix="sagent-claude-"))
        path = self._files / f"{kind}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        if not path.exists():
            path.write_text(text)
        return path


class _Claude:
    """One running ``claude``, and the conversation it has seen."""

    def __init__(self, process: Process, *, started_with: tuple[str, ...]) -> None:
        self.process = process
        self.started_with = started_with
        self.seen: tuple[SessionRecord, ...] = ()
        self.held: tuple[ToolCall, ...] = ()
        self.waiting: tuple[SessionRecord, ...] = ()
        """Notes that came with tool results, for the next user line to carry."""

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
            or tuple(records[: len(self.seen)]) != self.seen
        ):
            return None
        # Usage and settings records are ours alone; the model never sees them.
        unseen = [
            record
            for record in records[len(self.seen) :]
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
        limits: ModelLimits,
    ) -> bool:
        """Hand each result to the call it answers.

        Args:
          results: The results of the calls this process holds.
          mount: Where those calls wait.
          limits: The model's image limits, which each image is shrunk to.

        Returns:
          answered: ``False`` when a call never came, as for a tool it lacks.

        """
        calls = {call.call_id: call for call in self.held}
        try:
            for result in results:
                if isinstance(result, ToolResult):
                    await mount.answer(
                        calls[result.call_id],
                        _shrunk(result, limits=limits),
                    )
        except TimeoutError:
            return False  # A call the CLI never made, as for a tool it lacks.
        self.held = ()
        self.waiting += tuple(
            r for r in results if isinstance(r, (SystemMessage, TurnContext))
        )
        return True

    async def say(self, content: list[dict[str, object]]) -> None:
        stdin = self.process.stdin
        if stdin is None:
            raise ValueError("claude was started without stdin")
        line = {"type": "user", "message": {"role": "user", "content": content}}
        stdin.write(json_dumps(line).encode() + b"\n")
        await stdin.drain()

    async def read(self, reader: wire.StreamReader) -> AsyncIterator[TextChunk]:
        """Feed ``reader`` until the turn ends or the model calls our tools.

        Args:
          reader: Collects the reply.

        Yields:
          chunk: Each piece of text or reasoning, as the model writes it.

        Raises:
          DisconnectedError: ``claude`` exited, or went quiet for ten minutes.

        """
        stdout = self.process.stdout
        if stdout is None:
            raise ValueError("claude was started without stdout")
        # Which window stopped the turn, and when it resets, as the API's headers say.
        spent: dict[str, str] = {}
        while True:
            # A model that goes quiet this long has dropped the turn.
            raw = await asyncio.wait_for(stdout.readline(), timeout=600.0)
            if not raw:
                raise DisconnectedError("claude exited before its answer")
            event = json_loads(raw, dict[str, object])
            if event.get("type") == "rate_limit_event":
                spent = _spent(event)
                continue
            if event.get("type") == "result":
                if event.get("is_error") is True:
                    raise _failure(event, headers=spent)
                return
            if event.get("type") != "stream_event":
                continue
            inner = from_plain(event.get("event"), dict[str, object], default={})
            chunk = reader.feed(inner)
            if chunk is not None:
                yield chunk
            # It now waits on the held calls, which the caller runs.
            if (
                inner.get("type") == "message_delta"
                and reader.stop_reason == "tool_use"
            ):
                return

    def saw(self, records: Sequence[SessionRecord], *, response: Response) -> None:
        """Record that this process has seen ``records`` and given ``response``."""
        self.seen = (*records, *response.records)
        self.held = tuple(
            record for record in response.records if isinstance(record, ToolCall)
        )


def _content(
    records: Sequence[SessionRecord],
    *,
    replay: bool,
    limits: ModelLimits,
) -> list[dict[str, object]]:
    """Return one user line: the new user messages, or the whole conversation as text."""
    attachments = [
        attachment
        for record in records
        if isinstance(record, UserMessage)
        for attachment in record.attachments
    ]
    if replay:
        text = "\n\n".join(_transcript(records))
    else:
        text = "\n\n".join(line for record in records if (line := _said(record)))
    return wire.content(text, attachments, limits=limits)


def _shrunk[R: ToolResult](result: R, *, limits: ModelLimits) -> R:
    """Return ``result`` with each image shrunk to ``limits``, as the API path does."""
    return replace(
        result,
        attachments=tuple(
            _shrunk_attachment(a, limits=limits) for a in result.attachments
        ),
    )


def _shrunk_attachment(attachment: Attachment, *, limits: ModelLimits) -> Attachment:
    if not attachment.mime_descriptor.startswith("image/"):
        return attachment
    data, mime = image.shrink(
        attachment.data,
        max_edge_px=limits.max_image_edge_px,
        max_bytes=limits.max_image_bytes,
    )
    return Attachment(mime_descriptor=mime, data=data)


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
                lines.append(f"User: {text}")
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
                lines.append(
                    f"Assistant called {name} as {call_id}: {json_dumps(mutable(arguments))}",
                )
            case ToolResult():
                lines.append(f"{record.call_id} returned: {record.text()}")
            case _:
                pass
    return lines


def _digest(tools: Sequence[Tool]) -> str:
    offered = [
        {
            "name": tool.name,
            "description": tool.description,
            "schema": mutable(tool.schema),
        }
        for tool in tools
    ]
    return hashlib.sha256(json_dumps(offered).encode()).hexdigest()


def _spent(event: dict[str, object]) -> dict[str, str]:
    """Return a ``rate_limit_event`` that stopped the turn as the API's headers say it."""
    info = from_plain(event.get("rate_limit_info"), dict[str, object], default={})
    resets_at = info.get("resetsAt")
    if info.get("status") != "rejected" or not isinstance(resets_at, (int, float)):
        return {}
    return {
        "anthropic-ratelimit-unified-reset": str(resets_at),
        "anthropic-ratelimit-unified-representative-claim": str(
            info.get("rateLimitType") or "",
        ),
    }


def _failure(event: dict[str, object], *, headers: dict[str, str]) -> Exception:
    """Return the exception a failed ``result`` event means."""
    text = str(event.get("result") or event.get("errors") or event)
    reason = event.get("terminal_reason")
    if reason == "blocking_limit":
        return ContextOverflowError(text)
    if reason == "aborted_streaming":
        return DisconnectedError(text)
    status = event.get("api_error_status")
    return (
        wire.to_exception(status, body=text, headers=headers)
        if isinstance(status, int)
        else RuntimeError(text)
    )


def _said(record: SessionRecord) -> str:
    """Return what a new user line carries for ``record``: words, or a reminder."""
    if isinstance(record, UserMessage):
        return record.content or ""
    if isinstance(record, AgentToAgentMessage):
        body = record.content or ""
        return f"[from {record.sender}]: {body}" if record.sender is not None else body
    if isinstance(record, SystemMessage) and record.content is not None:
        return f"<system-reminder>\n{record.content}\n</system-reminder>"
    if isinstance(record, TurnContext) and record.text():
        return f"<system-reminder>\n{record.text()}\n</system-reminder>"
    return ""
