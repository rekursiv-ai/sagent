"""Agent Client Protocol (ACP) models, billed to the underlying agent harnesses.

Typical usage example:

  async with Acp() as acp:
      agy = acp["agy"]
      async for event in agy.stream(request):
          if isinstance(event, TextChunk):
              print(event.text, end="")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast, override

import contextlib

from sagent.acp.agy import AgyAcp
from sagent.acp.client import (
    AcpClient,
    AcpError,
    AcpSession,
    image_content,
    text_content,
)
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import (
    Response,
    TextChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from sagent.providers import acp
from sagent.providers.resources import SharedResources
from sagent.providers.table import HasModels
from sagent.session.types import (
    AnyToolResult,
    AssistantMessage,
    Attachment,
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    SessionRecord,
    ShellCommandResult,
    Thinking,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from types import ModuleType

    from treekle import PlainTree

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request
    from sagent.providers.types import ProviderSharedResources, UsageWindow


__all__ = [
    "Acp",
    "AcpModel",
]


class Acp(HasModels, AsyncResource):
    """Models reached through Agent Client Protocol (ACP) servers.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; None makes and closes its own.
      cwd: Directory the ACP subprocess runs in; current working directory by default.

    """

    section: ClassVar[str] = "acp"
    aliases: ClassVar[tuple[str, ...]] = ("acp-agent", "agent-client")
    package: ClassVar[ModuleType] = acp

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self._cwd = cwd or Path.cwd()
        self._started: set[AcpModel] = set()

    @property
    def name(self) -> str:
        """Return the name the catalog knows this provider by."""
        return self._name

    @property
    def cwd(self) -> Path:
        """Return the working directory for ACP agent processes."""
        return self._cwd

    @property
    def resources(self) -> ProviderSharedResources:
        """Return what this provider shares with others."""
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return rate limit windows (ACP does not report them directly)."""
        return ()

    @property
    def acp_servers(self) -> tuple[str, ...]:
        """Return supported ACP agent servers."""
        return ("agy",)

    @property
    def current_acp_server(self) -> str:
        """Return the active ACP agent server."""
        return "agy"

    def __getitem__(self, model: str) -> AcpModel:
        capability, settings = self.models().resolve(model)
        return AcpModel(
            provider=self,
            capability=capability,
            settings=settings,
            cwd=self._cwd,
            acp_server="agy",
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.models())

    @override
    async def aclose(self) -> None:
        for model in tuple(self._started):
            await model.aclose()
        if self._owns_resources:
            await self._resources.aclose()

    def started(self, model: AcpModel) -> None:
        """Track active models to close them when provider closes."""
        self._started.add(model)


class AcpModel(AsyncResource):
    """One ACP agent model, reached through an ACP subprocess."""

    def __init__(
        self,
        *,
        provider: Acp,
        capability: ModelCapability,
        settings: ModelSettings,
        cwd: Path,
        acp_server: str = "agy",
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._cwd = cwd
        self._acp_server = acp_server
        self._client: AcpClient | None = None
        self._session: AcpSession | None = None

    @property
    def acp_server(self) -> str:
        """Return the ACP agent server name."""
        return self._acp_server

    @property
    def provider(self) -> Acp:
        """Return the provider serving this model."""
        return self._provider

    @property
    def capability(self) -> ModelCapability:
        """Return the capability of this model."""
        return self._capability

    @property
    def settings(self) -> ModelSettings:
        """Return the chosen settings."""
        return self._settings

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        self._capability.check(settings)
        self._settings = settings

    @override
    async def aclose(self) -> None:
        client, self._client = self._client, None
        self._session = None
        if client is not None:
            await client.aclose()

    async def count(self, request: Request) -> int:
        """Return estimated tokens for request."""
        text = "\n\n".join(
            r.content or "" for r in request.records if isinstance(r, UserMessage)
        )
        chars_per_tok = self.capability.approx_chars_per_token
        return int(len(text) / chars_per_tok) if chars_per_tok > 0 else len(text)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return estimated tokens for an attachment."""
        if attachment.mime_descriptor.startswith("image/"):
            return 258
        chars_per_tok = self.capability.approx_chars_per_token
        return (
            int(len(attachment.data) / chars_per_tok)
            if chars_per_tok > 0
            else len(attachment.data)
        )

    async def stream(
        self, request: Request
    ) -> AsyncIterator[TextChunk | ToolCallChunk | ToolResultChunk | Response]:
        """Yield the reply chunks as streamed, then the finished Response."""
        self.provider.started(self)
        client = await self._ensure_client()
        session = await self._ensure_session(client)

        prompt_blocks = _extract_prompt_blocks(request.records)
        if not prompt_blocks:
            prompt_blocks = [text_content("continue")]

        full_text: list[str] = []
        thinking_text: list[str] = []
        active_calls: dict[str, ToolCall] = {}
        completed_call_ids: set[str] = set()

        async for update in client.prompt_stream(session, prompt_blocks):
            update_type = update.get("sessionUpdate")
            if update_type == "agent_thought_chunk":
                text = _extract_chunk_text(update.get("content"))
                if text:
                    thinking_text.append(text)
                    yield TextChunk(text=text, thinking=True)
            elif update_type == "agent_message_chunk":
                text = _extract_chunk_text(update.get("content"))
                if text:
                    full_text.append(text)
                    yield TextChunk(text=text, thinking=False)
            elif update_type == "tool_call":
                call_id = str(update.get("toolCallId") or "")
                if call_id:
                    name = str(update.get("name") or "tool")
                    args = _extract_tool_args(update)
                    title = str(update.get("title") or "")
                    call = ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=args,
                        extra={"title": title} if title else {},
                    )
                    active_calls[call_id] = call
                    yield ToolCallChunk(call=call, title=title)
                    status = str(update.get("status") or "")
                    if status in ("completed", "failed"):
                        tool_result = _build_tool_result(update, call)
                        completed_call_ids.add(call_id)
                        yield ToolResultChunk(result=tool_result)
            elif update_type == "tool_call_update":
                call_id = str(update.get("toolCallId") or "")
                status = str(update.get("status") or "")
                if (
                    call_id
                    and status in ("completed", "failed")
                    and call_id not in completed_call_ids
                ):
                    fallback_call = active_calls.get(call_id)
                    tool_result = _build_tool_result(update, fallback_call)
                    completed_call_ids.add(call_id)
                    yield ToolResultChunk(result=tool_result)

        for call_id in active_calls:
            if call_id not in completed_call_ids:
                yield ToolResultChunk(
                    result=UncategorizedToolResult(
                        call_id=call_id,
                        content="Tool completed",
                        is_error=False,
                    )
                )

        records: list[AssistantMessage | Thinking] = []
        if thinking_text:
            records.append(Thinking(content="".join(thinking_text)))
        records.append(AssistantMessage(content="".join(full_text)))

        response = Response(
            records=tuple(records),
            tokens=TokenCount(),
            spend=TokenCost(),
            stop_reason="end_turn",
        )
        yield response

    async def _ensure_client(self) -> AcpClient:
        if self._client is not None:
            return self._client
        # Currently defaults to AgyAcp for agy models
        agy = AgyAcp(
            dangerously_skip_permissions=True,
        )
        client = agy.client(cwd=self._cwd)
        await client.start()
        self._client = client
        return client

    async def _ensure_session(self, client: AcpClient) -> AcpSession:
        if self._session is not None:
            return self._session
        session = await client.new_session(cwd=self._cwd)
        wire_model = self.capability.wire_model_id or self.capability.model_id
        if wire_model and wire_model != "agy":
            with contextlib.suppress(AcpError):
                await client.set_model(session, wire_model)
        self._session = session
        return session


def _extract_prompt_blocks(
    records: Sequence[SessionRecord],
) -> list[dict[str, object]]:
    """Extract prompt text and image attachments from the latest user message."""
    user_record: UserMessage | None = None
    for rec in reversed(records):
        if isinstance(rec, UserMessage):
            user_record = rec
            break
    if user_record is None:
        return []

    blocks = [
        image_content(att.data, mime_type=att.mime_descriptor)
        for att in user_record.attachments
        if att.mime_descriptor.startswith("image/")
    ]

    text = user_record.content or ""
    if text:
        blocks.append(text_content(text))
    elif not blocks:
        blocks.append(text_content(""))
    return blocks


def _extract_chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        val = cast(Mapping[str, object], content).get("text")
        if isinstance(val, str):
            return val
    return ""


def _extract_tool_args(update: Mapping[str, object]) -> Mapping[str, PlainTree]:
    raw = update.get("args") or update.get("rawInput")
    if isinstance(raw, Mapping):
        return cast("Mapping[str, PlainTree]", raw)
    return {}


def _extract_tool_path(
    update: Mapping[str, object],
    fallback_call: ToolCall | None,
) -> str | None:
    locations = update.get("locations")
    if isinstance(locations, Sequence) and len(locations) > 0:
        first_loc = locations[0]
        if isinstance(first_loc, Mapping) and "path" in first_loc:
            path_val = first_loc["path"]
            if isinstance(path_val, str) and path_val:
                return path_val

    raw = update.get("args") or update.get("rawInput")
    if not isinstance(raw, Mapping) and fallback_call is not None:
        raw = fallback_call.arguments
    if isinstance(raw, Mapping):
        for key in ("Path", "path", "file_path", "filePath"):
            val = raw.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _extract_tool_output(update: Mapping[str, object]) -> str:
    raw_output = update.get("rawOutput")
    if isinstance(raw_output, Mapping):
        output_val = raw_output.get("output")
        if isinstance(output_val, str):
            return output_val
        stdout_val = raw_output.get("stdout")
        stderr_val = raw_output.get("stderr")
        if isinstance(stdout_val, str) or isinstance(stderr_val, str):
            return f"{stdout_val or ''}{stderr_val or ''}"
        result_val = raw_output.get("result")
        if isinstance(result_val, str):
            return result_val
        text_val = raw_output.get("text")
        if isinstance(text_val, str):
            return text_val
    elif isinstance(raw_output, str):
        return raw_output

    content = update.get("content")
    if isinstance(content, Sequence):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                inner = block.get("content")
                if isinstance(inner, Mapping):
                    text_val = inner.get("text")
                    if isinstance(text_val, str):
                        chunks.append(text_val)
                else:
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        chunks.append(block_text)
        if chunks:
            return "".join(chunks)
    elif isinstance(content, str):
        return content

    return ""


def _build_tool_result(
    update: Mapping[str, object],
    fallback_call: ToolCall | None,
) -> AnyToolResult:
    call_id = str(
        update.get("toolCallId") or (fallback_call.call_id if fallback_call else "")
    )
    name = str(update.get("name") or (fallback_call.name if fallback_call else "tool"))
    status = str(update.get("status") or "completed")
    is_error = status == "failed"
    output_text = _extract_tool_output(update)
    path = _extract_tool_path(update, fallback_call)

    if name in ("run_command", "bash", "shell"):
        raw_output = update.get("rawOutput")
        exit_code: int | None = None
        if isinstance(raw_output, Mapping):
            raw_code = raw_output.get("exit_code")
            if isinstance(raw_code, int):
                exit_code = raw_code
        if exit_code is None:
            exit_code = 1 if is_error else 0
        raw_args = update.get("args") or update.get("rawInput")
        if not isinstance(raw_args, Mapping) and fallback_call is not None:
            raw_args = fallback_call.arguments
        cmd = ()
        if isinstance(raw_args, Mapping):
            cmd_val = raw_args.get("CommandLine") or raw_args.get("command")
            if isinstance(cmd_val, str):
                cmd = (cmd_val,)
        return ShellCommandResult(
            call_id=call_id,
            command=cmd,
            stdout=output_text,
            exit_code=exit_code,
            is_error=is_error,
        )

    if name in ("read_file", "show_definition"):
        return FileReadResult(
            call_id=call_id,
            path=path,
            content=output_text,
            is_error=is_error,
        )

    if name == "write_file":
        return FileWriteResult(
            call_id=call_id,
            path=path,
            content=output_text,
            is_error=is_error,
        )

    if name == "edit_file":
        return FileEditResult(
            call_id=call_id,
            path=path,
            is_error=is_error,
        )

    return UncategorizedToolResult(
        call_id=call_id,
        content=output_text,
        is_error=is_error,
    )
