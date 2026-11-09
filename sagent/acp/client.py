"""A small, dependency-free ACP v1 client.

The ACP wire format is JSON-RPC 2.0 over the agent subprocess's stdin/stdout.
This module deliberately owns only transport, capability negotiation, session
lifecycle, prompt content, and update dispatch. Frontends decide how to render
updates and how to answer permission or filesystem requests.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import asyncio
import base64
import contextlib
import inspect
import json


if TYPE_CHECKING:
    from pathlib import Path


type JsonObject = dict[str, Any]
type AcpUpdate = Mapping[str, Any]
type UpdateHandler = Callable[[AcpUpdate], Awaitable[None] | None]
type RequestHandler = Callable[[str, JsonObject], Awaitable[object] | object]
type PermissionHandler = Callable[
    [JsonObject], Awaitable[JsonObject | PermissionDecision] | JsonObject | PermissionDecision
]


class AcpError(RuntimeError):
    """An ACP transport, protocol, or agent error."""


@dataclass(frozen=True, slots=True)
class AcpAgentCapabilities:
    """Capabilities advertised by an ACP agent during initialization."""

    raw: Mapping[str, Any]

    @property
    def load_session(self) -> bool:
        """Whether the agent supports resuming existing sessions."""
        return self.raw.get("loadSession") is True or self.raw.get("load_session") is True

    @property
    def image_prompt(self) -> bool:
        """Whether the agent accepts image blocks in prompts."""
        return self.raw.get("promptCapabilities", {}).get("image") is True

    @property
    def audio_prompt(self) -> bool:
        """Whether the agent accepts audio blocks in prompts."""
        return self.raw.get("promptCapabilities", {}).get("audio") is True

    @property
    def embedded_context(self) -> bool:
        """Whether the agent accepts embedded context blocks."""
        return self.raw.get("promptCapabilities", {}).get("embeddedContext") is True

    @property
    def session_capabilities(self) -> Mapping[str, Any]:
        """Session-level capabilities advertised by the agent."""
        caps: object = self.raw.get("sessionCapabilities") or self.raw.get("session_capabilities")
        if isinstance(caps, Mapping):
            return cast(Mapping[str, Any], caps)
        return {}


@dataclass(frozen=True, slots=True)
class AcpSession:
    """An ACP session identifier and its negotiated capabilities."""

    session_id: str
    capabilities: AcpAgentCapabilities
    available_models: tuple[Mapping[str, Any], ...] = ()
    current_model_id: str = ""
    config_options: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AcpPromptResult:
    """The terminal response returned by ``session/prompt``."""

    stop_reason: str
    raw: Mapping[str, Any]
    updates: tuple[AcpUpdate, ...] = ()


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """A client decision answering a ``session/request_permission`` request."""

    outcome: str
    option_id: str | None = None

    def to_json(self) -> JsonObject:
        """Return the JSON-RPC result payload for ACP."""
        if self.outcome == "selected":
            if self.option_id is None:
                msg = "selected outcome requires an option_id"
                raise ValueError(msg)
            return {"outcome": {"outcome": "selected", "optionId": self.option_id}}
        return {"outcome": {"outcome": "cancelled"}}

    @classmethod
    def allow(cls, option_id: str) -> Self:
        """Allow the tool call with a specific option."""
        return cls(outcome="selected", option_id=option_id)

    @classmethod
    def cancel(cls) -> Self:
        """Cancel or decline the tool call."""
        return cls(outcome="cancelled")

    @classmethod
    def auto_allow(cls, params: Mapping[str, Any]) -> Self:
        """Select the first allow option or cancel if none match."""
        options: object = params.get("options")
        if isinstance(options, Sequence):
            for opt in options:
                if isinstance(opt, Mapping):
                    opt_map = cast(Mapping[str, object], opt)
                    kind = str(opt_map.get("kind", ""))
                    opt_id = str(opt_map.get("optionId", ""))
                    if "allow" in kind or "allow" in opt_id:
                        return cls.allow(opt_id)
            if options:
                first = options[0]
                if isinstance(first, Mapping) and "optionId" in first:
                    first_map = cast(Mapping[str, object], first)
                    return cls.allow(str(first_map["optionId"]))
        return cls.cancel()


def _parse_json_message(raw: bytes) -> dict[str, Any]:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as error:
        msg = f"invalid ACP JSON: {error}"
        raise AcpError(msg) from error
    if not isinstance(parsed, dict):
        msg = "ACP message must be an object"
        raise AcpError(msg)
    return cast(dict[str, Any], parsed)


def _parse_session_models_and_config(
    result: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], str, tuple[Mapping[str, Any], ...]]:
    models_data = result.get("models")
    available_models: list[Mapping[str, Any]] = []
    current_model_id = ""
    if isinstance(models_data, Mapping):
        raw_avail = models_data.get("availableModels")
        if isinstance(raw_avail, Sequence):
            available_models.extend(
                cast(Mapping[str, Any], m) for m in raw_avail if isinstance(m, Mapping)
            )
        current_model_id = str(models_data.get("currentModelId") or "")
    raw_config = result.get("configOptions")
    config_options: list[Mapping[str, Any]] = []
    if isinstance(raw_config, Sequence):
        config_options.extend(
            cast(Mapping[str, Any], c) for c in raw_config if isinstance(c, Mapping)
        )
    return tuple(available_models), current_model_id, tuple(config_options)


class AcpClient:
    """Connect to one ACP agent subprocess.

    The client follows ACP v1's required lifecycle:
    ``initialize`` → ``session/new`` or ``session/load`` → ``session/prompt``.
    Server notifications and requests are handled concurrently with ordinary
    JSON-RPC responses, which is required for ACP's streaming updates.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        client_name: str = "sagent",
        client_version: str = "0.0.0",
        on_update: UpdateHandler | None = None,
        on_request: RequestHandler | None = None,
        on_permission: PermissionHandler | None = None,
    ) -> None:
        if not command:
            msg = "ACP command must not be empty"
            raise ValueError(msg)
        self.command = tuple(command)
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.client_name = client_name
        self.client_version = client_version
        self.on_update = on_update
        self.on_request = on_request
        self.on_permission = on_permission
        self.process: asyncio.subprocess.Process | None = None
        self.agent_capabilities: AcpAgentCapabilities | None = None
        self.agent_info: Mapping[str, Any] | None = None
        self._next_id = 0
        self._pending: dict[int | str, asyncio.Future[JsonObject]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._stderr_buffer: deque[str] = deque(maxlen=40)
        self._closed = False

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def start(self) -> Mapping[str, Any]:
        """Start the agent and complete ACP initialization."""
        if self.process is not None:
            msg = "ACP client is already started"
            raise RuntimeError(msg)
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(self.cwd),
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader = asyncio.create_task(self._read_stdout())
        self._stderr = asyncio.create_task(self._drain_stderr())
        result = await self.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        version = result.get("protocolVersion")
        if version != 1:
            msg = f"unsupported ACP protocol version: {version!r}"
            raise AcpError(msg)
        agent_caps: object = result.get("agentCapabilities")
        caps_mapping: Mapping[str, Any] = (
            cast(Mapping[str, Any], agent_caps) if isinstance(agent_caps, Mapping) else {}
        )
        self.agent_capabilities = AcpAgentCapabilities(caps_mapping)
        agent_info: object = result.get("agentInfo")
        self.agent_info = (
            cast(Mapping[str, Any], agent_info) if isinstance(agent_info, Mapping) else None
        )
        return result

    async def new_session(
        self,
        *,
        cwd: Path | None = None,
        mcp_servers: Sequence[Mapping[str, Any]] = (),
    ) -> AcpSession:
        """Create an ACP session after initialization."""
        result = await self.request(
            "session/new",
            {
                "cwd": str(cwd or self.cwd),
                "mcpServers": list(mcp_servers),
            },
        )
        session_id = result.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            msg = "session/new returned no sessionId"
            raise AcpError(msg)
        available_models, current_model_id, config_options = _parse_session_models_and_config(result)
        return AcpSession(
            session_id=session_id,
            capabilities=self.agent_capabilities or AcpAgentCapabilities({}),
            available_models=available_models,
            current_model_id=current_model_id,
            config_options=config_options,
        )

    async def load_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        mcp_servers: Sequence[Mapping[str, Any]] = (),
    ) -> AcpSession:
        """Resume an existing ACP session by ID."""
        result = await self.request(
            "session/load",
            {
                "sessionId": session_id,
                "cwd": str(cwd or self.cwd),
                "mcpServers": list(mcp_servers),
            },
        )
        returned_id = result.get("sessionId", session_id)
        if not isinstance(returned_id, str) or not returned_id:
            returned_id = session_id
        available_models, current_model_id, config_options = _parse_session_models_and_config(result)
        return AcpSession(
            session_id=returned_id,
            capabilities=self.agent_capabilities or AcpAgentCapabilities({}),
            available_models=available_models,
            current_model_id=current_model_id,
            config_options=config_options,
        )

    async def set_model(self, session: AcpSession, model_id: str) -> JsonObject:
        """Select the active model for an established ACP session."""
        return await self.request(
            "session/set_model",
            {
                "sessionId": session.session_id,
                "modelId": model_id,
            },
        )

    async def prompt(
        self,
        session: AcpSession,
        content: Sequence[Mapping[str, Any]] | str,
    ) -> AcpPromptResult:
        """Send one prompt and collect its streamed updates."""
        updates: list[AcpUpdate] = []
        previous = self.on_update

        async def collect(update: AcpUpdate) -> None:
            updates.append(dict(update))
            if previous is not None:
                result = previous(update)
                if inspect.isawaitable(result):
                    await result

        self.on_update = collect
        try:
            result = await self.request(
                "session/prompt",
                {
                    "sessionId": session.session_id,
                    "prompt": (
                        [{"type": "text", "text": content}]
                        if isinstance(content, str)
                        else list(content)
                    ),
                },
            )
        finally:
            self.on_update = previous
        return AcpPromptResult(
            stop_reason=str(result.get("stopReason", "unknown")),
            raw=result,
            updates=tuple(updates),
        )

    async def prompt_stream(
        self,
        session: AcpSession,
        content: Sequence[Mapping[str, Any]] | str,
    ) -> AsyncIterator[AcpUpdate]:
        """Stream updates from one prompt as they arrive."""
        queue: asyncio.Queue[AcpUpdate | BaseException | None] = asyncio.Queue()
        previous = self.on_update

        async def collect(update: AcpUpdate) -> None:
            queue.put_nowait(dict(update))
            if previous is not None:
                res = previous(update)
                if inspect.isawaitable(res):
                    await res

        self.on_update = collect

        async def run_prompt() -> None:
            try:
                await self.request(
                    "session/prompt",
                    {
                        "sessionId": session.session_id,
                        "prompt": (
                            [{"type": "text", "text": content}]
                            if isinstance(content, str)
                            else list(content)
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(run_prompt())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.on_update = previous
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def cancel(self, session: AcpSession) -> None:
        """Cancel the active prompt for ``session``."""
        await self.notify("session/cancel", {"sessionId": session.session_id})

    async def close_session(self, session: AcpSession) -> None:
        """Close a session when the agent advertises that capability."""
        capabilities = self.agent_capabilities or AcpAgentCapabilities({})
        if capabilities.session_capabilities.get("close") is not None:
            await self.request("session/close", {"sessionId": session.session_id})

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        """Send a JSON-RPC request and await its response."""
        process = self.process
        if process is None or process.stdin is None:
            msg = "ACP client is not started"
            raise AcpError(msg)
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: JsonObject) -> None:
        """Send a JSON-RPC notification."""
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def aclose(self) -> None:
        """Close the ACP session and subprocess without leaving orphan tasks."""
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
                with contextlib.suppress(ProcessLookupError):
                    await process.stdin.wait_closed()
            if process.returncode is None:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=2)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        for task in (self._reader, self._stderr):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        err_msg = "ACP client closed"
        if process is not None and process.returncode not in (None, 0):
            stderr_tail = "".join(self._stderr_buffer).strip()
            if stderr_tail:
                err_msg = f"ACP agent exited with code {process.returncode}: {stderr_tail}"
        error = AcpError(err_msg)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        self.process = None

    async def _write(self, message: JsonObject) -> None:
        process = self.process
        if process is None or process.stdin is None:
            msg = "ACP client is not started"
            raise AcpError(msg)
        process.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
        await process.stdin.drain()

    async def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            async for raw in process.stdout:
                typed_msg = _parse_json_message(raw)
                if "id" in typed_msg and ("result" in typed_msg or "error" in typed_msg):
                    self._resolve_response(typed_msg)
                    continue
                method = typed_msg.get("method")
                if isinstance(method, str):
                    await self._handle_request_or_notification(method, typed_msg)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- fail all blocked requests.
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
        finally:
            if self._pending:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                stderr_tail = "".join(self._stderr_buffer).strip()
                code = process.returncode
                if code not in (None, 0):
                    err_msg = (
                        f"ACP agent process exited with code {code}: {stderr_tail}"
                        if stderr_tail
                        else f"ACP agent process exited with code {code}"
                    )
                else:
                    err_msg = "ACP agent closed stdout unexpectedly"
                error = AcpError(err_msg)
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(error)

    def _resolve_response(self, message: Mapping[str, Any]) -> None:
        request_id: object = message.get("id")
        if not isinstance(request_id, (int, str)):
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        if "error" in message:
            error: object = message["error"]
            detail: object = (
                cast(Mapping[str, object], error).get("message", "unknown ACP error")
                if isinstance(error, Mapping)
                else error
            )
            future.set_exception(AcpError(str(detail)))
        else:
            result: object = message.get("result", {})
            result_dict: JsonObject = (
                cast(JsonObject, result) if isinstance(result, dict) else {"value": result}
            )
            future.set_result(result_dict)

    async def _handle_request_or_notification(
        self,
        method: str,
        message: Mapping[str, Any],
    ) -> None:
        params_raw: object = message.get("params")
        params: JsonObject = (
            cast(JsonObject, params_raw) if isinstance(params_raw, dict) else {}
        )
        if method == "session/update":
            if self.on_update is not None:
                update_body: object = params.get("update", params)
                update_mapping: AcpUpdate = (
                    cast(AcpUpdate, update_body)
                    if isinstance(update_body, Mapping)
                    else {}
                )
                result = self.on_update(update_mapping)
                if inspect.isawaitable(result):
                    await result
            return
        request_id: object = message.get("id")
        if not isinstance(request_id, (int, str)):
            return
        try:
            result = await self._handle_server_request(method, params)
            await self._write({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as error:  # noqa: BLE001 -- JSON-RPC error response.
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(error)},
                }
            )

    async def _handle_server_request(self, method: str, params: JsonObject) -> object:
        if method == "session/request_permission":
            if self.on_permission is not None:
                decision = self.on_permission(params)
                if inspect.isawaitable(decision):
                    decision = await decision
                if isinstance(decision, PermissionDecision):
                    return decision.to_json()
                return decision
            return PermissionDecision.cancel().to_json()
        if self.on_request is not None:
            res = self.on_request(method, params)
            if inspect.isawaitable(res):
                return await res
            return res
        msg = f"unsupported ACP server request: {method}"
        raise AcpError(msg)

    async def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        async for line in process.stderr:
            self._stderr_buffer.append(line.decode("utf-8", errors="replace"))


def text_content(text: str) -> JsonObject:
    """Return an ACP text prompt block."""
    return {"type": "text", "text": text}


def image_content(data: bytes, *, mime_type: str, uri: str | None = None) -> JsonObject:
    """Return an ACP image prompt block."""
    block: JsonObject = {
        "type": "image",
        "mimeType": mime_type,
        "data": base64.b64encode(data).decode("ascii"),
    }
    if uri is not None:
        block["uri"] = uri
    return block


__all__ = [
    "AcpAgentCapabilities",
    "AcpClient",
    "AcpError",
    "AcpPromptResult",
    "AcpSession",
    "AcpUpdate",
    "PermissionDecision",
    "PermissionHandler",
    "RequestHandler",
    "UpdateHandler",
    "image_content",
    "text_content",
]
