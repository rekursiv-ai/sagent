from asyncio import Lock, Queue
from collections.abc import Awaitable, Callable
from logging import Logger
from typing import Any

from slack_sdk.socket_mode.async_listeners import (
    AsyncSocketModeRequestListener,
    AsyncWebSocketMessageListener,
)
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

class AsyncBaseSocketModeClient:
    logger: Logger
    web_client: AsyncWebClient
    app_token: str
    wss_uri: str
    auto_reconnect_enabled: bool
    trace_enabled: bool
    closed: bool
    connect_operation_lock: Lock
    message_queue: Queue
    message_listeners: list[
        AsyncWebSocketMessageListener
        | Callable[[AsyncBaseSocketModeClient, dict, str | None], Awaitable[None]]
    ]
    socket_mode_request_listeners: list[
        AsyncSocketModeRequestListener
        | Callable[[AsyncBaseSocketModeClient, SocketModeRequest], Awaitable[None]]
    ]
    async def issue_new_wss_url(self) -> str: ...
    async def is_connected(self) -> bool: ...
    async def session_id(self) -> str: ...
    async def connect(self): ...
    async def disconnect(self): ...
    async def connect_to_new_endpoint(self, force: bool = ...) -> None: ...
    async def close(self) -> None: ...
    async def send_message(self, message: str): ...
    async def send_socket_mode_response(
        self, response: dict[str, Any] | SocketModeResponse
    ) -> None: ...
    async def enqueue_message(self, message: str) -> None: ...
    async def process_messages(self) -> None: ...
    async def process_message(self) -> None: ...
    async def run_message_listeners(self, message: dict, raw_message: str) -> None: ...
