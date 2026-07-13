from asyncio import AbstractEventLoop, Future, Lock, Queue
from collections.abc import Awaitable, Callable
from logging import Logger

from aiohttp import ClientWebSocketResponse, WSMessage
from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
from slack_sdk.socket_mode.async_listeners import (
    AsyncSocketModeRequestListener,
    AsyncWebSocketMessageListener,
)
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.web.async_client import AsyncWebClient

"""aiohttp based Socket Mode client

* https://docs.slack.dev/apis/events-api/using-socket-mode/
* https://docs.slack.dev/tools/python-slack-sdk/socket-mode/
* https://pypi.org/project/aiohttp/

"""

class SocketModeClient(AsyncBaseSocketModeClient):
    logger: Logger
    web_client: AsyncWebClient
    app_token: str
    wss_uri: str | None
    auto_reconnect_enabled: bool
    message_queue: Queue
    message_listeners: list[
        AsyncWebSocketMessageListener
        | Callable[[AsyncBaseSocketModeClient, dict, str | None], Awaitable[None]]
    ]
    socket_mode_request_listeners: list[
        AsyncSocketModeRequestListener
        | Callable[[AsyncBaseSocketModeClient, SocketModeRequest], Awaitable[None]]
    ]
    message_receiver: Future | None
    message_processor: Future
    proxy: str | None
    ping_interval: float
    trace_enabled: bool
    last_ping_pong_time: float | None
    current_session: ClientWebSocketResponse | None
    current_session_monitor: Future | None
    default_auto_reconnect_enabled: bool
    closed: bool
    stale: bool
    connect_operation_lock: Lock
    on_message_listeners: list[Callable[[WSMessage], Awaitable[None]]]
    on_error_listeners: list[Callable[[WSMessage], Awaitable[None]]]
    on_close_listeners: list[Callable[[WSMessage], Awaitable[None]]]
    def __init__(
        self,
        app_token: str,
        logger: Logger | None = ...,
        web_client: AsyncWebClient | None = ...,
        proxy: str | None = ...,
        auto_reconnect_enabled: bool = ...,
        ping_interval: float = ...,
        trace_enabled: bool = ...,
        on_message_listeners: list[Callable[[WSMessage], Awaitable[None]]] | None = ...,
        on_error_listeners: list[Callable[[WSMessage], Awaitable[None]]] | None = ...,
        on_close_listeners: list[Callable[[WSMessage], Awaitable[None]]] | None = ...,
        loop: AbstractEventLoop | None = ...,
    ) -> None: ...
    async def monitor_current_session(self) -> None: ...
    async def receive_messages(self) -> None: ...
    async def is_ping_pong_failing(self) -> bool: ...
    async def is_connected(self) -> bool: ...
    async def session_id(self) -> str: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def send_message(self, message: str) -> None: ...
    async def close(self) -> None: ...
    @classmethod
    def build_session_id(cls, session: ClientWebSocketResponse) -> str: ...
