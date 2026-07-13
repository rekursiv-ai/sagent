from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from logging import Logger
from queue import Queue
from threading import Lock

from slack_sdk.socket_mode.client import BaseSocketModeClient
from slack_sdk.socket_mode.listeners import (
    SocketModeRequestListener,
    WebSocketMessageListener,
)
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.web import WebClient

from .connection import Connection, ConnectionState
from ..interval_runner import IntervalRunner

"""The built-in Socket Mode client

* https://docs.slack.dev/apis/events-api/using-socket-mode/
* https://docs.slack.dev/tools/python-slack-sdk/socket-mode/

"""

class SocketModeClient(BaseSocketModeClient):
    logger: Logger
    web_client: WebClient
    app_token: str
    wss_uri: str | None
    message_queue: Queue
    message_listeners: list[
        WebSocketMessageListener
        | Callable[[BaseSocketModeClient, dict, str | None], None]
    ]
    socket_mode_request_listeners: list[
        SocketModeRequestListener
        | Callable[[BaseSocketModeClient, SocketModeRequest], None]
    ]
    current_session: Connection | None
    current_session_state: ConnectionState
    current_session_runner: IntervalRunner
    current_app_monitor: IntervalRunner
    current_app_monitor_started: bool
    message_processor: IntervalRunner
    message_workers: ThreadPoolExecutor
    auto_reconnect_enabled: bool
    default_auto_reconnect_enabled: bool
    trace_enabled: bool
    receive_buffer_size: int
    connect_operation_lock: Lock
    on_message_listeners: list[Callable[[str], None]]
    on_error_listeners: list[Callable[[Exception], None]]
    on_close_listeners: list[Callable[[int, str | None], None]]
    def __init__(
        self,
        app_token: str,
        logger: Logger | None = ...,
        web_client: WebClient | None = ...,
        auto_reconnect_enabled: bool = ...,
        trace_enabled: bool = ...,
        all_message_trace_enabled: bool = ...,
        ping_pong_trace_enabled: bool = ...,
        ping_interval: float = ...,
        receive_buffer_size: int = ...,
        concurrency: int = ...,
        proxy: str | None = ...,
        proxy_headers: dict[str, str] | None = ...,
        on_message_listeners: list[Callable[[str], None]] | None = ...,
        on_error_listeners: list[Callable[[Exception], None]] | None = ...,
        on_close_listeners: list[Callable[[int, str | None], None]] | None = ...,
    ) -> None: ...
    def session_id(self) -> str | None: ...
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def send_message(self, message: str) -> None: ...
    def close(self) -> None: ...
