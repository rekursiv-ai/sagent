from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from logging import Logger
from queue import Queue
from threading import Lock
from typing import Any

from slack_sdk.socket_mode.interval_runner import IntervalRunner
from slack_sdk.socket_mode.listeners import (
    SocketModeRequestListener,
    WebSocketMessageListener,
)
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web import WebClient

class BaseSocketModeClient:
    logger: Logger
    web_client: WebClient
    app_token: str
    wss_uri: str
    message_queue: Queue
    message_listeners: list[
        WebSocketMessageListener
        | Callable[[BaseSocketModeClient, dict, str | None], None]
    ]
    socket_mode_request_listeners: list[
        SocketModeRequestListener
        | Callable[[BaseSocketModeClient, SocketModeRequest], None]
    ]
    message_processor: IntervalRunner
    message_workers: ThreadPoolExecutor
    closed: bool
    connect_operation_lock: Lock
    def issue_new_wss_url(self) -> str: ...
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def connect_to_new_endpoint(self, force: bool = ...) -> None: ...
    def close(self) -> None: ...
    def send_message(self, message: str) -> None: ...
    def send_socket_mode_response(
        self, response: dict[str, Any] | SocketModeResponse
    ) -> None: ...
    def enqueue_message(self, message: str) -> None: ...
    def process_message(self) -> None: ...
    def run_message_listeners(self, message: dict, raw_message: str) -> None: ...
    def process_messages(self) -> None: ...
