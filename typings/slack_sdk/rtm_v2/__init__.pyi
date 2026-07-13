from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from logging import Logger
from queue import Queue
from ssl import SSLContext
from threading import Lock

import logging

from slack_sdk.socket_mode.builtin.connection import Connection, ConnectionState
from slack_sdk.socket_mode.interval_runner import IntervalRunner
from slack_sdk.web import WebClient

"""A Python module for interacting with Slack's RTM API."""

class RTMClient:
    token: str | None
    bot_id: str | None
    default_auto_reconnect_enabled: bool
    auto_reconnect_enabled: bool
    ssl: SSLContext | None
    proxy: str | None
    timeout: int
    base_url: str
    ping_interval: int
    logger: Logger
    web_client: WebClient
    current_session: Connection | None
    current_session_state: ConnectionState | None
    wss_uri: str | None
    message_queue: Queue
    message_listeners: list[Callable[[RTMClient, dict], None]]
    message_processor: IntervalRunner
    message_workers: ThreadPoolExecutor
    closed: bool
    connect_operation_lock: Lock
    on_message_listeners: list[Callable[[str], None]]
    on_error_listeners: list[Callable[[Exception], None]]
    on_close_listeners: list[Callable[[int, str | None], None]]
    def __init__(
        self,
        *,
        token: str | None = ...,
        web_client: WebClient | None = ...,
        auto_reconnect_enabled: bool = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        timeout: int = ...,
        base_url: str = ...,
        headers: dict | None = ...,
        ping_interval: int = ...,
        concurrency: int = ...,
        logger: logging.Logger | None = ...,
        on_message_listeners: list[Callable[[str], None]] | None = ...,
        on_error_listeners: list[Callable[[Exception], None]] | None = ...,
        on_close_listeners: list[Callable[[int, str | None], None]] | None = ...,
        trace_enabled: bool = ...,
        all_message_trace_enabled: bool = ...,
        ping_pong_trace_enabled: bool = ...,
    ) -> None: ...
    def on(self, event_type: str) -> Callable: ...
    def is_connected(self) -> bool: ...
    def issue_new_wss_url(self) -> str: ...
    def connect_to_new_endpoint(self, force: bool = ...) -> None: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def close(self) -> None: ...
    def start(self) -> None: ...
    def send(self, payload: dict | str) -> None: ...
    def enqueue_message(self, message: str) -> None: ...
    def process_message(self) -> None: ...
    def process_messages(self) -> None: ...
    def run_message_listeners(self, message: dict) -> None: ...
    def session_id(self) -> str | None: ...
    def run_all_message_listeners(self, message: str) -> None: ...
    def run_all_error_listeners(self, error: Exception) -> None: ...
    def run_all_close_listeners(self, code: int, reason: str | None = ...) -> None: ...
