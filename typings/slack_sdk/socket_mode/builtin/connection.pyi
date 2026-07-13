from collections.abc import Callable
from logging import Logger

import ssl

class ConnectionState:
    terminated: bool
    def __init__(self) -> None: ...

class Connection:
    url: str
    logger: Logger
    proxy: str | None
    proxy_headers: dict[str, str] | None
    trace_enabled: bool
    ping_pong_trace_enabled: bool
    last_ping_pong_time: float | None
    session_id: str
    sock: ssl.SSLSocket | None
    on_message_listener: Callable[[str], None] | None
    on_error_listener: Callable[[Exception], None] | None
    on_close_listener: Callable[[int, str | None], None] | None
    def __init__(
        self,
        url: str,
        logger: Logger,
        proxy: str | None = ...,
        proxy_headers: dict[str, str] | None = ...,
        ping_interval: float = ...,
        receive_timeout: float = ...,
        receive_buffer_size: int = ...,
        trace_enabled: bool = ...,
        all_message_trace_enabled: bool = ...,
        ping_pong_trace_enabled: bool = ...,
        on_message_listener: Callable[[str], None] | None = ...,
        on_error_listener: Callable[[Exception], None] | None = ...,
        on_close_listener: Callable[[int, str | None], None] | None = ...,
        connection_type_name: str = ...,
        ssl_context: ssl.SSLContext | None = ...,
    ) -> None: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_active(self) -> bool: ...
    def close(self) -> None: ...
    def ping(self, payload: str | bytes = ...) -> None: ...
    def pong(self, payload: str | bytes = ...) -> None: ...
    def send(self, payload: str) -> None: ...
    def check_state(self) -> None: ...
    def run_until_completion(self, state: ConnectionState) -> None: ...
