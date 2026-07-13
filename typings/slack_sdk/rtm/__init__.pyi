from collections.abc import Callable
from ssl import SSLContext
from typing import Any

import asyncio
import collections

"""A Python module for interacting with Slack's RTM API."""

class RTMClient:
    _callbacks: collections.defaultdict = ...
    def __init__(
        self,
        *,
        token: str,
        run_async: bool | None = ...,
        auto_reconnect: bool | None = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        timeout: int | None = ...,
        base_url: str | None = ...,
        connect_method: str | None = ...,
        ping_interval: int | None = ...,
        loop: asyncio.AbstractEventLoop | None = ...,
        headers: dict | None = ...,
    ) -> None: ...
    @staticmethod
    def run_on(*, event: str) -> Callable[..., Any]: ...
    @classmethod
    def on(cls, *, event: str, callback: Callable) -> None: ...
    def start(self) -> asyncio.Future | Any: ...
    def stop(self) -> None: ...
    async def async_stop(self) -> None: ...
    def send_over_websocket(self, *, payload: dict) -> Task[None]: ...
    async def ping(self) -> None: ...
    async def typing(self, *, channel: str) -> None: ...
