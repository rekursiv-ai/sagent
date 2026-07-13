from collections.abc import Callable

from slack_sdk.socket_mode.request import SocketModeRequest

class AsyncWebSocketMessageListener(Callable):
    async def __call__(
        self: AsyncBaseSocketModeClient, message: dict, raw_message: str | None = ...
    ): ...

class AsyncSocketModeRequestListener(Callable):
    async def __call__(self: AsyncBaseSocketModeClient, request: SocketModeRequest): ...
