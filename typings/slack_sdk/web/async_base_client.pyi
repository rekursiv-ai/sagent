from ssl import SSLContext

import logging

from aiohttp import FormData
from slack_sdk.http_retry.async_handler import AsyncRetryHandler

import aiohttp

from .async_slack_response import AsyncSlackResponse

class AsyncBaseClient:
    BASE_URL = ...
    def __init__(
        self,
        token: str | None = ...,
        base_url: str = ...,
        timeout: int = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        session: aiohttp.ClientSession | None = ...,
        trust_env_in_session: bool = ...,
        headers: dict | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        team_id: str | None = ...,
        logger: logging.Logger | None = ...,
        retry_handlers: list[AsyncRetryHandler] | None = ...,
    ) -> None: ...
    @property
    def logger(self) -> logging.Logger: ...
    async def api_call(
        self,
        api_method: str,
        *,
        http_verb: str = ...,
        files: dict | None = ...,
        data: dict | FormData | None = ...,
        params: dict | None = ...,
        json: dict | None = ...,
        headers: dict | None = ...,
        auth: dict | None = ...,
    ) -> AsyncSlackResponse: ...
