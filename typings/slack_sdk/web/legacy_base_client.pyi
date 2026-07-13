from ssl import SSLContext

import asyncio
import logging

from aiohttp import FormData

import aiohttp

from .legacy_slack_response import LegacySlackResponse as SlackResponse

"""A Python module for interacting with Slack's Web API."""

class LegacyBaseClient:
    BASE_URL = ...
    def __init__(
        self,
        token: str | None = ...,
        base_url: str = ...,
        timeout: int = ...,
        loop: asyncio.AbstractEventLoop | None = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        run_async: bool = ...,
        use_sync_aiohttp: bool = ...,
        session: aiohttp.ClientSession | None = ...,
        headers: dict | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        team_id: str | None = ...,
        logger: logging.Logger | None = ...,
    ) -> None: ...
    def api_call(
        self,
        api_method: str,
        *,
        http_verb: str = ...,
        files: dict | None = ...,
        data: dict | FormData = ...,
        params: dict | None = ...,
        json: dict | None = ...,
        headers: dict | None = ...,
        auth: dict | None = ...,
    ) -> asyncio.Future | SlackResponse: ...
    @staticmethod
    def validate_slack_signature(
        *, signing_secret: str, data: str, timestamp: str, signature: str
    ) -> bool: ...
