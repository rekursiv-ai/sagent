from ssl import SSLContext

import logging

from slack_sdk.http_retry.handler import RetryHandler

from .slack_response import SlackResponse

"""A Python module for interacting with Slack's Web API."""

class BaseClient:
    BASE_URL = ...
    def __init__(
        self,
        token: str | None = ...,
        base_url: str = ...,
        timeout: int = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        headers: dict | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        team_id: str | None = ...,
        logger: logging.Logger | None = ...,
        retry_handlers: list[RetryHandler] | None = ...,
    ) -> None: ...
    @property
    def logger(self) -> logging.Logger: ...
    def api_call(
        self,
        api_method: str,
        *,
        http_verb: str = ...,
        files: dict | None = ...,
        data: dict | None = ...,
        params: dict | None = ...,
        json: dict | None = ...,
        headers: dict | None = ...,
        auth: dict | None = ...,
    ) -> SlackResponse: ...
    @staticmethod
    def validate_slack_signature(
        *, signing_secret: str, data: str, timestamp: str, signature: str
    ) -> bool: ...
