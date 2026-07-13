from ssl import SSLContext
from typing import Any

import logging

from slack_sdk.http_retry.handler import RetryHandler

from .response import AuditLogsResponse

"""Audit Logs API is a set of APIs for monitoring what’s happening in your Enterprise Grid organization.

Refer to https://docs.slack.dev/tools/python-slack-sdk/audit-logs for details.
"""

class AuditLogsClient:
    BASE_URL = ...
    token: str
    timeout: int
    ssl: SSLContext | None
    proxy: str | None
    base_url: str
    default_headers: dict[str, str]
    logger: logging.Logger
    retry_handlers: list[RetryHandler]
    def __init__(
        self,
        token: str,
        timeout: int = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        base_url: str = ...,
        default_headers: dict[str, str] | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        logger: logging.Logger | None = ...,
        retry_handlers: list[RetryHandler] | None = ...,
    ) -> None: ...
    def schemas(
        self,
        *,
        query_params: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> AuditLogsResponse: ...
    def actions(
        self,
        *,
        query_params: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> AuditLogsResponse: ...
    def logs(
        self,
        *,
        latest: int | None = ...,
        oldest: int | None = ...,
        limit: int | None = ...,
        action: str | None = ...,
        actor: str | None = ...,
        entity: str | None = ...,
        cursor: str | None = ...,
        additional_query_params: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> AuditLogsResponse: ...
    def api_call(
        self,
        *,
        http_verb: str = ...,
        path: str,
        query_params: dict[str, Any] | None = ...,
        body_params: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> AuditLogsResponse: ...
