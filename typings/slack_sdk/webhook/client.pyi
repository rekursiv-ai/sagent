from collections.abc import Sequence
from ssl import SSLContext
from typing import Any

import logging

from slack_sdk.http_retry.handler import RetryHandler
from slack_sdk.models.attachments import Attachment
from slack_sdk.models.blocks import Block

from .webhook_response import WebhookResponse

class WebhookClient:
    url: str
    timeout: int
    ssl: SSLContext | None
    proxy: str | None
    default_headers: dict[str, str]
    logger: logging.Logger
    retry_handlers: list[RetryHandler]
    def __init__(
        self,
        url: str,
        timeout: int = ...,
        ssl: SSLContext | None = ...,
        proxy: str | None = ...,
        default_headers: dict[str, str] | None = ...,
        user_agent_prefix: str | None = ...,
        user_agent_suffix: str | None = ...,
        logger: logging.Logger | None = ...,
        retry_handlers: list[RetryHandler] | None = ...,
    ) -> None: ...
    def send(
        self,
        *,
        text: str | None = ...,
        attachments: Sequence[dict[str, Any] | Attachment] | None = ...,
        blocks: Sequence[dict[str, Any] | Block] | None = ...,
        response_type: str | None = ...,
        replace_original: bool | None = ...,
        delete_original: bool | None = ...,
        unfurl_links: bool | None = ...,
        unfurl_media: bool | None = ...,
        metadata: dict[str, Any] | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> WebhookResponse: ...
    def send_dict(
        self, body: dict[str, Any], headers: dict[str, str] | None = ...
    ) -> WebhookResponse: ...
