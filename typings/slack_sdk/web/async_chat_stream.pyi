from collections.abc import Sequence

import logging

from slack_sdk.models.blocks.blocks import Block
from slack_sdk.models.messages.chunk import Chunk
from slack_sdk.models.metadata import Metadata
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse

class AsyncChatStream:
    def __init__(
        self,
        client: AsyncWebClient,
        *,
        channel: str,
        logger: logging.Logger,
        thread_ts: str,
        buffer_size: int,
        recipient_team_id: str | None = ...,
        recipient_user_id: str | None = ...,
        task_display_mode: str | None = ...,
        **kwargs,
    ) -> None: ...
    async def append(
        self,
        *,
        markdown_text: str | None = ...,
        chunks: Sequence[dict | Chunk] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse | None: ...
    async def stop(
        self,
        *,
        markdown_text: str | None = ...,
        chunks: Sequence[dict | Chunk] | None = ...,
        blocks: str | Sequence[dict | Block] | None = ...,
        metadata: dict | Metadata | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
