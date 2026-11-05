"""Native Replicate text predictions, bounded polling, and explicit tool refusal.

Pass a priced ``models=ModelTable(...)`` whose wire IDs identify Replicate
models or pinned versions. Supported wrappers accept ``prompt``,
``system_prompt`` and ``max_tokens``, and return a string or string fragments.
Other schemas require a dedicated adapter. Responses are yielded when the
prediction completes; SSE URLs are never followed or sent credentials.
"""

from __future__ import annotations

from http import HTTPMethod
from typing import TYPE_CHECKING, ClassVar, NotRequired, TypedDict, override
from urllib.parse import urlsplit

import asyncio
import re

from treekle import json_dumps, json_loads

from sagent.model.exceptions import DisconnectedError
from sagent.providers import replicate
from sagent.providers.openai import wire
from sagent.providers.openai.compat import OpenAICompatible, OpenAIModel
from sagent.session.types import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from types import ModuleType

    from sagent.model.types import Request


__all__ = ["Replicate"]


class Replicate(OpenAICompatible):
    """Run prompt-based text predictions without claiming tool-call support."""

    section: ClassVar[str] = "replicate"
    aliases: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[ModuleType] = replicate
    url: ClassVar[str] = "https://api.replicate.com/v1"
    key_variable: ClassVar[str] = "REPLICATE_API_TOKEN"
    base_url_variable: ClassVar[str] = "REPLICATE_BASE_URL"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        if request.tools or any(
            isinstance(record, (ToolCall, ToolResult))
            for record in wire.flatten(request.records)
        ):
            raise ValueError("Replicate text wrappers do not support tools.")
        parts: list[str] = []
        for record in wire.flatten(request.records):
            if isinstance(record, (UserMessage, AssistantMessage)):
                if record.attachments:
                    raise ValueError(
                        "Replicate text wrappers do not support attachments.",
                    )
                role = "User" if isinstance(record, UserMessage) else "Assistant"
                parts.append(f"{role}: {record.content or ''}")
        return {
            "version": model.capability.wire_model_id or model.capability.model_id,
            "input": {
                "prompt": "\n\n".join(parts),
                "system_prompt": "\n\n".join(request.system),
                "max_tokens": request.response_tokens(model.capability.limits),
            },
        }

    async def prediction(self, url: str) -> _Prediction:
        """Get a prediction only from the trusted Replicate API host."""
        _trusted(url)
        return await self._prediction(HTTPMethod.GET, url=url)

    @override
    async def events(
        self,
        body: Mapping[str, object],
        *,
        headers: Mapping[str, str],
        retry: bool = True,
    ) -> AsyncIterator[dict[str, object]]:
        del headers, retry
        prediction = await self._prediction(
            HTTPMethod.POST,
            url=f"{(await self.endpoint()).rstrip('/')}/predictions",
            body=body,
        )
        async with asyncio.timeout(600):
            while prediction["status"] in ("starting", "processing"):
                url = prediction.get("urls", {}).get("get", "")
                prediction = await self.prediction(url)
                if prediction["status"] in ("starting", "processing"):
                    await asyncio.sleep(0.5)
        if prediction["status"] != "succeeded":
            raise RuntimeError(
                f"Replicate prediction {prediction['status']}: {prediction.get('error') or 'no error detail'}",
            )
        output = prediction.get("output")
        if output is None:
            raise DisconnectedError("Replicate returned no text output.")
        text = output if isinstance(output, str) else "".join(output)
        metrics = prediction.get("metrics", {})
        if "input_token_count" not in metrics or "output_token_count" not in metrics:
            raise RuntimeError(
                "Replicate text wrapper did not report token counts; billing cannot be represented.",
            )
        yield {
            "choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": metrics["input_token_count"],
                "completion_tokens": metrics["output_token_count"],
            },
        }

    async def _prediction(
        self,
        method: HTTPMethod,
        *,
        url: str,
        body: Mapping[str, object] | None = None,
    ) -> _Prediction:
        response = await self.resources.http.request(
            method,
            url,
            content=json_dumps(body) if body is not None else None,
            headers={
                **await self.credentials(),
                "content-type": "application/json",
                "Cancel-After": "10m",
            },
            follow_redirects=False,
        )
        if response.status_code >= 300:
            raise wire.to_exception(
                response.status_code,
                body=response.text,
                headers=response.headers,
            )
        return json_loads(response.content, _Prediction)


class _Prediction(TypedDict):
    status: str
    output: NotRequired[str | list[str] | None]
    error: NotRequired[str | None]
    urls: NotRequired[dict[str, str]]
    metrics: NotRequired[dict[str, int | float]]


def _trusted(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.replicate.com"
        or parsed.query
        or parsed.fragment
        or re.fullmatch(r"/v1/predictions/[a-zA-Z0-9_-]+", parsed.path) is None
    ):
        raise ValueError("Prediction URL is not a trusted Replicate API URL.")
