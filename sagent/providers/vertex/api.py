"""Vertex AI using native Gemini wire records, including thought signatures.

Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (default ``global``).
An explicit ``api_key`` or VERTEX_ACCESS_TOKEN supplies an OAuth access token,
not a Gemini API key. Otherwise ``gcloud auth application-default
print-access-token`` reads ADC through the installed Cloud SDK; override the
command with VERTEX_TOKEN_COMMAND. No undeclared google-auth dependency is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import asyncio
import os
import re
import shlex

from treekle import from_plain

from sagent.lib.posix.child import Child
from sagent.model.exceptions import AuthError, UnknownModelError
from sagent.providers import vertex
from sagent.providers.google.api import Google, GoogleModel


if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

    from sagent.model.capability import ModelCapability


__all__ = ["Vertex"]


class Vertex(Google):
    """Serve Vertex Gemini through native generateContent and countTokens."""

    section: ClassVar[str] = "vertex"
    aliases: ClassVar[tuple[str, ...]] = ("vertex-ai",)
    package: ClassVar[ModuleType] = vertex

    @override
    def __getitem__(self, model: str) -> GoogleModel:
        if (
            self._models is None
            and os.environ.get("GOOGLE_CLOUD_LOCATION", "global") != "global"
        ):
            raise UnknownModelError(
                "Regional Vertex pricing requires models=ModelTable(...).",
            )
        return super().__getitem__(model)

    @override
    def endpoint(self, method: str, *, model: str) -> str:
        base = self._base_url or os.environ.get("VERTEX_BASE_URL", "")
        if not base:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if not project or re.fullmatch(r"[a-zA-Z0-9_-]+", project) is None:
                raise AuthError("Set GOOGLE_CLOUD_PROJECT or pass base_url for Vertex.")
            if re.fullmatch(r"[a-z0-9-]+", location) is None:
                raise AuthError("GOOGLE_CLOUD_LOCATION is invalid.")
            host = (
                "aiplatform.googleapis.com"
                if location == "global"
                else f"{location}-aiplatform.googleapis.com"
            )
            base = f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google"
        return f"{base.rstrip('/')}/models/{model}:{method}"

    @override
    async def credentials(self) -> dict[str, str]:
        if token := self._api_key or os.environ.get("VERTEX_ACCESS_TOKEN", ""):
            return {"authorization": f"Bearer {token}"}
        command = os.environ.get(
            "VERTEX_TOKEN_COMMAND",
            "gcloud auth application-default print-access-token",
        )
        try:
            async with (
                asyncio.timeout(30),
                await Child.start(
                    *shlex.split(command),
                    group=True,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                ) as child,
            ):
                output, _ = await child.process.communicate()
                if child.process.returncode != 0 or not output.strip():
                    raise AuthError(
                        "Vertex token command failed; configure ADC or VERTEX_ACCESS_TOKEN.",
                    )
        except OSError as error:
            raise AuthError(
                "Cannot obtain Vertex access token; configure gcloud ADC or VERTEX_ACCESS_TOKEN.",
            ) from error
        return {"authorization": f"Bearer {output.decode().strip()}"}

    @override
    async def reauthenticate(self, *, rejected: Mapping[str, str]) -> bool:
        del rejected
        return not (self._api_key or os.environ.get("VERTEX_ACCESS_TOKEN", ""))

    @override
    async def count(
        self,
        body: Mapping[str, object],
        *,
        capability: ModelCapability,
    ) -> int:
        model = capability.wire_model_id or capability.model_id
        reply = await self.reply(self.endpoint("countTokens", model=model), body=body)
        return from_plain(reply.get("totalTokens"), int)
