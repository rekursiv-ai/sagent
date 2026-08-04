"""Defaults every ``Model`` backend shares.

A provider implements ``stream``, ``is_context_overflow``, and its wire
mapping. Everything else has one sensible answer that most transports
never override:

- ``buffer`` is ``stream`` with no publisher, in 7 of 7 providers.
- ``actual_*`` fall back to the ``approx_*`` heuristic unless the vendor
  exposes a real token-counting endpoint (only Anthropic does).
- ``is_retryable_provider_error`` is ``False`` unless the transport can
  recognize its own transient failures (2 of 7).
- ``usage_snapshot`` is ``None`` unless the transport reports quota
  windows (3 of 7).
- ``close`` is a no-op for backends holding no client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from sagent.lib import token_count


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.types.model import (
        ModelRequest,
        ModelResponse,
        UsageSnapshot,
    )
    from sagent.types.runtime import RuntimeEvent


__all__ = ["ModelDefaults"]


class _Transport(Protocol):
    """What a subclass must supply for these defaults to work."""

    def approx_text_tokens(self, text: str) -> int: ...
    def approx_image_tokens(self, data: bytes) -> int: ...
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse: ...


class ModelDefaults(_Transport, Protocol):
    """Mixin supplying the non-transport half of the ``Model`` Protocol."""

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Delegate to the local heuristic; no vendor endpoint by default."""
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Delegate to the local heuristic; no vendor endpoint by default."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Delegate to the local heuristic; no vendor endpoint by default."""
        return self.approx_request_tokens(request)

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Non-streaming send: ``stream`` with no publisher.

        Routed through ``stream`` rather than a vendor's non-streaming
        endpoint on purpose: those carry a fixed client timeout that
        large compaction prompts exceed, while the streaming path uses
        an idle-based one.
        """
        return await self.stream(request, None)

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """Whether the provider treats ``error`` as transient.

        ``False`` unless the transport can recognize its own transient
        failures: misclassifying a fatal error as retryable burns the
        whole retry budget on a request that can never succeed.
        """
        del error
        return False

    def usage_snapshot(self) -> UsageSnapshot | None:
        """Latest quota windows, or ``None`` when the transport reports none."""
        return None

    async def close(self) -> None:
        """Release transport resources; a no-op when there are none."""
        return
