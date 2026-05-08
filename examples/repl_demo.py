r"""Demo: run the REPL against a real provider/model.

Builds an ``Agent``, attaches the REPL bundle (rich console +
prompt-toolkit input), and drives it interactively. No session
persistence, no compactor, no tools -- a minimal showcase of the
dispatch loop in production.

Usage::

    uv --quiet run --frozen python -m \
        examples.repl_demo \
        --provider Anthropic --auth env --model claude-haiku-4-5

Or with the offline echo for iteration without an API key::

    uv --quiet run --frozen python -m \
        examples.repl_demo --offline
"""

from __future__ import annotations

from collections.abc import Callable

import argparse
import asyncio
import sys

from sagent.agent.agent import Agent
from sagent.agent.handlers import core_handlers
from sagent.custom_types import (
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.providers import build_provider
from sagent.repl import run_repl
from sagent.testing import MockModelCaps


class _OfflineEcho(MockModelCaps):
    """Offline stand-in that echoes the last user message back."""

    max_image_dim: int = 2000

    @property
    def max_request_tokens(self) -> int:
        return 100_000

    @property
    def model_id(self) -> str:
        return "offline-echo"

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self._echo(request)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del on_thinking
        if on_text is not None:
            for word in self._last_user(request).split():
                on_text(word + " ")
        return await self._echo(request)

    @staticmethod
    def _last_user(request: ModelRequest) -> str:
        for msg in reversed(request.messages):
            if msg.descriptor == "text/x-user-message":
                return str(msg.content)
        return ""

    async def _echo(self, request: ModelRequest) -> ModelResponse:
        text = self._last_user(request)
        return ModelResponse(
            content=MultipartMessage(
                (TextMessage(f"echo: {text}", "text/plain"),),
                "multipart/x-model-message",
            ),
            tokens=TokenCount(input_tokens=len(text), output_tokens=len(text)),
            stop_reason="model_finished",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--offline", action="store_true")
    _ = parser.add_argument("--provider", default="Anthropic")
    _ = parser.add_argument("--auth", default="env")
    _ = parser.add_argument("--model", default=None)
    _ = parser.add_argument("--account", default=None)
    args = parser.parse_args()

    if args.offline:
        model = _OfflineEcho()
    else:
        provider = build_provider(args.provider, args.auth, account=args.account)
        model = provider.model(args.model)

    sys.stderr.write(f"[v2] {model.model_id}\n")

    agent = Agent(model=model, handlers=core_handlers())
    asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
