"""MiniMax provider - OpenAI chat-completions compatible.

Usage::

    from sagent.providers import MiniMax

    provider = MiniMax.from_env()       # MINIMAX_API_KEY
    model = provider.model()            # minimax-3.0
    response = await model.buffer(request)

Self-hosted::

    provider = MiniMax.from_key("empty", base_url="http://gpu-box:8000/v1")

Reasoning surfaces via ``reasoning_content`` once ``reasoning_split`` is
set. Tool-calling uses the standard ``tool_calls`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.catalog.minimax import api, models
from sagent.catalog.table import ModelCatalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree
    from sagent.types.model import ModelRequest


class _MiniMaxModel(OpenAICompatModel):
    """MiniMax backend - reasoning surfaces via ``reasoning_content``."""

    _reasoning_field: ClassVar[str] = "reasoning_content"

    @override
    def _transform_body(
        self,
        body: dict[str, MutablePlainTree],
        request: ModelRequest,
    ) -> dict[str, MutablePlainTree]:
        """Split reasoning out of ``content`` and send the thinking switch."""
        del request
        # Without it, thinking stays "inside ``content`` wrapped in
        # ``<think>`` tags", which the parser would show as the answer.
        body["reasoning_split"] = True
        # Only M3 offers a switch; M2.x accepts ``disabled`` but ignores it,
        # so its rows offer nothing to send.
        if self.capability.thinking.budget == frozenset({"none"}):
            return body
        body["thinking"] = {
            "type": "disabled"
            if self.settings.thinking_budget == "none"
            else "adaptive",
        }
        return body


class MiniMax(OpenAICompat):
    """MiniMax provider (api.minimax.io)."""

    ENV_VAR: ClassVar[str] = "MINIMAX_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.minimax.io/v1"

    catalog = ModelCatalog(models=models(), transport=api())

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MiniMaxModel
