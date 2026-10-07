"""Moonshot provider - OpenAI chat-completions compatible.

Usage::

    from sagent.providers import Moonshot

    provider = Moonshot.from_env()          # MOONSHOT_API_KEY
    model = provider.model()            # kimi-3.0
    response = await model.buffer(request)

Self-hosted::

    provider = Moonshot.from_key("no-auth", base_url="http://gpu-box:8000/v1")

Moonshot streams reasoning text via ``reasoning_content`` (same as
DeepSeek/DashScope). Tool-calling uses the standard ``tool_calls`` block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from sagent.catalog.moonshot import models
from sagent.catalog.openai import compatible
from sagent.catalog.table import ModelCatalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree
    from sagent.types.model import ModelRequest


class _MoonshotModel(OpenAICompatModel):
    """Moonshot backend - surfaces ``reasoning_content`` as thinking."""

    _reasoning_field: ClassVar[str] = "reasoning_content"

    @override
    def _transform_body(
        self,
        body: dict[str, MutablePlainTree],
        request: ModelRequest,
    ) -> dict[str, MutablePlainTree]:
        """Drop the fixed sampler and map thinking onto Kimi's knobs."""
        del request
        # K3: "temperature=1.0 ... are fixed; omit them from requests"; K2.x:
        # "temperature is not modifiable -- use the default and do not pass
        # it explicitly".
        body.pop("temperature", None)
        # The base folds ``max`` onto OpenAI's ``high``; K3 takes the level
        # verbatim, and its row offers only levels the wire spells.
        if self.capability.effort_as_level and self.settings.thinking_effort != "none":
            body["reasoning_effort"] = self.settings.thinking_effort
        # Only K2.6 offers a switch; K3 and K2.7 Code "should not" be sent
        # the ``thinking`` parameter at all.
        if self.capability.thinking.budget == frozenset({"none", "auto"}):
            body["thinking"] = {
                "type": "disabled"
                if self.settings.thinking_budget == "none"
                else "enabled",
            }
        return body


class Moonshot(OpenAICompat):
    """Moonshot AI provider."""

    ENV_VAR: ClassVar[str] = "MOONSHOT_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.moonshot.ai/v1"

    catalog = ModelCatalog(models=models(), transport=compatible())

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _MoonshotModel
