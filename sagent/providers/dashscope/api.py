"""DashScope (Alibaba) provider - OpenAI compatible.

Usage::

    from sagent.providers import DashScope

    provider = DashScope.from_env()          # DASHSCOPE_API_KEY
    model = provider.model()            # qwen-plus-3.7
    response = await model.buffer(request)

Self-hosted (vLLM/SGLang on localhost)::

    provider = DashScope.from_key("empty", base_url="http://gpu-box:8000/v1")

Qwen surfaces reasoning via ``reasoning_content``. Every thinking knob is
read off the catalog row: ``thinking_budget`` toggles ``enable_thinking``,
and the effort rides as ``reasoning_effort`` on rows that take a level or
as a ``thinking_budget`` cap on rows that take tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, override

from sagent.catalog.dashscope import models, thinking_budget
from sagent.catalog.openai import compatible
from sagent.catalog.table import ModelCatalog
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree
    from sagent.types.model import (
        ModelRequest,
    )


class _DashScopeModel(OpenAICompatModel):
    """DashScope backend - reasoning_content, enable_thinking routing."""

    _reasoning_field: ClassVar[str] = "reasoning_content"

    # The base spells effort in OpenAI's vocabulary (``xhigh`` -> ``high``),
    # which DashScope either rejects or folds differently, so it is dropped
    # and re-derived from the row.
    @override
    def _transform_body(
        self,
        body: dict[str, MutablePlainTree],
        request: ModelRequest,
    ) -> dict[str, MutablePlainTree]:
        """Map sagent's thinking axes onto DashScope's knobs."""
        del request
        body.pop("reasoning_effort", None)
        # Only the models qwen-api-via-openai-chat-completions lists honor
        # ``max_completion_tokens``; every other id takes ``max_tokens``.
        cap = body.pop("max_completion_tokens", None)
        if cap is not None:
            field = (
                "max_completion_tokens"
                if self.capability.model_id in _COMPLETION_CAP
                else "max_tokens"
            )
            body[field] = cap
        offered = self.capability.thinking.budget
        # A row offering only ``none`` claims the model REJECTS the toggle
        # (the ``-instruct`` / ``-coder`` ids); never send one.
        if offered == frozenset({"none"}):
            return body
        settings = self.settings
        body["enable_thinking"] = settings.thinking_budget != "none"
        if settings.thinking_budget == "none" or settings.thinking_effort == "none":
            return body
        # Qwen 3.8 errors when ``reasoning_effort`` and ``thinking_budget``
        # arrive together, so a level row never sends the cap.
        if self.capability.effort_as_level:
            body["reasoning_effort"] = settings.thinking_effort
        elif settings.thinking_budget == "fixed":
            body["thinking_budget"] = int(thinking_budget(settings.thinking_effort))
        return body


_COMPLETION_CAP: Final = frozenset(
    {
        "qwen-max-3.8",
        "qwen-flash-3.8",
        "qwen-max-3.7",
        "qwen-plus-3.7",
        "qwen-flash-3.7",
        "qwen-plus-3.6",
        "qwen-flash-3.6",
    },
)
"""Rows the API page lists for ``max_completion_tokens`` (Max 3.7+, Plus/Flash 3.5+).

Absent: ``qwen3.8-27b`` and ``qwen3.8-2.4t-a95b`` are neither Max, Plus, nor
Flash; ``qwen3.6-max-preview`` predates the Max 3.7 floor.
"""


class DashScope(OpenAICompat):
    """DashScope (Alibaba) provider."""

    ENV_VAR: ClassVar[str] = "DASHSCOPE_API_KEY"
    # International endpoint. For mainland China use
    # dashscope.aliyuncs.com via the ``base_url=`` override.
    BASE_URL: ClassVar[str] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    catalog = ModelCatalog(models=models(), transport=compatible())

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _DashScopeModel
