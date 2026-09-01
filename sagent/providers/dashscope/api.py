"""DashScope (Alibaba) provider - OpenAI compatible.

Usage::

    from sagent.providers import DashScope

    provider = DashScope.from_env()          # DASHSCOPE_API_KEY
    model = provider.model()            # qwen3.6-plus
    response = await model.buffer(request)

Self-hosted (vLLM/SGLang on localhost)::

    provider = DashScope.from_key("empty", base_url="http://gpu-box:8000/v1")

Qwen3 surfaces reasoning via ``reasoning_content`` when
``enable_thinking=true`` is passed. The hybrid thinking/non-thinking
models default to non-thinking; set ``effort`` on the request to
switch behavior (mapped to ``enable_thinking`` in the body).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast, override

from sagent.catalog import dashscope as dashscope_catalog
from sagent.lib.custom_json import MutableJSON
from sagent.providers.openai.compat import (
    OpenAICompat,
    OpenAICompatModel,
)
from sagent.types.model import (
    ModelCapability,
    ModelRequest,
    ThinkingEffort,
)


# Non-reasoning variants that share a thinking prefix but reject the
# ``enable_thinking`` / ``thinking_budget`` knobs: the ``instruct`` and
# ``coder`` qwen3 ids (e.g. ``qwen3-235b-a22b-instruct-2507``,
# ``qwen3-coder-480b-a35b-instruct``). Matching the bare ``qwen3`` prefix is
# necessary to include the hybrid ``qwen3.6`` family, but these markers carve the
# pure non-reasoning models back out -- the same wire hazard the ``-thinking``
# suffix strip guards against, from the other direction. Markers match as whole
# hyphen-delimited SEGMENTS (not substrings), so a hypothetical ``qwen3-
# instructive-…`` id is NOT mis-flagged.
_NON_REASONING_MARKERS = frozenset({"instruct", "coder"})


def _is_non_reasoning_variant(model_id: str) -> bool:
    """True if ``model_id`` carries a non-reasoning marker as a hyphen segment."""
    return bool(_NON_REASONING_MARKERS.intersection(model_id.split("-")))


class _DashScopeModel(OpenAICompatModel):
    """DashScope backend - reasoning_content, enable_thinking routing."""

    _reasoning_field: ClassVar[str | None] = "reasoning_content"

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        """True for Qwen3/QwQ/QvQ models that accept ``enable_thinking``.

        A thinking prefix is necessary but not sufficient: the ``-instruct`` /
        ``-coder`` qwen3 ids share the prefix yet are pure non-reasoning models
        that reject the toggle, so they are excluded here.
        """
        if _is_non_reasoning_variant(model_id):
            return False
        # Prefixes WITHOUT a trailing hyphen match both the hyphenated ids
        # (``qwen3-32b``) and the dotted generation ids (``qwen3.6-plus``, the
        # default). A bare ``qwen3-`` silently excluded the entire qwen3.6
        # family -- including ``DEFAULT_MODEL`` -- from the effort knob.
        return any(
            model_id.startswith(p)
            for p in ("qwen3", "qwen-plus", "qwen-max", "qwq", "qvq")
        )

    @override
    def _transform_body(
        self,
        body: MutableJSON,
        request: ModelRequest,
    ) -> MutableJSON:
        """Map sagent's effort onto DashScope's thinking knobs.

        DashScope rejects ``reasoning_effort``; it exposes ``enable_thinking``
        (on/off) plus an optional ``thinking_budget`` reasoning-token cap. Read
        the RAW ``request.effort`` rather than the wire-mapped value the base
        already wrote into ``reasoning_effort`` -- the base maps ``none`` to
        ``minimal`` for OpenAI, which would otherwise read as "thinking on" and
        discard the level entirely.
        """
        body.pop("reasoning_effort", None)
        # A row that advertises no effort is claiming the model REJECTS the
        # knob (the ``-instruct`` / ``-coder`` / ``-turbo`` ids); never send one.
        if request.effort is None or not self.spec.supported_thinking_efforts:
            return body
        # The catalog holds the wire budget as data. An inline ladder here
        # drifted from it -- ``xhigh`` billed 24_576 reasoning tokens where
        # the row (and the UI reading it) said 20_480.
        budget = self.spec.supported_thinking_efforts.get(
            cast(ThinkingEffort, request.effort)
        )
        if budget is None:
            valid = ", ".join(self.spec.supported_thinking_efforts)
            raise ValueError(
                f"Unknown effort {request.effort!r} for {self._model_id}."
                f" Valid efforts: {valid}",
            )
        # Qwen spells "no reasoning" as a toggle, not a zero budget.
        body["enable_thinking"] = budget != "0"
        if budget != "0":
            body["thinking_budget"] = int(budget)
        return body


# DashScope/Qwen-VL preprocesses images server-side (Qwen resizes via
# min_pixels/max_pixels; object localization is robust 480-2560 px) and
# publishes no hard per-image pixel/byte reject the client must preempt, nor a
# request-body byte ceiling. Use the 0=unlimited sentinel rather than borrowing
# OpenAI's caps (verified Jun 2026;
# https://www.alibabacloud.com/help/en/model-studio/vision).


class DashScope(OpenAICompat):
    """DashScope (Alibaba) provider."""

    DEFAULT_MODEL: ClassVar[str] = "qwen3.6-plus"
    ENV_VAR: ClassVar[str] = "DASHSCOPE_API_KEY"
    # International endpoint. For mainland China use
    # dashscope.aliyuncs.com via the ``base_url=`` override.
    BASE_URL: ClassVar[str] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    # Model limits and pricing.
    # Source: https://help.aliyun.com/zh/model-studio/developer-reference/
    # Cross-ref: https://github.com/taylorwilsdon/llm-context-limits
    #
    # To add a new model: check the Alibaba Cloud Model Studio docs
    # for context window and max output tokens.
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = dashscope_catalog.MODELS
    """Per-model capability; transport limits live on ``TRANSPORT``."""

    MODEL_CLASS: ClassVar[type[OpenAICompatModel]] = _DashScopeModel
