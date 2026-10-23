"""DashScope/Qwen thinking-wire mappings.

Load model facts with ``sagent.catalog.models("dashscope")``.

Thinking axes: ``budget`` ``none`` is ``enable_thinking=false``; ``auto``
thinks at the model's default depth; ``fixed`` caps it with
``thinking_budget``. Rows with ``effort_as_level`` (Qwen 3.8) take the
effort as ``reasoning_effort`` instead, and reject ``thinking_budget``
alongside it. Effort ``none`` sends no depth knob at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sagent.types.capability import ThinkingEffort


__all__ = ["thinking_budget"]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinking_budget`` token cap an effort sends.

    Only rows without ``effort_as_level`` send it, and only under a
    ``fixed`` budget; ``0`` is the off value an effort of ``none`` maps to.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    """
    match effort:
        case "none":
            return "0"
        case "min":
            return "1024"
        case "low":
            return "4096"
        case "medium":
            return "8192"
        case "high":
            return "16384"
        case "xhigh":
            return "20480"
        case "max":
            return "24576"
