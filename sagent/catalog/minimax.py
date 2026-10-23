"""MiniMax chat-completions transport restrictions.

Load model facts with ``sagent.catalog.models("minimax")``.
Transport capabilities narrow those rows with ``&``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from sagent.catalog.openai import compatible


if TYPE_CHECKING:
    from sagent.types.capability import ModelCapability
    from sagent.types.cost import ServiceTier


__all__ = ["api"]


def api() -> ModelCapability:
    """Return the chat-completions transport with MiniMax's ``priority`` tier.

    Returns:
      capability: The shared compatible transport, plus ``priority``.

    """
    transport = compatible()
    priority: frozenset[ServiceTier] = frozenset({"priority"})
    return replace(transport, service_tier=transport.service_tier | priority)
