"""A model context's size: the price band it falls in, its tokens, and how it reads.

size = model.settings.context_size
size.context_name                       # smallcontext, midcontext, or fullcontext
size.tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


__all__ = [
    "ContextSize",
    "ContextSizeName",
]


# ``smallcontext`` and ``midcontext`` are the bands a model bills cheaper below; ``fullcontext``
# is its whole window.
type ContextSizeName = Literal[
    "smallcontext",
    "midcontext",
    "fullcontext",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextSize:
    """A context's size: the price band it falls in, its tokens, and how it reads."""

    context_name: ContextSizeName = "smallcontext"
    tokens: int = 0
    display_name: str = "0"
