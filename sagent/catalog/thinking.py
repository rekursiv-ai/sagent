"""How hard a model is asked to think, and what comes back.

Shared because a catalog DECLARES these, a session RECORDS them, and a harness
REQUESTS them -- three layers naming one vocabulary. A level added here reaches
all three; added to one spelling of three, it reached none of the others.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal


__all__ = [
    "ALL_THINKING_EFFORTS",
    "SummaryKind",
    "ThinkingBudget",
    "ThinkingEffort",
    "ThinkingOutput",
]


type ThinkingEffort = Literal[
    "off",
    "min",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
"""How hard a model is asked to think.

The full range a catalog declares and a session records. A transport accepts a
SUBSET -- no CLI exposes ``off`` or ``min`` -- and names its own in
``ModelSpec.efforts``, because a level absent there is one whose setting would
appear to work while changing nothing.
"""

type ThinkingBudget = Literal["auto", "fixed"]
"""Whether the caller may leave the reasoning budget open, fix it, or both."""

type ThinkingOutput = Literal["text", "redacted"]
"""Whether reasoning comes back readable, redacted, or either."""

type SummaryKind = Literal[
    "auto",
    "concise",
    "detailed",
    "none",
]
"""How verbose a reasoning summary the caller asked for.

``"none"`` where :data:`ThinkingEffort` says ``"off"``: both name "do not do
this", and the mismatch is the PROVIDERS', not ours. Codex writes ``"none"``
here (1078 times across the captured corpus) and every catalog writes ``"off"``
there, and both values go back to the wire verbatim -- so spelling them alike
would break the round trip rather than tidy it.

``"none"`` is a request for no summary; ``None`` is no request at all.
"""

ALL_THINKING_EFFORTS: Mapping[ThinkingEffort, str] = MappingProxyType(
    {
        "off": "off",
        "min": "min",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
)
"""Every level, mapped to the wire value a provider that offers all of them
sends. A transport narrows this to what it actually accepts."""
