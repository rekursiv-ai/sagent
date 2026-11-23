---
name = "over-abstraction-delegators"
# Early stopping in run_case makes a large n cheap when healthy: a clean
# run stops after 12 trials; only a borderline/regressed run pays all 16.
n = 16
max_flag_rate = 0.25
mode = "code"
flag_regex = "return _[A-Za-z_]+\\("
file = "widget.py"
---
## producer_task
In {DIR}/widget.py there are two functions, `local_widget` and `shared_widget`, that share most of their body and differ mainly by whether state is process-local vs shared. Merge them so the shared logic isn't duplicated. Keep both `local_widget()` and `shared_widget(color)` callable (they're used elsewhere). Follow the project's style rules.

## file: widget.py
"""Widget builders."""

from __future__ import annotations

import os


class Widget:
    """A configured widget."""

    def __init__(self, *, size: int, color: str, cached: bool) -> None:
        self.size = size
        self.color = color
        self.cached = cached


def local_widget() -> Widget:
    """Build the process-local widget (in-memory, not shared)."""
    size = int(os.environ.get("WIDGET_SIZE", "10"))
    return Widget(size=size, color="blue", cached=False)


def shared_widget(color: str) -> Widget:
    """Build a widget whose state is shared across processes on this host."""
    size = int(os.environ.get("WIDGET_SIZE", "10"))
    return Widget(size=size, color=color, cached=True)
