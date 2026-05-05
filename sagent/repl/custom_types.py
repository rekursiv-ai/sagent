"""Shared mutable state types for the REPL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.custom_types import Message


@dataclass(kw_only=True, slots=True)
class RenderState:
    """Mutable rendering state shared across event callbacks in one render frame."""

    console: Console
    out: Console
    buf: str = ""
    child_bufs: dict[str, str] = field(default_factory=dict)
    printed_header: bool = False
    done_event: Message | None = None
