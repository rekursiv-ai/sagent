"""Read the tool calls a local model wrote into its text, without I/O.

Qwen writes ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``;
DeepSeek and Kimi write ``<｜tool▁calls▁begin｜>`` blocks of ``function``
calls with fenced JSON arguments.

Typical usage example:

  calls, said = read_calls(text, offered={"read", "bash"})
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import re
import uuid

from treekle import from_plain, immutable, json_loads

from sagent.session.types import ToolCall


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


__all__ = [
    "read_calls",
]


def read_calls(text: str, *, offered: Iterable[str]) -> tuple[list[ToolCall], str]:
    """Return the tool calls written in ``text``, and the text without them.

    Args:
      text: What the model wrote.
      offered: The names of the tools offered. A call names one in any case,
        and takes the offered spelling, as the runtime looks tools up by it.

    Returns:
      calls: Each call read, in the order written.
      said: The text without those calls. A block that does not parse, or
        calls a tool not offered, stays in it.

    """
    names = {name.lower(): name for name in offered}
    calls, text = _strip(
        text,
        pattern=r"<tool_call>(.*?)</tool_call>",
        parse=_qwen_calls,
        names=names,
    )
    more, text = _strip(
        text,
        pattern="<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
        parse=_deepseek_calls,
        names=names,
    )
    return [*calls, *more], text.strip()


def _strip(
    text: str,
    *,
    pattern: str,
    parse: Callable[[str], list[ToolCall] | None],
    names: Mapping[str, str],
) -> tuple[list[ToolCall], str]:
    """Take each block ``pattern`` matches out of ``text`` that ``parse`` reads."""
    calls: list[ToolCall] = []
    kept: list[str] = []
    end = 0
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        parsed = parse(match.group(1))
        if parsed is None or any(call.name.lower() not in names for call in parsed):
            continue
        calls.extend(replace(call, name=names[call.name.lower()]) for call in parsed)
        kept.append(text[end : match.start()])
        end = match.end()
    return calls, "".join((*kept, text[end:]))


def _qwen_calls(body: str) -> list[ToolCall] | None:
    """Read ``{"name": ..., "arguments": {...}}``."""
    try:
        payload = json_loads(body.strip(), dict[str, object])
        name = from_plain(payload.get("name"), str)
        arguments = from_plain(payload.get("arguments") or {}, dict[str, object])
    except ValueError:
        return None
    # A name no tool has, the empty one included, is refused with the block.
    return [_tool_call(name, arguments)]


def _deepseek_calls(body: str) -> list[ToolCall] | None:
    r"""Read each ``function\n{name}\n```json\n{arguments}\n``` `` call of a block."""
    calls: list[ToolCall] = []
    for match in re.finditer(
        "<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>",
        body,
        flags=re.DOTALL,
    ):
        name = re.search(r"function\s*\n\s*([A-Za-z0-9_]+)", match.group(1))
        arguments = re.search(r"```json\s*(.*?)\s*```", match.group(1), flags=re.DOTALL)
        if name is None or arguments is None:
            return None
        try:
            parsed = json_loads(arguments.group(1), dict[str, object])
        except ValueError:
            return None
        calls.append(_tool_call(name.group(1), parsed))
    return calls or None


def _tool_call(name: str, arguments: Mapping[str, object]) -> ToolCall:
    return ToolCall(
        call_id=f"call_{uuid.uuid4().hex[:12]}",
        name=name,
        arguments=immutable(dict(arguments)),
    )
