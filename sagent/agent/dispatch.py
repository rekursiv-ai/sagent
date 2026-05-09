"""Tool dispatch helpers: read-only classification, invocation, validation."""

from __future__ import annotations

from collections.abc import Coroutine, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

import asyncio
import dataclasses
import logging

from sagent import agents_md
from sagent.custom_types import (
    Message,
    MultipartMessage,
    StreamingTool,
    TextMessage,
    Tool,
    is_streaming,
)
from sagent.lib.descriptors import (
    FILE_OP_TOOLS,
    READ_ONLY_TOOLS,
    is_multipart,
)
from sagent.lib.json import JSON
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    get_tool_name,
)
from sagent.tools.core import ToolState
from sagent.tools.input_errors import (
    INPUT_VALIDATION_PREFIX,
    MULTIPLE_TOOL_INPUT_ERRORS_HINT,
    TOOL_INPUT_RECOVERY_HINT,
    is_tool_input_error_text,
)
from sagent.tools.lib.bash import (
    cached_parse_bash as _cached_parse_bash,
    is_read_only as _bash_is_read_only,
)


logger = logging.getLogger(__name__)


def tc_tool_id(req: Message) -> str:
    """Return the full tool descriptor (``application/x-tool-*``) for a request.

    Works for both ``multipart/x-tool-call`` (new) and legacy
    ``application/x-tool-*`` (if ever encountered during migration).

    Args:
      req: Tool-call message.

    Returns:
      tool_id: Canonical ``application/x-tool-*`` descriptor.

    """
    if req.descriptor == "multipart/x-tool-call":
        return f"application/x-tool-{get_tool_name(req)}"
    return req.descriptor


def tc_directive(req: Message) -> JSON:
    """Return the JSON directive mapping for a tool-call request.

    Args:
      req: Tool-call message.

    Returns:
      directive: Frozen JSON mapping of tool inputs.

    """
    if req.descriptor == "multipart/x-tool-call":
        return get_directive(req)
    if req.descriptor.startswith("application/x-tool-"):
        return cast(JSON, req.content)
    return MappingProxyType({})


def conditional_rules_for_request(
    req: Message,
    tool_state: ToolState,
    seen: set[Path],
) -> str:
    """Return system-reminder text for conditional rules matching ``req``'s target.

    Returns empty string when the request isn't a file-op tool or no
    rules match. Dedupes within a dispatch batch via ``seen`` - three
    parallel Edits to matching files share the injection of each
    rule only once, instead of triplicating the same block.

    Args:
      req: Tool-call message.
      tool_state: Agent's mutable tool state.
      seen: Paths already injected this batch (mutated in-place).

    Returns:
      reminder: ``<system-reminder>`` block, or empty string.

    """
    if tc_tool_id(req) not in FILE_OP_TOOLS:
        return ""
    directive = tc_directive(req)
    target = directive.get("file_path")
    if not isinstance(target, str) or not target:
        return ""
    cfg = (
        agents_md.AgentsMdConfig(
            additional_dirs=[Path(d) for d in tool_state.additional_dirs],
        )
        if tool_state.additional_dirs
        else None
    )
    return agents_md.file_triggered_md_reminder(
        Path(tool_state.bash_cwd),
        [Path(target)],
        config=cfg,
        exclude_paths=seen,
    )


def tool_call_label(tool: Tool | None, req: Message) -> str:
    """Return the UI label for a tool call, using validation state if invalid."""
    if tool is None:
        return tc_tool_id(req)
    validation = _validation_details(tool, tc_directive(req))
    if validation is None:
        return tool.summary(req)
    missing, unexpected, _required, accepted = validation
    if missing:
        keys = ", ".join(f"`{k}`" for k in missing)
        return f"Invalid {tool.name} call: missing {keys}"
    keys = ", ".join(f"`{k}`" for k in unexpected)
    accepts = ", ".join(f"`{k}`" for k in accepted)
    suffix = f"; accepts {accepts}" if accepts else ""
    return f"Invalid {tool.name} call: unexpected {keys}{suffix}"


def is_request_read_only(req: Message, state: ToolState) -> bool:
    """Return True iff this tool call is safe to run concurrently.

    Bash classifications use ``state.bash_parse_cache`` so the parse is
    shared with the Bash tool's matcher dispatch later in the request.

    Args:
      req: Tool-call message.
      state: Agent's mutable tool state.

    Returns:
      read_only: Whether the call is side-effect-free.

    """
    tool_id = tc_tool_id(req)
    if tool_id in READ_ONLY_TOOLS:
        return True
    if tool_id == "application/x-tool-bash":
        directive = tc_directive(req)
        cmd = directive.get("command")
        if isinstance(cmd, str):
            trees = _cached_parse_bash(cmd, state.bash_parse_cache)
            return trees is not None and _bash_is_read_only(trees)
    return False


def invoke_tool(
    tools: dict[str, Tool],
    req: Message,
    events: asyncio.Queue[Message | None] | None = None,
) -> Coroutine[None, None, Message]:
    """Build the dispatch coroutine for one tool request.

    Returns an already-constructed coroutine ready to await. Hoisted
    out of ``_dispatch_tools`` so the dispatch path stays flat.

    Args:
      tools: Available tools keyed by descriptor.
      req: Tool-call message.
      events: Optional event queue for streaming tool output.

    Returns:
      coro: Coroutine that produces a ``multipart/x-tool-result`` message.

    """
    tool_id = tc_tool_id(req)
    t = tools.get(tool_id)
    if t is None:
        tool_name = get_tool_name(req)
        return _error_result(
            get_queue_id(req),
            req.id,
            f"Error: No such tool available: {tool_name}",
        )
    return _run_tool(t, req, events)


async def _run_tool(
    tool: Tool,
    req: Message,
    events: asyncio.Queue[Message | None] | None = None,
) -> Message:
    """Dispatch one tool call and produce a wire-facing tool result.

    Before dispatch, validate ``req.content`` against the tool's
    directive_schema and surface any mismatch as a structured error
    the model can act on (missing required field, unexpected key).
    Without this, a truncated tool_use from ``stop_reason=max_tokens``
    mid-JSON reaches the Python layer as a ``TypeError: ... missing
    required keyword-only argument`` and the model wedges
    re-generating the same malformed call.

    Handles both ``BatchTool`` (single ``await``) and ``StreamingTool``
    (async generator). For streaming tools, all yielded messages except
    the last are intermediate events; the final yield is the tool result.

    Tool exceptions propagate to the caller (``_invoke_tool_safe``)
    which converts them to ``<tool_use_error>``-wrapped tool results
    for the LLM and ``text/x-unexpected-error`` events for the REPL.
    """
    directive = tc_directive(req)
    tool_name = tool.name
    qid = get_queue_id(req)
    schema_error = _validate_tool_input(tool, directive)
    if schema_error is not None:
        logger.debug("Tool %s input invalid: %s", tool_name, schema_error)
        return MultipartMessage(
            (
                TextMessage(qid, "text/x-queue-id"),
                TextMessage(
                    schema_error,
                    "text/x-error",
                ),
            ),
            "multipart/x-tool-result",
            parent_id=req.id,
        )
    if is_streaming(tool):
        inner = await _drain_stream(tool, req, events)
    else:
        inner = await tool.run(req)
    # Set parent_id if the tool didn't (tools may set it themselves for tracing).
    if inner.parent_id == -1:
        inner = dataclasses.replace(inner, parent_id=req.id)
    # Wrap result in multipart/x-tool-result
    parts = (
        cast(tuple[Message, ...], inner.content)
        if is_multipart(inner.descriptor)
        else (inner,)
    )
    # Tools opt into the dim ``  ⎿ <summary>`` receipt line by returning
    # non-empty text from ``summary_result``. Skip on error -- the
    # ``text/x-error`` line already tells the story; a parallel summary
    # would just be noise.
    if not _has_error(parts):
        try:
            summary_text = tool.summary_result(inner)
        except Exception:  # noqa: BLE001 -- summary is best-effort UX, never fail dispatch
            logger.debug("summary_result raised for %s", tool.name, exc_info=True)
            summary_text = None
        if summary_text:
            parts = (
                *parts,
                TextMessage(str(summary_text), "text/x-tool-summary"),
            )
    return MultipartMessage(
        (TextMessage(qid, "text/x-queue-id"), *parts),
        "multipart/x-tool-result",
        parent_id=req.id,
    )


def _has_error(parts: tuple[Message, ...]) -> bool:
    """True if any part has an error descriptor."""
    return any(p.descriptor == "text/x-error" for p in parts)


async def _drain_stream(
    tool: StreamingTool,
    req: Message,
    events: asyncio.Queue[Message | None] | None = None,
) -> Message:
    """Consume a streaming tool's async generator, returning the final message.

    Intermediate yields are forwarded to the events queue (if active)
    as ``text/x-tool-stream-event`` messages. The last yielded message
    is treated as the tool result.

    Args:
      tool: Streaming tool to invoke.
      req: Tool-call request message.
      events: Optional event queue for streaming tool output.

    Returns:
      result: The final yielded Message.

    Raises:
      RuntimeError: If the generator yields nothing.

    """
    last: Message | None = None
    async for event in tool.run(req):
        if last is not None and events is not None:
            events.put_nowait(last)
        last = event
    if last is None:
        raise RuntimeError(f"StreamingTool {tool.name!r} yielded no messages")
    return last


def _validate_tool_input(
    tool: Tool,
    input_obj: Mapping[str, object],
) -> str | None:
    """Check ``input_obj`` against ``tool.directive_schema`` (required + known keys).

    Returns ``None`` when the input is well-formed; otherwise a
    human-readable error message the model should be able to act on
    directly. Echoes the tool directive_schema's ``required`` list so
    the model sees exactly what's expected. We don't type-check here:
    the actual call will surface wrong types with informative errors.
    """
    validation = _validation_details(tool, input_obj)
    if validation is None:
        return None
    missing, unexpected, req_list, accepted = validation
    issues = [f"The required parameter `{k}` is missing" for k in missing]
    issues.extend(f"Unexpected parameter `{k}`" for k in unexpected)
    header = (
        f"{tool.name} failed due to the following "
        f"{'issues' if len(issues) > 1 else 'issue'}:"
    )
    required_str = ", ".join(f"`{k}`" for k in req_list)
    accepted_str = ", ".join(f"`{k}`" for k in accepted)
    parts = [INPUT_VALIDATION_PREFIX + " " + "\n".join([header, *issues])]
    if required_str:
        parts.append(f"{tool.name} requires: {required_str}.")
    if unexpected and accepted_str:
        parts.append(f"{tool.name} accepts: {accepted_str}.")
    parts.append(TOOL_INPUT_RECOVERY_HINT)
    return "\n\n".join(parts)


def _validation_details(
    tool: Tool,
    input_obj: Mapping[str, object],
) -> tuple[list[str], list[str], list[str], list[str]] | None:
    schema = tool.directive_schema
    required = schema.get("required")
    req_list: list[str] = []
    if isinstance(required, (list, tuple)):
        req_list = [k for k in required if isinstance(k, str)]
    missing = [k for k in req_list if k not in input_obj]
    props = schema.get("properties")
    accepted: list[str] = []
    if isinstance(props, Mapping):
        accepted = [str(k) for k in props]
    unexpected: list[str] = []
    if schema.get("additionalProperties") is False:
        accepted_set = set(accepted)
        unexpected = [k for k in input_obj if k not in accepted_set]
    if not missing and not unexpected:
        return None
    return missing, unexpected, req_list, accepted


def has_tool_input_error(msg: Message) -> bool:
    """Return True when a tool result contains model-repairable input errors."""
    if not is_multipart(msg.descriptor):
        return False
    parts = cast(tuple[Message, ...], msg.content)
    return any(
        p.descriptor == "text/x-error" and is_tool_input_error_text(str(p.content))
        for p in parts
    )


def add_tool_input_batch_hint(results: list[Message]) -> list[Message]:
    """Add one aggregate recovery hint when a batch has multiple input errors."""
    bad = [i for i, result in enumerate(results) if has_tool_input_error(result)]
    if len(bad) < 2:
        return results
    out = list(results)
    first = bad[0]
    out[first] = _append_error_hint(out[first], MULTIPLE_TOOL_INPUT_ERRORS_HINT)
    return out


def _append_error_hint(msg: Message, hint: str) -> Message:
    parts = cast(tuple[Message, ...], msg.content)
    new_parts: list[Message] = []
    added = False
    for part in parts:
        if (
            not added
            and part.descriptor == "text/x-error"
            and is_tool_input_error_text(str(part.content))
        ):
            new_parts.append(
                dataclasses.replace(part, content=f"{part.content}\n\n{hint}")
            )
            added = True
        else:
            new_parts.append(part)
    return dataclasses.replace(msg, content=tuple(new_parts))


async def _error_result(queue_id: str, parent_id: int, message: str) -> Message:
    """Create an error tool result for the dispatch path."""
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            TextMessage(message, "text/x-error"),
        ),
        "multipart/x-tool-result",
        parent_id=parent_id,
    )
