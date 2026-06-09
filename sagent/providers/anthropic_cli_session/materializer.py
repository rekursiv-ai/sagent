"""Render a sagent tape (linearized into a ``ModelRequest``) as a
CLI-shaped session JSONL.

The goal is for ``claude --print --resume <uuid>`` to consume the
materialized file and pick up the conversation exactly as if claude
had written it.

Determinism contract:

- Same ``(session_id, cwd, messages)`` always produces a byte-identical
  file, EXCEPT for ``timestamp`` (which is taken from a caller-provided
  clock and defaults to a fixed epoch in test mode) and ``version``
  (probed from the live CLI).
- UUIDs are minted via UUIDv5 of ``(session_id, tape_index)`` against
  a fixed namespace. The parentUuid chain is linear (one chain, no
  side-chains, no branching).
- Synthetic ``requestId`` / ``message.id`` use the prefix ``mat_`` /
  ``msg_mat_`` so they are visibly materializer output, not claude
  output.

See ``format_spec.md`` next to this file for the wire-format reference,
and the worklog thread ``v2.1-cli-session-materialize`` for design
rationale.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import os
import re
import uuid as _uuid

from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultKind,
    UserMessage,
)


# Fixed UUIDv5 namespace for materializer-minted UUIDs. Pinned so
# golden tests across machines produce identical UUIDs.
_MAT_UUID_NS = _uuid.UUID("c0f3a1d2-7e5b-4c8a-9d6f-1a2b3c4d5e6f")

# Fixed epoch used when ``materialize_session`` is called without a
# ``clock``. Picked far in the past so we never collide with real CLI
# timestamps; the round-trip parser drops the field anyway.
_FIXED_TIMESTAMP = "2026-01-01T00:00:00.000Z"


def session_jsonl_path(session_id: str, *, cwd: Path, home: Path | None = None) -> Path:
    """Compute the on-disk path the CLI uses for this session.

    Mirrors the CLI's encoded-cwd convention: replace every non-
    ``[A-Za-z0-9-]`` character in the absolute cwd with ``-``. Leading
    ``/`` becomes a leading ``-`` (since ``/`` is non-alnum).

    Args:
        session_id: Session UUID (the file stem).
        cwd: Absolute working directory the CLI is spawned in.
        home: Override for ``$HOME`` (test seam).

    Returns:
        Absolute path under ``<home>/.claude/projects/<encoded-cwd>/<uuid>.jsonl``.

    """
    home = home or Path(os.environ.get("HOME", "~")).expanduser()
    encoded = re.sub(r"[^A-Za-z0-9-]", "-", str(cwd))
    return home / ".claude" / "projects" / encoded / f"{session_id}.jsonl"


def materialize_session(
    request: ModelRequest,
    *,
    session_id: str,
    cwd: Path,
    git_branch: str = "HEAD",
    cli_version: str = "0.0.0-materialized",
    timestamp: str = _FIXED_TIMESTAMP,
    home: Path | None = None,
    write: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    """Render ``request.messages`` as a CLI-shaped JSONL and (optionally) write it.

    The write is atomic: serialize to a sibling ``.tmp`` file, fsync,
    rename over the target path. ``--resume`` either sees the prior
    contents or the new contents in full, never a torn write.

    Args:
        request: The same ``ModelRequest`` the provider would consume
            on the wire. Its ``messages`` (a linearized
            ``UserMessage`` / ``AgentSendMessage`` / ``AssistantMessage``
            / ``ToolResult`` sequence — the resolved tape view) is
            the input.
        session_id: Session UUID; becomes the file stem and every
            entry's ``sessionId``.
        cwd: Spawn cwd; encoded into the directory path and copied into
            every entry's ``cwd``.
        git_branch: Reported as every entry's ``gitBranch``. ``"HEAD"``
            for detached.
        cli_version: Reported as every entry's ``version``. Should match
            the live CLI's ``claude --version`` (the provider probes it).
        timestamp: ISO-8601 UTC with ``Z`` suffix, ms precision. Same
            for every entry by design (the materializer doesn't
            reconstruct wall-clock per message; the CLI doesn't need
            it for ``--resume``).
        home: Override for ``$HOME`` (test seam).
        write: When False, build the entry list but don't touch disk.
            Used by tests + the tripwire.

    Returns:
        ``(path, entries)``: the target on-disk path (whether or not
        written) and the list of JSONL entry dicts in order.

    """
    entries = _build_entries(
        request,
        session_id=session_id,
        cwd=cwd,
        git_branch=git_branch,
        cli_version=cli_version,
        timestamp=timestamp,
    )
    path = session_jsonl_path(session_id, cwd=cwd, home=home)
    if write:
        _atomic_write(path, entries)
    return path, entries


# ---------------------------------------------------------------------------
# Building the entry list
# ---------------------------------------------------------------------------


@dataclass
class _Chain:
    """Carrier for the running parentUuid pointer + index counter.

    Lives across the linearization loop so deterministic UUID minting
    sees a monotonic index even when one logical sagent message expands
    to multiple JSONL entries (e.g. a string of tool results getting
    grouped into one user-role line — that still increments the index
    by one).
    """

    session_id: str
    parent_uuid: str | None = None
    index: int = 0

    def mint_uuid(self, kind: str) -> str:
        """Mint a UUIDv5 keyed by ``(session_id, index, kind)``."""
        ident = f"{self.session_id}:{self.index}:{kind}"
        u = _uuid.uuid5(_MAT_UUID_NS, ident)
        self.index += 1
        return str(u)

    def advance(self, uuid: str) -> None:
        self.parent_uuid = uuid


def _build_entries(
    request: ModelRequest,
    *,
    session_id: str,
    cwd: Path,
    git_branch: str,
    cli_version: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    """Linearize ``request.messages`` into CLI JSONL entries.

    Mirrors ``providers/anthropic.py:_build_messages``'s coalescing
    rule: consecutive ``ToolResult`` entries land in a single user-role
    JSONL line whose ``content`` is a list of ``tool_result`` blocks.
    """
    chain = _Chain(session_id=session_id)
    entries: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    base = _BaseFields(
        session_id=session_id,
        cwd=str(cwd),
        git_branch=git_branch,
        cli_version=cli_version,
        timestamp=timestamp,
    )

    for entry in request.messages:
        if isinstance(entry, (UserMessage, AgentSendMessage)):
            _flush_tool_results(entries, pending_tool_results, chain, base)
            blocks = _user_text_blocks(entry)
            if blocks:
                entries.append(_make_user_entry(blocks, chain, base))
        elif isinstance(entry, AssistantMessage):
            _flush_tool_results(entries, pending_tool_results, chain, base)
            blocks = _assistant_blocks(entry)
            if blocks:
                entries.append(_make_assistant_entry(blocks, chain, base, entry))
        elif isinstance(entry, ToolResult):  # pyright: ignore[reportUnnecessaryIsInstance] -- runtime-defensive: tape may carry future event types
            pending_tool_results.append(_tool_result_block(entry))
        # Else: unknown event type — skip silently. The provider's
        # linearizer already drops events that aren't model-visible.

    _flush_tool_results(entries, pending_tool_results, chain, base)
    return entries


# ---------------------------------------------------------------------------
# Per-message-type emitters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BaseFields:
    session_id: str
    cwd: str
    git_branch: str
    cli_version: str
    timestamp: str


def _common_metadata(base: _BaseFields) -> dict[str, Any]:
    """Fields every chain-bearing entry carries."""
    return {
        "sessionId": base.session_id,
        "cwd": base.cwd,
        "gitBranch": base.git_branch,
        "version": base.cli_version,
        "userType": "external",
        "entrypoint": "cli",
        "isSidechain": False,
        "timestamp": base.timestamp,
    }


def _make_user_entry(
    content_blocks: list[dict[str, Any]] | str,
    chain: _Chain,
    base: _BaseFields,
) -> dict[str, Any]:
    """Build a user-role JSONL line.

    ``content_blocks`` is either a plain string (operator text or
    AgentSend text) or a list of ``tool_result`` blocks.
    """
    uuid = chain.mint_uuid("user")
    parent = chain.parent_uuid
    chain.advance(uuid)
    return {
        **_common_metadata(base),
        "type": "user",
        "parentUuid": parent,
        "uuid": uuid,
        "message": {"role": "user", "content": content_blocks},
    }


def _make_assistant_entry(
    content_blocks: list[dict[str, Any]],
    chain: _Chain,
    base: _BaseFields,
    source: AssistantMessage,
) -> dict[str, Any]:
    """Build an assistant-role JSONL line.

    Synthetic ``requestId`` / ``message.id`` use the ``mat_`` prefix so
    they are recognizable as materializer output.
    """
    uuid = chain.mint_uuid("assistant")
    parent = chain.parent_uuid
    chain.advance(uuid)
    # The provider keeps the resolved model id off ``AssistantMessage``;
    # we don't have it at materialization time, and ``--resume`` does
    # not validate it. Use a stable placeholder so golden diffs hold.
    stop_reason = "tool_use" if source.tool_calls else "end_turn"
    return {
        **_common_metadata(base),
        "type": "assistant",
        "parentUuid": parent,
        "uuid": uuid,
        "requestId": f"req_mat_{chain.index - 1:08d}",
        "message": {
            "id": f"msg_mat_{chain.index - 1:08d}",
            "type": "message",
            "role": "assistant",
            "model": "materialized",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _flush_tool_results(
    entries: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    chain: _Chain,
    base: _BaseFields,
) -> None:
    """Emit accumulated tool_result blocks as one user-role entry."""
    if not pending:
        return
    entries.append(_make_user_entry(list(pending), chain, base))
    pending.clear()


def _user_text_blocks(entry: UserMessage | AgentSendMessage) -> Any:
    """Return a string (text-only) or content-block list (text + attachments).

    Attachments are not currently faithfully materialized — the CLI
    uses ``BytesMessage`` for image/PDF payloads and the JSONL embeds
    file references that the operator-side claude resolves locally.
    For now, attachments are dropped with a placeholder text marker so
    the chain remains valid; the tripwire flags this so we know to
    extend it.
    """
    if not entry.text and not entry.attachments:
        return ""
    if not entry.attachments:
        # Plain string is the most common form in real CLI files
        # (operator typing). Match it exactly.
        return entry.text or ""
    # Attachments present: emit a content-block list. Image/PDF
    # passthrough is a forward feature; for now drop a marker.
    blocks: list[dict[str, Any]] = []
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    blocks.extend(
        {
            "type": "text",
            "text": f"[materializer: dropped attachment ({att.descriptor})]",
        }
        for att in entry.attachments
    )
    return blocks


def _assistant_blocks(entry: AssistantMessage) -> list[dict[str, Any]]:
    """Build the content-block list for an AssistantMessage.

    Mirrors ``providers/anthropic.py:_assistant_blocks`` but emits
    plain dicts (no Anthropic SDK types). Orphan thinking blocks
    (signature without body) are elided to keep the API happy on a
    later wire send.

    Note on the thinking-end pad: Anthropic's API rejects an assistant
    *wire* message whose last content block is ``thinking``, but
    claude's CLI WRITES exactly that shape into its session JSONL
    (one assistant entry whose content is ``[thinking]`` only,
    followed by a separate assistant entry for the text). Whatever
    coalescing it does before the next API call happens internally;
    ``--resume`` consumes the unpadded shape fine. Verified against
    live CLI 2.1.168 output 2026-06-09. So we do NOT pad here.
    """
    blocks: list[dict[str, Any]] = [
        dict(tb)
        for tb in entry.thinking_blocks
        if _is_native_thinking(tb) and not _is_orphan_thinking(tb)
    ]
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    blocks.extend(_tool_use_block(tc) for tc in entry.tool_calls)
    return blocks


def _tool_use_block(tc: ToolCall) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tc.id,
        "name": tc.name,
        "input": dict(tc.args),
    }


def _tool_result_block(result: ToolResult) -> dict[str, Any]:
    """Render a single ToolResult as a tool_result content block.

    Always emits ``content`` as a list-of-blocks (the CLI accepts
    either form and the list form round-trips cleanly).
    """
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": [{"type": "text", "text": result.content}],
    }
    if result.is_error or result.kind is ToolResultKind.CANCELLED:
        block["is_error"] = True
    return block


def _is_native_thinking(block: Mapping[str, object]) -> bool:
    """Mirror ``providers/anthropic.py:_is_native_thinking`` minimally."""
    return block.get("type") in ("thinking", "redacted_thinking")


def _is_orphan_thinking(block: Mapping[str, object]) -> bool:
    """Mirror ``providers/anthropic.py:_is_orphan_thinking``."""
    return (
        block.get("type") == "thinking"
        and bool(block.get("signature"))
        and not block.get("thinking")
    )


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    """Serialize entries as NDJSON via temp + rename.

    Survives kill mid-write: an interrupted materialization leaves
    either the prior contents or no temp file at all, never a torn
    target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "\n".join(
        json.dumps(e, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        for e in entries
    )
    if payload:
        payload += "\n"
    # Open with O_CREAT|O_WRONLY|O_TRUNC: explicit truncate so a prior
    # short write doesn't leave tail bytes from the previous content.
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)
