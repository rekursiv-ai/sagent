"""Session serialization, repair, and ToolState reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import dataclasses
import hashlib
import json
import logging
import re
import time

from sagent.agent.dispatch import tc_directive, tc_tool_id
from sagent.custom_types import (
    JsonMessage,
    Message,
    MessageBase,
    Model,
    ModelSpec,
    MultipartMessage,
    TextMessage,
    TokenCount,
)
from sagent.lib.atomic_file import atomic_write
from sagent.lib.compaction import CLEARED
from sagent.lib.descriptors import flat_text, has_error
from sagent.lib.json import (
    MutableJSON,
    MutableJSONValue,
    float_val,
    int_val,
    json_freeze,
    json_unfreeze,
)
from sagent.lib.lazy_import import lazy_import
from sagent.lib.message import (
    get_queue_id,
    response_tool_calls,
    tool_call_message,
)
from sagent.sessions import parse_jsonl
from sagent.tools.core import ToolState


providers_lib = lazy_import("sagent.providers")


logger = logging.getLogger(__name__)

# File extensions whose Read tool result is a marker/rendering,
# not file bytes (image base64, extracted PDF text, notebook cell
# dump). Storing these as "content" on session resume would break
# change-detection diffs on the next model request.
_NON_TEXT_READ_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".pdf", ".ipynb"},
)
_NON_TEXT_READ_PREFIXES = (
    "[image:",
    "[Binary file:",
    "[Non-UTF-8 file:",
    "[PDF:",
)

_RE_SYSTEM_REMINDER = re.compile(
    r"<system-reminder>[\s\S]*?</system-reminder>",
)
_RE_LINE_NUMBER_PREFIX = re.compile(r"^\d+\t")


def restore_timing(msg: Message, data: MutableJSON) -> None:
    """Restore id/parent_id/timestamp from a session record onto ``msg``.

    Args:
      msg: Message to mutate in-place.
      data: Deserialized session record with ``_id``, ``_parent_id``,
          ``_timestamp`` keys.

    """
    for attr, key in (
        ("id", "_id"),
        ("parent_id", "_parent_id"),
        ("timestamp", "_timestamp"),
    ):
        v = data.get(key)
        if isinstance(v, int):
            object.__setattr__(msg, attr, v)


def text_from_msg(msg: Message) -> str:
    """Extract plain text from a Message.

    Args:
      msg: Source message (may be multipart).

    Returns:
      text: Concatenated plain-text content, including errors.

    """
    return flat_text(msg, include_errors=True)


def deserialize_message_legacy(data: MutableJSON) -> Message:
    """Deserialize a Message from the old role-based session format.

    Args:
      data: Dict with ``role`` key (``"user"``, ``"assistant"``, or
          tool result).

    Returns:
      msg: Reconstructed message.

    Raises:
      TypeError: If ``role`` is not a string.

    """
    role = data.get("role", "")
    if not isinstance(role, str):
        raise TypeError(f"Expected role to be str, got {type(role)}")
    if role == "user":
        msg = TextMessage(str(data.get("content", "")), "text/x-user-message")
        restore_timing(msg, data)
        return msg
    if role == "assistant":
        raw_tb = data.get("thinking_blocks", [])
        parts: list[Message] = [
            JsonMessage(
                json_freeze(cast(MutableJSON, tb)),
                "application/x-thinking-structured",
            )
            for tb in cast(list[MutableJSONValue], raw_tb)
            if isinstance(tb, dict)
        ]
        raw_content = data.get("content", "")
        if raw_content:
            parts.append(TextMessage(str(raw_content), "text/plain"))
        for tc in cast(list[MutableJSONValue], data.get("tool_calls", [])):
            if not isinstance(tc, dict):
                continue
            tc_d = cast(MutableJSON, tc)
            tc_id = str(tc_d["id"])
            tc_name = str(tc_d["name"])
            tc_input = tc_d.get("input", {})
            directive = json_freeze(
                cast(MutableJSON, tc_input) if isinstance(tc_input, dict) else {}
            )
            parts.append(tool_call_message(tc_id, tc_name, directive))
        raw_msg_id = data.get("message_id", "")
        all_parts: list[Message] = []
        if raw_msg_id:
            all_parts.append(TextMessage(str(raw_msg_id), "text/x-queue-id"))
        all_parts.extend(parts)
        msg = MultipartMessage(tuple(all_parts), "multipart/x-model-message")
        restore_timing(msg, data)
        return msg
    tool_call_id = str(data["tool_call_id"])
    content_str = str(data.get("content", ""))
    is_error = bool(data.get("is_error", False))
    result_parts: list[Message] = [
        TextMessage(tool_call_id, "text/x-queue-id"),
        TextMessage(content_str, "text/x-error" if is_error else "text/plain"),
        *[
            TextMessage(h, "text/x-hint-tool-use-nudge")
            for h in cast(list[MutableJSONValue], data.get("lint_hints", []))
            if isinstance(h, str)
        ],
    ]
    msg = MultipartMessage(tuple(result_parts), "multipart/x-tool-result")
    restore_timing(msg, data)
    return msg


def load_message(data: MutableJSON) -> Message:
    """Deserialize a Message from any session format (new or legacy role-based).

    Args:
      data: Serialized message dict.

    Returns:
      msg: Reconstructed message.

    """
    if "descriptor" not in data:
        return deserialize_message_legacy(data)
    return MessageBase.deserialize(data)


def read_jsonl(path: Path, label: str) -> list[MutableJSON]:
    """Read a legacy JSONL file, skipping malformed or non-object lines.

    Args:
      path: File to read.
      label: Human-readable label for the warning log on OS failure.

    Returns:
      records: Parsed JSON dicts, in file order. Non-dict lines and
          malformed JSON are skipped. Empty on any OSError.

    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read legacy %s.", label)
        return []
    return parse_jsonl(raw_text)


def normalize_tool_history(
    messages: list[Message],
    *,
    synthesize_missing: bool,
) -> list[Message]:
    """Normalize tool call/result ordering in live history.

    Anthropic's API requires every ``tool_use`` block in an assistant
    message to be immediately followed (in the next user message) by
    matching ``tool_result`` blocks for every id. When a request is
    cancelled mid-tool (Ctrl+C during tool execution), the assistant
    message with the tool_use is saved to the session but the
    tool_result isn't - resuming the session fails with a 400
    ``tool_use ids were found without tool_result blocks``.

    Drops unexpected tool_result messages from live history. When
    ``synthesize_missing`` is true, also inserts ``[interrupted]``
    results for live tool calls missing a result. The append-only
    session log remains the source for forensic recovery.

    Args:
      messages: Conversation history to repair.
      synthesize_missing: Whether to insert interrupted results for
          tool calls with no matching result.

    Returns:
      normalized: New list with orphan results dropped and, optionally,
          synthetic results injected.

    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.descriptor == "multipart/x-model-message":
            out.append(msg)
            tc_msgs = response_tool_calls(msg)
            if tc_msgs:
                expected = {get_queue_id(tc) for tc in tc_msgs}
                seen: set[str] = set()
                j = i + 1
                while (
                    j < len(messages)
                    and messages[j].descriptor == "multipart/x-tool-result"
                ):
                    qid = get_queue_id(messages[j])
                    if qid in expected and qid not in seen:
                        seen.add(qid)
                        out.append(messages[j])
                    j += 1
                if synthesize_missing:
                    out.extend(
                        MultipartMessage(
                            (
                                TextMessage(get_queue_id(tc), "text/x-queue-id"),
                                TextMessage("[interrupted]", "text/x-error"),
                            ),
                            "multipart/x-tool-result",
                        )
                        for tc in tc_msgs
                        if get_queue_id(tc) not in seen
                    )
                i = j
            else:
                i += 1
        elif msg.descriptor == "multipart/x-tool-result":
            i += 1
        else:
            out.append(msg)
            i += 1
    return out


def repair_dangling_tool_calls(messages: list[Message]) -> list[Message]:
    """Repair provider-request history by normalizing tool results."""
    return normalize_tool_history(messages, synthesize_missing=True)


def strip_read_result(content: str) -> str:
    r"""Reconstruct file content from a Read tool result.

    Strips <system-reminder> blocks and the ``N\t`` line-number
    prefix emitted by the Read tool so the rebuilt content cache
    matches disk bytes for diffing. Uses ``splitlines(keepends=True)``
    so trailing newlines are preserved - otherwise every diff against
    a file that ends in ``\n`` would show a phantom last-line change.

    Args:
      content: Raw Read tool output with line-number prefixes.

    Returns:
      text: Reconstructed file content.

    """
    cleaned = _RE_SYSTEM_REMINDER.sub("", content)
    lines = cleaned.splitlines(keepends=True)
    return "".join(_RE_LINE_NUMBER_PREFIX.sub("", line) for line in lines)


def rebuild_tool_state_from_messages(
    messages: list[Message],
    state: ToolState,
) -> None:
    """Rehydrate ToolState read tracking from the message history.

    Walks Read/Write/Edit tool uses and their results so that,
    after resuming a session, ``enforce_read``, ``check_stale``,
    and ``consume_changed_files`` have the content snapshots they
    need - without persisting a separate content cache.

    Args:
      messages: Conversation history to scan.
      state: ToolState to populate with file-content snapshots.

    """
    # Pass 1: collect tool uses keyed by id.
    # ``read_full_uses`` participates in content-cache hydration;
    # ``read_partial_uses`` only marks the path as read (we can't
    # reconstruct full-file bytes from a range). Both feed
    # ``enforce_read`` so resuming with partial reads still lets the
    # model Edit/Write that file without a spurious "not yet read".
    read_full_uses: dict[str, str] = {}
    read_partial_uses: dict[str, str] = {}
    write_uses: dict[str, tuple[str, str]] = {}
    edit_paths: dict[str, str] = {}
    for msg in messages:
        if msg.descriptor != "multipart/x-model-message":
            continue
        for tc in response_tool_calls(msg):
            directive = tc_directive(tc)
            fp_raw = directive.get("file_path", "")
            if not isinstance(fp_raw, str) or not fp_raw:
                continue
            tc_id = tc_tool_id(tc)
            qid = get_queue_id(tc)
            if tc_id == "application/x-tool-read":
                offset = directive.get("offset", 1)
                limit = directive.get("limit", 2000)
                is_full = offset in (1, 0, None) and limit in (2000, 0, None)
                if is_full:
                    read_full_uses[qid] = fp_raw
                else:
                    read_partial_uses[qid] = fp_raw
            elif tc_id == "application/x-tool-write":
                content = directive.get("content", "")
                if isinstance(content, str):
                    write_uses[qid] = (fp_raw, content)
            elif tc_id == "application/x-tool-edit":
                edit_paths[qid] = fp_raw

    # Pass 2: match results, populate the content cache.
    last_bash_cwd: str = ""
    for msg in messages:
        if msg.descriptor != "multipart/x-tool-result":
            continue
        cwd = _bash_cwd_from_result(msg)
        if cwd:
            last_bash_cwd = cwd
        if has_error(msg):
            continue
        tid = get_queue_id(msg)
        text = flat_text(msg)
        # Prefer the message's recorded file-stat (real mtime + path);
        # fall back to msg.timestamp/1e9 for sessions written before
        # ``application/x-file-stat`` parts existed.
        # backwards-compat: msg.timestamp is "later than real mtime"
        # so consume_changed_files always falls through to the content
        # diff path; equal content → empty diff → no false reminder.
        stat_path, stat_mtime, stat_sha = _file_stat_or_fallback(msg)
        if tid in read_full_uses:
            # Skip cleared/stub results - the earlier real Read
            # already populated the cache for this path.
            if text == CLEARED or text.startswith("[File unchanged"):
                continue
            fp = stat_path or read_full_uses[tid]
            # Skip content extraction for non-text reads (images,
            # PDFs, notebooks, binary, non-UTF-8). The result is a
            # marker or rendering, not file bytes - storing it as
            # "content" would corrupt change-detection diffs on the
            # next model request. We filter explicitly since Read returns a
            # string in all cases (including images, PDFs, etc.).
            suffix = Path(fp).suffix.lower()
            if suffix in _NON_TEXT_READ_EXTS or text.startswith(
                _NON_TEXT_READ_PREFIXES,
            ):
                state.mark_read(fp, mtime=stat_mtime)
            else:
                state.mark_read(
                    fp,
                    content=strip_read_result(text),
                    mtime=stat_mtime,
                )
        elif tid in read_partial_uses:
            # Partial reads: can't reconstruct full content, but the
            # path still counts as "read" for Edit/Write's invariant.
            state.mark_read(stat_path or read_partial_uses[tid], mtime=stat_mtime)
        elif tid in write_uses:
            fp_use, content = write_uses[tid]
            state.mark_read(stat_path or fp_use, content=content, mtime=stat_mtime)
        elif tid in edit_paths:
            fp = stat_path or edit_paths[tid]
            # Edit's directive carries old/new strings only — no post-edit
            # content. With ``stat_sha`` (new sessions) we verify disk
            # integrity: matching hash → disk IS the post-edit baseline,
            # cache it; mismatched → an external mutation between Edit
            # and resume, leave content empty so the next request's
            # ``consume_changed_files`` reports the full diff. Without
            # ``stat_sha`` (pre-file-stat sessions) we fall back to the
            # original optimistic behavior — cache current disk content
            # — to preserve pre-commit semantics on legacy data.
            try:
                disk = Path(fp).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                state.mark_read(fp, mtime=stat_mtime)
                continue
            if not stat_sha or (
                hashlib.sha256(disk.encode("utf-8")).hexdigest() == stat_sha
            ):
                state.mark_read(fp, content=disk, mtime=stat_mtime)
            else:
                state.mark_read(fp, mtime=stat_mtime)
    if last_bash_cwd:
        state.bash_cwd = last_bash_cwd


def _file_stat_or_fallback(msg: Message) -> tuple[str, float | None, str]:
    """Return (abs_path, mtime, sha256) from an x-file-stat part, or fallback.

    Returns ``("", None, "")`` when the part is absent so pre-file-stat
    sessions still load with the original semantics: ``mark_read`` falls
    through to ``stat()`` at rebuild time, matching pre-commit behavior
    across every code path. Using a synthetic ``msg.timestamp`` fallback
    here would trigger noisy full-file diffs on first resume request
    for paths without cached content (Edit, partial Read, binary Read),
    so we deliberately do not.
    """
    if not isinstance(msg, MultipartMessage):
        return "", None, ""
    for p in msg.content:
        if p.descriptor == "application/x-file-stat":
            data = cast(Mapping[str, object], p.content)
            path = str(data.get("path") or "")
            raw_mtime = data.get("mtime")
            mtime = float(raw_mtime) if isinstance(raw_mtime, (int, float)) else None
            sha = str(data.get("sha256") or "")
            return path, mtime, sha
    return "", None, ""  # backwards-compat: pre-file-stat sessions


def _bash_cwd_from_result(msg: Message) -> str:
    """Return the cwd from an ``application/x-bash-state`` part, or ''.

    Used to replay bash cwd from history rather than relying solely on
    SessionMeta's snapshot. Returns empty string when no part is present
    (pre-bash-state sessions).
    """
    if not isinstance(msg, MultipartMessage):
        return ""
    for p in msg.content:
        if p.descriptor == "application/x-bash-state":
            data = cast(Mapping[str, object], p.content)
            cwd = data.get("cwd")
            if isinstance(cwd, str) and cwd:
                return cwd
    return ""


# -- Session persistence (called from Agent) -------------------------------


def append_session(
    path: Path,
    *,
    meta: Mapping[str, object] | None = None,
    messages_delta: list[Message] | None = None,
    events: list[dict[str, object]] | None = None,
    clear: bool = False,
) -> None:
    """Append delta records to ``session.jsonl``.

    The session file is append-only. Multiple ``kind: meta`` lines may
    accumulate -- the loader takes the latest. Set ``clear=True`` to
    write a ``kind: clear`` barrier first; the loader treats this as a
    reset, dropping all prior messages from the live history view (file
    bytes preserved for forensics). Use ``clear=True`` whenever history
    was wholesale-replaced (``/clear``, compaction, uncompact, or repair
    inserting messages mid-history).

    Args:
      path: Destination file path; created if missing.
      meta: Optional session metadata dict (latest meta line wins on load).
      messages_delta: Conversation messages to append (typically the
          tail past ``Agent._persisted_idx``).
      events: Structured event entries to append.
      clear: True to emit a ``kind: clear`` barrier line before any
          other records in this batch.

    """
    parts: list[str] = []
    if clear:
        parts.append(json.dumps({"kind": "clear", "_timestamp": time.time_ns()}))
    if meta is not None:
        parts.append(json.dumps({"kind": "meta", **meta}))
    parts.extend(
        json.dumps({"kind": "message", **msg.serialize()})
        for msg in messages_delta or ()
    )
    parts.extend(
        json.dumps({"kind": "event", **entry}, default=str) for entry in events or ()
    )
    if not parts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for p in parts:
            _ = f.write(p + "\n")


def load_session(
    session_dir: Path,
    defaults: dict[str, object],
) -> tuple[MutableJSON, list[Message]] | None:
    """Load a session from disk, auto-migrating legacy layouts.

    Args:
      session_dir: Directory containing session files.
      defaults: Fallback values for migration (session_id,
          model_id, name, bash_cwd).

    Returns:
      result: ``(meta, messages)`` or ``None`` if no session exists.

    """
    session_file = session_dir / "session.jsonl"
    legacy_msgs = session_dir / "messages.jsonl"
    legacy_state = session_dir / "state.json"

    if not session_file.exists() and (legacy_msgs.exists() or legacy_state.exists()):
        _migrate_legacy_session(session_dir, defaults)

    if not session_file.exists():
        return None

    meta: MutableJSON = {}
    messages: list[Message] = []
    corrupt_preserved = False
    try:
        with session_file.open(encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    kind = record.get("kind")
                    if kind == "meta":
                        meta = record
                    elif kind == "message":
                        messages.append(load_message(record))
                    elif kind == "clear":
                        # Barrier: drop all prior messages from the live
                        # view. Bytes remain in the file for forensics.
                        messages = []
                except (json.JSONDecodeError, KeyError, TypeError):
                    if not corrupt_preserved:
                        _preserve_corrupt_session(session_file)
                        corrupt_preserved = True
                    logger.warning(
                        "Skipping corrupt session line %s in %s.",
                        line_num,
                        session_file,
                    )
    except OSError:
        logger.warning("Could not read session file, starting fresh.")
        return None
    return meta, normalize_tool_history(messages, synthesize_missing=False)


def _preserve_corrupt_session(session_file: Path) -> None:
    """Copy corrupt session bytes aside before recovery continues."""
    backup = session_file.with_name(f"{session_file.name}.corrupt-{time.time_ns()}")
    try:
        backup.write_bytes(session_file.read_bytes())
    except OSError:
        logger.exception("Could not preserve corrupt session file %s.", session_file)


def _migrate_legacy_session(
    session_dir: Path,
    defaults: dict[str, object],
) -> None:
    """Convert old 3-file layout to single session.jsonl."""
    legacy_msgs = session_dir / "messages.jsonl"
    legacy_state = session_dir / "state.json"
    legacy_events = session_dir / "events.jsonl"
    session_file = session_dir / "session.jsonl"
    logger.info(
        "Migrating legacy session layout in %s",
        session_dir,
    )

    state: MutableJSON = {}
    if legacy_state.exists():
        try:
            state = json.loads(
                legacy_state.read_text(encoding="utf-8"),
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Legacy state.json corrupt; skipping.")

    meta = {
        "kind": "meta",
        "session_id": state.get(
            "session_id",
            defaults.get("session_id"),
        ),
        "model_id": state.get(
            "model_id",
            defaults.get("model_id"),
        ),
        "name": state.get("name", defaults.get("name")),
        "tokens": {
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0),
            "cache_creation_tokens": state.get(
                "cache_creation_tokens",
                0,
            ),
            "cache_read_tokens": state.get("cache_read_tokens", 0),
        },
        "total_cost_usd": state.get("total_cost_usd", 0.0),
        "num_tool_call_rounds": state.get("turn_count", 0),
        "bash_cwd": state.get(
            "bash_cwd",
            defaults.get("bash_cwd"),
        ),
        "version": 2,
        "migrated_from": "legacy-v1",
    }
    with atomic_write(session_file) as f:
        _ = f.write(json.dumps(meta) + "\n")
        if legacy_msgs.exists():
            for msg_dict in read_jsonl(
                legacy_msgs,
                "messages.jsonl",
            ):
                msg = load_message(msg_dict)
                record = {
                    "kind": "message",
                    **msg.serialize(),
                }
                _ = f.write(json.dumps(record) + "\n")
        if legacy_events.exists():
            for entry in read_jsonl(
                legacy_events,
                "events.jsonl",
            ):
                _ = f.write(json.dumps({"kind": "event", **entry}) + "\n")
    for old in (legacy_msgs, legacy_events, legacy_state):
        if old.exists():
            old.rename(old.with_suffix(old.suffix + ".bak"))


# -- Session metadata -------------------------------------------------------


def parse_summary_pointers(raw: object) -> list[tuple[str, str]]:
    """Parse summary pointers from a deserialized JSON value.

    Args:
      raw: Expected ``list[list[str]]`` from session metadata.

    Returns:
      pointers: List of ``(path, topic)`` pairs.

    """
    if not isinstance(raw, list):
        return []
    return [(str(p[0]), str(p[1])) for p in cast(list[list[str]], raw) if len(p) >= 2]


@dataclasses.dataclass(slots=True, kw_only=True)
class SessionMeta:
    """Schema for session persistence -- single source of truth for save/load."""

    session_id: str = ""
    model_id: str = ""
    provider: str = ""
    auth: str = ""
    account: str = ""
    name: str = ""
    status: str = ""
    tokens: TokenCount = dataclasses.field(default_factory=TokenCount)
    total_cost_usd: float = 0.0
    num_tool_call_rounds: int = 0
    compact_count: int = 0
    summary_pointers: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    bash_cwd: str = ""
    total_active_elapsed_seconds: float = 0.0
    version: int = 2

    def serialize(self) -> MutableJSON:
        """Serialize to the canonical session metadata shape."""
        return json_unfreeze(dataclasses.asdict(self))

    @classmethod
    def deserialize(cls, d: MutableJSON) -> SessionMeta:
        """Construct from a deserialized session record.

        Args:
          d: Dict with session metadata keys.

        Returns:
          meta: Populated instance.

        """
        raw_tokens = d.get("tokens")
        tokens: Mapping[str, object] = (
            cast(Mapping[str, object], raw_tokens)
            if isinstance(raw_tokens, Mapping)
            else {}
        )
        return cls(
            session_id=str(d.get("session_id") or ""),
            model_id=str(d.get("model_id") or ""),
            provider=str(d.get("provider") or ""),
            auth=str(d.get("auth") or ""),
            account=str(d.get("account") or ""),
            name=str(d.get("name") or ""),
            status=str(d.get("status") or d.get("title") or ""),
            tokens=TokenCount(
                input_tokens=int_val(tokens.get("input_tokens"), 0),
                output_tokens=int_val(tokens.get("output_tokens"), 0),
                cache_creation_tokens=int_val(tokens.get("cache_creation_tokens"), 0),
                cache_read_tokens=int_val(tokens.get("cache_read_tokens"), 0),
            ),
            total_cost_usd=float_val(d.get("total_cost_usd"), 0),
            num_tool_call_rounds=int_val(d.get("num_tool_call_rounds"), 0),
            compact_count=int_val(d.get("compact_count"), 0),
            summary_pointers=parse_summary_pointers(d.get("summary_pointers")),
            bash_cwd=str(d.get("bash_cwd") or ""),
            total_active_elapsed_seconds=float_val(
                d.get("total_active_elapsed_seconds"), 0
            ),
            version=int_val(d.get("version"), 2),
        )


def restore_model(
    meta: SessionMeta,
) -> tuple[Model, ModelSpec] | None:
    """Rebuild model + spec from persisted session metadata.

    Args:
      meta: Session metadata with provider and model_id.

    Returns:
      result: ``(model, spec)`` on success, ``None`` on failure.

    """
    try:
        provider = providers_lib.build_provider(
            meta.provider, meta.auth, account=meta.account or None
        )
        model = provider.model(meta.model_id)
        spec = ModelSpec(
            provider=meta.provider,
            auth=meta.auth,
            model_id=model.model_id,
            account=meta.account or None,
        )
        logger.info("Restored model %s/%s", meta.provider, meta.model_id)
        return model, spec
    except (AttributeError, RuntimeError, ValueError):
        logger.warning(
            "Failed to restore model %s/%s; keeping default",
            meta.provider,
            meta.model_id,
        )
        return None
