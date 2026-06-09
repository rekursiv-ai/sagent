"""Tripwire: detect drift between claude-written and materializer-written JSONL.

Two entry points:

- :func:`structural_diff` — pure function that compares two JSONL
  payloads (either as paths or as already-parsed entry lists),
  ignoring the volatile fields documented in ``format_spec.md``
  (``timestamp``, ``requestId``, ``uuid``, ``parentUuid``,
  ``message.id``, ``message.usage``, etc.). Returns a list of
  ``DiffFinding`` entries; an empty list means "no drift, safe to
  enable materialization".

- :func:`run_canary_against_live_cli` / :func:`arun_canary_against_live_cli`
  — the v2.1-β startup probe. Spawns a 1-turn ``claude --print``
  against a fresh canary session, schema-checks every JSONL entry
  claude wrote, round-trips it through the materializer, and reports
  a :class:`CanaryResult` with ``is_safe`` + findings. ``serve.py``
  calls the async variant before agent build-up and clears
  ``SAGENT_CLI_OWN_SESSION`` if the verdict is dirty.

The diff is structural, not byte-level — claude splits each content
block into its own JSONL entry (thinking → own line, then tool_use
→ own line), while the materializer coalesces them into a single
assistant entry per ``AssistantMessage``. Both are valid for
``--resume``; the diff normalizes both sides to the same canonical
linearized-message form (the result of running
``parse_jsonl_to_messages`` on each side) and compares THAT.

That same comparison runs in the round-trip test
``test_real_claude_jsonl_round_trip`` — this module just packages it
behind a structured-verdict API so the boot path can act on the
result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import asyncio
import contextlib
import json
import logging
import os
import shutil
import threading
import uuid as _uuid

from sagent.providers.anthropic_cli_session.materializer import (
    materialize_session,
    session_jsonl_path,
)
from sagent.providers.anthropic_cli_session.parser import (
    iter_jsonl,
    parse_jsonl_to_messages,
)
from sagent.types.model import ModelContextEvent, ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


logger = logging.getLogger(__name__)


# Schema check: every chain-bearing entry must carry these fields.
# Pinned against CLI 2.1.168 -- see format_spec.md. Missing fields
# would make ``--resume`` reject the file or render an unparseable
# entry, so we flag them as drift.
_REQUIRED_FIELDS_CHAIN = ("type", "uuid", "timestamp", "sessionId", "version")
_REQUIRED_FIELDS_USER = ("message",)
_REQUIRED_FIELDS_ASSISTANT = ("message", "requestId")


# Entry types we accept on a claude-written canary JSONL. Anything
# else is an unrecognized type and signals format drift -- the
# materializer + parser will either drop it (lossy round-trip) or
# silently produce wrong output downstream.
_KNOWN_ENTRY_TYPES = frozenset(
    {
        # Chain-bearing
        "user",
        "assistant",
        "summary",
        # Sidecar (parser drops these; format_spec.md documents them)
        "system",
        "attachment",
        "custom-title",
        "agent-name",
        "mode",
        "permission-mode",
        "last-prompt",
        "file-history-snapshot",
        "queue-operation",
        "ai-title",
        "agent-setting",
    }
)


# Default canary timeout. A real ``claude --print`` "ping" round-trip
# typically completes in 2-5 seconds against haiku; 30s gives generous
# headroom for cold-cache + first-load. Slower than this means the
# CLI is in trouble and the operator should know.
_DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class DiffFinding:
    """One structural difference between two JSONL message lists.

    ``location`` describes WHERE the drift was found (a tape index,
    a field name); ``detail`` describes WHAT differs in a way the
    operator can read in a log line.
    """

    location: str
    detail: str


def structural_diff(
    left: Sequence[object] | Path,
    right: Sequence[object] | Path,
) -> list[DiffFinding]:
    """Compare two linearized message lists structurally.

    Both arguments may be either:

    - a ``Path`` to a JSONL file (will be parsed via
      :func:`parse_jsonl_to_messages`), or
    - an already-parsed list of ``UserMessage``/``AssistantMessage``/
      ``ToolResult``/``AgentSendMessage`` instances.

    The diff ignores volatile fields (timestamps, request IDs,
    UUIDs, model.id, model.usage, thinking signatures) per
    ``format_spec.md``. Returns ``[]`` when the two sides agree on
    the meaningful content; otherwise a non-empty list of
    findings, one per differing position.
    """
    left_msgs = _ensure_parsed(left)
    right_msgs = _ensure_parsed(right)

    findings: list[DiffFinding] = []

    if len(left_msgs) != len(right_msgs):
        findings.append(
            DiffFinding(
                location="<top-level>",
                detail=(
                    f"length mismatch: left has {len(left_msgs)} entries, "
                    f"right has {len(right_msgs)}"
                ),
            )
        )

    common = min(len(left_msgs), len(right_msgs))
    for i in range(common):
        findings.extend(_compare_one(i, left_msgs[i], right_msgs[i]))

    return findings


def is_safe_to_enable(findings: Sequence[DiffFinding]) -> bool:
    """The verdict the boot path acts on: ``True`` iff no findings."""
    return len(findings) == 0


def _ensure_parsed(source: Sequence[object] | Path) -> list[object]:
    if isinstance(source, Path):
        return parse_jsonl_to_messages(source)
    return list(source)


def _compare_one(i: int, a: object, b: object) -> list[DiffFinding]:
    """Compare two messages at the same tape position."""
    if type(a) is not type(b):
        return [
            DiffFinding(
                location=f"[{i}]",
                detail=f"type mismatch: {type(a).__name__} vs {type(b).__name__}",
            )
        ]

    if isinstance(a, UserMessage) and isinstance(b, UserMessage):
        return _compare_text(i, a.text, b.text)
    if isinstance(a, AssistantMessage) and isinstance(b, AssistantMessage):
        return _compare_assistant(i, a, b)
    if isinstance(a, ToolResult) and isinstance(b, ToolResult):
        return _compare_tool_result(i, a, b)
    # Unknown type for either side — surface as a finding rather than
    # asserting, so the operator gets a useful log line.
    return [
        DiffFinding(
            location=f"[{i}]",
            detail=f"unhandled message type {type(a).__name__}",
        )
    ]


def _compare_text(i: int, a: str, b: str) -> list[DiffFinding]:
    if a == b:
        return []
    return [
        DiffFinding(
            location=f"[{i}].text",
            detail=f"text differs: {a!r} vs {b!r}",
        )
    ]


def _compare_assistant(
    i: int, a: AssistantMessage, b: AssistantMessage
) -> list[DiffFinding]:
    out: list[DiffFinding] = []
    if a.text != b.text:
        out.append(
            DiffFinding(
                location=f"[{i}].text",
                detail=f"text differs: {a.text!r} vs {b.text!r}",
            )
        )
    if len(a.tool_calls) != len(b.tool_calls):
        out.append(
            DiffFinding(
                location=f"[{i}].tool_calls",
                detail=(
                    f"tool_call count differs: {len(a.tool_calls)} vs "
                    f"{len(b.tool_calls)}"
                ),
            )
        )
    for j, (tca, tcb) in enumerate(zip(a.tool_calls, b.tool_calls, strict=False)):
        out.extend(_compare_tool_call(i, j, tca, tcb))
    # thinking_blocks are intentionally NOT compared:
    # - signatures are opaque + provider-minted (volatile field)
    # - claude may split each thinking block to its own JSONL line
    #   while we coalesce; round-trip preserves the AssistantMessage
    #   text + tool_calls but loses thinking-block grouping fidelity.
    return out


def _compare_tool_call(i: int, j: int, a: ToolCall, b: ToolCall) -> list[DiffFinding]:
    out: list[DiffFinding] = []
    if a.id != b.id:
        out.append(
            DiffFinding(
                location=f"[{i}].tool_calls[{j}].id",
                detail=f"id differs: {a.id!r} vs {b.id!r}",
            )
        )
    if a.name != b.name:
        out.append(
            DiffFinding(
                location=f"[{i}].tool_calls[{j}].name",
                detail=f"name differs: {a.name!r} vs {b.name!r}",
            )
        )
    if dict(a.args) != dict(b.args):
        out.append(
            DiffFinding(
                location=f"[{i}].tool_calls[{j}].args",
                detail=f"args differ: {dict(a.args)!r} vs {dict(b.args)!r}",
            )
        )
    return out


def _compare_tool_result(i: int, a: ToolResult, b: ToolResult) -> list[DiffFinding]:
    out: list[DiffFinding] = []
    if a.call_id != b.call_id:
        out.append(
            DiffFinding(
                location=f"[{i}].call_id",
                detail=f"call_id differs: {a.call_id!r} vs {b.call_id!r}",
            )
        )
    if a.content != b.content:
        out.append(
            DiffFinding(
                location=f"[{i}].content",
                detail=f"content differs: {a.content!r} vs {b.content!r}",
            )
        )
    if a.is_error != b.is_error:
        out.append(
            DiffFinding(
                location=f"[{i}].is_error",
                detail=f"is_error differs: {a.is_error} vs {b.is_error}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Live canary runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryResult:
    """Verdict + diagnostics from one canary run.

    ``is_safe`` is the gate the boot path checks before flipping
    ``SAGENT_CLI_OWN_SESSION=1`` for the rest of the process. The
    findings list explains why if it's False; the operator log shows
    them verbatim.
    """

    is_safe: bool
    findings: list[DiffFinding]
    claude_jsonl_path: Path | None
    """Where claude wrote the canary session, before cleanup."""


def run_canary_against_live_cli(
    *,
    session_id: str | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    model: str = "claude-haiku-4-5",
    prompt: str = "ping",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    cleanup: bool = True,
) -> CanaryResult:
    """Run a 1-turn canary through ``claude --print`` and verdict the JSONL.

    Steps:

    1. Mint a fresh canary session UUID (never collides with a real
       role's session).
    2. Spawn ``claude --print --input-format stream-json
       --session-id <uuid>`` with a tiny prompt.
    3. Read the JSONL claude wrote to
       ``<home>/.claude/projects/-<encoded-cwd>/<uuid>.jsonl``.
    4. Schema-check: every entry's ``type`` is in
       :data:`_KNOWN_ENTRY_TYPES`, every chain-bearing entry has the
       required fields.
    5. Round-trip check: parse claude's JSONL → re-materialize → reparse
       → ``structural_diff`` vs the parsed original. Catches lossy
       mappings.
    6. Cleanup: delete the canary JSONL (unless ``cleanup=False``).

    Returns a :class:`CanaryResult` with the verdict + findings + path.
    The boot path treats ``not result.is_safe`` as "do not flip
    ``materialize_session=True`` for this boot".

    Failure modes that map to ``is_safe=False`` rather than raising:

    - ``claude`` CLI not on PATH
    - Subprocess timeout
    - Subprocess non-zero exit
    - Empty / missing JSONL after the spawn
    - Any of the above happens silently; one finding per failure.

    Errors that DO raise: programming bugs (TypeError, ValueError on
    bad args). I/O errors during cleanup are swallowed.
    """
    if session_id is None:
        # Use a fresh UUIDv4 so the canary session is guaranteed not
        # to collide with any real role's UUIDv5-derived session.
        session_id = str(_uuid.uuid4())
    if cwd is None:
        cwd = Path.cwd()
    if home is None:
        home = Path(os.environ.get("HOME", "~")).expanduser()

    # Pick the right entry depending on whether we're already inside
    # an event loop. The boot path calls ``await
    # arun_canary_against_live_cli(...)`` directly so it never gets
    # here in production; this sync entry exists for tests and ad-hoc
    # use.
    try:
        return asyncio.run(
            _run_canary_async(
                session_id=session_id,
                cwd=cwd,
                home=home,
                model=model,
                prompt=prompt,
                timeout_s=timeout_s,
                cleanup=cleanup,
            )
        )
    except RuntimeError as exc:
        # ``asyncio.run`` refuses to nest. Fall back to a thread that
        # owns its own loop.
        if "already running" in str(exc).lower():
            return _run_canary_from_inside_loop(
                session_id=session_id,
                cwd=cwd,
                home=home,
                model=model,
                prompt=prompt,
                timeout_s=timeout_s,
                cleanup=cleanup,
            )
        raise


async def arun_canary_against_live_cli(
    *,
    session_id: str | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    model: str = "claude-haiku-4-5",
    prompt: str = "ping",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    cleanup: bool = True,
) -> CanaryResult:
    """Async entry for callers already inside an event loop.

    Same behaviour as :func:`run_canary_against_live_cli`, awaited
    directly instead of bouncing through ``asyncio.run``. Use this
    from ``serve.py:_amain`` (which is already async) to avoid the
    thread-fallback path.
    """
    if session_id is None:
        session_id = str(_uuid.uuid4())
    if cwd is None:
        cwd = Path.cwd()
    if home is None:
        home = Path(  # noqa: ASYNC240 -- env lookup + string ops, no disk I/O
            os.environ.get("HOME", "~")
        ).expanduser()
    return await _run_canary_async(
        session_id=session_id,
        cwd=cwd,
        home=home,
        model=model,
        prompt=prompt,
        timeout_s=timeout_s,
        cleanup=cleanup,
    )


def _run_canary_from_inside_loop(
    *,
    session_id: str,
    cwd: Path,
    home: Path,
    model: str,
    prompt: str,
    timeout_s: float,
    cleanup: bool,
) -> CanaryResult:
    """Fallback for callers that already own an event loop.

    Spins up a thread to run the canary in its own loop. The boot
    path doesn't use this (it calls
    :func:`arun_canary_against_live_cli` directly); this exists for
    tests and ad-hoc sync callers that happen to be inside a loop.
    """
    result_container: list[CanaryResult] = []
    error_container: list[BaseException] = []

    def _runner() -> None:
        try:
            result_container.append(
                asyncio.run(
                    _run_canary_async(
                        session_id=session_id,
                        cwd=cwd,
                        home=home,
                        model=model,
                        prompt=prompt,
                        timeout_s=timeout_s,
                        cleanup=cleanup,
                    )
                )
            )
        except BaseException as exc:  # noqa: BLE001 -- propagate to main thread
            error_container.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_container:
        raise error_container[0]
    return result_container[0]


async def _run_canary_async(
    *,
    session_id: str,
    cwd: Path,
    home: Path,
    model: str,
    prompt: str,
    timeout_s: float,
    cleanup: bool,
) -> CanaryResult:
    """The actual async canary body. Factored so sync entry can wrap."""
    expected_path = session_jsonl_path(session_id, cwd=cwd, home=home)

    # Spawn claude
    spawn_findings = await _spawn_canary_claude(
        session_id=session_id,
        cwd=cwd,
        model=model,
        prompt=prompt,
        timeout_s=timeout_s,
    )
    if spawn_findings:
        return CanaryResult(
            is_safe=False, findings=spawn_findings, claude_jsonl_path=None
        )

    # Read what claude wrote
    if not expected_path.exists():
        return CanaryResult(
            is_safe=False,
            findings=[
                DiffFinding(
                    location="<canary>",
                    detail=f"claude --print did not write {expected_path}",
                )
            ],
            claude_jsonl_path=None,
        )

    schema_findings = _schema_check(expected_path)
    roundtrip_findings = _roundtrip_check(
        expected_path,
        session_id=session_id,
        cwd=cwd,
        home=home,
    )

    if cleanup:
        _cleanup_canary_files(expected_path)

    all_findings = schema_findings + roundtrip_findings
    return CanaryResult(
        is_safe=is_safe_to_enable(all_findings),
        findings=all_findings,
        claude_jsonl_path=expected_path,
    )


async def _spawn_canary_claude(
    *,
    session_id: str,
    cwd: Path,
    model: str,
    prompt: str,
    timeout_s: float,
) -> list[DiffFinding]:
    """Spawn ``claude --print`` with the canary prompt. Returns findings on failure."""
    if shutil.which("claude") is None:
        return [
            DiffFinding(
                location="<canary>",
                detail="`claude` CLI is not on PATH; cannot run canary",
            )
        ]

    argv = [
        "claude",
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--system-prompt",
        "You are a canary probe. Reply with one short word.",
        "--session-id",
        session_id,
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--permission-mode",
        "bypassPermissions",
    ]
    stdin_payload = (
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": prompt}},
        )
        + "\n"
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        return [
            DiffFinding(
                location="<canary>",
                detail=f"failed to spawn claude subprocess: {exc}",
            )
        ]

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin_payload.encode("utf-8")),
            timeout=timeout_s,
        )
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        return [
            DiffFinding(
                location="<canary>",
                detail=(
                    f"claude --print canary timed out after {timeout_s}s; "
                    "tripwire treats this as drift"
                ),
            )
        ]

    if proc.returncode != 0:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")[:500]
        return [
            DiffFinding(
                location="<canary>",
                detail=(f"claude --print exited {proc.returncode}: {stderr_text!r}"),
            )
        ]

    # stdout is the stream-json output we don't actually parse here --
    # we read the on-disk JSONL instead. Discard.
    _ = stdout_bytes
    return []


def _schema_check(path: Path) -> list[DiffFinding]:
    """Validate every JSONL entry against the pinned schema.

    Two checks per entry:

    1. Its ``type`` is in :data:`_KNOWN_ENTRY_TYPES`. An unknown
       type means the CLI added a new entry shape our parser will
       silently drop or our materializer can't reproduce.
    2. Chain-bearing entries (``user``, ``assistant``) carry the
       required fields. Missing fields would make ``--resume`` reject
       the file (claude's own loader is strict about ``parentUuid``
       and ``message`` shape).
    """
    findings: list[DiffFinding] = []
    try:
        entries = list(iter_jsonl(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [
            DiffFinding(
                location="<canary>",
                detail=f"failed to read/parse claude JSONL at {path}: {exc}",
            )
        ]

    for i, entry in enumerate(entries):
        t = entry.get("type")
        if t not in _KNOWN_ENTRY_TYPES:
            findings.append(
                DiffFinding(
                    location=f"entry[{i}].type",
                    detail=(
                        f"unknown entry type {t!r}; CLI may have added a new "
                        "shape we don't materialize"
                    ),
                )
            )
            continue
        if t == "user":
            findings.extend(_check_required_fields(i, entry, _REQUIRED_FIELDS_USER))
        elif t == "assistant":
            findings.extend(
                _check_required_fields(i, entry, _REQUIRED_FIELDS_ASSISTANT)
            )
        # Chain fields apply to ``user`` / ``assistant`` / ``summary``;
        # sidecars don't carry them and that's expected.
        if t in ("user", "assistant", "summary"):
            findings.extend(_check_required_fields(i, entry, _REQUIRED_FIELDS_CHAIN))
    return findings


def _check_required_fields(
    i: int, entry: dict[str, object], required: Sequence[str]
) -> list[DiffFinding]:
    out: list[DiffFinding] = []
    out.extend(
        DiffFinding(
            location=f"entry[{i}].{field}",
            detail=f"required field {field!r} missing from claude entry",
        )
        for field in required
        if field not in entry
    )
    return out


def _roundtrip_check(
    claude_path: Path,
    *,
    session_id: str,
    cwd: Path,
    home: Path,
) -> list[DiffFinding]:
    """Parse claude's JSONL, re-materialize, reparse, structural_diff."""
    try:
        original_msgs = parse_jsonl_to_messages(claude_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            DiffFinding(
                location="<canary>",
                detail=f"failed to parse claude JSONL: {exc}",
            )
        ]

    # Materialize to a temp session id under the same home so the path
    # is computed identically. We use a "-canary-mat" suffix on the
    # session id to keep claude's canary file and ours from colliding.
    mat_session_id = f"{session_id}-canary-mat"
    try:
        mat_path, _ = materialize_session(
            ModelRequest(messages=cast(list[ModelContextEvent], original_msgs)),
            session_id=mat_session_id,
            cwd=cwd,
            home=home,
        )
    except (OSError, ValueError, TypeError) as exc:
        return [
            DiffFinding(
                location="<canary>",
                detail=f"materializer raised on claude-parsed messages: {exc}",
            )
        ]

    try:
        roundtripped = parse_jsonl_to_messages(mat_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            DiffFinding(
                location="<canary>",
                detail=f"failed to reparse materialized canary: {exc}",
            )
        ]
    finally:
        with contextlib.suppress(OSError):
            mat_path.unlink()

    return structural_diff(original_msgs, roundtripped)


def _cleanup_canary_files(claude_path: Path) -> None:
    """Best-effort: remove the canary session JSONL.

    Failure is silent because we don't want a transient OSError on
    boot to mask the verdict. The next canary will overwrite via the
    materializer's atomic write anyway (or land on a different UUID).
    """
    try:
        claude_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("canary cleanup: failed to delete %s: %s", claude_path, exc)
