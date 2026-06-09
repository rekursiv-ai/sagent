"""Per-agent runtime trace writer.

Each agent's ``runtime.publish`` is called for every RuntimeEvent — model
call lifecycle (``ModelCallStarted``, ``ModelResponsePartial``,
``ModelResponseComplete``, ``ModelIdle``), inbox arrivals
(``AgentSendMessage``, ``UserMessage``), tool labels (``ToolLabel``),
cohort transitions (``CohortStarted``, ``CohortComplete``), session
saves (``SaveSession``), etc.

Chat/'s parity: in the chat runtime each ``chat consume`` worker
captured its claude subprocess's NDJSON stream into a per-role
``sessions/<role>.trace.jsonl``; the ``GET /api/trace/<role>``
endpoint surfaced it to the debug page. We restore the same surface
here by attaching a ``TraceWriter`` observer to each sagent agent at
build time.

Each line is a single JSON dict with ``_event`` carrying the typed
RuntimeEvent class name plus the event's fields. Non-JSON-serializable
fields (asyncio tasks, MCP bridge handles, etc.) fall through to
``repr()``.

Cheap append per event; one ``fcntl.LOCK_EX`` per write keeps lines
atomic if two agents ever shared the same file (they don't — one file
per agent — but the locking is harmless and matches ``shim.append_record``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dataclasses
import fcntl
import json

# Trace files co-locate with ``main.jsonl`` and the per-role
# ``*.mcp.json`` configs in the plugin's *data* dir (configurable
# via ``SAGENT_DATA_DIR``; see :mod:`mcp_sagent.delivery`). Imported
# at module load — lives in the same Python process as ``serve.py``
# so the env-resolved value reflects the launching process's
# environment.
import sys


_plugin_root = Path(__file__).resolve().parent.parent
if str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))
from mcp_sagent import delivery


_SESSIONS_DIR = delivery.SESSIONS_DIR


def _iso_now() -> str:
    dt = datetime.now(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert sagent event objects to JSON-able primitives.

    Order of attempts:

    1. ``None`` / bool / int / float / str — already JSON-able.
    2. dataclass → ``asdict`` (recurses).
    3. ``Mapping`` → dict with recursively-converted values.
    4. ``list`` / ``tuple`` / ``set`` → list with recursively-converted items.
    5. Bytes → repr (the substring of trace records that contain binary
       data is rare — file uploads etc. — and a clean repr is enough
       for debug rendering).
    6. Asyncio tasks / file handles / functions / anything else with
       no obvious serialization → ``repr(obj)``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _to_jsonable(dataclasses.asdict(obj))
        except TypeError:
            # Some dataclasses contain un-asdict-able fields (e.g. an
            # asyncio.Task). Fall through to field-by-field.
            out: dict[str, Any] = {}
            for f in dataclasses.fields(obj):
                try:
                    out[f.name] = _to_jsonable(getattr(obj, f.name))
                except Exception:
                    out[f.name] = repr(getattr(obj, f.name, None))
            return out
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return repr(obj)
    return repr(obj)


def trace_path_for(role_name: str, *, sessions_dir: Path | str | None = None) -> Path:
    base = Path(sessions_dir) if sessions_dir is not None else _SESSIONS_DIR
    return base / f"{role_name}.trace.jsonl"


class TraceWriter:
    """Observer that serializes RuntimeEvents to a per-role JSONL file."""

    def __init__(self, role_name: str, *, sessions_dir: Path | None = None) -> None:
        self.role_name = role_name
        self.path = trace_path_for(role_name, sessions_dir=sessions_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Counter so the on-disk record carries an ordinal that's stable
        # across restarts of the JSONL reader — ``GET /api/trace`` can
        # use it to dedupe + jump to ``around=N``.
        self._ordinal = self._count_existing()

    def _count_existing(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def __call__(self, event) -> None:
        cls = type(event).__name__
        payload = _to_jsonable(event)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        record = {
            "_event": cls,
            "_ts": _iso_now(),
            "_n": self._ordinal,
            **payload,
        }
        self._ordinal += 1
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def install_on(
    agent, role_name: str, *, sessions_dir: Path | None = None
) -> TraceWriter:
    """Attach a TraceWriter observer to an agent and return it."""
    writer = TraceWriter(role_name, sessions_dir=sessions_dir)
    agent.runtime.observers.append(writer)
    return writer
