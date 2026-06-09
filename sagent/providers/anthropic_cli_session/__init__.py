"""Sagent-owned CLI session JSONL materializer.

See ``format_spec.md`` in this directory for the wire format this
module emits, and the worklog thread ``v2.1-cli-session-materialize``
on the claude-config repo for design rationale.

Public surface:

- :func:`materialize_session`: render a ``ModelRequest`` (the same
  linearized tape view the provider would consume on the wire) as a
  CLI-shaped JSONL file at the operator's HOME-relative session path.
- :func:`session_jsonl_path`: compute the on-disk path for a given
  ``(session_id, cwd)`` pair using claude's encoded-cwd convention.
"""

from sagent.providers.anthropic_cli_session.materializer import (
    materialize_session,
    session_jsonl_path,
)
from sagent.providers.anthropic_cli_session.parser import (
    iter_jsonl,
    parse_jsonl_to_messages,
)
from sagent.providers.anthropic_cli_session.tripwire import (
    CanaryResult,
    DiffFinding,
    arun_canary_against_live_cli,
    is_safe_to_enable,
    run_canary_against_live_cli,
    structural_diff,
)


__all__ = [
    "CanaryResult",
    "DiffFinding",
    "arun_canary_against_live_cli",
    "is_safe_to_enable",
    "iter_jsonl",
    "materialize_session",
    "parse_jsonl_to_messages",
    "run_canary_against_live_cli",
    "session_jsonl_path",
    "structural_diff",
]
