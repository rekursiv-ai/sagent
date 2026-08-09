"""Cross-layer invariants for shortened tool results.

Eight independent sites can shorten a tool result -- the tool's own
limit, ``truncate``, disk persistence, request materialization, the
compactor, and the renderer. Each was written locally and none knows
about the others, so the composition is what breaks: a cap removed in
one layer arms a different layer's failure mode.

These tests assert the two rules that must hold across the whole
pipeline, independent of which layer does the shortening:

1. **Self-describing** -- a shortened result says so. Silence is
   indistinguishable from a complete answer.
2. **Recoverable** -- a tool that can shorten offers a way to reach the
   rest (``offset``), or hands back a path that can be re-read.

They live at package root rather than under ``tools/`` or ``agent/``
because the invariant spans both.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import asyncio
import tempfile

import pytest

from sagent.agent.result_storage import post_process_result
from sagent.request_materialization import (
    ELIDED_TOOL_RESULT_TAG,
    materialize_messages,
)
from sagent.testing import with_fake_agent
from sagent.tools.core import TOOL_RESULT_MAX_CHARS
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


# Budget shape of a mid-size production model (opus-5: 200k window,
# ~3 chars/token), spelled out so the arithmetic in these tests is
# legible rather than imported and opaque.
_PERSIST_THRESHOLD = 150_000
_MESSAGE_BUDGET = 300_000


def _write_lines(path: Path, count: int) -> Path:
    """Write ``count`` numbered source-like lines and return the path."""
    path.write_text(
        "".join(f"{i:06d}\tdef f{i}(): return {i}\n" for i in range(count)),
        encoding="utf-8",
    )
    return path


def test_large_read_is_not_silently_elided(tmp_path: Path) -> None:
    """A big Read must reach the model as content, never as a bare tag.

    ``Read`` bypasses both bounds that protect every other tool: it
    calls ``asyncio.to_thread`` directly rather than ``run_sync`` (so
    ``TOOL_RESULT_MAX_CHARS`` never applies) and it sits in
    ``PERSIST_EXEMPT_TOOLS`` (so disk offload is skipped). The result
    lands in ``materialize_messages`` over-budget and is replaced by
    ``<elided>`` -- total content loss with no notice and no path back.
    """
    big = _write_lines(tmp_path / "big.py", 40_000)
    with with_fake_agent():
        result = asyncio.run(Read().run({"file_path": str(big)}))

    messages = [
        UserMessage(text="read it"),
        AssistantMessage(
            text="", tool_calls=(ToolCall(id="c1", name="Read", args={}),)
        ),
        ToolResult(call_id="c1", content=result.content),
    ]
    materialized = materialize_messages(
        messages, tool_result_budget_chars=_MESSAGE_BUDGET
    )
    tool_results = [m for m in materialized if isinstance(m, ToolResult)]

    assert tool_results, "the Read result was dropped from the request entirely"
    assert tool_results[0].content != ELIDED_TOOL_RESULT_TAG, (
        "a large Read reached the model as a bare elision tag: the file"
        " contents were lost with no notice and no way to recover them"
    )


def test_read_result_stays_within_the_persist_threshold(tmp_path: Path) -> None:
    """Read's own limit must keep one result under the offload threshold.

    ``PERSIST_EXEMPT_TOOLS`` justifies exempting Read on the grounds
    that its "output already bounded by the tool's own internal cap".
    That claim has to be true, or the exemption removes the only
    remaining bound.
    """
    big = _write_lines(tmp_path / "big.py", 40_000)
    with with_fake_agent():
        result = asyncio.run(Read().run({"file_path": str(big)}))
    assert len(result.content) <= _PERSIST_THRESHOLD, (
        f"Read returned {len(result.content):,} chars, over the"
        f" {_PERSIST_THRESHOLD:,}-char persist threshold it is exempt from"
    )


def test_unbounded_grep_stays_within_the_result_cap(tmp_path: Path) -> None:
    """A default grep over a large tree must not overrun the result cap.

    With ``keep_first`` defaulting to unlimited, a wide match set is
    head-truncated by ``truncate`` at the framework boundary -- and
    Grep's ``offset`` cannot recover the tail once the characters are
    gone rather than the rows.
    """
    for i in range(2_000):
        (tmp_path / f"f{i:05d}.txt").write_text("hit line\n" * 40, encoding="utf-8")
    with with_fake_agent():
        result = asyncio.run(
            Grep().run(
                {"pattern": "hit", "path": str(tmp_path), "output_mode": "content"}
            )
        )
    assert len(result.content) <= TOOL_RESULT_MAX_CHARS, (
        f"grep produced {len(result.content):,} chars against a"
        f" {TOOL_RESULT_MAX_CHARS:,} cap; the overflow is cut mid-stream"
    )


@pytest.mark.parametrize("tool", [Glob(), Grep()])
def test_shortening_tools_expose_offset(tool: Glob | Grep) -> None:
    """Any tool that can shorten its output must offer a way back.

    Shortening without pagination is the silent-cut this whole layer
    exists to prevent: the model is told rows were dropped but has no
    argument that would return them.
    """
    schema = tool.directive_schema
    assert isinstance(schema, Mapping)
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    assert "offset" in properties, (
        f"{type(tool).__name__} can shorten its result but exposes no"
        " 'offset', so the omitted rows are unreachable"
    )


def test_persisted_result_carries_a_path_back() -> None:
    """Disk offload is the one lossless shortening path; keep it honest."""
    body = "y" * 3_000_000
    with tempfile.TemporaryDirectory() as tmp:
        out = post_process_result(
            ToolResult(call_id="c1", content=body),
            "Bash",
            session_dir=Path(tmp),
            persist_threshold=_PERSIST_THRESHOLD,
            message_budget_chars=_MESSAGE_BUDGET,
            used_message_chars=0,
        )
    assert "tool-results" in out.content, "persisted result lost its path back"
    assert len(out.content) < len(body)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
