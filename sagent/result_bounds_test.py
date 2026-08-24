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
from typing import Final, cast

import asyncio
import dataclasses
import json
import re
import tempfile

import pytest

from sagent.agent.result_storage import post_process_result
from sagent.request_materialization import (
    ELIDED_TOOL_RESULT_TAG,
    materialize_messages,
)
from sagent.testing import FakeAgent, with_fake_agent
from sagent.tools.core import result_token_budget
from sagent.tools.glob_tool import Glob
from sagent.tools.grep import Grep
from sagent.tools.read import Read
from sagent.types.runtime import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


# Budget shape of a mid-size production model (opus-5: 200k window,
# ~3 chars/token), spelled out so the arithmetic in these tests is
# legible rather than imported and opaque.
_PERSIST_TOKENS = 50_000
_MESSAGE_BUDGET_TOKENS = 100_000


def _write_lines(path: Path, count: int) -> Path:
    """Write ``count`` numbered source-like lines and return the path."""
    path.write_text(
        "".join(f"{i:06d}\tdef f{i}(): return {i}\n" for i in range(count)),
        encoding="utf-8",
    )
    return path


def test_large_read_is_not_silently_elided(tmp_path: Path) -> None:
    """A big Read must reach the model as content, never as a bare tag.

    Read calls ``asyncio.to_thread`` directly rather than ``run_sync``,
    so the framework's own backstop never applies to it. If its own bound
    ever fails, the result lands in ``materialize_messages`` over-budget
    and is replaced by ``<elided>`` -- total content loss with no notice
    and no path back.
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
        messages, tool_result_budget_tokens=_MESSAGE_BUDGET_TOKENS
    )
    tool_results = [m for m in materialized if isinstance(m, ToolResult)]

    assert tool_results, "the Read result was dropped from the request entirely"
    assert tool_results[0].content != ELIDED_TOOL_RESULT_TAG, (
        "a large Read reached the model as a bare elision tag: the file"
        " contents were lost with no notice and no way to recover them"
    )


def _wide_jsonl(path: Path, count: int, width: int) -> Path:
    """Write ``count`` JSONL records of ~``width`` chars and return the path."""
    path.write_text(
        "".join('{"x":"' + "a" * width + '"}\n' for _ in range(count)),
        encoding="utf-8",
    )
    return path


def _one_huge_line(path: Path, width: int) -> Path:
    """Write a single line wider than any budget, with no newline to split."""
    path.write_text('{"x":"' + "a" * width + '"}\n', encoding="utf-8")
    return path


def _notebook(path: Path, cells: int, width: int) -> Path:
    """Write a notebook whose cells and outputs are both large."""
    body = "x" * width
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [body],
                        "outputs": [{"text": [body]}],
                    }
                    for _ in range(cells)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _oversized(tmp_path: Path) -> dict[str, tuple[object, Mapping[str, object]]]:
    """One over-budget invocation per text-producing tool.

    ``narrow`` and ``wide`` differ ONLY in line width. A bound expressed
    in lines passes one and fails the other by that ratio, which is the
    defect: the budget is not counted in lines.
    """
    for i in range(400):
        (tmp_path / f"f{i:04d}.txt").write_text("hit line\n" * 40, encoding="utf-8")
    return {
        "read-narrow": (
            Read(),
            {"file_path": str(_write_lines(tmp_path / "n.py", 4000))},
        ),
        "read-wide": (
            Read(),
            {"file_path": str(_wide_jsonl(tmp_path / "w.jsonl", 4000, 3_260))},
        ),
        # A minified blob: ONE line, so a unit-wise bound has nothing to
        # split and emits it whole. This is the wedge shape reduced to its
        # minimum -- session ``190b6baec7ed``'s file merely had wide lines.
        "read-one-line": (
            Read(),
            {"file_path": str(_one_huge_line(tmp_path / "min.json", 3_000_000))},
        ),
        "read-notebook": (
            Read(),
            {"file_path": str(_notebook(tmp_path / "n.ipynb", 400, 3_000))},
        ),
        "glob": (Glob(), {"pattern": "**/*.txt", "path": str(tmp_path)}),
        "grep": (
            Grep(),
            {"pattern": "hit", "path": str(tmp_path), "output_mode": "content"},
        ),
    }


_OVERSIZED_CASES: Final = (
    "read-narrow",
    "read-wide",
    "read-one-line",
    "read-notebook",
    "glob",
    "grep",
)


@pytest.mark.parametrize("case", _OVERSIZED_CASES)
def test_a_tool_result_never_exceeds_the_token_budget(
    case: str, tmp_path: Path
) -> None:
    """No tool may return more than the active model's per-result budget.

    THE invariant this module exists to enforce, and the one the wedge in
    session ``190b6baec7ed`` violated: a single ``Read`` returned 11.1M
    characters -- 40.9x its budget -- because the cap was expressed in
    LINES (``max_result_chars // 80``) while the budget is a token count.
    Wide JSONL lines average ~3.3k chars, so the guessed width was off by
    40x and the result overflowed every downstream bound.

    Parameterized over every text-producing tool, and over both a narrow
    and a wide file: a line- or match-shaped bound passes one and fails
    the other, which is precisely the bug. ``read-notebook`` covers
    ``_read_notebook``, which never reaches the windowing path at all and
    so historically had no cap of any kind.
    """
    tool, args = _oversized(tmp_path)[case]
    with with_fake_agent() as agent:
        result = asyncio.run(cast("Tool", tool).run(args))
        budget = agent.max_result_tokens
        used = agent.approx_text_tokens(result.content)
    assert used <= budget, (
        f"{case} returned {used:,} tokens against a {budget:,}-token budget"
        f" ({used / budget:.1f}x over)"
    )


def test_unbounded_grep_stays_within_the_result_cap(tmp_path: Path) -> None:
    """A default grep over a large tree must not overrun the result cap.

    With ``keep_first`` defaulting to unlimited, a wide match set is
    head-truncated by ``truncate`` at the framework boundary -- and
    Grep's ``offset`` cannot recover the tail once the characters are
    gone rather than the rows.
    """
    for i in range(200):
        (tmp_path / f"f{i:05d}.txt").write_text("hit line\n" * 40, encoding="utf-8")
    with with_fake_agent():
        result = asyncio.run(
            Grep().run(
                {"pattern": "hit", "path": str(tmp_path), "output_mode": "content"}
            )
        )
    with with_fake_agent() as agent:
        used = agent.approx_text_tokens(result.content)
    assert used <= result_token_budget(), (
        f"grep produced {used:,} tokens against a"
        f" {result_token_budget():,}-token budget; the overflow is cut mid-stream"
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


def test_a_resume_offset_never_points_behind_what_was_shown(tmp_path: Path) -> None:
    """A continuation must continue, not re-fetch what the reader already has.

    ``keep_last`` slices the TAIL, and the slice discards where that tail
    began. The resume note was then phrased from the caller's ``offset``
    -- still ``0`` on this path -- so it named a position at the HEAD
    while the body showed the tail: every row between them was
    unreachable, and following the note re-fetched rows already seen.

    "Recoverable" (see this module's docstring) means the offered
    argument reaches the omitted rows. A note that points backwards
    fails that whether or not it is well-formed.
    """
    rows = 400
    path = tmp_path / "wide.txt"
    path.write_text(
        "".join(f"hit{i:04d} {'w' * 400}\n" for i in range(rows)), encoding="utf-8"
    )
    # ``keep_last`` must be a REAL tail (strictly fewer than the matches),
    # or the slice is a no-op and the discarded start never matters. The
    # budget must also be tight enough to withhold rows, since a reply
    # that fits offers no resume note to check.
    with with_fake_agent(agent=dataclasses.replace(FakeAgent(), max_result_tokens=500)):
        result = asyncio.run(
            Grep().run(
                {
                    "pattern": "hit",
                    "path": str(path),
                    "output_mode": "content",
                    "keep_last": rows // 2,
                }
            )
        )
    shown = re.findall(r"hit(\d+) ", result.content)
    resume = re.search(r"offset=(\d+)", result.content)
    if resume is None:
        return  # Nothing was withheld, so there is no claim to check.
    assert shown, "a resume note was offered but no rows were shown"
    assert int(resume.group(1)) > int(shown[-1]), (
        f"resume says offset={resume.group(1)} but rows through"
        f" {shown[-1]} are already shown; the note points backwards"
    )


def test_an_oversized_error_result_keeps_a_path_back(tmp_path: Path) -> None:
    """A huge failure must stay readable, like a huge success.

    Error results skipped disk off-load, but request materialization
    elides ANY over-budget result -- so a large traceback reached the
    model as ``<elided>`` with nothing to re-read, while the identical
    body as a success was written to disk with a path. That inverts the
    priority: the failing case is the one whose detail is wanted.
    """
    body = "traceback line\n" * 100_000
    processed = post_process_result(
        ToolResult(call_id="err", content=body, is_error=True),
        "Bash",
        session_dir=tmp_path,
        persist_tokens=_PERSIST_TOKENS,
        message_budget_tokens=_MESSAGE_BUDGET_TOKENS,
        used_message_tokens=0,
    )
    assert "tool-results" in processed.content, (
        "an oversized error result was not off-loaded, so materialization"
        " will elide it with no way to read the failure"
    )
    assert processed.is_error, "off-loading must not silently clear is_error"


def test_persisted_result_carries_a_path_back() -> None:
    """Disk offload is the one lossless shortening path; keep it honest."""
    body = "y" * 3_000_000
    with tempfile.TemporaryDirectory() as tmp:
        out = post_process_result(
            ToolResult(call_id="c1", content=body),
            "Bash",
            session_dir=Path(tmp),
            persist_tokens=_PERSIST_TOKENS,
            message_budget_tokens=_MESSAGE_BUDGET_TOKENS,
            used_message_tokens=0,
        )
    assert "tool-results" in out.content, "persisted result lost its path back"
    assert len(out.content) < len(body)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
