"""Tests for ``compaction.pruner``: deterministic observation and thinking pruner."""

from __future__ import annotations

import time

from sagent.compaction.pruner import (
    DEFAULT_TAIL_TURNS,
    elide_historical_thinking,
    prune_messages,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def _create_read_pair(
    call_id: str,
    path: str,
    content: str,
) -> tuple[AssistantMessage, ToolResult]:
    """Create an assistant read call and corresponding tool result."""
    am = AssistantMessage(
        tool_calls=(
            ToolCall(
                id=call_id,
                name="Read",
                args={"file_path": path},
            ),
        ),
    )
    tr = ToolResult(call_id=call_id, content=content)
    return am, tr


def _create_bash_pair(
    call_id: str,
    command: str,
    stdout: str,
    rc: int = 0,
) -> tuple[AssistantMessage, ToolResult]:
    """Create an assistant bash call and corresponding tool result."""
    am = AssistantMessage(
        tool_calls=(
            ToolCall(
                id=call_id,
                name="Bash",
                args={"command": command},
            ),
        ),
    )
    content = f"{stdout}\n[exit code: {rc}]" if rc != 0 else stdout
    tr = ToolResult(call_id=call_id, content=content, is_error=(rc != 0))
    return am, tr


def _create_thinking_assistant(
    thought: str,
    sig: str = "sig_default",
    tool_call: ToolCall | None = None,
) -> AssistantMessage:
    """Create an assistant message with a thinking block."""
    calls = (tool_call,) if tool_call is not None else ()
    return AssistantMessage(
        thinking_blocks=({"type": "thinking", "thinking": thought},),
        thought_signature=sig,
        tool_calls=calls,
    )


def test_superseded_file_read_deduplication() -> None:
    """A file read at turn T1 is collapsed when re-read at turn T2 outside tail."""
    events: list[ModelContextEvent] = [
        UserMessage(text="Inspect model architecture"),
    ]
    am1, tr1 = _create_read_pair("c_read_1", "src/model.py", "class ResNet:\n" * 50)
    events.extend([am1, tr1])

    am2, tr2 = _create_bash_pair("c_bash_2", "python -V", stdout="Python 3.12.0", rc=0)
    events.extend([am2, tr2])

    am3, tr3 = _create_read_pair(
        "c_read_2", "src/model.py", "class ResNet:\n# v2\n" * 50
    )
    events.extend([am3, tr3])

    # 6 trailing turns to form the active tail
    for i in range(5, 11):
        events.append(UserMessage(text=f"Tail prompt {i}"))
        events.append(AssistantMessage(text=f"Tail response {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)

    res_t1 = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_read_1"
    )
    assert "superseded" in res_t1.content.lower()

    res_t2 = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_read_2"
    )
    assert "class ResNet:" in res_t2.content


def test_clean_bash_stdout_scrubbing_to_summary_line() -> None:
    """Successful bash command (rc=0) older than 2 turns outside tail is collapsed."""
    events: list[ModelContextEvent] = [
        UserMessage(text="Run build and tests"),
    ]
    clean_stdout = "compiling sagent/core...\n" * 150
    am1, tr1 = _create_bash_pair("c_bash_clean", "cargo build", clean_stdout, rc=0)
    events.extend([am1, tr1])

    # Intermediary turn older than tail
    events.append(UserMessage(text="Next step"))
    events.append(AssistantMessage(text="Continuing..."))

    # 6 active tail turns
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail user {i}"))
        events.append(AssistantMessage(text=f"Tail asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)

    res_bash = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_bash_clean"
    )
    assert len(res_bash.content.splitlines()) == 1
    assert "Command succeeded" in res_bash.content
    assert "rc=0" in res_bash.content


def test_failed_bash_stderr_and_stdout_strictly_preserved() -> None:
    """Commands that failed (rc!=0, is_error=True) are preserved verbatim."""
    events: list[ModelContextEvent] = [
        UserMessage(text="Execute compilation"),
    ]
    error_traceback = (
        "Traceback (most recent call last):\n"
        '  File "sagent/agent.py", line 650, in bound_result\n'
        "TypeError: unsupported operand type(s)\n"
        "[exit code: 1]"
    )
    am_fail, tr_fail = _create_bash_pair(
        "c_bash_err", "pytest tests/", error_traceback, rc=1
    )
    events.extend([am_fail, tr_fail])

    # 6 active tail turns
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail {i}"))
        events.append(AssistantMessage(text=f"Ack {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)

    res_err = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_bash_err"
    )
    assert res_err.content == tr_fail.content
    assert res_err.is_error is True
    assert "Traceback" in res_err.content


def test_active_working_tail_protected() -> None:
    """The most recent 6 turns are 100% untouched."""
    events: list[ModelContextEvent] = []
    # 2 old turns
    am_old, tr_old = _create_read_pair(
        "c_old_r1", "shared.py", "shared content 1\n" * 100
    )
    events.extend([am_old, tr_old])

    # 6 active tail turns
    for i in range(1, 7):
        if i == 1:
            am_tail, tr_tail = _create_read_pair(
                "c_tail_r1", "shared.py", "tail content 1"
            )
            events.extend([am_tail, tr_tail])
        elif i == 2:
            am_tail2, tr_tail2 = _create_read_pair(
                "c_tail_r2", "shared.py", "tail content 2"
            )
            events.extend([am_tail2, tr_tail2])
        else:
            events.append(UserMessage(text=f"User {i}"))
            events.append(AssistantMessage(text=f"Asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)

    tr1 = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_tail_r1"
    )
    assert tr1.content == "tail content 1"

    tr2 = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_tail_r2"
    )
    assert tr2.content == "tail content 2"


def test_historical_thinking_block_elision_older_than_n_minus_1() -> None:
    """Historical thinking blocks older than N-1 are cleared while preserving thought_signature."""
    events: list[ModelContextEvent] = [
        UserMessage(text="Turn 1 prompt"),
        _create_thinking_assistant("Thinking step 1", sig="sig_step_1"),
        UserMessage(text="Turn 2 prompt"),
        _create_thinking_assistant("Thinking step 2", sig="sig_step_2"),
        UserMessage(text="Turn 3 prompt"),
        _create_thinking_assistant("Thinking step 3 (N-1)", sig="sig_step_3"),
    ]

    pruned = elide_historical_thinking(events)

    asst1 = pruned[1]
    assert isinstance(asst1, AssistantMessage)
    assert asst1.thinking_blocks == ()

    asst2 = pruned[3]
    assert isinstance(asst2, AssistantMessage)
    assert asst2.thinking_blocks == ()

    asst3 = pruned[5]
    assert isinstance(asst3, AssistantMessage)
    assert len(asst3.thinking_blocks) == 1
    assert asst3.thinking_blocks[0]["thinking"] == "Thinking step 3 (N-1)"
    assert asst3.thought_signature == "sig_step_3"


def test_tool_pairing_invariant_all_call_ids_paired() -> None:
    """100% of ToolCall.id and ToolResult.call_id remain paired after pruning."""
    events: list[ModelContextEvent] = [UserMessage(text="Start task")]
    for i in range(1, 15):
        if i % 2 == 0:
            am, tr = _create_read_pair(
                f"call_{i}", f"file_{i % 3}.py", "content\n" * 50
            )
        else:
            am, tr = _create_bash_pair(f"call_{i}", f"cmd_{i}", "output\n" * 50, rc=0)
        events.extend([am, tr])

    pruned = prune_messages(events, tail_turns=6, elide_thinking=True)

    call_ids = {
        tc.id for e in pruned if isinstance(e, AssistantMessage) for tc in e.tool_calls
    }
    result_ids = {e.call_id for e in pruned if isinstance(e, ToolResult)}
    assert call_ids == result_ids


def test_short_session_under_6_turns_untouched() -> None:
    """A sequence with <= 6 turns total is completely protected (0 pruning)."""
    events: list[ModelContextEvent] = []
    am1, tr1 = _create_read_pair("c_short_1", "foo.py", "foo content 1")
    am2, tr2 = _create_bash_pair("c_short_2", "pytest", "all passed\n" * 60, rc=0)
    am3, tr3 = _create_read_pair("c_short_3", "foo.py", "foo content 2")
    events.extend([am1, tr1, am2, tr2, am3, tr3])

    pruned = prune_messages(events, tail_turns=DEFAULT_TAIL_TURNS, elide_thinking=False)

    tr1_out = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_short_1"
    )
    assert tr1_out.content == "foo content 1"

    tr2_out = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_short_2"
    )
    assert "all passed" in tr2_out.content


def test_pruning_latency_under_1ms() -> None:
    """Pruning overhead is strictly sub-millisecond."""
    events: list[ModelContextEvent] = [UserMessage(text="Benchmark prompt")]
    for i in range(1, 30):
        am, tr = _create_read_pair(
            f"c_bench_{i}", f"file_{i % 5}.py", "sample line\n" * 100
        )
        events.extend([am, tr])

    t0 = time.perf_counter()
    _ = prune_messages(events, tail_turns=6, elide_thinking=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 1.0


def test_short_bash_discovery_stdout_preserved() -> None:
    """Discovery bash outputs (<= 10 lines and <= 500 chars) are kept verbatim."""
    events: list[ModelContextEvent] = [UserMessage(text="Inspect repository")]
    discovery_stdout = "e512c8a15cf35fb44d990b691e669d7872de4584\n"
    am1, tr1 = _create_bash_pair(
        "c_bash_discovery", "git rev-parse HEAD", discovery_stdout, rc=0
    )
    events.extend([am1, tr1])

    # Intermediary turn older than tail
    events.append(UserMessage(text="Next step"))
    events.append(AssistantMessage(text="Continuing..."))

    # 6 active tail turns
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail user {i}"))
        events.append(AssistantMessage(text=f"Tail asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)
    res_bash = next(
        e
        for e in pruned
        if isinstance(e, ToolResult) and e.call_id == "c_bash_discovery"
    )
    assert res_bash.content == discovery_stdout


def test_read_in_pre_tail_superseded_by_read_in_active_tail() -> None:
    """A pre-tail file read is superseded when re-read inside the active tail."""
    events: list[ModelContextEvent] = [UserMessage(text="Initial inspection")]
    am_old, tr_old = _create_read_pair(
        "c_read_old", "config.py", "OLD CONFIG CONTENT\n"
    )
    events.extend([am_old, tr_old])

    # Intermediary turn older than tail
    events.append(UserMessage(text="Middle step"))
    events.append(AssistantMessage(text="Continuing..."))

    # Active tail turns: re-read config.py in turn 3 of tail
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail user {i}"))
        if i == 3:
            am_tail, tr_tail = _create_read_pair(
                "c_read_tail", "config.py", "FRESH ACTIVE CONFIG CONTENT\n"
            )
            events.extend([am_tail, tr_tail])
        else:
            events.append(AssistantMessage(text=f"Tail asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)
    res_old = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_read_old"
    )
    res_tail = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_read_tail"
    )
    assert "[Read of config.py superseded by later turn]" in res_old.content
    assert res_tail.content == "FRESH ACTIVE CONFIG CONTENT\n"


def test_elide_historical_thinking_preserves_encrypted_reasoning_blocks() -> None:
    """Historical assistant messages preserve encrypted_content for OpenAI all_turns."""
    encrypted_block = {
        "type": "reasoning",
        "encrypted_content": "opaque-token-for-openai-reasoning-context",
    }
    plaintext_thought_block = {
        "type": "thinking",
        "thinking": "This internal thinking text should be pruned",
    }

    turn1_asst = AssistantMessage(
        text="Turn 1 answer",
        thinking_blocks=(encrypted_block, plaintext_thought_block),
    )
    turn2_asst = AssistantMessage(
        text="Turn 2 answer",
        thinking_blocks=(plaintext_thought_block,),
    )

    events: list[ModelContextEvent] = [
        UserMessage(text="Turn 1 prompt"),
        turn1_asst,
        UserMessage(text="Turn 2 prompt"),
        turn2_asst,
    ]

    pruned = elide_historical_thinking(events)

    # Turn 1 is historical relative to Turn 2:
    asst1_pruned = pruned[1]
    assert isinstance(asst1_pruned, AssistantMessage)
    assert len(asst1_pruned.thinking_blocks) == 1
    assert asst1_pruned.thinking_blocks[0] == encrypted_block

    # Turn 2 is the latest assistant message: thinking blocks kept verbatim
    asst2_pruned = pruned[3]
    assert isinstance(asst2_pruned, AssistantMessage)
    assert asst2_pruned.thinking_blocks == (plaintext_thought_block,)


def test_disjoint_windowed_reads_not_superseded() -> None:
    """Disjoint windowed reads of the same file are preserved and not superseded."""
    events: list[ModelContextEvent] = [UserMessage(text="Inspect head of file")]
    am1 = AssistantMessage(
        tool_calls=(
            ToolCall(
                id="c_read_w1",
                name="Read",
                args={"file_path": "src/model.py", "offset": 1, "limit": 50},
            ),
        ),
    )
    tr1 = ToolResult(call_id="c_read_w1", content="HEAD OF FILE\n" * 50)
    events.extend([am1, tr1])

    # Intermediary turn older than tail
    events.append(UserMessage(text="Next step"))
    events.append(AssistantMessage(text="Continuing..."))

    # Active tail turns: read disjoint window lines 100-150 in turn 3 of tail
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail user {i}"))
        if i == 3:
            am_tail = AssistantMessage(
                tool_calls=(
                    ToolCall(
                        id="c_read_w2",
                        name="Read",
                        args={"file_path": "src/model.py", "offset": 100, "limit": 50},
                    ),
                ),
            )
            tr_tail = ToolResult(call_id="c_read_w2", content="MIDDLE OF FILE\n" * 50)
            events.extend([am_tail, tr_tail])
        else:
            events.append(AssistantMessage(text=f"Tail asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)
    res_w1 = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_read_w1"
    )
    assert "HEAD OF FILE" in res_w1.content
    assert "superseded" not in res_w1.content


def test_bash_timeout_reason_marker_strictly_preserved() -> None:
    """A bash command terminated by timeout is preserved and not scrubbed as rc=0."""
    events: list[ModelContextEvent] = [UserMessage(text="Run build")]
    stdout = "building packages...\n" * 50
    content = f"{stdout}\n[timeout after 30.0s]"
    am1 = AssistantMessage(
        tool_calls=(
            ToolCall(
                id="c_bash_to",
                name="Bash",
                args={"command": "cargo build"},
            ),
        ),
    )
    tr1 = ToolResult(call_id="c_bash_to", content=content, is_error=False)
    events.extend([am1, tr1])

    # Intermediary turn older than tail
    events.append(UserMessage(text="Next step"))
    events.append(AssistantMessage(text="Continuing..."))

    # 6 active tail turns
    for i in range(1, 7):
        events.append(UserMessage(text=f"Tail user {i}"))
        events.append(AssistantMessage(text=f"Tail asst {i}"))

    pruned = prune_messages(events, tail_turns=6, elide_thinking=False)
    res_bash = next(
        e for e in pruned if isinstance(e, ToolResult) and e.call_id == "c_bash_to"
    )
    assert "timeout after 30.0s" in res_bash.content
    assert "Command succeeded" not in res_bash.content


def test_hollow_assistant_thinking_elision_sets_placeholder_text() -> None:
    """Historical assistant message with only thinking gets placeholder text to avoid empty wire message."""
    thought_block = {"type": "thinking", "thinking": "Thinking about the task..."}
    hollow_asst = AssistantMessage(
        text="",
        tool_calls=(),
        thinking_blocks=(thought_block,),
    )
    latest_asst = AssistantMessage(
        text="Final answer",
        thinking_blocks=(thought_block,),
    )
    events: list[ModelContextEvent] = [
        UserMessage(text="Turn 1 prompt"),
        hollow_asst,
        UserMessage(text="Turn 2 prompt"),
        latest_asst,
    ]

    pruned = elide_historical_thinking(events)
    asst1_pruned = pruned[1]
    assert isinstance(asst1_pruned, AssistantMessage)
    assert asst1_pruned.text == "[Historical reasoning elided]"
    assert asst1_pruned.thinking_blocks == ()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
