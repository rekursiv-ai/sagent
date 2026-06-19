"""Tests for ``compaction.history``: shared history mutators and estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast, override

from sagent.compaction import scrunch, summary
from sagent.compaction.history import (
    append_to_first_user,
    estimate_entry_tokens,
)
from sagent.lib.token_count import entry_tokens
from sagent.testing import MockModelCaps
from sagent.types.model import Model, ModelRequest, ModelResponse
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


def test_estimate_entry_tokens_shared_across_compaction_modules() -> None:
    # ``estimate_entry_tokens`` is the canonical token estimator; ``summary``
    # and ``scrunch`` must reuse it, not keep private copies that silently
    # drift from one another (one rule everywhere).
    assert summary.estimate_entry_tokens is estimate_entry_tokens
    assert scrunch.estimate_entry_tokens is estimate_entry_tokens


def test_estimate_entry_tokens_delegates_to_wire_walker() -> None:
    # The per-entry estimator must be the same one the request builder uses.
    model = cast(Model, MockModelCaps())
    entries: list[ModelContextEvent] = [
        UserMessage(text="hello"),
        AssistantMessage(text="hi"),
    ]
    expected = sum(entry_tokens(e, model) for e in entries)
    assert estimate_entry_tokens(model, entries) == expected


def test_append_to_first_user_inserts_when_absent() -> None:
    events: list[ModelContextEvent] = []
    append_to_first_user(events, "seed")
    first = events[0]
    assert isinstance(first, UserMessage)
    assert first.text == "seed"


def test_append_to_first_user_does_not_reorder_before_agent_send() -> None:
    """With only an ``AgentSendMessage`` (a user-role wire message) present,
    the injected text must not be prepended ahead of it -- that reorders the
    conversation. It must attach to the agent-send or land after it.
    """
    asm = AgentSendMessage(source="rev", text="hello")
    events: list[ModelContextEvent] = [asm]
    append_to_first_user(events, "ctx")
    # The agent-send must remain first; nothing inserted before it.
    assert events[0] is asm or (
        isinstance(events[0], AgentSendMessage) and events[0].source == "rev"
    )


@dataclass(slots=True, kw_only=True)
class _NonLinearModel(MockModelCaps):
    """Tokenizer where token count is the number of DISTINCT characters.

    Deliberately non-linear in length: ``"x" * 1000`` -> 1 token, but real
    varied text -> many. A probe that samples ``"x" * 1000`` to recover a
    chars-per-token ratio (the deleted ``chars_to_tokens``) would compute a
    ~1000:1 ratio and wildly under-count real text. Tokenizing the actual
    entry text gives the right answer.
    """

    model_id: str = "nonlinear"
    max_request_tokens: int = 200_000
    received_texts: list[str] = field(default_factory=list)

    @override
    def approx_text_tokens(self, text: str) -> int:
        self.received_texts.append(text)
        return len(set(text))

    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del request, publish
        return ModelResponse(message=AssistantMessage(text=""))

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request)


def test_estimate_entry_tokens_tokenizes_real_text_not_a_sample_probe() -> None:
    """``estimate_entry_tokens`` must tokenize the entries' actual text via
    the model, not reconstruct a ratio from a fixed dummy sample.

    Under a non-linear tokenizer, the two diverge by orders of magnitude.
    """
    model = _NonLinearModel()
    entries: list[ModelContextEvent] = [
        UserMessage(text="abcde"),  # 5 distinct -> 5 tokens
        AssistantMessage(text="wxyz"),  # 4 distinct -> 4 tokens
    ]
    got = estimate_entry_tokens(model, entries)
    # Real per-entry tokenization: 5 + 4 = 9. The old sample-probe approach
    # (ratio from "x"*1000 == 1 token => ratio 1000) would yield ~0.
    assert got == 9
    # And it tokenized the real entry text, never a "xxxx..." sample.
    assert "abcde" in model.received_texts
    assert "wxyz" in model.received_texts
    assert not any(set(t) == {"x"} and len(t) >= 100 for t in model.received_texts)


def test_estimate_entry_tokens_counts_every_wire_surface() -> None:
    """``estimate_entry_tokens`` must count the same per-entry surfaces the
    request walker bills: tool-call id/name/args JSON, thinking blocks, and
    attachments -- not just bare text.

    Equivalence with ``approx_request_tokens`` over a messages-only request
    (no system, no tools) is the contract: one per-entry estimator, used by
    both compaction sizing and request building.
    """
    model = cast(Model, MockModelCaps())
    entries: list[ModelContextEvent] = [
        UserMessage(text="hello"),
        AssistantMessage(
            text="reasoning",
            tool_calls=(
                ToolCall(id="call_abc", name="Read", args={"file_path": "/x/y.py"}),
            ),
            thinking_blocks=(
                {"type": "thinking", "thinking": "deep" * 50, "signature": "sig"},
            ),
        ),
        ToolResult(call_id="call_abc", content="file contents here"),
    ]
    via_history = estimate_entry_tokens(model, entries)
    via_request = model.approx_request_tokens(
        ModelRequest(messages=entries, system=None, tools=None)
    )
    assert via_history == via_request


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
