"""Tests for ``lib.token_count``: wire-aware approx walker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from sagent.lib.token_count import approx_request_tokens


if TYPE_CHECKING:
    import pytest
from sagent.lib.custom_json import JSON
from sagent.testing import MockModelCaps
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AssistantMessage,
    BytesMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


@dataclass(slots=True, kw_only=True)
class _TokenModel(MockModelCaps):
    """Trivial estimator: text length // 4, fixed image cost."""

    model_id: str = "tok"
    max_request_tokens: int = 100_000

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 7


@dataclass(slots=True, kw_only=True)
class _StubTool:
    """Tool stub satisfying the rich ``Tool`` protocol surface."""

    name: str = "Stub"
    tool_id: str = "application/x-tool-stub"
    description: str = ""
    directive_schema: JSON = field(default_factory=lambda: {"type": "object"})
    clearable_results: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def summary_result(self, result: ToolResult) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        return ""

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


def _req(messages: list[ModelContextEvent], **kwargs: object) -> ModelRequest:
    return ModelRequest(messages=messages, **kwargs)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType] -- test helper


def test_user_message_text() -> None:
    n = approx_request_tokens(
        _req([UserMessage(text="abcdefgh")], system="sys"),
        _TokenModel(),
    )
    # ``"sys"`` → 0 (3 chars // 4); ``"abcdefgh"`` → 2.
    assert n == 0 + 2


def test_user_with_image_attachment() -> None:
    n = approx_request_tokens(
        _req(
            [
                UserMessage(
                    text="",
                    attachments=(
                        BytesMessage(data=b"\x89PNG", descriptor="image/png"),
                    ),
                ),
            ],
        ),
        _TokenModel(),
    )
    assert n == 7


def test_user_with_pdf_attachment_counted() -> None:
    """PDF attachments contribute to the token estimate.

    Anthropic and Google providers ship PDFs on the wire; the estimator
    must include them or compaction fires after the request is already
    oversized.
    """
    n = approx_request_tokens(
        _req(
            [
                UserMessage(
                    text="",
                    attachments=(
                        BytesMessage(data=b"%PDF", descriptor="application/pdf"),
                    ),
                ),
            ],
        ),
        _TokenModel(),
    )
    assert n == 7


def test_user_with_unknown_attachment_zero_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown descriptors contribute zero but emit a warning.

    Silent-drop was the F58 bug; the warning is the defensive log that
    keeps a future provider addition from re-introducing the regression.
    """
    with caplog.at_level("WARNING", logger="sagent.lib.token_count"):
        n = approx_request_tokens(
            _req(
                [
                    UserMessage(
                        text="",
                        attachments=(
                            BytesMessage(
                                data=b"hello",
                                descriptor="application/x-unknown",
                            ),
                        ),
                    ),
                ],
            ),
            _TokenModel(),
        )
    assert n == 0
    assert any("application/x-unknown" in r.message for r in caplog.records)


def test_assistant_text() -> None:
    n = approx_request_tokens(
        _req([AssistantMessage(text="abcdefgh")]),
        _TokenModel(),
    )
    assert n == 2


def test_tool_result_text_with_image() -> None:
    n = approx_request_tokens(
        _req(
            [
                ToolResult(
                    call_id="c1",
                    content="result",
                    attachments=(
                        BytesMessage(data=b"\x89PNG", descriptor="image/jpeg"),
                    ),
                ),
            ],
        ),
        _TokenModel(),
    )
    # ``"result"`` → 1 (6 chars // 4); image → 7.
    assert n == 1 + 7


def test_assistant_tool_call_args_counted() -> None:
    """``ToolCall.args`` are summed (this was the silent undercount bug)."""
    tc = ToolCall(id="t1", name="Bash", args={"cmd": "x" * 100})
    n = approx_request_tokens(
        _req([AssistantMessage(text="", tool_calls=(tc,))]),
        _TokenModel(),
    )
    # JSON-encoded ``{id, name, args}`` is longer than the cmd alone.
    assert n > 100 // 4


def test_assistant_thinking_signature_counted() -> None:
    """Thinking-block signatures and bodies are summed."""
    n = approx_request_tokens(
        _req(
            [
                AssistantMessage(
                    text="",
                    thinking_blocks=(
                        {
                            "type": "thinking",
                            "signature": "s" * 80,
                            "thinking": "t" * 40,
                        },
                    ),
                ),
            ],
        ),
        _TokenModel(),
    )
    # 80 // 4 + 40 // 4 = 20 + 10.
    assert n == 30


def test_reasoning_text_blocks_count_tokens() -> None:
    """OpenAI/Moonshot reasoning blocks (``type=reasoning``) bill via ``text``.

    These blocks reach ``AssistantMessage.thinking_blocks`` from the OpenAI-
    compat / OpenAI-subscription parsers as ``{"type":"reasoning","text":...}``.
    Without counting their ``text`` field a long reasoning trace contributes
    zero to the estimate, so proactive compaction fires far too late and the
    provider 400s with context overflow.
    """
    msg = AssistantMessage(
        text="hi",
        thinking_blocks=({"type": "reasoning", "text": "x" * 4000},),
    )
    n = approx_request_tokens(_req([msg]), _TokenModel())
    # 4000 chars // 4 = 1000 tokens from the reasoning text; ``"hi"`` rounds
    # to 0 from the // 4 truncation.
    assert n == 1000


def test_tools_schema_counted() -> None:
    """``request.tools`` schemas are summed via JSON-serialized form."""
    tool = _StubTool(directive_schema={"type": "object", "x": "y" * 100})
    n = approx_request_tokens(_req([], tools=[tool]), _TokenModel())
    # ``json.dumps(...)`` of the schema has ≥ 100 chars; expect ≥ 25 tokens.
    assert n >= 25


def test_empty_request_zero() -> None:
    assert approx_request_tokens(_req([]), _TokenModel()) == 0


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
