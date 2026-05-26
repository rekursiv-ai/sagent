"""Advisor: consult a more capable model as a sub-agent tool.

Exposes an LLM as an ``advisor`` tool the main agent can call when
stuck. Each consult runs a fresh sub-agent: the advisor sees only
the prompt string the executor passes - no shared history, no tools,
no cross-consult memory.

See https://claude.com/blog/the-advisor-strategy for the strategy:
pair a cheap executor (Sonnet/Haiku) with Opus as advisor and get
near-Opus intelligence at a fraction of the cost. This module is a
client-side approximation of Anthropic's server-side ``advisor`` tool
- two round-trips per consult instead of one, but provider-agnostic
and fully observable in the REPL.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from sagent.agent import runtime as agent_runtime
from sagent.lib import debug_log
from sagent.lib.json import JSON, json_freeze
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import Model, ModelRequest


_SYSTEM = (
    "You advise a coding agent that is stuck on a decision. Read the"
    " question and return a concise plan, correction, or stop signal."
    " You have no tools and cannot act - your reply goes only to the"
    " executor. Be direct and specific; skip preamble."
)

_DESCRIPTION = (
    "Consult a more capable advisor model for guidance. The advisor"
    " has no tools and sees only the prompt you send. Typical"
    " triggers: a tool call has failed twice, you're choosing between"
    " two approaches without clear evidence, or you're about to make"
    " a non-obvious design decision. Include the situation, options"
    " considered, and the specific decision you need help with."
)

SYSTEM_NUDGE = (
    "# Advisor\n\n"
    "An `advisor` tool is available - a more capable model with no"
    " tools of its own. It returns a short plan, correction, or stop"
    " signal.\n\n"
    "Consult it when:\n"
    "- A tool call fails twice and the cause isn't obvious.\n"
    "- You're choosing between two approaches without clear evidence"
    " for either.\n"
    "- You're about to make a non-obvious, hard-to-reverse decision.\n"
    "- You've made no progress on the current sub-task for two model requests.\n"
    "\n"
    "Skip it for tasks you already know how to do, simple lookups, or"
    " style questions.\n\n"
    "The advisor sees only the prompt you send - no history, no tool"
    " results, no files. Write self-contained prompts: state the"
    " situation, the options you've considered, and the specific"
    " decision you need help with."
)


class _AdvisorModel:
    """Bridge a provider ``Model`` to the runtime ``Model`` protocol."""

    def __init__(self, inner: Model) -> None:
        self._inner = inner

    async def stream(
        self,
        history: list[HistoryEntry],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        """Stream a provider response and adapt it to the runtime ``Model`` protocol.

        Args:
          history: Conversation history for the advisor consult.
          system: System prompt to send.
          tools: Runtime-side tools forwarded by the engine (ignored).
          on_text: Callback for each streamed text chunk.
          on_thinking: Callback for each streamed thinking chunk.

        Returns:
          message: Final ``AssistantMessage`` from the inner provider.

        """
        del tools
        request = ModelRequest(messages=history, system=system or None)
        response = await self._inner.stream(request, on_text, on_thinking)
        return response.message


class Advisor:
    """``Tool`` wrapper exposing an LLM as an advisor sub-agent."""

    name: str = "advisor"
    tool_id: str = "application/x-tool-advisor"
    clearable_results: bool = False
    description: str = _DESCRIPTION
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Self-contained question for the advisor: the"
                        " situation, options considered, and the decision"
                        " you need help with. No shared context."
                    ),
                },
            },
            "required": ["prompt"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return the status-pane label for a pending advisor consult.

        Args:
          args: Directive arguments (ignored).

        Returns:
          label: ``Advisor consulting…`` line shown before invocation.

        """
        del args
        return "Advisor consulting…"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for advisor consults.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return the advisor system-prompt nudge.

        Returns:
          contribution: Advisor invocation guidance for the system prompt.

        """
        return SYSTEM_NUDGE

    def __init__(
        self,
        *,
        model: Model,
        max_uses: int | None = None,
        system: str = _SYSTEM,
    ) -> None:
        self._model = model
        self._max_uses = max_uses
        self._system = system
        self._uses = 0

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Run a fresh sub-agent consultation and return the advice.

        Args:
          args: Directive with the ``prompt`` string.

        Returns:
          result: Advisor's reply, or a quota-exhausted error.

        """
        prompt = str(args.get("prompt", ""))
        if self._max_uses is not None and self._uses >= self._max_uses:
            return ToolResult(
                call_id="",
                content=f"Advisor quota exhausted ({self._max_uses} uses).",
                is_error=True,
            )
        self._uses += 1
        debug_log.trace(
            "advisor_invoke",
            model=self._model.model_id,
            prompt_len=len(prompt),
            uses=self._uses,
            max_uses=self._max_uses,
        )
        runtime = agent_runtime.AgentRuntime(
            model=_AdvisorModel(self._model),
            system=self._system,
            tools=[],
        )
        history = await runtime.run(UserMessage(text=prompt))
        for m in reversed(history):
            if isinstance(m, AssistantMessage):
                return ToolResult(call_id="", content=m.text)
        return ToolResult(call_id="", content="")
