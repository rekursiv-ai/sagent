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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import logging

from sagent.agent.agent import Agent, ObservedModel
from sagent.agent.retry import (
    RateLimitError,
    RetriesExhaustedError,
    RetryDeferredError,
    send_with_retry,
)
from sagent.agent.state import current_agent_var
from sagent.lib import debug_log
from sagent.lib.codec import PlainTree, immutable
from sagent.types.model import (
    Model,
    ModelRequest,
    ModelTerminationError,
    StreamInterruptedError,
)
from sagent.types.runtime import ToolResult, UserMessage


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)


SYSTEM_NUDGE: Final = (
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


class Advisor:
    """``Tool`` wrapper exposing an LLM as an advisor sub-agent."""

    name: str = "advisor"
    tool_id: str = "application/x-tool-advisor"
    clearable_results: bool = False
    description: str = (
        "Consult a more capable advisor model for guidance. The advisor"
        " has no tools and sees only the prompt you send. Typical"
        " triggers: a tool call has failed twice, you're choosing between"
        " two approaches without clear evidence, or you're about to make"
        " a non-obvious design decision. Include the situation, options"
        " considered, and the specific decision you need help with."
    )
    directive_schema: Mapping[str, PlainTree] = immutable(
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
        },
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
        system: str = (
            "You advise a coding agent that is stuck on a decision. Read the"
            " question and return a concise plan, correction, or stop signal."
            " You have no tools and cannot act - your reply goes only to the"
            " executor. Be direct and specific; skip preamble."
        ),
    ) -> None:
        self._model = model
        self._max_uses = max_uses
        self._system = system
        self._uses = 0

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: advisor calls are independent."""
        del args
        return None

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
            model=self._model.tagged_model_id,
            prompt_len=len(prompt),
            uses=self._uses,
            max_uses=self._max_uses,
        )
        request = ModelRequest(
            messages=[UserMessage(text=prompt)],
            system=self._system,
        )
        try:
            response = await send_with_retry(
                self._consult_model(),
                request,
                max_attempts=3,
                persistent_retry=False,
                publish_recoverable=lambda text: logger.info(
                    "advisor recoverable: %s",
                    text,
                ),
            )
        # Every way ``send_with_retry`` ends a consult is this tool's answer, not a
        # crash. The calling agent's budget cap is not among them: it must stop
        # that agent, not read as bad advice.
        except (
            ModelTerminationError,
            RateLimitError,
            RetriesExhaustedError,
            RetryDeferredError,
            StreamInterruptedError,
        ) as exc:
            return ToolResult(
                call_id="",
                content=f"Advisor consult failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(call_id="", content=response.message.text)

    # The consult bills the running agent: wrap the model so its cost and budget
    # cap see the call, exactly like the agent's own turns.
    def _consult_model(self) -> Model:
        """Return the advisor model, accounted against the current agent."""
        agent = current_agent_var.get(None)
        if isinstance(agent, Agent):
            return ObservedModel(self._model, agent.record_side_response)
        return self._model
