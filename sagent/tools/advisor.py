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

Usage::

    from sagent.tools.advisor import Advisor
    from sagent.agent import Agent
    from sagent.providers import resolve_model

    _, sonnet = resolve_model("anthropic", "claude-sonnet-4-6")
    _, opus = resolve_model("anthropic", "claude-opus-4-7")

    advisor = Advisor(model=opus)  # max_uses=None → unlimited
    agent = Agent(model=sonnet, tools=[..., advisor])

Or via the CLI::

    ./cli.py --model claude-sonnet-4-6 --advisor claude-opus-4-7
"""

from __future__ import annotations

from sagent.agent import Agent
from sagent.custom_types import Message, Model, TextMessage
from sagent.lib import debug_log
from sagent.lib.json import JSON, json_freeze
from sagent.lib.message import get_directive


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


# System-prompt section added when the advisor is enabled. Tool
# descriptions are low-salience vs system text - this nudge tells
# the executor concretely when to reach for the advisor.
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


class Advisor:
    """``Tool`` wrapper exposing an LLM as an advisor sub-agent.

    Each call runs a fresh ``Agent`` with empty history, so advice is
    independent across consults. ``max_uses`` caps invocations per
    instance (``None`` = unlimited - bounded only by the host agent's
    ``--max-tool-call-rounds`` CLI flag).
    """

    name: str = "advisor"
    tool_id: str = "application/x-tool-advisor"
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
    supports_microcompaction: bool = False

    def summary(self, msg: Message) -> str:
        """Return a short label for this advisor consultation.

        Args:
          msg: Tool call message (unused).

        Returns:
          label: Static "Advisor consulting…" string.

        """
        del msg
        return "Advisor consulting…"

    def prompt(self) -> str:
        """Return the system prompt nudge for advisor usage.

        Returns:
          prompt: Multi-paragraph guidance on when to consult the advisor.

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

    async def run(self, msg: Message) -> Message:
        """Run a fresh sub-agent consultation and return the advice.

        Args:
          msg: Tool call message with ``prompt`` field.

        Returns:
          result: Advisor's response or quota-exhausted error.

        """
        directive = get_directive(msg)
        prompt = str(directive.get("prompt", ""))
        if self._max_uses is not None and self._uses >= self._max_uses:
            return TextMessage(
                f"Advisor quota exhausted ({self._max_uses} uses).",
                "text/x-error",
            )
        self._uses += 1
        debug_log.trace(
            "advisor_invoke",
            model=self._model.model_id,
            prompt_len=len(prompt),
            uses=self._uses,
            max_uses=self._max_uses,
        )
        sub = Agent(
            name="advisor",
            description="Advisor sub-agent.",
            model=self._model,
            system=self._system,
            tools=[],
        )
        return await sub.run(json_freeze({"prompt": prompt}))
