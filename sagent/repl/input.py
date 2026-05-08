"""``PromptInputHandler`` + ``LoginHandler`` + ``InputSource`` abstraction.

The input handler is a long-running spawned handler. It loops on an
``InputSource`` (real prompt-toolkit in production; ``StubInputSource``
in tests), classifies each line via :func:`repl.slash.parse_slash`, and
dispatches the resulting :class:`SlashAction` to ``agent.inbox``.

Slash commands flow through the FIFO with typed text so user intent
order is preserved. Only urgent actions (``/clear``, ``/abort``) jump
the queue -- they exist to preempt in-flight model/tool work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

from sagent import providers as _providers
from sagent.agent.handlers.base import (
    InlineHandler,
    SpawnedHandler,
)
from sagent.custom_types import TextMessage
from sagent.repl.slash import (
    QUIT_WORDS,
    SlashAction,
    dispatch,
    parse_slash,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message
    from sagent.repl.render import Printer


__all__ = [
    "QUIT_WORDS",
    "InputSource",
    "LoginHandler",
    "PromptInputHandler",
    "StubInputSource",
]


class InputSource(Protocol):
    """Source of user input lines.

    Implementations include a real prompt-toolkit session and the
    in-process :class:`StubInputSource` used by tests.
    """

    async def next_line(self) -> str | None:
        """Return the next line, or ``None`` to terminate the input loop."""
        ...


class StubInputSource:
    """In-process queue of pre-staged lines for tests.

    Attributes:
      lines: Remaining lines to deliver, in order. ``None`` ends the loop.

    """

    def __init__(self, lines: list[str | None]) -> None:
        self._lines: list[str | None] = list(lines)

    async def next_line(self) -> str | None:
        if not self._lines:
            return None
        return self._lines.pop(0)


class PromptInputHandler(SpawnedHandler):
    """Long-running spawned handler that reads input and posts to the inbox.

    Registers against ``text/x-bootstrap`` so the dispatch loop fires
    it once at startup; the handler then reads lines forever (or until
    the source returns ``None``). The CLI orchestrator posts a single
    ``text/x-bootstrap`` after constructing the agent to start input.
    """

    descriptors: tuple[str, ...] = ("text/x-bootstrap",)

    def __init__(
        self,
        source: InputSource,
        *,
        printer: Printer | None = None,
    ) -> None:
        self._source = source
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        while True:
            line = await self._source.next_line()
            if line is None:
                _ = agent.inbox.put(TextMessage("", "text/x-quit"))
                return
            action = parse_slash(line)
            if action is None:
                continue
            if action.descriptor == "text/x-user-message":
                # Pass the original (untrimmed) line through.
                _ = agent.inbox.put(TextMessage(line, "text/x-user-message"))
                continue
            self._apply(agent, action)
            if action.quit:
                return

    def _apply(self, agent: Agent, action: SlashAction) -> None:
        """Dispatch ``action`` and emit any echo line."""
        dispatch(agent, action)
        if action.echo is not None and self._printer is not None:
            self._printer.write_line(action.echo)


class LoginHandler(InlineHandler):
    """Handle ``text/x-login-request`` by invoking the provider's ``login``.

    Slash-command parsing emits ``text/x-login-request`` so both REPL
    paths (active keybinding, idle prompt) take the same code path; the
    actual re-auth lives here.
    """

    descriptors: tuple[str, ...] = ("text/x-login-request",)

    def __init__(self, printer: Printer | None = None) -> None:
        self._printer = printer

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        del msg
        spec = agent.model_spec
        if spec is None:
            self._write("[/login] agent has no model spec")
            return
        prov_cls = getattr(_providers, spec.provider, None)
        if prov_cls is None:
            self._write(f"[/login] unknown provider {spec.provider!r}")
            return
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            self._write(f"[/login] {spec.provider} has no login method")
            return
        try:
            login_fn()
            self._write(f"[/login] {spec.provider} re-authenticated")
        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
            self._write(f"[/login] {exc}")

    def _write(self, line: str) -> None:
        if self._printer is not None:
            self._printer.write_line(line)
