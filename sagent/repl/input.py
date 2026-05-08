"""``PromptInputHandler`` and ``InputSource`` abstraction.

The input handler is a long-running spawned handler. It loops on an
``InputSource`` (real prompt-toolkit in production; ``StubInputSource``
in tests), classifies each line, and posts the appropriate descriptor
to ``agent.inbox``::

    line == "/quit"               ->  text/x-quit            (FIFO)
    line == "/clear"              ->  text/x-clear-request   (FIFO)
    line == "/compact" or
    line.startswith("/compact ")  ->  text/x-compact-request (FIFO)
    line == "/uncompact" or
    line.startswith("/uncompact ")->  text/x-uncompact-request
    line.startswith("/abort")     ->  text/x-abort           (put_left)
    line == ""                    ->  ignore
    other                         ->  text/x-user-message    (FIFO)

Slash commands flow through the FIFO with typed text so user intent
order is preserved. Only ``/abort`` jumps the queue -- it exists to
interrupt in-flight model/tool work, not to displace queued input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

from sagent import providers as _providers
from sagent.agent.handlers.base import SpawnedHandler
from sagent.custom_types import TextMessage


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message
    from sagent.repl.render import Printer


_QUIT_WORDS = ("/quit",)


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
                agent.inbox.put(TextMessage("", "text/x-quit"))
                return
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in _QUIT_WORDS:
                agent.inbox.put(TextMessage("", "text/x-quit"))
                return
            if stripped == "/clear" or stripped.startswith("/clear "):
                # /clear is urgent (matches v1): wipe before any
                # queued messages so the user's intent is honored even
                # while a model call is in flight.
                reason = stripped.removeprefix("/clear").strip()
                agent.inbox.put_left(
                    TextMessage(reason, "text/x-clear-request"),
                )
                note = f" ({reason})" if reason else ""
                self._echo(f"[/clear] history cleared{note}")
                continue
            if stripped == "/compact" or stripped.startswith("/compact "):
                instructions = stripped.removeprefix("/compact").strip()
                agent.inbox.put(
                    TextMessage(instructions, "text/x-compact-request"),
                )
                note = f" ({instructions})" if instructions else ""
                self._echo(f"[/compact] queued{note}")
                continue
            if stripped == "/uncompact" or stripped.startswith("/uncompact "):
                instructions = stripped.removeprefix("/uncompact").strip()
                agent.inbox.put(
                    TextMessage(instructions, "text/x-uncompact-request"),
                )
                note = f" ({instructions})" if instructions else ""
                self._echo(f"[/uncompact] queued{note}")
                continue
            if stripped == "/model" or stripped.startswith("/model "):
                args_str = stripped.removeprefix("/model").strip()
                agent.inbox.put(
                    TextMessage(args_str, "text/x-model-switch-request"),
                )
                # ModelSwitchHandler echoes its own status.
                continue
            if stripped == "/provider" or stripped.startswith("/provider "):
                # Sugar for ``/model --provider PROV [model_id]``.
                rest = stripped.removeprefix("/provider").strip()
                args_str = f"--provider {rest}" if rest else ""
                agent.inbox.put(
                    TextMessage(args_str, "text/x-model-switch-request"),
                )
                continue
            if stripped.startswith("/abort"):
                agent.inbox.put_left(TextMessage("", "text/x-abort"))
                self._echo("[/abort] cancelling in-flight tasks")
                continue
            if stripped == "/login":
                self._do_login(agent)
                continue
            if stripped.startswith("/"):
                # Unknown slash command. Surface to the user; do NOT
                # send to the LLM.
                cmd = stripped.split(maxsplit=1)[0]
                agent.inbox.put(
                    TextMessage(
                        f"unknown command: {cmd}. Supported: "
                        "/clear /compact /uncompact /model /provider "
                        "/abort /quit",
                        "text/x-error",
                    ),
                )
                continue
            agent.inbox.put(TextMessage(line, "text/x-user-message"))

    def _echo(self, line: str) -> None:
        """Write a confirmation line to the printer if one was supplied."""
        if self._printer is not None:
            self._printer.write_line(line)

    def _do_login(self, agent: Agent) -> None:
        """Re-authenticate the current provider via its ``login`` classmethod."""
        spec = agent.model_spec
        if spec is None:
            self._echo("[/login] agent has no model spec")
            return
        prov_name = spec.provider
        prov_cls = getattr(_providers, prov_name, None)
        if prov_cls is None:
            self._echo(f"[/login] unknown provider {prov_name!r}")
            return
        login_fn = getattr(prov_cls, "login", None)
        if login_fn is None:
            self._echo(f"[/login] {prov_name} has no login method")
            return
        try:
            login_fn()
            self._echo(f"[/login] {prov_name} re-authenticated")
        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
            self._echo(f"[/login] {exc}")
