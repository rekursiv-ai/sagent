"""Local, read-only inspection of commands and diffs hidden by previews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Protocol, Self, runtime_checkable


if TYPE_CHECKING:
    from types import TracebackType


@dataclass(frozen=True, slots=True)
class Detail:
    """One original renderer payload, stored outside the model context."""

    number: int
    title: str
    kind: Literal["command", "diff"]
    path: Path

    def read(self) -> str:
        """Read the original text without newline conversion."""
        with self.path.open(encoding="utf-8", newline="") as file:
            return file.read()


class ToolDetails:
    """Private temporary files keep large payloads out of the UI's memory.

    The store lives for one REPL invocation. Replay rebuilds it from the
    session tape, so IDs are local to the current invocation.
    """

    def __init__(self) -> None:
        self._directory: TemporaryDirectory[str] | None = None
        self.entries: list[Detail] = []

    def __enter__(self) -> Self:
        """Return the store owned by this REPL."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove temporary payloads even on cancellation."""
        if self._directory is not None:
            self._directory.cleanup()

    def add(self, title: str, kind: Literal["command", "diff"], text: str) -> Detail:
        """Save the exact text before any display shortening."""
        number = len(self.entries) + 1
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="sagent-details-")
        path = Path(self._directory.name) / f"{number}.txt"
        with path.open("w", encoding="utf-8", newline="") as file:
            path.chmod(0o600)
            _ = file.write(text)
        detail = Detail(number, " ".join(title.split())[:160], kind, path)
        self.entries.append(detail)
        return detail

    def get(self, number: int = 0) -> Detail | None:
        """Return a numbered entry, or the latest entry for zero."""
        if not self.entries:
            return None
        if number == 0:
            return self.entries[-1]
        if 1 <= number <= len(self.entries):
            return self.entries[number - 1]
        return None


@runtime_checkable
class DetailsPrinter(Protocol):
    """Optional terminal capability; headless printers need not implement it."""

    async def inspect_details(self, number: int = 0) -> None:
        """Open a local inspector without submitting input to the agent."""
        ...


def display_text(text: str) -> str:
    """Make terminal control characters visible while retaining the raw file."""
    return "".join(
        char
        if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159)
        else repr(char)[1:-1]
        for char in text
    )
