"""A journal that keeps every record in a list, to be read back without a file.

An agent given no journal keeps its session in one, which nothing resumes; a
caller can also inspect a session in one, or resume one it built.

Typical usage example:

  journal = RecordingJournal()
  journal.append(UserMessage(content="hi"))
  list(journal)                                  # The last ContextClear onward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.session.types import ContextClear


if TYPE_CHECKING:
    from collections.abc import Iterator

    from sagent.session.types import SessionRecord


__all__ = [
    "RecordingJournal",
]


class RecordingJournal:
    """A journal with no file: it holds the records, and is gone when they are."""

    def __init__(self) -> None:
        self._records: list[SessionRecord] = []
        # Where the last window starts in ``_records``.
        self._start = 0

    @property
    def path(self) -> None:
        """No file."""
        return

    def append(self, *records: SessionRecord) -> None:
        """Keep ``records``; a ``ContextClear`` starts a new window."""
        for record in records:
            if isinstance(record, ContextClear):
                self._start = len(self._records)
            self._records.append(record)

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield the records from the last ``ContextClear`` on."""
        return iter(tuple(self._records[self._start :]))

    def __len__(self) -> int:
        """Return how many records ``history`` yields."""
        return len(self._records)

    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        """Yield every record kept, from record ``start``."""
        return iter(tuple(self._records[start:]))

    def flush(self) -> None:
        """Do nothing; nothing is buffered."""
