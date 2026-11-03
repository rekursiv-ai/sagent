"""A journal stored as one JSON line per record, locked to one process at a time.

Each line is a record as :func:`~sagent.session.types.stored_plain` writes
it, read back through :func:`~sagent.session.types.stored_record`.

Typical usage example:

  async with JsonlJournal(path) as journal:     # A second open fails.
      list(journal)                              # The last ContextClear onward.
      list(journal.history(start=n))             # Record n onward; none before read.
      journal.append(UserMessage(content="hi"))
      journal.flush()                            # On disk, even after a power cut.

  read_record(line)                              # One line, as a follower reads it.
  record_line(record)                            # The line a journal writes for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import fcntl
import itertools
import json
import os

from treekle import json_loads

from sagent.lib.files.owner_only import open_private
from sagent.lib.posix.pids import lock_holder
from sagent.session.types import (
    ContextClear,
    IncompleteRecord,
    SessionRecord,
    stored_plain,
    stored_record,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path


__all__ = [
    "JournalInUseError",
    "JsonlJournal",
    "read_record",
    "record_line",
]


class JournalInUseError(RuntimeError):
    """Another process holds the journal open."""


class JsonlJournal(AsyncResource):
    """A journal in a JSON-lines file, the session's directory naming the session.

    Args:
      path: The file; it and its directory are made private when created.

    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self._window: list[SessionRecord] = []
        # Where each record's line starts in the file, so a read from record n
        # decodes none before it; the lock keeps any other writer out.
        self._starts: list[int] = []
        self._size = 0

    @property
    def path(self) -> Path:
        """The file."""
        return self._path

    @override
    async def aopen(self) -> None:
        self._fd = _open_locked(self._path)
        self._window = _last_window(self._path)
        self._starts, self._size = _line_starts(self._path)

    @override
    async def aclose(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.fsync(fd)
            os.close(fd)

    def append(self, *records: SessionRecord) -> None:
        """Write ``records`` after the last; a ``ContextClear`` starts a new window."""
        self._write(records)
        self._keep(records)

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield the records from the last ``ContextClear`` on."""
        return iter(tuple(self._window))

    def __len__(self) -> int:
        """Return how many records ``history`` yields."""
        return len(self._starts)

    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        """Yield the file's records from record ``start``; a torn line as incomplete.

        The last window is held in memory; earlier records are read from the file
        as they are iterated, so a caller may iterate on a worker thread.

        Args:
          start: The number of the first record; those before it are not read.

        Returns:
          records: The file's records from ``start`` as of this call, oldest first.

        Raises:
          RuntimeError: The journal is not open.

        """
        if self._fd is None:
            raise RuntimeError(f"Journal {self._path} is not open.")
        first = len(self._starts) - len(self._window)
        window = tuple(self._window[max(0, start - first) :])
        if start >= first:
            return iter(window)
        end = self._starts[first] if first < len(self._starts) else self._size
        return itertools.chain(_read(self._path, self._starts[start], end), window)

    def flush(self) -> None:
        """Make what was written survive a power cut."""
        if self._fd is not None:
            os.fsync(self._fd)

    def _write(self, records: Sequence[SessionRecord]) -> None:
        if self._fd is None:
            raise RuntimeError(f"Journal {self._path} is not open.")
        each = [record_line(record) for record in records]
        # A write that raised partway, on a full disk, left lines ``_starts`` does
        # not hold, the last perhaps torn: they are indexed, and a torn one ended,
        # or the next record would join it and later ones start before their index.
        ending = b""
        if os.fstat(self._fd).st_size != self._size:
            starts, end = _line_starts(self._path, offset=self._size)
            self._keep(list(_read(self._path, self._size, end)))
            self._starts.extend(starts)
            self._size = end
            if os.pread(self._fd, 1, self._size - 1) != b"\n":
                ending = b"\n"
        # ``os.write`` may stop short, on a full disk or a signal; the rest of
        # the line must follow, or the next record joins a torn one.
        rest = memoryview(ending + b"".join(each))
        while rest:
            written = os.write(self._fd, rest)
            if not written:
                raise OSError(
                    f"Journal {self._path} wrote nothing; the disk may be full.",
                )
            rest = rest[written:]
        self._size += len(ending)
        for line in each:
            self._starts.append(self._size)
            self._size += len(line)

    # ``history`` serves record n from the window when n is in it, so the window
    # holds exactly the records ``_starts`` counts from the last ``ContextClear``.
    def _keep(self, records: Iterable[SessionRecord]) -> None:
        """Add ``records`` to the window; a ``ContextClear`` starts a new one."""
        for record in records:
            if isinstance(record, ContextClear):
                self._window = []
            self._window.append(record)


def read_record(line: bytes) -> SessionRecord:
    """Return the record one line of a journal holds.

    Args:
      line: One line of the file, with or without its newline.

    Returns:
      record: The record; an ``IncompleteRecord`` for a line torn or not a record.

    """
    return _record(line.removesuffix(b"\n"))


def record_line(record: SessionRecord) -> bytes:
    """Return the line a journal writes for ``record``, which :func:`read_record` reads.

    Args:
      record: The record.

    Returns:
      line: Its JSON, newline included.

    """
    # ``projects.py`` finds a session's title and model by each line's first bytes,
    # ``{"py/object": "UserMessage"``: ``json.dumps``'s default separators.
    return (json.dumps(stored_plain(record)) + "\n").encode()


def _open_locked(path: Path) -> int:
    """Open ``path`` for appending, locked against any other process."""
    fd = open_private(path, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        holder = lock_holder(path)
        where = "another process" if holder is None else f"pid {holder}"
        raise JournalInUseError(f"Journal {path} is open in {where}.") from None
    # A crash mid-write leaves a line without its newline; ending it keeps the
    # next record from joining the torn one into a line nothing can read.
    size = os.fstat(fd).st_size
    if size and os.pread(fd, 1, size - 1) != b"\n":
        os.write(fd, b"\n")
    return fd


# Blank lines are skipped as ``history`` skips them, or record numbers would shift.
def _line_starts(path: Path, *, offset: int = 0) -> tuple[list[int], int]:
    """Return where each record's line from ``offset`` starts in ``path``, and its size."""
    starts: list[int] = []
    size = offset
    with path.open("rb") as file:
        _ = file.seek(offset)
        for line in file:
            if line.strip():
                starts.append(size)
            size += len(line)
    return starts, size


def _read(path: Path, begin: int, end: int) -> Iterator[SessionRecord]:
    """Yield the records of ``path``'s lines from byte ``begin`` to ``end``, when asked."""
    with path.open("rb") as file:
        _ = file.seek(begin)
        data = file.read(end - begin)
    for line in data.split(b"\n"):
        if line.strip():
            yield _record(line)


def _last_window(path: Path) -> list[SessionRecord]:
    """Return the records from the last ``ContextClear`` on, reading backward."""
    window: list[SessionRecord] = []
    for line in _reversed_lines(path):
        record = _record(line)
        window.append(record)
        if isinstance(record, ContextClear):
            break
    window.reverse()
    return window


def _record(line: bytes) -> SessionRecord:
    try:
        return stored_record(json_loads(line))
    except ValueError:
        return IncompleteRecord(text=line.decode(errors="replace").rstrip("\n"))


def _reversed_lines(path: Path) -> Iterator[bytes]:
    # ``b"\n"`` never occurs inside a multibyte UTF-8 character, so splitting
    # raw chunks on it never cuts a character.
    with path.open("rb") as file:
        position = file.seek(0, os.SEEK_END)
        tail = b""
        while position:
            step = min(1 << 20, position)
            position -= step
            file.seek(position)
            lines = (file.read(step) + tail).split(b"\n")
            tail = lines.pop(0)
            yield from (line for line in reversed(lines) if line.strip())
        if tail.strip():
            yield tail
