"""A journal stored as one JSON line per record, locked to one process at a time.

Each record's JSON fields are written as the plain objects they hold, as
:data:`~sagent.session.types.CONCRETE_TYPES` says, and read back frozen.

Typical usage example:

  async with JsonlJournal(path) as journal:     # A second open fails.
      list(journal)                              # The last ContextClear onward.
      journal.append(UserMessage(content="hi"))
      journal.flush()                            # On disk, even after a power cut.

  read_record(line)                              # One line, as a follower reads it.
  record_line(record)                            # The line a journal writes for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import fcntl
import os

from treekle import json_dumps, json_loads

from sagent.session.types import (
    CONCRETE_TYPES,
    ContextClear,
    IncompleteRecord,
    SessionRecord,
    record_classes,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
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
        self._classes = record_classes()

    @property
    def path(self) -> Path:
        """The file."""
        return self._path

    @override
    async def aopen(self) -> None:
        self._fd = _open_locked(self._path)
        self._window = _last_window(self._path, classes=self._classes)

    @override
    async def aclose(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.fsync(fd)
            os.close(fd)

    def append(self, *records: SessionRecord) -> None:
        """Write ``records`` after the last; a ``ContextClear`` starts a new window."""
        self._write(records)
        for record in records:
            if isinstance(record, ContextClear):
                self._window = []
            self._window.append(record)

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield the records from the last ``ContextClear`` on."""
        return iter(tuple(self._window))

    def history(self) -> Iterator[SessionRecord]:
        """Yield every record in the file, from the first; a torn line as incomplete.

        Returns:
          records: The file's records, oldest first.

        Raises:
          RuntimeError: The journal is not open.

        """
        if self._fd is None:
            raise RuntimeError(f"Journal {self._path} is not open.")
        with self._path.open("rb") as file:
            records = [
                _record(line, classes=self._classes) for line in file if line.strip()
            ]
        return iter(records)

    def flush(self) -> None:
        """Make what was written survive a power cut."""
        if self._fd is not None:
            os.fsync(self._fd)

    def _write(self, records: Sequence[SessionRecord]) -> None:
        if self._fd is None:
            raise RuntimeError(f"Journal {self._path} is not open.")
        lines = b"".join(_line(record, classes=self._classes) for record in records)
        # ``os.write`` may stop short, on a full disk or a signal; the rest of
        # the line must follow, or the next record joins a torn one.
        rest = memoryview(lines)
        while rest:
            written = os.write(self._fd, rest)
            if not written:
                raise OSError(
                    f"Journal {self._path} wrote nothing; the disk may be full.",
                )
            rest = rest[written:]


def read_record(line: bytes) -> SessionRecord:
    """Return the record one line of a journal holds.

    Args:
      line: One line of the file, with or without its newline.

    Returns:
      record: The record; an ``IncompleteRecord`` for a line torn or not a record.

    """
    return _record(line.removesuffix(b"\n"), classes=record_classes())


def record_line(record: SessionRecord) -> bytes:
    """Return the line a journal writes for ``record``, which :func:`read_record` reads.

    Args:
      record: The record.

    Returns:
      line: Its JSON, newline included.

    """
    return _line(record, classes=record_classes())


def _line(record: SessionRecord, *, classes: Mapping[str, type]) -> bytes:
    return (
        json_dumps(
            record,
            classes=classes,
            sort=False,
            default_concrete_types=CONCRETE_TYPES,
        )
        + "\n"
    ).encode()


def _open_locked(path: Path) -> int:
    """Open ``path`` for appending, locked against any other process."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise JournalInUseError(f"Journal {path} is open in another process.") from None
    # ``O_CREAT``'s mode applies only to a new file; one made before, or by
    # another tool, keeps whatever it had.
    os.fchmod(fd, 0o600)
    # A crash mid-write leaves a line without its newline; ending it keeps the
    # next record from joining the torn one into a line nothing can read.
    size = os.fstat(fd).st_size
    if size and os.pread(fd, 1, size - 1) != b"\n":
        os.write(fd, b"\n")
    return fd


def _last_window(path: Path, *, classes: Mapping[str, type]) -> list[SessionRecord]:
    """Return the records from the last ``ContextClear`` on, reading backward."""
    window: list[SessionRecord] = []
    for line in _reversed_lines(path):
        record = _record(line, classes=classes)
        window.append(record)
        if isinstance(record, ContextClear):
            break
    window.reverse()
    return window


def _record(line: bytes, *, classes: Mapping[str, type]) -> SessionRecord:
    try:
        read = json_loads(
            line,
            SessionRecord,
            classes=classes,
            default_concrete_types=CONCRETE_TYPES,
        )
        return cast(SessionRecord, read)
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
            yield from (line for line in reversed(lines) if line)
        if tail:
            yield tail
