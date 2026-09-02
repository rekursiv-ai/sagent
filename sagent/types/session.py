"""Where one agent's history is persisted, and what identifies it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import uuid


__all__ = [
    "Session",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    """One agent's identity and, optionally, where it is written down.

    A directory NAMES its session, so ``dir.name`` and ``id`` must agree;
    ``__post_init__`` enforces it.
    """

    id: str
    """Stable identifier for this agent's tape and registry entry."""

    dir: Path | None = None
    """Directory holding the transcript; ``None`` disables persistence."""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Session.id must be non-empty")
        if self.dir is not None and self.dir.name != self.id:
            raise ValueError(
                f"Session.dir must be named for its session: dir={self.dir.name!r}"
                f" but id={self.id!r}"
            )

    @classmethod
    def at(cls, path: Path | str) -> Session:
        """Build a persisted session from the directory that will hold it.

        Args:
          path: Directory to persist into; its name becomes the id.

        Returns:
          session: Persisted session identified by ``path``'s name.

        """
        directory = Path(path)
        return cls(id=directory.name, dir=directory)

    @classmethod
    def ephemeral(cls) -> Session:
        """Build an unpersisted session with a fresh id.

        Returns:
          session: In-memory session; ``dir`` is ``None``.

        """
        return cls(id=uuid.uuid4().hex[:8])
