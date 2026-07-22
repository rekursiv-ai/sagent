"""Persistent transport routing learned by automatic web fetches."""

from __future__ import annotations

from pathlib import Path

import fcntl
import logging
import os

from sagent.lib.userdirs import state_dir


logger = logging.getLogger(__name__)


__all__ = [
    "remember_zendriver_domain",
    "zendriver_domains",
    "zendriver_domains_path",
]


def _normalize(line: str) -> str:
    """Return the canonical (stripped, casefolded) form of one domain line."""
    return line.strip().casefold()


def _read_all(file_descriptor: int) -> bytes:
    """Read one open file descriptor to EOF."""
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_all(file_descriptor: int, data: bytes) -> None:
    """Write ``data`` fully, honoring short writes."""
    view = memoryview(data)
    while view:
        view = view[os.write(file_descriptor, view) :]


def zendriver_domains_path() -> Path:
    """Return the writable per-user automatic-Zendriver domain list."""
    return state_dir("loop") / "web" / "zendriver-domains.txt"


def _bundled_domains_path() -> Path:
    """Return the read-only domain defaults shipped alongside this module.

    Optional: ``_read_domains`` returns an empty set when the file is absent, so
    a checkout without it simply starts with no bundled defaults.
    """
    return Path(__file__).parent / "zendriver-domains.txt"


def _read_domains(path: Path) -> frozenset[str]:
    """Read one locked domain list, returning empty when absent or unreadable.

    The list is an optional, rebuildable cache on the per-fetch hot path, so a
    missing, permission-denied, or corrupt (non-UTF-8) file degrades to no
    learned routing rather than aborting every automatic fetch.
    """
    try:
        file_descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return frozenset()
    except OSError:
        logger.warning("Ignoring unreadable Zendriver domain list at %s.", path)
        return frozenset()
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_SH)
        raw = _read_all(file_descriptor)
    finally:
        os.close(file_descriptor)
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        logger.warning("Ignoring undecodable Zendriver domain list at %s.", path)
        return frozenset()
    return frozenset(
        normalized for line in text.splitlines() if (normalized := _normalize(line))
    )


def zendriver_domains(*, path: Path | None = None) -> frozenset[str]:
    """Return domains whose successful fallback established a browser requirement.

    Args:
      path: Domain-list path. Defaults to :func:`zendriver_domains_path`.

    Returns:
      domains: Normalized domains currently routed directly to Zendriver.

    """
    if path is not None:
        return _read_domains(path)
    return _read_domains(_bundled_domains_path()) | _read_domains(
        zendriver_domains_path()
    )


def remember_zendriver_domain(domain: str, *, path: Path | None = None) -> None:
    """Atomically add ``domain`` to the cross-process Zendriver domain list.

    Args:
      domain: DNS hostname whose browser fallback succeeded.
      path: Domain-list path. Defaults to :func:`zendriver_domains_path`.

    Raises:
      ValueError: If ``domain`` is empty or not one line.

    """
    normalized = domain.strip().casefold()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"Invalid Zendriver domain: {domain!r}.")
    target = zendriver_domains_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(
        target,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        existing = _read_all(file_descriptor).decode()
        domains = {
            value for line in existing.splitlines() if (value := _normalize(line))
        }
        if normalized in domains:
            return
        domains.add(normalized)
        payload = "".join(f"{value}\n" for value in sorted(domains)).encode()
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(file_descriptor, 0)
        _write_all(file_descriptor, payload)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
