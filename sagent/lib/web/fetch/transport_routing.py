"""Persistent transport routing learned by automatic web fetches."""

from __future__ import annotations

from pathlib import Path

import fcntl
import os


__all__ = [
    "remember_zendriver_domain",
    "zendriver_domains",
    "zendriver_domains_path",
]


def zendriver_domains_path() -> Path:
    """Return the maintained automatic-Zendriver domain manifest path."""
    return Path(__file__).parent / "zendriver-domains.txt"


def zendriver_domains(*, path: Path | None = None) -> frozenset[str]:
    """Return domains whose successful fallback established a browser requirement.

    Args:
      path: Domain-list path. Defaults to :func:`zendriver_domains_path`.

    Returns:
      domains: Normalized domains currently routed directly to Zendriver.

    """
    target = zendriver_domains_path() if path is None else path
    try:
        file_descriptor = os.open(target, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return frozenset()
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_SH)
        text = os.read(file_descriptor, 1 << 20).decode()
    finally:
        os.close(file_descriptor)
    return frozenset(line for line in map(str.strip, text.splitlines()) if line)


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
        domains = set(os.read(file_descriptor, 1 << 20).decode().splitlines())
        if normalized in domains:
            return
        domains.add(normalized)
        payload = "".join(f"{value}\n" for value in sorted(domains)).encode()
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(file_descriptor, 0)
        _ = os.write(file_descriptor, payload)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
