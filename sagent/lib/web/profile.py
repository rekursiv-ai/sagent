"""Cross-process browser profiles: a per-``(egress_ip, domain)`` cookie + UA jar.

A scraper that presents as a brand-new anonymous client on every request looks
exactly like a bot: it never sends back the session cookie a real browser keeps,
so the server's per-session budget never applies and it trips the anti-scrape
limit fast. A :class:`Profile` fixes that -- it persists the ``User-Agent`` and
the cookie jar a domain handed us, keyed by the public egress IP so a VPN
rotation yields a fresh identity, and shared across processes (an ``fcntl``-locked
JSON file under :func:`sagent.lib.web.userdirs.data_dir`) so several workers behind
one exit IP behave like one user with several browser tabs open.

This module is a pure data store and the cookie/UA vocabulary: it loads, saves,
and merges profiles, and parses/draws the values a profile holds. It owns no
orchestration and takes no dependency on ``fetch`` or egress resolution -- the
orchestrator (:func:`sagent.lib.web.fetch.fetch`) drives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import cache
from pathlib import Path
from urllib.parse import quote

import fcntl
import json
import os
import threading
import time

from sagent.lib.userdirs import data_dir


__all__ = [
    "Profile",
    "ProfileStore",
    "parse_set_cookie",
]


def parse_set_cookie(header_value: str) -> dict[str, str]:
    """Parse a ``Set-Cookie`` header value into live ``{name: value}`` pairs.

    Keeps only the leading ``name=value`` of each cookie, dropping attributes
    (``Path`` / ``Domain`` / ``Secure`` / ``SameSite`` / ...): we are the client
    deciding what to send back, not a browser enforcing scope. A ``Max-Age`` of
    zero or less, or a past ``Expires``, marks a deletion and is omitted.

    Multiple ``Set-Cookie`` headers are newline-separated by
    :func:`sagent.lib.web.fetch.common.join_headers` -- Set-Cookie is RFC-exempt
    from comma folding because a cookie value may itself contain ``", "``, so a
    newline (never present in a header value) is the unambiguous separator.

    Args:
      header_value: The (possibly newline-separated) ``Set-Cookie`` header value.

    Returns:
      cookies: Live cookie name-value pairs; empty when none are live.

    """
    cookies: dict[str, str] = {}
    for chunk in header_value.split("\n"):
        parts = chunk.split(";")
        head = parts[0].strip()
        if "=" not in head:
            continue
        name, value = head.split("=", 1)
        name = name.strip()
        if not name:
            continue
        if _is_deletion(parts[1:]):
            continue
        cookies[name] = value.strip()
    return cookies


@dataclass(frozen=True, slots=True, kw_only=True)
class Profile:
    """One browsing identity for a ``(egress_ip, domain)``: its UA and cookies.

    Attributes:
      ua: The ``User-Agent`` this identity presents. Frozen for its life so a
        stable cookie never rides a shifting UA (a bot tell).
      cookies: Live cookie jar (``name -> value``) for the domain.
      created: Unix time the identity was minted, for TTL eviction.

    """

    ua: str
    cookies: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class ProfileStore:
    """Persists :class:`Profile` per ``(egress_ip, domain)`` across processes.

    Each key maps to one ``fcntl``-locked JSON file under ``base_dir``. Reads
    honor ``ttl_sec`` (a profile older than the TTL is treated as absent, so a
    stale identity is retired rather than reused). Writes are a locked
    read-modify-write, so concurrent workers behind one exit IP never corrupt
    the jar.

    Args:
      base_dir: Directory holding the per-key files. Defaults to
        ``data_dir("web_profiles")``. Created on first write.
      ttl_sec: Seconds a profile stays valid after ``created``. Default 12h.

    """

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        ttl_sec: float = 12 * 3600.0,
    ) -> None:
        self._base = data_dir("sagent.lib.web") if base_dir is None else base_dir
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()

    @classmethod
    @cache
    def shared(cls) -> ProfileStore:
        """The process-wide store, built once (the default for every fetch)."""
        return cls()

    def load(self, egress_ip: str, domain: str) -> Profile | None:
        """Return the live profile for the key, or ``None`` if absent or stale.

        A corrupt or partial file (a crash mid-write, a manual edit, a disk
        error) is treated as absent -- it self-heals on the next :meth:`save` --
        so a bad file never propagates an exception out of a fetch. An
        expired file is unlinked, not merely ignored, so stale keys don't
        accumulate.
        """
        path = self._path(egress_ip, domain)
        with self._lock:
            raw = self._read(path)
            if raw is None:
                return None
            profile = _try_decode(raw)
            if profile is None:
                return None  # corrupt: absent, self-heals on next save.
            if time.time() - profile.created > self._ttl_sec:
                path.unlink(missing_ok=True)
                return None
            return profile

    def save(self, egress_ip: str, domain: str, profile: Profile) -> None:
        """Persist ``profile`` for the key, replacing any prior one."""
        path = self._path(egress_ip, domain)
        with self._lock:
            self._write(path, _encode(profile))

    def discard(self, egress_ip: str, domain: str) -> None:
        """Delete the profile for the key (a burn); absent keys are a no-op."""
        path = self._path(egress_ip, domain)
        with self._lock:
            path.unlink(missing_ok=True)

    def update_cookies(
        self, egress_ip: str, domain: str, cookies: dict[str, str]
    ) -> None:
        """Merge ``cookies`` into an existing key's jar (locked read-modify-write).

        Preserves prior cookies (new names add, repeated names overwrite). A
        no-op when no profile exists for the key -- a jar has no identity without
        a User-Agent, and the store does not mint one; :meth:`save` establishes
        the profile first.
        """
        if not cookies:
            return
        path = self._path(egress_ip, domain)
        with self._lock:
            raw = self._read(path)
            if raw is None:
                return
            profile = _try_decode(raw)
            if profile is None:
                return  # corrupt: nothing to merge into; heals on next save.
            merged = {**profile.cookies, **cookies}
            self._write(
                path,
                _encode(
                    Profile(ua=profile.ua, cookies=merged, created=profile.created)
                ),
            )

    def _path(self, egress_ip: str, domain: str) -> Path:
        # One file per identity. Percent-encode each part before joining so an
        # IPv6 literal's colons (in the egress OR the domain) neither collide
        # with the "|" separator nor produce a colon in the filename (illegal on
        # some filesystems). quote(safe="") escapes ":" -> "%3A", so distinct
        # (egress, domain) pairs never map to one path.
        key = f"{quote(egress_ip, safe='')}|{quote(domain, safe='')}"
        return self._base / f"{key}.json"

    def _read(self, path: Path, *, max_bytes: int = 1 << 20) -> bytes | None:
        """Read a key's bytes under the file lock, or ``None`` when absent.

        ``max_bytes`` (default 1 MiB) caps a runaway file; cookie jars are small.
        """
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            data = os.read(fd, max_bytes)
        finally:
            os.close(fd)
        return data or None

    def _write(self, path: Path, data: bytes) -> None:
        """Atomically replace a key's file: write a temp sibling, then rename.

        ``os.replace`` is atomic on POSIX, so a concurrent reader (or a crash)
        sees either the complete old file or the complete new one -- never a
        truncated partial. A crash after the temp is written but before the
        rename leaves only the stray temp (cleaned on the next successful write
        to the key), never a corrupt file at the key path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # A per-pid temp name avoids two writers colliding on one temp file; the
        # atomic rename serializes them so the key always holds a complete file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
        try:
            _ = os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.replace(path)


def _is_deletion(attributes: list[str]) -> bool:
    """Whether cookie attributes mark it expired (past ``Max-Age`` or ``Expires``).

    ``Max-Age`` wins when present (RFC 6265 precedence); otherwise a past
    ``Expires`` date is a deletion. A malformed value is treated as non-deletion
    (keep the cookie) rather than raising.
    """
    for attr in attributes:
        key, _, value = attr.strip().partition("=")
        if key.strip().lower() == "max-age":
            try:
                return int(value.strip()) <= 0
            except ValueError:
                return False
    for attr in attributes:
        key, _, value = attr.strip().partition("=")
        if key.strip().lower() == "expires":
            expires = parsedate_to_datetime_or_none(value.strip())
            if expires is not None:
                return expires <= datetime.now(UTC)
    return False


def parsedate_to_datetime_or_none(value: str) -> datetime | None:
    """Parse an HTTP-date; return ``None`` on any malformed value."""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    # A naive datetime (no tz in the header) is interpreted as UTC, per HTTP.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _encode(profile: Profile) -> bytes:
    """Serialize a profile to JSON bytes."""
    return json.dumps(
        {
            "ua": profile.ua,
            "cookies": profile.cookies,
            "created": profile.created,
        }
    ).encode()


def _try_decode(raw: bytes) -> Profile | None:
    """Deserialize JSON bytes to a profile, or ``None`` if corrupt/malformed.

    A partial write, a manual edit, or a disk error yields bytes that are not a
    valid profile; the store treats those as absent so a bad file never raises
    out of a fetch (it self-heals on the next save).
    """
    try:
        obj = json.loads(raw)
        return Profile(
            ua=obj["ua"], cookies=dict(obj["cookies"]), created=obj["created"]
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
