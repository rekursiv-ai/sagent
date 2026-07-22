"""Per-user filesystem locations following OS conventions.

Rolled in-house rather than depending on ``platformdirs``: the surface
we need is a handful of lines of platform branching, and a 100KB
third-party module earns its keep only when it handles complexity the
caller cannot trivially reproduce. The corners ``platformdirs``
covers that we skip -- AppData redirection via ``SHGetKnownFolderPath``,
roaming profiles, appauthor/version subdirs, Android, iOS -- do not
apply to our development tools.
"""

from __future__ import annotations

from pathlib import Path

import os
import sys


__all__ = [
    "cache_dir",
    "config_dir",
    "data_dir",
    "resolve_working_dir",
    "state_dir",
]


def data_dir(app: str, platform: str = sys.platform) -> Path:
    """Resolve the per-user data directory for ``app``.

    POSIX systems honor an explicit ``XDG_DATA_HOME``. Without one, macOS uses
    ``Application Support`` and Linux/BSD use ``~/.local/share``. Windows uses
    ``LOCALAPPDATA``. The
    Windows branch reads the env var rather than calling
    ``SHGetKnownFolderPath``, so AppData redirected via group policy
    is not detected -- acceptable for development tools, not for
    shipped end-user software.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Absolute path to the application's data directory. The
        directory is not created.

    References:
      https://specifications.freedesktop.org/basedir-spec/latest/

    """
    if platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / app
    if xdg_data_home := os.environ.get("XDG_DATA_HOME"):
        return Path(xdg_data_home) / app
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app
    return Path.home() / ".local" / "share" / app


def config_dir(app: str, platform: str = sys.platform) -> Path:
    """Resolve the per-user config directory for ``app``.

    POSIX systems honor an explicit ``XDG_CONFIG_HOME``. Without one, this is
    distinct from :func:`data_dir` on Linux/BSD (``~/.config`` vs
    ``~/.local/share``), while macOS and Windows collapse the two locations.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Absolute path to the application's config directory. The
        directory is not created.

    References:
      https://specifications.freedesktop.org/basedir-spec/latest/

    """
    if platform == "win32":
        return data_dir(app, platform=platform)
    if xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg_config_home) / app
    if platform == "darwin":
        return data_dir(app, platform=platform)
    return Path.home() / ".config" / app


def cache_dir(app: str, platform: str = sys.platform) -> Path:
    """Resolve the per-user cache directory for ``app``.

    XDG ``$XDG_CACHE_HOME`` is for non-essential, regenerable data --
    downloaded model weights, build artifacts, memoized computation. POSIX
    systems honor an explicit override. Without one, the cache is distinct from
    :func:`data_dir` on Linux/BSD, while macOS and Windows use native locations.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Absolute path to the application's cache directory. The
        directory is not created.

    References:
      https://specifications.freedesktop.org/basedir-spec/latest/

    """
    if platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / app / "Cache"
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache_home) / app
    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / app
    return Path.home() / ".cache" / app


def state_dir(app: str, platform: str = sys.platform) -> Path:
    """Resolve the per-user state directory for ``app``.

    XDG ``$XDG_STATE_HOME`` is for state that should persist between runs but
    is not configuration or user data -- session captures, logs, undo
    histories. POSIX systems honor an explicit override. Without one, the
    state directory is distinct on Linux/BSD, while macOS and Windows use their
    native application-data location.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Absolute path to the application's state directory. The
        directory is not created.

    References:
      https://specifications.freedesktop.org/basedir-spec/latest/

    """
    if platform == "win32":
        return data_dir(app, platform=platform)
    if xdg_state_home := os.environ.get("XDG_STATE_HOME"):
        return Path(xdg_state_home) / app
    if platform == "darwin":
        return data_dir(app, platform=platform)
    return Path.home() / ".local" / "state" / app


def resolve_working_dir(
    base_dir: Path | str | None,
    working_dir: Path | str,
) -> Path:
    """Resolve a Config's ``working_dir`` against an optional ``base_dir``.

    The one path-composition rule every path-owning Config shares: when
    ``base_dir`` is ``None`` the ``working_dir`` is its own absolute logical
    root; otherwise ``working_dir`` is made relative (its leading slash
    stripped) and joined beneath ``base_dir``. Stripping the slash is required
    because ``Path("/a") / "/b" == Path("/b")`` -- an absolute right operand
    discards the base (POSIX semantics), so a logical root like
    ``"/checkpoints"`` would otherwise ignore its owner's ``base_dir``.

    Args:
      base_dir: Owner-supplied root, or ``None`` when the Config is its own root.
      working_dir: The Config's opinionated logical location.

    Returns:
      resolved: ``working_dir`` as a ``Path``, beneath ``base_dir`` when given.

    """
    if base_dir is None:
        return Path(working_dir)
    return Path(base_dir) / str(working_dir).lstrip("/")
