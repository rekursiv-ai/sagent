"""Utilities for per-user filesystem locations following OS conventions.

These helpers satisfy the XDG Base Directory Specification:
https://specifications.freedesktop.org/basedir/latest/

Each function reads one environment variable and falls back to a platform
default when it is unset *or empty*, per spec:

==========  =================  ==============  =============================
Function    Env var (POSIX)    Linux/BSD       macOS
==========  =================  ==============  =============================
data_dir    XDG_DATA_HOME      ~/.local/share  ~/Library/Application Support
config_dir  XDG_CONFIG_HOME    ~/.config       ~/Library/Application Support
cache_dir   XDG_CACHE_HOME     ~/.cache        ~/Library/Caches
state_dir   XDG_STATE_HOME     ~/.local/state  ~/Library/Application Support
==========  =================  ==============  =============================

On Windows all four read ``LOCALAPPDATA`` (default ``~/AppData/Local``) and
ignore the XDG variables; only ``cache_dir`` differs, appending a ``Cache``
leaf. That branch reads the variable rather than calling
``SHGetKnownFolderPath``, so AppData redirected via group policy is not
detected -- acceptable for development tools, not for shipped end-user
software.

Two deliberate deviations from the specification:

* A relative ``XDG_*_HOME`` is honored, where the spec says to treat it as
  invalid and ignore it. Tests and sandboxes point these at relative scratch
  dirs, and silently substituting ``$HOME`` would write outside the sandbox --
  a wrong path that is loud beats a right-looking one that escapes isolation.
* The ``XDG_*`` variables are honored on macOS too, ahead of
  ``Library/Application Support``. The spec is silent on macOS; an operator who
  exports the variable means it on every platform.

Consequently the returned path is absolute only when the consulted variable is
absolute, which is the usual case but not guaranteed.

Note: we opted to not use the third-party ``platformdirs`` because we only need
a very tiny surface: four base directories on Linux, macOS, and Windows.
"""

from __future__ import annotations

from pathlib import Path

import os
import sys


__all__ = [
    "cache_dir",
    "config_dir",
    "data_dir",
    "state_dir",
]


def data_dir(app: str, platform: str = sys.platform) -> Path:
    """Resolve the per-user data directory for ``app``.

    Holds user-specific data files: the durable, portable content a user would
    expect to keep -- the things worth backing up and carrying to another
    machine. State that is merely resumable belongs in :func:`state_dir`, and
    regenerable content in :func:`cache_dir`.

    Reads ``XDG_DATA_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Path to the application's data directory, absolute unless the
        consulted environment variable was relative. The directory is not
        created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

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

    Holds user-specific configuration files: settings the user chose, which the
    application reads to decide how to behave. Anything the application itself
    wrote to remember what it was doing is state, not config.

    Reads ``XDG_CONFIG_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Path to the application's config directory, absolute unless the
        consulted environment variable was relative. The directory is not
        created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

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

    Holds user-specific non-essential (cached) data -- downloaded model
    weights, build artifacts, memoized computation. Every file here must be
    reconstructible: deleting the whole tree may cost time but must not lose
    anything.

    Reads ``XDG_CACHE_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Path to the application's cache directory, absolute unless the
        consulted environment variable was relative. The directory is not
        created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

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

    Holds state that should persist across restarts but is not important or
    portable enough for :func:`data_dir`. The spec names two kinds: action
    history (logs, recently-used files, session captures) and the state needed
    to resume where the application left off (view, layout, open files, undo
    history).

    Reads ``XDG_STATE_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      app: Application name. Used as the leaf directory.
      platform: ``sys.platform`` string. Override for testing; the
        default closes over the host's ``sys.platform``.

    Returns:
      path: Path to the application's state directory, absolute unless the
        consulted environment variable was relative. The directory is not
        created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

    """
    if platform == "win32":
        return data_dir(app, platform=platform)
    if xdg_state_home := os.environ.get("XDG_STATE_HOME"):
        return Path(xdg_state_home) / app
    if platform == "darwin":
        return data_dir(app, platform=platform)
    return Path.home() / ".local" / "state" / app
