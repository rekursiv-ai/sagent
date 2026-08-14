"""Utilities for per-user filesystem locations following OS conventions.

These helpers satisfy the XDG Base Directory Specification:
https://specifications.freedesktop.org/basedir/latest/

Each function returns a BASE directory and takes no arguments. The
namespace is a path segment the caller joins, exactly like every other
segment::

    config_dir() / "rekursiv-ai" / "secrets"
    cache_dir() / "uv"                          # another vendor's dir

Each reads one environment variable and falls back to a platform default
when it is unset *or empty*, per spec:

==========  =================  ==============  =============================
Function    Env var (POSIX)    Linux/BSD       macOS
==========  =================  ==============  =============================
data_dir    XDG_DATA_HOME      ~/.local/share  ~/Library/Application Support
config_dir  XDG_CONFIG_HOME    ~/.config       ~/Library/Application Support
cache_dir   XDG_CACHE_HOME     ~/.cache        ~/Library/Caches
state_dir   XDG_STATE_HOME     ~/.local/state  ~/Library/Application Support
==========  =================  ==============  =============================

On Windows all four read ``LOCALAPPDATA`` (default ``~/AppData/Local``) and
ignore the XDG variables. That branch reads the variable rather than calling
``SHGetKnownFolderPath``, so AppData redirected via group policy is not
detected -- acceptable for development tools, not for shipped end-user
software.

Windows convention places a ``Cache`` leaf *below* the application name
(``…/Local/<app>/Cache``). A base directory has no application name to sit
above, so that leaf is not expressible here and ``cache_dir`` returns the
same ``LOCALAPPDATA`` root as the others. No shell caller runs on Windows
(see ``userdirs.sh``), and callers that care can append ``"Cache"`` after
their own namespace segment.

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


def data_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user data base directory.

    Holds user-specific data files: the durable, portable content a user would
    expect to keep -- the things worth backing up and carrying to another
    machine. State that is merely resumable belongs in :func:`state_dir`, and
    regenerable content in :func:`cache_dir`.

    Reads ``XDG_DATA_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      platform: ``sys.platform`` string. Override for testing; ``None``
        reads ``sys.platform`` at CALL time. A ``= sys.platform`` default
        would bind at import, freezing the value before any monkeypatch.

    Returns:
      path: The data base directory, absolute unless the consulted
        environment variable was relative. Join your own namespace segment;
        nothing is created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

    """
    platform = platform or sys.platform
    if platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if xdg_data_home := os.environ.get("XDG_DATA_HOME"):
        return Path(xdg_data_home)
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".local" / "share"


def config_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user config base directory.

    Holds user-specific configuration files: settings the user chose, which the
    application reads to decide how to behave. Anything the application itself
    wrote to remember what it was doing is state, not config.

    Reads ``XDG_CONFIG_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      platform: ``sys.platform`` string. Override for testing; ``None``
        reads ``sys.platform`` at CALL time. A ``= sys.platform`` default
        would bind at import, freezing the value before any monkeypatch.

    Returns:
      path: The config base directory, absolute unless the consulted
        environment variable was relative. Join your own namespace segment;
        nothing is created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg_config_home)
    if platform == "darwin":
        return data_dir(platform=platform)
    return Path.home() / ".config"


def cache_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user cache base directory.

    Holds user-specific non-essential (cached) data -- downloaded model
    weights, build artifacts, memoized computation. Every file here must be
    reconstructible: deleting the whole tree may cost time but must not lose
    anything.

    Reads ``XDG_CACHE_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      platform: ``sys.platform`` string. Override for testing; ``None``
        reads ``sys.platform`` at CALL time. A ``= sys.platform`` default
        would bind at import, freezing the value before any monkeypatch.

    Returns:
      path: The cache base directory, absolute unless the consulted
        environment variable was relative. Join your own namespace segment;
        nothing is created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg_cache_home)
    if platform == "darwin":
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


def state_dir(*, platform: str | None = None) -> Path:
    """Resolve the per-user state base directory.

    Holds state that should persist across restarts but is not important or
    portable enough for :func:`data_dir`. The spec names two kinds: action
    history (logs, recently-used files, session captures) and the state needed
    to resume where the application left off (view, layout, open files, undo
    history).

    Reads ``XDG_STATE_HOME``, or ``LOCALAPPDATA`` on Windows.

    Args:
      platform: ``sys.platform`` string. Override for testing; ``None``
        reads ``sys.platform`` at CALL time. A ``= sys.platform`` default
        would bind at import, freezing the value before any monkeypatch.

    Returns:
      path: The state base directory, absolute unless the consulted
        environment variable was relative. Join your own namespace segment;
        nothing is created.

    References:
      https://specifications.freedesktop.org/basedir/latest/

    """
    platform = platform or sys.platform
    if platform == "win32":
        return data_dir(platform=platform)
    if xdg_state_home := os.environ.get("XDG_STATE_HOME"):
        return Path(xdg_state_home)
    if platform == "darwin":
        return data_dir(platform=platform)
    return Path.home() / ".local" / "state"
