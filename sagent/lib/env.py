"""Environment-variable helpers."""

from __future__ import annotations

import os


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_truthy(name: str) -> bool:
    """Return ``True`` iff ``os.environ[name]`` is a conventional truthy string.

    Args:
      name: Environment variable name.

    Returns:
      is_truthy: Whether the value is in {1, true, yes, on}.

    """
    return os.environ.get(name, "").lower() in _TRUTHY
