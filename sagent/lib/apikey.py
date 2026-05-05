"""Centralized API key access."""

from __future__ import annotations

import os


def get(name: str) -> str:
    return os.environ.get(name, "")
