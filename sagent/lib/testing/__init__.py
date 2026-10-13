"""Testing utilities.

- ``runner``: ``test_main``, the footer every ``*_test.py`` runs.
- ``resources``: what a test process may use, wired in once by a conftest.
- ``checks``: assertions and helpers a single test calls.
"""

from __future__ import annotations

from sagent.lib.testing.runner import test_main


__all__ = [
    "test_main",
]
