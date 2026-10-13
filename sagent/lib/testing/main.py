"""Standalone test runner, and a cap that proves code does not recurse.

``test_main`` is the standalone-execution entry point used by ~every
``*_test.py`` (the ``if __name__ == "__main__"`` guard). It lives in its own
module so importing it pulls in nothing but pytest.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import sys
import traceback

import pytest


if TYPE_CHECKING:
    from collections.abc import Generator


def test_main(test_file: str) -> None:
    """Run pytest on a test file with standard flags.

    Usage:
        if __name__ == "__main__":
            from sagent.lib.testing.main import test_main

            test_main(__file__)

    Args:
        test_file: The test file path (usually __file__)

    """
    sys.exit(
        pytest.main(
            [
                test_file,
                "-v",
                "-s",  # Don't capture output (show print statements)
                "-W",  # Warning filter (overrides -Werror for specific warning)
                "ignore::pytest.PytestAssertRewriteWarning",  # Ignore assertion rewrite warnings (happens during direct execution)
                *sys.argv[1:],
            ],
        ),
    )


@contextmanager
def recursion_headroom(frames: int) -> Generator[None]:
    """Cap the recursion limit ``frames`` above the caller's own depth.

    Inside, input nested deeper than ``frames`` raises ``RecursionError`` in any
    code that recurses once per level, so a test of iterative code needs only a
    small input instead of one deeper than the default limit.

    Args:
      frames: How many frames deeper than the caller the code may go.

    Yields:
      None: The limit is restored on exit.

    """
    before = sys.getrecursionlimit()
    sys.setrecursionlimit(sum(1 for _ in traceback.walk_stack(None)) + frames)
    try:
        yield
    finally:
        sys.setrecursionlimit(before)
