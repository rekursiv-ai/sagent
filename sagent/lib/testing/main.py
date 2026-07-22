"""Standalone test runner.

``test_main`` is the standalone-execution entry point used by ~every
``*_test.py`` (the ``if __name__ == "__main__"`` guard). It lives in its own
module so importing it pulls in nothing but pytest.
"""

from __future__ import annotations

import sys

import pytest


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
