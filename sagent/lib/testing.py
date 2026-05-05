"""Testing utilities."""

from __future__ import annotations

import sys

import pytest


try:
    import torch
except ImportError:  # torch is optional; cleanup_cuda + get_device no-op without it
    torch = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]


def get_device():
    """Get device for tests (prefer CUDA). Requires torch."""
    if torch is None:
        raise RuntimeError("torch is not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory before and after each test (no-op without torch)."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_main(test_file: str) -> None:
    """Run pytest on a test file with standard flags.

    Usage:
        if __name__ == "__main__":
            from sagent.lib.testing import test_main

            test_main(__file__)

    Args:
        test_file: The test file path (usually __file__)

    """
    sys.exit(
        pytest.main(
            [
                test_file,  # The test file to run
                "-v",  # Verbose output
                "-s",  # Don't capture output (show print statements)
                "-W",  # Warning filter (overrides -Werror for specific warning)
                "ignore::pytest.PytestAssertRewriteWarning",  # Ignore assertion rewrite warnings (happens during direct execution)
                *sys.argv[
                    1:
                ],  # Pass through command-line arguments (e.g., -k for test filtering)
            ],
        ),
    )
