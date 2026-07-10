"""Testing utilities."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import sys

import pytest


if TYPE_CHECKING:
    import torch as torch_typed


try:
    import torch
except ImportError:  # torch is optional; cleanup_cuda + get_device no-op without it.
    torch = None  # ty: ignore[invalid-assignment] -- optional dep sentinel; pyright infers Module|None


def get_device() -> torch_typed.device:
    """Return the preferred test device, CUDA when available.

    Returns:
      device: ``cuda`` if a CUDA device is present, else ``cpu``.

    Raises:
      RuntimeError: If torch is not installed.

    """
    if torch is None:
        raise RuntimeError("torch is not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cleanup_cuda() -> Generator[None]:
    """Reclaim CUDA memory symmetrically around each test (no-op on CPU)."""
    _reclaim_cuda()
    yield
    _reclaim_cuda()


def _reclaim_cuda() -> None:
    """Synchronize and empty the CUDA cache when a CUDA device is present."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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
                test_file,
                "-v",
                "-s",  # Don't capture output (show print statements)
                "-W",  # Warning filter (overrides -Werror for specific warning)
                "ignore::pytest.PytestAssertRewriteWarning",  # Ignore assertion rewrite warnings (happens during direct execution)
                *sys.argv[1:],
            ],
        ),
    )
