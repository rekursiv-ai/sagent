"""Export-root pytest marker rollups for sagent.

Binds the resource-marker hook at the repository root so it reaches tests
living OUTSIDE the ``sagent`` package -- ``examples/`` in particular. A
conftest hook applies to its own directory and below, so ``sagent/conftest.py``
alone would leave an ``examples`` test's resource marker with no timeout, no CI
skip policy, and no error to say so.
"""

from sagent.lib.testing.resource_markers import pytest_collection_modifyitems


__all__ = ["pytest_collection_modifyitems"]
