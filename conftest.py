"""Export-root pytest setup for sagent: marker rollups and math-thread caps.

Binds the resource-marker hook at the repository root so it reaches tests
living OUTSIDE the ``sagent`` package -- ``examples/`` in particular. A
conftest hook applies to its own directory and below, so ``sagent/conftest.py``
alone would leave an ``examples`` test's resource marker with no timeout, no CI
skip policy, and no error to say so.

The math-thread caps run here, at the root, because this conftest loads before
any test or package conftest imports NumPy or torch. In the monorepo the
repo-root conftest applies them; the public tree has no such root.
"""

from sagent.lib.testing.resource_markers import pytest_collection_modifyitems
from sagent.lib.testing.threads import cap_math_threads


__all__ = ["pytest_collection_modifyitems"]


cap_math_threads()
