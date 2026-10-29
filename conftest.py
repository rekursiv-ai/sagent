"""Export-root pytest setup for sagent: resource markers, user dirs, thread caps.

Binds the resource-marker hook and the per-user-directory isolation at the
repository root so they reach tests living OUTSIDE the ``sagent`` package --
the pre-commit gates' tests under ``tools/`` in particular. A conftest hook
applies to its own directory and below, so a package-level conftest alone
would leave a ``tools`` test's resource marker with no timeout, no CI skip
policy, and no error to say so. In the monorepo the repo-root conftest does
all of this; the public tree has no such root, so without it a session test
reads and writes the real per-user directories.

The math-thread caps run here, at the root, because this conftest loads before
any test or package conftest imports NumPy or torch.
"""

from __future__ import annotations

from sagent.lib.testing.resources import (
    cap_math_threads,
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


__all__ = [
    "isolate_user_dirs",
    "pytest_collection_modifyitems",
    "pytest_configure",
]


cap_math_threads()
