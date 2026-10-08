"""Native math-thread caps for a test process, set before any native library loads.

Call :func:`cap_math_threads` at import time from a package's own conftest::

    from sagent.lib.testing.threads import cap_math_threads

    cap_math_threads()

Lives beside the other testing helpers rather than in one package's conftest: a
conftest does not ship in a wheel, so an exported package whose tests import
another package's conftest gets nothing, and its xdist workers each run a
full-width BLAS/OpenMP pool.
"""

from __future__ import annotations

from typing import Protocol, cast

import os
import sys


__all__ = ["cap_math_threads"]


def cap_math_threads() -> None:
    """Give each process one math thread, before any native library loads.

    xdist parallelizes at the process level, so a worker that also spawns a
    full-width BLAS/OpenMP pool oversubscribes the box: N workers x N threads.
    The effect is not a mild slowdown -- an 8-worker run turns 2ms training
    steps into seconds and trips per-test timeouts.

    Must run before NumPy/PyTorch/SciPy import: torch reads ``OMP_NUM_THREADS``
    at import and pins its ATen intra-op pool to match. Conftest import is early
    enough; ``addopts`` is not.

    ``MKL_CBWR`` pins a CPU-independent GEMM kernel. It is NOT redundant with
    the bfb harness's float64 upcast: that upcast removes the float32 kernel's
    error, but a float64 GEMM's reduction order still varies with the kernel
    MKL selects, and a float64 difference lands on a different float32 bit
    whenever the exact value sits near a rounding boundary. Absorbed almost
    always, not always -- which is a test that fails on one machine in many,
    the worst failure a golden can have. Removing it was measured inert on an
    AMD host, where MKL takes a generic path anyway, and broke an Intel one.
    MKL reads it at its first GEMM, so it must be set before any matmul runs.

    Numba's ``prange`` pool is ``NUMBA_NUM_THREADS`` wide, every CPU by
    default, so an xdist worker keeps two: under ``pytest -n 32`` unit tests of
    10 ms took up to 2 s. A serial run keeps every CPU (an 8,192-world Craftax
    pool builds in 0.29 s on 128 threads, 7.4 s on two). Idle pool threads
    must sleep either way: spinning, they slowed every later test in the
    process 20-30x. A pytest plugin imports Numba before any conftest, and
    Numba re-reads its environment only at compile time, so a kernel loaded
    from cache could launch the pool at the stale width; reloading here, before
    any test runs, fixes the width before launch, after which Numba refuses to
    change it.

    Every variable uses ``setdefault``, so an explicit
    ``OMP_NUM_THREADS=8 pytest`` always wins.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",  # macOS Accelerate.
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")  # noqa: TID251 -- Thread cap the operator may override; not a provisioned cache path.
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")  # noqa: TID251 -- Kernel pin the operator may override; not a provisioned cache path.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # noqa: TID251 -- Thread policy the operator may override; not a provisioned cache path.
    if "PYTEST_XDIST_WORKER" in os.environ:
        os.environ.setdefault("NUMBA_NUM_THREADS", "2")  # noqa: TID251 -- Thread cap the operator may override; not a provisioned cache path.
    numba_config = cast(_NumbaConfig | None, sys.modules.get("numba.core.config"))
    if numba_config is not None:
        numba_config.reload_config()


class _NumbaConfig(Protocol):
    def reload_config(self) -> None: ...
