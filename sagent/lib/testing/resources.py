"""What a test process may use: directories, threads, stack depth, time, services.

A package's conftest wires these in once::

    from sagent.lib.testing.resources import (
        cap_math_threads,
        isolate_user_dirs,
        pytest_collection_modifyitems,
        pytest_configure,
    )

    cap_math_threads()

    __all__ = [
        "isolate_user_dirs",
        "pytest_collection_modifyitems",
        "pytest_configure",
    ]

They live here rather than in a conftest because a conftest does not ship in a
wheel: an exported package whose tests import another package's conftest gets
nothing.

- Directories: ``isolate_user_dirs`` points every ``userdirs`` lookup at a
  per-test tmp directory, so a test can neither clobber the developer's
  real config, data, cache, or state, nor pass against what happens to be
  there. ``@pytest.mark.real_user_dirs`` opts out, for a test that drives a
  live service with the operator's real credentials. It cannot reach a
  module-level constant that called a ``userdirs`` helper at import time:
  imports run before any fixture.
- Threads: ``cap_math_threads`` gives each process one math thread.
- Stack depth: ``recursion_headroom`` caps how deep a test may recurse.
- Time and services: ``pytest_collection_modifyitems`` turns each resource
  marker into a timeout, family markers, and a skip where the resource is
  absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from importlib import util
from itertools import compress, islice, product
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import ast
import inspect
import os
import re
import sys
import tempfile
import traceback

from _pytest.mark.expression import Expression

import pytest


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


_CWD: Final = Path(__file__).resolve().parent


_IDENT: Final = re.compile(r"[A-Za-z_][\w:.+-]*")
"""A marker name as ``-m`` tokenizes one."""


_KEYWORDS: Final = frozenset(("and", "or", "not", "True", "False"))
"""Words a ``-m`` expression reserves, which name no marker."""


def pytest_configure(config: pytest.Config) -> None:
    """Declare the ``real_user_dirs`` marker. Re-export beside the fixture.

    Declared in code rather than a pyproject ``markers`` list because this
    module is vendored into every exported package and each carries its OWN
    pyproject: a marker listed only in the monorepo's is unregistered
    downstream, and every export runs ``filterwarnings = ["error"]``, which
    turns pytest's unknown-mark warning into a collection error. The marker
    travels with the code that honors it.

    It must be the hook, registered at configure time: the unknown-mark
    warning fires during COLLECTION, so the fixture cannot declare it late.
    A conftest with no hook of its own re-exports this one by name::

        from sagent.lib.testing.resources import (
            isolate_user_dirs,
            pytest_configure,
        )

        __all__ = ["isolate_user_dirs", "pytest_configure"]

    A conftest that already defines ``pytest_configure`` imports this under
    another name and calls it from its own body.

    Args:
      config: The pytest config to add the marker line to.

    """
    config.addinivalue_line(
        "markers",
        "real_user_dirs: opt out of XDG isolation; the test drives a live"
        " service with the operator's real credentials, which a tmp XDG root"
        " would replace with an empty directory",
    )
    # A run that deselects ``cuda`` runs only CPU tests, and a test that is
    # not marked may not reach a GPU: hidden, its stray CUDA use fails here as
    # it does in CI, instead of passing or not on whatever else holds the GPU.
    # Set before any test imports torch, and inherited by xdist workers.
    markexpr = str(config.getoption("markexpr", default=""))
    if markexpr and not _selects_any(markexpr, marker="cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(autouse=True)
def isolate_user_dirs(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path | None:
    """Point every XDG base directory at a per-test tmp root.

    Sets the environment rather than patching the ``userdirs`` functions: the
    env var is the seam production reads, so a test exercises the real
    resolution -- including a wrong ``app`` argument, which a patched function
    would hide. Patching also only reaches the one module that imported the
    helper, leaving every other caller pointed at the real directory.

    A test that drives a LIVE external service opts out with
    ``@pytest.mark.real_user_dirs``: its credentials are the operator's real
    ones, and a tmp XDG root silently replaces them with an empty directory.
    The marker exists so that need is expressible HERE. Without it the opt-out
    has to be spelled at the call site -- the caller hand-builds
    ``Path.home() / ".local/state/..."`` to dodge the fixture, reintroducing
    exactly the per-platform layout ``userdirs`` exists to delete.

    Args:
      request: Pytest request; checked for real_user_dirs marker.
      tmp_path_factory: Factory for creating per-test tmp directories.
      monkeypatch: Fixture for temporary environment modifications.

    Returns:
      root: The tmp directory the four XDG variables point at, or ``None``
        when the test opted out. Assertions normally do not need it; call the
        ``userdirs`` helper instead.

    """
    # FixtureRequest.node is an un-annotated abstract property upstream
    # (_pytest.fixtures); function scope makes it the test Item.
    node = cast(pytest.Item, request.node)
    if node.get_closest_marker("real_user_dirs") is not None:
        return None
    # Its OWN directory, deliberately NOT a subdirectory of ``tmp_path``.
    # Placing it under ``tmp_path`` costs one fewer allocation but is wrong: a
    # large family of tests asserts on the CONTENTS of ``tmp_path`` to prove an
    # atomic write left no temp file behind, and a ``userdirs`` entry appearing
    # there fails 15 of them (``temp file leaked: ['userdirs']``). The XDG root
    # must be invisible to the directory under test.
    # Keep these roots out of numbered allocation's per-test sibling scans.
    parent = tmp_path_factory.getbasetemp() / "userdirs"
    parent.mkdir(mode=0o700, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="isolate-", dir=parent))
    # This IS the fixture that repoints userdirs, so it spells the variables.
    for leaf in ("config", "data", "cache", "state"):
        (root / leaf).mkdir(mode=0o700)
        variable = f"XDG_{leaf.upper()}_HOME"  # house-ignore[xdg-literal] -- This IS the fixture that repoints userdirs, so it spells the variables.
        monkeypatch.setenv(variable, str(root / leaf))
    return root


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
    # The walk starts at this generator's frame and skips it and the
    # ``_GeneratorContextManager.__enter__`` that resumed it, so it counts from the
    # ``with`` statement's frame. ``walk_stack(None)`` starts at a different frame
    # per version: the ``with`` frame on 3.12, but this one on 3.14, which left
    # the limit two frames looser there.
    sys.setrecursionlimit(
        sum(1 for _ in islice(traceback.walk_stack(inspect.currentframe()), 2, None))
        + frames,
    )
    try:
        yield
    finally:
        sys.setrecursionlimit(before)


class MarkedItem(Protocol):
    """The marker surface the rollup reads and writes on a collected test.

    Narrower than ``pytest.Item`` on purpose: this is the whole contract, so a
    caller holding anything marker-shaped satisfies it.
    """

    def iter_markers(self, name: str | None = ...) -> Iterator[pytest.Mark]:
        """Iterate over markers, optionally filtered by name."""
        ...

    def add_marker(
        self,
        marker: str | pytest.MarkDecorator,
        *,
        append: bool = ...,
    ) -> None:
        """Add marker."""
        ...


class GoldenMarkedItem(MarkedItem, Protocol):
    """Marker surface plus the collected module and test function."""

    @property
    def module(self) -> object | None:
        """The collected module, if the item has one."""
        ...

    @property
    def obj(self) -> object:
        """The collected test function."""
        ...

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        """Return the nearest marker with ``name``."""
        ...


def resource_marker_family(
    marker: str,
    *,
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> str:
    """Return the selector family encoded by a resource marker prefix.

    Args:
      marker: Marker.
      resource_families: Resource families.

    Returns:
      family: The str.

    """
    family, separator, _specific = marker.partition("_")
    if not separator:
        raise ValueError("Expected separator.")
    if family not in resource_families:
        raise ValueError("Expected family in resource_families.")
    return family


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Derive timeouts and skips from resource markers, and mark golden tests.

    Args:
      config: Config.
      items: Items.

    """
    marker = "golden: tests that assert against a golden"
    if marker not in config.getini("markers"):
        config.addinivalue_line("markers", marker)
    apply_resource_markers(
        items,
        resource_markers=registered_resource_markers(config),
    )
    # The mark exists only for ``-m`` selection, and the call-graph walk costs
    # tens of seconds over the whole tree, so a run that cannot select on it
    # skips it.
    if "golden" in str(config.getoption("markexpr", default="")):
        apply_golden_marker(cast(Sequence[GoldenMarkedItem], items))


def apply_golden_marker(items: Sequence[GoldenMarkedItem]) -> None:
    """Mark test items whose call graph reaches a golden assertion.

    Args:
      items: Items.

    """
    # A golden is a host-agnostic CPU record; a GPU test's output is not portable.
    for item in items:
        module = item.module
        path = getattr(module, "__file__", None)
        name = getattr(item.obj, "__name__", None)
        if (
            isinstance(path, str)
            and isinstance(name, str)
            and not any(m.name.startswith("gpu_") for m in item.iter_markers())
            and _test_reaches_golden(Path(path), name)
            and item.get_closest_marker("golden") is None
        ):
            item.add_marker(pytest.mark.golden)


@cache
def module_path(module_name: str) -> Path | None:
    """Return the in-repo ``.py`` file behind ``module_name``, without importing it.

    Args:
      module_name: Dotted module name.

    Returns:
      path: The source file, or None when it is unresolvable, compiled, or outside
        the repository.

    """
    try:
        spec = util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
        return None
    path = Path(spec.origin)
    if path.suffix != ".py" or not path.is_relative_to(_CWD.parents[2]):
        return None
    return path


def registered_resource_markers(
    config: pytest.Config,
    *,
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> tuple[str, ...]:
    """Return registered concrete resource markers from pytest config.

    Args:
      config: Pytest config with marker registry.
      resource_families: Marker family names to filter by.

    Returns:
      result: Concrete marker names (e.g., "bench_throughput", "gpu_cuda").

    """
    configured = cast(list[str], config.getini("markers"))
    marker_names = tuple(marker.partition(":")[0] for marker in configured)
    return tuple(
        marker
        for marker in marker_names
        if marker.partition("_")[0] in resource_families and "_" in marker
    )


def resource_marker_aliases(
    marker: str,
    *,
    aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("bench_compile_time", ("performance",)),
        ("bench_memory", ("performance",)),
        ("bench_statistical", ("performance",)),
        ("bench_throughput", ("performance",)),
        ("bench_wallclock", ("performance",)),
        ("browser_chrome", ("integration",)),
        ("browser_zendriver", ("integration",)),
        ("cli_bash", ("slow",)),
        ("cli_real_llm", ("integration",)),
        ("cli_docker", ("integration",)),
        ("cli_git", ("integration",)),
        ("cli_node", ("integration",)),
        ("cli_precommit", ("ci_smoke",)),
        ("cli_python_subprocess", ("integration",)),
        ("cli_rsync", ("integration",)),
        ("cli_ssh", ("integration",)),
        ("cli_uv", ("ci_smoke",)),
        ("compute_distributed", ("slow",)),
        ("compute_jax_jit", ("slow",)),
        ("compute_large_fixture", ("slow",)),
        ("compute_torch_compile", ("slow",)),
        ("compute_training", ("slow",)),
        ("db_pglite", ("integration",)),
        ("db_pgvector", ("integration",)),
        ("db_postgres", ("integration",)),
        ("gpu_cuda_runtime", ("cuda",)),
        ("gpu_flash_attention", ("cuda",)),
        ("gpu_jax_cuda", ("cuda",)),
        ("gpu_nvidia", ("cuda",)),
        ("gpu_torch_cuda", ("cuda",)),
        ("gpu_torch_mps", ("integration",)),
        ("gpu_triton", ("cuda",)),
        ("network_anthropic", ("integration",)),
        ("network_duckduckgo", ("integration",)),
        ("network_e2b", ("integration",)),
        ("network_gemini", ("integration",)),
        ("network_google_search", ("integration",)),
        ("network_github", ("integration",)),
        ("network_huggingface", ("integration",)),
        ("network_kaggle", ("integration",)),
        ("network_localhost", ("integration",)),
        ("network_openai", ("integration",)),
        ("network_openml", ("integration",)),
        ("network_openreview", ("integration",)),
        ("network_pypi", ("integration",)),
        ("network_searxng", ("integration",)),
        ("network_gpu_cloud", ("cluster",)),
        ("network_slack", ("integration",)),
        ("network_together", ("cluster",)),
        ("network_wandb", ("integration",)),
    ),
) -> tuple[str, ...]:
    """Return legacy selector marks for a concrete resource marker.

    Args:
      marker: Concrete resource marker name (e.g., "bench_throughput").
      aliases: Mapping from markers to legacy skip/group names.

    Returns:
      candidate_aliases: Legacy marker names that should be added to the test.

    Raises:
      pytest.UsageError: marker is not in the aliases table.

    """
    for candidate, candidate_aliases in aliases:
        if marker == candidate:
            return candidate_aliases
    # UsageError, not ValueError: this is reached from a collection hook, and
    # pytest renders anything else as an INTERNALERROR traceback that buries the
    # marker name.
    raise pytest.UsageError(f"Unknown resource marker: {marker}")


def resource_marker_timeout(
    marker: str,
    *,
    category_timeouts: tuple[tuple[str, int], ...] = (
        ("bench", 600),
        ("browser", 180),
        ("cli", 120),
        ("compute", 300),
        ("db", 120),
        ("gpu", 300),
        ("network", 180),
    ),
    specific_timeouts: tuple[tuple[str, int], ...] = (
        ("bench_throughput", 600),
        ("cli_real_llm", 1800),
        ("cli_docker", 300),
        ("cli_precommit", 300),
        # A spawned interpreter re-imports the tree it collects, which the
        # 120s `cli` default does not cover: marker_tiers_test's collection
        # measured 69s on a developer box and timed out 108 times on a loaded
        # 2-vCPU CI runner. A timeout is a ceiling, not a schedule, so this
        # cannot slow a subprocess test that already finishes quickly.
        ("cli_python_subprocess", 300),
        ("cli_rsync", 180),
        ("cli_ssh", 300),
        ("cli_uv", 180),
        ("compute_large_fixture", 180),
        ("compute_torch_compile", 900),
        ("db_pglite", 180),
        ("network_gpu_cloud", 300),
        ("network_together", 4800),
    ),
) -> int:
    """Return a marker's specific timeout, falling back to its category.

    Args:
      marker: Marker.
      category_timeouts: Category timeouts.
      specific_timeouts: Specific timeouts.

    Returns:
      result: The int.

    """
    specific = dict(specific_timeouts)
    if marker in specific:
        return specific[marker]
    return dict(category_timeouts)[resource_marker_family(marker)]


def apply_resource_markers(
    items: Sequence[MarkedItem],
    *,
    resource_markers: tuple[str, ...],
    ci_skipped_marks: tuple[str, ...] = (
        "cluster",
        "cuda",
        "performance",
        "bench_compile_time",
        "bench_memory",
        "bench_statistical",
        "bench_throughput",
        "bench_wallclock",
        "gpu_cuda_runtime",
        "gpu_flash_attention",
        "gpu_jax_cuda",
        "gpu_nvidia",
        "gpu_torch_cuda",
        "gpu_triton",
        "network_gpu_cloud",
        "network_together",
    ),
    live_llm_marks: tuple[str, ...] = ("cli_real_llm",),
    live_llm_env_var: str = "RUN_REAL_LLM",
) -> None:
    """Apply virtual family markers, timeout budgets, and skip policy.

    Args:
      items: Test items to mark up.
      resource_markers: Registered concrete resource marker names.
      ci_skipped_marks: Markers that skip in CI (not RUN_INTEGRATION=1).
      live_llm_marks: Markers that skip unless RUN_REAL_LLM=1.
      live_llm_env_var: Environment variable enabling live LLM tests.

    """
    known_resources = set(resource_markers)
    for item in items:
        # One marker walk per item: ``get_closest_marker`` re-walks the whole
        # parent chain per name, so asking it per registered marker costs a walk
        # per marker per item on every repo-wide collection.
        existing = {marker.name for marker in item.iter_markers()}
        _fail_on_unknown_resource_markers(existing, resource_markers=known_resources)
        carried = existing & known_resources
        resource_timeouts = [resource_marker_timeout(marker) for marker in carried]
        resource_aliases = {
            alias for marker in carried for alias in resource_marker_aliases(marker)
        }
        # A root conftest and a package conftest both bind this hook, so an item
        # is walked once per binding. EVERY mark added below is therefore guarded
        # by what the item already carries -- one rule, rather than a per-branch
        # check that the next branch forgets.
        for alias in sorted(resource_aliases - existing):
            alias_marker = cast(pytest.MarkDecorator, getattr(pytest.mark, alias))
            item.add_marker(alias_marker)
        if resource_timeouts and "timeout" not in existing:
            item.add_marker(pytest.mark.timeout(max(resource_timeouts)))
        if "skip" not in existing:
            _apply_skip_policy(
                item,
                names=existing,
                live_llm_marks=live_llm_marks,
                live_llm_env_var=live_llm_env_var,
                ci_skipped_marks=ci_skipped_marks,
            )


class _NumbaConfig(Protocol):
    def reload_config(self) -> None: ...


# A ``cuda`` test also carries other markers (``slow``, ``gpu_cuda_runtime``), so
# testing an item holding ``cuda`` alone hid the GPU from ``-m "cuda and slow"``.
# Every other name the expression mentions is tried both ways instead.
def _selects_any(markexpr: str, *, marker: str) -> bool:
    """Return whether ``markexpr`` selects some test that carries ``marker``."""
    expression = Expression.compile(markexpr)
    others = sorted(set(_IDENT.findall(markexpr)) - _KEYWORDS - {marker})
    return any(
        expression.evaluate(_Carries(names={marker, *compress(others, carried)}))
        for carried in product((False, True), repeat=len(others))
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Carries:
    """A ``-m`` matcher for an item that carries exactly ``names``."""

    names: set[str]

    def __call__(self, name: str, /, **kwargs: str | int | bool | None) -> bool:
        del kwargs
        return name in self.names


class _ModuleSource:
    """Parsed functions, module-level names, and imports for one source module."""

    def __init__(
        self,
        tree: ast.Module,
        path: Path,
        *,
        mentions_testdata: bool,
    ) -> None:
        self.path = path
        self.mentions_testdata = mentions_testdata
        self.functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.golden_names = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and _names_checked_in_testdata(node.value)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.imports: dict[str, tuple[str, str | None]] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[alias.asname or alias.name.split(".")[-1]] = (
                        alias.name,
                        None,
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (
                        node.module,
                        alias.name,
                    )


def _apply_skip_policy(
    item: MarkedItem,
    *,
    names: set[str],
    live_llm_marks: tuple[str, ...],
    live_llm_env_var: str,
    ci_skipped_marks: tuple[str, ...],
) -> None:
    """Skip an item whose resource is unavailable in this environment."""
    for mark in live_llm_marks:
        if mark in names and not os.environ.get(live_llm_env_var, ""):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"{mark} test skipped"
                        f" (set {live_llm_env_var}=1 to run live model CLIs)"
                    ),
                ),
            )
            return
    if os.environ.get("CI", "") and not os.environ.get("RUN_INTEGRATION", ""):
        for mark in ci_skipped_marks:
            if mark in names:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"{mark} test skipped in CI"
                            " (no live credentials/services/devices;"
                            " set RUN_INTEGRATION=1 to opt in)"
                        ),
                    ),
                )
                return


def _fail_on_unknown_resource_markers(
    names: set[str],
    *,
    resource_markers: set[str],
    resource_families: tuple[str, ...] = (
        "bench",
        "browser",
        "cli",
        "compute",
        "db",
        "gpu",
        "network",
    ),
) -> None:
    """Fail collection when a resource-prefixed marker is not registered."""
    for name in names - resource_markers:
        family, separator, _specific = name.partition("_")
        if separator and family in resource_families:
            raise pytest.UsageError(f"Unknown resource marker: {name}")


@cache
def _test_reaches_golden(path: Path, test_name: str) -> bool:
    """Return whether one test function's transitive calls reach a golden."""
    seen: set[tuple[Path, str]] = set()
    pending = [(path, test_name)]
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        reaches, callees = _function_facts(*key)
        if reaches:
            return True
        pending.extend(callees)
    return False


# Every test walks the same shared helpers, so each function is read once per
# process and the per-test walk only follows the cached edges.
@cache
def _function_facts(
    path: Path,
    function_name: str,
) -> tuple[bool, tuple[tuple[Path, str], ...]]:
    """Return whether a function touches a golden itself, and what it calls."""
    module = _module_source(path)
    if function_name.startswith("assert_") and _imports_golden_support(path):
        return True, ()
    function = module.functions.get(function_name)
    if function is None:
        module_name, symbol = module.imports.get(function_name, ("", None))
        if symbol is None:
            return False, ()
        if _is_golden_symbol(module_name, symbol):
            return True, ()
        target_path = module_path(module_name)
        return False, () if target_path is None else ((target_path, symbol),)
    if module.mentions_testdata and _reads_checked_in_testdata(function, module):
        return True, ()
    callees: list[tuple[Path, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = _call_target(node, module)
        if target is None:
            continue
        target_path, target_name, is_golden = target
        if is_golden:
            return True, ()
        if target_path is not None:
            callees.append((target_path, target_name))
    return False, tuple(callees)


@cache
def _module_source(path: Path) -> _ModuleSource:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # A shared worktree can move a module between its import and this read.
        text = ""
    return _ModuleSource(
        ast.parse(text, filename=str(path)),
        path,
        mentions_testdata="testdata" in text,
    )


@cache
def _imports_golden_support(path: Path) -> bool:
    """Return whether a module transitively imports the golden or bfb helpers."""
    seen: set[Path] = set()
    pending = [path]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for module_name, _symbol in _module_source(current).imports.values():
            if module_name.rsplit(".", 1)[-1] in {"golden", "bfb"}:
                return True
            target_path = module_path(module_name)
            if target_path is not None:
                pending.append(target_path)
    return False


def _call_target(
    call: ast.Call,
    module: _ModuleSource,
) -> tuple[Path | None, str, bool] | None:
    function = call.func
    module_name: str | None = None
    symbol: str | None = None
    if isinstance(function, ast.Name):
        if function.id in module.functions:
            return module.path, function.id, False
        imported = module.imports.get(function.id)
        if imported is None:
            return None
        module_name, symbol = imported
    elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        imported = module.imports.get(function.value.id)
        if imported is None:
            return None
        parent, submodule = imported
        # ``from pkg import mod`` binds a module; its attribute is a function there.
        module_name = parent if submodule is None else f"{parent}.{submodule}"
        symbol = function.attr
    else:
        return None
    if symbol is not None and _is_golden_symbol(module_name, symbol):
        return None, symbol, True
    target_path = module_path(module_name)
    if target_path is None or symbol is None:
        return None
    return target_path, symbol, False


def _names_checked_in_testdata(node: ast.AST) -> bool:
    """Return whether ``node`` spells a path into a checked-in ``testdata/`` dir."""
    return any(
        isinstance(child, ast.Constant) and child.value == "testdata"
        for child in ast.walk(node)
    )


def _reads_checked_in_testdata(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module: _ModuleSource,
) -> bool:
    """Return whether a test body names a ``testdata/`` path, directly or by name."""
    return _names_checked_in_testdata(function) or any(
        isinstance(node, ast.Name) and node.id in module.golden_names
        for node in ast.walk(function)
    )


def _is_golden_symbol(module_name: str, symbol: str) -> bool:
    return module_name.rsplit(".", 1)[-1] in {"golden", "bfb"} and (
        symbol.startswith("assert_") or symbol == "expect_golden_mismatch"
    )
