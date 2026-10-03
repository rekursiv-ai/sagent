"""Shared pytest resource-marker rollups and timeout budgets."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import ast
import os

import pytest


_CWD: Final = Path(__file__).resolve().parent


if TYPE_CHECKING:
    from collections.abc import Iterator


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
        ("network_shadeform", ("cluster",)),
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
        ("network_shadeform", 300),
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
        "network_shadeform",
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
        if mark in names and not os.environ.get(live_llm_env_var):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"{mark} test skipped"
                        f" (set {live_llm_env_var}=1 to run live model CLIs)"
                    ),
                ),
            )
            return
    if os.environ.get("CI") and not os.environ.get("RUN_INTEGRATION"):
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
