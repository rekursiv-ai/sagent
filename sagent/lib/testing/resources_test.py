"""Tests for ``sagent.lib.testing.resources``."""

from __future__ import annotations

from collections.abc import Callable
from importlib import machinery, util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import inspect
import os
import subprocess
import sys
import textwrap
import traceback

import pytest

from sagent.lib.testing import resources
from sagent.lib.testing.resources import (
    apply_golden_marker,
    apply_resource_markers,
    cap_math_threads,
    module_path,
    pytest_collection_modifyitems,
    recursion_headroom,
    resource_marker_aliases,
    resource_marker_timeout,
)
from sagent.lib.userdirs import cache_dir, config_dir, data_dir, state_dir


if TYPE_CHECKING:
    from collections.abc import Iterator


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def fresh_call_graph() -> Iterator[None]:
    """Forget cached sources: each test writes its own modules under tmp_path."""
    caches = (
        resources.module_path,
        resources._module_source,
        resources._function_facts,
        resources._imports_golden_support,
        resources._test_reaches_golden,
    )
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


def _run_pytest(tmp_path: Path, *, bindings: int) -> subprocess.CompletedProcess[str]:
    """Collect a tree carrying an unregistered marker under ``bindings`` hooks."""
    conftest = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, {root!r})

        from sagent.lib.testing.resources import pytest_collection_modifyitems

        __all__ = ["pytest_collection_modifyitems"]
        """,
    ).format(root=str(_CWD.parents[2]))
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "conftest.py").write_text(conftest, encoding="utf-8")
    if bindings > 1:
        (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    (package / "probe_test.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.network_notreal
            def test_probe():
                pass
            """,
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "pkg/probe_test.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.cli_python_subprocess
def test_unknown_resource_marker_reports_a_usage_error(tmp_path: Path) -> None:
    """An unregistered resource marker must report, not crash the runner.

    The check runs inside ``pytest_collection_modifyitems``. pytest turns an
    arbitrary exception escaping a hook into an INTERNALERROR traceback, which
    buries the one line naming the offending marker; ``UsageError`` is the
    exception it renders as a plain message instead.
    """
    result = _run_pytest(tmp_path, bindings=1)

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "INTERNALERROR" not in output
    assert "Unknown resource marker: network_notreal" in output


@pytest.mark.cli_python_subprocess
def test_double_bound_hook_reports_a_usage_error(tmp_path: Path) -> None:
    """Two conftests binding the hook report the same way as one.

    A root conftest and a package conftest both bind it whenever the export
    ships tests outside the package directory, so the second binding must not
    change how a bad marker surfaces.
    """
    result = _run_pytest(tmp_path, bindings=2)

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "INTERNALERROR" not in output
    assert "Unknown resource marker: network_notreal" in output


def test_collection_hook_registers_the_golden_marker() -> None:
    registered: list[tuple[str, str]] = []

    config = _Config(registered)
    pytest_collection_modifyitems(cast(pytest.Config, cast(object, config)), [])
    pytest_collection_modifyitems(cast(pytest.Config, cast(object, config)), [])

    assert registered == [("markers", "golden: tests that assert against a golden")]


@pytest.mark.parametrize(("markexpr", "marked"), [("golden", True), ("", False)])
def test_golden_marks_are_derived_only_when_selected(
    tmp_path: Path,
    markexpr: str,
    *,
    marked: bool,
) -> None:
    module = tmp_path / "selected_test.py"
    module.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n"
        "\ndef test_selected():\n    check()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_selected")
    item.module = type("Module", (), {"__file__": str(module)})

    pytest_collection_modifyitems(
        cast(pytest.Config, cast(object, _Config([], markexpr=markexpr))),
        cast(list[pytest.Item], cast(object, [item])),
    )

    assert ("golden" in [m.name for m in item.iter_markers()]) is marked


def test_unknown_alias_lookup_reports_a_usage_error() -> None:
    """``resource_marker_aliases`` rejects an unknown marker the same way.

    It is reached from the same hook, so a bare exception here escapes with the
    same INTERNALERROR shape as the collection check.
    """
    with pytest.raises(
        pytest.UsageError,
        match="Unknown resource marker: network_nope",
    ):
        resource_marker_aliases("network_nope")


def test_indirect_golden_caller_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "bridge_test.py"
    module.write_text(
        "import numerics.testing.bfb as bfb\n"
        "\ndef test_indirect():\n    bfb.assert_bfb_against_golden()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_indirect")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_imported_name_call_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "name_test.py"
    module.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n"
        "\ndef test_name():\n    check()\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_name")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_importing_bfb_without_calling_an_assertion_is_not_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "bfb.py").write_text(
        "def host_agnostic_numerics():\n    pass\n",
        encoding="utf-8",
    )
    module = tmp_path / "plain_test.py"
    module.write_text(
        "from bfb import host_agnostic_numerics\n"
        "\ndef test_plain():\n    host_agnostic_numerics()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    item = _Item(test_name="test_plain")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_test_loading_a_checked_in_tensor_golden_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "direct_test.py"
    module.write_text(
        "import torch\n"
        "\ndef test_direct():\n"
        '    torch.load(_CWD / "testdata" / "g.pt")\n',
        encoding="utf-8",
    )
    item = _Item(test_name="test_direct")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_test_reading_a_module_level_golden_path_is_marked(tmp_path: Path) -> None:
    module = tmp_path / "constant_test.py"
    module.write_text(
        "from numerics.testing.golden import read_tensors\n"
        '\nGOLDEN = _CWD / "testdata" / "data.pt"\n'
        "\ndef test_constant():\n    read_tensors(GOLDEN)\n",
        encoding="utf-8",
    )
    item = _Item(test_name="test_constant")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_call_through_a_from_imported_submodule_is_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "records.py").write_text(
        'def golden_path(name):\n    return _CWD / "testdata" / name\n',
        encoding="utf-8",
    )
    module = tmp_path / "submodule_test.py"
    module.write_text(
        "from pkg import records\n"
        '\ndef test_submodule():\n    records.golden_path("g.pt")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setattr(
        "sagent.lib.testing.resources._CWD",
        tmp_path / "a" / "b",
    )
    item = _Item(test_name="test_submodule")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert [marker.name for marker in item.iter_markers()] == ["golden"]


def test_test_writing_a_temporary_checkpoint_is_not_marked(tmp_path: Path) -> None:
    module = tmp_path / "checkpoint_test.py"
    module.write_text(
        "import torch\n"
        "\ndef test_checkpoint(tmp_path):\n"
        '    torch.load(tmp_path / "ckpt.pt")\n',
        encoding="utf-8",
    )
    item = _Item(test_name="test_checkpoint")
    item.module = type("Module", (), {"__file__": str(module)})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_a_test_module_moved_away_after_import_is_not_marked(tmp_path: Path) -> None:
    item = _Item(test_name="test_gone")
    item.module = type("Module", (), {"__file__": str(tmp_path / "gone_test.py")})

    apply_golden_marker([item])

    assert list(item.iter_markers()) == []


def test_module_path_ignores_invalid_and_external_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_spec(name: str) -> object:
        del name
        raise ValueError("module __spec__ is None")

    monkeypatch.setattr(util, "find_spec", invalid_spec)
    assert module_path("torch._C._jit_tree_views") is None

    external = tmp_path / "external.py"
    external.write_text("", encoding="utf-8")
    spec = util.spec_from_file_location("external", external)
    assert spec is not None

    def external_spec(name: str) -> machinery.ModuleSpec:
        del name
        return spec

    monkeypatch.setattr(util, "find_spec", external_spec)
    assert module_path("external") is None

    monkeypatch.undo()
    assert module_path("no_such_module_anywhere") is None
    assert module_path("sys") is None


def test_openml_cold_fetch_keeps_its_hosted_runner_timeout_budget() -> None:
    """A cold OpenML download needs the previously established 120s budget."""
    assert resource_marker_timeout("network_openml") >= 120


def test_e2b_auth_keeps_integration_selection_and_network_deadline() -> None:
    """Selecting the E2B resource must preserve the real-network test tier."""
    item = _Item("network_e2b")
    apply_resource_markers([item], resource_markers=("network_e2b",))
    assert item.get_closest_marker("integration") is not None
    timeout = item.get_closest_marker("timeout")
    assert timeout is not None
    assert timeout.args == (resource_marker_timeout("network_e2b"),)


def test_resource_marker_family_rejects_malformed_names() -> None:
    with pytest.raises(ValueError, match="separator"):
        resources.resource_marker_family("network")
    with pytest.raises(ValueError, match="family"):
        resources.resource_marker_family("unknown_probe")


def test_unknown_resource_marker_is_rejected() -> None:
    with pytest.raises(pytest.UsageError, match="network_missing"):
        resources._fail_on_unknown_resource_markers(
            {"network_missing"},
            resource_markers=set(),
        )


def test_golden_call_graph_import_edges_are_resolved(tmp_path: Path) -> None:
    direct = tmp_path / "direct.py"
    direct.write_text(
        "import numerics.testing.bfb as bfb\n",
        encoding="utf-8",
    )
    assert resources._function_facts(direct, "assert_anything") == (True, ())

    imported = tmp_path / "imported.py"
    imported.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden\n",
        encoding="utf-8",
    )
    assert resources._function_facts(
        imported,
        "assert_bfb_against_golden",
    ) == (True, ())
    aliased = tmp_path / "aliased.py"
    aliased.write_text(
        "from numerics.testing.bfb import assert_bfb_against_golden as check\n",
        encoding="utf-8",
    )
    assert resources._function_facts(aliased, "check") == (True, ())


def test_call_graph_edges_keep_only_calls_it_can_follow(tmp_path: Path) -> None:
    module = tmp_path / "calls_test.py"
    module.write_text(
        textwrap.dedent(
            """
            from pathlib import Path

            def helper():
                pass

            def test_calls():
                helper()
                print()
                local = object()
                local.method()
                helper.__call__.__call__()
                Path()
            """,
        ),
        encoding="utf-8",
    )

    # ``pathlib`` lives outside the repository, so ``Path()`` is no edge.
    assert resources._function_facts(module, "test_calls") == (
        False,
        ((module, "helper"),),
    )
    assert resources._function_facts(module, "Path") == (False, ())


def test_a_test_calling_itself_does_not_reach_a_golden(tmp_path: Path) -> None:
    module = tmp_path / "recursive_test.py"
    module.write_text("def test_self():\n    test_self()\n", encoding="utf-8")

    assert resources._test_reaches_golden(module, "test_self") is False


def test_import_graph_without_golden_support_is_false(tmp_path: Path) -> None:
    module = tmp_path / "plain.py"
    module.write_text("import pathlib\n", encoding="utf-8")
    assert resources._imports_golden_support(module) is False


def test_import_graph_skips_a_seen_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root.py"
    root.write_text("import bridge\n", encoding="utf-8")
    bridge = tmp_path / "bridge.py"
    bridge.write_text("import bridge\n", encoding="utf-8")

    def find_bridge(name: str) -> Path | None:
        return bridge if name == "bridge" else None

    monkeypatch.setattr(resources, "module_path", find_bridge)
    assert resources._imports_golden_support(root) is False


def test_import_graph_follows_local_modules_to_golden_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root.py"
    bridge = tmp_path / "bridge.py"
    root.write_text("import bridge\n", encoding="utf-8")
    bridge.write_text("import numerics.testing.bfb\n", encoding="utf-8")

    def find_bridge(name: str) -> Path | None:
        return bridge if name == "bridge" else None

    monkeypatch.setattr(resources, "module_path", find_bridge)
    assert resources._imports_golden_support(root) is True


class _Config:
    """Minimal stand-in for the config surface the collection hook reads."""

    def __init__(
        self,
        registered: list[tuple[str, str]],
        *,
        markexpr: str = "",
    ) -> None:
        self.markers: list[str] = []
        self.registered = registered
        self.markexpr = markexpr

    def getini(self, name: str) -> list[str]:
        if name == "markers":
            return self.markers
        return []

    def getoption(self, name: str, default: object = None) -> object:
        return self.markexpr if name == "markexpr" else default

    def addinivalue_line(self, name: str, value: str) -> None:
        self.registered.append((name, value))
        self.markers.append(value)


class _Item:
    """Minimal stand-in exposing the marker surface the rollup touches."""

    def __init__(self, *marks: str, test_name: str = "test_probe") -> None:
        self.module: object | None = None
        self.obj = SimpleNamespace(__name__=test_name)
        self.own_markers: list[pytest.Mark] = [
            cast(pytest.MarkDecorator, getattr(pytest.mark, mark)).mark
            for mark in marks
        ]

    def iter_markers(self, name: str | None = None) -> Iterator[pytest.Mark]:
        for marker in self.own_markers:
            if name is None or marker.name == name:
                yield marker

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        for marker in self.own_markers:
            if marker.name == name:
                return marker
        return None

    def add_marker(
        self,
        marker: str | pytest.MarkDecorator,
        *,
        append: bool = True,
    ) -> None:
        del append
        assert isinstance(marker, pytest.MarkDecorator)
        self.own_markers.append(marker.mark)


def test_applying_markers_twice_leaves_one_alias() -> None:
    """Rolling up twice must not stack duplicate alias marks on an item.

    A root conftest and a package conftest both bind this hook, so an item is
    walked once per binding. Without idempotence the aliases accumulate, and the
    duplicates travel with the item into every later report.
    """
    item = _Item("network_github")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("network_github",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("integration") == 1


def test_applying_markers_twice_leaves_one_live_llm_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live-LLM skip is added once, however many bindings walk the item.

    Every mark this function adds has to survive a second pass unchanged --
    an alias, a timeout, and a skip alike -- because a duplicate skip reason
    travels into every later report exactly as a duplicate alias does.
    """
    monkeypatch.delenv("RUN_REAL_LLM", raising=False)
    item = _Item("cli_real_llm")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("cli_real_llm",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("skip") == 1


def test_applying_markers_twice_leaves_one_ci_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CI skip is added once, however many bindings walk the item."""
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUN_INTEGRATION", raising=False)
    item = _Item("gpu_nvidia")
    for _ in range(2):
        apply_resource_markers([item], resource_markers=("gpu_nvidia",))

    names = [marker.name for marker in item.iter_markers()]
    assert names.count("skip") == 1


_HELPERS = (
    (config_dir, "XDG_CONFIG_HOME"),
    (data_dir, "XDG_DATA_HOME"),
    (cache_dir, "XDG_CACHE_HOME"),
    (state_dir, "XDG_STATE_HOME"),
)


def test_every_helper_resolves_under_the_isolated_root() -> None:
    """Default: all four helpers land in this test's tmp root, not the user's."""
    for helper, variable in _HELPERS:
        assert str(helper() / "rekursiv-ai").startswith(os.environ[variable])


def test_isolated_base_directories_exist() -> None:
    for helper, _ in _HELPERS:
        assert helper().is_dir()
        assert helper().stat().st_mode & 0o777 == 0o700


def test_configure_declares_the_exported_opt_out_marker(
    pytestconfig: pytest.Config,
) -> None:
    resources.pytest_configure(pytestconfig)
    assert pytestconfig.getini("markers")[-1] == (
        "real_user_dirs: opt out of XDG isolation; the test drives a live"
        " service with the operator's real credentials, which a tmp XDG root"
        " would replace with an empty directory"
    )


@pytest.mark.parametrize(
    ("markexpr", "hidden"),
    [
        ("not ci_smoke and not cuda and not integration", True),
        # A ``cuda`` test that is also ``ci_smoke`` is selected, so it keeps the GPU.
        ("(not cluster and not cuda) or ci_smoke", False),
        ("", False),
        ("cuda", False),
        ("cuda and slow", False),
        ("gpu_cuda_runtime", False),
        ("slow and not cuda", True),
        ("not integration", False),
    ],
)
def test_a_run_that_deselects_cuda_sees_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
    markexpr: str,
    *,
    hidden: bool,
) -> None:
    """An unmarked test cannot reach a GPU, so it must run on the CPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    resources.pytest_configure(
        cast(pytest.Config, cast(object, _Config([], markexpr=markexpr))),
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ("" if hidden else "0,1")


def test_isolation_does_not_scan_prior_test_directories(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with patch("os.scandir", wraps=os.scandir) as scandir:
        root = _allocate(request, tmp_path_factory, monkeypatch)

    assert isinstance(root, Path)
    assert root.is_dir()
    scandir.assert_not_called()


def test_isolation_does_not_grow_pytests_numbered_directory_parent(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path_factory.getbasetemp()
    before = set(base.iterdir())
    for _ in range(3):
        _allocate(request, tmp_path_factory, monkeypatch)

    assert len(set(base.iterdir()) - before) <= 1


def test_isolated_roots_are_unique_and_outside_tmp_path(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _allocate(request, tmp_path_factory, monkeypatch)
    assert isinstance(first, Path)
    (config_dir() / "rekursiv-ai").touch()
    second = _allocate(request, tmp_path_factory, monkeypatch)

    assert isinstance(second, Path)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.is_relative_to(tmp_path_factory.getbasetemp())
    assert second.is_relative_to(tmp_path_factory.getbasetemp())
    assert not (config_dir() / "rekursiv-ai").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.real_user_dirs
def test_real_user_dirs_marker_restores_the_operators_directories(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``real_user_dirs`` reaches the operator's own dirs.

    A live-service test needs its real credentials; without this marker the
    only way to reach them is to hand-build ``Path.home() / ".local/state"``
    at the call site, which reintroduces the per-platform layout ``userdirs``
    exists to delete. Asserting through the helper keeps that unnecessary.
    """
    for helper, _ in _HELPERS:
        resolved = helper() / "rekursiv-ai"
        assert not resolved.is_relative_to(tmp_path_factory.getbasetemp()), (
            f"{helper.__name__} still resolved into a pytest tmp root"
        )


def _allocate(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path | None:
    """Call the fixture body directly, outside pytest's fixture machinery."""
    # ``inspect.unwrap`` returns ``Any``; the wrapped function IS the fixture,
    # so its signature is the one declared on ``isolate_user_dirs``.
    body = cast(
        Callable[
            [pytest.FixtureRequest, pytest.TempPathFactory, pytest.MonkeyPatch],
            Path | None,
        ],
        inspect.unwrap(resources.isolate_user_dirs),
    )
    return body(request, tmp_path_factory, monkeypatch)


@pytest.fixture(autouse=True)
def numba_config(monkeypatch: pytest.MonkeyPatch) -> _NumbaConfig:
    """Stand in for a loaded ``numba.core.config`` that counts its reloads.

    The real one would re-read the environment these tests rewrite and keep the
    rewritten width for every later test in the process.
    """
    config = _NumbaConfig()
    monkeypatch.setitem(sys.modules, "numba.core.config", config)
    return config


def test_unset_variables_get_one_thread_and_the_pinned_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    _unset(monkeypatch, *names, "MKL_CBWR", "OPENBLAS_CORETYPE", "OMP_WAIT_POLICY")

    cap_math_threads()

    assert [os.environ[name] for name in names] == ["1"] * len(names)
    assert os.environ["MKL_CBWR"] == "COMPATIBLE"
    assert os.environ["OPENBLAS_CORETYPE"] == "Haswell"
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"


def test_an_operator_setting_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_CBWR", "AUTO")
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Zen")

    cap_math_threads()

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_CBWR"] == "AUTO"
    assert os.environ["OPENBLAS_CORETYPE"] == "Zen"


@pytest.mark.parametrize(("worker", "expected"), [("gw0", "2"), (None, None)])
def test_numba_keeps_two_threads_only_in_an_xdist_worker(
    monkeypatch: pytest.MonkeyPatch,
    worker: str | None,
    expected: str | None,
) -> None:
    _unset(monkeypatch, "NUMBA_NUM_THREADS", "PYTEST_XDIST_WORKER")
    if worker is not None:
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)

    cap_math_threads()

    assert os.environ.get("NUMBA_NUM_THREADS") == expected


def test_a_loaded_numba_rereads_its_environment(numba_config: _NumbaConfig) -> None:
    cap_math_threads()

    assert numba_config.reloads == 1


def test_an_absent_numba_is_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "numba.core.config")

    cap_math_threads()

    assert "numba.core.config" not in sys.modules


class _NumbaConfig(ModuleType):
    """Counts reloads in place of ``numba.core.config``."""

    def __init__(self) -> None:
        super().__init__("numba.core.config")
        self.reloads = 0

    def reload_config(self) -> None:
        self.reloads += 1


def _unset(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Remove ``names`` for this test; teardown restores each, absent ones too."""
    for name in names:
        # ``delenv`` of an absent name records nothing to undo, so a value the
        # code under test then sets would outlive the test. Setting first
        # records the original, absence included.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _depth(levels: int) -> int:
    return 0 if levels == 0 else 1 + _depth(levels - 1)


def test_recursing_past_the_headroom_raises() -> None:
    with recursion_headroom(50), pytest.raises(RecursionError):
        _depth(100)


def test_recursing_within_the_headroom_runs() -> None:
    with recursion_headroom(50):
        assert _depth(20) == 20


def test_the_headroom_counts_from_the_with_statements_own_depth() -> None:
    depth = sum(1 for _ in traceback.walk_stack(inspect.currentframe()))
    with recursion_headroom(50):
        assert sys.getrecursionlimit() == depth + 50


def test_the_recursion_limit_is_restored_on_exit() -> None:
    before = sys.getrecursionlimit()

    with pytest.raises(RecursionError), recursion_headroom(50):
        _depth(100)

    assert sys.getrecursionlimit() == before


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
