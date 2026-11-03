r"""Classify collected tests into CI tiers, and narrow a collection to a listing.

CI selects tests in one job and runs them in another, handing the list over as
an uploaded artifact. Redirecting ``--collect-only -q`` into that artifact made
its format whatever the installed plugins happened to print: pytest appends a
``N/M tests collected`` summary, and py-pglite prints a banner. Those lines
reach the next job as file arguments, and ONE unmatchable argument makes pytest
collect nothing at all -- not "the rest minus one". Measured 2026-08-19: a
979-nodeid selection plus one summary line collected zero tests, exited 5, and
failed a required check for hours while the artifact looked the right size.

Writing from the hook instead makes the file this plugin's output rather than
the console's. Nothing another plugin prints can reach it.

Enable with ``-p tools.testing.listing`` -- the dotted
path, which is what resolves from the repository root -- or by naming it in
``PYTEST_PLUGINS``. Then set any of:

- ``PYTEST_COLLECT_SELECTED``: write all selected nodeids to that one file.
- ``PYTEST_COLLECT_SELECTED_DIR``: write PR CI's per-hook selected files and
  ``selected-counts.json`` from one testmon collection.
- ``PYTEST_COLLECT_TIERS``: write ``<tier>\\t<file>`` for every test file and
  CI tier it holds tests for -- the classification main records for PRs.

or pass ``--test-files LISTING`` to collect only the files that listing names,
one per line, without entering any directory that holds none of them. It is an
option, not a variable, because tests start pytest runs of their own: they
inherit the environment, and a listing in it narrowed their run to nothing.

A listing may also split one file between shards (``shard_plan.py``): a
``<file>::<test>`` line runs only the tests so named from that file, and a
``!<file>::<test>`` line drops that test from a file the listing names whole.
A test named without its ``[case]`` names every case. The shard holding the
file whole runs every test no other shard names, a new one included.

The pre-commit ``pytest`` gate selects the same way, from the dependency
graph's nodeids. Nodeids as arguments are unsafe: once one points into a
module that skips at module level, pytest 9.2 drops every other argument.

Without one of these the plugin does nothing, so the same invocation works
uninstrumented.
"""

from __future__ import annotations

from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import json
import os

from tools.gate import gate
from tools.testing.tiers import TIERS


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class _Mark(Protocol):
    @property
    def name(self) -> str: ...


class _Item(Protocol):
    """The part of ``pytest.Item`` the writer reads."""

    @property
    def nodeid(self) -> str: ...

    def iter_markers(self) -> Iterable[_Mark]: ...


class _Session(Protocol):
    @property
    def items(self) -> Sequence[_Item]: ...


class _Config(Protocol):
    """The part of ``pytest.Config`` the collection filter reads."""

    @property
    def rootpath(self) -> Path: ...

    def getini(self, name: str) -> list[str]: ...

    def getoption(self, name: str) -> Path | None: ...


class _Parser(Protocol):
    def addoption(self, *names: str, **attrs: object) -> None: ...


class _Hooks(Protocol):
    def pytest_deselected(self, *, items: Sequence[_Item]) -> None: ...


class _ItemConfig(_Config, Protocol):
    """The part of ``pytest.Config`` the item filter reads."""

    @property
    def hook(self) -> _Hooks: ...


def pytest_addoption(parser: _Parser) -> None:
    """Register ``--test-files``.

    Args:
      parser: pytest's command-line parser.

    """
    parser.addoption(
        "--test-files",
        type=Path,
        help="Collect only the test files this file lists, one per line.",
    )


def pytest_collection_finish(session: _Session) -> None:
    """Write selected node ids once every deselector has run.

    NOT ``pytest_collection_modifyitems``. pytest runs that hook LIFO, so an
    ordinary implementation sees the item list BEFORE testmon deselects from
    its own ``trylast`` one -- and registering ``trylast`` here merely ties
    with testmon, resolved by registration order. Measured against real
    testmon: where it selected 1 of 2, both spellings of ``modifyitems`` wrote
    2 and this hook wrote 1. Collection finish runs strictly after every
    ``modifyitems``, so the ordering question does not arise.

    It matters because the consuming job runs exactly this file: recording the
    pre-deselection list forfeits the selection this job exists to compute,
    degrading testmon from per-test to per-file granularity.

    Args:
      session: pytest session with finalized item collection.

    """
    environ = os.environ
    if gate.flagged(environ, "PYTEST_COLLECT_SELECTED"):
        _write_nodeids(
            Path(environ["PYTEST_COLLECT_SELECTED"]),
            nodeids=(item.nodeid for item in session.items),
        )
    if gate.flagged(environ, "PYTEST_COLLECT_SELECTED_DIR"):
        _write_tier_nodeids(
            Path(environ["PYTEST_COLLECT_SELECTED_DIR"]),
            items=session.items,
        )
    if gate.flagged(environ, "PYTEST_COLLECT_TIERS"):
        _write_tier_files(Path(environ["PYTEST_COLLECT_TIERS"]), items=session.items)


def pytest_ignore_collect(collection_path: Path, config: _Config) -> bool | None:
    """Skip every test file the ``--test-files`` listing does not name.

    A directory is skipped unless a listed file sits under it, so neither its
    test modules nor its conftests are imported. A file that is not a test
    module by ``python_files`` is left to pytest: a conftest hook may collect it.

    Args:
      collection_path: Path pytest is about to collect.
      config: The session's config.

    Returns:
      ignore: True to skip the path; None to let other hooks decide.

    """
    if (listing := config.getoption("test_files")) is None:
        return None
    files, directories = _listed(listing, root=config.rootpath)
    if collection_path.is_dir():
        return None if collection_path in directories else True
    patterns = config.getini("python_files")
    if collection_path not in files and any(
        fnmatch(collection_path.name, pattern) for pattern in patterns
    ):
        return True
    return None


def pytest_collection_modifyitems[ItemT: _Item](
    config: _ItemConfig,
    items: list[ItemT],
) -> None:
    """Deselect the tests of a split file that the listing gives another shard.

    Args:
      config: The session's config.
      items: Collected tests, narrowed in place.

    """
    if (listing := config.getoption("test_files")) is None:
        return
    named, dropped = _split_tests(listing)
    kept: list[ItemT] = []
    deselected: list[ItemT] = []
    for item in items:
        file = item.nodeid.partition("::")[0]
        # The dependency graph names ``f::t``; pytest collects ``f::t[case]``.
        names = {item.nodeid, item.nodeid.partition("[")[0]}
        runs_here = not names & dropped and (
            file not in named or bool(names & named[file])
        )
        (kept if runs_here else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def _write_tier_nodeids(destination: Path, *, items: Sequence[_Item]) -> None:
    """Write one selected-nodeid file per PR pytest hook."""
    by_hook = {
        hook: [item.nodeid for item in items if tier.selects(_markers(item))]
        for hook, tier in TIERS.items()
    }
    for hook_id, nodeids in by_hook.items():
        _write_nodeids(destination / f"{hook_id}.txt", nodeids=nodeids)
    # pragma: no mutate start -- every encoding mutant is UTF-8 on a UTF-8 host.
    (destination / "selected-counts.json").write_text(
        json.dumps({hook_id: len(nodeids) for hook_id, nodeids in by_hook.items()})
        + "\n",
        encoding="utf-8",
    )
    # pragma: no mutate end


def _write_tier_files(destination: Path, *, items: Sequence[_Item]) -> None:
    r"""Write ``<tier>\\t<file>`` once for each file and tier it holds tests for."""
    lines = sorted(
        {
            f"{tier}\t{item.nodeid.partition('::')[0]}"
            for item in items
            for tier, spec in TIERS.items()
            if spec.selects(_markers(item))
        },
    )
    _write_nodeids(destination, nodeids=lines)


def _markers(item: _Item) -> set[str]:
    """Return marker names declared on ``item``."""
    return {marker.name for marker in item.iter_markers()}


@cache
def _listed(listing: Path, *, root: Path) -> tuple[frozenset[Path], frozenset[Path]]:
    """Return the listed files under ``root``, and every directory holding one."""
    files = frozenset(
        root / line.removeprefix("!").partition("::")[0]
        for line in listing.read_bytes().decode().splitlines()
        if line
    )
    directories = frozenset(
        parent
        for file in files
        for parent in file.parents
        if parent.is_relative_to(root)
    )
    return files, directories


@cache
def _split_tests(listing: Path) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Return the tests named for files the listing splits, and the tests it drops."""
    lines = [line for line in listing.read_bytes().decode().splitlines() if line]
    whole = {line for line in lines if "::" not in line}
    named: dict[str, set[str]] = {}
    for line in lines:
        file = line.partition("::")[0]
        if "::" in line and not line.startswith("!") and file not in whole:
            named.setdefault(file, set()).add(line)
    return (
        {file: frozenset(tests) for file, tests in named.items()},
        frozenset(line.removeprefix("!") for line in lines if line.startswith("!")),
    )


def _write_nodeids(path: Path, *, nodeids: Iterable[str]) -> None:
    """Write ``nodeids`` to ``path``, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # pragma: no mutate start -- every encoding mutant is UTF-8 on a UTF-8 host.
    path.write_text("".join(f"{nodeid}\n" for nodeid in nodeids), encoding="utf-8")
    # pragma: no mutate end
