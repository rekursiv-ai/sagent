"""Give each pytest invocation a private basetemp beneath one configured parent.

Pytest normally places every process owned by one Unix user beneath one
predictable ``pytest-of-USER`` directory. Each process gets a numbered child,
but session cleanup scans stale siblings in the shared parent. A concurrent
process can therefore fail while removing another process's read-only or
partially materialized fixture.

The configured parent holds one opaque directory per pytest invocation. Xdist
workers remain children of that directory, while unrelated pytest processes
never share a cleanup root.

A plugin, not conftest hooks: the rootdir conftest of every tree that runs
these tests -- the monorepo's and each export's -- names it in
``pytest_plugins``.
"""

from __future__ import annotations

from typing import Final, Protocol, cast

import contextlib
import os
import pathlib
import tempfile

import pytest

from sagent.lib.files import tempclaim


CLAIMED_BASETEMP: Final = pytest.StashKey[bool]()
"""Set when this process claimed its basetemp rather than being given one."""


class XdistNode(Protocol):
    """The part of xdist's ``WorkerController`` ``pytest_configure_node`` uses."""

    config: pytest.Config
    workerinput: dict[str, object]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``pytest_temp_root``, the parent of the invocation directories."""
    parser.addini(
        "pytest_temp_root",
        "Parent directory for invocation-isolated pytest directories",
        default="",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Give pytest one private cleanup root before it builds its temp factory.

    An explicit ``--basetemp`` remains authoritative. Otherwise this creates a
    collision-resistant child beneath ``pytest_temp_root``, holds a lock beside
    it for the life of this process, and removes it when this pytest process
    is unconfigured. A process killed first (a timeout, SIGKILL, OOM) releases
    the lock but leaves the child behind, so each run first sweeps this user's
    unlocked siblings. Temporary-directory filesystem work begins only when pytest
    invokes this hook, never when Python imports the module.

    Args:
      config: This process's config; its ``basetemp`` option is set here.

    """
    if vars(config.option).get("basetemp") is not None:
        return
    configured_root = cast(str, config.getini("pytest_temp_root"))
    parent = (
        pathlib.Path(configured_root)
        if configured_root
        else pathlib.Path(tempfile.gettempdir()) / "pytest"
    )
    if not parent.is_absolute():
        raise pytest.UsageError("pytest_temp_root must be an absolute path")
    # Shared across users on a multi-tenant host. Sticky 1777 (like /tmp
    # itself) keeps each user's 0700 child unlinkable by anyone else. mkdir
    # applies the caller's umask (0002 -> 0775), so create under umask 0:
    # mkdir-then-chmod leaves a window where a second user's run sees 0775 and
    # fails with EACCES on its child. The chmod remains to repair roots left at
    # 0775 by older runs; only the owner may chmod, so a non-owner's call is
    # expected to fail once the mode is already correct.
    #
    # Provisioning owns every ancestor of this shared root. Changing one can
    # make SSH reject a user's home directory or erase a scratch mount's setgid
    # policy, so pytest creates and repairs only its dedicated directory.
    old_umask = os.umask(0)
    try:
        parent.mkdir(exist_ok=True, mode=0o1777)
        with contextlib.suppress(PermissionError):
            parent.chmod(0o1777)
    finally:
        os.umask(old_umask)
    # The root is pytest's alone, so an empty prefix also sweeps basetemps
    # leaked before claims carried locks.
    claim = contextlib.ExitStack()
    config.option.basetemp = claim.enter_context(
        tempclaim.claimed_directory(parent, prefix=""),
    )
    config.add_cleanup(claim.close)
    config.stash[CLAIMED_BASETEMP] = True


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: XdistNode) -> None:
    """Tell an xdist worker its basetemp lies in a tree this controller deletes."""
    if node.config.stash.get(CLAIMED_BASETEMP, False):
        node.workerinput["claimed_basetemp"] = True


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    """Make an xdist worker delete its own tree as its session finishes.

    Left to the controller, every worker's tree went in one serial delete after
    the summary: 14.7s of a full run. Deleted by each worker, the trees go in
    parallel. Here, not at unconfigure: xdist reports the worker finished at the
    end of this hook and the controller then kills it, so a delete still
    running cut ``rmtree`` off mid-scan and printed an unclosed-scandir warning
    per worker. Every test's fixtures are torn down by now, so no patched ``os``
    function is in place. A worker under an explicit ``--basetemp`` keeps its
    tree, as pytest does.

    Args:
      session: This process's session; only an xdist worker's config has
        ``workerinput``.

    """
    config = session.config
    # Xdist sets ``workerinput`` on a worker's config, untyped, and on no other.
    workerinput = cast(dict[str, object], vars(config).get("workerinput", {}))
    if workerinput.get("claimed_basetemp"):
        tempclaim.remove_tree(pathlib.Path(cast(str, config.option.basetemp)))
