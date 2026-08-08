"""Autouse isolation for per-user directories.

Re-export :func:`isolate_user_dirs` from a conftest to point every
``sagent.lib.userdirs`` lookup at a per-test tmp directory::

    from sagent.lib.testing.userdirs_fixture import isolate_user_dirs

    __all__ = ["isolate_user_dirs"]

Two problems this solves, both of which bit before it existed.

A test that writes through ``config_dir``/``data_dir``/``cache_dir``/
``state_dir`` without isolation writes into the DEVELOPER'S real directories --
clobbering live sessions, caches, and credentials, and passing against whatever
happens to be there. Opt-in isolation is not enough: the failure is silent, so
the file that forgets it is exactly the file nobody notices.

A test that asserts a literal ``~/.config/...`` instead of calling the helper
encodes the Linux layout and fails on macOS, where the same call resolves to
``~/Library/Application Support``. With this fixture active an assertion can
simply call the helper: it returns a tmp path, so the expectation follows the
code under test rather than restating one platform's answer.

Lives beside the other testing helpers rather than in a repo-root conftest: a
root conftest does not ship, so an exported package would otherwise lose the
isolation entirely. Each package re-exports this fixture from its own conftest.

One thing it CANNOT reach: a module-level constant that calls a ``userdirs``
helper at import time. Imports run before any fixture, so the value is frozen
against the developer's real directories and every later test sees it --
``jobber.lifecycle.receipts.DEFAULT_RECEIPT_DIR`` is the live example, and a
test that writes through one is writing to a real directory no matter what
this fixture does. Prefer resolving inside the function (or a dataclass
``field(default_factory=...)``, which runs per-instance and does follow the
fixture); when a module constant is genuinely right, its tests must pass an
explicit path rather than trusting isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest


if TYPE_CHECKING:
    from pathlib import Path


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

        from sagent.lib.testing.userdirs_fixture import (
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
    root = tmp_path_factory.mktemp("userdirs")
    for variable, leaf in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_STATE_HOME", "state"),
    ):
        monkeypatch.setenv(variable, str(root / leaf))
    return root
