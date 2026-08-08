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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_user_dirs(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point every XDG base directory at a per-test tmp root.

    Sets the environment rather than patching the ``userdirs`` functions: the
    env var is the seam production reads, so a test exercises the real
    resolution -- including a wrong ``app`` argument, which a patched function
    would hide. Patching also only reaches the one module that imported the
    helper, leaving every other caller pointed at the real directory.

    Returns:
      root: The tmp directory the four XDG variables point at. Assertions
        normally do not need it; call the ``userdirs`` helper instead.

    """
    root = tmp_path_factory.mktemp("userdirs")
    for variable, leaf in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_STATE_HOME", "state"),
    ):
        monkeypatch.setenv(variable, str(root / leaf))
    return root
