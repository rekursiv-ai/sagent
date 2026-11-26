"""Check the per-invocation basetemp claim and each worker's own delete."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import stat
import tempfile

import pytest

from sagent.lib.testing import basetemp


if TYPE_CHECKING:
    from collections.abc import Callable


def test_the_temp_root_is_an_ini_option_defaulting_to_empty() -> None:
    parser = _Parser()

    basetemp.pytest_addoption(cast(pytest.Parser, cast(object, parser)))

    assert parser.ini == {
        "pytest_temp_root": (
            "Parent directory for invocation-isolated pytest directories",
            "",
        ),
    }


def test_an_unset_basetemp_is_claimed_beneath_the_temp_root_until_cleanup(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    config = _Config(temp_root=str(root))

    basetemp.pytest_configure(config.as_config())

    claimed = config.claimed()
    assert claimed.parent == root
    assert claimed.is_dir()
    # Sticky and world-writable, as /tmp is, for every user's runs.
    assert stat.S_IMODE(root.stat().st_mode) == 0o1777
    assert config.stash[basetemp.CLAIMED_BASETEMP]
    for cleanup in config.cleanups:
        cleanup()
    assert list(root.iterdir()) == []


def test_an_unset_temp_root_claims_beneath_the_system_temp_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config = _Config(temp_root="")

    basetemp.pytest_configure(config.as_config())

    assert config.claimed().parent == tmp_path / "pytest"
    for cleanup in config.cleanups:
        cleanup()


def test_an_explicit_basetemp_is_left_as_given(tmp_path: Path) -> None:
    given = str(tmp_path / "given")
    config = _Config(temp_root=str(tmp_path / "unused"), basetemp=given)

    basetemp.pytest_configure(config.as_config())

    assert config.option.basetemp == given
    assert not (tmp_path / "unused").exists()
    assert basetemp.CLAIMED_BASETEMP not in config.stash
    assert config.cleanups == []


def test_a_relative_temp_root_is_refused() -> None:
    with pytest.raises(pytest.UsageError, match="must be an absolute path"):
        basetemp.pytest_configure(_Config(temp_root="relative").as_config())


@pytest.mark.parametrize(
    ("claimed", "told"),
    [(True, {"claimed_basetemp": True}), (False, {})],
    ids=["claimed", "given"],
)
def test_only_a_controller_that_claimed_its_basetemp_tells_each_worker(
    *,
    claimed: bool,
    told: dict[str, object],
) -> None:
    stash = pytest.Stash()
    if claimed:
        stash[basetemp.CLAIMED_BASETEMP] = True
    workerinput: dict[str, object] = {}

    basetemp.pytest_configure_node(
        cast(
            "basetemp.XdistNode",
            SimpleNamespace(
                config=SimpleNamespace(stash=stash),
                workerinput=workerinput,
            ),
        ),
    )

    assert workerinput == told


def test_a_worker_under_a_claimed_basetemp_deletes_its_tree_at_session_finish(
    tmp_path: Path,
) -> None:
    """The controller alone deleting every worker's tree took 14.7s after the summary."""
    tree = tmp_path / "popen-gw0"
    (tree / "test_a0").mkdir(parents=True)

    basetemp.pytest_sessionfinish(
        _session(basetemp=str(tree), extra={"workerinput": {"claimed_basetemp": True}}),
    )

    assert not tree.exists()


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("explicit/popen-gw0", {"workerinput": {}}),
        ("claimed", {}),
    ],
    ids=["worker-of-explicit-basetemp", "controller"],
)
def test_only_a_worker_of_a_claimed_basetemp_deletes_at_session_finish(
    tmp_path: Path,
    name: str,
    extra: dict[str, object],
) -> None:
    tree = tmp_path / name
    tree.mkdir(parents=True)

    basetemp.pytest_sessionfinish(_session(basetemp=str(tree), extra=extra))

    assert tree.is_dir()


@pytest.mark.parametrize(
    "hook",
    [basetemp.pytest_configure, basetemp.pytest_sessionfinish],
    ids=["configure", "sessionfinish"],
)
def test_the_claim_and_the_delete_run_first(hook: Callable[..., None]) -> None:
    """The claim precedes the temp factory; the delete precedes xdist's kill.

    In unconfigure the delete raced xdist's kill: an rmtree cut off mid-scan warned.
    """
    options = cast(dict[str, bool | str | None], vars(hook).get("pytest_impl", {}))

    assert options.get("tryfirst") is True


class _Parser:
    """The ``addini`` of a ``pytest.Parser``, recording each option's help and default."""

    def __init__(self) -> None:
        self.ini: dict[str, tuple[str, str]] = {}

    def addini(self, name: str, help: str, *, default: str) -> None:
        self.ini[name] = (help, default)


class _Option:
    """The ``basetemp`` of ``pytest.Config.option``."""

    def __init__(self, basetemp: str | None) -> None:
        self.basetemp: Path | str | None = basetemp


class _Config:
    """What ``pytest_configure`` reads and writes of a ``pytest.Config``."""

    def __init__(self, *, temp_root: str, basetemp: str | None = None) -> None:
        self.option = _Option(basetemp)
        self.stash = pytest.Stash()
        self.cleanups: list[Callable[[], object]] = []
        self._temp_root = temp_root

    def getini(self, name: str) -> str:
        """Return the configured ``pytest_temp_root``."""
        assert name == "pytest_temp_root", name
        return self._temp_root

    def add_cleanup(self, cleanup: Callable[[], object]) -> None:
        """Keep ``cleanup`` for the test to run, as unconfigure would."""
        self.cleanups.append(cleanup)

    def claimed(self) -> Path:
        """Return the directory the hook claimed as the basetemp."""
        assert isinstance(self.option.basetemp, Path), self.option.basetemp
        return self.option.basetemp

    def as_config(self) -> pytest.Config:
        """Return this fake as the config the hook is typed to take."""
        return cast(pytest.Config, cast(object, self))


def _session(
    *,
    basetemp: str,
    extra: dict[str, object] | None = None,
) -> pytest.Session:
    """Return a session whose config holds ``basetemp`` and ``extra``'s attributes."""
    config = SimpleNamespace(option=SimpleNamespace(basetemp=basetemp), **(extra or {}))
    return cast(pytest.Session, cast(object, SimpleNamespace(config=config)))


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
