"""Tests for :mod:`sagent.lib.userdirs`."""

from __future__ import annotations

from pathlib import Path

import os
import sys

import pytest

from sagent.lib.userdirs import (
    base_dir_variables,
    cache_dir,
    config_dir,
    data_dir,
    runtime_dir_variables,
    state_dir,
)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def _home(cls: type[Path]) -> Path:
        del cls
        return tmp_path

    monkeypatch.setattr(Path, "home", classmethod(_home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return tmp_path


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_data_dir_shape(home: Path, plat: str) -> None:
    result = data_dir(platform=plat)

    if plat == "linux":
        assert result == home / ".local" / "share"
    elif plat == "darwin":
        assert result == home / "Library" / "Application Support"
    else:
        assert result == home / "AppData" / "Local"


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_config_dir_shape(home: Path, plat: str) -> None:
    result = config_dir(platform=plat)

    if plat == "linux":
        assert result == home / ".config"
    elif plat == "darwin":
        assert result == home / "Library" / "Application Support"
    else:
        assert result == home / "AppData" / "Local"


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_cache_dir_shape(home: Path, plat: str) -> None:
    result = cache_dir(platform=plat)

    if plat == "linux":
        assert result == home / ".cache"
    elif plat == "darwin":
        assert result == home / "Library" / "Caches"
    else:
        assert result == home / "AppData" / "Local"


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_state_dir_shape(home: Path, plat: str) -> None:
    result = state_dir(platform=plat)

    if plat == "linux":
        assert result == home / ".local" / "state"
    elif plat == "darwin":
        assert result == home / "Library" / "Application Support"
    else:
        assert result == home / "AppData" / "Local"


def test_base_dir_carries_no_app_leaf(home: Path) -> None:
    """The base is a base: the caller's namespace is the next segment.

    Guards the CORE-004 shape from the other direction -- previously the
    risk was a doubled ``app`` leaf; now it is a base that smuggles one.
    """
    del home
    assert data_dir(platform="win32").name == "Local"
    assert (data_dir(platform="win32") / "rekursiv-ai").name == "rekursiv-ai"


@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_data_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "custom"))
    assert data_dir(platform=plat) == tmp_path / "custom"


@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_config_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert config_dir(platform=plat) == tmp_path / "cfg"


@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_cache_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert cache_dir(platform=plat) == tmp_path / "cache"


@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_state_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert state_dir(platform=plat) == tmp_path / "state"


def test_data_dir_localappdata_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert data_dir(platform="win32") == tmp_path / "AppData" / "Local"


def test_base_dir_variables_are_those_set_and_not_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # Empty reads as unset, so a child given it would resolve nothing differently.
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("EDITOR", "vim")

    assert base_dir_variables() == {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


@pytest.mark.parametrize("plat", ["linux", "darwin", "win32"])
def test_a_child_given_the_base_dir_variables_alone_resolves_the_same_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
) -> None:
    """A service manager starts a child with little of its parent's environment."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    helpers = (data_dir, config_dir, cache_dir, state_dir)
    parent = [helper(platform=plat) for helper in helpers]
    given = base_dir_variables()

    monkeypatch.setattr(os, "environ", {})
    for name, value in given.items():
        monkeypatch.setenv(name, value)

    assert [helper(platform=plat) for helper in helpers] == parent


@pytest.mark.parametrize(
    ("plat", "made", "found"),
    [("linux", True, True), ("linux", False, False), ("darwin", True, False)],
)
def test_a_runtime_dir_not_set_is_logind_s_where_it_keeps_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plat: str,
    made: bool,
    found: bool,
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", plat)
    if made:
        (tmp_path / str(os.getuid())).mkdir()

    variables = runtime_dir_variables(logind=tmp_path)

    assert variables == (
        {"XDG_RUNTIME_DIR": str(tmp_path / str(os.getuid()))} if found else {}
    )


def test_a_runtime_dir_set_is_kept_as_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1001")

    assert runtime_dir_variables(logind=Path("/nowhere")) == {
        "XDG_RUNTIME_DIR": "/run/user/1001",
    }


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
