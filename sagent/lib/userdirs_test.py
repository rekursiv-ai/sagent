"""Tests for :mod:`sagent.lib.userdirs`."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.lib.userdirs import cache_dir, config_dir, data_dir, state_dir


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def _home(_cls: type[Path]) -> Path:
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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
