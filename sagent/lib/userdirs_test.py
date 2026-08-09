"""Tests for :mod:`sagent.lib.userdirs`."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.lib.userdirs import (
    cache_dir,
    config_dir,
    data_dir,
    resolve_working_dir,
    state_dir,
)


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


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_data_dir_shape(home: Path, platform: str) -> None:
    result = data_dir("myapp", platform=platform)

    if platform == "linux":
        assert result == home / ".local" / "share" / "myapp"
    elif platform == "darwin":
        assert result == home / "Library" / "Application Support" / "myapp"
    else:
        assert result == home / "AppData" / "Local" / "myapp"


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_config_dir_shape(home: Path, platform: str) -> None:
    result = config_dir("myapp", platform=platform)

    if platform == "linux":
        assert result == home / ".config" / "myapp"
    elif platform == "darwin":
        assert result == home / "Library" / "Application Support" / "myapp"
    else:
        assert result == home / "AppData" / "Local" / "myapp"


def test_data_dir_win32_single_leaf(home: Path) -> None:
    # ``home`` fixture isolates env/home; its presence is the effect.
    del home
    # Leaf appears exactly once -- no base/app/app double-nest (CORE-004).
    result = data_dir("loop", platform="win32")
    assert result.name == "loop"
    assert result.parent.name != "loop"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_data_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "custom"))
    assert data_dir("myapp", platform=platform) == tmp_path / "custom" / "myapp"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_config_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert config_dir("myapp", platform=platform) == tmp_path / "cfg" / "myapp"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_cache_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert cache_dir("myapp", platform=platform) == tmp_path / "cache" / "myapp"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_state_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert state_dir("myapp", platform=platform) == tmp_path / "state" / "myapp"


def test_data_dir_localappdata_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert (
        data_dir("myapp", platform="win32") == tmp_path / "AppData" / "Local" / "myapp"
    )


def test_resolve_working_dir_without_base_is_its_own_root() -> None:
    assert resolve_working_dir(None, "/checkpoints") == Path("/checkpoints")


def test_resolve_working_dir_nests_absolute_working_dir_under_base() -> None:
    """An absolute ``working_dir`` must still land beneath ``base_dir``.

    ``Path("/a") / "/b" == Path("/b")`` -- POSIX makes an absolute right
    operand discard the base. Without the leading-slash strip, a logical
    root like ``"/checkpoints"`` would silently escape its owner's
    ``base_dir`` and write to the filesystem root.
    """
    assert resolve_working_dir("/base", "/checkpoints") == Path("/base/checkpoints")


def test_resolve_working_dir_accepts_relative_working_dir() -> None:
    assert resolve_working_dir("/base", "runs/exp1") == Path("/base/runs/exp1")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
