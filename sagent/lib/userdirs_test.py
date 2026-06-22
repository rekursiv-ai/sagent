"""Tests for :mod:`sagent.lib.userdirs`."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.lib.userdirs import config_dir, data_dir


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    def _home(_cls: type[Path]) -> Path:
        return tmp_path

    monkeypatch.setattr(Path, "home", classmethod(_home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
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


def test_data_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "custom"))
    assert data_dir("myapp", platform="linux") == tmp_path / "custom" / "myapp"


def test_config_dir_xdg_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert config_dir("myapp", platform="linux") == tmp_path / "cfg" / "myapp"


def test_data_dir_localappdata_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert (
        data_dir("myapp", platform="win32") == tmp_path / "AppData" / "Local" / "myapp"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
