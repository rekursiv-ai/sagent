"""Tests for sagent.bin.check_wheel."""

from __future__ import annotations

from pathlib import Path

import zipfile

import pytest

from sagent.bin import check_wheel


def _write_wheel(path: Path, names: list[str]) -> None:
    """Write a minimal zip file with the requested names."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            content = (
                "[console_scripts]\n"
                "sagent = sagent.bin.cli:main\n"
                "sagent-slack = sagent.bin.slack:main\n"
                if name.endswith(".dist-info/entry_points.txt")
                else ""
            )
            archive.writestr(name, content)


class TestCheckWheel:
    def test_accepts_required_public_wheel_surface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        dist = tmp_path / "dist"
        dist.mkdir()
        _write_wheel(
            dist / "sagent-0.1.0-py3-none-any.whl",
            [
                "sagent/__init__.py",
                "sagent/bin/cli.py",
                "sagent/bin/slack.py",
                "sagent/assets/sagent.yaml",
                "sagent/assets/default/prompt.md",
                "sagent/assets/default/tools_bash.md",
                "sagent/assets/slack/default.md",
                "sagent/lib/web/fetch.py",
                "sagent/lib/web/search.py",
                "sagent-0.1.0.dist-info/entry_points.txt",
            ],
        )

        assert check_wheel.main() == 0

    def test_rejects_missing_prompt_assets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        dist = tmp_path / "dist"
        dist.mkdir()
        _write_wheel(
            dist / "sagent-0.1.0-py3-none-any.whl",
            [
                "sagent/__init__.py",
                "sagent/bin/cli.py",
                "sagent/bin/slack.py",
                "sagent-0.1.0.dist-info/entry_points.txt",
            ],
        )

        with pytest.raises(SystemExit, match="missing required wheel entries"):
            check_wheel.main()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
