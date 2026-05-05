"""Tests for sagent.memory."""

from __future__ import annotations

from pathlib import Path

from sagent import memory


class TestMemoryDir:
    def test_under_projects(self, tmp_path: Path) -> None:
        d = memory.memory_dir("/some/cwd", projects_dir=tmp_path / "projects")
        assert d.parent.name != ""
        assert d.name == "memory"

    def test_ensure_creates(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/some/cwd", projects_dir=tmp_path / "projects")
        assert d.exists()


class TestLoadIndex:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        assert (
            memory.load_index("/nonexistent", projects_dir=tmp_path / "projects") == ""
        )

    def test_reads_content(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        (d / "MEMORY.md").write_text("# Memory Index\n- [A](a.md) - x\n")
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert "Memory Index" in content
        assert "[A](a.md)" in content

    def test_line_truncation(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        lines = [f"- line {i}\n" for i in range(500)]
        (d / "MEMORY.md").write_text("".join(lines))
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert content.count("\n") <= 205
        assert "truncated" in content

    def test_byte_truncation(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        (d / "MEMORY.md").write_text("a" * 30_000)
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert len(content.encode()) <= 26_000
        assert "truncated" in content


class TestBuildSystemSection:
    def test_mentions_path(self, tmp_path: Path) -> None:
        out = memory.build_system_section(
            "/some/cwd", projects_dir=tmp_path / "projects"
        )
        assert "auto memory" in out
        assert "memory" in out.lower()

    def test_empty_index_note(self, tmp_path: Path) -> None:
        out = memory.build_system_section(
            "/nonexistent", projects_dir=tmp_path / "projects"
        )
        assert "no memories yet" in out.lower()

    def test_with_index(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        (d / "MEMORY.md").write_text("- [Entry](e.md) - hook\n")
        out = memory.build_system_section("/x", projects_dir=tmp_path / "projects")
        assert "[Entry](e.md)" in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
