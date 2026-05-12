"""Tests for ``memory``: per-project memory dir, index truncation, prompt section."""

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
        _ = (d / "MEMORY.md").write_text("# Memory Index\n- [A](a.md) - x\n")
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert "Memory Index" in content
        assert "[A](a.md)" in content

    def test_line_truncation(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        lines = [f"- line {i}\n" for i in range(500)]
        _ = (d / "MEMORY.md").write_text("".join(lines))
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert content.count("\n") <= 205
        assert "truncated" in content

    def test_byte_truncation(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        _ = (d / "MEMORY.md").write_text("a" * 30_000)
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert len(content.encode()) <= 26_000
        assert "truncated" in content

    def test_byte_truncation_cuts_at_newline(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        # Few lines, each large, so the byte cap fires after the line cap is a
        # no-op. The final newline before the 25 KB mark is the cut point.
        _ = (d / "MEMORY.md").write_text("a" * 20_000 + "\n" + "b" * 20_000 + "\n")
        content = memory.load_index("/x", projects_dir=tmp_path / "projects")
        assert "truncated" in content
        body = content.split("\n\n[MEMORY.md truncated:")[0]
        # The cut keeps everything up to and including the newline after the a's.
        assert body.endswith("\n")
        assert "a" * 20_000 in body
        assert "b" * 20_000 not in body

    def test_unreadable_index_returns_empty(self, tmp_path: Path) -> None:
        d = memory.ensure_memory_dir("/x", projects_dir=tmp_path / "projects")
        _ = (d / "MEMORY.md").write_bytes(b"\xff\xfe not utf-8 \xc3\x28")
        assert memory.load_index("/x", projects_dir=tmp_path / "projects") == ""


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
        _ = (d / "MEMORY.md").write_text("- [Entry](e.md) - hook\n")
        out = memory.build_system_section("/x", projects_dir=tmp_path / "projects")
        assert "[Entry](e.md)" in out


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
