# pytest fixtures look unused to pyright; pytest wires them by name
"""Tests for tools.skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.dotsagent import parse_frontmatter
from sagent.lib.json import JSON, json_freeze
from sagent.tools import skill as skill_mod
from sagent.tools.core import get_tool_state


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-skill"),),
        "multipart/x-tool-call",
    )


class TestFrontmatterParser:
    def test_no_frontmatter(self) -> None:
        meta, body = parse_frontmatter("# hello\n")
        assert meta == {}
        assert body == "# hello\n"

    def test_with_frontmatter(self) -> None:
        text = "---\nname: foo\ndescription: bar baz\n---\n# Body\n"
        meta, body = parse_frontmatter(text)
        assert meta == {"name": "foo", "description": "bar baz"}
        assert body.strip() == "# Body"

    def test_strips_quotes(self) -> None:
        text = "---\nname: 'foo'\ndescription: \"bar\"\n---\nbody\n"
        meta, _ = parse_frontmatter(text)
        assert meta == {"name": "foo", "description": "bar"}


class TestDiscover:
    def test_no_skills(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        assert skill_mod.discover(tmp_path) == []

    def test_project_skill(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        skill_dir = tmp_path / ".sagent" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: do things\n---\n# Body\n",
        )
        skills = skill_mod.discover(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "do things"
        assert skills[0].source == "project"

    def test_project_shadows_user(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        user_root = tmp_path / "user" / "skills"
        user_root.mkdir(parents=True)
        user_skill = user_root / "dup"
        user_skill.mkdir()
        (user_skill / "SKILL.md").write_text(
            "---\nname: dup\ndescription: user version\n---\n",
        )
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", (user_root,))

        cwd = tmp_path / "project"
        cwd.mkdir()
        proj_skill = cwd / ".sagent" / "skills" / "dup"
        proj_skill.mkdir(parents=True)
        (proj_skill / "SKILL.md").write_text(
            "---\nname: dup\ndescription: project version\n---\n",
        )

        skills = skill_mod.discover(cwd)
        assert len(skills) == 1
        assert skills[0].description == "project version"

    def test_derives_name_from_dirname(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        skill_dir = tmp_path / ".sagent" / "skills" / "auto-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("body only\n")
        skills = skill_mod.discover(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "auto-name"

    def test_rejects_invalid_frontmatter_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        skill_dir = tmp_path / ".sagent" / "skills" / "safe-dir"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: </skill><script>\ndescription: bad\n---\nbody\n",
        )
        assert skill_mod.discover(tmp_path) == []

    def test_rejects_invalid_directory_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        skill_dir = tmp_path / ".sagent" / "skills" / "bad name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("body\n")
        assert skill_mod.discover(tmp_path) == []

    def test_imports_agents_only_when_requested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
        skill_dir = tmp_path / ".agents" / "skills" / "agents"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: agents\ndescription: imported\n---\n"
        )

        assert skill_mod.discover(tmp_path) == []
        skills = skill_mod.discover(tmp_path, import_roots=("agents",))

        assert [skill.name for skill in skills] == ["agents"]
        assert {skill.source for skill in skills} == {"import"}


class TestFormat:
    def test_empty(self) -> None:
        assert skill_mod.format_listing([]) == ""

    def test_contains_names(self) -> None:
        infos = [
            skill_mod.SkillInfo(
                name="a",
                description="first",
                body="b",
                source="project",
                path=Path("/x"),
            ),
            skill_mod.SkillInfo(
                name="b",
                description="second",
                body="b",
                source="user",
                path=Path("/y"),
            ),
        ]
        out = skill_mod.format_listing(infos)
        assert "a" in out
        assert "b" in out
        assert "first" in out
        assert "second" in out


@pytest.fixture
def skill_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Skill tool tests need an empty user-skills dir + fresh cwd."""
    monkeypatch.setattr(skill_mod, "_USER_SKILL_ROOTS", ())
    get_tool_state().bash_cwd = str(tmp_path)
    return tmp_path


class TestSkillTool:
    @pytest.mark.anyio
    async def test_unknown_skill_errors(self, skill_cwd: Path) -> None:
        del skill_cwd
        tool = skill_mod.Skill()
        resp = await tool.run(_msg(json_freeze({"skill": "nope"})))
        assert resp.descriptor == "text/x-error"
        assert "Unknown skill" in str(resp.content)

    @pytest.mark.anyio
    async def test_returns_body(self, skill_cwd: Path) -> None:
        d = skill_cwd / ".sagent" / "skills" / "s1"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: s1\ndescription: a\n---\nhello world\n",
        )
        tool = skill_mod.Skill()
        resp = await tool.run(_msg(json_freeze({"skill": "s1"})))
        assert isinstance(resp, TextMessage)
        assert "hello world" in resp.content
        assert "<skill" in resp.content

    @pytest.mark.anyio
    async def test_args_appended(self, skill_cwd: Path) -> None:
        d = skill_cwd / ".sagent" / "skills" / "s1"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: s1\ndescription: a\n---\nbody\n",
        )
        tool = skill_mod.Skill()
        resp = await tool.run(_msg(json_freeze({"skill": "s1", "args": "xyz"})))
        assert isinstance(resp, TextMessage)
        assert "Arguments: xyz" in resp.content

    def test_prompt_lists_discovered_skills(
        self,
        skill_cwd: Path,
    ) -> None:
        # The tool self-reports its per-request listing - adding Skill to
        # an agent auto-wires the catalog into the system prompt.
        d = skill_cwd / ".sagent" / "skills" / "demo"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: demo\ndescription: exercise the hook\n---\nbody\n",
        )
        section = skill_mod.Skill().prompt()
        assert "# Skills" in section
        assert "demo" in section
        assert "exercise the hook" in section

    def test_prompt_empty_when_no_skills(
        self,
        skill_cwd: Path,
    ) -> None:
        del skill_cwd
        assert skill_mod.Skill().prompt() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
