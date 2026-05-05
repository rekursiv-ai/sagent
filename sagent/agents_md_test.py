"""Tests for sagent.agents_md."""

from __future__ import annotations

from pathlib import Path

import pytest

from sagent import agents_md


@pytest.fixture
def cfg(tmp_path: Path) -> agents_md.AgentsMdConfig:
    """Config with system/user dirs isolated to tmp."""
    return agents_md.AgentsMdConfig(
        system_dir=tmp_path / "__system__",
        user_dir=tmp_path / "__user__",
    )


class TestDiscovery:
    def test_empty_when_no_files(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        assert agents_md._discover(tmp_path, cfg) == []

    def test_finds_project_file(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("# project rules\nrule 1\n")
        files = agents_md._discover(tmp_path, cfg)
        assert len(files) == 1
        assert files[0].memory_type == "Project"
        assert "rule 1" in files[0].content

    def test_walks_up_interleaves_project_and_local(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        (parent / "AGENTS.md").write_text("parent-p\n")
        (parent / "AGENTS.local.md").write_text("parent-l\n")
        (child / "AGENTS.md").write_text("child-p\n")
        (child / "AGENTS.local.md").write_text("child-l\n")
        files = agents_md._discover(child, cfg)
        order = [f.content.strip() for f in files]
        assert order.index("parent-p") < order.index("parent-l")
        assert order.index("parent-l") < order.index("child-p")
        assert order.index("child-p") < order.index("child-l")

    def test_finds_dot_sagent_agents_md(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        dot = tmp_path / ".sagent"
        dot.mkdir()
        (dot / "AGENTS.md").write_text("dotsagentmd\n")
        files = agents_md._discover(tmp_path, cfg)
        assert any("dotsagentmd" in f.content for f in files)

    def test_finds_rules_recursive(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        (rules / "sub").mkdir(parents=True)
        (rules / "a.md").write_text("rule-a\n")
        (rules / "sub" / "b.md").write_text("rule-b\n")
        files = agents_md._discover(tmp_path, cfg)
        contents = [f.content for f in files]
        assert any("rule-a" in c for c in contents)
        assert any("rule-b" in c for c in contents)

    def test_user_global_loads_first(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "__user__"
        user_dir.mkdir()
        (user_dir / "AGENTS.md").write_text("user-rules\n")
        cfg = agents_md.AgentsMdConfig(
            system_dir=tmp_path / "__system__",
            user_dir=user_dir,
        )
        (tmp_path / "AGENTS.md").write_text("proj\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].memory_type == "User"
        assert "user-rules" in files[0].content

    def test_managed_loads_before_user(self, tmp_path: Path) -> None:
        managed_dir = tmp_path / "managed-fake"
        managed_dir.mkdir()
        (managed_dir / "AGENTS.md").write_text("managed-rule\n")
        cfg = agents_md.AgentsMdConfig(
            system_dir=managed_dir,
            user_dir=tmp_path / "__user__",
        )
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].memory_type == "Managed"
        assert "managed-rule" in files[0].content

    def test_symlink_dedup(self, tmp_path: Path, cfg: agents_md.AgentsMdConfig) -> None:
        real = tmp_path / "real"
        link = tmp_path / "link"
        real.mkdir()
        link.symlink_to(real)
        (real / "AGENTS.md").write_text("once\n")
        child = link / "child"
        child.mkdir()
        files = agents_md._discover(child, cfg)
        contents = [f.content for f in files if "once" in f.content]
        assert len(contents) == 1


class TestFrontmatter:
    def test_strips_frontmatter_block(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text(
            "---\npaths: ['**/*.py']\n---\nreal content\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "real content" in files[0].content
        assert "---" not in files[0].content

    def test_parses_paths_list(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text(
            "---\npaths:\n  - '**/*.py'\n  - 'src/**'\n---\nc\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].globs == ["**/*.py", "src"]

    def test_paths_star_star_dropped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("---\npaths:\n  - '**'\n---\nc\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].globs == []

    def test_trailing_star_star_stripped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("---\npaths:\n  - 'src/**'\n---\nc\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].globs == ["src"]


class TestHtmlComments:
    def test_block_comment_stripped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text(
            "rule 1\n\n<!-- note to self -->\n\nrule 2\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "note to self" not in files[0].content
        assert "rule 1" in files[0].content
        assert "rule 2" in files[0].content

    def test_code_fence_comments_preserved(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text(
            "before\n\n```html\n<!-- keep this -->\n```\n\nafter\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "keep this" in files[0].content


class TestIncludes:
    def test_include_relative(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("main\n@./sub.md\n")
        (tmp_path / "sub.md").write_text("sub-content\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].path.name == "AGENTS.md"
        assert files[1].path.name == "sub.md"
        assert "sub-content" in files[1].content
        assert files[1].parent == files[0].path

    def test_include_cycle_detected(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("main\n@./other.md\n")
        (tmp_path / "other.md").write_text("other\n@./AGENTS.md\n")
        files = agents_md._discover(tmp_path, cfg)
        paths = [f.path for f in files]
        assert len(paths) == len(set(paths)) == 2

    def test_include_in_code_block_ignored(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("main\n\n```\n@./sub.md\n```\n")
        (tmp_path / "sub.md").write_text("should-not-load\n")
        files = agents_md._discover(tmp_path, cfg)
        assert all("should-not-load" not in f.content for f in files)

    def test_include_depth_cap(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        for i in range(10):
            nxt = f"@./f{i + 1}.md\n" if i < 9 else ""
            (tmp_path / f"f{i}.md").write_text(f"level{i}\n{nxt}")
        (tmp_path / "AGENTS.md").write_text("root\n@./f0.md\n")
        files = agents_md._discover(tmp_path, cfg)
        contents = " ".join(f.content for f in files)
        assert "level0" in contents
        assert "level3" in contents
        assert "level4" not in contents


class TestFormat:
    def test_empty(self) -> None:
        assert agents_md._format_for_prompt([]) == ""

    def test_includes_path_and_description(self, tmp_path: Path) -> None:
        f = agents_md._AgentMdFile(
            path=tmp_path / "AGENTS.md",
            content="hello",
            memory_type="Project",
        )
        out = agents_md._format_for_prompt([f])
        assert "hello" in out
        assert str(tmp_path / "AGENTS.md") in out
        assert "project directives, version-controlled" in out

    def test_large_file_not_truncated(self, tmp_path: Path) -> None:
        f = agents_md._AgentMdFile(
            path=tmp_path / "AGENTS.md",
            content="x" * 50_000,
            memory_type="Project",
        )
        out = agents_md._format_for_prompt([f])
        assert len(out) >= 50_000


class TestAddDir:
    def test_additional_dirs_discovered(self, tmp_path: Path) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "AGENTS.md").write_text("extra-rule\n")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        cfg = agents_md.AgentsMdConfig(
            system_dir=tmp_path / "__s__",
            user_dir=tmp_path / "__u__",
            additional_dirs=[extra],
        )
        files = agents_md._discover(cwd, cfg)
        assert any("extra-rule" in f.content for f in files)

    def test_no_additional_dirs_by_default(
        self,
        tmp_path: Path,
        cfg: agents_md.AgentsMdConfig,
    ) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "AGENTS.md").write_text("extra-rule\n")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        files = agents_md._discover(cwd, cfg)
        assert all("extra-rule" not in f.content for f in files)


class TestConditionalFilter:
    def test_unconditional_only_drops_globbed(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "unconditional.md").write_text("u\n")
        (rules / "conditional.md").write_text("---\npaths: ['**/*.py']\n---\nc\n")
        all_files = agents_md._discover(tmp_path, cfg)
        kept = [f for f in all_files if not f.globs]
        assert any("u" in f.content for f in kept)
        assert all(f.content.strip() != "c" for f in kept)


class TestConditionalMatching:
    def _setup_python_rule(self, tmp_path: Path) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\n# python style\nuse ruff\n"
        )

    def test_matches_py_file(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        (tmp_path / "foo.py").write_text("x = 1\n")
        matches = agents_md._matching_file_triggered_md(
            tmp_path,
            [tmp_path / "foo.py"],
            config=cfg,
        )
        assert len(matches) == 1
        assert "python style" in matches[0].content

    def test_no_match_for_non_py(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        (tmp_path / "README.md").write_text("text\n")
        matches = agents_md._matching_file_triggered_md(
            tmp_path,
            [tmp_path / "README.md"],
            config=cfg,
        )
        assert matches == []

    def test_unconditional_rule_not_returned(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules_dir = tmp_path / ".sagent" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "always.md").write_text("unconditional\n")
        matches = agents_md._matching_file_triggered_md(
            tmp_path,
            [tmp_path / "foo.py"],
            config=cfg,
        )
        assert matches == []

    def test_reminder_wraps_in_system_reminder(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        out = agents_md.file_triggered_md_reminder(
            tmp_path,
            [tmp_path / "foo.py"],
            config=cfg,
        )
        assert out.startswith("<system-reminder>")
        assert out.endswith("</system-reminder>")
        assert "use ruff" in out

    def test_reminder_empty_when_no_match(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        out = agents_md.file_triggered_md_reminder(
            tmp_path,
            [tmp_path / "README.md"],
            config=cfg,
        )
        assert out == ""

    def test_dir_glob_via_trailing_star(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "backend.md").write_text("---\npaths: ['src/**']\n---\nbackend-rule\n")
        matched = agents_md._matching_file_triggered_md(
            tmp_path,
            [tmp_path / "src" / "api.py"],
            config=cfg,
        )
        assert len(matched) == 1
        assert "backend-rule" in matched[0].content

    def test_dedup_across_multiple_matches(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        matched = agents_md._matching_file_triggered_md(
            tmp_path,
            [tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"],
            config=cfg,
        )
        assert len(matched) == 1

    def test_exclude_paths_dedup(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        self._setup_python_rule(tmp_path)
        seen: set[Path] = set()
        out1 = agents_md.file_triggered_md_reminder(
            tmp_path,
            [tmp_path / "foo.py"],
            config=cfg,
            exclude_paths=seen,
        )
        assert "use ruff" in out1
        assert len(seen) == 1
        out2 = agents_md.file_triggered_md_reminder(
            tmp_path,
            [tmp_path / "bar.py"],
            config=cfg,
            exclude_paths=seen,
        )
        assert out2 == ""


class TestBuildSection:
    def test_empty_when_no_files(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        assert agents_md.build_section(tmp_path, config=cfg) == ""

    def test_integrates(self, tmp_path: Path, cfg: agents_md.AgentsMdConfig) -> None:
        (tmp_path / "AGENTS.md").write_text("# my rules\n")
        out = agents_md.build_section(tmp_path, config=cfg)
        assert "my rules" in out

    def test_skips_conditional_rules_in_prompt(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text("---\npaths: ['**/*.py']\n---\npython-rule\n")
        (tmp_path / "AGENTS.md").write_text("always-loaded\n")
        out = agents_md.build_section(tmp_path, config=cfg)
        assert "always-loaded" in out
        assert "python-rule" not in out
