"""Tests for ``agents_md``: discovery, frontmatter, includes, formatting."""

from __future__ import annotations

from pathlib import Path

import logging
import platform

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
        _ = (tmp_path / "AGENTS.md").write_text("# project rules\nrule 1\n")
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
        _ = (parent / "AGENTS.md").write_text("parent-p\n")
        _ = (parent / "AGENTS.local.md").write_text("parent-l\n")
        _ = (child / "AGENTS.md").write_text("child-p\n")
        _ = (child / "AGENTS.local.md").write_text("child-l\n")
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
        _ = (dot / "AGENTS.md").write_text("dotsagentmd\n")
        files = agents_md._discover(tmp_path, cfg)
        assert any("dotsagentmd" in f.content for f in files)

    def test_finds_rules_recursive(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        (rules / "sub").mkdir(parents=True)
        _ = (rules / "a.md").write_text("rule-a\n")
        _ = (rules / "sub" / "b.md").write_text("rule-b\n")
        files = agents_md._discover(tmp_path, cfg)
        contents = [f.content for f in files]
        assert any("rule-a" in c for c in contents)
        assert any("rule-b" in c for c in contents)

    def test_user_global_loads_first(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "__user__"
        user_dir.mkdir()
        _ = (user_dir / "AGENTS.md").write_text("user-rules\n")
        cfg = agents_md.AgentsMdConfig(
            system_dir=tmp_path / "__system__",
            user_dir=user_dir,
        )
        _ = (tmp_path / "AGENTS.md").write_text("proj\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].memory_type == "User"
        assert "user-rules" in files[0].content

    def test_managed_loads_before_user(self, tmp_path: Path) -> None:
        managed_dir = tmp_path / "managed-fake"
        managed_dir.mkdir()
        _ = (managed_dir / "AGENTS.md").write_text("managed-rule\n")
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
        _ = (real / "AGENTS.md").write_text("once\n")
        child = link / "child"
        child.mkdir()
        files = agents_md._discover(child, cfg)
        contents = [f.content for f in files if "once" in f.content]
        assert len(contents) == 1


class TestFrontmatter:
    def test_strips_frontmatter_block(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text(
            "---\npaths: ['**/*.py']\n---\nreal content\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "real content" in files[0].content
        assert "---" not in files[0].content

    def test_parses_paths_list(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text(
            "---\npaths:\n  - '**/*.py'\n  - 'src/**'\n---\nc\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        # ``src/**`` is preserved verbatim; the recursive-glob suffix is
        # part of the matcher contract, not a typo for the bare dir name.
        assert files[0].globs == ["**/*.py", "src/**"]

    def test_paths_star_star_dropped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        """Bare ``**`` matches everything, so it's equivalent to no constraint."""
        _ = (tmp_path / "AGENTS.md").write_text("---\npaths:\n  - '**'\n---\nc\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].globs == []

    def test_recursive_glob_preserved(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        """``src/**`` is preserved verbatim (regression: used to collapse to ``src``)."""
        _ = (tmp_path / "AGENTS.md").write_text("---\npaths:\n  - 'src/**'\n---\nc\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].globs == ["src/**"]


class TestHtmlComments:
    def test_block_comment_stripped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text(
            "rule 1\n\n<!-- note to self -->\n\nrule 2\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "note to self" not in files[0].content
        assert "rule 1" in files[0].content
        assert "rule 2" in files[0].content

    def test_code_fence_comments_preserved(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text(
            "before\n\n```html\n<!-- keep this -->\n```\n\nafter\n"
        )
        files = agents_md._discover(tmp_path, cfg)
        assert "keep this" in files[0].content


class TestIncludes:
    def test_include_relative(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("main\n@./sub.md\n")
        _ = (tmp_path / "sub.md").write_text("sub-content\n")
        files = agents_md._discover(tmp_path, cfg)
        assert files[0].path.name == "AGENTS.md"
        assert files[1].path.name == "sub.md"
        assert "sub-content" in files[1].content
        assert files[1].parent == files[0].path

    def test_include_cycle_detected(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("main\n@./other.md\n")
        _ = (tmp_path / "other.md").write_text("other\n@./AGENTS.md\n")
        files = agents_md._discover(tmp_path, cfg)
        paths = [f.path for f in files]
        assert len(paths) == len(set(paths)) == 2

    def test_include_in_code_block_ignored(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("main\n\n```\n@./sub.md\n```\n")
        _ = (tmp_path / "sub.md").write_text("should-not-load\n")
        files = agents_md._discover(tmp_path, cfg)
        assert all("should-not-load" not in f.content for f in files)

    def test_include_depth_cap(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        for i in range(10):
            nxt = f"@./f{i + 1}.md\n" if i < 9 else ""
            _ = (tmp_path / f"f{i}.md").write_text(f"level{i}\n{nxt}")
        _ = (tmp_path / "AGENTS.md").write_text("root\n@./f0.md\n")
        files = agents_md._discover(tmp_path, cfg)
        contents = " ".join(f.content for f in files)
        assert "level0" in contents
        assert "level3" in contents
        assert "level4" not in contents


class TestFormat:
    def test_empty(self) -> None:
        assert agents_md._format_for_prompt([], agents_md._DEFAULT_PREAMBLE) == ""

    def test_includes_path_and_description(self, tmp_path: Path) -> None:
        f = agents_md._AgentMdFile(
            path=tmp_path / "AGENTS.md",
            content="hello",
            memory_type="Project",
        )
        out = agents_md._format_for_prompt([f], agents_md._DEFAULT_PREAMBLE)
        assert "hello" in out
        assert str(tmp_path / "AGENTS.md") in out
        assert "project directives, version-controlled" in out

    def test_large_file_not_truncated(self, tmp_path: Path) -> None:
        f = agents_md._AgentMdFile(
            path=tmp_path / "AGENTS.md",
            content="x" * 50_000,
            memory_type="Project",
        )
        out = agents_md._format_for_prompt([f], agents_md._DEFAULT_PREAMBLE)
        assert len(out) >= 50_000

    def test_custom_preamble_replaces_default(self, tmp_path: Path) -> None:
        """Recipe-overridable preamble flows through ``build_section``."""
        _ = (tmp_path / "AGENTS.md").write_text("rule\n")
        cfg = agents_md.AgentsMdConfig(
            system_dir=tmp_path / "__s__",
            user_dir=tmp_path / "__u__",
            preamble="CUSTOM HEADER",
        )
        out = agents_md.build_section(tmp_path, config=cfg)
        assert out.startswith("CUSTOM HEADER")
        assert "Project and user directives follow" not in out


class TestAddDir:
    def test_additional_dirs_discovered(self, tmp_path: Path) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        _ = (extra / "AGENTS.md").write_text("extra-rule\n")
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
        _ = (extra / "AGENTS.md").write_text("extra-rule\n")
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
        _ = (rules / "unconditional.md").write_text("u\n")
        _ = (rules / "conditional.md").write_text("---\npaths: ['**/*.py']\n---\nc\n")
        all_files = agents_md._discover(tmp_path, cfg)
        kept = [f for f in all_files if not f.globs]
        assert any("u" in f.content for f in kept)
        assert all(f.content.strip() != "c" for f in kept)

    def test_conditional_rules_for_paths_matches_globs(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        _ = (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\npython-rule\n"
        )
        _ = (rules / "text.md").write_text("---\npaths: ['**/*.txt']\n---\ntext-rule\n")
        out, matched = agents_md.conditional_rules_for_paths(
            tmp_path,
            [tmp_path / "main.py"],
            config=cfg,
            exclude=set(),
        )
        assert "<system-reminder>" in out
        assert "python-rule" in out
        assert "text-rule" not in out
        assert matched == {str((rules / "python.md").resolve())}


class TestBuildSection:
    def test_empty_when_no_files(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        assert agents_md.build_section(tmp_path, config=cfg) == ""

    def test_integrates(self, tmp_path: Path, cfg: agents_md.AgentsMdConfig) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("# my rules\n")
        out = agents_md.build_section(tmp_path, config=cfg)
        assert "my rules" in out

    def test_skips_conditional_rules_in_prompt(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        _ = (rules / "python.md").write_text(
            "---\npaths: ['**/*.py']\n---\npython-rule\n"
        )
        _ = (tmp_path / "AGENTS.md").write_text("always-loaded\n")
        out = agents_md.build_section(tmp_path, config=cfg)
        assert "always-loaded" in out
        assert "python-rule" not in out


class TestDefaultSystemDir:
    def test_posix_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert agents_md._default_system_dir() == Path("/etc/sagent")

    def test_windows_uses_programdata_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setenv("PROGRAMDATA", "D:/Custom")
        assert agents_md._default_system_dir() == Path("D:/Custom") / "sagent"

    def test_windows_falls_back_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.delenv("PROGRAMDATA", raising=False)
        assert agents_md._default_system_dir() == Path("C:/ProgramData") / "sagent"


class TestProcessErrorPaths:
    def test_rglob_oserror_yields_empty(
        self,
        tmp_path: Path,
        cfg: agents_md.AgentsMdConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rules = tmp_path / ".sagent" / "rules"
        rules.mkdir(parents=True)
        _ = (rules / "a.md").write_text("a\n")

        original_rglob = Path.rglob

        def fake_rglob(self: Path, pattern: str) -> object:
            if self == rules:
                raise OSError("permission denied")
            return original_rglob(self, pattern)

        monkeypatch.setattr(Path, "rglob", fake_rglob)
        out = agents_md._load_md_dir(rules, "Project", set(), cfg)
        assert out == []

    def test_resolve_oserror_skips_file(
        self,
        tmp_path: Path,
        cfg: agents_md.AgentsMdConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "AGENTS.md"
        _ = target.write_text("content\n")
        original_resolve = Path.resolve

        def fake_resolve(self: Path, strict: bool = False) -> Path:
            if self == target:
                raise OSError("ELOOP")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        out = agents_md._process(target, "Project", set(), 0, None, cfg)
        assert out == []

    def test_unreadable_file_skipped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        target = tmp_path / "AGENTS.md"
        _ = target.write_bytes(b"\xff\xfe\xfa not utf-8 \xc3\x28")
        out = agents_md._process(target, "Project", set(), 0, None, cfg)
        assert out == []

    def test_empty_body_skipped(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("---\npaths: ['x']\n---\n\n   \n")
        out = agents_md._discover(tmp_path, cfg)
        assert out == []

    def test_large_file_warns(
        self,
        tmp_path: Path,
        cfg: agents_md.AgentsMdConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("x" * (cfg.large_threshold + 1))
        with caplog.at_level(logging.WARNING, logger=agents_md.__name__):
            files = agents_md._discover(tmp_path, cfg)
        assert len(files) == 1
        assert any("threshold" in record.message for record in caplog.records)

    def test_dedup_key_resolve_failure_uses_raw(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "AGENTS.md"
        original_resolve = Path.resolve

        def fake_resolve(self: Path, strict: bool = False) -> Path:
            if self == target:
                raise OSError("ELOOP")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        assert agents_md._dedup_key(target) == str(target)


class TestExtractPathGlobs:
    def test_string_paths_splits_comma_and_newline(self) -> None:
        meta: dict[str, object] = {"paths": "a/**, b/**\nc.py"}
        globs = agents_md._extract_path_globs(meta)
        assert globs == ["a/**", "b/**", "c.py"]


class TestStripHtmlBlockComments:
    def test_html_block_without_close_kept(self) -> None:
        text = "before\n\n<!-- unclosed\n\nafter\n"
        assert agents_md._strip_html_block_comments(text) == text

    def test_residue_replaces_first_line(self) -> None:
        text = "intro\n<!-- comment --> keep this line\ntail\n"
        out = agents_md._strip_html_block_comments(text)
        assert "comment" not in out
        assert "keep this line" in out
        assert "intro" in out
        assert "tail" in out

    def test_multiline_comment_residue_drops_inner_lines(self) -> None:
        text = "<!--\ncomment-inner\n--> keep tail\n"
        out = agents_md._strip_html_block_comments(text)
        assert "comment-inner" not in out
        assert "keep tail" in out


class TestExpandIncludePath:
    def test_empty_after_anchor_strip_returns_none(self, tmp_path: Path) -> None:
        assert agents_md._expand_include_path("#frag", tmp_path) is None

    def test_home_relative(self, tmp_path: Path) -> None:
        result = agents_md._expand_include_path("~/foo.md", tmp_path)
        assert result is not None
        assert result == (Path.home() / "foo.md").resolve()

    def test_root_ref_rejected(self, tmp_path: Path) -> None:
        assert agents_md._expand_include_path("/", tmp_path) is None

    def test_absolute_ref(self, tmp_path: Path) -> None:
        target = tmp_path / "abs.md"
        result = agents_md._expand_include_path(str(target), tmp_path)
        assert result == target.resolve()

    def test_leading_invalid_char_rejected(self, tmp_path: Path) -> None:
        assert agents_md._expand_include_path("!bang.md", tmp_path) is None


class TestWalkTokensHtmlBlockIncludes:
    def test_include_inside_html_comment_residue_scanned(
        self, tmp_path: Path, cfg: agents_md.AgentsMdConfig
    ) -> None:
        _ = (tmp_path / "AGENTS.md").write_text(
            "head\n\n<!-- ignore --> @./inc.md\n\nbody\n"
        )
        _ = (tmp_path / "inc.md").write_text("inc-body\n")
        files = agents_md._discover(tmp_path, cfg)
        assert any("inc-body" in f.content for f in files)

    def test_extract_includes_from_html_block_with_comment(
        self, tmp_path: Path
    ) -> None:
        _ = (tmp_path / "inc.md").write_text("x\n")
        text = "<!-- note --> @./inc.md\n"
        out = agents_md._extract_includes(text, tmp_path)
        assert out == [(tmp_path / "inc.md").resolve()]

    def test_extract_includes_html_block_only_comment_skipped(
        self, tmp_path: Path
    ) -> None:
        _ = (tmp_path / "inc.md").write_text("x\n")
        out = agents_md._extract_includes("<!-- @./inc.md -->\n", tmp_path)
        assert out == []


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
