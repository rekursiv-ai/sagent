"""Tests for AGENTS.md discovery: tiers, frontmatter, comments, includes, rules."""

from __future__ import annotations

from pathlib import Path

import logging
import platform

import pytest

from sagent.agent import agents_md
from sagent.agent.agents_md import AgentsMdConfig, Directives, directives
from sagent.lib.userdirs import data_dir


@pytest.fixture
def cfg(tmp_path: Path) -> AgentsMdConfig:
    """Return a config whose managed and user tiers live under ``tmp_path``."""
    return AgentsMdConfig(system_dir=tmp_path / "system", user_dir=tmp_path / "user")


def rules_for(
    root: Path,
    paths: list[Path],
    *,
    exclude: set[str],
    config: AgentsMdConfig,
) -> tuple[str, set[str]]:
    """Return the rules ``paths`` match among the root's files, not ``exclude``."""
    window = Directives(root=root, config=config)
    _ = window.start()
    return window.found([], paths, given=exclude)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text)
    return path


def _order(section: str, *needles: str) -> list[int]:
    """Return where each of ``needles`` first appears in ``section``."""
    return [section.index(n) for n in needles]


def test_no_files_is_no_section(tmp_path: Path, cfg: AgentsMdConfig) -> None:
    assert directives(tmp_path, config=cfg) == ""


def test_a_project_file_is_headed_by_its_path_under_the_preamble(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    path = _write(tmp_path / "AGENTS.md", "rule 1\n")

    section = directives(tmp_path, config=cfg)

    assert section == (
        f"{cfg.preamble}\n\nContents of {path.resolve()} (project directives,"
        f" version-controlled; governs only {tmp_path.resolve()} and below):\nrule 1"
    )


def test_a_subtrees_file_says_it_governs_that_tree_alone(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    sub = tmp_path / "vendor" / "lib"
    _write(sub / "AGENTS.md", "lib rule\n")
    _write(sub / ".sagent" / "rules" / "r.md", "lib dot rule\n")
    _write(sub / "AGENTS.local.md", "lib local\n")
    window = Directives(root=tmp_path, config=cfg)
    window.start()

    section, _ = window.found([sub], [sub / "a.py"], given=set())

    governs = f"governs only {sub.resolve()} and below):\n"
    assert f"{governs}lib rule" in section
    assert f"{governs}lib dot rule" in section
    assert f"user-local; {governs}lib local" in section


def test_a_custom_preamble_replaces_the_default(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "rule\n")
    custom = AgentsMdConfig(
        system_dir=cfg.system_dir,
        user_dir=cfg.user_dir,
        preamble="CUSTOM HEADER",
    )

    section = directives(tmp_path, config=custom)

    assert section.startswith("CUSTOM HEADER\n\n")
    assert AgentsMdConfig().preamble not in section


def test_tiers_load_managed_then_user_then_project_root_down(tmp_path: Path) -> None:
    cfg = AgentsMdConfig(system_dir=tmp_path / "system", user_dir=tmp_path / "user")
    _write(cfg.system_dir / "AGENTS.md", "managed-rule\n")
    _write(cfg.system_dir / "rules" / "m.md", "managed-dir-rule\n")
    _write(cfg.user_dir / "AGENTS.md", "user-rule\n")
    _write(cfg.user_dir / "rules" / "u.md", "user-dir-rule\n")
    parent = tmp_path / "grand" / "parent"
    child = parent / "child"
    _write(tmp_path / "grand" / "AGENTS.md", "grand-project\n")
    _write(parent / "AGENTS.md", "parent-project\n")
    _write(parent / "AGENTS.local.md", "parent-local\n")
    _write(child / "AGENTS.md", "child-project\n")
    _write(child / ".sagent" / "AGENTS.md", "child-dot\n")
    _write(child / ".sagent" / "rules" / "r.md", "child-rule\n")
    _write(child / "AGENTS.local.md", "child-local\n")

    section = directives(child, config=cfg)

    order = _order(
        section,
        "managed-rule",
        "managed-dir-rule",
        "user-rule",
        "user-dir-rule",
        "grand-project",
        "parent-project",
        "parent-local",
        "child-project",
        "child-dot",
        "child-rule",
        "child-local",
    )
    assert order == sorted(order)
    assert "(system-wide directives)" in section
    assert "(user-specific directives)" in section
    assert "(project directives, user-local; governs only" in section


def test_the_user_tier_defaults_to_the_users_data_dir(tmp_path: Path) -> None:
    _write(data_dir() / "rekursiv-ai" / "sagent" / "AGENTS.md", "from-data-dir\n")

    section = directives(tmp_path, config=AgentsMdConfig(system_dir=tmp_path / "s"))

    assert "from-data-dir" in section
    assert "(user-specific directives)" in section


def test_rules_are_found_recursively_in_sorted_order(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    rules = tmp_path / ".sagent" / "rules"
    _write(rules / "b.md", "rule-b\n")
    _write(rules / "a" / "z.md", "rule-az\n")
    _write(rules / "notes.txt", "not-markdown\n")
    (rules / "dir.md").mkdir()

    section = directives(tmp_path, config=cfg)

    assert section.index("rule-az") < section.index("rule-b")
    assert "not-markdown" not in section


def test_additional_dirs_load_after_the_working_directory(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write(extra / "AGENTS.md", "extra-rule\n")
    _write(cwd / "AGENTS.md", "cwd-rule\n")
    cfg = AgentsMdConfig(
        system_dir=tmp_path / "system",
        user_dir=tmp_path / "user",
        additional_dirs=(extra,),
    )

    section = directives(cwd, config=cfg)

    assert section.index("cwd-rule") < section.index("extra-rule")
    plain = AgentsMdConfig(system_dir=cfg.system_dir, user_dir=cfg.user_dir)
    assert "extra-rule" not in directives(cwd, config=plain)


@pytest.mark.parametrize(
    ("tiers", "kept"),
    [
        (frozenset({"Managed", "User"}), {"system-rule", "user-rule"}),
        (frozenset[str](), set[str]()),
    ],
    ids=["no-project", "none"],
)
def test_only_the_tiers_named_are_read(
    tmp_path: Path,
    tiers: frozenset[agents_md.Tier],
    kept: set[str],
) -> None:
    """An agent whose directory another account writes reads none of its files."""
    hosts_own = _write(tmp_path / "host" / "private.txt", "host-only\n")
    cwd = tmp_path / "cwd"
    _write(cwd / "AGENTS.md", f"planted @{hosts_own}\n")
    _write(cwd / "AGENTS.local.md", "local-rule\n")
    _write(cwd / ".sagent" / "rules" / "r.md", "---\npaths: '*.py'\n---\nrule\n")
    _write(tmp_path / "extra" / "AGENTS.md", "extra-rule\n")
    _write(tmp_path / "system" / "AGENTS.md", "system-rule\n")
    _write(tmp_path / "user" / "AGENTS.md", "user-rule\n")
    cfg = AgentsMdConfig(
        system_dir=tmp_path / "system",
        user_dir=tmp_path / "user",
        additional_dirs=(tmp_path / "extra",),
        tiers=tiers,
    )

    section = directives(cwd, config=cfg)
    rules, matched = rules_for(cwd, [cwd / "a.py"], exclude=set(), config=cfg)

    words = set(section.split())
    assert {"system-rule", "user-rule"} & words == kept
    assert not {"planted", "host-only", "local-rule", "extra-rule"} & words
    assert (rules, matched) == ("", set())


def test_a_file_reached_twice_through_a_symlink_loads_once(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    real = tmp_path / "real"
    _write(real / "AGENTS.md", "once\n")
    (tmp_path / "link").symlink_to(real)
    cfg = AgentsMdConfig(
        system_dir=cfg.system_dir,
        user_dir=cfg.user_dir,
        additional_dirs=(tmp_path / "link",),
    )

    assert directives(real, config=cfg).count("once") == 1


def test_frontmatter_and_a_byte_order_mark_are_stripped(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "\ufeff---\ntitle: x\n---\nreal content\n")

    section = directives(tmp_path, config=cfg)

    assert section.endswith(":\nreal content")
    assert "title" not in section


def test_malformed_frontmatter_is_dropped_and_the_body_kept(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "---\npaths: [unclosed\n---\nbody\n")

    section = directives(tmp_path, config=cfg)

    assert section.endswith(":\nbody")


@pytest.mark.parametrize("meta", ["- a\n- b", "just words", "42"])
def test_frontmatter_that_is_no_mapping_is_dropped_and_the_body_kept(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    meta: str,
) -> None:
    _write(tmp_path / "AGENTS.md", f"---\n{meta}\n---\nbody\n")

    assert directives(tmp_path, config=cfg).endswith(":\nbody")
    assert rules_for(tmp_path, [tmp_path / "a.py"], exclude=set(), config=cfg) == (
        "",
        set(),
    )


def test_a_file_with_paths_is_a_rule_not_a_directive(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    rules = tmp_path / ".sagent" / "rules"
    _write(rules / "python.md", "---\npaths: ['**/*.py']\n---\npython-rule\n")
    _write(tmp_path / "AGENTS.md", "always-loaded\n")

    section = directives(tmp_path, config=cfg)

    assert "always-loaded" in section
    assert "python-rule" not in section


def test_a_bare_double_star_is_no_condition(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "---\npaths:\n  - '**'\n---\neverywhere\n")

    assert "everywhere" in directives(tmp_path, config=cfg)


def test_rules_for_returns_the_rules_a_path_matches(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    rules = tmp_path / ".sagent" / "rules"
    python = _write(rules / "python.md", "---\npaths: ['**/*.py']\n---\npython-rule\n")
    _write(rules / "text.md", "---\npaths: ['**/*.txt']\n---\ntext-rule\n")
    _write(tmp_path / "AGENTS.md", "always-loaded\n")

    text, matched = rules_for(
        tmp_path,
        [tmp_path / "main.py"],
        exclude=set(),
        config=cfg,
    )

    assert text.startswith(cfg.preamble)
    assert "python-rule" in text
    assert "text-rule" not in text
    assert "always-loaded" not in text
    assert matched == {str(python.resolve())}


def test_rules_for_skips_a_rule_already_given(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    rule = _write(
        tmp_path / ".sagent" / "rules" / "py.md",
        "---\npaths: ['**/*.py']\n---\npy\n",
    )

    text, matched = rules_for(
        tmp_path,
        [tmp_path / "a.py"],
        exclude={str(rule.resolve())},
        config=cfg,
    )

    assert (text, matched) == ("", set())


@pytest.mark.parametrize(
    ("paths", "target", "matches"),
    [
        ("'src/**'", "src/deep/a.py", True),
        ("'src/**'", "lib/a.py", False),
        ("'**/*.py'", "top.py", True),
        ("'a.py, b.py'", "b.py", True),
        ('"a.py\\nb.py"', "b.py", True),
    ],
)
def test_rule_globs_match_paths_relative_to_the_working_directory(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    paths: str,
    target: str,
    matches: bool,
) -> None:
    _write(
        tmp_path / ".sagent" / "rules" / "r.md",
        f"---\npaths: {paths}\n---\nthe-rule\n",
    )

    text, _ = rules_for(tmp_path, [tmp_path / target], exclude=set(), config=cfg)

    assert ("the-rule" in text) is matches


def test_a_path_outside_the_working_directory_matches_as_absolute(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    cwd = tmp_path / "cwd"
    outside = tmp_path / "elsewhere" / "x.py"
    _write(
        cwd / ".sagent" / "rules" / "r.md",
        f"---\npaths: ['{outside.resolve().as_posix()}']\n---\nabs-rule\n",
    )

    text, _ = rules_for(cwd, [outside], exclude=set(), config=cfg)

    assert "abs-rule" in text


def test_block_comments_are_stripped_but_code_is_kept(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(
        tmp_path / "AGENTS.md",
        "rule 1\n\n<!-- note to self -->\n\nrule 2\n\n"
        "```html\n<!-- keep this -->\n```\n\n"
        "<!-- gone --> kept residue\n\n"
        "<!--\ninner-line\n--> kept tail\n\n"
        "> <!-- quoted comment -->\n\n"
        "<!-- unclosed\n",
    )

    section = directives(tmp_path, config=cfg)

    assert "note to self" not in section
    assert "gone" not in section
    assert "inner-line" not in section
    for kept in ("rule 1", "rule 2", "keep this", "kept residue", "kept tail"):
        assert kept in section
    assert "<!-- unclosed" in section
    assert "quoted comment" in section


def test_a_file_of_only_comments_and_frontmatter_is_skipped(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "---\ntitle: x\n---\n<!-- only -->\n\n   \n")

    assert directives(tmp_path, config=cfg) == ""


def test_an_unreadable_file_is_skipped(tmp_path: Path, cfg: AgentsMdConfig) -> None:
    _ = (tmp_path / "AGENTS.md").write_bytes(b"\xff\xfe not utf-8 \xc3\x28")

    assert directives(tmp_path, config=cfg) == ""


def test_a_large_file_is_kept_whole_and_warned_about(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write(tmp_path / "AGENTS.md", "x" * (cfg.large_threshold + 1))

    with caplog.at_level(logging.WARNING, logger=agents_md.__name__):
        section = directives(tmp_path, config=cfg)

    assert "x" * (cfg.large_threshold + 1) in section
    assert any("characters (over" in m for m in caplog.messages)


def test_an_include_loads_after_the_file_that_names_it(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "main\n@docs/sub.md\n")
    _write(tmp_path / "docs" / "sub.md", "sub-content\n@Other.md#a#b\n")
    _write(tmp_path / "docs" / "Other.md", "other-content\n")

    section = directives(tmp_path, config=cfg)

    order = _order(section, "main", "sub-content", "other-content")
    assert order == sorted(order)
    # An include governs what the file naming it governs.
    assert f"governs only {tmp_path.resolve()} and below):\nsub-content" in section


@pytest.mark.parametrize(
    "line",
    [
        "```\n@inc.md\n```",
        "`@inc.md`",
        "<!-- @inc.md -->",
        "mail me@inc.md",
        "@#frag",
        "@/",
    ],
)
def test_includes_in_code_comments_or_words_are_not_followed(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    line: str,
) -> None:
    _write(tmp_path / "AGENTS.md", f"main\n\n{line}\n")
    _write(tmp_path / "inc.md", "should-not-load\n")

    assert "should-not-load" not in directives(tmp_path, config=cfg)


def test_an_include_must_start_with_a_path_character(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "main\n\n@!bang.md\n")
    _write(tmp_path / "!bang.md", "should-not-load\n")

    assert "should-not-load" not in directives(tmp_path, config=cfg)


def test_a_comment_nested_in_a_quote_hides_only_what_it_wraps(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "> <!-- @./hidden.md --> @./shown.md\n")
    _write(tmp_path / "hidden.md", "hidden-body\n")
    _write(tmp_path / "shown.md", "shown-body\n")

    section = directives(tmp_path, config=cfg)

    assert "shown-body" in section
    assert "hidden-body" not in section


def test_an_include_after_a_comment_is_followed(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "head\n\n<!-- ignore --> @./inc.md\n\nbody\n")
    _write(tmp_path / "inc.md", "inc-body\n")

    assert "inc-body" in directives(tmp_path, config=cfg)


def test_absolute_home_and_escaped_space_includes_resolve(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    absolute = _write(tmp_path / "abs.md", "abs-body\n")
    _write(home / "h.md", "home-body\n")
    _write(tmp_path / "my file.md", "spaced-body\n")
    cwd = tmp_path / "cwd"
    _write(cwd / "AGENTS.md", f"@{absolute}\n@~/h.md\n@../my\\ file.md\n")

    section = directives(cwd, config=cfg)

    for body in ("abs-body", "home-body", "spaced-body"):
        assert body in section


def test_an_include_cycle_loads_each_file_once(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "main\n@./other.md\n")
    _write(tmp_path / "other.md", "other\n@./AGENTS.md\n")

    section = directives(tmp_path, config=cfg)

    assert section.count("Contents of") == 2


def test_an_include_of_a_rule_governs_only_the_rules_paths(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    extra = _write(tmp_path / "docs" / "extra.md", "SECRET\n")
    _write(
        tmp_path / ".sagent" / "rules" / "py.md",
        f"---\npaths: ['*.py']\n---\n@{extra}\n",
    )

    text, _ = rules_for(tmp_path, [tmp_path / "a.py"], exclude=set(), config=cfg)

    assert "SECRET" not in directives(tmp_path, config=cfg)
    assert "SECRET" in text


def test_the_window_opens_with_the_roots_directives(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "AGENTS.md", "root-rule\n")
    _write(tmp_path / "sub" / "AGENTS.md", "sub-rule\n")
    window = Directives(root=tmp_path, config=cfg)

    prompt = window.start()

    assert prompt == directives(tmp_path, config=cfg)
    assert "sub-rule" not in prompt
    # The root's own files never come again: the prompt holds them.
    assert window.found([tmp_path], [tmp_path / "a.py"], given=set()) == ("", set())


def test_a_directory_below_is_read_once_its_files_are_seen(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    sub = _write(tmp_path / "sub" / "AGENTS.md", "sub-rule\n")
    deep = tmp_path / "sub" / "deep"
    deep.mkdir()
    window = Directives(root=tmp_path, config=cfg)
    _ = window.start()

    text, keys = window.found([deep], [deep / "x.py"], given=set())
    again = window.found([deep], [deep / "x.py"], given=keys)

    assert "sub-rule" in text
    assert text.startswith(cfg.preamble)
    assert keys == {str(sub.resolve())}
    assert again == ("", set())


def test_a_directory_beside_the_root_is_read_up_to_what_was_read(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(tmp_path / "other" / "AGENTS.md", "other-rule\n")
    window = Directives(root=root, config=cfg)
    _ = window.start()

    text, _ = window.found([tmp_path / "other" / "src"], [], given=set())

    assert "other-rule" in text


def test_a_rule_below_waits_for_a_file_it_governs(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(
        tmp_path / "sub" / ".sagent" / "rules" / "py.md",
        "---\npaths: ['**/*.py']\n---\npy-rule\n",
    )
    window = Directives(root=tmp_path, config=cfg)
    _ = window.start()
    sub = tmp_path / "sub"

    before = window.found([sub], [sub / "notes.md"], given=set())
    text, _ = window.found([sub], [sub / "notes.md", sub / "a.py"], given=set())

    assert before == ("", set())
    assert "py-rule" in text


def test_a_long_file_found_later_is_cut_short(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    sub = _write(tmp_path / "sub" / "AGENTS.md", "x" * (cfg.large_threshold + 5))
    window = Directives(root=tmp_path, config=cfg)
    _ = window.start()

    text, _ = window.found([tmp_path / "sub"], [], given=set())

    assert "x" * (cfg.large_threshold + 1) not in text
    assert f"[... 5 more characters; Read {sub.resolve()} for the rest.]" in text


def test_a_new_window_reads_every_directory_again(
    tmp_path: Path,
    cfg: AgentsMdConfig,
) -> None:
    _write(tmp_path / "sub" / "AGENTS.md", "sub-rule\n")
    window = Directives(root=tmp_path, config=cfg)
    _ = window.start()
    first, _ = window.found([tmp_path / "sub"], [], given=set())
    _ = window.start()

    second, _ = window.found([tmp_path / "sub"], [], given=set())

    assert "sub-rule" in first
    assert second == first


def test_no_project_tier_reads_no_directory_later(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / "AGENTS.md", "sub-rule\n")
    window = Directives(
        root=tmp_path,
        config=AgentsMdConfig(
            system_dir=tmp_path / "system",
            user_dir=tmp_path / "user",
            tiers=frozenset({"Managed", "User"}),
        ),
    )
    _ = window.start()

    assert window.found([tmp_path / "sub"], [], given=set()) == ("", set())


def test_includes_nest_at_most_five_deep(tmp_path: Path, cfg: AgentsMdConfig) -> None:
    for i in range(10):
        _write(tmp_path / f"f{i}.md", f"level{i}\n@./f{i + 1}.md\n")
    _write(tmp_path / "AGENTS.md", "root\n@./f0.md\n")

    section = directives(tmp_path, config=cfg)

    assert "level3" in section
    assert "level4" not in section


def test_a_rules_dir_that_cannot_be_listed_is_skipped(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = tmp_path / ".sagent" / "rules"
    _write(rules / "a.md", "rule-a\n")
    _write(tmp_path / "AGENTS.md", "project\n")

    def refuse(self: Path, pattern: str) -> list[Path]:
        raise OSError(f"cannot list {self} for {pattern}")

    monkeypatch.setattr(Path, "rglob", refuse)

    section = directives(tmp_path, config=cfg)

    assert "project" in section
    assert "rule-a" not in section


def test_a_file_that_cannot_be_resolved_is_skipped(
    tmp_path: Path,
    cfg: AgentsMdConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _write(tmp_path / "AGENTS.md", "content\n")
    resolve = Path.resolve

    def refuse(self: Path, *, strict: bool = False) -> Path:
        if self == target:
            raise OSError("ELOOP")
        return resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", refuse)

    assert directives(tmp_path, config=cfg) == ""


@pytest.mark.parametrize(
    ("system", "programdata", "expected"),
    [
        ("Linux", None, Path("/etc/sagent")),
        ("Windows", "D:/Custom", Path("D:/Custom/sagent")),
        ("Windows", None, Path("C:/ProgramData/sagent")),
    ],
)
def test_the_managed_tier_defaults_by_platform(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    programdata: str | None,
    expected: Path,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    if programdata is None:
        monkeypatch.delenv("PROGRAMDATA", raising=False)
    else:
        monkeypatch.setenv("PROGRAMDATA", programdata)

    assert AgentsMdConfig().system_dir == expected


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
